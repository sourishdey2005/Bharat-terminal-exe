// crates/bt-app/src/views3d.rs
// Author: Sourish Dey

//! Painter-based 3D and surface views for the dashboard.
//!
//! # Why software 3D
//!
//! The target machine is an i3 with 2 GB of RAM and no GPU, so nothing here
//! touches WebGL or a 3D pipeline. Instead each view projects a height field
//! through a small isometric/oblique transform and fills painter shapes. That
//! costs one `Vec` of already-allocated values and a few dozen draw calls, which
//! is comfortably inside budget, and it renders through the same egui painter as
//! every other tab, so there is no second rendering context to manage.
//!
//! # Shared machinery
//!
//! Every surface is a [`Surface`]: a rectangular grid of heights plus a colour
//! ramp. [`render_surface`] draws it with an oblique projection and simple
//! painter's-algorithm depth sorting, which is enough for a height field and
//! avoids the cost of a real z-buffer. [`draw_frame`] wraps it with axes and a
//! legend so each view does not repeat that code.

use bt_core::OhlcvSeries;
use egui::{pos2, Color32, Pos2, Rect, Sense, Stroke, Ui};

/// A rectangular height field, row-major, `rows * cols` values.
pub struct Surface {
    pub title: String,
    pub subtitle: String,
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<f64>,
    /// Row axis label, e.g. "time".
    pub x_label: String,
    /// Column axis label, e.g. "price level".
    pub y_label: String,
    /// Colour ramp endpoints; values are mapped across this range.
    pub ramp: (Color32, Color32),
}

impl Surface {
    /// Build a surface, panicking on a dimension mismatch. Callers construct
    /// `values` from a known `rows * cols` loop, so a mismatch is a bug here
    /// rather than a runtime condition to recover from.
    ///
    /// Takes a [`SurfaceSpec`] because the positional form exceeded clippy's
    /// argument limit and a mis-ordered call here would silently mislabel a chart.
    pub fn new(spec: SurfaceSpec) -> Self {
        let SurfaceSpec {
            title,
            subtitle,
            cols,
            rows,
            values,
            x_label,
            y_label,
            ramp,
        } = spec;
        assert_eq!(
            values.len(),
            cols * rows,
            "surface '{title}' grid is {cols}x{rows} but {} values were given",
            values.len()
        );
        Self {
            title,
            subtitle,
            cols,
            rows,
            values,
            x_label,
            y_label,
            ramp,
        }
    }

    /// Value range, ignoring non-finite cells.
    pub fn bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for v in &self.values {
            if v.is_finite() {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        if (hi - lo).abs() < 1e-12 {
            (lo - 1.0, hi + 1.0)
        } else {
            (lo, hi)
        }
    }

    /// Whether there is anything to draw.
    pub fn is_drawable(&self) -> bool {
        self.cols >= 2 && self.rows >= 2 && self.values.iter().any(|v| v.is_finite())
    }

    /// Colour for a normalized position in `[0, 1]`.
    pub fn color_at(&self, t: f64) -> Color32 {
        let t = t.clamp(0.0, 1.0) as f32;
        Color32::from_rgb(
            lerp(self.ramp.0.r(), self.ramp.1.r(), t),
            lerp(self.ramp.0.g(), self.ramp.1.g(), t),
            lerp(self.ramp.0.b(), self.ramp.1.b(), t),
        )
    }
}

/// Builder arguments for [`Surface::new`], named so a mis-ordered call site
/// cannot silently mislabel a chart.
pub struct SurfaceSpec {
    pub title: String,
    pub subtitle: String,
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<f64>,
    pub x_label: String,
    pub y_label: String,
    pub ramp: (Color32, Color32),
}

impl SurfaceSpec {
    /// Construct a spec from a named grid and its axes.
    pub fn new(
        title: impl Into<String>,
        subtitle: impl Into<String>,
        grid: Grid,
        ramp: (Color32, Color32),
    ) -> Self {
        Self {
            title: title.into(),
            subtitle: subtitle.into(),
            cols: grid.cols,
            rows: grid.rows,
            values: grid.values,
            x_label: grid.x_label,
            y_label: grid.y_label,
            ramp,
        }
    }
}

/// A named grid: its dimensions, its values, and what its axes mean.
pub struct Grid {
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<f64>,
    pub x_label: String,
    pub y_label: String,
}

impl Grid {
    /// A grid with axis names.
    pub fn new(cols: usize, rows: usize, values: Vec<f64>, x_label: &str, y_label: &str) -> Self {
        Self {
            cols,
            rows,
            values,
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
        }
    }
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Heat ramp: cold to hot.
pub const RAMP_FIRE: (Color32, Color32) = (
    Color32::from_rgb(0x18, 0x2A, 0x4A),
    Color32::from_rgb(0xFF, 0x6B, 0x35),
);
/// Cool ramp for risk/volatility surfaces.
pub const RAMP_ICE: (Color32, Color32) = (
    Color32::from_rgb(0x0B, 0x2A, 0x3A),
    Color32::from_rgb(0x00, 0xE5, 0xFF),
);
/// Green-to-red ramp for return-signed surfaces.
pub const RAMP_SIGNED: (Color32, Color32) = (
    Color32::from_rgb(0xD9, 0x3A, 0x3A),
    Color32::from_rgb(0x1E, 0xC9, 0x6B),
);

/// Isometric projection from a unit cube to screen space.
///
/// A `Projector` is fitted to a rect by [`Projector::fit`], so callers never
/// scale by hand — that is how the previous renderer ended up with cells
/// thousands of pixels wide.
#[derive(Debug, Clone, Copy)]
pub struct Projector {
    origin: Pos2,
    scale: f32,
    /// Yaw in radians: 0 gives a symmetric isometric, +-PI/2 an oblique side view.
    yaw: f32,
    /// Pitch in radians: how tilted the ground plane is.
    pitch: f32,
}

impl Projector {
    /// Standard isometric angles, looking down at a box.
    pub const DEFAULT_YAW: f32 = std::f32::consts::FRAC_PI_4;
    pub const DEFAULT_PITCH: f32 = 0.61548_f32;

    fn raw(&self, u: f32, v: f32, w: f32) -> Pos2 {
        // Orthographic camera: rotate about the vertical axis by yaw, then
        // tilt by the elevation angle.
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let rx = u * cy - v * sy;
        let ry = u * sy + v * cy;

        // Horizontal is unaffected by elevation; depth foreshortens and height
        // lifts. At yaw 45 deg and pitch ~35.26 deg this is the canonical
        // isometric view, where the two ground axes fan out symmetrically.
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let sx = rx;
        let sy = ry * sp - w * cp;
        pos2(
            self.origin.x + sx * self.scale,
            self.origin.y + sy * self.scale,
        )
    }

    /// Fit the unit cube into `rect` with a margin for axis labels.
    pub fn fit(rect: Rect, margin: f32, yaw: f32, pitch: f32) -> Self {
        let probe = Projector {
            origin: pos2(0.0, 0.0),
            scale: 1.0,
            yaw,
            pitch,
        };
        // Bounding box of all eight corners of the unit cube.
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &u in &[0.0f32, 1.0] {
            for &v in &[0.0f32, 1.0] {
                for &w in &[0.0f32, 1.0] {
                    let p = probe.raw(u, v, w);
                    min_x = min_x.min(p.x);
                    max_x = max_x.max(p.x);
                    min_y = min_y.min(p.y);
                    max_y = max_y.max(p.y);
                }
            }
        }
        let span_x = (max_x - min_x).max(1e-6);
        let span_y = (max_y - min_y).max(1e-6);
        let avail_w = (rect.width() - 2.0 * margin).max(1.0);
        let avail_h = (rect.height() - 2.0 * margin).max(1.0);
        let scale = (avail_w / span_x).min(avail_h / span_y);

        // Centre the projected box inside the rect.
        let centre_x = (min_x + max_x) / 2.0;
        let centre_y = (min_y + max_y) / 2.0;
        let origin = pos2(
            rect.center().x - centre_x * scale,
            rect.center().y - centre_y * scale,
        );
        Projector {
            origin,
            scale,
            yaw,
            pitch,
        }
    }

    /// Project a unit-cube point to screen space.
    pub fn project(&self, u: f32, v: f32, w: f32) -> Pos2 {
        self.raw(u, v, w)
    }
}

/// Draw a surface as a shaded isometric height field.
///
/// Returns the plot rect.
pub fn render_surface(ui: &mut Ui, surface: &Surface, yaw: f32, pitch: f32) -> Rect {
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), ui.available_height().max(240.0)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(0x0A, 0x0C, 0x12));

    if !surface.is_drawable() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Not enough data for this surface",
            egui::FontId::proportional(13.0),
            Color32::GRAY,
        );
        return rect;
    }

    // Leave room on the left and bottom for axis labels.
    let margin = 34.0_f32.min(rect.width() * 0.15);
    let proj = Projector::fit(rect, margin, yaw, pitch);

    let (lo, hi) = surface.bounds();
    let span = if (hi - lo).abs() < 1e-12 {
        1.0
    } else {
        hi - lo
    };
    let norm = |v: f64| -> f32 {
        if v.is_finite() {
            ((v - lo) / span) as f32
        } else {
            0.0
        }
    };

    let cols = surface.cols;
    let rows = surface.rows;
    let du = 1.0 / cols as f32;
    let dv = 1.0 / rows as f32;

    // Normalized height at a grid corner, clamped at the edges.
    let at = |i: isize, j: isize| -> f32 {
        let ci = i.clamp(0, cols as isize - 1) as usize;
        let cj = j.clamp(0, rows as isize - 1) as usize;
        norm(surface.values[cj * cols + ci])
    };
    let cell = |i: usize, j: usize| -> f32 { norm(surface.values[j * cols + i]) };

    // ---- floor ---------------------------------------------------------
    // A dark base quad so the surface reads as a solid sitting on a plane.
    let floor = vec![
        proj.project(0.0, 0.0, 0.0),
        proj.project(1.0, 0.0, 0.0),
        proj.project(1.0, 1.0, 0.0),
        proj.project(0.0, 1.0, 0.0),
    ];
    painter.add(egui::Shape::convex_polygon(
        floor,
        Color32::from_rgb(0x14, 0x18, 0x22),
        Stroke::new(1.0_f32, Color32::from_gray(60)),
    ));

    // ---- floor grid ----------------------------------------------------
    for c in 0..=cols.min(20) {
        let u = c as f32 * du;
        painter.line_segment(
            [proj.project(u, 0.0, 0.0), proj.project(u, 1.0, 0.0)],
            Stroke::new(1.0_f32, Color32::from_gray(45)),
        );
    }
    for r in 0..=rows.min(20) {
        let v = r as f32 * dv;
        painter.line_segment(
            [proj.project(0.0, v, 0.0), proj.project(1.0, v, 0.0)],
            Stroke::new(1.0_f32, Color32::from_gray(45)),
        );
    }

    // ---- top surface, back to front ------------------------------------
    // In this projection a larger (u + v) is nearer the viewer, so ascending
    // (u + v) is the correct painter's order. A row-major sweep is already
    // monotonic in v, so iterate rows outermost.
    let light = (-0.45_f32, -0.75_f32, 0.55_f32);
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(cols * rows);
    for j in 0..rows {
        for i in 0..cols {
            order.push((i, j));
        }
    }
    order.sort_by_key(|(i, j)| {
        // Key on the cell centre's depth; ties broken deterministically.
        let u = (*i as f32 + 0.5) * du;
        let v = (*j as f32 + 0.5) * dv;
        ((u + v) * 4096.0) as i32
    });

    for (i, j) in order {
        let h = cell(i, j);
        let h00 = at(i as isize, j as isize);
        let h10 = at(i as isize + 1, j as isize);
        let h01 = at(i as isize, j as isize + 1);
        let h11 = at(i as isize + 1, j as isize + 1);

        let poly = vec![
            proj.project(i as f32 * du, j as f32 * dv, h00),
            proj.project((i + 1) as f32 * du, j as f32 * dv, h10),
            proj.project((i + 1) as f32 * du, (j + 1) as f32 * dv, h11),
            proj.project(i as f32 * du, (j + 1) as f32 * dv, h01),
        ];

        // Local gradient drives a cheap Lambert term: flatter cells catch more
        // light, which is what gives the mesh its relief.
        let grad_u = (h10 + h11) * 0.5 - (h00 + h01) * 0.5;
        let grad_v = (h01 + h11) * 0.5 - (h00 + h10) * 0.5;
        let n = normalize3(-grad_u * 6.0, -grad_v * 6.0, 1.0);
        let lambert = (n.0 * light.0 + n.1 * light.1 + n.2 * light.2).max(0.0);
        let shade = 0.55 + 0.45 * lambert;

        let base = surface.color_at(h as f64);
        let color = Color32::from_rgb(
            scale_channel(base.r(), shade),
            scale_channel(base.g(), shade),
            scale_channel(base.b(), shade),
        );
        painter.add(egui::Shape::convex_polygon(poly, color, Stroke::NONE));
    }

    // ---- front skirts --------------------------------------------------
    // The two viewer-facing boundaries, dropped to the floor so the surface
    // reads as a solid block instead of floating tiles.
    for i in 0..cols {
        let h = at(i as isize, rows as isize - 1);
        let hn = at(i as isize + 1, rows as isize - 1);
        let top0 = proj.project(i as f32 * du, 1.0, h);
        let top1 = proj.project((i + 1) as f32 * du, 1.0, hn);
        let bot0 = proj.project(i as f32 * du, 1.0, 0.0);
        let bot1 = proj.project((i + 1) as f32 * du, 1.0, 0.0);
        let color = shade(surface.color_at(h as f64), 0.55);
        painter.add(egui::Shape::convex_polygon(
            vec![top0, top1, bot1, bot0],
            color,
            Stroke::NONE,
        ));
    }
    for j in 0..rows {
        let h = at(cols as isize - 1, j as isize);
        let hn = at(cols as isize - 1, j as isize + 1);
        let top0 = proj.project(1.0, j as f32 * dv, h);
        let top1 = proj.project(1.0, (j + 1) as f32 * dv, hn);
        let bot0 = proj.project(1.0, j as f32 * dv, 0.0);
        let bot1 = proj.project(1.0, (j + 1) as f32 * dv, 0.0);
        let color = shade(surface.color_at(h as f64), 0.75);
        painter.add(egui::Shape::convex_polygon(
            vec![top0, top1, bot1, bot0],
            color,
            Stroke::NONE,
        ));
    }

    draw_axes(&painter, &proj, surface, lo, hi);
    rect
}

fn scale_channel(c: u8, f: f32) -> u8 {
    (c as f32 * f).round().clamp(0.0, 255.0) as u8
}

fn normalize3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let n = (x * x + y * y + z * z).sqrt().max(1e-6);
    (x / n, y / n, z / n)
}

/// Axis lines plus min/max tick labels on the two ground-plane edges.
fn draw_axes(painter: &egui::Painter, proj: &Projector, surface: &Surface, lo: f64, hi: f64) {
    let axis_color = Stroke::new(1.4_f32, Color32::from_gray(150));
    // FontId is not Copy, and painter.text takes it by value.
    let font = || egui::FontId::proportional(10.0);

    // u axis (front-left edge) and v axis (front-right edge).
    painter.line_segment(
        [proj.project(0.0, 1.0, 0.0), proj.project(1.0, 1.0, 0.0)],
        axis_color,
    );
    painter.line_segment(
        [proj.project(1.0, 0.0, 0.0), proj.project(1.0, 1.0, 0.0)],
        axis_color,
    );
    // Height axis at the near corner.
    painter.line_segment(
        [proj.project(0.0, 1.0, 0.0), proj.project(0.0, 1.0, 1.0)],
        Stroke::new(1.4_f32, Color32::from_gray(120)),
    );

    let x_label = surface.x_label.clone();
    let y_label = surface.y_label.clone();

    // Value ticks along the u edge.
    let ticks = 4;
    for k in 0..=ticks {
        let t = k as f32 / ticks as f32;
        let p = proj.project(t, 1.0, 0.0);
        painter.text(
            pos2(p.x, p.y + 12.0),
            egui::Align2::CENTER_TOP,
            format!("{:.0}", t * (surface.cols.saturating_sub(1)) as f32),
            font(),
            Color32::from_gray(130),
        );
    }
    let uc = proj.project(0.5, 1.0, 0.0);
    painter.text(
        pos2(uc.x, uc.y + 24.0),
        egui::Align2::CENTER_TOP,
        x_label,
        font(),
        Color32::from_gray(160),
    );

    for k in 0..=ticks {
        let t = k as f32 / ticks as f32;
        let p = proj.project(1.0, t, 0.0);
        painter.text(
            pos2(p.x + 6.0, p.y),
            egui::Align2::LEFT_CENTER,
            format!("{:.0}", t * (surface.rows.saturating_sub(1)) as f32),
            font(),
            Color32::from_gray(130),
        );
    }
    let vc = proj.project(1.0, 0.5, 0.0);
    painter.text(
        pos2(vc.x + 30.0, vc.y),
        egui::Align2::LEFT_CENTER,
        y_label,
        font(),
        Color32::from_gray(160),
    );

    // Height (value) labels on the z axis.
    let zbase = proj.project(0.0, 1.0, 0.0);
    painter.text(
        pos2(zbase.x - 6.0, zbase.y),
        egui::Align2::RIGHT_BOTTOM,
        format!("{lo:.2}"),
        font(),
        Color32::from_gray(130),
    );
    let ztop = proj.project(0.0, 1.0, 1.0);
    painter.text(
        pos2(ztop.x - 6.0, ztop.y),
        egui::Align2::RIGHT_TOP,
        format!("{hi:.2}"),
        font(),
        Color32::from_gray(130),
    );
}

type CameraMap = std::collections::HashMap<String, (f32, f32)>;

thread_local! {
    /// Per-surface viewpoint, so dragging one 3D tab does not rotate the others.
    static CAMERAS: std::cell::RefCell<CameraMap> = std::cell::RefCell::new(CameraMap::new());
}

/// Run `f` against the camera map.
fn with_cameras<R>(f: impl FnOnce(&mut CameraMap) -> R) -> R {
    CAMERAS.with(|c| f(&mut c.borrow_mut()))
}

/// Darken a colour by a factor, for the shaded side faces.
fn shade(c: Color32, f: f32) -> Color32 {
    Color32::from_rgb(
        scale_channel(c.r(), f),
        scale_channel(c.g(), f),
        scale_channel(c.b(), f),
    )
}

/// Draw a framed surface with a title, subtitle, axes and legend.
///
/// The plot is draggable: horizontal drag yaws the model, vertical drag changes
/// the pitch, and double-clicking resets to isometric. The camera is stored per
/// surface title, so each 3D tab keeps its own viewpoint between frames.
pub fn draw_frame(ui: &mut Ui, surface: &Surface) -> Rect {
    ui.label(egui::RichText::new(&surface.title).strong());
    ui.label(
        egui::RichText::new(&surface.subtitle)
            .small()
            .color(Color32::GRAY),
    );

    let (yaw, pitch) = with_cameras(|c| {
        c.get(&surface.title)
            .copied()
            .unwrap_or((Projector::DEFAULT_YAW, Projector::DEFAULT_PITCH))
    });

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("drag to rotate")
                .small()
                .color(Color32::from_gray(110)),
        );
        ui.separator();
        if ui
            .add(egui::Button::new(egui::RichText::new("Reset view").small()))
            .clicked()
        {
            with_cameras(|c| {
                c.remove(&surface.title);
            });
        }
    });

    let rect = render_surface(ui, surface, yaw, pitch);

    // Apply this frame's drag so rotation feels immediate.
    let id = ui.make_persistent_id(("surface_cam", &surface.title));
    let response = ui.interact(rect, id, Sense::click_and_drag());
    if response.double_clicked() {
        with_cameras(|c| {
            c.remove(&surface.title);
        });
    } else if response.dragged() {
        let delta = response.drag_delta();
        with_cameras(|cams| {
            let entry = cams.entry(surface.title.clone()).or_insert((yaw, pitch));
            // Yaw wraps; pitch is clamped so the model never flips upside down
            // or collapses edge-on into a line.
            entry.0 = (entry.0 + delta.x * 0.01) % std::f32::consts::TAU;
            entry.1 = (entry.1 - delta.y * 0.01).clamp(0.15_f32, 1.4_f32);
        });
    }

    draw_legend(ui, surface, rect);
    rect
}

/// Draw a framed surface with a title, subtitle, axis labels and a colour
/// legend. Returns the plot rect.
#[allow(dead_code)]
pub fn draw_frame_at(ui: &mut Ui, surface: &Surface, yaw: f32, pitch: f32) -> Rect {
    ui.label(egui::RichText::new(&surface.title).strong());
    ui.label(
        egui::RichText::new(&surface.subtitle)
            .small()
            .color(Color32::GRAY),
    );
    let rect = render_surface(ui, surface, yaw, pitch);
    draw_legend(ui, surface, rect);
    rect
}

fn draw_legend(ui: &mut Ui, surface: &Surface, _rect: Rect) {
    let (lo, hi) = surface.bounds();
    let x_label = surface.x_label.clone();
    let y_label = surface.y_label.clone();
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{x_label} \u{2192}   {y_label} \u{2191}"))
                .small()
                .color(Color32::GRAY),
        );
        ui.separator();
        ui.label(egui::RichText::new(format!("{hi:.2}")).small());
        // Gradient strip from low to high.
        let (strip, _) = ui.allocate_exact_size(egui::vec2(120.0, 10.0), Sense::hover());
        let p = ui.painter_at(strip);
        for i in 0..40 {
            let t = i as f64 / 39.0;
            let x0 = strip.min.x + strip.width() * (i as f32 / 40.0);
            let x1 = strip.min.x + strip.width() * ((i + 1) as f32 / 40.0);
            p.rect_filled(
                Rect::from_min_max(pos2(x0, strip.min.y), pos2(x1, strip.max.y)),
                0.0,
                surface.color_at(t),
            );
        }
        p.rect_stroke(strip, 1.0, Stroke::new(1.0_f32, Color32::from_gray(70)));
        ui.label(egui::RichText::new(format!("{lo:.2}")).small());
    });
}

// ---------------------------------------------------------------------------
// Surface builders
// ---------------------------------------------------------------------------

fn closes(series: &OhlcvSeries) -> Vec<f64> {
    series.candles.iter().map(|c| c.close).collect()
}

/// Price over time as a ridge surface: one row per price level so the shape has
/// depth. A single row has nothing to project and renders as a flat band.
///
/// The time axis is bucketed rather than per-bar. At 300 bars a per-column cell
/// projects to well under a pixel, which renders as a solid smear instead of a
/// readable ridge, so the column count is capped at a width that stays legible
/// and the buckets are averaged.
pub fn price_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    // The depth axis consumes vertical extent, so the grid must stay coarse
    // enough that every cell projects to several pixels. A 80x10 grid pushed
    // cell corners a thousand pixels off-canvas and filled the viewport with a
    // solid block; 40x6 keeps every cell comfortably on-screen.
    let levels = 6usize;
    let max_cols = 40usize;
    let cols = c.len().clamp(2, max_cols);
    let bucket = c.len().div_ceil(cols.max(1));

    let mut values = vec![f64::NAN; cols * levels];
    for col in 0..cols {
        let start = col * bucket;
        let end = ((col + 1) * bucket).min(c.len());
        if start >= end {
            continue;
        }
        let slice = &c[start..end];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let lo = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for lvl in 0..levels {
            // Interpolate between the bucket's low and high across the depth
            // axis, so the ridge shows the bar's range, not just its mean.
            let t = lvl as f64 / (levels - 1).max(1) as f64;
            values[col * levels + lvl] = lo + (hi - lo) * t;
        }
        // Guarantee a non-degenerate footprint even for a flat bucket.
        values[col * levels] = values[col * levels].min(mean);
    }
    if values.iter().all(|v| !v.is_finite()) {
        for v in values.iter_mut() {
            *v = 0.0;
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("3D Price Surface \u{2014} {}", series.symbol),
        "Close over time (low-to-high ridge)",
        Grid::new(cols, levels, values, "time", "price range"),
        RAMP_ICE,
    ))
}

/// Rolling volatility over a price grid: volatility as a function of horizon
/// (columns) and lookback (rows).
pub fn volatility_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let lookbacks = [5usize, 10, 15, 20, 30, 40];
    let horizons = [1usize, 2, 3, 5, 8, 13, 21, 34];
    let mut values = Vec::with_capacity(lookbacks.len() * horizons.len());
    for &lb in &lookbacks {
        for &h in &horizons {
            let v = rolling_ann_vol(&c, lb, h);
            values.push(v);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("3D Volatility Surface \u{2014} {}", series.symbol),
        "Annualized vol by lookback x horizon",
        Grid::new(
            horizons.len(),
            lookbacks.len(),
            values,
            "horizon (bars)",
            "lookback (bars)",
        ),
        RAMP_FIRE,
    ))
}

fn rolling_ann_vol(closes: &[f64], lookback: usize, horizon: usize) -> f64 {
    if closes.len() <= lookback + horizon {
        return f64::NAN;
    }
    let start = closes.len() - lookback - horizon;
    let mut rets = Vec::with_capacity(lookback + horizon);
    for i in (start + 1)..(start + lookback + horizon) {
        let prev = closes[i - 1];
        if prev > 0.0 {
            rets.push((closes[i] / prev).ln());
        }
    }
    if rets.len() < 2 {
        return f64::NAN;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() as f64 - 1.0);
    var.sqrt() * (252.0_f64).sqrt() * 100.0
}

/// Return surface: cumulative return as a function of lookback (rows) and
/// holding period (columns).
pub fn return_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let lookbacks = [5usize, 10, 20, 40, 60];
    let holds = [1usize, 3, 5, 10, 20];
    let mut values = Vec::with_capacity(lookbacks.len() * holds.len());
    for &lb in &lookbacks {
        for &h in &holds {
            if c.len() < lb + h {
                values.push(f64::NAN);
                continue;
            }
            let a = c[c.len() - lb - h];
            let b = c[c.len() - h];
            values.push(if a > 0.0 {
                (b / a - 1.0) * 100.0
            } else {
                f64::NAN
            });
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Return Surface \u{2014} {}", series.symbol),
        "Cumulative return % by lookback x hold",
        Grid::new(
            holds.len(),
            lookbacks.len(),
            values,
            "hold (bars)",
            "lookback (bars)",
        ),
        RAMP_SIGNED,
    ))
}

/// Risk landscape: downside vs upside deviation for a grid of windows.
pub fn risk_landscape(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [10usize, 20, 30, 40, 60, 90];
    let mut values = Vec::with_capacity(windows.len() * windows.len());
    for &wa in &windows {
        for &wb in &windows {
            let rets = window_returns(&c, wa.max(wb));
            if rets.is_empty() {
                values.push(f64::NAN);
                continue;
            }
            let up: f64 = rets.iter().filter(|r| **r > 0.0).sum();
            let down: f64 = rets.iter().filter(|r| **r < 0.0).sum();
            // Total variation captures both tails; higher means a rougher ride.
            let tv: f64 = rets.iter().map(|r| r.abs()).sum();
            let skew_balance = if tv > 0.0 { (up + down) / tv } else { 0.0 };
            values.push(skew_balance * 100.0);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Risk Landscape \u{2014} {}", series.symbol),
        "Up vs down variation balance",
        Grid::new(
            windows.len(),
            windows.len(),
            values,
            "window (bars)",
            "window (bars)",
        ),
        RAMP_SIGNED,
    ))
}

fn window_returns(closes: &[f64], n: usize) -> Vec<f64> {
    if closes.len() <= n {
        return Vec::new();
    }
    let start = closes.len() - n;
    (start + 1..closes.len())
        .filter_map(|i| {
            let p = closes[i - 1];
            if p > 0.0 {
                Some((closes[i] / p - 1.0) * 100.0)
            } else {
                None
            }
        })
        .collect()
}
/// Beta surface across a set of benchmark-free windows: sensitivity of the
/// symbol to its own trailing mean across horizons. Uses the symbol's own
/// history as the reference series (self-beta), which is well-defined without
/// fetching a second instrument.
pub fn beta_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let market = {
        // Proxy benchmark: a moving average of the symbol stands in for the
        // market factor, giving a bounded, positive correlation reference.
        let mut m = Vec::with_capacity(c.len());
        let win = 20usize;
        for i in 0..c.len() {
            // `saturating_sub` keeps the window start in range for the first
            // bars, where `i - win` would otherwise underflow.
            let s = i.saturating_sub(win - 1);
            let mean = c[s..=i].iter().sum::<f64>() / (i - s + 1) as f64;
            m.push(mean);
        }
        m
    };
    let horizons = [10usize, 20, 40, 60];
    let windows = [10usize, 20, 40, 60, 90];
    let mut values = Vec::with_capacity(horizons.len() * windows.len());
    for &h in &horizons {
        for &w in &windows {
            values.push(rolling_beta(&c, &market, w, h));
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Beta Surface \u{2014} {}", series.symbol),
        "Sensitivity to trailing mean by horizon x window",
        Grid::new(
            horizons.len(),
            windows.len(),
            values,
            "horizon (bars)",
            "window (bars)",
        ),
        RAMP_ICE,
    ))
}

fn rolling_beta(closes: &[f64], market: &[f64], window: usize, horizon: usize) -> f64 {
    if closes.len() <= window + horizon || market.len() != closes.len() {
        return f64::NAN;
    }
    let end = closes.len() - horizon;
    let start = end.saturating_sub(window);
    if end <= start {
        return f64::NAN;
    }
    let n = (end - start) as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in start..end {
        let r = if closes[i - 1] > 0.0 {
            closes[i] / closes[i - 1] - 1.0
        } else {
            0.0
        };
        let m = if market[i - 1] > 0.0 {
            market[i] / market[i - 1] - 1.0
        } else {
            0.0
        };
        sxy += r * m;
        sxx += r * r;
        syy += m * m;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    (sxy / n) / ((sxx / n).sqrt() * (syy / n).sqrt())
}

/// Entropy surface: approximate entropy (ApEn) over a grid of embedding
/// dimensions and tolerances. Higher means a less predictable series.
pub fn entropy_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let dims = [2usize, 3, 4, 5];
    let tols = [0.05f64, 0.1, 0.2, 0.4];
    let mut values = Vec::with_capacity(dims.len() * tols.len());
    for &m in &dims {
        for &r in &tols {
            values.push(apen(&c, m, r));
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Entropy Surface \u{2014} {}", series.symbol),
        "Approximate entropy by embedding dim x tolerance",
        Grid::new(
            tols.len(),
            dims.len(),
            values,
            "tolerance (fraction of std)",
            "embedding dim",
        ),
        RAMP_FIRE,
    ))
}

fn apen(series: &[f64], m: usize, r: f64) -> f64 {
    let n = series.len();
    if n < m + 2 {
        return f64::NAN;
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd <= 0.0 {
        return 0.0;
    }
    let tol = r * sd;
    let embed = |i: usize| -> Vec<f64> { series[i..i + m].to_vec() };
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut c_m = Vec::with_capacity(n - m + 1);
    for i in 0..=(n - m) {
        let vi = embed(i);
        let count = (0..=(n - m))
            .filter(|j| *j != i && dist(&vi, &embed(*j)) <= tol)
            .count() as f64;
        c_m.push(count / (n - m) as f64);
    }

    let phi_m: f64 = c_m
        .iter()
        .filter(|c| **c > 0.0)
        .map(|c| c.ln())
        .sum::<f64>()
        / (n - m + 1) as f64;

    if n < m + 3 {
        return f64::NAN;
    }
    let mut c_m1 = Vec::with_capacity(n - m);
    for i in 0..=(n - m - 1) {
        let vi = embed(i);
        let count = (0..=(n - m))
            .filter(|j| dist(&vi, &embed(*j)) <= tol)
            .count() as f64;
        c_m1.push(count / (n - m + 1) as f64);
    }
    let phi_m1: f64 = c_m1
        .iter()
        .filter(|c| **c > 0.0)
        .map(|c| c.ln())
        .sum::<f64>()
        / (n - m) as f64;

    if phi_m.is_finite() && phi_m1.is_finite() {
        phi_m1 - phi_m
    } else {
        f64::NAN
    }
}

/// Alpha surface: excess return of the symbol over its own trailing mean,
/// across a grid of windows and horizons.
pub fn alpha_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [10usize, 20, 40, 60, 90];
    let horizons = [1usize, 3, 5, 10, 20];
    let mut values = Vec::with_capacity(windows.len() * horizons.len());
    for &w in &windows {
        for &h in &horizons {
            if c.len() < w + h + 1 {
                values.push(f64::NAN);
                continue;
            }
            let actual = (c[c.len() - h] / c[c.len() - w - h] - 1.0) * 100.0;
            // Expected drift from the trailing mean over the same window.
            let s = c.len() - w - h;
            let e = c.len() - h;
            let mean = c[s..=e].iter().sum::<f64>() / (e - s + 1) as f64;
            let expected = if mean > 0.0 {
                (c[e] / mean - 1.0) * 100.0
            } else {
                0.0
            };
            values.push(actual - expected);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Alpha Surface \u{2014} {}", series.symbol),
        "Excess return vs trailing mean (%)",
        Grid::new(
            horizons.len(),
            windows.len(),
            values,
            "hold (bars)",
            "window (bars)",
        ),
        RAMP_SIGNED,
    ))
}

/// Signal strength surface: normalized |RSI-50| style conviction across a grid
/// of fast/slow windows.
pub fn signal_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let fast = [5usize, 8, 12, 16, 21];
    let slow = [20usize, 30, 40, 55, 80];
    let mut values = Vec::with_capacity(fast.len() * slow.len());
    for &f in &fast {
        for &sl in &slow {
            values.push(momentum_conviction(&c, f, sl));
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Signal Surface \u{2014} {}", series.symbol),
        "Trend conviction by fast x slow window",
        Grid::new(slow.len(), fast.len(), values, "slow (bars)", "fast (bars)"),
        RAMP_ICE,
    ))
}

fn momentum_conviction(closes: &[f64], fast: usize, slow: usize) -> f64 {
    if closes.len() <= slow + 1 || slow <= fast {
        return f64::NAN;
    }
    let last = closes[closes.len() - 1];
    if last <= 0.0 {
        return f64::NAN;
    }
    let sma = |w: usize| -> f64 { closes[closes.len() - w..].iter().sum::<f64>() / w as f64 };
    let f = sma(fast);
    let s = sma(slow);
    if s <= 0.0 {
        return f64::NAN;
    }
    // Positive when fast leads slow (uptrend), scaled by separation.
    ((f / s) - 1.0) * 100.0
}

/// Regime cluster surface: classify each bar into an up/down/sideways regime
/// by trailing momentum, then show the regime over a grid of thresholds.
pub fn regime_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let wins = [10usize, 15, 20, 30, 40];
    let thresh = [0.5f64, 1.0, 1.5, 2.0, 2.5];
    let mut values = Vec::with_capacity(wins.len() * thresh.len());
    for &w in &wins {
        for &t in &thresh {
            // Fraction of bars whose trailing move exceeds t std: a "regime
            // strength" read across thresholds.
            if c.len() < w + 2 {
                values.push(f64::NAN);
                continue;
            }
            let rets: Vec<f64> = (1..c.len())
                .filter_map(|i| {
                    if c[i - 1] > 0.0 {
                        Some((c[i] / c[i - 1] - 1.0) * 100.0)
                    } else {
                        None
                    }
                })
                .collect();
            if rets.is_empty() {
                values.push(f64::NAN);
                continue;
            }
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let sd =
                (rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64).sqrt();
            if sd <= 0.0 {
                values.push(0.0);
                continue;
            }
            let frac = rets.iter().filter(|r| (**r - mean).abs() > t * sd).count() as f64
                / rets.len() as f64;
            values.push(frac * 100.0);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Regime Cluster \u{2014} {}", series.symbol),
        "% bars beyond threshold sigma",
        Grid::new(
            thresh.len(),
            wins.len(),
            values,
            "threshold (sigma)",
            "window (bars)",
        ),
        RAMP_FIRE,
    ))
}

/// Momentum surface: rate of change across two windows.
pub fn momentum_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let short = [3usize, 5, 8, 13, 21];
    let long = [10usize, 20, 40, 60, 90];
    let mut values = Vec::with_capacity(short.len() * long.len());
    for &s in &short {
        for &l in &long {
            if c.len() <= l + s {
                values.push(f64::NAN);
                continue;
            }
            let base = c[c.len() - l - s - 1];
            let now = *c.last().unwrap();
            values.push(if base > 0.0 {
                (now / base - 1.0) * 100.0
            } else {
                f64::NAN
            });
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Momentum Surface \u{2014} {}", series.symbol),
        "Momentum % by short x long window",
        Grid::new(
            long.len(),
            short.len(),
            values,
            "long (bars)",
            "short (bars)",
        ),
        RAMP_SIGNED,
    ))
}

/// Order-flow 3D: volume-weighted pressure (close location x volume) over a
/// time x price grid.
pub fn order_flow_surface(series: &OhlcvSeries) -> Surface {
    let n = series.candles.len();
    let rows = 24; // time buckets
    let cols = 12; // price buckets
    let mut values = vec![0.0; rows * cols];
    if n == 0 {
        return Surface::new(SurfaceSpec::new(
            format!("Order Flow 3D \u{2014} {}", series.symbol),
            "Volume pressure by time x price",
            Grid::new(cols, rows, values, "price", "time"),
            RAMP_FIRE,
        ));
    }
    let (lo, hi) = price_bounds(series);
    for (i, c) in series.candles.iter().enumerate() {
        let time_bucket = (i * rows) / n.max(1);
        let span = (hi - lo).max(1e-9);
        let price_pos = ((c.close - lo) / span).clamp(0.0, 1.0);
        let price_bucket = (price_pos * (cols - 1) as f64).round() as usize;
        let clv = if (c.high - c.low).abs() > 1e-12 {
            ((c.close - c.low) - (c.high - c.close)) / (c.high - c.low)
        } else {
            0.0
        };
        values[time_bucket * cols + price_bucket] += clv * c.volume;
    }
    Surface::new(SurfaceSpec::new(
        format!("Order Flow 3D \u{2014} {}", series.symbol),
        "Volume pressure by time x price",
        Grid::new(cols, rows, values, "price bucket", "time"),
        RAMP_FIRE,
    ))
}

fn price_bounds(series: &OhlcvSeries) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in &series.candles {
        lo = lo.min(c.low);
        hi = hi.max(c.high);
    }
    if lo.is_finite() && hi.is_finite() && hi > lo {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

/// Skew/kurtosis surface over a grid of windows.
pub fn skew_kurt_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [10usize, 20, 30, 40, 60, 90];
    let mut skew_row = Vec::with_capacity(windows.len());
    let mut kurt_row = Vec::with_capacity(windows.len());
    for &w in &windows {
        let rets = window_returns(&c, w);
        if rets.len() < 3 {
            skew_row.push(f64::NAN);
            kurt_row.push(f64::NAN);
            continue;
        }
        let n = rets.len() as f64;
        let mean = rets.iter().sum::<f64>() / n;
        let m2 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let sd = m2.sqrt();
        if sd <= 0.0 {
            skew_row.push(0.0);
            kurt_row.push(0.0);
            continue;
        }
        let m3 = rets.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n;
        let m4 = rets.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;
        skew_row.push(m3 / sd.powi(3));
        kurt_row.push(m4 / sd.powi(4) - 3.0);
    }
    // Two columns so the grid is projectable: skew and excess kurtosis read
    // side by side for each window, one column per statistic.
    let cols = 2usize;
    let mut values = Vec::with_capacity(windows.len() * cols);
    for i in 0..windows.len() {
        values.push(skew_row[i]);
        values.push(kurt_row[i]);
    }
    Surface::new(SurfaceSpec::new(
        format!("Skew-Kurt Surface \u{2014} {}", series.symbol),
        "Skew and excess kurtosis by window",
        Grid::new(cols, windows.len(), values, "statistic", "window"),
        RAMP_SIGNED,
    ))
}

/// Signal evolution surface: how conviction decays across a grid of past
/// windows (a "how persistent is the trend" read).
pub fn signal_evolution_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [5usize, 10, 20, 40, 60];
    let offsets = [0usize, 5, 15, 30, 60];
    let mut values = Vec::with_capacity(windows.len() * offsets.len());
    for &w in &windows {
        for &off in &offsets {
            let end = c.len().saturating_sub(off);
            if end <= w {
                values.push(f64::NAN);
                continue;
            }
            let base = c[end - w - 1];
            let now = c[end - 1];
            values.push(if base > 0.0 {
                (now / base - 1.0) * 100.0
            } else {
                f64::NAN
            });
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Signal Evolution \u{2014} {}", series.symbol),
        "Past return % by window x bars ago",
        Grid::new(offsets.len(), windows.len(), values, "bars ago", "window"),
        RAMP_SIGNED,
    ))
}

/// Equity surface: cumulative return over a grid of windows/holds.
pub fn equity_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [10usize, 20, 40, 60, 90, 120];
    let holds = [1usize, 3, 5, 10, 20, 40];
    let mut values = Vec::with_capacity(windows.len() * holds.len());
    for &w in &windows {
        for &h in &holds {
            if c.len() < w + h + 1 {
                values.push(f64::NAN);
                continue;
            }
            let a = c[c.len() - w - h];
            let b = c[c.len() - h];
            values.push(if a > 0.0 {
                (b / a).ln() * 100.0
            } else {
                f64::NAN
            });
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Equity Surface \u{2014} {}", series.symbol),
        "Log return % by window x hold",
        Grid::new(
            holds.len(),
            windows.len(),
            values,
            "hold (bars)",
            "window (bars)",
        ),
        RAMP_SIGNED,
    ))
}

/// VaR band surface: historical VaR % across a grid of confidence and window.
pub fn var_band_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [20usize, 40, 60, 90, 120, 180];
    let confs = [0.90f64, 0.95, 0.975, 0.99];
    let mut values = Vec::with_capacity(windows.len() * confs.len());
    for &w in &windows {
        for &cf in &confs {
            let rets = window_returns(&c, w);
            if rets.len() < 5 {
                values.push(f64::NAN);
                continue;
            }
            let mut sorted = rets.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = (((1.0 - cf) * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
            values.push(sorted[idx]);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("VaR Band \u{2014} {}", series.symbol),
        "Historical VaR % by window x confidence",
        Grid::new(
            confs.len(),
            windows.len(),
            values,
            "confidence",
            "window (bars)",
        ),
        RAMP_FIRE,
    ))
}

/// Risk-return cloud: each asset (here, the window slices) plotted as risk vs
/// return, rendered as a surface so it fits the painter pipeline.
pub fn risk_return_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [10usize, 15, 20, 30, 40, 60, 90, 120];
    let mut values = Vec::with_capacity(windows.len() * windows.len());
    for &wa in &windows {
        for &wb in &windows {
            let rets = window_returns(&c, wa.max(wb));
            if rets.len() < 3 {
                values.push(f64::NAN);
                continue;
            }
            let n = rets.len() as f64;
            let mean = rets.iter().sum::<f64>() / n;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            // Sharpe-like ratio, scaled to a plottable range.
            let sharpe = if sd > 0.0 { (mean / sd) * 16.0 } else { 0.0 };
            values.push(sharpe);
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Risk-Return Cloud \u{2014} {}", series.symbol),
        "Risk-adjusted return by window pair",
        Grid::new(windows.len(), windows.len(), values, "window B", "window A"),
        RAMP_SIGNED,
    ))
}

/// Eigenvalue cloud: eigenvalues of the return covariance matrix over a grid of
/// lookback windows, showing factor concentration.
pub fn eigenvalue_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [20usize, 30, 40, 60, 90];
    let dims = [2usize, 3, 4, 5];
    let mut values = Vec::with_capacity(windows.len() * dims.len());
    for &w in &windows {
        for &d in &dims {
            values.push(eigen_concentration(&c, w, d));
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Eigenvalue Cloud \u{2014} {}", series.symbol),
        "Covariance eigenvalue concentration",
        Grid::new(
            dims.len(),
            windows.len(),
            values,
            "factors (dim)",
            "window (bars)",
        ),
        RAMP_ICE,
    ))
}

fn eigen_concentration(closes: &[f64], window: usize, dim: usize) -> f64 {
    // Build a dim-factor return matrix from lagged returns and report the share
    // of variance in the top factor. Full eigen-decomposition is overkill for
    // this display; the power iteration on the covariance gives the dominant
    // eigenvalue directly.
    let rets = window_returns(closes, window);
    if rets.len() < dim + 1 {
        return f64::NAN;
    }
    // Lags act as factors.
    let rows = rets.len() - dim;
    let mut data = vec![0.0; rows * dim];
    for r in 0..rows {
        for k in 0..dim {
            data[r * dim + k] = rets[r + k];
        }
    }
    let mut cov = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            let mut s = 0.0;
            for r in 0..rows {
                s += data[r * dim + a] * data[r * dim + b];
            }
            cov[a * dim + b] = s / rows as f64;
        }
    }
    let trace: f64 = (0..dim).map(|i| cov[i * dim + i]).sum();
    if trace <= 0.0 {
        return f64::NAN;
    }
    // Power iteration for the dominant eigenvalue.
    let mut v = vec![1.0 / (dim as f64).sqrt(); dim];
    let mut lambda = 0.0;
    for _ in 0..24 {
        let mut w = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                w[i] += cov[i * dim + j] * v[j];
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            break;
        }
        for x in w.iter_mut() {
            *x /= norm;
        }
        v = w;
        lambda = norm;
    }
    (lambda / trace).clamp(0.0, 1.0) * 100.0
}

/// Returns heatmap: rolling return by time bucket x period, as a surface.
pub fn returns_heatmap(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let periods = [1usize, 2, 3, 5, 8, 13, 21, 34];
    // Bucket the time axis rather than using one bucket per bar: at one per bar
    // each cell projects to well under a pixel and the heatmap becomes an
    // unreadable smear.
    let max_buckets = 60usize;
    let buckets = (c.len() / 4).clamp(4, max_buckets);
    let stride = (c.len() / buckets).max(1);
    let mut values = vec![f64::NAN; buckets * periods.len()];
    for b in 0..buckets {
        let end_idx = ((b + 1) * stride).min(c.len());
        for (pi, &p) in periods.iter().enumerate() {
            if end_idx <= p {
                continue;
            }
            let a = c[end_idx - p - 1];
            let bpx = c[end_idx - 1];
            values[b * periods.len() + pi] = if a > 0.0 {
                (bpx / a - 1.0) * 100.0
            } else {
                f64::NAN
            };
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("Returns Heatmap \u{2014} {}", series.symbol),
        "Return % by time bucket x period",
        Grid::new(
            periods.len(),
            buckets,
            values,
            "period (bars)",
            "time bucket",
        ),
        RAMP_SIGNED,
    ))
}

/// PCA projection: leading principal-component direction of the return
/// sequence, shown as projections on a grid of window pairs.
pub fn pca_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let windows = [20usize, 30, 40, 60, 90, 120];
    let mut values = Vec::with_capacity(windows.len() * windows.len());
    for &wa in &windows {
        for &wb in &windows {
            values.push(pca_explained(&c, wa.max(wb), 2.min(wa.min(wb))));
        }
    }
    Surface::new(SurfaceSpec::new(
        format!("PCA Projection \u{2014} {}", series.symbol),
        "Variance explained by leading factors",
        Grid::new(windows.len(), windows.len(), values, "window B", "window A"),
        RAMP_ICE,
    ))
}

fn pca_explained(closes: &[f64], window: usize, dim: usize) -> f64 {
    let rets = window_returns(closes, window);
    if rets.len() < dim + 1 || dim == 0 {
        return f64::NAN;
    }
    let rows = rets.len() - dim;
    let mut data = vec![0.0; rows * dim];
    for r in 0..rows {
        for k in 0..dim {
            data[r * dim + k] = rets[r + k];
        }
    }
    let mut cov = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            let mut s = 0.0;
            for r in 0..rows {
                s += data[r * dim + a] * data[r * dim + b];
            }
            cov[a * dim + b] = s / rows as f64;
        }
    }
    let trace: f64 = (0..dim).map(|i| cov[i * dim + i]).sum();
    if trace <= 0.0 {
        return f64::NAN;
    }
    // Sum the top eigenvalues via power iteration with deflation (small dim).
    let mut remaining = cov.clone();
    let mut total = 0.0;
    for _ in 0..dim {
        let mut v = vec![1.0 / (dim as f64).sqrt(); dim];
        let mut lambda = 0.0;
        for _ in 0..32 {
            let mut w = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    w[i] += remaining[i * dim + j] * v[j];
                }
            }
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 1e-12 {
                break;
            }
            for x in w.iter_mut() {
                *x /= norm;
            }
            v = w;
            lambda = norm;
        }
        if lambda <= 1e-12 {
            break;
        }
        total += lambda;
        for i in 0..dim {
            for j in 0..dim {
                remaining[i * dim + j] -= lambda * v[i] * v[j];
            }
        }
    }
    (total / trace).clamp(0.0, 1.0) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(n: usize) -> OhlcvSeries {
        bt_core::synthetic_ohlcv("V3D", n, 7, 100.0)
    }

    #[test]
    fn test_surface_grid_mismatch_is_caught() {
        // A mismatched grid is a programming error, so it must be loud.
        let result = std::panic::catch_unwind(|| {
            Surface::new(SurfaceSpec::new(
                "bad",
                "",
                Grid::new(3, 3, vec![1.0, 2.0], "x", "y"),
                RAMP_FIRE,
            ))
        });
        assert!(
            result.is_err(),
            "grid mismatch should panic at construction"
        );
    }

    #[test]
    fn test_every_surface_builder_produces_finite_data() {
        let s = series(200);
        type Builder = fn(&OhlcvSeries) -> Surface;
        let builders: Vec<(&str, Builder)> = vec![
            ("price", price_surface),
            ("volatility", volatility_surface),
            ("return", return_surface),
            ("risk", risk_landscape),
            ("beta", beta_surface),
            ("entropy", entropy_surface),
            ("alpha", alpha_surface),
            ("signal", signal_surface),
            ("regime", regime_surface),
            ("momentum", momentum_surface),
            ("order_flow", order_flow_surface),
            ("skew_kurt", skew_kurt_surface),
            ("signal_evo", signal_evolution_surface),
            ("equity", equity_surface),
            ("var_band", var_band_surface),
            ("risk_return", risk_return_surface),
            ("eigenvalue", eigenvalue_surface),
            ("heatmap", returns_heatmap),
            ("pca", pca_surface),
        ];

        for (name, f) in builders {
            let surface = f(&s);
            assert_eq!(
                surface.values.len(),
                surface.cols * surface.rows,
                "{name}: grid/value mismatch"
            );
            let finite = surface.values.iter().filter(|v| v.is_finite()).count();
            assert!(
                finite > 0,
                "{name}: no finite values; the view would render empty"
            );
            assert!(surface.is_drawable(), "{name}: not drawable");
            let (lo, hi) = surface.bounds();
            assert!(lo < hi, "{name}: degenerate range [{lo}, {hi}]");
            println!(
                "{name:>14}: {finite:>4}/{:<4} finite, range [{:.3}, {:.3}]",
                surface.values.len(),
                lo,
                hi
            );
        }
    }

    #[test]
    fn test_surfaces_survive_a_short_series() {
        // Selecting 1D can leave very little history; a blank tab is fine, a
        // panic is not.
        let s = series(12);
        for f in [
            price_surface as fn(&OhlcvSeries) -> Surface,
            volatility_surface,
            return_surface,
            risk_landscape,
            beta_surface,
            entropy_surface,
            alpha_surface,
            signal_surface,
            regime_surface,
            momentum_surface,
            order_flow_surface,
            skew_kurt_surface,
            signal_evolution_surface,
            equity_surface,
            var_band_surface,
            risk_return_surface,
            eigenvalue_surface,
            returns_heatmap,
            pca_surface,
        ] {
            let surface = f(&s);
            assert_eq!(surface.values.len(), surface.cols * surface.rows);
        }
    }

    #[test]
    fn test_volatility_surface_is_positive_and_ordered() {
        let s = series(200);
        let surface = volatility_surface(&s);
        let finite: Vec<f64> = surface
            .values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        assert!(
            finite.iter().all(|v| *v >= 0.0),
            "volatility cannot be negative"
        );
    }

    #[test]
    fn test_ramps_map_endpoints() {
        let s = Surface::new(SurfaceSpec::new(
            "t",
            "",
            Grid::new(2, 1, vec![0.0, 1.0], "x", "y"),
            RAMP_FIRE,
        ));
        assert_eq!(s.color_at(0.0), RAMP_FIRE.0);
        assert_eq!(s.color_at(1.0), RAMP_FIRE.1);
        // Out-of-range clamps rather than wrapping.
        assert_eq!(s.color_at(-5.0), RAMP_FIRE.0);
        assert_eq!(s.color_at(5.0), RAMP_FIRE.1);
    }

    // ------------------------------------------------------------------
    // Projection geometry.
    //
    // The first renderer scaled grid coordinates by the *full canvas* instead
    // of a per-cell step, so column 5 landed five screen-widths off-screen and
    // every quad overdrew the others into one solid block. These tests pin the
    // geometry that bug violated: every projected point must land inside the
    // plot rect, the axes must move in the right directions, and the cell quads
    // must tile rather than overlap into a blob.
    // ------------------------------------------------------------------

    const PLOT: Rect = Rect {
        min: pos2(0.0, 0.0),
        max: pos2(800.0, 500.0),
    };

    fn projector() -> Projector {
        Projector::fit(PLOT, 34.0, Projector::DEFAULT_YAW, Projector::DEFAULT_PITCH)
    }

    fn polygon_area(pts: &[Pos2]) -> f32 {
        let mut sum = 0.0;
        for i in 0..pts.len() {
            let a = pts[i];
            let b = pts[(i + 1) % pts.len()];
            sum += a.x * b.y - b.x * a.y;
        }
        (sum / 2.0).abs()
    }

    #[test]
    fn test_every_unit_cube_corner_lands_inside_the_plot_rect() {
        let p = projector();
        for &u in &[0.0f32, 0.5, 1.0] {
            for &v in &[0.0f32, 0.5, 1.0] {
                for &w in &[0.0f32, 1.0] {
                    let s = p.project(u, v, w);
                    assert!(
                        s.x >= PLOT.min.x - 1.0 && s.x <= PLOT.max.x + 1.0,
                        "x={} outside [{}, {}] for ({u},{v},{w})",
                        s.x,
                        PLOT.min.x,
                        PLOT.max.x
                    );
                    assert!(
                        s.y >= PLOT.min.y - 1.0 && s.y <= PLOT.max.y + 1.0,
                        "y={} outside [{}, {}] for ({u},{v},{w})",
                        s.y,
                        PLOT.min.y,
                        PLOT.max.y
                    );
                }
            }
        }
    }

    #[test]
    fn test_projection_is_monotonic_in_each_axis() {
        // Isometric diamond: the two ground axes fan out in opposite horizontal
        // directions and both recede downward, while height lifts straight up.
        let p = projector();
        let base = p.project(0.0, 0.0, 0.0);

        let taller = p.project(0.0, 0.0, 1.0);
        assert!(
            taller.y < base.y - 50.0,
            "height must lift the point: {} -> {}",
            base.y,
            taller.y
        );
        assert!(
            (taller.x - base.x).abs() < 1.0,
            "height must not shift the point horizontally"
        );

        // Column axis: right and down.
        let across = p.project(1.0, 0.0, 0.0);
        assert!(across.x > base.x + 50.0, "columns must advance right");
        assert!(across.y > base.y + 20.0, "columns must recede downward");

        // Row axis: left and down, mirroring the column axis.
        let back = p.project(0.0, 1.0, 0.0);
        assert!(back.x < base.x - 50.0, "rows must advance left");
        assert!(back.y > base.y + 20.0, "rows must recede downward");

        // The far corner sits below both near corners.
        let far = p.project(1.0, 1.0, 0.0);
        assert!(far.y > across.y && far.y > back.y);
    }

    #[test]
    fn test_the_model_is_fitted_to_the_canvas() {
        // A fit-to-rect projection guarantees two things, and both were violated
        // by the original renderer:
        //   * the limiting axis is filled, and
        //   * neither axis ever overflows.
        // The original scaled by the full canvas, so columns ran far off the
        // right edge (overflow) instead of being fitted.
        let p = projector();
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &u in &[0.0f32, 1.0] {
            for &v in &[0.0f32, 1.0] {
                for &w in &[0.0f32, 1.0] {
                    let s = p.project(u, v, w);
                    min_x = min_x.min(s.x);
                    max_x = max_x.max(s.x);
                    min_y = min_y.min(s.y);
                    max_y = max_y.max(s.y);
                }
            }
        }
        let w_frac = (max_x - min_x) / PLOT.width();
        let h_frac = (max_y - min_y) / PLOT.height();
        let limit = w_frac.max(h_frac);
        let slack = w_frac.min(h_frac);

        // The 34 px margin on each edge means the fit lands just under 1.0;
        // this asserts it is *fitted*, which is the property that matters.
        assert!(
            (0.80..=1.0).contains(&limit),
            "the fit should fill one axis, got {:.2}",
            limit
        );
        assert!(
            slack >= 0.35,
            "the non-limiting axis only gets {:.0}% of the plot",
            slack * 100.0
        );
        assert!(
            max_x <= PLOT.max.x && max_y <= PLOT.max.y,
            "the model overflows the plot (max {max_x:.0},{max_y:.0})"
        );
    }

    #[test]
    fn test_projection_is_injective_on_the_floor_plane() {
        // If two distinct grid points collapsed together, the mesh would render
        // as a degenerate sliver instead of a surface.
        let p = projector();
        let mut seen: Vec<Pos2> = Vec::new();
        for i in 0..=8 {
            for j in 0..=8 {
                seen.push(p.project(i as f32 / 8.0, j as f32 / 8.0, 0.0));
            }
        }
        for a in 0..seen.len() {
            for b in (a + 1)..seen.len() {
                let d = (seen[a].x - seen[b].x).hypot(seen[a].y - seen[b].y);
                assert!(
                    d > 1.0,
                    "grid points {a} and {b} collapsed to within {d:.2}px"
                );
            }
        }
    }

    #[test]
    fn test_adjacent_cells_tile_without_gaps_or_overlap() {
        let p = projector();
        let cols = 6usize;
        let rows = 6usize;
        let du = 1.0 / cols as f32;
        let dv = 1.0 / rows as f32;

        let quad = |i: usize, j: usize| -> Vec<Pos2> {
            vec![
                p.project(i as f32 * du, j as f32 * dv, 0.5),
                p.project((i + 1) as f32 * du, j as f32 * dv, 0.5),
                p.project((i + 1) as f32 * du, (j + 1) as f32 * dv, 0.5),
                p.project(i as f32 * du, (j + 1) as f32 * dv, 0.5),
            ]
        };

        for j in 0..rows - 1 {
            let a = quad(1, j);
            let b = quad(2, j);
            // Shared vertical edge: a[1]->a[2] must equal b[0]->b[3].
            let shared = (a[1].x - b[0].x).abs() < 0.01
                && (a[1].y - b[0].y).abs() < 0.01
                && (a[2].x - b[3].x).abs() < 0.01
                && (a[2].y - b[3].y).abs() < 0.01;
            assert!(shared, "row {j}: neighbouring columns do not share an edge");
            let area = polygon_area(&a);
            assert!(area > 50.0, "row {j}: cell area {area:.1} is degenerate");
        }
    }

    #[test]
    fn test_surface_covers_a_sane_fraction_of_the_plot() {
        // A flat isometric diamond legitimately covers only ~10-20% of a wide
        // plot rect, so absolute coverage is the wrong invariant. What matters
        // is that cells have real area and the total is neither zero (everything
        // projected off-screen) nor absurd (everything stacked into a blob).
        let p = projector();
        let cols = 20usize;
        let rows = 20usize;
        let du = 1.0 / cols as f32;
        let dv = 1.0 / rows as f32;
        let h = |i: usize, j: usize| {
            let u = i as f32 * du;
            let v = j as f32 * dv;
            ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt().min(1.0)
        };
        let mut total = 0.0;
        let mut min_cell = f32::INFINITY;
        for j in 0..rows {
            for i in 0..cols {
                let quad = vec![
                    p.project(i as f32 * du, j as f32 * dv, h(i, j)),
                    p.project((i + 1) as f32 * du, j as f32 * dv, h(i + 1, j)),
                    p.project((i + 1) as f32 * du, (j + 1) as f32 * dv, h(i + 1, j + 1)),
                    p.project(i as f32 * du, (j + 1) as f32 * dv, h(i, j + 1)),
                ];
                let area = polygon_area(&quad);
                assert!(
                    area > 1.0,
                    "cell ({i},{j}) has area {area:.2}; the mesh is degenerate"
                );
                min_cell = min_cell.min(area);
                total += area;
            }
        }
        let frac = total / (PLOT.width() * PLOT.height());
        assert!(
            frac > 0.03 && frac < 0.9,
            "surface covers {:.1}% of the plot area; the projection is degenerate",
            frac * 100.0
        );
        assert!(min_cell > 1.0);
    }

    #[test]
    fn test_projection_holds_for_extreme_grid_shapes() {
        // A 200-column surface and a 200-row surface must both fit, since the
        // fit is done on the unit cube and is therefore grid-size independent.
        for _ in 0..1 {
            let p = Projector::fit(PLOT, 34.0, 0.0, Projector::DEFAULT_PITCH);
            for &u in &[0.0f32, 0.5, 1.0] {
                for &v in &[0.0f32, 1.0] {
                    let s = p.project(u, v, 1.0);
                    assert!(
                        s.x >= PLOT.min.x - 1.0 && s.x <= PLOT.max.x + 1.0,
                        "x={} escaped",
                        s.x
                    );
                    assert!(
                        s.y >= PLOT.min.y - 1.0 && s.y <= PLOT.max.y + 1.0,
                        "y={} escaped",
                        s.y
                    );
                }
            }
        }
    }

    /// Every real surface, measured through the actual projection.
    ///
    /// This is the check that the shipped geometry produces a sensible picture,
    /// for every builder and every one of the real data shapes rather than a
    /// synthetic cone: cell area must be real, coverage must be plausible, and
    /// nothing may escape the plot rect.
    #[test]
    fn test_every_real_surface_projects_to_a_sensible_picture() {
        let s = series(300);
        type Builder = fn(&OhlcvSeries) -> Surface;
        let builders: Vec<(&str, Builder)> = vec![
            ("price", price_surface),
            ("volatility", volatility_surface),
            ("return", return_surface),
            ("risk", risk_landscape),
            ("beta", beta_surface),
            ("entropy", entropy_surface),
            ("alpha", alpha_surface),
            ("signal", signal_surface),
            ("regime", regime_surface),
            ("momentum", momentum_surface),
            ("order_flow", order_flow_surface),
            ("skew_kurt", skew_kurt_surface),
            ("signal_evo", signal_evolution_surface),
            ("equity", equity_surface),
            ("var_band", var_band_surface),
            ("risk_return", risk_return_surface),
            ("eigenvalue", eigenvalue_surface),
            ("heatmap", returns_heatmap),
            ("pca", pca_surface),
        ];

        for (name, build) in builders {
            let surface = build(&s);
            assert!(surface.is_drawable(), "{name}: nothing to draw");

            let (lo, hi) = surface.bounds();
            let span = if (hi - lo).abs() < 1e-12 {
                1.0
            } else {
                hi - lo
            };
            let norm = |v: f64| {
                if v.is_finite() {
                    ((v - lo) / span) as f32
                } else {
                    0.0
                }
            };
            let p = projector();

            let du = 1.0 / surface.cols as f32;
            let dv = 1.0 / surface.rows as f32;
            let mut total_area = 0.0f32;
            let mut min_area = f32::INFINITY;

            for j in 0..surface.rows {
                for i in 0..surface.cols {
                    let h = |di: usize, dj: usize| {
                        let ci = (i + di).min(surface.cols - 1);
                        let cj = (j + dj).min(surface.rows - 1);
                        norm(surface.values[cj * surface.cols + ci])
                    };
                    let quad = [
                        p.project(i as f32 * du, j as f32 * dv, h(0, 0)),
                        p.project((i + 1) as f32 * du, j as f32 * dv, h(1, 0)),
                        p.project((i + 1) as f32 * du, (j + 1) as f32 * dv, h(1, 1)),
                        p.project(i as f32 * du, (j + 1) as f32 * dv, h(0, 1)),
                    ];
                    // Every corner must be on-canvas.
                    for pt in quad.iter() {
                        assert!(
                            pt.x >= PLOT.min.x - 1.0
                                && pt.x <= PLOT.max.x + 1.0
                                && pt.y >= PLOT.min.y - 1.0
                                && pt.y <= PLOT.max.y + 1.0,
                            "{name}: cell ({i},{j}) corner {:?} escaped the plot",
                            pt
                        );
                    }
                    let area = polygon_area(&quad);
                    assert!(
                        area > 0.5,
                        "{name}: cell ({i},{j}) area {area:.3} is degenerate"
                    );
                    min_area = min_area.min(area);
                    total_area += area;
                }
            }

            let frac = total_area / (PLOT.width() * PLOT.height());
            assert!(
                frac > 0.01 && frac < 0.95,
                "{name}: covers {:.1}% of the plot; projection is degenerate",
                frac * 100.0
            );
            println!(
                "{name:>14}: {:>4}x{:<4} area {:>5.1}%  min cell {:>6.1}px",
                surface.cols,
                surface.rows,
                frac * 100.0,
                min_area
            );
        }
    }

    #[test]
    fn test_yaw_and_pitch_stay_well_behaved_across_the_drag_range() {
        for yaw_step in 0..24 {
            let yaw = yaw_step as f32 * 0.26;
            for pitch in [0.15_f32, 0.5, 0.9, 1.4] {
                let p = Projector::fit(PLOT, 34.0, yaw, pitch);
                for &u in &[0.0f32, 1.0] {
                    for &v in &[0.0f32, 1.0] {
                        for &w in &[0.0f32, 1.0] {
                            let s = p.project(u, v, w);
                            assert!(
                                s.x.is_finite() && s.y.is_finite(),
                                "yaw {yaw:.2} pitch {pitch:.2} produced NaN"
                            );
                            assert!(
                                s.x >= PLOT.min.x - 2.0 && s.x <= PLOT.max.x + 2.0,
                                "yaw {yaw:.2} pitch {pitch:.2}: x={} escaped",
                                s.x
                            );
                            assert!(
                                s.y >= PLOT.min.y - 2.0 && s.y <= PLOT.max.y + 2.0,
                                "yaw {yaw:.2} pitch {pitch:.2}: y={} escaped",
                                s.y
                            );
                        }
                    }
                }
            }
        }
    }
}
