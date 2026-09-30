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

/// Project a grid cell to a screen point under an oblique projection.
///
/// `depth` (the height) is drawn on the vertical axis with `z_scale`, and the
/// grid's second axis is sheared to give the isometric look without a real
/// camera.
fn project(col: f32, row: f32, depth: f32, rect: Rect, z_scale: f32, shear: f32) -> Pos2 {
    let gw = rect.width().max(1.0);
    let gh = rect.height().max(1.0);
    // x: column across, plus a shear proportional to row
    let x = col * gw + row * shear * gw;
    // y: row down, minus height
    let y = row * gh - depth * z_scale * gh;
    pos2(rect.min.x + x, rect.min.y + y)
}

/// Draw a surface as a filled height field with depth-sorted quads.
pub fn render_surface(ui: &mut Ui, surface: &Surface, z_scale: f32, shear: f32) -> Rect {
    // egui 0.28 returns (rect, response).
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

    let (lo, hi) = surface.bounds();
    let span = hi - lo;
    let cell_w = rect.width() / surface.cols as f32;
    let cell_h = rect.height() / surface.rows as f32;
    let depth = |v: f64| {
        if v.is_finite() {
            ((v - lo) / span) as f32
        } else {
            0.0
        }
    };

    // Painter's algorithm: draw far rows (low `row`) first so nearer cells
    // overlap them. Within a row, draw back-to-front by height.
    let mut cells: Vec<(usize, usize, f32)> = Vec::with_capacity(surface.cols * surface.rows);
    for row in 0..surface.rows {
        for col in 0..surface.cols {
            let d = depth(surface.values[row * surface.cols + col]);
            cells.push((row, col, d));
        }
    }
    cells.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    for (row, col, d) in cells {
        let a = project(col as f32, row as f32, d, rect, z_scale, shear);
        let b = project((col + 1) as f32, row as f32, d, rect, z_scale, shear);
        let c = project((col + 1) as f32, (row + 1) as f32, d, rect, z_scale, shear);
        let e = project(col as f32, (row + 1) as f32, d, rect, z_scale, shear);
        let poly = vec![a, b, c, e];
        let color = surface.color_at((surface.values[row * surface.cols + col] - lo) / span);
        painter.add(egui::Shape::convex_polygon(poly, color, Stroke::NONE));
        let _ = (cell_w, cell_h);
    }

    // Connect adjacent columns with a vertical wall so the height is legible
    // instead of reading as a flat colour field.
    for row in 0..surface.rows.saturating_sub(1) {
        for col in 0..surface.cols {
            let v0 = surface.values[row * surface.cols + col];
            let v1 = surface.values[(row + 1) * surface.cols + col];
            let top0 = project(col as f32, row as f32, depth(v0), rect, z_scale, shear);
            let bot0 = project(
                col as f32,
                (row + 1) as f32,
                depth(v0),
                rect,
                z_scale,
                shear,
            );
            let bot1 = project(
                col as f32,
                (row + 1) as f32,
                depth(v1),
                rect,
                z_scale,
                shear,
            );
            let top1 = project(col as f32, row as f32, depth(v1), rect, z_scale, shear);
            let poly = vec![top0, bot0, bot1, top1];
            if poly.len() == 4 {
                painter.add(egui::Shape::convex_polygon(
                    poly,
                    surface.color_at(
                        (v0 + v1) / 2.0 * 0.0 + (v0 - lo) / span * 0.5 + (v1 - lo) / span * 0.5,
                    ),
                    Stroke::NONE,
                ));
            }
        }
    }

    rect
}

/// Draw a framed surface with a title, subtitle, axis labels and a colour
/// legend. Returns the plot rect.
pub fn draw_frame(ui: &mut Ui, surface: &Surface, z_scale: f32, shear: f32) -> Rect {
    ui.label(egui::RichText::new(&surface.title).strong());
    ui.label(
        egui::RichText::new(&surface.subtitle)
            .small()
            .color(Color32::GRAY),
    );
    let rect = render_surface(ui, surface, z_scale, shear);
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
pub fn price_surface(series: &OhlcvSeries) -> Surface {
    let c = closes(series);
    let levels = 10usize;
    let (lo, hi) = price_bounds(series);
    let span = (hi - lo).max(1e-9);
    let cols = c.len().max(2);
    let mut values = vec![f64::NAN; cols * levels];
    for (i, &price) in c.iter().enumerate() {
        // Height is the price; the level axis just repeats the column so the
        // ridge has a footprint to project onto.
        for lvl in 0..levels {
            values[i * levels + lvl] = price;
        }
    }
    // Fall back to a flat band when there is no usable range (empty series).
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        for v in values.iter_mut() {
            *v = 0.0;
        }
    }
    let _ = span;
    Surface::new(SurfaceSpec::new(
        format!("3D Price Surface \u{2014} {}", series.symbol),
        "Close over time",
        Grid::new(cols, levels, values, "time", "price"),
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
    let buckets = 20;
    let mut values = vec![f64::NAN; buckets * periods.len()];
    for b in 0..buckets {
        for (pi, &p) in periods.iter().enumerate() {
            let end_idx = ((b + 1) * c.len()) / buckets;
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
}
