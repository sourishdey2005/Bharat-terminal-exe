//! Headless render check: drives the real UI code through an egui context.
//!
//! Screen capture in this environment is unreliable (it fails for known-good
//! binaries too), and the numeric projection tests cannot show whether the
//! painter actually emits drawable geometry. This runs the actual draw closures
//! with a synthetic input and sums the *tessellated* triangles, which is exactly
//! what the GPU would receive. So "the 3D view draws something" becomes a
//! measured number rather than a screenshot someone has to eyeball.
//!
//! `egui::Context::run` needs no window and no GPU, so this works in CI.

use super::views3d;
use egui::epaint::{Primitive, TessellationOptions};
use egui::{Context, Event, RawInput, Rect, Vec2};

fn canvas() -> Rect {
    Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(1920.0, 1000.0))
}

/// Ink for one frame: how many meshes, and how much of the canvas the *surface*
/// paints.
///
/// Shape batches merge the background rect, grid lines and surface cells into
/// one mesh, so meshes cannot be separated and the background cannot be removed
/// by skipping a mesh. Instead only triangles carrying a *ramp* colour are
/// counted: the plot background, the grid lines and the axes are all neutral
/// greys, while the surface cells are the only thing coloured from the ramp.
///
/// That isolates exactly the quantity under test — did the height field paint
/// real, correctly-sized cells? The failure mode this catches is the original
/// bug, where cells were scaled by the whole canvas and overdrew the viewport;
/// here that shows up as near-total coverage.
fn ink_of(out: &egui::FullOutput, canvas: egui::Vec2) -> (usize, f32) {
    const GW: usize = 200;
    const GH: usize = 104;

    let mut meshes = 0usize;
    let mut ramp_tris = 0usize;
    let mut grid = vec![false; GW * GH];
    let mut tess = egui::epaint::Tessellator::new(
        out.pixels_per_point,
        TessellationOptions::default(),
        [out.textures_delta.set.len(), 32],
        Vec::new(),
    );

    for prim in tess.tessellate_shapes(out.shapes.clone()) {
        if let Primitive::Mesh(m) = prim.primitive {
            meshes += 1;
            for idx in m.indices.chunks(3) {
                let va = &m.vertices[idx[0] as usize];
                let vb = &m.vertices[idx[1] as usize];
                let vc = &m.vertices[idx[2] as usize];
                // Only ramp-coloured triangles: the surface cells. Greys are the
                // background, floor and axes.
                if !is_ramp_color(va.color) {
                    continue;
                }
                ramp_tris += 1;
                let (a, b, c) = (va.pos, vb.pos, vc.pos);
                if !(a.x.is_finite() && b.x.is_finite() && c.x.is_finite()) {
                    continue;
                }
                // Convert pixel bounds into *grid* indices. Doing this with raw
                // pixel values against a 200-wide grid clamps every write to the
                // last column, which silently reports zero coverage even for
                // large, valid triangles.
                let sx = canvas.x / GW as f32;
                let sy = canvas.y / GH as f32;
                let min_x = ((a.x.min(b.x).min(c.x)) / sx).floor().max(0.0) as usize;
                let max_x = ((a.x.max(b.x).max(c.x)) / sx).ceil().min(GW as f32) as usize;
                let min_y = ((a.y.min(b.y).min(c.y)) / sy).floor().max(0.0) as usize;
                let max_y = ((a.y.max(b.y).max(c.y)) / sy).ceil().min(GH as f32) as usize;
                if max_x < min_x || max_y < min_y {
                    continue;
                }
                let min_x = min_x.min(GW - 1);
                let max_x = max_x.min(GW - 1);
                let min_y = min_y.min(GH - 1);
                let max_y = max_y.min(GH - 1);
                for gy in min_y..=max_y {
                    for gx in min_x..=max_x {
                        let px = (gx as f32 + 0.5) * sx;
                        let py = (gy as f32 + 0.5) * sy;
                        if point_in_tri((px, py), a, b, c) {
                            grid[gy * GW + gx] = true;
                        }
                    }
                }
            }
        }
    }
    let covered = grid.iter().filter(|c| **c).count();
    println!("  ink: {meshes} meshes, {ramp_tris} ramp triangles");
    (meshes, covered as f32 / (GW * GH) as f32)
}

/// Whether a colour comes from one of the surface ramps rather than the greys
/// used for the background, floor grid and axes.
///
/// The ramps are blue-to-orange, teal-to-cyan and red-to-green. None of them is
/// ever near-neutral grey once shaded, whereas every chrome colour *is* neutral
/// by construction. Requiring real saturation (max-min channel spread) is what
/// separates them.
fn is_ramp_color(c: egui::Color32) -> bool {
    let [r, g, b, _] = c.to_array();
    let hi = r.max(g).max(b) as i16;
    let lo = r.min(g).min(b) as i16;
    (hi - lo) > 30
}

/// Barycentric point-in-triangle test.
fn point_in_tri(p: (f32, f32), a: egui::Pos2, b: egui::Pos2, c: egui::Pos2) -> bool {
    let d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if d.abs() < 1e-9 {
        return false;
    }
    let l1 = ((b.y - c.y) * (p.0 - c.x) + (c.x - b.x) * (p.1 - c.y)) / d;
    let l2 = ((c.y - a.y) * (p.0 - c.x) + (a.x - c.x) * (p.1 - c.y)) / d;
    let l3 = 1.0 - l1 - l2;
    l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0
}

/// Run `f` against a fresh context and report what it drew.
fn ink_for(events: Vec<Event>, f: impl FnOnce(&mut egui::Ui)) -> (usize, f32) {
    let ctx = Context::default();
    let area = canvas();
    let raw = RawInput {
        screen_rect: Some(area),
        events,
        ..Default::default()
    };
    let mut called = false;
    let out = ctx.run(raw, |ctx| {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                called = true;
                f(ui);
            });
    });
    assert!(called, "the draw closure never ran");
    ink_of(&out, area.size())
}

/// [`ink_for`] but using [`paint_of`]'s all-pixel accounting.
fn paint_for(events: Vec<Event>, f: impl FnOnce(&mut egui::Ui)) -> (usize, f32) {
    let ctx = Context::default();
    let area = canvas();
    let raw = RawInput {
        screen_rect: Some(area),
        events,
        ..Default::default()
    };
    let mut called = false;
    let out = ctx.run(raw, |ctx| {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                called = true;
                f(ui);
            });
    });
    assert!(called, "the draw closure never ran");
    paint_of(&out, area.size())
}

/// A surface that is a clean ramp, so the renderer must emit a full mesh.
fn ramp_surface(cols: usize, rows: usize) -> views3d::Surface {
    let mut values = Vec::with_capacity(cols * rows);
    for j in 0..rows {
        for i in 0..cols {
            values.push((i + j) as f64 / (cols + rows) as f64);
        }
    }
    views3d::Surface::new(views3d::SurfaceSpec::new(
        "Test Surface",
        "ramp",
        views3d::Grid::new(cols, rows, values, "x", "y"),
        views3d::RAMP_FIRE,
    ))
}

#[test]
fn test_render_surface_emits_real_geometry() {
    let surface = ramp_surface(12, 8);
    let (meshes, cover) = ink_for(vec![], |ui| {
        let rect = views3d::render_surface(ui, &surface, 0.0, 0.61548);
        assert!(
            rect.width() > 1000.0 && rect.height() > 400.0,
            "plot rect {rect:?} is too small"
        );
    });
    assert!(meshes > 0, "no meshes emitted at all");
    // A 3D height field that fills the canvas edge to edge is precisely the bug
    // this harness exists to catch; a real surface covers a solid but partial
    // share of it.
    assert!(
        cover > 0.02 && cover < 0.95,
        "surface covers {:.1}% of the canvas; projection is degenerate",
        cover * 100.0
    );
    println!(
        "render_surface: {meshes} meshes, {:.1}% canvas",
        cover * 100.0
    );
}

#[test]
fn test_every_surface_kind_renders_without_panicking_and_draws() {
    let series = bt_core::synthetic_ohlcv("RENDER", 300, 7, 100.0);
    type Builder = fn(&bt_core::OhlcvSeries) -> views3d::Surface;
    let builders: Vec<(&str, Builder)> = vec![
        ("price", views3d::price_surface),
        ("volatility", views3d::volatility_surface),
        ("return", views3d::return_surface),
        ("risk", views3d::risk_landscape),
        ("beta", views3d::beta_surface),
        ("entropy", views3d::entropy_surface),
        ("alpha", views3d::alpha_surface),
        ("signal", views3d::signal_surface),
        ("regime", views3d::regime_surface),
        ("momentum", views3d::momentum_surface),
        ("order_flow", views3d::order_flow_surface),
        ("skew_kurt", views3d::skew_kurt_surface),
        ("signal_evo", views3d::signal_evolution_surface),
        ("equity", views3d::equity_surface),
        ("var_band", views3d::var_band_surface),
        ("risk_return", views3d::risk_return_surface),
        ("eigenvalue", views3d::eigenvalue_surface),
        ("heatmap", views3d::returns_heatmap),
        ("pca", views3d::pca_surface),
    ];

    for (name, build) in builders {
        let surface = build(&series);
        let (meshes, cover) = ink_for(vec![], |ui| {
            views3d::render_surface(ui, &surface, 0.0, 0.61548);
        });
        assert!(meshes > 0, "{name}: no meshes emitted");
        assert!(
            cover > 0.01 && cover < 0.95,
            "{name}: covers {:.1}% of the canvas; projection is degenerate",
            cover * 100.0
        );
        println!(
            "{name:>14}: {meshes:>4} meshes, {:.1}% canvas",
            cover * 100.0
        );
    }
}

#[test]
fn test_draw_frame_renders_title_axes_and_legend() {
    let surface = ramp_surface(10, 6);
    let (meshes, cover) = ink_for(vec![], |ui| {
        views3d::draw_frame(ui, &surface);
    });
    // Title, subtitle, hint row, mesh, axis lines, tick labels, legend strip.
    assert!(meshes > 0, "draw_frame emitted no meshes");
    assert!(
        cover > 0.05,
        "draw_frame only covered {:.1}% of the canvas",
        cover * 100.0
    );
    println!("draw_frame: {meshes} meshes, {:.1}% canvas", cover * 100.0);
}

#[test]
fn test_rotation_sweep_renders_without_panicking() {
    let surface = ramp_surface(10, 6);
    for step in 0..8 {
        let yaw = step as f32 * 0.4;
        let (_, cover) = ink_for(vec![], |ui| {
            views3d::render_surface(ui, &surface, yaw, 0.8);
        });
        assert!(
            cover > 0.01,
            "yaw {yaw:.1}: covered only {:.1}% of the canvas; the view collapsed",
            cover * 100.0
        );
    }
}

#[test]
fn test_extreme_canvas_sizes_still_draw() {
    // A minimised window must not produce inverted geometry or a panic.
    for (w, h) in [(320.0_f32, 240.0_f32), (3840.0, 2160.0)] {
        let surface = ramp_surface(10, 6);
        let ctx = Context::default();
        let rect = Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(w, h));
        let raw = RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        };
        let out = ctx.run(raw, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::none())
                .show(ctx, |ui| {
                    views3d::render_surface(ui, &surface, 0.4, 0.6);
                });
        });
        let (_, cover) = ink_of(&out, rect.size());
        assert!(
            cover > 0.001,
            "{w}x{h}: covered only {:.2}% at this size",
            cover * 100.0
        );
    }
}

#[test]
fn test_pointer_over_the_plot_does_not_crash() {
    // `draw_frame` registers a drag interaction; exercise it with a real pointer.
    let surface = ramp_surface(8, 5);
    let events = vec![
        Event::PointerMoved(egui::pos2(600.0, 400.0)),
        Event::PointerButton {
            pos: egui::pos2(600.0, 400.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        },
        Event::PointerMoved(egui::pos2(660.0, 430.0)),
        Event::PointerButton {
            pos: egui::pos2(660.0, 430.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        },
    ];
    let (meshes, cover) = ink_for(events, |ui| {
        views3d::draw_frame(ui, &surface);
    });
    assert!(meshes > 0, "drag frame emitted no meshes");
    assert!(
        cover > 0.05,
        "drag frame covered only {:.1}% of the canvas",
        cover * 100.0
    );
    println!("drag frame: {meshes} meshes, {:.1}% canvas", cover * 100.0);
}

// ---------------------------------------------------------------------------
// v4.2 chart and panel render tests.
//
// These assert that the panels put *pixels* on the screen, not merely that they
// return without panicking. A plot that renders an empty plot area looks
// identical to a working one in a unit test that only checks for absence of
// panic, which is why the existing ink_of harness is used here.
// ---------------------------------------------------------------------------

/// Like [`ink_of`] but counts *every* painted pixel, not just saturated ones.
///
/// [`ink_of`] deliberately ignores greys so it can isolate the 3D surface cells
/// from the plot chrome. That is wrong for a line chart, where the whole subject
/// is a two-pixel stroke, and for a text panel, which is entirely grey. This
/// variant answers "did anything get drawn at all", which is the right question
/// for those two.
fn paint_of(out: &egui::FullOutput, canvas: egui::Vec2) -> (usize, f32) {
    let mut meshes = 0usize;
    let mut painted = 0usize;
    let mut tris = 0usize;
    let mut tess = egui::epaint::Tessellator::new(
        out.pixels_per_point,
        TessellationOptions::default(),
        [out.textures_delta.set.len(), 32],
        Vec::new(),
    );
    for prim in tess.tessellate_shapes(out.shapes.clone()) {
        if let Primitive::Mesh(m) = prim.primitive {
            meshes += 1;
            // A triangle larger than roughly 400px^2 is plot chrome (background,
            // panel rect), not the content. Subtracting that keeps "coverage"
            // meaningful for a chart or a panel of text.
            for idx in m.indices.chunks(3) {
                let a = m.vertices[idx[0] as usize].pos;
                let b = m.vertices[idx[1] as usize].pos;
                let c = m.vertices[idx[2] as usize].pos;
                if !(a.x.is_finite() && b.x.is_finite() && c.x.is_finite()) {
                    continue;
                }
                let area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs();
                if area > 400.0 {
                    continue;
                }
                tris += 1;
                painted += area as usize;
            }
        }
    }
    let total = (canvas.x * canvas.y) as usize;
    println!("  paint: {meshes} meshes, {tris} content triangles");
    (meshes, painted as f32 / total as f32)
}

/// A close series with a rise, a real drawdown, and a recovery.
fn volatile_closes(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut p = 100.0;
    for i in 0..n {
        // Two cycles of up 4%, down 18%, then a slow recovery.
        let phase = i % 60;
        let r = if phase < 30 { 0.004 } else { -0.008 };
        p *= 1.0 + r;
        if i == 90 {
            p *= 0.55; // a sharp, unmistakable crash
        }
        out.push(p);
    }
    out
}

fn closes_to_candles(closes: &[f64]) -> Vec<bt_core::Candle> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &c)| bt_core::Candle::new(i as f64, c, c * 1.002, c * 0.998, c, 1000.0 + i as f64))
        .collect()
}

#[test]
fn the_underwater_plot_renders_a_visible_area() {
    let closes = volatile_closes(200);
    let candles = closes_to_candles(&closes);
    let (meshes, cover) = paint_for(vec![], |ui| {
        let uw = bt_analytics::drawdown_recovery::from_candles(&candles)
            .expect("underwater geometry");
        // The deepest drawdown here is the scripted 45% crash, so the series
        // must actually reach a large negative value for the chart to be worth
        // drawing.
        assert!(uw.max_depth() > 0.4, "fixture too shallow: {}", uw.max_depth());
        assert!(!uw.episodes.is_empty(), "fixture produced no episodes");

        egui_plot::Plot::new("uw_test")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(300.0)
            .show(ui, |plot_ui| {
                let pts: egui_plot::PlotPoints = uw
                    .series
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(i, v)| [i as f64, *v * 100.0])
                    .collect();
                plot_ui.line(egui_plot::Line::new(pts).color(egui::Color32::RED).width(2.0));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(egui::Color32::GRAY));
            });
    });
    assert!(meshes > 0, "underwater plot emitted no meshes");
    // A line chart paints far less than a filled surface; the threshold is
    // deliberately low and exists to catch "nothing was drawn at all".
    assert!(cover > 0.0005, "underwater plot drew almost nothing ({:.4}%)", cover * 100.0);
    println!("underwater plot: {meshes} meshes, {:.4}% painted", cover * 100.0);
}

#[test]
fn the_forecast_cone_renders_all_four_bands() {
    use bt_analytics::forecast_cone as fc;
    let horizon = 24usize;
    let median: Vec<f64> = (0..horizon).map(|i| 100.0 + i as f64 * 0.4).collect();
    let (p10, p25, p50, p75, p90) = fc::synthetic_quantiles(&median, 3.0).expect("quantiles");
    let cone = fc::build_cone(&p10, &p25, &p50, &p75, &p90).expect("cone");
    assert_eq!(cone.bands.len(), 4);

    // Count how many distinct shaded polygons actually reach the tessellator.
    let mut shaded = 0usize;
    let (meshes, cover) = ink_for(vec![], |ui| {
        egui_plot::Plot::new("cone_test")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(300.0)
            .show(ui, |plot_ui| {
                for (band, alpha) in cone.bands.iter().zip([0.10_f32, 0.16, 0.22, 0.30]) {
                    // The ring arrives already ordered: lower edge forward,
                    // upper edge back, then the closing vertex.
                    let pts: egui_plot::PlotPoints = band
                        .ring
                        .iter()
                        .map(|(x, y)| [*x, *y])
                        .collect();
                    plot_ui.polygon(
                        egui_plot::Polygon::new(pts)
                            .fill_color(egui::Color32::from_rgb(0, 191, 255).gamma_multiply(alpha)),
                    );
                    shaded += 1;
                }
                let median_pts: egui_plot::PlotPoints = cone
                    .median
                    .iter()
                    .enumerate()
                    .map(|(i, v)| [i as f64, *v])
                    .collect();
                plot_ui.line(egui_plot::Line::new(median_pts).color(egui::Color32::WHITE));
            });
    });
    assert_eq!(shaded, 4, "not every band was submitted");
    assert!(meshes > 0, "cone emitted no meshes");
    assert!(cover > 0.10, "cone covered only {:.1}%", cover * 100.0);
    println!("cone: {meshes} meshes, {:.1}% canvas, {shaded} bands", cover * 100.0);
}

#[test]
fn the_advisor_panel_renders_on_real_data() {
    let closes = volatile_closes(150);
    let candles = closes_to_candles(&closes);
    let (meshes, cover) = paint_for(vec![], |ui| {
        let view = crate::panel_advisor::AdvisorView {
            symbol: "RELIANCE.NS",
            candles: &candles,
            forecast_change_pct: Some(1.5),
            watchsignal: None,
            capital: crate::panel_advisor::EXAMPLE_CAPITAL,
        };
        crate::panel_advisor::draw(ui, &view);
    });
    assert!(meshes > 0, "advisor panel emitted no meshes");
    assert!(cover > 0.0002, "advisor panel drew almost nothing ({:.4}%)", cover * 100.0);
    println!("advisor panel: {meshes} meshes, {:.4}% painted", cover * 100.0);
}

#[test]
fn the_advisor_panel_renders_on_empty_input_without_panicking() {
    // A chart that has not loaded is the state a user sees first, so it must
    // render the WAIT verdict rather than dividing by a zero history length.
    let candles: Vec<bt_core::Candle> = Vec::new();
    let (meshes, _) = ink_for(vec![], |ui| {
        let view = crate::panel_advisor::AdvisorView {
            symbol: "NEW.NS",
            candles: &candles,
            forecast_change_pct: None,
            watchsignal: None,
            capital: 0.0,
        };
        crate::panel_advisor::draw(ui, &view);
    });
    assert!(meshes > 0, "empty advisor panel emitted no meshes");
}
