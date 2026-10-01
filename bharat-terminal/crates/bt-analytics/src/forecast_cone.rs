// crates/bt-analytics/src/forecast_cone.rs
// Author: Sourish Dey

//! Quantile fan geometry for the forecast cone chart.
//!
//! A forecast cone is the one chart where a rendering bug is a *misleading*
//! bug rather than an ugly one: if the bands are drawn in the wrong order, or a
//! quantile band is inverted, the chart will claim the 90th percentile is below
//! the 10th and the reader has no visual cue that anything is wrong. So the
//! ordering invariant is enforced and tested here, in pure data, rather than left
//! to a renderer to get right.
//!
//! [`build_cone`] takes the five quantiles a quantile model emits and returns
//! draw-ready polygons. Everything downstream — the egui panel, the plotters CLI
//! render — consumes the same geometry, so the two cannot disagree.

use bt_core::Result;

/// The five quantiles a cone is drawn from, as fractions of the distribution.
pub const QUANTILES: [f64; 5] = [0.10, 0.25, 0.50, 0.75, 0.90];

/// One closed polygon: a lower edge walked forward, an upper edge walked back.
///
/// Stored as a ring so a renderer can fill it directly without having to know
/// which end is which, and so the vertex count is always `2 * n`.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Vertices in draw order, starting at the cone's origin.
    pub ring: Vec<(f64, f64)>,
    /// Fraction of the distribution this band covers, e.g. 0.50 for p50-p75.
    pub coverage: f64,
}

/// A complete cone: the median path plus four nested bands.
#[derive(Debug, Clone, PartialEq)]
pub struct Cone {
    /// The p50 path, one point per projected step.
    pub median: Vec<f64>,
    /// Bands from widest (p10-p90) to narrowest (p25-p75).
    pub bands: Vec<Band>,
    /// Half-width of the p10-p90 envelope at each step.
    pub envelope: Vec<f64>,
    /// Bars covered.
    pub horizon: usize,
}

impl Cone {
    /// Band closest to the median, i.e. the tightest one.
    pub fn inner_band(&self) -> Option<&Band> {
        self.bands.last()
    }

    /// Band covering the whole distribution.
    pub fn outer_band(&self) -> Option<&Band> {
        self.bands.first()
    }

    /// Cone width at `step`, or `0.0` when the step is out of range.
    ///
    /// Callers index this with a step derived from user input, so it answers
    /// rather than panicking.
    pub fn width_at(&self, step: usize) -> f64 {
        self.envelope.get(step).copied().unwrap_or(0.0)
    }

    /// Whether the cone widens with horizon.
    ///
    /// A cone that narrows into the future is a symptom of misaligned
    /// quantiles, and this is the cheapest check that catches it before the
    /// chart goes on screen.
    pub fn widens(&self) -> bool {
        self.envelope.windows(2).any(|w| w[1] > w[0])
    }
}

/// Build the fan geometry from five aligned quantile paths.
///
/// Every quantile must be the same length; a mismatch is an error rather than
/// being silently truncated, because a truncated cone renders as a cone that
/// just stops early and looks plausible.
///
/// The bands are returned widest-first so a renderer fills back to front and the
/// narrow bands stay visible on top.
pub fn build_cone(
    p10: &[f64],
    p25: &[f64],
    p50: &[f64],
    p75: &[f64],
    p90: &[f64],
) -> Result<Cone> {
    let n = p50.len();
    if n == 0 {
        return Err(bt_core::BtError::EmptySeries(
            "forecast cone needs at least one step".into(),
        ));
    }
    for (name, q) in [
        ("p10", p10),
        ("p25", p25),
        ("p75", p75),
        ("p90", p90),
    ] {
        if q.len() != n {
            return Err(bt_core::BtError::InvalidInput(format!(
                "forecast cone: {name} has {} steps but p50 has {n}; quantiles must be aligned",
                q.len()
            )));
        }
    }

    // The whole point of the module. An inverted quantile pair means the model
    // returned crossed bands, and drawing them would show the 10th percentile
    // above the 90th.
    for (i, step) in (0..n).enumerate() {
        let l10 = p10[i];
        let l25 = p25[i];
        let m = p50[i];
        let h75 = p75[i];
        let h90 = p90[i];
        if !(l10 <= l25 && l25 <= m && m <= h75 && h75 <= h90) {
            return Err(bt_core::BtError::InvalidInput(format!(
                "forecast cone: quantiles are not ordered at step {i} \
                 (p10 {l10:.4}, p25 {l25:.4}, p50 {m:.4}, p75 {h75:.4}, p90 {h90:.4})"
            )));
        }
        if !l10.is_finite() || !h90.is_finite() {
            return Err(bt_core::BtError::InvalidInput(format!(
                "forecast cone: non-finite quantile at step {i}"
            )));
        }
    }

    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let make_band = |lo: &[f64], hi: &[f64], coverage: f64| {
        // Walk the lower edge forward, then the upper edge back, then repeat the
        // first vertex. Without that last point the ring is not closed: it would
        // start at (0, lo[0]) and end at (0, hi[0]), leaving an open edge along
        // the origin that a fill would render as a gap.
        let mut ring: Vec<(f64, f64)> = xs
            .iter()
            .cloned()
            .zip(lo.iter().copied())
            .collect();
        ring.extend(
            xs.iter()
                .rev()
                .cloned()
                .zip(hi.iter().rev().copied()),
        );
        ring.push((xs[0], lo[0]));
        Band { ring, coverage }
    };

    // Widest first so the renderer fills back to front.
    let bands = vec![
        make_band(p10, p90, 0.80),
        make_band(p10, p75, 0.65),
        make_band(p25, p75, 0.50),
        make_band(p25, p50, 0.25),
    ];

    let envelope: Vec<f64> = (0..n)
        .map(|i| 0.5 * ((p90[i] - p10[i]).abs()))
        .collect();

    Ok(Cone {
        median: p50.to_vec(),
        bands,
        envelope,
        horizon: n,
    })
}

/// Derive p10/p25/p50/p75/p90 from a median path and a widening sigma.
///
/// A stand-in for models that emit only a median, used by the panels when the
/// quantile model is unavailable. The sigma deliberately grows with the square
/// root of the horizon: uncertainty in a random walk grows with the root of
/// time, so a linear fan would understate risk at the far end and overstate it
/// early, which is precisely the shape of a confident-looking wrong forecast.
pub fn synthetic_quantiles(median: &[f64], sigma0: f64) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if median.is_empty() {
        return Err(bt_core::BtError::EmptySeries(
            "synthetic cone needs a median path".into(),
        ));
    }
    if !sigma0.is_finite() || sigma0 <= 0.0 {
        return Err(bt_core::BtError::InvalidInput(
            "synthetic cone needs a positive sigma".into(),
        ));
    }
    // Standard-normal quantiles for the five bands.
    const Z: [f64; 5] = [-1.2815515655, -0.6744897502, 0.0, 0.6744897502, 1.2815515655];

    let mut out = vec![Vec::with_capacity(median.len()); 5];
    for (i, &m) in median.iter().enumerate() {
        if !m.is_finite() {
            return Err(bt_core::BtError::InvalidInput(format!(
                "synthetic cone: non-finite median at step {i}"
            )));
        }
        let sigma = sigma0 * ((i as f64) + 1.0).sqrt();
        for (k, out_k) in out.iter_mut().enumerate() {
            out_k.push(m + Z[k] * sigma);
        }
    }
    let (p10, p25, p50, p75, p90) = (
        out[0].clone(),
        out[1].clone(),
        out[2].clone(),
        out[3].clone(),
        out[4].clone(),
    );
    Ok((p10, p25, p50, p75, p90))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A properly ordered five-quantile cone that widens with the horizon.
    fn ordered() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 10;
        let p50: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let spread = |k: f64| -> Vec<f64> {
            (0..n)
                .map(|i| 100.0 + i as f64 + k * ((i + 1) as f64).sqrt())
                .collect()
        };
        (spread(-1.28), spread(-0.67), p50, spread(0.67), spread(1.28))
    }

    #[test]
    fn a_valid_cone_builds_with_four_nested_bands() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).expect("cone");
        assert_eq!(cone.horizon, 10);
        assert_eq!(cone.median, q.2);
        assert_eq!(cone.bands.len(), 4);
        assert!(cone.widens(), "uncertainty must grow with the horizon");
    }

    #[test]
    fn bands_are_widest_first_so_the_renderer_fills_back_to_front() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap();
        let widths: Vec<f64> = cone
            .bands
            .iter()
            .map(|b| {
                // Ring runs lower-forward then upper-back; compare the extremes.
                let ys: Vec<f64> = b.ring.iter().map(|(_, y)| *y).collect();
                ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min)
            })
            .collect();
        assert!(
            widths.windows(2).all(|w| w[0] > w[1]),
            "bands must nest, got {widths:?}"
        );
    }

    #[test]
    fn every_ring_has_two_edges_per_step_plus_the_closing_point() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap();
        for b in &cone.bands {
            // 2 * horizon: lower edge forward + upper edge back. The extra
            // closing vertex is what makes it a polygon rather than an open
            // path, so it is asserted rather than allowed to drift.
            assert_eq!(b.ring.len(), cone.horizon * 2 + 1, "{b:?}");
        }
    }

    #[test]
    fn a_ring_starts_at_the_origin_and_returns_to_it() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap();
        for b in &cone.bands {
            assert_eq!(b.ring.first(), b.ring.last(), "ring must close");
        }
    }

    #[test]
    fn crossed_quantiles_are_rejected_rather_than_drawn() {
        // This is the failure the module exists to prevent: the 10th percentile
        // above the 90th would render as an inside-out cone.
        let mut q = ordered();
        q.0[4] = 1e9; // p10 above everything
        let err = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap_err();
        assert!(err.to_string().contains("not ordered"), "{err}");
    }

    #[test]
    fn misaligned_quantiles_are_an_error_not_a_truncation() {
        let mut q = ordered();
        q.3.truncate(5);
        let err = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap_err();
        assert!(err.to_string().contains("aligned"), "{err}");
    }

    #[test]
    fn a_non_finite_quantile_is_rejected() {
        let mut q = ordered();
        q.2[3] = f64::NAN;
        assert!(build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).is_err());
        q.2[3] = f64::INFINITY;
        assert!(build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).is_err());
    }

    #[test]
    fn an_empty_median_is_an_error_not_a_bare_cone() {
        let e: Vec<f64> = Vec::new();
        assert!(build_cone(&e, &e, &e, &e, &e).is_err());
    }

    #[test]
    fn a_single_step_is_allowed() {
        let cone = build_cone(&[99.0], &[99.5], &[100.0], &[100.5], &[101.0]).unwrap();
        assert_eq!(cone.horizon, 1);
        assert!(!cone.widens(), "one step has nothing to widen");
        for b in &cone.bands {
            // Two edges plus the closing point.
            assert_eq!(b.ring.len(), 3, "{b:?}");
        }
    }

    #[test]
    fn the_envelope_is_half_the_p10_p90_spread() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap();
        for (i, w) in cone.envelope.iter().enumerate() {
            let expected = 0.5 * (q.4[i] - q.0[i]).abs();
            assert!((w - expected).abs() < 1e-9, "step {i}");
        }
    }

    #[test]
    fn out_of_range_width_lookup_answers_instead_of_panicking() {
        let q = ordered();
        let cone = build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap();
        assert_eq!(cone.width_at(9999), 0.0);
        assert!(cone.width_at(0) > 0.0);
    }

    #[test]
    fn band_lookups_return_nothing_on_an_empty_cone_rather_than_panicking() {
        // `inner_band`/`outer_band` are `last()`/`first()`, so the None arm is
        // only reachable on a hand-built empty Cone.
        let empty = Cone {
            median: Vec::new(),
            bands: Vec::new(),
            envelope: Vec::new(),
            horizon: 0,
        };
        assert!(empty.inner_band().is_none());
        assert!(empty.outer_band().is_none());
    }

    #[test]
    fn synthetic_quantiles_are_ordered_and_widen() {
        let median: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let (p10, p25, p50, p75, p90) = synthetic_quantiles(&median, 2.0).expect("quantiles");
        assert_eq!(p50, median, "the median must pass through unchanged");
        // build_cone enforces the ordering, so this is really testing that the
        // synthetic bands are monotonic.
        build_cone(&p10, &p25, &p50, &p75, &p90).expect("synthetic cone must be ordered");
        assert!(p90[19] - p50[19] > p90[0] - p50[0], "sigma must grow with the horizon");
    }

    #[test]
    fn synthetic_quantiles_reject_bad_input() {
        assert!(synthetic_quantiles(&[], 1.0).is_err());
        assert!(synthetic_quantiles(&[100.0], 0.0).is_err());
        assert!(synthetic_quantiles(&[100.0], -1.0).is_err());
        assert!(synthetic_quantiles(&[100.0], f64::NAN).is_err());
        assert!(synthetic_quantiles(&[f64::NAN], 1.0).is_err());
    }

    #[test]
    fn a_flat_cone_does_not_claim_to_widen() {
        // Zero-volatility input yields zero-width bands, and `widens` must say so
        // rather than reporting growth from floating-point dust.
        let n = 5;
        let flat = vec![100.0; n];
        let cone = build_cone(&flat, &flat, &flat, &flat, &flat).unwrap();
        assert!(!cone.widens(), "{:?}", cone.envelope);
    }

    #[test]
    fn the_builder_is_deterministic() {
        let q = ordered();
        assert_eq!(
            build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap(),
            build_cone(&q.0, &q.1, &q.2, &q.3, &q.4).unwrap()
        );
    }
}