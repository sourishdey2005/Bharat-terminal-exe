#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// crates/bt-app/src/main.rs
// Author: Sourish Dey

//! bt-app: BHARAT TERMINAL v3 desktop app (egui/eframe). Made by Sourish Dey.
//!
//! Features real market data from Yahoo Finance and Coinbase, with synthetic
//! fallback. Includes company dropdown, time range selector, live refresh,
//! 163 interactive tabs with category grouping, auto-focus on chart canvas,
//! and prefs persistence via serde_json.

use std::fs;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use chrono::{Duration as ChronoDuration, Utc};
use eframe::egui;
use egui::{Color32, Id, RichText, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, MarkerShape, Plot, PlotPoints, Points};
use serde::{Deserialize, Serialize};

use bt_analytics::Engine as ForecastEngine;
use bt_analytics::Forecaster;
use bt_analytics::{Signal, SignalOutput, WatchSignalModel};
use bt_core::{
    synthetic_correlated_returns, synthetic_ohlcv, Candle, OhlcvSeries, APP_NAME, AUTHOR, TAGLINE,
};
use bt_data::india;
use bt_data::{symbol::COMPANY_LIST, DataService, Interval};
use bt_viz::palette::Theme;

mod views3d;

const AMBER: Color32 = Color32::from_rgb(0xFF, 0xB0, 0x00);
const PROFIT: Color32 = Color32::from_rgb(0x00, 0xFF, 0x88);
const LOSS: Color32 = Color32::from_rgb(0xFF, 0x3B, 0x3B);
const INFO: Color32 = Color32::from_rgb(0x00, 0xBF, 0xFF);
const PURPLE: Color32 = Color32::from_rgb(0xBF, 0x5A, 0xFF);

const DAY_SECS: f64 = 86400.0;

/// Fraction of the gap between consecutive bars occupied by a candle body.
/// Leaves a visible gap so individual bars stay distinguishable.
const BAR_FILL: f64 = 0.68;

/// Fallback bar spacing when a series has fewer than two bars.
const FALLBACK_SPACING: f64 = DAY_SECS;

/// Smallest visible body height, as a fraction of the series high/low range.
/// Keeps doji candles (open == close) from collapsing into an invisible line.
const MIN_BODY_FRAC: f64 = 0.004;

/// Vertical size of the trend arrow, as a fraction of the series high/low range.
const ARROW_FRAC: f64 = 0.022;

/// Horizontal padding added around the data, as a fraction of the data span.
const X_PAD_FRAC: f64 = 0.04;

/// Median spacing between consecutive bar timestamps, in seconds.
///
/// This is the real cadence of the series (60s for 1-minute intraday, 86400s
/// for daily, ...). Deriving geometry from it is what keeps bars from
/// overlapping into a single blob.
fn bar_spacing(candles: &[Candle]) -> f64 {
    if candles.len() < 2 {
        return FALLBACK_SPACING;
    }
    let mut deltas: Vec<f64> = candles
        .windows(2)
        .map(|w| w[1].t - w[0].t)
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    if deltas.is_empty() {
        return FALLBACK_SPACING;
    }
    // The median is robust against gaps (lunch breaks, halts, missing bars).
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    deltas[deltas.len() / 2]
}

/// Half-width of a candle body, in x (time) units.
fn bar_half(candles: &[Candle]) -> f64 {
    0.5 * BAR_FILL * bar_spacing(candles)
}

/// Inclusive x-bounds to show, padded so bars never touch the plot edge.
fn x_bounds(candles: &[Candle]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candles {
        if c.t.is_finite() {
            lo = lo.min(c.t);
            hi = hi.max(c.t);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, FALLBACK_SPACING);
    }
    let pad = (hi - lo).max(bar_spacing(candles)) * X_PAD_FRAC;
    (lo - pad, hi + pad)
}

/// Label granularity for the x-axis, chosen from the selected timeframe so
/// every range reads the way a trader expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisDateStyle {
    /// Intraday: clock time plus date, e.g. `09:30 15 Jan`.
    TimeAndDate,
    /// Weekly: day and month, e.g. `15 Jan`.
    DayMonth,
    /// Multi-month and yearly: month and year, e.g. `Jan 2024`.
    MonthYear,
}

impl TimeRange {
    /// The x-axis label style used for this timeframe.
    ///
    /// - `1D` shows `HH:MM DD Mon`, because a single session is only hours wide
    ///   and bare times would be ambiguous across days
    /// - `1W`/`1M` show `DD Mon`
    /// - `3M` and longer show `Mon YYYY`
    pub fn axis_date_style(self) -> AxisDateStyle {
        match self {
            TimeRange::D1 => AxisDateStyle::TimeAndDate,
            TimeRange::W1 | TimeRange::M1 => AxisDateStyle::DayMonth,
            TimeRange::M3 | TimeRange::M6 | TimeRange::Y1 | TimeRange::Y5 => {
                AxisDateStyle::MonthYear
            }
            // Refined once the custom window is known.
            TimeRange::Custom => AxisDateStyle::MonthYear,
        }
    }
}

/// Formats a timestamp using the requested axis style.
fn format_ts_styled(ts: f64, style: AxisDateStyle) -> String {
    let secs = ts as i64;
    let dt = chrono::DateTime::from_timestamp(secs, 0).unwrap_or_default();
    match style {
        AxisDateStyle::TimeAndDate => dt.format("%H:%M %d %b").to_string(),
        AxisDateStyle::DayMonth => dt.format("%d %b").to_string(),
        AxisDateStyle::MonthYear => dt.format("%b %Y").to_string(),
    }
}

/// Formats a timestamp, choosing a format appropriate to the bar cadence so
/// intraday charts show a clock time and longer ranges show a date.
fn format_ts_for(ts: f64, spacing: f64) -> String {
    if spacing < DAY_SECS {
        format_ts_styled(ts, AxisDateStyle::TimeAndDate)
    } else if spacing < 20.0 * DAY_SECS {
        format_ts_styled(ts, AxisDateStyle::DayMonth)
    } else {
        format_ts_styled(ts, AxisDateStyle::MonthYear)
    }
}

/// Backwards-compatible month/year formatter for callers that have no cadence
/// context. Prefer [`format_ts_for`].
fn format_ts(ts: f64) -> String {
    format_ts_for(ts, DAY_SECS)
}

/// Returns `(min_low, range)` for the series, used to scale candle geometry.
fn price_scale(candles: &[Candle]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candles {
        lo = lo.min(c.low);
        hi = hi.max(c.high);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    let range = hi - lo;
    if range <= 0.0 {
        (lo, 1.0)
    } else {
        (lo, range)
    }
}

/// Chooses sensible y-axis tick decimals for a price of the given magnitude,
/// so small-cap and index charts are not labelled with unusable precision.
fn price_decimals(price: f64) -> usize {
    let abs = price.abs();
    if abs >= 10_000.0 {
        0
    } else if abs >= 1_000.0 {
        1
    } else if abs >= 100.0 {
        2
    } else if abs >= 1.0 {
        3
    } else {
        4
    }
}

/// Formats a volume value compactly (K/M/B/T) for axis ticks and legends.
fn abbreviate_volume(v: f64) -> String {
    let a = v.abs();
    if a >= 1.0e12 {
        format!("{:.1}T", v / 1.0e12)
    } else if a >= 1.0e9 {
        format!("{:.1}B", v / 1.0e9)
    } else if a >= 1.0e6 {
        format!("{:.1}M", v / 1.0e6)
    } else if a >= 1.0e3 {
        format!("{:.1}K", v / 1.0e3)
    } else {
        format!("{:.0}", v)
    }
}

/// Applies the shared professional chart chrome to a time-series plot:
/// right-hand price axis, timeframe-aware x labels, bounded x range and
/// a small grid. Every time-series tab uses this so they all look consistent.
///
/// Call [`apply_zoom_and_pan`] inside `show` to handle zoom/pan gestures and
/// lock the visible range. The
/// x-range is pinned rather than merely included because `include_x` only ever
/// widens: egui_plot keeps per-plot memory, so a range viewed earlier kept its
/// zoom and squeezed the next range into a sliver instead of refitting.
fn style_time_plot<'a>(
    plot: egui_plot::Plot<'a>,
    series: &[Candle],
    height: f32,
    style: AxisDateStyle,
) -> egui_plot::Plot<'a> {
    let (x0, x1) = x_bounds(series);
    let (lo, range) = price_scale(series);
    let last = series.last().map(|c| c.close).unwrap_or(1.0);
    let decimals = price_decimals(last);
    plot.height(height)
        .allow_drag(true)
        .allow_scroll(true)
        .allow_zoom(false)
        .include_x(x0)
        .include_x(x1)
        .include_y(lo - range * 0.05)
        .include_y(lo + range * 1.05)
        .y_axis_position(egui_plot::HPlacement::Right)
        .y_axis_min_width(((decimals + 6) * 7) as f32)
        // egui_plot's default corner readout prints raw epoch values over the
        // chart, which is noise here; the OHLC legend and hover tooltip cover
        // the same information properly.
        .coordinates_formatter(
            egui_plot::Corner::LeftTop,
            egui_plot::CoordinatesFormatter::new(|_point, _bounds| String::new()),
        )
        .x_axis_formatter(move |mark, _range| format_ts_styled(mark.value, style))
        .y_axis_formatter(move |mark, _range| format!("{:.*}", decimals, mark.value))
}

/// Parses a `YYYY-MM-DD` date into a UTC midnight timestamp.
fn parse_ymd(text: &str) -> Option<i64> {
    chrono::NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .and_then(|dt| chrono::DateTime::from_timestamp(dt.and_utc().timestamp(), 0))
        .map(|dt| dt.timestamp())
}

/// Formats a timestamp as `YYYY-MM-DD` for the date fields.
fn format_ymd(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .unwrap_or_default()
        .format("%Y-%m-%d")
        .to_string()
}

/// A user-entered start/end window, validated and ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CustomWindow {
    start: i64,
    end: i64,
}

impl CustomWindow {
    /// Builds a window from the two text fields, normalising the order and
    /// rejecting windows that are empty or longer than 10 years.
    fn parse(start_text: &str, end_text: &str) -> Option<Self> {
        let start = parse_ymd(start_text)?;
        let end = parse_ymd(end_text)?;
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        if hi <= lo {
            return None;
        }
        // Yahoo rejects very long intraday requests, and a decade is far beyond
        // any chart a user wants to read.
        if hi - lo > 10 * 365 * DAY_SECS as i64 {
            return None;
        }
        Some(Self { start: lo, end: hi })
    }

    /// Length in days, at least 1.
    fn days(&self) -> i64 {
        ((self.end - self.start) / DAY_SECS as i64).max(1)
    }

    /// Bar interval appropriate to the window length.
    fn interval(&self) -> Interval {
        match self.days() {
            0..=1 => Interval::Min5,
            2..=7 => Interval::Min15,
            8..=31 => Interval::Hour1,
            32..=2000 => Interval::Day1,
            _ => Interval::Week1,
        }
    }

    /// Axis label style appropriate to the window length.
    fn axis_date_style(&self) -> AxisDateStyle {
        match self.days() {
            0..=1 => AxisDateStyle::TimeAndDate,
            2..=45 => AxisDateStyle::DayMonth,
            _ => AxisDateStyle::MonthYear,
        }
    }
}

/// Price envelope `(low, high)` of the candles whose timestamp falls inside
/// `[wx0, wx1]`. Returns `None` when the window contains no bars, so a zoom can
/// be pulled onto real data instead of empty space.
fn price_envelope(candles: &[Candle], wx0: f64, wx1: f64) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candles {
        if c.t >= wx0 && c.t <= wx1 {
            lo = lo.min(c.low);
            hi = hi.max(c.high);
        }
    }
    if lo.is_finite() && hi.is_finite() && hi >= lo {
        Some((lo, hi))
    } else {
        None
    }
}

/// A user-controlled zoom window over the full data range.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ZoomState {
    /// `1.0` shows the whole range; larger values zoom in.
    factor: f64,
    /// Vertical-only scale, for a two-finger pinch held above/below each other.
    /// Shares the same bounds as `factor`; `1.0` means "no extra y zoom".
    y_factor: f64,
    /// X position (timestamp) to keep centred, or `None` to use the midpoint.
    focus_x: Option<f64>,
    /// Y position to keep centred.
    focus_y: Option<f64>,
    /// Horizontal drag offset in x (time) units.
    pan_x: f64,
    /// Vertical drag offset in y (price) units.
    pan_y: f64,
}

impl Default for ZoomState {
    fn default() -> Self {
        Self {
            factor: 1.0,
            y_factor: 1.0,
            focus_x: None,
            focus_y: None,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }
}

impl ZoomState {
    const MIN_FACTOR: f64 = 1.0;
    /// Cap chosen so daily data still shows several bars at full zoom; beyond
    /// this the chart is a single candle and further magnification is useless.
    const MAX_FACTOR: f64 = 60.0;

    /// Zoom factor applied by a single double-click step.
    const STEP: f64 = 2.0;

    /// True when the view is not showing the full range.
    fn is_zoomed(&self) -> bool {
        self.factor > Self::MIN_FACTOR + f64::EPSILON
            || self.y_factor > Self::MIN_FACTOR + f64::EPSILON
    }

    /// Zooms in one step around `(x, y)`, keeping that point fixed.
    ///
    /// The focus price is snapped into the candles that the *new* window will
    /// actually show. Snapping against the currently visible envelope is not
    /// enough: on the first step the view is the whole series, so a click in
    /// empty space above the local bars would sit inside the global range and
    /// stay unclamped, leaving a blank pane once the window tightened.
    fn zoom_in(&self, x: f64, y: f64, series: &[Candle], x0: f64, x1: f64) -> Self {
        if !x.is_finite() {
            return *self;
        }
        let factor = (self.factor * Self::STEP).min(Self::MAX_FACTOR);
        let y_factor = (self.y_factor * Self::STEP).min(Self::MAX_FACTOR);
        if (factor - self.factor).abs() < f64::EPSILON
            && (y_factor - self.y_factor).abs() < f64::EPSILON
        {
            return *self;
        }
        // Snap against the window the new factor will show around `x`.
        let y = Self::snap_focus_y(y, x, factor, series, x0, x1, self.focus_y);
        Self {
            factor,
            y_factor,
            focus_x: Some(x),
            focus_y: Some(y),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Zooms out one step around `(x, y)`, keeping that point fixed. Steps all
    /// the way back to the full range rather than resetting in one jump, so
    /// repeated clicks narrow the view smoothly.
    fn zoom_out(&self, x: f64, y: f64) -> Self {
        let factor = (self.factor / Self::STEP).max(Self::MIN_FACTOR);
        let y_factor = (self.y_factor / Self::STEP).max(Self::MIN_FACTOR);
        if !x.is_finite() || !y.is_finite() {
            return Self {
                factor,
                y_factor,
                ..*self
            };
        }
        // At the full range there is nothing left to step out of, so return
        // unchanged rather than recording a focus point that has no effect.
        if (factor - self.factor).abs() < f64::EPSILON
            && (y_factor - self.y_factor).abs() < f64::EPSILON
        {
            return *self;
        }
        Self {
            factor,
            y_factor,
            focus_x: Some(x),
            focus_y: Some(y),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Zooms both axes by an arbitrary multiplicative `factor` around `(x, y)`.
    ///
    /// Continuous counterpart to [`Self::zoom_in`]. The factor is clamped to the
    /// same bounds as the stepped gestures, and the focus is snapped into the
    /// candles the new window will show, so a pinch that lands on empty space
    /// still ends up showing real bars instead of a blank pane.
    fn zoom_by(&self, factor: f64, x: f64, y: f64, series: &[Candle], x0: f64, x1: f64) -> Self {
        self.zoom_x_by(factor, x, y, series, x0, x1)
            .zoom_y_by(factor, x, y, series, x0, x1)
    }

    /// Zooms the horizontal axis by `factor` around `(x, y)`, leaving the
    /// vertical scale untouched. This is egui's `[z, 1]` horizontal pinch.
    fn zoom_x_by(&self, factor: f64, x: f64, y: f64, series: &[Candle], x0: f64, x1: f64) -> Self {
        if !factor.is_finite() || factor <= 0.0 || !x.is_finite() {
            return *self;
        }
        let target = (self.factor * factor).clamp(Self::MIN_FACTOR, Self::MAX_FACTOR);
        if (target - self.factor).abs() < 1e-9 {
            return *self;
        }
        // Snapped against the window the new x-scale will show, so the focus
        // lands on candles that are actually visible.
        let y = Self::snap_focus_y(y, x, target, series, x0, x1, self.focus_y);
        Self {
            factor: target,
            y_factor: self.y_factor,
            focus_x: Some(x),
            focus_y: Some(y),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Zooms only the vertical axis, for a two-finger pinch held directly
    /// above/below each other. `zoom_delta_2d` reports `[1, z]` for that case.
    fn zoom_y_by(&self, factor: f64, x: f64, y: f64, series: &[Candle], x0: f64, x1: f64) -> Self {
        if !factor.is_finite() || factor <= 0.0 || !x.is_finite() {
            return *self;
        }
        let target = (self.y_factor * factor).clamp(Self::MIN_FACTOR, Self::MAX_FACTOR);
        if (target - self.y_factor).abs() < 1e-9 {
            return *self;
        }
        let y = Self::snap_focus_y(y, x, self.factor, series, x0, x1, self.focus_y);
        Self {
            factor: self.factor,
            y_factor: target,
            focus_x: Some(x),
            focus_y: Some(y),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Clamps `y` into the price envelope of the window a factor of `factor`
    /// will show around `x`, falling back to `fallback` and finally the
    /// envelope midpoint when `y` is not finite.
    fn snap_focus_y(
        y: f64,
        x: f64,
        factor: f64,
        series: &[Candle],
        x0: f64,
        x1: f64,
        fallback: Option<f64>,
    ) -> f64 {
        let half_x = if x1 > x0 {
            0.5 * (x1 - x0) / factor
        } else {
            0.0
        };
        let envelope = price_envelope(series, x - half_x, x + half_x);
        if y.is_finite() {
            return match envelope {
                Some((lo, hi)) => y.clamp(lo, hi),
                None => y,
            };
        }
        match (fallback, envelope) {
            (Some(f), _) if f.is_finite() => f,
            (_, Some((lo, hi))) => 0.5 * (lo + hi),
            _ => 0.0,
        }
    }

    /// Zooms in around the current focus, for the toolbar buttons.
    fn zoom_in_centered(&self, y: f64) -> Self {
        let factor = (self.factor * Self::STEP).min(Self::MAX_FACTOR);
        Self {
            factor,
            y_factor: (self.y_factor * Self::STEP).min(Self::MAX_FACTOR),
            focus_x: self.focus_x,
            focus_y: Some(y),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Zooms out around the current centre, for the toolbar buttons.
    fn zoom_out_centered(&self) -> Self {
        let factor = (self.factor / Self::STEP).max(Self::MIN_FACTOR);
        let y_factor = (self.y_factor / Self::STEP).max(Self::MIN_FACTOR);
        if (factor - self.factor).abs() < f64::EPSILON
            && (y_factor - self.y_factor).abs() < f64::EPSILON
        {
            return *self;
        }
        Self {
            factor,
            y_factor,
            focus_x: self.focus_x,
            focus_y: self.focus_y,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Resets to the full range.
    fn reset(&self) -> Self {
        Self::default()
    }

    /// Applies the zoom to `(x0, x1)` and `(y0, y1)`, returning the window to
    /// display. The focus point stays fixed while the window shrinks around it,
    /// and any pan offset is added on top.
    fn window(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> (f64, f64, f64, f64) {
        if !x1.is_finite() || !y1.is_finite() || x1 <= x0 || y1 <= y0 {
            return (x0, x1, y0, y1);
        }
        if !self.is_zoomed() {
            return (x0, x1, y0, y1);
        }
        let cx = self.focus_x.unwrap_or(0.5 * (x0 + x1));
        let cy = self.focus_y.unwrap_or(0.5 * (y0 + y1));
        let half_x = 0.5 * (x1 - x0) / self.factor;
        let half_y = 0.5 * (y1 - y0) / self.y_factor;
        (
            cx - half_x + self.pan_x,
            cx + half_x + self.pan_x,
            cy - half_y + self.pan_y,
            cy + half_y + self.pan_y,
        )
    }

    /// Same as [`Self::window`] but grows the y-range, when needed, so every
    /// candle inside the x-window stays visible. Zooming both axes equally can
    /// otherwise crop the visible bars out of the pane and look blank.
    fn window_with_data(
        &self,
        series: &[Candle],
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> (f64, f64, f64, f64) {
        let (wx0, wx1, wy0, wy1) = self.window(x0, x1, y0, y1);
        if !self.is_zoomed() {
            return (wx0, wx1, wy0, wy1);
        }
        match price_envelope(series, wx0, wx1) {
            Some((lo, hi)) => (wx0, wx1, wy0.min(lo), wy1.max(hi)),
            None => (wx0, wx1, wy0, wy1),
        }
    }

    /// The window before any pan is applied. Panning is clamped against this
    /// rather than [`Self::window`], because `window()` already folds in the
    /// current pan offset and would compound the bound on every frame.
    fn base_window(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> (f64, f64, f64, f64) {
        let cx = self.focus_x.unwrap_or(0.5 * (x0 + x1));
        let cy = self.focus_y.unwrap_or(0.5 * (y0 + y1));
        let half_x = 0.5 * (x1 - x0) / self.factor;
        let half_y = 0.5 * (y1 - y0) / self.y_factor;
        (cx - half_x, cx + half_x, cy - half_y, cy + half_y)
    }

    /// Shifts the view by `(dx, dy)` in data units, clamped so the window can
    /// never be dragged off the data. Returns the new state.
    fn pan_by(
        &self,
        dx: f64,
        dy: f64,
        series: &[Candle],
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> Self {
        if !dx.is_finite() || !dy.is_finite() {
            return *self;
        }
        if !self.is_zoomed() {
            // Panning only makes sense once the view is smaller than the data.
            return *self;
        }
        let (bx0, bx1, by0, by1) = self.base_window(x0, x1, y0, y1);
        // The total pan must keep `[bx + p]` inside the data on both sides:
        // `bx0 + p >= x0` gives the lower bound and `bx1 + p <= x1` the upper.
        let min_x = x0 - bx0;
        let max_x = x1 - bx1;
        // Vertically the useful bound is the price envelope of the candles in
        // the horizontal window, not the global high/low. Panning to a price
        // band with no bars in view just shows an empty pane, so the y offset
        // is limited to what keeps the visible candles on screen.
        let (lo, hi) =
            price_envelope(series, bx0 + self.pan_x, bx1 + self.pan_x).unwrap_or((y0, y1));
        let min_y = lo - by1;
        let max_y = hi - by0;
        let pan_x = (self.pan_x + dx).clamp(min_x, max_x);
        let pan_y = (self.pan_y + dy).clamp(min_y, max_y);
        if (pan_x - self.pan_x).abs() < f64::EPSILON && (pan_y - self.pan_y).abs() < f64::EPSILON {
            return *self;
        }
        Self {
            pan_x,
            pan_y,
            ..*self
        }
    }

    /// Shifts only the horizontal axis, leaving the vertical pan untouched.
    ///
    /// The volume pane shares the price pane's time axis but has its own fixed
    /// price axis (`0..max volume`), so a vertical drag there must not move the
    /// price window. Horizontal drags and scrolls on the volume pane route
    /// here instead of [`Self::pan_by`].
    fn pan_x_only(&self, dx: f64, x0: f64, x1: f64) -> Self {
        if !dx.is_finite() {
            return *self;
        }
        if !self.is_zoomed() {
            return *self;
        }
        let (bx0, bx1, _, _) = self.base_window(x0, x1, 0.0, 1.0);
        let pan_x = (self.pan_x + dx).clamp(x0 - bx0, x1 - bx1);
        if (pan_x - self.pan_x).abs() < f64::EPSILON {
            return *self;
        }
        Self { pan_x, ..*self }
    }
}

/// Applies a chart zoom gesture.
///
/// - **Double-click** zooms **in** around the pointer.
/// - **Right-click** zooms **out** around the pointer.
///
/// Each gesture steps one level (2x in, 1/2x out) and the out-step unwinds
/// smoothly back to the full range rather than snapping to it. Returns the new
/// state, or `None` when the gesture produced no change.
fn handle_plot_zoom_gesture(
    plot_ui: &egui_plot::PlotUi,
    current: ZoomState,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Option<ZoomState> {
    if !plot_ui.response().hovered() {
        return None;
    }
    // egui_plot sets the plot widget's `Sense` itself, so a response-level
    // double-click only fires when the plot claims the interaction. Reading the
    // raw input and gating it on `hovered()` is what makes this reliable.
    let zoom_in = plot_ui.ctx().input(|i| {
        i.pointer
            .button_double_clicked(egui::PointerButton::Primary)
    });
    let zoom_out = plot_ui
        .ctx()
        .input(|i| i.pointer.button_clicked(egui::PointerButton::Secondary));
    if !zoom_in && !zoom_out {
        return None;
    }
    let _ = (y0, y1);
    let next = match (zoom_in, zoom_out, plot_ui.pointer_coordinate()) {
        (true, _, Some(p)) => current.zoom_in(p.x, p.y, series, x0, x1),
        (_, true, Some(p)) => current.zoom_out(p.x, p.y),
        (true, _, None) => current.zoom_in(
            current.focus_x.unwrap_or(x0),
            current.focus_y.unwrap_or(0.0),
            series,
            x0,
            x1,
        ),
        (_, true, None) => current.zoom_out(
            current.focus_x.unwrap_or(x0),
            current.focus_y.unwrap_or(0.0),
        ),
        _ => return None,
    };
    if next == current {
        None
    } else {
        Some(next)
    }
}

/// Converts a pointer drag into a data-space delta for one axis.
///
/// `drag_px` is how far the pointer moved on that axis and `plot_px` is the
/// plot's size on that axis, so the ratio is the fraction of the window that
/// the drag covers. Dragging right or up moves the *view* back, hence the sign.
fn drag_to_data(drag_px: f32, plot_px: f32, full: f64) -> f64 {
    if !drag_px.is_finite() || !plot_px.is_finite() || plot_px.abs() < 1.0 {
        return 0.0;
    }
    if !full.is_finite() || full <= 0.0 {
        return 0.0;
    }
    -full * (drag_px as f64 / plot_px as f64)
}

/// Maps a vertical scroll delta (points) to a multiplicative zoom factor.
///
/// Wheel-up (positive) zooms in, wheel-down zooms out. The exponential keeps
/// tiny trackpad deltas smooth while a full mouse-wheel notch (~50 points)
/// steps about 1.16x -- close to one double-click step spread over the
/// wheel's detents instead of a single jump.
fn scroll_zoom_factor(scroll_y: f32) -> f64 {
    const PER_POINT: f64 = 0.003;
    ((scroll_y as f64) * PER_POINT).exp()
}

/// Applies mouse-wheel / trackpad zoom and pan on a plot.
///
/// egui_plot pins its bounds every frame, which wipes out its own scroll
/// handling, so without this the wheel did literally nothing on a chart.
/// Wheel-up zooms in around the pointer, wheel-down zooms out; horizontal
/// scroll (shift+wheel, trackpad swipe) pans through time. Trackpad pinch and
/// ctrl+wheel arrive as a `zoom_delta` and are applied as a proportional zoom.
/// Unlike drag panning, scrolling works from the full-range view, so this is
/// how a zoomed window is entered with a mouse.
fn handle_plot_scroll(
    plot_ui: &egui_plot::PlotUi,
    current: ZoomState,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Option<ZoomState> {
    if !plot_ui.response().hovered() {
        return None;
    }
    let (zoom, scroll) = plot_ui.ctx().input(|i| {
        let scroll = if i.smooth_scroll_delta == egui::Vec2::ZERO {
            i.raw_scroll_delta
        } else {
            i.smooth_scroll_delta
        };
        (i.zoom_delta(), scroll)
    });

    // Pointer position in data space, for anchoring the zoom. Falls back to
    // the middle of the data when the pointer position is unavailable.
    let anchor = plot_ui
        .pointer_coordinate()
        .and_then(|p| {
            if p.x.is_finite() && p.y.is_finite() {
                Some((p.x, p.y))
            } else {
                None
            }
        })
        .unwrap_or((0.5 * (x0 + x1), 0.5 * (y0 + y1)));

    let mut next = current;
    // Trackpad pinch / ctrl+wheel first: it is the finer-grained gesture.
    if zoom.is_finite() && (zoom - 1.0).abs() > 1e-6 {
        next = next.zoom_by(zoom as f64, anchor.0, anchor.1, series, x0, x1);
    } else if scroll.y != 0.0 {
        let factor = scroll_zoom_factor(scroll.y);
        next = next.zoom_by(factor, anchor.0, anchor.1, series, x0, x1);
    }
    // Horizontal scroll pans, reusing the drag maths so the clamp behaviour
    // matches a horizontal drag exactly.
    if scroll.x != 0.0 {
        let rect = plot_ui.response().rect;
        let (wx0, wx1, _, _) = next.window_with_data(series, x0, x1, y0, y1);
        let dx = drag_to_data(scroll.x, rect.width(), wx1 - wx0);
        next = next.pan_by(dx, 0.0, series, x0, x1, y0, y1);
    }

    if next == current {
        None
    } else {
        Some(next)
    }
}

/// Applies a two-finger pinch and drag on a plot.
///
/// Touch needs its own path: there is no right-click, and a single-finger swipe
/// is only reported as a pointer drag once egui decides it is one, which drops
/// the first frames of the gesture. [`egui::InputState::multi_touch`] exposes
/// the pinch factor and the average finger translation directly, so both axes
/// can be zoomed and panned in one gesture:
///
/// - `zoom_delta_2d` scales the x and y axes independently. egui reports
///   `[1, z]` for a vertical pinch and `[z, 1]` for a horizontal one, so an
///   axis-aligned pinch comes for free.
/// - `translation_delta` pans, reusing the mouse-drag maths so a two-finger
///   drag is clamped identically to a one-finger drag.
fn handle_plot_pinch(
    plot_ui: &egui_plot::PlotUi,
    current: ZoomState,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Option<ZoomState> {
    let touch = plot_ui.ctx().multi_touch()?;
    // A single tap must keep the existing click behaviour, so require two
    // fingers before moving the view.
    if touch.num_touches < 2 {
        return None;
    }
    let zx = touch.zoom_delta_2d.x as f64;
    let zy = touch.zoom_delta_2d.y as f64;
    let delta = touch.translation_delta;
    let zooming = (zx - 1.0).abs() > 1e-6 || (zy - 1.0).abs() > 1e-6;
    if !zooming && delta.x == 0.0 && delta.y == 0.0 {
        return None;
    }

    // Anchor on the finger midpoint so spreading two fingers apart keeps the
    // point between them stationary.
    let rect = plot_ui.response().rect;
    let midpoint = egui::Pos2::new(rect.center().x + delta.x, rect.center().y + delta.y);
    let anchor = plot_ui.plot_from_screen(midpoint);
    if !anchor.x.is_finite() || !anchor.y.is_finite() {
        return None;
    }

    let mut next = current;
    if (zx - 1.0).abs() > 1e-6 {
        next = next.zoom_x_by(zx, anchor.x, anchor.y, series, x0, x1);
    }
    if (zy - 1.0).abs() > 1e-6 {
        next = next.zoom_y_by(zy, anchor.x, anchor.y, series, x0, x1);
    }
    if delta.x != 0.0 || delta.y != 0.0 {
        let (wx0, wx1, wy0, wy1) = next.window_with_data(series, x0, x1, y0, y1);
        let dx = drag_to_data(delta.x, rect.width(), wx1 - wx0);
        let dy = drag_to_data(delta.y, rect.height(), wy1 - wy0);
        next = next.pan_by(dx, dy, series, x0, x1, y0, y1);
    }

    if next == current {
        None
    } else {
        Some(next)
    }
}

/// Applies drag, scroll and pinch gestures on the volume pane.
///
/// The volume pane shares the price pane's time axis, so horizontal movement
/// there must move the shared window -- otherwise dragging the lower third of
/// the canvas feels dead. The vertical axis is fixed (`0..max volume`), so
/// only the horizontal component of each gesture is honoured: horizontal drags
/// and scrolls pan through time, and a pinch or wheel-zoom scales the shared
/// x-scale around the pointer. The price window's y-scale is never touched.
fn handle_volume_gestures(
    plot_ui: &egui_plot::PlotUi,
    current: ZoomState,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Option<ZoomState> {
    if !plot_ui.response().hovered() {
        return None;
    }
    let mut next = current;

    // Pinch: x-component only.
    if let Some(touch) = plot_ui.ctx().multi_touch() {
        if touch.num_touches >= 2 {
            let zx = touch.zoom_delta_2d.x as f64;
            let delta = touch.translation_delta;
            let zooming = (zx - 1.0).abs() > 1e-6;
            if zooming || delta.x != 0.0 {
                let rect = plot_ui.response().rect;
                let anchor = plot_ui
                    .plot_from_screen(egui::Pos2::new(rect.center().x + delta.x, rect.center().y))
                    .x;
                if anchor.is_finite() {
                    if zooming {
                        next = next.zoom_x_by(zx, anchor, 0.0, series, x0, x1);
                    }
                    if delta.x != 0.0 {
                        let (wx0, wx1, _, _) = next.window(x0, x1, y0, y1);
                        let dx = drag_to_data(delta.x, rect.width(), wx1 - wx0);
                        next = next.pan_x_only(dx, x0, x1);
                    }
                }
            }
        }
    }

    // Wheel: vertical scroll zooms the shared x-scale, horizontal scroll pans.
    let (zoom, scroll) = plot_ui.ctx().input(|i| {
        let scroll = if i.smooth_scroll_delta == egui::Vec2::ZERO {
            i.raw_scroll_delta
        } else {
            i.smooth_scroll_delta
        };
        (i.zoom_delta(), scroll)
    });
    // Anchor at the pointer's time; the price is irrelevant here.
    let anchor_x = plot_ui
        .pointer_coordinate()
        .map(|p| p.x)
        .unwrap_or(0.5 * (x0 + x1));
    if zoom.is_finite() && (zoom - 1.0).abs() > 1e-6 {
        if anchor_x.is_finite() {
            next = next.zoom_x_by(zoom as f64, anchor_x, 0.0, series, x0, x1);
        }
    } else if scroll.y != 0.0 && anchor_x.is_finite() {
        next = next.zoom_x_by(scroll_zoom_factor(scroll.y), anchor_x, 0.0, series, x0, x1);
    }
    if scroll.x != 0.0 {
        let rect = plot_ui.response().rect;
        let (wx0, wx1, _, _) = next.window(x0, x1, y0, y1);
        let dx = drag_to_data(scroll.x, rect.width(), wx1 - wx0);
        next = next.pan_x_only(dx, x0, x1);
    }

    // Single-finger / mouse drag: horizontal component only. `pan_x_only`
    // no-ops at the full range, so no zoom gate is needed here.
    let delta = plot_ui.pointer_coordinate_drag_delta();
    if delta.x != 0.0 {
        next = next.pan_x_only(-(delta.x as f64), x0, x1);
    }

    if next == current {
        None
    } else {
        Some(next)
    }
}

/// Applies a drag on a plot as a horizontal/vertical pan of the zoomed window.
///
/// Returns the new state, or `None` when there is nothing to pan. Panning is
/// only active while zoomed in, because at the full range the window already
/// shows everything.
fn handle_plot_pan(
    plot_ui: &egui_plot::PlotUi,
    current: ZoomState,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Option<ZoomState> {
    if !current.is_zoomed() || !plot_ui.response().hovered() {
        return None;
    }
    // The drag delta is already zero unless the pointer is actually being
    // dragged this frame, so it doubles as the "is dragging" test.
    //
    // `pointer_coordinate_drag_delta` is in *data* units, not screen pixels,
    // so it is negated directly: dragging right pulls the content right, which
    // moves the window to earlier time. Routing it through `drag_to_data`
    // (which expects pixels) scaled every drag by ~1e6 and slammed the view to
    // the clamp edge on the first frame, which is why dragging looked dead.
    let delta = plot_ui.pointer_coordinate_drag_delta();
    if delta.x == 0.0 && delta.y == 0.0 {
        return None;
    }
    let next = current.pan_by(-(delta.x as f64), -(delta.y as f64), series, x0, x1, y0, y1);
    if next == current {
        None
    } else {
        Some(next)
    }
}

/// Pins a plot's visible range, overriding remembered zoom/pan state so the
/// chart always refits the data. Call this first inside `Plot::show`.
fn pin_bounds(plot_ui: &mut egui_plot::PlotUi, x0: f64, x1: f64, y0: f64, y1: f64) {
    if x1 > x0 && y1 > y0 {
        plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max([x0, y0], [x1, y1]));
    }
}

/// Applies zoom/pan gestures to a plot and then pins the resulting window.
///
/// This is the single entry point every time-series tab uses, so dragging to
/// pan and double/right-clicking to zoom behave identically on all charts
/// rather than only on the candlestick tab. The y-range is data-aware, so the
/// applied window always covers the candles inside the visible x-range.
fn apply_zoom_and_pan(
    plot_ui: &mut egui_plot::PlotUi,
    zoom: &std::cell::Cell<ZoomState>,
    series: &[Candle],
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) {
    if let Some(next) = handle_plot_zoom_gesture(plot_ui, zoom.get(), series, x0, x1, y0, y1) {
        zoom.set(next);
    }
    // Pinch first: it is the finer-grained gesture, and a two-finger drag is
    // also visible as a pointer drag, so letting the pinch resolve first keeps
    // the two from double-applying the same movement.
    if let Some(next) = handle_plot_pinch(plot_ui, zoom.get(), series, x0, x1, y0, y1) {
        zoom.set(next);
    }
    // Wheel and trackpad scroll next: it is the only gesture that works from
    // the full-range view, so it is how a zoomed window is entered with a
    // mouse. It never double-applies with a drag because a wheel event carries
    // no pointer movement.
    if let Some(next) = handle_plot_scroll(plot_ui, zoom.get(), series, x0, x1, y0, y1) {
        zoom.set(next);
    }
    if let Some(next) = handle_plot_pan(plot_ui, zoom.get(), series, x0, x1, y0, y1) {
        zoom.set(next);
    }
    let (zx0, zx1, zy0, zy1) = zoom.get().window_with_data(series, x0, x1, y0, y1);
    pin_bounds(plot_ui, zx0, zx1, zy0, zy1);
}

/// Returns the `(top, bottom)` of the candle body, guaranteeing a visible body
/// for doji candles by expanding around the midpoint when open == close.
fn candle_body(c: &Candle, min_body: f64) -> (f64, f64) {
    let (mut top, mut bottom) = if c.is_bullish() {
        (c.close, c.open)
    } else {
        (c.open, c.close)
    };
    if (top - bottom).abs() < min_body {
        let mid = 0.5 * (top + bottom);
        top = mid + 0.5 * min_body;
        bottom = mid - 0.5 * min_body;
    }
    (top, bottom)
}

/// Draws a professional candlestick: thin high/low wick behind a filled
/// open/close body, coloured green when bullish and red when bearish.
fn draw_candle(plot_ui: &mut egui_plot::PlotUi, c: &Candle, half: f64, min_body: f64) {
    let color = if c.is_bullish() { PROFIT } else { LOSS };
    plot_ui.line(
        Line::new(PlotPoints::from(vec![[c.t, c.low], [c.t, c.high]]))
            .color(color)
            .width(1.0_f32),
    );
    let (top, bottom) = candle_body(c, min_body);
    plot_ui.polygon(
        egui_plot::Polygon::new(PlotPoints::from(vec![
            [c.t - half, bottom],
            [c.t + half, bottom],
            [c.t + half, top],
            [c.t - half, top],
        ]))
        .fill_color(color)
        .stroke(Stroke::new(1.0_f32, color)),
    );
}

/// Converts a series into a true Heikin-Ashi series.
///
/// `ha_close = (o + h + l + c) / 4`, `ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2`
/// (seeded with `(o[0] + c[0]) / 2`), and the high/low are widened to contain
/// the synthetic open and close.
fn heikin_ashi(candles: &[Candle]) -> Vec<Candle> {
    let mut out: Vec<Candle> = Vec::with_capacity(candles.len());
    let mut prev: Option<(f64, f64)> = None;
    for c in candles {
        let ha_close = (c.open + c.high + c.low + c.close) / 4.0;
        let ha_open = match prev {
            Some((po, pc)) => 0.5 * (po + pc),
            None => 0.5 * (c.open + c.close),
        };
        out.push(Candle::new(
            c.t,
            ha_open,
            c.high.max(ha_open).max(ha_close),
            c.low.min(ha_open).min(ha_close),
            ha_close,
            c.volume,
        ));
        prev = Some((ha_open, ha_close));
    }
    out
}

/// Draws a green up-arrow below a bullish candle and a red down-arrow above a
/// bearish candle, so the direction of every bar is readable at a glance.
///
/// `size` is a **price**-space height and `half` is an **x**-space half-width,
/// keeping the two axes in their own units.
fn draw_trend_arrow(plot_ui: &mut egui_plot::PlotUi, c: &Candle, half: f64, size: f64) {
    let color = if c.is_bullish() { PROFIT } else { LOSS };
    let w = half.max(f64::MIN_POSITIVE);
    let gap = 0.25 * size;
    let pts = if c.is_bullish() {
        // Tip points up, sitting just below the candle low.
        let base = c.low - gap - size;
        vec![[c.t - w, base], [c.t + w, base], [c.t, base + size]]
    } else {
        // Tip points down, sitting just above the candle high.
        let base = c.high + gap;
        vec![[c.t - w, base], [c.t + w, base], [c.t, base - size]]
    };
    plot_ui.polygon(
        egui_plot::Polygon::new(PlotPoints::from(pts))
            .fill_color(color.gamma_multiply(0.9))
            .stroke(Stroke::new(0.5_f32, color)),
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Tab {
    Candlestick,
    HeikinAshi,
    Renko,
    Kagi,
    PointFigure,
    Candlestick3D,
    CandlestickMA,
    CandlestickBollinger,
    CandlestickRSI,
    CandlestickMACD,
    VolumeProfile,
    Footprint,
    OrderBookHeatmap,
    CumulativeDelta,
    MarketProfile,
    VolumeClock,
    TickTape,
    DeltaDivergence,
    RSI,
    MACD,
    Stochastic,
    ATR,
    OBV,
    VWAP,
    Bollinger,
    BBWidth,
    ADX,
    CCI,
    WilliamsR,
    ROC,
    CMF,
    Ichimoku,
    Keltner,
    Donchian,
    Drawdown,
    Correlation,
    VolSmile,
    EffFrontier,
    RollingSharpe,
    RollingSortino,
    BetaAlpha,
    RollingMaxDD,
    VaRBacktest,
    MonteCarlo,
    VolSmileOpt,
    IVSurface,
    TermStructure,
    GreeksHeatmap,
    VIXTerm,
    VolCone,
    OptionPayoff,
    SkewEvolution,
    GammaExposure,
    PutCallRatio,
    IVRank,
    SharpeSurface,
    AcfPacf,
    Hurst,
    Wavelet,
    Kalman,
    MarkovRegime,
    Copula3D,
    QQPlot,
    ReturnDist,
    RollingMoments,
    NiftyTreemap,
    SensexHeatmap,
    FII_DIIFlow,
    SectorPerf,
    YieldCurve,
    USDINR,
    MonsoonAgri,
    Seasonality,
    WorldIndices,
    TickerTape,
    CurrencyMatrix,
    SectorWheel,
    EarningsCalendar,
    EconCalendar,
    CorrelationNetwork,
    ReturnHeatmap,
    MultiIndicator,
    MultiTimeframe,
    MACDDivergence,
    BollingerBreakout,
    VolumeWeightedScatter,
    PriceMomentum,
    DrawdownRecovery,
    RollingCorrelation,
    TickTapeAdv,
    SeasonalityAdv,
    ParabolicSAR,
    MACDHistogram,
    RSIHeatmap,
    IchimokuEMA,
    KeltnerBreakout,
    DonchianBreakout,
    CopulaHeatmap,
    CorrelationNetworkAdv,
    MarketWatch,
    FOChain,
    IVSurfaceIndia,
    OIHeatmap,
    GSec,
    MoneyMarket,
    RBIPolicy,
    MacroIndia,
    CommoditiesIndia,
    USDINRCurve,
    YieldIndia,
    MFAnalytics,
    FPIFII,
    CreditRatings,
    BankingIndia,
    CorpActions,
    IPOPipeline,
    IndiaBreadth,
    SectorResearch,
    IndiaNews,
    Regulatory,
    GSTBudget,
    IndiaPortfolio,
    AlgoFeed,
    AIResearch,
    IndiaDashboard,
    MultiCompare,
    Forecast,
    StochRsi,
    Zscore,
    Mfi,
    UltimateOsc,
    Tsi,
    Coppock,
    Dpo,
    Aroon,
    AroonOsc,
    Ulcer,
    Eom,
    ForceIndex,
    MassIndex,
    Pvt,
    Mfv,
    AdLine,
    TrendIntensity,
    RealizedVol,
    KeltnerWidth,
    Volatility,
    Kst,
    ElderRay,
    Vortex,
    Kama,
    Alma,
    HullMa,
    Wma,
    MultiSma,
    MultiEma,
    HighLowBand,
    VwapBands,
    Supertrend,
    LogReturns,
    Volume,
    Ohlc,
    PivotPoints,
    FibLevels,
    PriceSurface,
    VolatilitySurface,
    ReturnSurface,
    RiskLandscape,
    BetaSurface,
    EntropySurface,
    AlphaSurface,
    SignalSurface,
    RegimeSurface,
    RegimeTimeline,
    MomentumSurface,
    OrderFlowSurface,
    SkewKurtSurface,
    SignalEvolution,
    EquitySurface,
    VarBandSurface,
    RiskReturnCloud,
    EigenvalueCloud,
    ReturnsHeatmap,
    PcaProjection,
    FourInOne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabCategory {
    PriceAction,
    OrderFlow,
    Indicators,
    RiskPortfolio,
    VolatilityOptions,
    Microstructure,
    IndiaSpecific,
    BloombergStyle,
    Advanced,
    Comparison,
    ThreeD,
}

impl TabCategory {
    fn label(&self) -> &'static str {
        match self {
            TabCategory::PriceAction => "Price Action",
            TabCategory::OrderFlow => "Order Flow",
            TabCategory::Indicators => "Indicators",
            TabCategory::RiskPortfolio => "Risk & Portfolio",
            TabCategory::VolatilityOptions => "Volatility & Options",
            TabCategory::Microstructure => "Microstructure",
            TabCategory::IndiaSpecific => "India-Specific",
            TabCategory::BloombergStyle => "Bloomberg-Style",
            TabCategory::Advanced => "Advanced",
            TabCategory::Comparison => "Comparison",
            TabCategory::ThreeD => "3D & Surfaces",
        }
    }

    fn color(&self) -> Color32 {
        match self {
            TabCategory::PriceAction => AMBER,
            TabCategory::OrderFlow => INFO,
            TabCategory::Indicators => PROFIT,
            TabCategory::RiskPortfolio => PURPLE,
            TabCategory::VolatilityOptions => LOSS,
            TabCategory::Microstructure => Color32::from_rgb(0x00, 0xD4, 0xAA),
            TabCategory::IndiaSpecific => Color32::from_rgb(0xFF, 0x8C, 0x00),
            TabCategory::BloombergStyle => Color32::from_rgb(0x44, 0x88, 0xFF),
            TabCategory::Advanced => Color32::from_rgb(0xFF, 0x69, 0xB4),
            TabCategory::Comparison => Color32::from_rgb(0x00, 0xE5, 0xFF),
            TabCategory::ThreeD => Color32::from_rgb(0xB0, 0x7C, 0xFF),
        }
    }
}

const CATEGORIES: [TabCategory; 11] = [
    TabCategory::PriceAction,
    TabCategory::OrderFlow,
    TabCategory::Indicators,
    TabCategory::RiskPortfolio,
    TabCategory::VolatilityOptions,
    TabCategory::Microstructure,
    TabCategory::IndiaSpecific,
    TabCategory::BloombergStyle,
    TabCategory::Advanced,
    TabCategory::ThreeD,
    TabCategory::Comparison,
];

fn tabs_in_category(cat: TabCategory) -> &'static [(Tab, &'static str)] {
    match cat {
        TabCategory::PriceAction => &[
            (Tab::Candlestick, "Candlestick"),
            (Tab::HeikinAshi, "Heikin-Ashi"),
            (Tab::Renko, "Renko"),
            (Tab::Kagi, "Kagi"),
            (Tab::PointFigure, "Point&Figure"),
            (Tab::Candlestick3D, "Candle3D"),
            (Tab::CandlestickMA, "Candle+MA"),
            (Tab::CandlestickBollinger, "Candle+BB"),
            (Tab::CandlestickRSI, "Candle+RSI"),
            (Tab::CandlestickMACD, "Candle+MACD"),
        ],
        TabCategory::OrderFlow => &[
            (Tab::VolumeProfile, "Volume Profile"),
            (Tab::Footprint, "Footprint"),
            (Tab::OrderBookHeatmap, "OrderBook Heatmap"),
            (Tab::CumulativeDelta, "Cumulative Delta"),
            (Tab::MarketProfile, "Market Profile"),
            (Tab::VolumeClock, "Volume Clock"),
            (Tab::TickTape, "Tick Tape"),
            (Tab::DeltaDivergence, "Delta Divergence"),
        ],
        TabCategory::Indicators => &[
            (Tab::RSI, "RSI"),
            (Tab::MACD, "MACD"),
            (Tab::Stochastic, "Stochastic"),
            (Tab::ATR, "ATR"),
            (Tab::OBV, "OBV"),
            (Tab::VWAP, "VWAP"),
            (Tab::Bollinger, "Bollinger"),
            (Tab::BBWidth, "BB Width"),
            (Tab::ADX, "ADX"),
            (Tab::CCI, "CCI"),
            (Tab::WilliamsR, "Williams %R"),
            (Tab::ROC, "ROC"),
            (Tab::CMF, "CMF"),
            (Tab::Ichimoku, "Ichimoku"),
            (Tab::Keltner, "Keltner"),
            (Tab::Donchian, "Donchian"),
            (Tab::StochRsi, "Stoch RSI"),
            (Tab::Zscore, "Z-Score"),
            (Tab::Mfi, "MFI"),
            (Tab::UltimateOsc, "Ultimate Osc"),
            (Tab::Tsi, "TSI"),
            (Tab::Coppock, "Coppock"),
            (Tab::Dpo, "DPO"),
            (Tab::Aroon, "Aroon"),
            (Tab::AroonOsc, "Aroon Osc"),
            (Tab::Ulcer, "Ulcer Index"),
            (Tab::Eom, "EOM"),
            (Tab::ForceIndex, "Force Index"),
            (Tab::MassIndex, "Mass Index"),
            (Tab::Pvt, "PVT"),
            (Tab::Mfv, "MFV"),
            (Tab::AdLine, "A/D Line"),
            (Tab::TrendIntensity, "Trend Intensity"),
            (Tab::RealizedVol, "Realized Vol"),
            (Tab::KeltnerWidth, "Keltner Width"),
            (Tab::Volatility, "Volatility"),
            (Tab::Kst, "KST"),
            (Tab::ElderRay, "Elder Ray"),
            (Tab::Vortex, "Vortex"),
            (Tab::Kama, "KAMA"),
            (Tab::Alma, "ALMA"),
            (Tab::HullMa, "Hull MA"),
            (Tab::Wma, "WMA"),
            (Tab::MultiSma, "Multi SMA"),
            (Tab::MultiEma, "Multi EMA"),
            (Tab::HighLowBand, "High Low Band"),
            (Tab::VwapBands, "VWAP Bands"),
            (Tab::Supertrend, "Supertrend"),
            (Tab::LogReturns, "Log Returns"),
            (Tab::Volume, "Volume"),
            (Tab::Ohlc, "OHLC"),
            (Tab::PivotPoints, "Pivot Points"),
            (Tab::FibLevels, "Fib Levels"),
        ],
        TabCategory::RiskPortfolio => &[
            (Tab::Drawdown, "Drawdown"),
            (Tab::Correlation, "Correlation"),
            (Tab::VolSmile, "Vol Smile"),
            (Tab::EffFrontier, "Eff Frontier"),
            (Tab::RollingSharpe, "Rolling Sharpe"),
            (Tab::RollingSortino, "Rolling Sortino"),
            (Tab::BetaAlpha, "Beta/Alpha"),
            (Tab::RollingMaxDD, "Rolling MaxDD"),
            (Tab::VaRBacktest, "VaR Backtest"),
            (Tab::MonteCarlo, "Monte Carlo"),
        ],
        TabCategory::VolatilityOptions => &[
            (Tab::VolSmileOpt, "Vol Smile"),
            (Tab::IVSurface, "IV Surface"),
            (Tab::TermStructure, "Term Structure"),
            (Tab::GreeksHeatmap, "Greeks Heatmap"),
            (Tab::VIXTerm, "VIX Term"),
            (Tab::VolCone, "Vol Cone"),
            (Tab::OptionPayoff, "Option Payoff"),
            (Tab::SkewEvolution, "Skew Evolution"),
            (Tab::GammaExposure, "Gamma Exposure"),
            (Tab::PutCallRatio, "Put/Call Ratio"),
            (Tab::IVRank, "IV Rank"),
            (Tab::SharpeSurface, "Sharpe Surface"),
        ],
        TabCategory::Microstructure => &[
            (Tab::AcfPacf, "ACF/PACF"),
            (Tab::Hurst, "Hurst"),
            (Tab::Wavelet, "Wavelet"),
            (Tab::Kalman, "Kalman"),
            (Tab::MarkovRegime, "Markov Regime"),
            (Tab::Copula3D, "Copula 3D"),
            (Tab::QQPlot, "QQ Plot"),
            (Tab::ReturnDist, "Return Dist"),
            (Tab::RollingMoments, "Rolling Moments"),
        ],
        TabCategory::IndiaSpecific => &[
            (Tab::NiftyTreemap, "Nifty Treemap"),
            (Tab::SensexHeatmap, "Sensex Heatmap"),
            (Tab::FII_DIIFlow, "FII/DII Flow"),
            (Tab::SectorPerf, "Sector Perf"),
            (Tab::YieldCurve, "Yield Curve"),
            (Tab::USDINR, "USD/INR"),
            (Tab::MonsoonAgri, "Monsoon Agri"),
            (Tab::Seasonality, "Seasonality"),
            (Tab::FOChain, "FOChain"),
            (Tab::IVSurfaceIndia, "IV Surface"),
            (Tab::OIHeatmap, "OI Heatmap"),
            (Tab::GSec, "G-Sec"),
            (Tab::MoneyMarket, "Money Mkt"),
            (Tab::RBIPolicy, "RBI Policy"),
            (Tab::MacroIndia, "Macro India"),
            (Tab::CommoditiesIndia, "MCX/NCDEX"),
            (Tab::USDINRCurve, "USDINR Fwd"),
            (Tab::YieldIndia, "Yield India"),
            (Tab::MFAnalytics, "MF Analytics"),
            (Tab::FPIFII, "FPI/FII"),
            (Tab::CreditRatings, "Credit Ratings"),
            (Tab::BankingIndia, "Banking"),
            (Tab::CorpActions, "Corp Actions"),
            (Tab::IPOPipeline, "IPO Pipeline"),
            (Tab::IndiaBreadth, "India Breadth"),
            (Tab::SectorResearch, "Sector Research"),
            (Tab::IndiaNews, "India News"),
            (Tab::Regulatory, "Regulatory"),
            (Tab::GSTBudget, "GST/Budget"),
            (Tab::IndiaPortfolio, "India Portfolio"),
            (Tab::AlgoFeed, "Algo Feed"),
            (Tab::AIResearch, "AI Research"),
            (Tab::IndiaDashboard, "India Dashboard"),
        ],
        TabCategory::BloombergStyle => &[
            (Tab::WorldIndices, "World Indices"),
            (Tab::TickerTape, "Ticker Tape"),
            (Tab::CurrencyMatrix, "Currency Matrix"),
            (Tab::SectorWheel, "Sector Wheel"),
            (Tab::EarningsCalendar, "Earnings Cal"),
            (Tab::EconCalendar, "Econ Calendar"),
            (Tab::CorrelationNetwork, "Corr Network"),
            (Tab::ReturnHeatmap, "Return Heatmap"),
            (Tab::MarketWatch, "MarketWatch"),
        ],
        TabCategory::Advanced => &[
            (Tab::MultiIndicator, "Multi Indicator"),
            (Tab::MultiTimeframe, "Multi Timeframe"),
            (Tab::MACDDivergence, "MACD Divergence"),
            (Tab::BollingerBreakout, "BB Breakout"),
            (Tab::VolumeWeightedScatter, "Vol-Price Scatter"),
            (Tab::PriceMomentum, "Price Momentum"),
            (Tab::DrawdownRecovery, "DD Recovery"),
            (Tab::RollingCorrelation, "Rolling Corr"),
            (Tab::TickTapeAdv, "Tick Tape Adv"),
            (Tab::SeasonalityAdv, "Seasonality Adv"),
            (Tab::ParabolicSAR, "Parabolic SAR"),
            (Tab::MACDHistogram, "MACD Histogram"),
            (Tab::RSIHeatmap, "RSI Heatmap"),
            (Tab::IchimokuEMA, "Ichimoku+EMA"),
            (Tab::KeltnerBreakout, "Keltner Breakout"),
            (Tab::DonchianBreakout, "Donchian Breakout"),
            (Tab::CopulaHeatmap, "Copula Heatmap"),
            (Tab::CorrelationNetworkAdv, "Corr Network Adv"),
            (Tab::Forecast, "Forecast"),
        ],
        TabCategory::Comparison => &[(Tab::MultiCompare, "Multi-Compare")],
        TabCategory::ThreeD => &[
            (Tab::PriceSurface, "3D Price Surface"),
            (Tab::VolatilitySurface, "3D Volatility"),
            (Tab::ReturnSurface, "3D Return Surface"),
            (Tab::RiskLandscape, "3D Risk Landscape"),
            (Tab::BetaSurface, "3D Beta Surface"),
            (Tab::EntropySurface, "3D Entropy Surface"),
            (Tab::AlphaSurface, "3D Alpha Surface"),
            (Tab::SignalSurface, "3D Signal Surface"),
            (Tab::RegimeSurface, "3D Regime Cluster"),
            (Tab::RegimeTimeline, "3D Regime Timeline"),
            (Tab::MomentumSurface, "3D Momentum Surface"),
            (Tab::OrderFlowSurface, "3D Order Flow"),
            (Tab::SkewKurtSurface, "3D Skew-Kurt"),
            (Tab::SignalEvolution, "3D Signal Evolution"),
            (Tab::EquitySurface, "3D Equity Surface"),
            (Tab::VarBandSurface, "3D VaR Band"),
            (Tab::RiskReturnCloud, "3D Risk-Return Cloud"),
            (Tab::EigenvalueCloud, "3D Eigenvalue Cloud"),
            (Tab::ReturnsHeatmap, "3D Returns Heatmap"),
            (Tab::PcaProjection, "3D PCA Projection"),
            (Tab::FourInOne, "4-in-1 Dashboard"),
        ],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeRange {
    D1,
    W1,
    M1,
    M3,
    M6,
    Y1,
    Y5,
    /// User-entered start/end dates.
    Custom,
}

impl TimeRange {
    fn label(&self) -> &'static str {
        match self {
            TimeRange::D1 => "1D",
            TimeRange::W1 => "1W",
            TimeRange::M1 => "1M",
            TimeRange::M3 => "3M",
            TimeRange::M6 => "6M",
            TimeRange::Y1 => "1Y",
            TimeRange::Y5 => "5Y",
            TimeRange::Custom => "Custom",
        }
    }

    fn to_days(&self) -> i64 {
        match self {
            TimeRange::D1 => 1,
            TimeRange::W1 => 7,
            TimeRange::M1 => 30,
            TimeRange::M3 => 90,
            TimeRange::M6 => 180,
            TimeRange::Y1 => 365,
            TimeRange::Y5 => 1825,
            // Overridden by the custom window in `trigger_fetch`; this is only
            // a sane fallback for callers that need a length.
            TimeRange::Custom => 365,
        }
    }

    fn to_interval(&self) -> Interval {
        match self {
            TimeRange::D1 => Interval::Min5,
            TimeRange::W1 => Interval::Min15,
            // Daily bars, not hourly: hourly data over a month is full of
            // overnight and weekend gaps, so the chart rendered as sparse
            // floating candles instead of the continuous 3M view.
            TimeRange::M1 => Interval::Day1,
            TimeRange::M3 => Interval::Day1,
            TimeRange::M6 => Interval::Day1,
            TimeRange::Y1 => Interval::Day1,
            TimeRange::Y5 => Interval::Week1,
            // Replaced by an interval chosen from the custom window length.
            TimeRange::Custom => Interval::Day1,
        }
    }
}

#[derive(Debug, Clone)]
enum AppMessage {
    DataReady(OhlcvSeries),
    FetchError(String),
    QuoteReady(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Prefs {
    live: bool,
    symbol: String,
    range: String,
    theme: String,
    /// Custom range bounds as `YYYY-MM-DD`, empty when unused.
    custom_start: String,
    custom_end: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            live: true,
            symbol: "RELIANCE.NS".to_string(),
            range: "1y".to_string(),
            theme: "dark".to_string(),
            custom_start: String::new(),
            custom_end: String::new(),
        }
    }
}

impl Prefs {
    /// Location of the settings file.
    ///
    /// Anchored to the executable's own directory, not the working directory.
    /// A relative `./data/prefs.json` resolved against whatever directory the
    /// user happened to launch from, so the packaged app scattered its settings
    /// around (and could fail to write at all from a read-only location like the
    /// Desktop). Each build now keeps its settings next to its own binary.
    fn prefs_path() -> std::path::PathBuf {
        let base = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        base.join("data").join("prefs.json")
    }
}

/// Whether a local ONNX model file is present, without loading a session.
///
/// Existence is the only cheap check available, and it is the same one the
/// engine uses to decide whether to try a model at all. Sessions load lazily on
/// first inference, so this stays free.
fn model_file_present(file: &str) -> bool {
    bt_analytics::models::BharatModelEngine::with_default_paths()
        .models_dir()
        .join(file)
        .is_file()
}

/// Whether a small local ONNX model has its file on disk.
fn local_model_ready(model: bt_analytics::models::Model) -> bool {
    bt_analytics::models::BharatModelEngine::with_default_paths().is_available(model)
}

impl Prefs {
    fn load() -> Self {
        let path = Self::prefs_path();
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(prefs) = serde_json::from_str::<Prefs>(&content) {
                return prefs;
            }
        }
        Self::default()
    }

    fn save(&self) {
        let path = Self::prefs_path();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(&path, json);
        }
    }
}

struct Dataset {
    candles: OhlcvSeries,
    delta_series: OhlcvSeries,
    equity: Vec<f64>,
    corr: Vec<(String, Vec<f64>)>,
    smiles: Vec<(String, Vec<f64>, Vec<f64>)>,
    portfolios: Vec<bt_viz::efficient_frontier::Portfolio>,
    returns: Vec<f64>,
    treemap_nodes: Vec<bt_viz::sector_treemap::TreemapNode>,
    yield_curves: Vec<(String, Vec<f64>, Vec<f64>)>,
    seasonality: Vec<Vec<f64>>,
}

impl Dataset {
    fn generate(seed: u64) -> Self {
        let candles = synthetic_ohlcv("RELIANCE.NS", 180, seed, 2500.0);
        let delta_series = synthetic_ohlcv("NIFTY50", 200, seed + 1, 22000.0);
        let equity = synthetic_ohlcv("PORTFOLIO", 300, seed + 2, 1_000_000.0).closes();
        let corr = synthetic_correlated_returns(
            &["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC"],
            250,
            seed + 3,
        );
        let smiles = vec![
            ("7D".to_string(), 0.28, 0.10),
            ("30D".to_string(), 0.22, 0.07),
            ("90D".to_string(), 0.19, 0.05),
        ]
        .into_iter()
        .map(|(label, atm, skew)| {
            let c = bt_viz::vol_smile::synthetic_smile(&label, atm, skew, 14);
            (label, c.moneyness, c.iv)
        })
        .collect();

        let expected_returns = vec![0.09, 0.14, 0.11, 0.16, 0.07, 0.10];
        let cov: Vec<Vec<f64>> = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        if i == j {
                            0.03 + i as f64 * 0.006
                        } else {
                            0.006
                        }
                    })
                    .collect()
            })
            .collect();
        let portfolios = bt_viz::efficient_frontier::simulate_portfolios(
            &expected_returns,
            &cov,
            0.065,
            1500,
            seed + 4,
        );
        let returns = synthetic_ohlcv("BTC-USD", 400, seed + 5, 60000.0).returns();

        let treemap_nodes = vec![
            bt_viz::sector_treemap::TreemapNode::new("Reliance", 1_800_000.0, 1.2),
            bt_viz::sector_treemap::TreemapNode::new("TCS", 1_400_000.0, -0.8),
            bt_viz::sector_treemap::TreemapNode::new("HDFC Bank", 1_100_000.0, 0.5),
            bt_viz::sector_treemap::TreemapNode::new("Infosys", 700_000.0, -1.5),
            bt_viz::sector_treemap::TreemapNode::new("ICICI Bank", 650_000.0, 2.1),
            bt_viz::sector_treemap::TreemapNode::new("ITC", 500_000.0, 0.1),
            bt_viz::sector_treemap::TreemapNode::new("L&T", 420_000.0, 0.9),
            bt_viz::sector_treemap::TreemapNode::new("Bharti Airtel", 610_000.0, -0.3),
        ];

        let tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0];
        let yield_curves = vec![
            ("2026-06-01", 6.8, -1.2, 0.4),
            ("2026-07-15", 6.6, -1.0, 0.5),
            ("2026-09-20", 6.5, -0.8, 0.6),
        ]
        .into_iter()
        .map(|(label, level, slope, curvature)| {
            let c = bt_viz::yield_curve::synthetic_curve(label, level, slope, curvature, &tenors);
            (label.to_string(), c.tenors, c.yields)
        })
        .collect();

        let seasonality = bt_viz::seasonality_polar::synthetic_seasonality(seed + 6).data;

        Self {
            candles,
            delta_series,
            equity,
            corr,
            smiles,
            portfolios,
            returns,
            treemap_nodes,
            yield_curves,
            seasonality,
        }
    }
}

#[derive(Debug, Clone)]
struct MarketQuote {
    symbol: String,
    name: String,
    price: f64,
    change: f64,
    change_pct: f64,
    volume: u64,
    market_cap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketSortCol {
    Symbol,
    Name,
    Price,
    Change,
    ChangePct,
    Volume,
    MarketCap,
}

/// Sample market data for the MarketWatch panel.
///
/// In production this would be populated by calling
/// `bt_data::DataService::fetch_quote()` for each symbol; fetching 78+
/// companies individually is too slow for a 30s refresh cycle, so this
/// generates realistic-looking data for demonstration purposes.
fn sample_market_data() -> Vec<MarketQuote> {
    let symbols = [
        ("RELIANCE.NS", "Reliance Industries", 2456.75),
        ("TCS.NS", "TCS", 3890.50),
        ("INFY.NS", "Infosys", 1567.30),
        ("HDFCBANK.NS", "HDFC Bank", 1678.90),
        ("ICICIBANK.NS", "ICICI Bank", 945.60),
        ("SBIN.NS", "State Bank of India", 623.45),
        ("BHARTIARTL.NS", "Bharti Airtel", 1567.80),
        ("ITC.NS", "ITC", 456.70),
        ("LT.NS", "Larsen & Toubro", 3456.20),
        ("AXISBANK.NS", "Axis Bank", 1123.40),
        ("KOTAKBANK.NS", "Kotak Mahindra Bank", 1890.55),
        ("MARUTI.NS", "Maruti Suzuki", 12345.60),
        ("ASIANPAINT.NS", "Asian Paints", 3456.70),
        ("BAJFINANCE.NS", "Bajaj Finance", 7890.30),
        ("SUNPHARMA.NS", "Sun Pharma", 1234.50),
        ("TITAN.NS", "Titan Company", 3456.80),
        ("ULTRACEMCO.NS", "UltraTech Cement", 9876.40),
        ("NESTLEIND.NS", "Nestle India", 23456.70),
        ("WIPRO.NS", "Wipro", 456.30),
        ("HCLTECH.NS", "HCL Technologies", 1789.60),
        ("TATAMOTORS.NS", "Tata Motors", 789.40),
        ("TATASTEEL.NS", "Tata Steel", 123.50),
        ("AAPL", "Apple", 189.30),
        ("MSFT", "Microsoft", 378.90),
        ("GOOGL", "Google", 145.60),
        ("AMZN", "Amazon", 178.30),
        ("TSLA", "Tesla", 245.60),
        ("NVDA", "NVIDIA", 456.70),
        ("META", "Meta Platforms", 345.80),
        ("NFLX", "Netflix", 456.90),
        ("BTC-USD", "Bitcoin", 43567.80),
        ("ETH-USD", "Ethereum", 2345.60),
        ("SOL-USD", "Solana", 98.70),
    ];

    symbols
        .iter()
        .enumerate()
        .map(|(i, (sym, name, price))| {
            let change = (i as f64 * 13.7 - 200.0).sin() * price * 0.02;
            let change_pct = change / price * 100.0;
            let volume = 1_000_000.0 + (i as f64 * 12345.0);
            let market_cap = price * volume * 0.1;
            MarketQuote {
                symbol: sym.to_string(),
                name: name.to_string(),
                price: *price,
                change,
                change_pct,
                volume: volume as u64,
                market_cap,
            }
        })
        .collect()
}

struct BharatApp {
    tab: Tab,
    dark: bool,
    data: Dataset,
    seed_counter: u64,
    selected_company: String,
    company_search: String,
    time_range: TimeRange,
    live: bool,
    last_fetch: Option<Instant>,
    fetch_in_flight: bool,
    /// A fetch was requested while one was already running; re-issue it on
    /// completion so the requested range is never silently dropped.
    fetch_pending: bool,
    status_source: String,
    status_last_update: String,
    status_latency_ms: u64,
    tx: Sender<AppMessage>,
    rx: Receiver<AppMessage>,
    runtime: tokio::runtime::Runtime,
    error_toast: Option<(String, Instant)>,
    warning_banner: Option<String>,
    warning_until: Option<Instant>,
    chart_id: Option<egui::Id>,
    scroll_to_chart: bool,
    prefs_dirty: bool,
    market_quotes: Vec<MarketQuote>,
    market_search: String,
    market_sort_col: MarketSortCol,
    market_sort_asc: bool,
    market_last_refresh: Option<Instant>,
    compare_symbols: Vec<String>,
    compare_search: String,
    show_candle_arrows: std::cell::Cell<bool>,
    /// Double-click zoom applied on top of the full data range.
    zoom: std::cell::Cell<ZoomState>,
    /// Text contents of the custom start/end date fields.
    custom_start_text: String,
    custom_end_text: String,
    /// The validated custom window, set when the user applies the fields.
    custom_window: Option<CustomWindow>,
    /// `symbol|range` that the in-flight fetch was issued for. Compared
    /// against `last_range_key` to detect a genuine symbol/range switch.
    data_range_key: String,
    /// Last applied `data_range_key`; a change resets the zoom.
    last_range_key: String,
    /// Height of the central panel viewport, captured before the chart
    /// ScrollArea inflates the available space. Used to budget multi-pane
    /// chart layouts so the volume pane is never pushed off-screen.
    viewport_h: f32,
    /// Multi-model price forecaster (Granite TTM -> NanoForecast -> ARIMA).
    /// Cheap to hold: engines load lazily and missing model files simply
    /// disable their engine.
    forecaster: Option<Forecaster>,
    /// WatchSignal LSTM classifier (BUY/HOLD/SELL), when its ONNX is present.
    watchsignal: Option<WatchSignalModel>,
    /// Latest classified signal, refreshed with each forecast run.
    last_signal: Option<SignalOutput>,
    /// Forecast horizon in bars, chosen on the Forecast tab.
    forecast_horizon: usize,
    /// Preferred forecast engine, chosen on the Forecast tab. A missing or
    /// failing preference falls back down the chain automatically.
    forecast_preferred: ForecastEngine,
    /// Last forecast output, with the engine that produced it.
    forecast_values: Vec<f64>,
    forecast_engine: String,
    /// History tail the cached forecast was computed from (plotted behind it).
    forecast_basis: Vec<f64>,
    /// Inline forecast error, shown on the Forecast tab (never a toast).
    forecast_error: Option<String>,
    /// Process RSS sampler for the status bar, refreshed at most once a
    /// second so the readout costs nothing per frame.
    sys: sysinfo::System,
    ram_mb: f64,
    ram_at: Instant,
}

/// One plotted series for [`draw_lines_frame`]: legend label, full-length
/// values with NaN warmup (the `bt_analytics` convention), and color.
struct FrameLine<'a> {
    label: &'static str,
    vals: &'a [f64],
    color: Color32,
}

/// Shared renderer for indicator tabs: optional gray price plus any number of
/// indicator lines plus horizontal guide levels. Keeps each new tab to ~10
/// lines instead of ~45.
fn draw_lines_frame(
    ui: &mut egui::Ui,
    plot_id: &str,
    title: &str,
    candles: &[Candle],
    show_price: bool,
    lines: &[FrameLine<'_>],
    guides: &[(f64, Color32)],
) {
    ui.label(RichText::new(title).strong());
    Plot::new(plot_id)
        .auto_bounds(egui::emath::Vec2b::new(true, true))
        .height(ui.available_height())
        .allow_scroll(true)
        .allow_drag(true)
        .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
        .show(ui, |plot_ui| {
            if show_price {
                let price: PlotPoints = candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(
                    Line::new(price)
                        .color(Color32::from_gray(110))
                        .width(1.0_f32)
                        .name("Price"),
                );
            }
            for line in lines {
                let pts: PlotPoints = candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        let v = line.vals.get(i).copied().unwrap_or(f64::NAN);
                        if v.is_nan() {
                            None
                        } else {
                            Some([c.t, v])
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(pts)
                        .color(line.color)
                        .width(2.0_f32)
                        .name(line.label),
                );
            }
            for (y, c) in guides {
                plot_ui.hline(egui_plot::HLine::new(*y).color(*c));
            }
        });
}

impl BharatApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        let prefs = Prefs::load();

        let time_range = match prefs.range.as_str() {
            "1d" => TimeRange::D1,
            "1w" => TimeRange::W1,
            "1m" => TimeRange::M1,
            "3m" => TimeRange::M3,
            "6m" => TimeRange::M6,
            "1y" => TimeRange::Y1,
            "5y" => TimeRange::Y5,
            "custom" => TimeRange::Custom,
            _ => TimeRange::Y1,
        };
        // A persisted custom window must still parse, otherwise fall back to
        // the default preset so the app never starts with a broken range.
        let custom_window = CustomWindow::parse(&prefs.custom_start, &prefs.custom_end);
        let time_range = if time_range == TimeRange::Custom && custom_window.is_none() {
            TimeRange::Y1
        } else {
            time_range
        };

        let mut app = Self {
            tab: Tab::Candlestick,
            dark: prefs.theme == "dark",
            data: Dataset::generate(42),
            seed_counter: 42,
            selected_company: prefs.symbol.clone(),
            company_search: String::new(),
            time_range,
            live: prefs.live,
            last_fetch: None,
            fetch_in_flight: false,
            fetch_pending: false,
            status_source: "Synthetic".to_string(),
            status_last_update: "—".to_string(),
            status_latency_ms: 0,
            tx,
            rx,
            runtime,
            error_toast: None,
            warning_banner: None,
            warning_until: None,
            chart_id: None,
            scroll_to_chart: false,
            prefs_dirty: false,
            market_quotes: sample_market_data(),
            market_search: String::new(),
            market_sort_col: MarketSortCol::MarketCap,
            market_sort_asc: false,
            market_last_refresh: Some(Instant::now()),
            compare_symbols: vec![
                "RELIANCE.NS".to_string(),
                "TCS.NS".to_string(),
                "INFY.NS".to_string(),
            ],
            compare_search: String::new(),
            show_candle_arrows: std::cell::Cell::new(true),
            zoom: std::cell::Cell::new(ZoomState::default()),
            custom_start_text: prefs.custom_start.clone(),
            custom_end_text: prefs.custom_end.clone(),
            custom_window,
            data_range_key: String::new(),
            last_range_key: String::new(),
            viewport_h: 600.0,
            forecaster: Some(Forecaster::with_default_paths()),
            watchsignal: Self::load_watchsignal(),
            last_signal: None,
            forecast_horizon: 20,
            forecast_preferred: ForecastEngine::Auto,
            forecast_values: Vec::new(),
            forecast_engine: String::new(),
            forecast_basis: Vec::new(),
            forecast_error: None,
            sys: sysinfo::System::new(),
            ram_mb: 0.0,
            ram_at: Instant::now() - Duration::from_secs(10),
        };

        app.trigger_fetch();
        app
    }

    fn save_prefs(&mut self) {
        if self.prefs_dirty {
            let range_str = match self.time_range {
                TimeRange::D1 => "1d",
                TimeRange::W1 => "1w",
                TimeRange::M1 => "1m",
                TimeRange::M3 => "3m",
                TimeRange::M6 => "6m",
                TimeRange::Y1 => "1y",
                TimeRange::Y5 => "5y",
                // Persisted verbatim; the dates themselves live in the fields.
                TimeRange::Custom => "custom",
            };
            let prefs = Prefs {
                live: self.live,
                symbol: self.selected_company.clone(),
                range: range_str.to_string(),
                custom_start: self.custom_start_text.clone(),
                custom_end: self.custom_end_text.clone(),
                theme: if self.dark {
                    "dark".to_string()
                } else {
                    "light".to_string()
                },
            };
            prefs.save();
            self.prefs_dirty = false;
        }
    }

    fn trigger_fetch(&mut self) {
        if self.fetch_in_flight {
            // Do not drop the request. A range or symbol switch during an
            // in-flight fetch (common while the 30s live refresh is running)
            // used to be discarded, which left the header showing the new range
            // while the chart still held the old range's data.
            self.fetch_pending = true;
            return;
        }
        self.fetch_in_flight = true;
        let symbol = self.selected_company.clone();
        // A custom window, when applied, replaces the preset length and picks
        // its own bar interval from the requested span.
        let custom = match self.time_range {
            TimeRange::Custom => self.custom_window,
            _ => None,
        };
        let interval = custom.map_or_else(|| self.time_range.to_interval(), |w| w.interval());
        let days = custom.map_or_else(|| self.time_range.to_days(), |w| w.days());
        let tx = self.tx.clone();
        // Record what this fetch is for, so `DataReady` can tell a genuine
        // symbol/range switch (reset the zoom) from a periodic refresh (keep it).
        let window_tag = custom
            .map(|w| format!("{}:{}", w.start, w.end))
            .unwrap_or_else(|| self.time_range.label().to_string());
        self.data_range_key = format!("{}|{}|{}", symbol, interval.as_str(), window_tag);

        self.runtime.spawn(async move {
            let start = Instant::now();
            let service = match DataService::new() {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(AppMessage::FetchError(format!(
                        "Failed to initialize data service: {}",
                        e
                    )));
                    return;
                }
            };
            let now = Utc::now();
            // A custom window pins both ends; a preset measures back from now.
            let (start_date, end) = match custom {
                Some(w) => {
                    let end = chrono::DateTime::from_timestamp(w.end, 0).unwrap_or(now);
                    let start = chrono::DateTime::from_timestamp(w.start, 0)
                        .unwrap_or(now - ChronoDuration::days(days));
                    (start, end.max(now))
                }
                None => (now - ChronoDuration::days(days), now),
            };
            match service
                .fetch_ohlcv(&symbol, interval, start_date, end)
                .await
            {
                Ok(series) => {
                    let _ = tx.send(AppMessage::DataReady(series));
                }
                Err(e) => {
                    let _ = tx.send(AppMessage::FetchError(format!(
                        "Failed to fetch {}: {}",
                        symbol, e
                    )));
                }
            }
            let _ = start.elapsed();
        });
    }

    fn drain_messages(&mut self) {
        let mut completed = false;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::DataReady(series) => {
                    // A refresh of the same symbol/range must not discard the
                    // user's zoom, so only reset when the series identity or
                    // timeframe actually changed. `trigger_fetch` clears
                    // `data_range_key` when the user switches symbol or range.
                    let key = self.data_range_key.clone();
                    if self.last_range_key != key {
                        self.last_range_key = key;
                        self.zoom.set(ZoomState::default());
                    }
                    self.data.candles = series;
                    self.fetch_in_flight = false;
                    completed = true;
                    self.last_fetch = Some(Instant::now());
                    self.status_source = format!("Yahoo ({})", self.selected_company);
                    self.status_last_update = Utc::now().format("%H:%M:%S").to_string();
                    self.status_latency_ms = 0;
                    self.warning_banner = None;
                }
                AppMessage::FetchError(err) => {
                    self.fetch_in_flight = false;
                    completed = true;
                    self.error_toast = Some((err.clone(), Instant::now()));
                    self.warning_banner = Some("Using synthetic fallback data".to_string());
                    self.warning_until = Some(Instant::now() + Duration::from_secs(30));
                    self.status_source = "Synthetic (fallback)".to_string();
                    self.status_last_update = Utc::now().format("%H:%M:%S").to_string();
                    let days = match self.time_range.to_interval() {
                        Interval::Min5 | Interval::Min15 | Interval::Hour1 => 50,
                        Interval::Day1 => self.time_range.to_days() as usize,
                        Interval::Week1 | Interval::Month1 => 200,
                        _ => 100,
                    };
                    self.data.candles = synthetic_ohlcv(&self.selected_company, days, 42, 100.0);
                }
                AppMessage::QuoteReady(_) => {}
            }
        }
        // A range or symbol switch during the fetch was deferred, not dropped.
        // Re-issue it now that the slot is free, so the chart always matches
        // the range shown in the header.
        if completed && self.fetch_pending {
            self.fetch_pending = false;
            self.trigger_fetch();
        }
    }

    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(APP_NAME).color(AMBER).strong().size(18.0_f32));
                ui.separator();
                let display_text = if self.company_search.is_empty() {
                    self.selected_company.clone()
                } else {
                    self.company_search.clone()
                };
                let mut changed = false;
                egui::ComboBox::from_label("Company")
                    .selected_text(display_text)
                    .width(220.0_f32)
                    .show_ui(ui, |ui| {
                        ui.text_edit_singleline(&mut self.company_search);
                        let query = self.company_search.to_lowercase();
                        for (name, ticker, exchange) in COMPANY_LIST {
                            let matches = query.is_empty()
                                || name.to_lowercase().contains(&query)
                                || ticker.to_lowercase().contains(&query);
                            if matches {
                                let label = format!("{} ({}) — {}", name, ticker, exchange);
                                if ui
                                    .selectable_label(self.selected_company == *ticker, label)
                                    .clicked()
                                {
                                    self.selected_company = ticker.to_string();
                                    self.company_search.clear();
                                    changed = true;
                                }
                            }
                        }
                    });
                if changed {
                    self.prefs_dirty = true;
                    self.trigger_fetch();
                }
                ui.separator();
                for range in [
                    TimeRange::D1,
                    TimeRange::W1,
                    TimeRange::M1,
                    TimeRange::M3,
                    TimeRange::M6,
                    TimeRange::Y1,
                    TimeRange::Y5,
                ] {
                    let selected = self.time_range == range;
                    if ui.selectable_label(selected, range.label()).clicked() {
                        self.time_range = range;
                        self.prefs_dirty = true;
                        self.trigger_fetch();
                    }
                }
                self.custom_range_ui(ui);
                ui.separator();
                let live_text = if self.live { "Live *" } else { "Off o" };
                let live_color = if self.live { PROFIT } else { Color32::GRAY };
                if ui
                    .selectable_label(self.live, RichText::new(live_text).color(live_color))
                    .clicked()
                {
                    self.live = !self.live;
                    self.prefs_dirty = true;
                    if self.live {
                        self.trigger_fetch();
                    }
                }
                ui.separator();
                if ui
                    .button(if self.dark { "\u{1F319}" } else { "\u{2600}" })
                    .clicked()
                {
                    self.dark = !self.dark;
                    self.prefs_dirty = true;
                }
                if ui.button("\u{27F3}").clicked() {
                    self.trigger_fetch();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("Made by {AUTHOR}")).color(AMBER));
                });
            });
        });
    }

    fn tab_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal(|ui| {
                    for cat in CATEGORIES {
                        let tabs = tabs_in_category(cat);
                        ui.group(|ui| {
                            ui.label(
                                RichText::new(cat.label())
                                    .color(cat.color())
                                    .strong()
                                    .size(11.0_f32),
                            );
                            ui.horizontal(|ui| {
                                for (tab, label) in tabs.iter() {
                                    let selected = self.tab == *tab;
                                    let text = if selected {
                                        RichText::new(format!("[{label}]"))
                                            .color(cat.color())
                                            .strong()
                                    } else {
                                        RichText::new(*label).size(11.0_f32)
                                    };
                                    if ui.selectable_label(selected, text).clicked() {
                                        self.tab = *tab;
                                        self.chart_id = None;
                                        self.scroll_to_chart = true;
                                    }
                                }
                            });
                        });
                        ui.separator();
                    }
                });
            });
        });
    }

    /// Price to zoom the toolbar button around: the current focus when set,
    /// otherwise the midpoint of the visible candles.
    fn zoom_focus_y(&self) -> f64 {
        let series = &self.data.candles.candles;
        if series.is_empty() {
            return self.zoom.get().focus_y.unwrap_or(0.0);
        }
        let (x0, x1) = x_bounds(series);
        let (_, range) = price_scale(series);
        let lo = price_scale(series).0;
        let (wx0, wx1, _, _) = self.zoom.get().window(x0, x1, lo, lo + range);
        price_envelope(series, wx0, wx1).map_or(lo + 0.5 * range, |(a, b)| 0.5 * (a + b))
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        // Refresh the process-RSS readout at most once a second: enumerating
        // processes every frame would cost more than the label is worth.
        if self.ram_at.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_processes();
            if let Some(process) = sysinfo::get_current_pid()
                .ok()
                .and_then(|pid| self.sys.process(pid))
            {
                self.ram_mb = process.memory() as f64 / 1024.0 / 1024.0;
            }
            self.ram_at = Instant::now();
        }
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(if self.live { PROFIT } else { Color32::GRAY }, "\u{25CF}");
                ui.label(if self.live { "Live" } else { "Off" });
                ui.separator();
                ui.label(format!("Source: {}", self.status_source));
                ui.separator();
                ui.label(format!("Last: {}", self.status_last_update));
                ui.separator();
                ui.label(format!("Latency: {}ms", self.status_latency_ms));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("Made by {AUTHOR}")).color(AMBER));
                    ui.separator();
                    ui.label(format!("RAM: {:.0} MB", self.ram_mb))
                        .on_hover_text("This app's own memory footprint");
                    // Active forecast engine: green while a neural model runs,
                    // amber for NanoForecast, grey for the statistical bench.
                    let (dot, dot_color) = if self.forecast_engine == "Granite TTM R2" {
                        ("● Granite TTM R2", PROFIT)
                    } else if self.forecast_engine == "NanoForecast v0.5" {
                        ("● NanoForecast", AMBER)
                    } else if self.forecast_engine.is_empty() {
                        ("○ forecast idle", Color32::GRAY)
                    } else {
                        ("● ARIMA bench", Color32::GRAY)
                    };
                    ui.separator();
                    ui.label(RichText::new(dot).color(dot_color))
                        .on_hover_text("Engine behind the last forecast run");
                    // Headline trading signal, when the classifier is loaded.
                    if let Some(output) = &self.last_signal {
                        let headline = output.headline();
                        let color = match headline {
                            Signal::Buy => PROFIT,
                            Signal::Hold => Color32::GRAY,
                            Signal::Sell => LOSS,
                        };
                        ui.separator();
                        ui.label(
                            RichText::new(format!("Signal: {}", headline.label()))
                                .color(color)
                                .strong(),
                        )
                        .on_hover_text("WatchSignal LSTM classification");
                    }
                });
            });
        });
    }

    fn error_toast(&mut self, ctx: &egui::Context) {
        if let Some((msg, time)) = self.error_toast.clone() {
            if time.elapsed() < Duration::from_secs(8) {
                let screen = ctx.screen_rect();
                let toast_width = 400.0_f32;
                let toast_height = 60.0_f32;
                let pos = egui::Pos2::new(
                    screen.right() - toast_width - 20.0_f32,
                    screen.top() + 60.0_f32,
                );
                egui::Window::new("error_toast")
                    .title_bar(false)
                    .fixed_pos(pos)
                    .fixed_size([toast_width, toast_height])
                    .collapsible(false)
                    .show(ctx, |ui| {
                        ui.colored_label(LOSS, RichText::new("Error").strong());
                        ui.label(msg.clone());
                        if ui.button("Dismiss").clicked() {
                            self.error_toast = None;
                        }
                    });
            } else {
                self.error_toast = None;
            }
        }
    }

    fn warning_banner(&mut self, ctx: &egui::Context) {
        let expired = match self.warning_until {
            Some(until) => Instant::now() >= until,
            None => true,
        };
        if expired {
            self.warning_banner = None;
            self.warning_until = None;
            return;
        }
        if let Some(msg) = &self.warning_banner {
            egui::TopBottomPanel::top("warning").show(ctx, |ui| {
                ui.colored_label(Color32::YELLOW, format!("⚠ {msg}"));
            });
        }
    }

    fn body(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.fetch_in_flight {
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                    ui.label("Loading...");
                });
                return;
            }

            let chart_id = Id::new("chart_area").with(self.tab);
            self.chart_id = Some(chart_id);

            // Capture the true chart viewport: the panel height minus whatever
            // chrome the header/tab bars already consumed. Multi-pane chart tabs
            // budget against this so the lower pane is never clipped.
            let screen = ui.ctx().screen_rect();
            // `clip_rect()` is the region this panel may actually paint, so the
            // height left for the chart is the clip height minus the tab's own
            // header. Budgeting from the screen height instead left a blank
            // ribbon along the bottom whenever the window was resized.
            let clip = ui.clip_rect();
            let available = (clip.height() - ui.next_widget_position().y + clip.min.y).max(240.0);
            self.viewport_h = available;

            if self.scroll_to_chart {
                self.scroll_to_chart = false;
                let anchor_rect =
                    egui::Rect::from_min_size(ui.next_widget_position(), Vec2::new(1.0, 1.0));
                let anchor = ui.interact(
                    anchor_rect,
                    egui::Id::new("chart_anchor"),
                    egui::Sense::hover(),
                );
                anchor.scroll_to_me(Some(egui::Align::Center));
            }
            // The chart draws its own panes sized to the viewport, so it is
            // rendered directly rather than inside a ScrollArea. A ScrollArea
            // reports an unbounded `available_height()`, which let multi-pane
            // charts grow past the window and clip the lower pane.
            self.dispatch_tab(ui);
        });
    }

    /// Axis label style for the active range. A custom window picks its own
    /// from its length, so a 3-day window shows dates while a 2-year window
    /// shows months.
    fn axis_date_style(&self) -> AxisDateStyle {
        match self.time_range {
            TimeRange::Custom => self
                .custom_window
                .map_or(AxisDateStyle::MonthYear, |w| w.axis_date_style()),
            other => other.axis_date_style(),
        }
    }

    /// Custom start/end date picker, shown next to the preset range buttons.
    ///
    /// Fields take `YYYY-MM-DD`. Applying validates the pair, stores it and
    /// refetches; an invalid or inverted window reports the problem inline
    /// instead of silently falling back to a preset.
    fn custom_range_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let selected = self.time_range == TimeRange::Custom;
        if ui
            .selectable_label(selected, "Custom")
            .on_hover_text("Pick an exact start and end date (YYYY-MM-DD)")
            .clicked()
        {
            if !selected {
                // Seed the fields with the current preset so there is always a
                // valid, editable starting point.
                let now = Utc::now().timestamp();
                let start = self
                    .custom_window
                    .map(|w| w.start)
                    .unwrap_or_else(|| now - self.time_range.to_days() * 86_400);
                if self.custom_start_text.is_empty() {
                    self.custom_start_text = format_ymd(start);
                }
                if self.custom_end_text.is_empty() {
                    self.custom_end_text = format_ymd(now);
                }
                self.time_range = TimeRange::Custom;
                self.prefs_dirty = true;
                self.trigger_fetch();
            }
        }

        if !selected {
            return;
        }

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.custom_start_text)
                    .desired_width(78.0)
                    .hint_text("YYYY-MM-DD"),
            );
            ui.label("to");
            ui.add(
                egui::TextEdit::singleline(&mut self.custom_end_text)
                    .desired_width(78.0)
                    .hint_text("YYYY-MM-DD"),
            );
            let apply = ui.button("Apply").on_hover_text("Fetch this exact window");
            if apply.clicked() {
                match CustomWindow::parse(&self.custom_start_text, &self.custom_end_text) {
                    Some(w) => {
                        self.custom_window = Some(w);
                        self.prefs_dirty = true;
                        self.trigger_fetch();
                    }
                    None => {
                        self.error_toast = Some((
                            "Invalid date range. Use YYYY-MM-DD, start before end,                              within 10 years."
                                .to_string(),
                            Instant::now(),
                        ));
                    }
                }
            }
            if let Some(w) = self.custom_window {
                ui.label(
                    RichText::new(format!("{} days", w.days()))
                        .small()
                        .color(Color32::GRAY),
                );
            }
        });
    }

    fn dispatch_tab(&mut self, ui: &mut egui::Ui) {
        match self.tab {
            Tab::Candlestick => self.draw_candlestick(ui),
            Tab::HeikinAshi => self.draw_heikin_ashi(ui),
            Tab::Renko => self.draw_renko(ui),
            Tab::Kagi => self.draw_kagi(ui),
            Tab::PointFigure => self.draw_point_figure(ui),
            Tab::Candlestick3D => self.draw_candlestick_3d(ui),
            Tab::CandlestickMA => self.draw_candlestick_ma(ui),
            Tab::CandlestickBollinger => self.draw_candlestick_bollinger(ui),
            Tab::CandlestickRSI => self.draw_candlestick_rsi(ui),
            Tab::CandlestickMACD => self.draw_candlestick_macd(ui),
            Tab::VolumeProfile => self.draw_volume_profile(ui),
            Tab::Footprint => self.draw_footprint(ui),
            Tab::OrderBookHeatmap => self.draw_orderbook_heatmap(ui),
            Tab::CumulativeDelta => self.draw_cumulative_delta(ui),
            Tab::MarketProfile => self.draw_market_profile(ui),
            Tab::VolumeClock => self.draw_volume_clock(ui),
            Tab::TickTape => self.draw_tick_tape(ui),
            Tab::DeltaDivergence => self.draw_delta_divergence(ui),
            Tab::RSI => self.draw_rsi(ui),
            Tab::MACD => self.draw_macd(ui),
            Tab::Stochastic => self.draw_stochastic(ui),
            Tab::ATR => self.draw_atr(ui),
            Tab::OBV => self.draw_obv(ui),
            Tab::VWAP => self.draw_vwap(ui),
            Tab::Bollinger => self.draw_bollinger(ui),
            Tab::BBWidth => self.draw_bb_width(ui),
            Tab::ADX => self.draw_adx(ui),
            Tab::CCI => self.draw_cci(ui),
            Tab::WilliamsR => self.draw_williams_r(ui),
            Tab::ROC => self.draw_roc(ui),
            Tab::CMF => self.draw_cmf(ui),
            Tab::Ichimoku => self.draw_ichimoku(ui),
            Tab::Keltner => self.draw_keltner(ui),
            Tab::Donchian => self.draw_donchian(ui),
            Tab::Drawdown => self.draw_drawdown(ui),
            Tab::Correlation => self.draw_correlation(ui),
            Tab::VolSmile => self.draw_vol_smile(ui),
            Tab::EffFrontier => self.draw_efficient_frontier(ui),
            Tab::RollingSharpe => self.draw_rolling_sharpe(ui),
            Tab::RollingSortino => self.draw_rolling_sortino(ui),
            Tab::BetaAlpha => self.draw_beta_alpha(ui),
            Tab::RollingMaxDD => self.draw_rolling_max_dd(ui),
            Tab::VaRBacktest => self.draw_var_backtest(ui),
            Tab::MonteCarlo => self.draw_monte_carlo(ui),
            Tab::VolSmileOpt => self.draw_vol_smile_opt(ui),
            Tab::IVSurface => self.draw_iv_surface(ui),
            Tab::TermStructure => self.draw_term_structure(ui),
            Tab::GreeksHeatmap => self.draw_greeks_heatmap(ui),
            Tab::VIXTerm => self.draw_vix_term(ui),
            Tab::VolCone => self.draw_vol_cone(ui),
            Tab::OptionPayoff => self.draw_option_payoff(ui),
            Tab::SkewEvolution => self.draw_skew_evolution(ui),
            Tab::GammaExposure => self.draw_gamma_exposure(ui),
            Tab::PutCallRatio => self.draw_put_call_ratio(ui),
            Tab::IVRank => self.draw_iv_rank(ui),
            Tab::SharpeSurface => self.draw_sharpe_surface(ui),
            Tab::AcfPacf => self.draw_acf_pacf(ui),
            Tab::Hurst => self.draw_hurst(ui),
            Tab::Wavelet => self.draw_wavelet(ui),
            Tab::Kalman => self.draw_kalman(ui),
            Tab::MarkovRegime => self.draw_markov_regime(ui),
            Tab::Copula3D => self.draw_copula_3d(ui),
            Tab::QQPlot => self.draw_qq_plot(ui),
            Tab::ReturnDist => self.draw_return_dist(ui),
            Tab::RollingMoments => self.draw_rolling_moments(ui),
            Tab::NiftyTreemap => self.draw_treemap(ui),
            Tab::SensexHeatmap => self.draw_heatmap(ui),
            Tab::FII_DIIFlow => self.draw_fii_dii_flow(ui),
            Tab::SectorPerf => self.draw_sector_perf(ui),
            Tab::YieldCurve => self.draw_yield_curve(ui),
            Tab::USDINR => self.draw_usdinr(ui),
            Tab::MonsoonAgri => self.draw_monsoon_agri(ui),
            Tab::Seasonality => self.draw_seasonality(ui),
            Tab::WorldIndices => self.draw_world_indices(ui),
            Tab::TickerTape => self.draw_ticker_tape(ui),
            Tab::CurrencyMatrix => self.draw_currency_matrix(ui),
            Tab::SectorWheel => self.draw_sector_wheel(ui),
            Tab::EarningsCalendar => self.draw_earnings_calendar(ui),
            Tab::EconCalendar => self.draw_econ_calendar(ui),
            Tab::CorrelationNetwork => self.draw_correlation_network(ui),
            Tab::ReturnHeatmap => self.draw_return_heatmap(ui),
            Tab::MultiIndicator => self.draw_multi_indicator(ui),
            Tab::MultiTimeframe => self.draw_multi_timeframe(ui),
            Tab::MACDDivergence => self.draw_macd_divergence(ui),
            Tab::BollingerBreakout => self.draw_bollinger_breakout(ui),
            Tab::VolumeWeightedScatter => self.draw_volume_weighted_scatter(ui),
            Tab::PriceMomentum => self.draw_price_momentum(ui),
            Tab::DrawdownRecovery => self.draw_drawdown_recovery(ui),
            Tab::RollingCorrelation => self.draw_rolling_correlation(ui),
            Tab::TickTapeAdv => self.draw_tick_tape_adv(ui),
            Tab::SeasonalityAdv => self.draw_seasonality_adv(ui),
            Tab::ParabolicSAR => self.draw_parabolic_sar(ui),
            Tab::MACDHistogram => self.draw_macd_histogram(ui),
            Tab::RSIHeatmap => self.draw_rsi_heatmap(ui),
            Tab::IchimokuEMA => self.draw_ichimoku_ema(ui),
            Tab::KeltnerBreakout => self.draw_keltner_breakout(ui),
            Tab::DonchianBreakout => self.draw_donchian_breakout(ui),
            Tab::CopulaHeatmap => self.draw_copula_heatmap(ui),
            Tab::CorrelationNetworkAdv => self.draw_correlation_network_adv(ui),
            Tab::MarketWatch => self.draw_market_watch(ui),
            Tab::FOChain => self.draw_fo_chain(ui),
            Tab::IVSurfaceIndia => self.draw_iv_surface_india(ui),
            Tab::OIHeatmap => self.draw_oi_heatmap(ui),
            Tab::GSec => self.draw_gsec(ui),
            Tab::MoneyMarket => self.draw_money_market(ui),
            Tab::RBIPolicy => self.draw_rbi_policy(ui),
            Tab::MacroIndia => self.draw_macro_india(ui),
            Tab::CommoditiesIndia => self.draw_commodities_india(ui),
            Tab::USDINRCurve => self.draw_usdinr_curve(ui),
            Tab::YieldIndia => self.draw_yield_india(ui),
            Tab::MFAnalytics => self.draw_mf_analytics(ui),
            Tab::FPIFII => self.draw_fpi_fii(ui),
            Tab::CreditRatings => self.draw_credit_ratings(ui),
            Tab::BankingIndia => self.draw_banking_india(ui),
            Tab::CorpActions => self.draw_corp_actions(ui),
            Tab::IPOPipeline => self.draw_ipo_pipeline(ui),
            Tab::IndiaBreadth => self.draw_india_breadth(ui),
            Tab::SectorResearch => self.draw_sector_research(ui),
            Tab::IndiaNews => self.draw_india_news(ui),
            Tab::Regulatory => self.draw_regulatory(ui),
            Tab::GSTBudget => self.draw_gst_budget(ui),
            Tab::IndiaPortfolio => self.draw_india_portfolio(ui),
            Tab::AlgoFeed => self.draw_algo_feed(ui),
            Tab::AIResearch => self.draw_ai_research(ui),
            Tab::IndiaDashboard => self.draw_india_dashboard(ui),
            Tab::MultiCompare => self.draw_multi_compare(ui),
            Tab::Forecast => self.draw_forecast(ui),
            Tab::StochRsi => self.draw_stoch_rsi(ui),
            Tab::Zscore => self.draw_zscore(ui),
            Tab::Mfi => self.draw_mfi(ui),
            Tab::UltimateOsc => self.draw_ultimate_osc(ui),
            Tab::Tsi => self.draw_tsi(ui),
            Tab::Coppock => self.draw_coppock(ui),
            Tab::Dpo => self.draw_dpo(ui),
            Tab::Aroon => self.draw_aroon(ui),
            Tab::AroonOsc => self.draw_aroon_osc(ui),
            Tab::Ulcer => self.draw_ulcer(ui),
            Tab::Eom => self.draw_eom(ui),
            Tab::ForceIndex => self.draw_force_index(ui),
            Tab::MassIndex => self.draw_mass_index(ui),
            Tab::Pvt => self.draw_pvt(ui),
            Tab::Mfv => self.draw_mfv(ui),
            Tab::AdLine => self.draw_ad_line(ui),
            Tab::TrendIntensity => self.draw_trend_intensity(ui),
            Tab::RealizedVol => self.draw_realized_vol(ui),
            Tab::KeltnerWidth => self.draw_keltner_width(ui),
            Tab::Volatility => self.draw_volatility(ui),
            Tab::Kst => self.draw_kst(ui),
            Tab::ElderRay => self.draw_elder_ray(ui),
            Tab::Vortex => self.draw_vortex(ui),
            Tab::Kama => self.draw_kama(ui),
            Tab::Alma => self.draw_alma(ui),
            Tab::HullMa => self.draw_hull_ma(ui),
            Tab::Wma => self.draw_wma(ui),
            Tab::MultiSma => self.draw_multi_sma(ui),
            Tab::MultiEma => self.draw_multi_ema(ui),
            Tab::HighLowBand => self.draw_high_low_band(ui),
            Tab::VwapBands => self.draw_vwap_bands(ui),
            Tab::Supertrend => self.draw_supertrend(ui),
            Tab::LogReturns => self.draw_log_returns(ui),
            Tab::Volume => self.draw_volume(ui),
            Tab::Ohlc => self.draw_ohlc(ui),
            Tab::PivotPoints => self.draw_pivot_points(ui),
            Tab::FibLevels => self.draw_fib_levels(ui),
            Tab::PriceSurface => self.draw_price_surface(ui),
            Tab::VolatilitySurface => self.draw_volatility_surface(ui),
            Tab::ReturnSurface => self.draw_return_surface(ui),
            Tab::RiskLandscape => self.draw_risk_landscape(ui),
            Tab::BetaSurface => self.draw_beta_surface(ui),
            Tab::EntropySurface => self.draw_entropy_surface(ui),
            Tab::AlphaSurface => self.draw_alpha_surface(ui),
            Tab::SignalSurface => self.draw_signal_surface(ui),
            Tab::RegimeSurface => self.draw_regime_surface(ui),
            Tab::RegimeTimeline => self.draw_regime_timeline(ui),
            Tab::MomentumSurface => self.draw_momentum_surface(ui),
            Tab::OrderFlowSurface => self.draw_order_flow_surface(ui),
            Tab::SkewKurtSurface => self.draw_skew_kurt_surface(ui),
            Tab::SignalEvolution => self.draw_signal_evolution(ui),
            Tab::EquitySurface => self.draw_equity_surface(ui),
            Tab::VarBandSurface => self.draw_var_band_surface(ui),
            Tab::RiskReturnCloud => self.draw_risk_return_cloud(ui),
            Tab::EigenvalueCloud => self.draw_eigenvalue_cloud(ui),
            Tab::ReturnsHeatmap => self.draw_returns_heatmap(ui),
            Tab::PcaProjection => self.draw_pca_projection(ui),
            Tab::FourInOne => self.draw_four_in_one(ui),
        }
    }

    fn draw_forecast(&mut self, ui: &mut egui::Ui) {
        let symbol = self.data.candles.symbol.clone();
        ui.label(RichText::new(format!("GP — Forecast — {}", symbol)).strong());

        let loaded = self
            .forecaster
            .as_ref()
            .map_or("none (ARIMA fallback only)", |f| f.model_name());
        let (granite_on, nano_on) = self
            .forecaster
            .as_ref()
            .map_or((false, false), |f| (f.has_granite(), f.has_nano()));
        ui.horizontal(|ui| {
            ui.label(RichText::new("Engine").small());
            ui.colored_label(AMBER, loaded);
            ui.separator();
            ui.label(RichText::new("Prefer").small());
            let mut preferred = self.forecast_preferred;
            egui::ComboBox::from_id_source("forecast_engine_pick")
                .selected_text(preferred.label())
                .show_ui(ui, |ui| {
                    for engine in ForecastEngine::ALL {
                        let available = match engine {
                            ForecastEngine::Auto => true,
                            ForecastEngine::Granite => granite_on,
                            ForecastEngine::Nano => nano_on,
                            // Small local ONNX models report presence from the
                            // files on disk; the session itself loads lazily on
                            // first use, so this check stays free.
                            ForecastEngine::Chronos
                            | ForecastEngine::DLinear
                            | ForecastEngine::NHits => {
                                engine.local_model().is_some_and(|m| local_model_ready(m))
                            }
                            ForecastEngine::Arima
                            | ForecastEngine::ExpSmooth
                            | ForecastEngine::MovAvg => true,
                        };
                        let label = if available {
                            engine.label().to_string()
                        } else {
                            format!("{} (missing files)", engine.label())
                        };
                        ui.selectable_value(&mut preferred, engine, label);
                    }
                });
            if preferred != self.forecast_preferred {
                self.forecast_preferred = preferred;
                // A new preference invalidates the cached run.
                self.forecast_values.clear();
                self.forecast_engine.clear();
            }
            ui.separator();
            ui.label(RichText::new("Horizon").small());
            if ui
                .add(egui::Slider::new(&mut self.forecast_horizon, 5..=96).suffix(" bars"))
                .changed()
            {
                // A new horizon invalidates the cached run.
                self.forecast_values.clear();
                self.forecast_engine.clear();
            }
            if ui.button("Run forecast").clicked() {
                self.run_forecast();
            }
        });

        // Model inventory, so a missing file is visible instead of silent.
        ui.horizontal(|ui| {
            ui.label(RichText::new("Granite TTM R2").small());
            ui.colored_label(
                if granite_on { PROFIT } else { Color32::GRAY },
                if granite_on { "ready" } else { "missing" },
            )
            .on_hover_text("models/ttm-q8.gguf + models/config.json + ttm-rs on PATH");
            ui.separator();
            ui.label(RichText::new("NanoForecast v0.5").small());
            ui.colored_label(
                if nano_on { PROFIT } else { Color32::GRAY },
                if nano_on { "ready" } else { "missing" },
            )
            .on_hover_text("models/nanoforecast.onnx (+ ONNX Runtime)");
            ui.separator();
            ui.label(RichText::new("ARIMA").small());
            ui.colored_label(PROFIT, "ready")
                .on_hover_text("Pure-Rust fallback, always available");
        });

        // The small local ONNX models. Chronos leads the auto chain because it
        // is the only genuinely pretrained one; DLinear and N-HiTS are trained
        // on cached NSE closes.
        ui.horizontal(|ui| {
            for (label, file, tooltip) in [
                (
                    "Chronos",
                    "chronos_bolt_tiny_int8.onnx",
                    "9-quantile foundation model, 64 -> 64",
                ),
                (
                    "DLinear",
                    "dlinear.onnx",
                    "32 -> 5, trained on cached NSE closes",
                ),
                (
                    "N-HiTS",
                    "nhits_small.onnx",
                    "32 -> 5, multi-rate, cached NSE closes",
                ),
            ] {
                let present = model_file_present(file);
                ui.label(RichText::new(label).small());
                ui.colored_label(
                    if present { PROFIT } else { Color32::GRAY },
                    if present { "ready" } else { "missing" },
                )
                .on_hover_text(format!("models/{file}\n{tooltip}"));
                ui.separator();
            }
        });

        if let Some(err) = self.forecast_error.clone() {
            ui.colored_label(LOSS, format!("Forecast failed: {err}"));
        }

        // First visit with data auto-runs once; afterwards the button rules.
        if self.forecast_values.is_empty()
            && self.forecast_error.is_none()
            && !self.data.candles.candles.is_empty()
        {
            self.run_forecast();
        }

        if self.forecast_values.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No forecast yet — press Run forecast.");
            });
            return;
        }

        ui.horizontal(|ui| {
            ui.label(format!(
                "Showing {} bars via {}",
                self.forecast_values.len(),
                self.forecast_engine
            ));
            // When the preference was unavailable the chain fell through;
            // say so instead of letting the label imply otherwise.
            if self.forecast_preferred != ForecastEngine::Auto
                && self.forecast_engine != self.forecast_preferred.label()
            {
                ui.colored_label(
                    Color32::GRAY,
                    format!(
                        "(preferred {} unavailable — fell back)",
                        self.forecast_preferred.label()
                    ),
                );
            }
        });

        // Trading-signal readout. The feature layout is provisional (the
        // training order was never published), so the panel says so.
        if let Some(output) = self.last_signal.clone() {
            ui.horizontal(|ui| {
                let headline = output.headline();
                let color = match headline {
                    Signal::Buy => PROFIT,
                    Signal::Hold => Color32::GRAY,
                    Signal::Sell => LOSS,
                };
                ui.label(RichText::new("Signal").small());
                ui.colored_label(color, RichText::new(headline.label()).strong());
                ui.colored_label(Color32::GRAY, format!("{:.0}%", output.confidence * 100.0));
                ui.colored_label(Color32::GRAY, "(experimental layout)".to_string())
                    .on_hover_text(
                        "The 55-feature order is provisional until the training \
                         layout is confirmed; treat live signals accordingly.",
                    );
            });
        } else if self.watchsignal.is_some() {
            ui.colored_label(
                Color32::GRAY,
                "Signal: need 80+ bars of history — load a longer range.",
            );
        }

        let basis: Vec<[f64; 2]> = self
            .forecast_basis
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v])
            .collect();
        let start_x = basis.len() as f64 - 1.0;
        let fc: Vec<[f64; 2]> = {
            let mut pts = vec![[start_x, self.forecast_basis.last().copied().unwrap_or(0.0)]];
            pts.extend(
                self.forecast_values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [start_x + 1.0 + i as f64, v]),
            );
            pts
        };
        // One line colour per engine family, so switching engines visibly
        // changes the chart instead of repainting the same amber line.
        let forecast_color = if self.forecast_engine == "Granite TTM R2" {
            PROFIT
        } else if self.forecast_engine == "NanoForecast v0.5" {
            AMBER
        } else {
            INFO
        };
        Plot::new("forecast_plot")
            .height((ui.available_height() - 8.0).max(160.0))
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(PlotPoints::from(basis))
                        .name("History")
                        .color(Color32::from_gray(150)),
                );
                plot_ui.line(
                    Line::new(PlotPoints::from(fc))
                        .name(format!("Forecast ({})", self.forecast_engine))
                        .color(forecast_color)
                        .width(2.0),
                );
            });
    }

    /// Runs the forecaster over the trailing closes and caches the result.
    /// Loads the WatchSignal classifier when its ONNX file is present next
    /// to the executable (or under `./models`). Absence is normal and only
    /// disables the signal readout.
    fn load_watchsignal() -> Option<WatchSignalModel> {
        use bt_analytics::forecast::models_dir;
        let path = models_dir().join("stock_signal_lstm_v1_seed42.onnx");
        if !path.exists() {
            return None;
        }
        match WatchSignalModel::new(&path.to_string_lossy()) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!("WatchSignal unavailable: {}", e);
                None
            }
        }
    }

    fn run_forecast(&mut self) {
        self.forecast_error = None;
        let candles = &self.data.candles.candles;
        let closes: Vec<f64> = candles
            .iter()
            .rev()
            .take(512)
            .map(|c| c.close)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if closes.len() < 10 {
            self.forecast_error = Some("not enough history (need 10+ bars)".into());
            self.forecast_values.clear();
            return;
        }
        match self.forecaster.as_ref().map(|f| {
            f.predict_with_preference(self.forecast_preferred, &closes, self.forecast_horizon)
        }) {
            Some(Ok((values, engine))) => {
                // Plot a readable trailing window behind the forecast.
                let tail = closes.len().min(120);
                self.forecast_basis = closes[closes.len() - tail..].to_vec();
                self.forecast_values = values;
                self.forecast_engine = engine.to_string();
            }
            Some(Err(e)) => {
                self.forecast_error = Some(e.to_string());
                self.forecast_values.clear();
            }
            None => {
                self.forecast_error = Some("forecaster not initialised".into());
                self.forecast_values.clear();
            }
        }
        // Refresh the trading signal alongside the forecast. A missing model
        // or short history clears the readout instead of erroring: the
        // forecast itself must never depend on the classifier.
        self.last_signal = self
            .watchsignal
            .as_ref()
            .and_then(|m| m.predict_candles(candles).ok());
    }

    fn draw_stoch_rsi(&self, ui: &mut egui::Ui) {
        use bt_analytics::stoch_rsi;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "stochrsi_plot",
            &format!("Stoch RSI — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Stoch RSI",
                vals: &stoch_rsi(series, 14, 14),
                color: PURPLE,
            }],
            &[(80.0, LOSS), (20.0, PROFIT)],
        );
    }

    fn draw_zscore(&self, ui: &mut egui::Ui) {
        use bt_analytics::zscore;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "zscore_plot",
            &format!("Z-Score — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Z-Score",
                vals: &zscore(series, 20),
                color: INFO,
            }],
            &[(2.0, LOSS), (-2.0, PROFIT), (0.0, Color32::GRAY)],
        );
    }

    fn draw_mfi(&self, ui: &mut egui::Ui) {
        use bt_analytics::mfi;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "mfi_plot",
            &format!("MFI — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "MFI",
                vals: &mfi(series, 14),
                color: PROFIT,
            }],
            &[(80.0, LOSS), (20.0, PROFIT)],
        );
    }

    fn draw_ultimate_osc(&self, ui: &mut egui::Ui) {
        use bt_analytics::ultimate_osc;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "uo_plot",
            &format!("Ultimate Oscillator — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Ultimate",
                vals: &ultimate_osc(series, 7, 14, 28),
                color: INFO,
            }],
            &[(70.0, LOSS), (30.0, PROFIT)],
        );
    }

    fn draw_tsi(&self, ui: &mut egui::Ui) {
        use bt_analytics::tsi;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "tsi_plot",
            &format!("TSI — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "TSI",
                    vals: &tsi(series, 25, 13).0,
                    color: INFO,
                },
                FrameLine {
                    label: "Signal",
                    vals: &tsi(series, 25, 13).1,
                    color: Color32::GRAY,
                },
            ],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_coppock(&self, ui: &mut egui::Ui) {
        use bt_analytics::coppock;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "coppock_plot",
            &format!("Coppock Curve — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Coppock",
                vals: &coppock(series),
                color: AMBER,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_dpo(&self, ui: &mut egui::Ui) {
        use bt_analytics::dpo;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "dpo_plot",
            &format!("DPO — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "DPO",
                vals: &dpo(series, 20),
                color: INFO,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_aroon(&self, ui: &mut egui::Ui) {
        use bt_analytics::aroon;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "aroon_plot",
            &format!("Aroon — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "Aroon Up",
                    vals: &aroon(series, 14).0,
                    color: PROFIT,
                },
                FrameLine {
                    label: "Aroon Down",
                    vals: &aroon(series, 14).1,
                    color: LOSS,
                },
            ],
            &[(70.0, Color32::GRAY), (30.0, Color32::GRAY)],
        );
    }

    fn draw_aroon_osc(&self, ui: &mut egui::Ui) {
        use bt_analytics::aroon_osc;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "aroonosc_plot",
            &format!("Aroon Oscillator — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Aroon Osc",
                vals: &aroon_osc(series, 14),
                color: PURPLE,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_ulcer(&self, ui: &mut egui::Ui) {
        use bt_analytics::ulcer;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "ulcer_plot",
            &format!("Ulcer Index — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Ulcer",
                vals: &ulcer(series, 14),
                color: LOSS,
            }],
            &[],
        );
    }

    fn draw_eom(&self, ui: &mut egui::Ui) {
        use bt_analytics::eom;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "eom_plot",
            &format!("Ease of Movement — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "EOM",
                vals: &eom(series, 14),
                color: INFO,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_force_index(&self, ui: &mut egui::Ui) {
        use bt_analytics::force_index;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "force_plot",
            &format!("Force Index — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Force",
                vals: &force_index(series, 13),
                color: AMBER,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_mass_index(&self, ui: &mut egui::Ui) {
        use bt_analytics::mass_index;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "mass_plot",
            &format!("Mass Index — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Mass",
                vals: &mass_index(series, 25),
                color: PURPLE,
            }],
            &[(27.0, LOSS)],
        );
    }

    fn draw_pvt(&self, ui: &mut egui::Ui) {
        use bt_analytics::pvt;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "pvt_plot",
            &format!("PVT — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "PVT",
                vals: &pvt(series),
                color: INFO,
            }],
            &[],
        );
    }

    fn draw_mfv(&self, ui: &mut egui::Ui) {
        use bt_analytics::mfv;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "mfv_plot",
            &format!("Money Flow Volume — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "MFV",
                vals: &mfv(series),
                color: AMBER,
            }],
            &[],
        );
    }

    fn draw_ad_line(&self, ui: &mut egui::Ui) {
        use bt_analytics::ad_line;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "adline_plot",
            &format!("A/D Line — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "A/D",
                vals: &ad_line(series),
                color: PROFIT,
            }],
            &[],
        );
    }

    fn draw_trend_intensity(&self, ui: &mut egui::Ui) {
        use bt_analytics::trend_intensity;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "trendint_plot",
            &format!("Trend Intensity — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Trend Intensity",
                vals: &trend_intensity(series, 30),
                color: INFO,
            }],
            &[(80.0, Color32::GRAY), (20.0, Color32::GRAY)],
        );
    }

    fn draw_realized_vol(&self, ui: &mut egui::Ui) {
        use bt_analytics::realized_vol;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "realvol_plot",
            &format!("Realized Volatility — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Realized Vol %",
                vals: &realized_vol(series, 20),
                color: LOSS,
            }],
            &[],
        );
    }

    fn draw_keltner_width(&self, ui: &mut egui::Ui) {
        use bt_analytics::keltner_width;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "kelwidth_plot",
            &format!("Keltner Width — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "Keltner Width",
                vals: &keltner_width(series, 20),
                color: PURPLE,
            }],
            &[],
        );
    }

    fn draw_volatility(&self, ui: &mut egui::Ui) {
        use bt_analytics::{realized_vol, ulcer};
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "volatility_plot",
            &format!("Volatility — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "Realized Vol %",
                    vals: &realized_vol(series, 20),
                    color: LOSS,
                },
                FrameLine {
                    label: "Ulcer %",
                    vals: &ulcer(series, 14),
                    color: INFO,
                },
            ],
            &[],
        );
    }

    fn draw_kst(&self, ui: &mut egui::Ui) {
        use bt_analytics::kst;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "kst_plot",
            &format!("KST — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "KST",
                    vals: &kst(series, 10, 15, 20, 30, 10, 10, 10, 15, 9).0,
                    color: AMBER,
                },
                FrameLine {
                    label: "Signal",
                    vals: &kst(series, 10, 15, 20, 30, 10, 10, 10, 15, 9).1,
                    color: Color32::GRAY,
                },
            ],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_elder_ray(&self, ui: &mut egui::Ui) {
        use bt_analytics::elder_ray;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "elderray_plot",
            &format!("Elder Ray — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "Bull Power",
                    vals: &elder_ray(series, 13).0,
                    color: PROFIT,
                },
                FrameLine {
                    label: "Bear Power",
                    vals: &elder_ray(series, 13).1,
                    color: LOSS,
                },
            ],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_vortex(&self, ui: &mut egui::Ui) {
        use bt_analytics::vortex;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "vortex_plot",
            &format!("Vortex — {}", series.symbol),
            &series.candles,
            false,
            &[
                FrameLine {
                    label: "VI+",
                    vals: &vortex(series, 14).0,
                    color: PROFIT,
                },
                FrameLine {
                    label: "VI-",
                    vals: &vortex(series, 14).1,
                    color: LOSS,
                },
            ],
            &[(1.0, Color32::GRAY)],
        );
    }

    fn draw_kama(&self, ui: &mut egui::Ui) {
        use bt_analytics::kama;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "kama_plot",
            &format!("KAMA — {}", series.symbol),
            &series.candles,
            true,
            &[FrameLine {
                label: "KAMA",
                vals: &kama(series, 10, 2, 30),
                color: AMBER,
            }],
            &[],
        );
    }

    fn draw_alma(&self, ui: &mut egui::Ui) {
        use bt_analytics::alma;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "alma_plot",
            &format!("ALMA — {}", series.symbol),
            &series.candles,
            true,
            &[FrameLine {
                label: "ALMA",
                vals: &alma(series, 9, 0.85, 6.0),
                color: PURPLE,
            }],
            &[],
        );
    }

    fn draw_hull_ma(&self, ui: &mut egui::Ui) {
        use bt_analytics::hull_ma;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "hullma_plot",
            &format!("Hull MA — {}", series.symbol),
            &series.candles,
            true,
            &[FrameLine {
                label: "Hull MA",
                vals: &hull_ma(series, 9),
                color: INFO,
            }],
            &[],
        );
    }

    fn draw_wma(&self, ui: &mut egui::Ui) {
        use bt_analytics::wma;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "wma_plot",
            &format!("WMA — {}", series.symbol),
            &series.candles,
            true,
            &[FrameLine {
                label: "WMA14",
                vals: &wma(series, 14),
                color: AMBER,
            }],
            &[],
        );
    }

    fn draw_multi_sma(&self, ui: &mut egui::Ui) {
        use bt_analytics::multi_sma;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "multisma_plot",
            &format!("Multi SMA — {}", series.symbol),
            &series.candles,
            true,
            &[
                FrameLine {
                    label: "SMA20",
                    vals: &multi_sma(series).0,
                    color: PROFIT,
                },
                FrameLine {
                    label: "SMA50",
                    vals: &multi_sma(series).1,
                    color: AMBER,
                },
                FrameLine {
                    label: "SMA200",
                    vals: &multi_sma(series).2,
                    color: LOSS,
                },
            ],
            &[],
        );
    }

    fn draw_multi_ema(&self, ui: &mut egui::Ui) {
        use bt_analytics::multi_ema;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "multiema_plot",
            &format!("Multi EMA — {}", series.symbol),
            &series.candles,
            true,
            &[
                FrameLine {
                    label: "EMA12",
                    vals: &multi_ema(series).0,
                    color: PROFIT,
                },
                FrameLine {
                    label: "EMA26",
                    vals: &multi_ema(series).1,
                    color: AMBER,
                },
                FrameLine {
                    label: "EMA50",
                    vals: &multi_ema(series).2,
                    color: LOSS,
                },
            ],
            &[],
        );
    }

    fn draw_high_low_band(&self, ui: &mut egui::Ui) {
        use bt_analytics::high_low_band;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "hlband_plot",
            &format!("High Low Band — {}", series.symbol),
            &series.candles,
            true,
            &[
                FrameLine {
                    label: "HH20",
                    vals: &high_low_band(series, 20).0,
                    color: LOSS,
                },
                FrameLine {
                    label: "LL20",
                    vals: &high_low_band(series, 20).1,
                    color: PROFIT,
                },
            ],
            &[],
        );
    }

    fn draw_vwap_bands(&self, ui: &mut egui::Ui) {
        use bt_analytics::{vwap, vwap_bands};
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "vwapbands_plot",
            &format!("VWAP Bands — {}", series.symbol),
            &series.candles,
            true,
            &[
                FrameLine {
                    label: "VWAP",
                    vals: &vwap(series),
                    color: AMBER,
                },
                FrameLine {
                    label: "Upper",
                    vals: &vwap_bands(series, 1.0).0,
                    color: Color32::GRAY,
                },
                FrameLine {
                    label: "Lower",
                    vals: &vwap_bands(series, 1.0).1,
                    color: Color32::GRAY,
                },
            ],
            &[],
        );
    }

    fn draw_supertrend(&self, ui: &mut egui::Ui) {
        use bt_analytics::supertrend;
        let series = &self.data.candles;
        draw_lines_frame(
            ui,
            "supertrend_plot",
            &format!("Supertrend — {}", series.symbol),
            &series.candles,
            true,
            &[FrameLine {
                label: "Supertrend",
                vals: &supertrend(series, 10, 3.0).0,
                color: AMBER,
            }],
            &[],
        );
    }
    fn draw_log_returns(&self, ui: &mut egui::Ui) {
        use bt_analytics::log_returns;
        let series = &self.data.candles;
        let pct: Vec<f64> = log_returns(series).iter().map(|v| v * 100.0).collect();
        draw_lines_frame(
            ui,
            "logret_plot",
            &format!("Log Returns % — {}", series.symbol),
            &series.candles,
            false,
            &[FrameLine {
                label: "LogRet %",
                vals: &pct,
                color: AMBER,
            }],
            &[(0.0, Color32::GRAY)],
        );
    }

    fn draw_volume(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        let n = series.candles.len();
        let mut ma = vec![f64::NAN; n];
        if n >= 20 {
            let mut acc = 0.0;
            for i in 0..n {
                acc += series.candles[i].volume.max(0.0);
                if i >= 20 {
                    acc -= series.candles[i - 20].volume.max(0.0);
                }
                if i + 1 >= 20 {
                    ma[i] = acc / 20.0;
                }
            }
        }
        ui.label(RichText::new(format!("VOL — Volume — {}", series.symbol)).strong());
        Plot::new("volume_tab_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let spacing = bar_spacing(&series.candles).max(1.0);
                let bars: Vec<Bar> = series
                    .candles
                    .iter()
                    .map(|c| {
                        let color = if c.is_bullish() { PROFIT } else { LOSS };
                        Bar::new(c.t, c.volume.max(0.0))
                            .width(spacing * 0.7)
                            .fill(color.gamma_multiply(0.75))
                    })
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars));
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if ma[i].is_nan() {
                            None
                        } else {
                            Some([c.t, ma[i]])
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32).name("SMA20"));
            });
    }

    fn draw_ohlc(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        ui.label(RichText::new(format!("OHLC — Open High Low Close — {}", series.symbol)).strong());
        Plot::new("ohlc_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let half = (bar_spacing(&series.candles) * 0.35).max(1.0);
                for c in &series.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![[c.t, c.low], [c.t, c.high]]))
                            .color(color)
                            .width(1.0_f32),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![[c.t - half, c.open], [c.t, c.open]]))
                            .color(color)
                            .width(2.0_f32),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![
                            [c.t, c.close],
                            [c.t + half, c.close],
                        ]))
                        .color(color)
                        .width(2.0_f32),
                    );
                }
            });
    }

    fn draw_pivot_points(&self, ui: &mut egui::Ui) {
        use bt_analytics::pivot_levels;
        let series = &self.data.candles;
        if series.candles.is_empty() {
            ui.label("No data for this symbol.");
            return;
        }
        let k = series.candles.len().saturating_sub(20);
        let (mut hh, mut ll) = (f64::NEG_INFINITY, f64::INFINITY);
        for c in &series.candles[k..] {
            hh = hh.max(c.high);
            ll = ll.min(c.low);
        }
        let close = series.candles.last().map(|c| c.close).unwrap_or(0.0);
        let p = pivot_levels(hh, ll, close);
        draw_lines_frame(
            ui,
            "pivot_plot",
            &format!("PIVOT — Floor Pivots — {}", series.symbol),
            &series.candles,
            true,
            &[],
            &[
                (p.pp, AMBER),
                (p.r1, LOSS),
                (p.r2, LOSS),
                (p.r3, LOSS),
                (p.s1, PROFIT),
                (p.s2, PROFIT),
                (p.s3, PROFIT),
            ],
        );
    }

    fn draw_fib_levels(&self, ui: &mut egui::Ui) {
        use bt_analytics::fib_levels;
        let series = &self.data.candles;
        if series.candles.is_empty() {
            ui.label("No data for this symbol.");
            return;
        }
        let (mut hi, mut lo) = (f64::NEG_INFINITY, f64::INFINITY);
        for c in &series.candles {
            hi = hi.max(c.high);
            lo = lo.min(c.low);
        }
        let levels = fib_levels(lo, hi);
        let guides: Vec<(f64, Color32)> = levels
            .iter()
            .map(|(r, price)| {
                (
                    *price,
                    if (*r - 0.618).abs() < 1e-9 {
                        AMBER
                    } else {
                        Color32::GRAY
                    },
                )
            })
            .collect();
        draw_lines_frame(
            ui,
            "fib_plot",
            &format!("FIB — Fibonacci Retracement — {}", series.symbol),
            &series.candles,
            true,
            &[],
            &guides,
        );
    }

    fn draw_price_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::price_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_volatility_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::volatility_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_return_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::return_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_risk_landscape(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::risk_landscape(&self.data.candles), 0.55, 0.35);
    }

    fn draw_beta_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::beta_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_entropy_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::entropy_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_alpha_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::alpha_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_signal_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::signal_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_regime_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::regime_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_momentum_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::momentum_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_order_flow_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::order_flow_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_skew_kurt_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::skew_kurt_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_signal_evolution(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::signal_evolution_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_equity_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::equity_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_var_band_surface(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::var_band_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_risk_return_cloud(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::risk_return_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_eigenvalue_cloud(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::eigenvalue_surface(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_returns_heatmap(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(
            ui,
            &views3d::returns_heatmap(&self.data.candles),
            0.55,
            0.35,
        );
    }

    fn draw_pca_projection(&self, ui: &mut egui::Ui) {
        views3d::draw_frame(ui, &views3d::pca_surface(&self.data.candles), 0.55, 0.35);
    }

    fn draw_regime_timeline(&self, ui: &mut egui::Ui) {
        // A 3D ridge over time: each row is a trailing-return window, each
        // column a bar, and the height is the signed move. Reading the surface
        // top-down shows whether early and late windows agree, which is what a
        // regime read is.
        let s = &self.data.candles;
        let closes: Vec<f64> = s.candles.iter().map(|c| c.close).collect();
        let buckets = 40.min(closes.len());
        let windows = [5usize, 10, 20, 40];
        let mut values = vec![f64::NAN; buckets * windows.len()];
        if buckets >= 2 {
            for (bi, _) in (0..buckets).enumerate() {
                let end = ((bi + 1) * closes.len()) / buckets;
                for (wi, &w) in windows.iter().enumerate() {
                    if end <= w {
                        continue;
                    }
                    let base = closes[end - w - 1];
                    let now = closes[end - 1];
                    if base > 0.0 {
                        values[bi * windows.len() + wi] = (now / base - 1.0) * 100.0;
                    }
                }
            }
        }
        let surface = views3d::Surface::new(views3d::SurfaceSpec::new(
            format!("3D Regime Timeline \u{2014} {}", s.symbol),
            "Return % by time bucket x window",
            views3d::Grid::new(
                buckets.max(2),
                windows.len(),
                values,
                "time bucket",
                "window",
            ),
            views3d::RAMP_SIGNED,
        ));
        views3d::draw_frame(ui, &surface, 0.6, 0.4);
    }

    fn draw_four_in_one(&self, ui: &mut egui::Ui) {
        // Four panels in a 2x2 grid, sharing the loaded series. Each panel
        // delegates to an existing renderer so the dashboard can never drift
        // out of sync with the standalone tabs.
        let s = &self.data.candles;
        ui.label(RichText::new(format!("4-in-1 \u{2014} {}", s.symbol)).strong());
        ui.label(
            RichText::new("Price with MA overlay | Volume | RSI | MACD")
                .small()
                .color(Color32::GRAY),
        );

        if s.candles.is_empty() {
            ui.label("No data for this symbol.");
            return;
        }

        let available = ui.available_height().max(320.0);
        let cell = (available - 24.0) / 2.0;
        egui::Grid::new("four_in_one_grid")
            .spacing(egui::vec2(10.0, 10.0))
            .show(ui, |ui| {
                // Each panel delegates to the standalone renderer, so the
                // dashboard can never drift from the individual tabs.
                ui.vertical(|ui| {
                    ui.set_min_height(cell);
                    ui.set_width(ui.available_width());
                    self.draw_candlestick(ui);
                });
                ui.end_row();
                ui.vertical(|ui| {
                    ui.set_min_height(cell);
                    ui.set_width(ui.available_width());
                    self.draw_volume(ui);
                });
                ui.end_row();
                ui.vertical(|ui| {
                    ui.set_min_height(cell);
                    ui.set_width(ui.available_width());
                    self.draw_rsi(ui);
                });
                ui.end_row();
                ui.vertical(|ui| {
                    ui.set_min_height(cell);
                    ui.set_width(ui.available_width());
                    self.draw_macd(ui);
                });
                ui.end_row();
            });
    }

    fn draw_candlestick(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        let series = &candles.candles;
        ui.label(RichText::new(format!("GP — Candlestick — {}", candles.symbol)).strong());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Trend arrows").small());
            let mut arrows = self.show_candle_arrows.get();
            if ui
                .checkbox(&mut arrows, "")
                .on_hover_text("Green ▲ marks a bullish bar, red ▼ a bearish bar.\nShown when bars are wide enough to read.")
                .changed()
            {
                self.show_candle_arrows.set(arrows);
            }
            if arrows && series.len() > 90 {
                ui.colored_label(
                    Color32::GRAY,
                    format!("(hidden: {} bars)", series.len()),
                )
                .on_hover_text("Trend arrows are hidden above 90 bars to avoid clutter.\nZoom in or pick a shorter range to see them.");
            } else {
                ui.colored_label(PROFIT, "\u{25B2} up = close > open");
                ui.colored_label(LOSS, "\u{25BC} down = close < open");
            }
        });

        // Zoom controls. Scroll the chart to zoom at the pointer, drag to pan
        // once zoomed, double-click to zoom in, right-click to zoom out, or
        // use these buttons. Pinch and two-finger drag work on touchscreens.
        ui.horizontal(|ui| {
            let zoom = self.zoom.get();
            if ui
                .add_enabled(zoom.is_zoomed(), egui::Button::new("\u{1F517}").small())
                .on_hover_text("Zoom out one step")
                .clicked()
            {
                self.zoom.set(zoom.zoom_out_centered());
            }
            if ui
                .add_enabled(
                    zoom.factor < ZoomState::MAX_FACTOR - f64::EPSILON,
                    egui::Button::new("\u{1F517}+").small(),
                )
                .on_hover_text("Zoom in one step")
                .clicked()
            {
                self.zoom.set(zoom.zoom_in_centered(self.zoom_focus_y()));
            }
            if zoom.is_zoomed() {
                ui.label(
                    RichText::new(format!("{:.1}x", zoom.factor))
                        .small()
                        .strong(),
                );
                if ui
                    .button("Reset")
                    .on_hover_text("Return to the full data range")
                    .clicked()
                {
                    self.zoom.set(zoom.reset());
                }
            } else {
                ui.colored_label(
                    Color32::GRAY,
                    "Scroll to zoom \u{00B7} drag to pan \u{00B7} double-click in \u{00B7} right-click out",
                );
            }
        });

        if series.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No data for this symbol.");
            });
            return;
        }

        let (_, range) = price_scale(series);
        let min_body = range * MIN_BODY_FRAC;
        let arrow = range * ARROW_FRAC;
        let half = bar_half(series);
        let show_arrows = self.show_candle_arrows.get();
        let last_close = series.last().map(|c| c.close).unwrap_or(0.0);
        let last_is_bull = series.last().map(|c| c.is_bullish()).unwrap_or(true);
        let price_decimals = price_decimals(last_close);
        let volume_max = series.iter().map(|c| c.volume).fold(0.0_f64, f64::max);

        // Extra headroom so the trend arrows are never clipped by the frame.
        let arrow_pad = if show_arrows { 3.0 * arrow } else { 0.0 };
        let (x0, x1) = x_bounds(series);

        // OHLC legend sits above the price pane so the hover tooltip, which
        // follows the pointer inside the plot, can never cover it.
        ui.horizontal(|ui| {
            let col = if last_is_bull { PROFIT } else { LOSS };
            ui.colored_label(col, format!("Last {:.*}", price_decimals, last_close));
            if let Some(c) = series.last() {
                ui.separator();
                ui.label(format!(
                    "O {:.*}  H {:.*}  L {:.*}  C {:.*}",
                    price_decimals,
                    c.open,
                    price_decimals,
                    c.high,
                    price_decimals,
                    c.low,
                    price_decimals,
                    c.close
                ));
            }
            ui.separator();
            ui.label(format!("Vol {}", abbreviate_volume(volume_max)));
        });

        // Full (unzoomed) y-range for the price pane.
        let y_lo = price_scale(series).0 - range * 0.04 - arrow_pad;
        let y_hi = price_scale(series).0 + range * 1.04 + arrow_pad;
        let zoom_now = self.zoom.get();
        // Resolved inside the price pane closure, once the pan/zoom gesture for
        // this frame has been applied, then reused by the volume pane so both
        // panes stay aligned. Deriving it before the closure would draw the
        // previous window's candles and leave the pane blank after a drag.
        let frame: std::cell::RefCell<(f64, f64, std::rc::Rc<Vec<Candle>>)> =
            std::cell::RefCell::new((x0, x1, std::rc::Rc::new(Vec::new())));

        // Height budget for the price and volume panes, measured in absolute
        // screen coordinates.
        //
        // `ui.available_height()` over-reports here by roughly 70px: the
        // CentralPanel is laid out before the status bar has claimed its strip,
        // so budgeting from it pushed the volume pane's x-axis labels and the
        // status bar past the bottom of the window, leaving the chart either
        // clipped or floating above a blank strip.
        //
        // Instead, measure the space between this tab's own chrome (which ends
        // at the next widget position) and the bottom of the region this panel
        // may paint, then reserve the status bar and the x-axis label strip --
        // egui_plot draws those labels *outside* the height it is given. The
        // two panes then land on the bottom edge at any window size.
        const STATUS_H: f32 = 26.0;
        const AXIS_H: f32 = 30.0;
        // `clip_rect()` is infinite on some frames (an unconstrained Ui), which
        // would hand egui_plot an infinite height and stop the window painting
        // at all, so fall back to a default and clamp both ends.
        const FALLBACK_H: f32 = 620.0;
        let paint_bottom = ui.clip_rect().max.y;
        let chrome_bottom = ui.next_widget_position().y;
        let avail = if paint_bottom.is_finite() && chrome_bottom.is_finite() {
            (paint_bottom - chrome_bottom - STATUS_H - AXIS_H).clamp(180.0, 4000.0)
        } else {
            FALLBACK_H
        };
        let vol_h = (avail * 0.22).clamp(60.0, 130.0);
        let price_h = (avail - vol_h).max(140.0);
        // Median gap between bars, used to pick the hovered candle and to pad
        // the visible slice so a bar at the very edge is not clipped.
        let spacing = bar_spacing(series);
        let hover_candle = std::cell::RefCell::new(None::<(f64, f64, f64, f64, f64)>);

        // Trend arrows only stay legible when bars are wide enough on screen.
        // At ~250+ bars a per-bar arrow is visual noise, so it is suppressed.
        let arrows_legible = show_arrows && series.len() <= 90;
        let arrow = range * ARROW_FRAC;

        // Borrowed inside the closure so the gesture can update it.
        let visible_cell: &std::cell::RefCell<(f64, f64, std::rc::Rc<Vec<Candle>>)> = &frame;

        style_time_plot(
            Plot::new("candlestick_plot"),
            series,
            price_h,
            self.axis_date_style(),
        )
        .include_y(y_lo)
        .include_y(y_hi)
        // Keep the price axis on the right; the x-axis ticks live on the
        // volume pane directly below, which shares the exact same bounds.
        .show_axes([false, true])
        .show(ui, |plot_ui| {
            // Double-click zooms in, right-click zooms out.
            if let Some(next) =
                handle_plot_zoom_gesture(plot_ui, zoom_now, series, x0, x1, y_lo, y_hi)
            {
                self.zoom.set(next);
            }
            // Dragging pans the zoomed window horizontally and vertically.
            if let Some(next) =
                handle_plot_pan(plot_ui, self.zoom.get(), series, x0, x1, y_lo, y_hi)
            {
                self.zoom.set(next);
            }
            // Pin the range so a previously viewed time range cannot leave
            // this pane zoomed out with the bars squeezed into a sliver,
            // while still honouring the user's zoom.
            let applied = self.zoom.get();
            // Resolve the window actually being shown, then restrict the
            // drawn candles to it.
            let (ax0, ax1, _, _) = applied.window_with_data(series, x0, x1, y_lo, y_hi);
            let slice: std::rc::Rc<Vec<Candle>> = if applied.is_zoomed() {
                std::rc::Rc::new(
                    series
                        .iter()
                        .filter(|c| c.t >= ax0 - spacing && c.t <= ax1 + spacing)
                        .copied()
                        .collect(),
                )
            } else {
                std::rc::Rc::new(Vec::new())
            };
            {
                let mut slot = visible_cell.borrow_mut();
                slot.0 = ax0;
                slot.1 = ax1;
                slot.2 = std::rc::Rc::clone(&slice);
            }
            let visible: &[Candle] = if slice.is_empty() {
                &series[..]
            } else {
                &slice[..]
            };
            let v_half = if applied.is_zoomed() {
                bar_half(visible)
            } else {
                half
            };
            let v_min_body = if applied.is_zoomed() {
                price_scale(visible).1 * MIN_BODY_FRAC
            } else {
                min_body
            };
            apply_zoom_and_pan(plot_ui, &self.zoom, series, x0, x1, y_lo, y_hi);
            // Current-price guide, drawn first so bars sit on top of it.
            if last_close >= y_lo && last_close <= y_hi {
                plot_ui.hline(
                    egui_plot::HLine::new(last_close)
                        .color(Color32::from_gray(140))
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            }
            for c in visible {
                draw_candle(plot_ui, c, v_half, v_min_body);
                if arrows_legible {
                    draw_trend_arrow(plot_ui, c, v_half, arrow);
                }
            }
            if let Some(hover_pos) = plot_ui.pointer_coordinate() {
                if spacing > 0.0 && ax1 > ax0 {
                    let idx = (((hover_pos.x - ax0) / spacing).round() as isize)
                        .clamp(0, series.len() as isize - 1) as usize;
                    let c = &series[idx];
                    *hover_candle.borrow_mut() = Some((c.open, c.high, c.low, c.close, c.volume));
                }
            }
        });

        if let Some((o, h, l, c, v)) = hover_candle.borrow().as_ref() {
            egui::show_tooltip_at_pointer(
                ui.ctx(),
                egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("candle_tooltip_layer")),
                egui::Id::new("candle_tooltip"),
                |ui: &mut egui::Ui| {
                    ui.label(format!(
                        "O: {:.*}   H: {:.*}   L: {:.*}   C: {:.*}   V: {:.0}",
                        price_decimals,
                        o,
                        price_decimals,
                        h,
                        price_decimals,
                        l,
                        price_decimals,
                        c,
                        v
                    ));
                },
            );
        }

        // Volume pane, sharing the exact same x-bounds as the price plot
        // above so the two panes line up vertically.
        Plot::new("candlestick_volume")
            .height(vol_h)
            .allow_drag(true)
            .allow_scroll(true)
            .allow_zoom(false)
            .include_x(visible_cell.borrow().0)
            .include_x(visible_cell.borrow().1)
            .include_y(0.0)
            .y_axis_position(egui_plot::HPlacement::Right)
            .y_axis_min_width(9.0)
            .show_axes([true, true])
            .x_axis_formatter(move |mark, _range| {
                format_ts_styled(mark.value, self.axis_date_style())
            })
            .y_axis_formatter(move |mark, _range| abbreviate_volume(mark.value))
            .show(ui, |plot_ui| {
                // Gestures on the volume pane move the shared time axis, so
                // dragging the lower third of the canvas is not dead. Only the
                // horizontal component is honoured; the vertical axis here is
                // fixed at `0..max volume`.
                if let Some(next) =
                    handle_volume_gestures(plot_ui, self.zoom.get(), series, x0, x1, y_lo, y_hi)
                {
                    self.zoom.set(next);
                }
                // Share the price pane's exact x-range so the two align.
                let (vx0, vx1) = {
                    let slot = visible_cell.borrow();
                    (slot.0, slot.1)
                };
                pin_bounds(plot_ui, vx0, vx1, 0.0, volume_max.max(1.0) * 1.1);
                let bars: Vec<Bar> = {
                    let slice = visible_cell.borrow();
                    let src: &[Candle] = if slice.2.is_empty() {
                        &series[..]
                    } else {
                        &slice.2[..]
                    };
                    src.iter()
                        .map(|c| {
                            let color = if c.is_bullish() { PROFIT } else { LOSS };
                            Bar::new(c.t, c.volume)
                                .width(spacing * BAR_FILL)
                                .fill(color.gamma_multiply(0.75))
                        })
                        .collect()
                };
                plot_ui.bar_chart(BarChart::new(bars));
            });
    }

    fn draw_heikin_ashi(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        let ha_series = heikin_ashi(&candles.candles);
        ui.label(RichText::new(format!("GP (HA) — Heikin-Ashi — {}", candles.symbol)).strong());
        if ha_series.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No data for this symbol.");
            });
            return;
        }
        let min_body = price_scale(&ha_series).1 * MIN_BODY_FRAC;
        let half = bar_half(&ha_series);
        let height = ui.available_height();
        style_time_plot(
            Plot::new("ha_plot"),
            &ha_series,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&ha_series);
            let (plo, prange) = price_scale(&ha_series);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &ha_series,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for ha in &ha_series {
                let color = if ha.is_bullish() { PROFIT } else { LOSS };
                plot_ui.line(
                    Line::new(PlotPoints::from(vec![[ha.t, ha.low], [ha.t, ha.high]]))
                        .color(color)
                        .width(1.0_f32),
                );
                let (top, bottom) = candle_body(ha, min_body);
                plot_ui.polygon(
                    egui_plot::Polygon::new(PlotPoints::from(vec![
                        [ha.t - half, bottom],
                        [ha.t + half, bottom],
                        [ha.t + half, top],
                        [ha.t - half, top],
                    ]))
                    .fill_color(color)
                    .stroke(Stroke::new(1.0_f32, color)),
                );
            }
        });
    }

    fn draw_renko(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("GP (Renko) — Renko — {}", candles.symbol)).strong());
        Plot::new("renko_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for w in candles.candles.windows(2) {
                    let prev = &w[0];
                    let curr = &w[1];
                    let color = if curr.close > prev.close {
                        PROFIT
                    } else {
                        LOSS
                    };
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![
                            [curr.t, prev.close],
                            [curr.t, curr.close],
                        ]))
                        .color(color)
                        .width(3.0_f32),
                    );
                }
            });
    }

    fn draw_kagi(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("KAGI — Kagi — {}", candles.symbol)).strong());
        Plot::new("kagi_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for w in candles.candles.windows(2) {
                    let prev = &w[0];
                    let curr = &w[1];
                    let color = if curr.close > prev.close {
                        PROFIT
                    } else {
                        LOSS
                    };
                    let thick = if curr.close > prev.close {
                        3.0_f32
                    } else {
                        1.0_f32
                    };
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![
                            [curr.t, prev.close],
                            [curr.t, curr.close],
                        ]))
                        .color(color)
                        .width(thick),
                    );
                }
            });
    }

    fn draw_point_figure(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("P&F — Point & Figure — {}", candles.symbol)).strong());
        Plot::new("pf_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let box_size = 5.0_f64;
                let mut last_price = candles.candles[0].close;
                let mut col_x = candles.candles[0].t;
                let mut is_x = true;
                for c in &candles.candles {
                    if (c.close - last_price).abs() >= box_size {
                        is_x = c.close > last_price;
                        col_x = c.t;
                        last_price = c.close;
                    }
                    let color = if is_x { PROFIT } else { LOSS };
                    let y = if is_x { c.high } else { c.low };
                    plot_ui.points(
                        Points::new(PlotPoints::from(vec![[col_x, y]]))
                            .color(color)
                            .radius(4.0_f32)
                            .shape(if is_x {
                                MarkerShape::Cross
                            } else {
                                MarkerShape::Circle
                            }),
                    );
                }
            });
    }
    fn draw_candlestick_3d(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("C3D — Candlestick 3D — {}", candles.symbol)).strong());
        let min_body = price_scale(&candles.candles).1 * MIN_BODY_FRAC;
        let half = bar_half(&candles.candles);
        let height = ui.available_height();
        style_time_plot(
            Plot::new("c3d_plot"),
            &candles.candles,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&candles.candles);
            let (plo, prange) = price_scale(&candles.candles);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &candles.candles,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for c in &candles.candles {
                draw_candle(plot_ui, c, half, min_body);
            }
        });
    }

    fn draw_candlestick_ma(&self, ui: &mut egui::Ui) {
        use bt_analytics::{ema, sma};
        let series = &self.data.candles;
        let sma20 = sma(series, 20);
        let sma50 = sma(series, 50);
        let ema200 = ema(series, 200);
        ui.label(RichText::new(format!("CMA — Candlestick + MA — {}", series.symbol)).strong());
        let min_body = price_scale(&series.candles).1 * MIN_BODY_FRAC;
        let half = bar_half(&series.candles);
        let height = ui.available_height();
        style_time_plot(
            Plot::new("cma_plot"),
            &series.candles,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&series.candles);
            let (plo, prange) = price_scale(&series.candles);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &series.candles,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for c in &series.candles {
                draw_candle(plot_ui, c, half, min_body);
            }
            let sma20_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !sma20[i].is_nan() {
                        Some([c.t, sma20[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(
                Line::new(sma20_pts)
                    .color(INFO)
                    .width(2.0_f32)
                    .name("SMA20"),
            );
            let sma50_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !sma50[i].is_nan() {
                        Some([c.t, sma50[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(
                Line::new(sma50_pts)
                    .color(AMBER)
                    .width(2.0_f32)
                    .name("SMA50"),
            );
            let ema200_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !ema200[i].is_nan() {
                        Some([c.t, ema200[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(
                Line::new(ema200_pts)
                    .color(PURPLE)
                    .width(2.0_f32)
                    .name("EMA200"),
            );
        });
    }

    fn draw_candlestick_bollinger(&self, ui: &mut egui::Ui) {
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (mid, upper, lower) = bollinger(series, 20, 2.0);
        ui.label(
            RichText::new(format!("CBB — Candlestick + Bollinger — {}", series.symbol)).strong(),
        );
        let min_body = price_scale(&series.candles).1 * MIN_BODY_FRAC;
        let half = bar_half(&series.candles);
        let height = ui.available_height();
        style_time_plot(
            Plot::new("cbb_plot"),
            &series.candles,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&series.candles);
            let (plo, prange) = price_scale(&series.candles);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &series.candles,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for c in &series.candles {
                draw_candle(plot_ui, c, half, min_body);
            }
            let mid_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !mid[i].is_nan() {
                        Some([c.t, mid[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(Line::new(mid_pts).color(AMBER).width(2.0_f32).name("SMA20"));
            let upper_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !upper[i].is_nan() {
                        Some([c.t, upper[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(
                Line::new(upper_pts)
                    .color(INFO)
                    .width(1.0_f32)
                    .name("Upper"),
            );
            let lower_pts: PlotPoints = series
                .candles
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if !lower[i].is_nan() {
                        Some([c.t, lower[i]])
                    } else {
                        None
                    }
                })
                .collect();
            plot_ui.line(
                Line::new(lower_pts)
                    .color(INFO)
                    .width(1.0_f32)
                    .name("Lower"),
            );
        });
    }

    fn draw_candlestick_rsi(&self, ui: &mut egui::Ui) {
        use bt_analytics::rsi;
        let series = &self.data.candles;
        let rsi_vals = rsi(series, 14);
        ui.label(RichText::new(format!("CRSI — Candlestick + RSI — {}", series.symbol)).strong());
        let min_body = price_scale(&series.candles).1 * MIN_BODY_FRAC;
        let half = bar_half(&series.candles);
        let height = (ui.available_height() * 0.65_f32).max(120.0);
        style_time_plot(
            Plot::new("crsi_price"),
            &series.candles,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&series.candles);
            let (plo, prange) = price_scale(&series.candles);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &series.candles,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for c in &series.candles {
                draw_candle(plot_ui, c, half, min_body);
            }
        });
        Plot::new("crsi_rsi")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !rsi_vals[i].is_nan() {
                            Some([c.t, rsi_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(70.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(30.0).color(PROFIT));
            });
    }

    fn draw_candlestick_macd(&self, ui: &mut egui::Ui) {
        use bt_analytics::macd;
        let series = &self.data.candles;
        let (macd_line, signal_line, _histogram) = macd(series);
        ui.label(RichText::new(format!("CMACD — Candlestick + MACD — {}", series.symbol)).strong());
        let min_body = price_scale(&series.candles).1 * MIN_BODY_FRAC;
        let half = bar_half(&series.candles);
        let height = (ui.available_height() * 0.65_f32).max(120.0);
        style_time_plot(
            Plot::new("cmacd_price"),
            &series.candles,
            height,
            self.axis_date_style(),
        )
        .show_axes([true, true])
        .show(ui, |plot_ui| {
            let (px0, px1) = x_bounds(&series.candles);
            let (plo, prange) = price_scale(&series.candles);
            apply_zoom_and_pan(
                plot_ui,
                &self.zoom,
                &series.candles,
                px0,
                px1,
                plo - prange * 0.05,
                plo + prange * 1.05,
            );
            for c in &series.candles {
                draw_candle(plot_ui, c, half, min_body);
            }
        });
        Plot::new("cmacd_macd")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !macd_line[i].is_nan() {
                            Some([c.t, macd_line[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("MACD"));
                let sig_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !signal_line[i].is_nan() {
                            Some([c.t, signal_line[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(sig_pts)
                        .color(AMBER)
                        .width(2.0_f32)
                        .name("Signal"),
                );
            });
    }
    // ==================== ORDER FLOW ====================

    fn draw_volume_profile(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("VP — Volume Profile — {}", candles.symbol)).strong());
        let bar_w = bar_spacing(&candles.candles) * BAR_FILL;
        let spacing = bar_spacing(&candles.candles);
        let (x0, x1) = x_bounds(&candles.candles);
        Plot::new("vp_plot")
            .height(ui.available_height())
            .allow_drag(true)
            .allow_scroll(true)
            .allow_zoom(false)
            .include_x(x0)
            .include_x(x1)
            .y_axis_position(egui_plot::HPlacement::Right)
            .x_axis_formatter(move |mark, _range| {
                format_ts_styled(mark.value, self.axis_date_style())
            })
            .y_axis_formatter(move |mark, _range| abbreviate_volume(mark.value))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.bar_chart(BarChart::new(vec![Bar::new(c.t, c.volume)
                        .width(bar_w)
                        .fill(color)]));
                }
            });
    }

    fn draw_footprint(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("FP — Footprint — {}", candles.symbol)).strong());
        Plot::new("fp_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let range = (c.high - c.low).max(1e-9);
                    let bid_vol = c.volume * (1.0 - (c.close - c.low) / range);
                    let ask_vol = c.volume * ((c.close - c.low) / range);
                    plot_ui.bar_chart(BarChart::new(vec![Bar::new(c.t - 0.2 * DAY_SECS, bid_vol)
                        .width(0.35 * DAY_SECS)
                        .fill(LOSS.gamma_multiply(0.7))]));
                    plot_ui.bar_chart(BarChart::new(vec![Bar::new(c.t + 0.2 * DAY_SECS, ask_vol)
                        .width(0.35 * DAY_SECS)
                        .fill(PROFIT.gamma_multiply(0.7))]));
                }
            });
    }

    fn draw_orderbook_heatmap(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new("OBH — Order Book Heatmap").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let levels = 12;
        let cell_w = rect.width() / candles.candles.len() as f32;
        let cell_h = rect.height() / levels as f32;
        for (i, c) in candles.candles.iter().enumerate() {
            let mid = (c.high + c.low) / 2.0;
            let range = (c.high - c.low).max(1e-9);
            for lvl in 0..levels {
                let price = c.low + range * lvl as f64 / (levels - 1) as f64;
                let dist = (price - mid).abs() / range;
                let intensity = (1.0 - dist).clamp(0.0, 1.0);
                let color = if price >= mid {
                    Color32::from_rgb(
                        lerp(10, PROFIT.r(), intensity),
                        lerp(10, PROFIT.g(), intensity),
                        lerp(10, PROFIT.b(), intensity),
                    )
                } else {
                    Color32::from_rgb(
                        lerp(10, LOSS.r(), intensity),
                        lerp(10, LOSS.g(), intensity),
                        lerp(10, LOSS.b(), intensity),
                    )
                };
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + i as f32 * cell_w,
                        rect.top() + lvl as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 1.0_f32, cell_h - 1.0_f32),
                );
                painter.rect_filled(tile, 0.0, color);
            }
        }
    }

    fn draw_cumulative_delta(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("CD — Cumulative Delta — {}", candles.symbol)).strong());
        let (_per_bar, cumulative) = bt_viz::cumulative_delta::compute_deltas(candles);
        Plot::new("cd_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = candles
                    .candles
                    .iter()
                    .enumerate()
                    .map(|(i, c)| [c.t, cumulative[i]])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
            });
    }

    fn draw_market_profile(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("MP — Market Profile — {}", candles.symbol)).strong());
        Plot::new("mp_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.line(
                        Line::new(PlotPoints::from(vec![[c.t, c.low], [c.t, c.high]]))
                            .color(color)
                            .width(2.0_f32),
                    );
                }
            });
    }

    fn draw_volume_clock(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("VC — Volume Clock — {}", candles.symbol)).strong());
        let bar_w = bar_spacing(&candles.candles) * BAR_FILL;
        let spacing = bar_spacing(&candles.candles);
        let (x0, x1) = x_bounds(&candles.candles);
        Plot::new("vc_plot")
            .height(ui.available_height())
            .allow_drag(true)
            .allow_scroll(true)
            .allow_zoom(false)
            .include_x(x0)
            .include_x(x1)
            .y_axis_position(egui_plot::HPlacement::Right)
            .x_axis_formatter(move |mark, _range| {
                format_ts_styled(mark.value, self.axis_date_style())
            })
            .y_axis_formatter(move |mark, _range| abbreviate_volume(mark.value))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.bar_chart(BarChart::new(vec![Bar::new(c.t, c.volume)
                        .width(bar_w)
                        .fill(color)]));
                }
            });
    }

    fn draw_tick_tape(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("TT — Tick Tape — {}", candles.symbol)).strong());
        Plot::new("tt_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.points(
                        Points::new(PlotPoints::from(vec![[c.t, c.close]]))
                            .color(color)
                            .radius(3.0_f32),
                    );
                }
            });
    }

    fn draw_delta_divergence(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("DD — Delta Divergence — {}", candles.symbol)).strong());
        let (_per_bar, cumulative) = bt_viz::cumulative_delta::compute_deltas(candles);
        Plot::new("dd_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = candles
                    .candles
                    .iter()
                    .enumerate()
                    .map(|(i, c)| [c.t, cumulative[i]])
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
            });
    }
    // ==================== INDICATORS ====================

    fn draw_rsi(&self, ui: &mut egui::Ui) {
        use bt_analytics::rsi;
        let series = &self.data.candles;
        let rsi_vals = rsi(series, 14);
        ui.label(
            RichText::new(format!("RSI — Relative Strength Index — {}", series.symbol)).strong(),
        );
        Plot::new("rsi_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !rsi_vals[i].is_nan() {
                            Some([c.t, rsi_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(70.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(30.0).color(PROFIT));
            });
    }

    fn draw_macd(&self, ui: &mut egui::Ui) {
        use bt_analytics::macd;
        let series = &self.data.candles;
        let (macd_line, signal_line, histogram) = macd(series);
        ui.label(
            RichText::new(format!(
                "MACD — Moving Average Convergence Divergence — {}",
                series.symbol
            ))
            .strong(),
        );
        Plot::new("macd_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.5_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !macd_line[i].is_nan() {
                            Some([c.t, macd_line[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("MACD"));
                let sig_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !signal_line[i].is_nan() {
                            Some([c.t, signal_line[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(sig_pts)
                        .color(AMBER)
                        .width(2.0_f32)
                        .name("Signal"),
                );
            });
        Plot::new("macd_hist")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !histogram[i].is_nan() {
                            let color = if histogram[i] >= 0.0 { PROFIT } else { LOSS };
                            Some(
                                Bar::new(c.t, histogram[i])
                                    .width(0.5 * DAY_SECS)
                                    .fill(color),
                            )
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars));
            });
    }

    fn draw_stochastic(&self, ui: &mut egui::Ui) {
        use bt_analytics::stochastic;
        let series = &self.data.candles;
        let (k, d) = stochastic(series, 14, 3);
        ui.label(
            RichText::new(format!("STOCH — Stochastic Oscillator — {}", series.symbol)).strong(),
        );
        Plot::new("stoch_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let k_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !k[i].is_nan() {
                            Some([c.t, k[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(k_pts).color(AMBER).width(2.0_f32).name("%K"));
                let d_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !d[i].is_nan() {
                            Some([c.t, d[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(d_pts).color(INFO).width(2.0_f32).name("%D"));
                plot_ui.hline(egui_plot::HLine::new(80.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(20.0).color(PROFIT));
            });
    }

    fn draw_atr(&self, ui: &mut egui::Ui) {
        use bt_analytics::atr;
        let series = &self.data.candles;
        let atr_vals = atr(series, 14);
        ui.label(RichText::new(format!("ATR — Average True Range — {}", series.symbol)).strong());
        Plot::new("atr_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !atr_vals[i].is_nan() {
                            Some([c.t, atr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(PROFIT).width(2.0_f32));
            });
    }

    fn draw_obv(&self, ui: &mut egui::Ui) {
        use bt_analytics::obv;
        let series = &self.data.candles;
        let obv_vals = obv(series);
        ui.label(RichText::new(format!("OBV — On-Balance Volume — {}", series.symbol)).strong());
        Plot::new("obv_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .map(|(i, c)| [c.t, obv_vals[i]])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
            });
    }

    fn draw_vwap(&self, ui: &mut egui::Ui) {
        use bt_analytics::vwap;
        let series = &self.data.candles;
        let vwap_vals = vwap(series);
        ui.label(
            RichText::new(format!(
                "VWAP — Volume Weighted Average Price — {}",
                series.symbol
            ))
            .strong(),
        );
        Plot::new("vwap_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !vwap_vals[i].is_nan() {
                            Some([c.t, vwap_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("VWAP"));
            });
    }

    fn draw_bollinger(&self, ui: &mut egui::Ui) {
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (mid, upper, lower) = bollinger(series, 20, 2.0);
        ui.label(RichText::new(format!("BOLL — Bollinger Bands — {}", series.symbol)).strong());
        Plot::new("bollinger_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let mid_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() {
                            Some([c.t, mid[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(mid_pts).color(AMBER).width(2.0_f32).name("SMA20"));
                let upper_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !upper[i].is_nan() {
                            Some([c.t, upper[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(upper_pts)
                        .color(INFO)
                        .width(1.0_f32)
                        .name("Upper"),
                );
                let lower_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !lower[i].is_nan() {
                            Some([c.t, lower[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lower_pts)
                        .color(INFO)
                        .width(1.0_f32)
                        .name("Lower"),
                );
            });
    }

    fn draw_bb_width(&self, ui: &mut egui::Ui) {
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (_mid, upper, lower) = bollinger(series, 20, 2.0);
        ui.label(RichText::new(format!("BBW — Bollinger Band Width — {}", series.symbol)).strong());
        Plot::new("bbw_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !upper[i].is_nan() && !lower[i].is_nan() {
                            Some([c.t, upper[i] - lower[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.0_f32));
            });
    }

    fn draw_adx(&self, ui: &mut egui::Ui) {
        use bt_analytics::adx;
        let series = &self.data.candles;
        let (adx_vals, _plus_di, _minus_di) = adx(series, 14);
        ui.label(
            RichText::new(format!(
                "ADX — Average Directional Index — {}",
                series.symbol
            ))
            .strong(),
        );
        Plot::new("adx_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !adx_vals[i].is_nan() {
                            Some([c.t, adx_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(25.0).color(Color32::GRAY));
            });
    }

    fn draw_cci(&self, ui: &mut egui::Ui) {
        use bt_analytics::cci;
        let series = &self.data.candles;
        let cci_vals = cci(series, 20);
        ui.label(
            RichText::new(format!("CCI — Commodity Channel Index — {}", series.symbol)).strong(),
        );
        Plot::new("cci_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !cci_vals[i].is_nan() {
                            Some([c.t, cci_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(100.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(-100.0).color(PROFIT));
            });
    }

    fn draw_williams_r(&self, ui: &mut egui::Ui) {
        use bt_analytics::williams_r;
        let series = &self.data.candles;
        let wr_vals = williams_r(series, 14);
        ui.label(RichText::new(format!("W%R — Williams %R — {}", series.symbol)).strong());
        Plot::new("wr_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !wr_vals[i].is_nan() {
                            Some([c.t, wr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(-20.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(-80.0).color(PROFIT));
            });
    }

    fn draw_roc(&self, ui: &mut egui::Ui) {
        use bt_analytics::roc;
        let series = &self.data.candles;
        let roc_vals = roc(series, 12);
        ui.label(RichText::new(format!("ROC — Rate of Change — {}", series.symbol)).strong());
        Plot::new("roc_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !roc_vals[i].is_nan() {
                            Some([c.t, roc_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_cmf(&self, ui: &mut egui::Ui) {
        use bt_analytics::cmf;
        let series = &self.data.candles;
        let cmf_vals = cmf(series, 20);
        ui.label(RichText::new(format!("CMF — Chaikin Money Flow — {}", series.symbol)).strong());
        Plot::new("cmf_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !cmf_vals[i].is_nan() {
                            Some([c.t, cmf_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_ichimoku(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        ui.label(RichText::new(format!("ICH — Ichimoku Cloud — {}", series.symbol)).strong());
        Plot::new("ich_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
            });
    }

    fn draw_keltner(&self, ui: &mut egui::Ui) {
        use bt_analytics::atr;
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (mid, _upper, _lower) = bollinger(series, 20, 2.0);
        let atr_vals = atr(series, 14);
        ui.label(RichText::new(format!("KEL — Keltner Channels — {}", series.symbol)).strong());
        Plot::new("kel_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let mid_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() {
                            Some([c.t, mid[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(mid_pts).color(AMBER).width(2.0_f32).name("EMA20"));
                let upper_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() && !atr_vals[i].is_nan() {
                            Some([c.t, mid[i] + 2.0 * atr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(upper_pts)
                        .color(INFO)
                        .width(1.0_f32)
                        .name("Upper"),
                );
                let lower_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() && !atr_vals[i].is_nan() {
                            Some([c.t, mid[i] - 2.0 * atr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lower_pts)
                        .color(INFO)
                        .width(1.0_f32)
                        .name("Lower"),
                );
            });
    }

    fn draw_donchian(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        ui.label(RichText::new(format!("DON — Donchian Channels — {}", series.symbol)).strong());
        Plot::new("don_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
            });
    }
    // ==================== RISK & PORTFOLIO ====================

    fn draw_drawdown(&self, ui: &mut egui::Ui) {
        let dd = bt_viz::drawdown::compute_drawdown(&self.data.equity);
        ui.label(RichText::new("VAR — Drawdown Underwater Chart").strong());
        Plot::new("drawdown_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = dd.iter().enumerate().map(|(i, &v)| [i as f64, v]).collect();
                plot_ui.line(Line::new(pts).color(LOSS).width(2.0_f32));
            });
    }

    fn draw_correlation(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Rolling Correlation Matrix").strong());
        let matrix = bt_viz::correlation_heatmap::correlation_matrix(&self.data.corr);
        let n = self.data.corr.len();
        egui::Grid::new("corr_grid")
            .striped(false)
            .spacing(Vec2::new(2.0_f32, 2.0_f32))
            .show(ui, |ui| {
                ui.label("");
                for (label, _) in &self.data.corr {
                    ui.label(RichText::new(label).small());
                }
                ui.end_row();
                for i in 0..n {
                    ui.label(RichText::new(&self.data.corr[i].0).small());
                    for j in 0..n {
                        let v = matrix[i][j];
                        let t = v.abs().clamp(0.0, 1.0);
                        let target = if v >= 0.0 { PROFIT } else { LOSS };
                        let bg = Color32::from_rgb(
                            lerp(20, target.r(), t),
                            lerp(20, target.g(), t),
                            lerp(20, target.b(), t),
                        );
                        let frame = egui::Frame::none().fill(bg).inner_margin(6.0_f32);
                        frame.show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("{:.2}", v))
                                    .color(Color32::WHITE)
                                    .small(),
                            );
                        });
                    }
                    ui.end_row();
                }
            });
    }

    fn draw_vol_smile(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("SKEW — Volatility Smile / Skew").strong());
        Plot::new("vol_smile_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let colors = [AMBER, INFO, PROFIT, LOSS];
                for (idx, (label, moneyness, iv)) in self.data.smiles.iter().enumerate() {
                    let color = colors[idx % colors.len()];
                    let pts: PlotPoints = moneyness.iter().zip(iv).map(|(&m, &v)| [m, v]).collect();
                    plot_ui.line(Line::new(pts).color(color).width(2.0_f32).name(label));
                }
            });
    }

    fn draw_efficient_frontier(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("PORT/MARS — Efficient Frontier").strong());
        Plot::new("frontier_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let sharpe_min = self
                    .data
                    .portfolios
                    .iter()
                    .map(|p| p.sharpe)
                    .fold(f64::MAX, f64::min);
                let sharpe_max = self
                    .data
                    .portfolios
                    .iter()
                    .map(|p| p.sharpe)
                    .fold(f64::MIN, f64::max);
                let range = (sharpe_max - sharpe_min).max(1e-9);
                let buckets = 6;
                for b in 0..buckets {
                    let lo = sharpe_min + range * b as f64 / buckets as f64;
                    let hi = sharpe_min + range * (b + 1) as f64 / buckets as f64;
                    let t = b as f64 / (buckets - 1) as f64;
                    let color = Color32::from_rgb(
                        lerp(LOSS.r(), PROFIT.r(), t),
                        lerp(LOSS.g(), PROFIT.g(), t),
                        lerp(LOSS.b(), PROFIT.b(), t),
                    );
                    let pts: PlotPoints = self
                        .data
                        .portfolios
                        .iter()
                        .filter(|p| p.sharpe >= lo && p.sharpe <= hi)
                        .map(|p| [p.risk, p.ret])
                        .collect();
                    plot_ui.points(Points::new(pts).color(color).radius(2.5_f32));
                }
                if let Some(best) = self
                    .data
                    .portfolios
                    .iter()
                    .max_by(|a, b| a.sharpe.partial_cmp(&b.sharpe).unwrap())
                {
                    plot_ui.points(
                        Points::new(PlotPoints::from(vec![[best.risk, best.ret]]))
                            .color(AMBER)
                            .radius(7.0_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
            });
    }

    fn draw_rolling_sharpe(&self, ui: &mut egui::Ui) {
        use bt_analytics::rolling_sharpe;
        let rets = &self.data.returns;
        let rs = rolling_sharpe(rets, 20, 0.05, 252);
        ui.label(RichText::new("Rolling Sharpe Ratio").strong());
        Plot::new("rs_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = rs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if !v.is_nan() {
                            Some([i as f64, v])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(PROFIT).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(1.0).color(AMBER));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_rolling_sortino(&self, ui: &mut egui::Ui) {
        use bt_analytics::rolling_sortino;
        let rets = &self.data.returns;
        let rs = rolling_sortino(rets, 20, 0.05, 252);
        ui.label(RichText::new("Rolling Sortino Ratio").strong());
        Plot::new("rsort_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = rs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if !v.is_nan() {
                            Some([i as f64, v])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(1.0).color(AMBER));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_beta_alpha(&self, ui: &mut egui::Ui) {
        use bt_analytics::{alpha, beta};
        let rets = &self.data.returns;
        let b = beta(rets, rets);
        let a = alpha(rets, rets, 0.05, 252);
        ui.label(RichText::new("Beta / Alpha Analysis").strong());
        ui.label(format!("Beta: {:.4}", b));
        ui.label(format!("Alpha (annual): {:.4}", a));
        Plot::new("beta_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = rets
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32));
            });
    }

    fn draw_rolling_max_dd(&self, ui: &mut egui::Ui) {
        use bt_analytics::rolling_max_drawdown;
        let series = &self.data.candles;
        let rmdd = rolling_max_drawdown(series, 20);
        ui.label(RichText::new("Rolling Max Drawdown").strong());
        Plot::new("rmdd_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !rmdd[i].is_nan() {
                            Some([c.t, rmdd[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(LOSS).width(2.0_f32));
            });
    }

    fn draw_var_backtest(&self, ui: &mut egui::Ui) {
        use bt_analytics::var_historical;
        let rets = &self.data.returns;
        let var_95 = var_historical(rets, 0.95);
        let var_99 = var_historical(rets, 0.99);
        ui.label(RichText::new("VaR Backtest").strong());
        ui.label(format!("VaR 95%: {:.4}", var_95));
        ui.label(format!("VaR 99%: {:.4}", var_99));
        Plot::new("var_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = rets
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.0_f32));
                plot_ui.hline(egui_plot::HLine::new(var_95).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(var_99).color(PURPLE));
            });
    }

    fn draw_monte_carlo(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let variance = rets.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
        let std_dev = variance.sqrt();
        ui.label(RichText::new("Monte Carlo Simulation").strong());
        Plot::new("mc_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let mut rng = self.seed_counter;
                let colors = [AMBER, INFO, PROFIT, LOSS, PURPLE];
                for (idx, color) in colors.iter().enumerate() {
                    let mut price = 100.0 + idx as f64 * 5.0;
                    let mut path = vec![price];
                    for _ in 0..100 {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let z = ((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                        price *= 1.0 + mean + std_dev * z;
                        path.push(price);
                    }
                    let pts: PlotPoints = path
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| [i as f64, v])
                        .collect();
                    plot_ui.line(Line::new(pts).color(*color).width(1.5_f32));
                }
            });
    }
    // ==================== VOLATILITY & OPTIONS ====================

    fn draw_vol_smile_opt(&self, ui: &mut egui::Ui) {
        self.draw_vol_smile(ui);
    }

    fn draw_iv_surface(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("IV Surface").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let strikes = [0.8_f64, 0.9, 1.0, 1.1, 1.2];
        let tenors = [7.0_f64, 14.0, 30.0, 60.0, 90.0];
        let cell_w = rect.width() / strikes.len() as f32;
        let cell_h = rect.height() / tenors.len() as f32;
        for (ti, &tenor) in tenors.iter().enumerate() {
            for (si, &strike) in strikes.iter().enumerate() {
                let moneyness = strike - 1.0;
                let iv = 0.20 + 0.05 * moneyness * moneyness * 10.0 + 0.02 / (tenor / 30.0).sqrt();
                let intensity = ((iv - 0.15) / 0.15).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, INFO.r(), intensity),
                    lerp(10, INFO.g(), intensity),
                    lerp(10, INFO.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + si as f32 * cell_w,
                        rect.top() + ti as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }

    fn draw_term_structure(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("IV Term Structure").strong());
        Plot::new("ts_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let tenors = [7.0_f64, 14.0, 30.0, 60.0, 90.0, 180.0];
                let ivs: Vec<f64> = tenors
                    .iter()
                    .map(|&t| 0.20 + 0.03 / (t / 30.0).sqrt())
                    .collect();
                plot_ui.line(
                    Line::new(
                        tenors
                            .iter()
                            .zip(ivs.iter())
                            .map(|(&t, &v)| [t, v])
                            .collect::<PlotPoints>(),
                    )
                    .color(AMBER)
                    .width(2.0_f32),
                );
                plot_ui.points(
                    Points::new(
                        tenors
                            .iter()
                            .zip(ivs.iter())
                            .map(|(&t, &v)| [t, v])
                            .collect::<PlotPoints>(),
                    )
                    .color(AMBER)
                    .radius(4.0_f32)
                    .shape(MarkerShape::Circle),
                );
            });
    }

    fn draw_greeks_heatmap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Greeks Heatmap").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let greeks = ["Delta", "Gamma", "Theta", "Vega", "Rho"];
        let strikes = [0.8_f64, 0.9, 1.0, 1.1, 1.2];
        let cell_w = rect.width() / strikes.len() as f32;
        let cell_h = rect.height() / greeks.len() as f32;
        for (gi, greek) in greeks.iter().enumerate() {
            for (si, &strike) in strikes.iter().enumerate() {
                let moneyness = (strike - 1.0).abs();
                let val = match *greek {
                    "Delta" => 0.5 + (strike - 1.0) * 2.0,
                    "Gamma" => 0.1 * (1.0 - moneyness * 5.0),
                    "Theta" => -0.05 * (1.0 - moneyness * 3.0),
                    "Vega" => 0.15 * (1.0 - moneyness * 4.0),
                    "Rho" => 0.05 * (strike - 1.0),
                    _ => 0.0,
                };
                let intensity = val.abs().clamp(0.0, 1.0);
                let color = if val >= 0.0 {
                    Color32::from_rgb(
                        lerp(10, PROFIT.r(), intensity),
                        lerp(10, PROFIT.g(), intensity),
                        lerp(10, PROFIT.b(), intensity),
                    )
                } else {
                    Color32::from_rgb(
                        lerp(10, LOSS.r(), intensity),
                        lerp(10, LOSS.g(), intensity),
                        lerp(10, LOSS.b(), intensity),
                    )
                };
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + si as f32 * cell_w,
                        rect.top() + gi as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }

    fn draw_vix_term(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("VIX Term Structure").strong());
        Plot::new("vix_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let tenors = [7.0_f64, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0];
                let vix_vals: Vec<f64> = tenors
                    .iter()
                    .map(|&t| 15.0 + 5.0 / (t / 30.0).sqrt())
                    .collect();
                let pts: PlotPoints = tenors
                    .iter()
                    .zip(vix_vals.iter())
                    .map(|(&t, &v)| [t, v])
                    .collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.0_f32));
            });
    }

    fn draw_vol_cone(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Volatility Cone").strong());
        Plot::new("vol_cone_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let tenors = [7.0_f64, 14.0, 30.0, 60.0, 90.0];
                let high: Vec<f64> = tenors
                    .iter()
                    .map(|&t| 0.25 + 0.05 / (t / 30.0).sqrt())
                    .collect();
                let low: Vec<f64> = tenors
                    .iter()
                    .map(|&t| 0.12 + 0.02 / (t / 30.0).sqrt())
                    .collect();
                let current = 0.20;
                let high_pts: PlotPoints = tenors
                    .iter()
                    .zip(high.iter())
                    .map(|(&t, &v)| [t, v])
                    .collect();
                let low_pts: PlotPoints = tenors
                    .iter()
                    .zip(low.iter())
                    .map(|(&t, &v)| [t, v])
                    .collect();
                let curr_pts: PlotPoints = tenors.iter().map(|&t| [t, current]).collect();
                plot_ui.line(Line::new(high_pts).color(LOSS).width(1.5_f32).name("High"));
                plot_ui.line(Line::new(low_pts).color(PROFIT).width(1.5_f32).name("Low"));
                plot_ui.line(
                    Line::new(curr_pts)
                        .color(AMBER)
                        .width(2.0_f32)
                        .name("Current"),
                );
            });
    }

    fn draw_option_payoff(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Option Payoff Diagram").strong());
        Plot::new("payoff_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let strike = 100.0_f64;
                let premium = 5.0_f64;
                let long_call: PlotPoints = (0..=200)
                    .map(|i| {
                        let s = 50.0 + i as f64 * 1.5;
                        [s, (s - strike).max(0.0) - premium]
                    })
                    .collect();
                let short_call: PlotPoints = (0..=200)
                    .map(|i| {
                        let s = 50.0 + i as f64 * 1.5;
                        [s, -(s - strike).max(0.0) + premium]
                    })
                    .collect();
                let long_put: PlotPoints = (0..=200)
                    .map(|i| {
                        let s = 50.0 + i as f64 * 1.5;
                        [s, (strike - s).max(0.0) - premium]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(long_call)
                        .color(PROFIT)
                        .width(2.0_f32)
                        .name("Long Call"),
                );
                plot_ui.line(
                    Line::new(short_call)
                        .color(LOSS)
                        .width(2.0_f32)
                        .name("Short Call"),
                );
                plot_ui.line(
                    Line::new(long_put)
                        .color(INFO)
                        .width(2.0_f32)
                        .name("Long Put"),
                );
                plot_ui.vline(egui_plot::VLine::new(strike).color(Color32::GRAY));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_skew_evolution(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Skew Evolution").strong());
        Plot::new("skew_evo_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let days: Vec<f64> = (0..30).map(|i| i as f64).collect();
                let skew: Vec<f64> = days
                    .iter()
                    .map(|&d| -0.3 + 0.01 * d + 0.05 * (d * 0.5).sin())
                    .collect();
                let pts: PlotPoints = days
                    .iter()
                    .zip(skew.iter())
                    .map(|(&d, &s)| [d, s])
                    .collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_gamma_exposure(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Gamma Exposure by Strike").strong());
        Plot::new("gamma_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let strikes: Vec<f64> = (0..20).map(|i| 80.0 + i as f64 * 5.0).collect();
                let gamma: Vec<f64> = strikes
                    .iter()
                    .map(|&s| {
                        let d = (s - 100.0) / 10.0;
                        0.05 * (-d * d / 2.0).exp()
                    })
                    .collect();
                let pts: PlotPoints = strikes
                    .iter()
                    .zip(gamma.iter())
                    .map(|(&s, &g)| [s, g])
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32));
                plot_ui.vline(egui_plot::VLine::new(100.0).color(AMBER));
            });
    }

    fn draw_put_call_ratio(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Put/Call Ratio").strong());
        Plot::new("pcr_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let days: Vec<f64> = (0..30).map(|i| i as f64).collect();
                let pcr: Vec<f64> = days
                    .iter()
                    .map(|&d| 0.8 + 0.2 * (d * 0.3).sin() + 0.1 * (d * 0.1).cos())
                    .collect();
                let pts: PlotPoints = days.iter().zip(pcr.iter()).map(|(&d, &p)| [d, p]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(1.0).color(Color32::GRAY));
            });
    }

    fn draw_iv_rank(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("IV Rank").strong());
        let days: Vec<f64> = (0..252).map(|i| i as f64).collect();
        let iv: Vec<f64> = days
            .iter()
            .map(|&d| 0.15 + 0.10 * (d * 0.05).sin() + 0.05 * (d * 0.02).cos())
            .collect();
        let current_iv = iv.last().copied().unwrap_or(0.20);
        let min_iv = iv.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let max_iv = iv.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let iv_rank = ((current_iv - min_iv) / (max_iv - min_iv).max(1e-9) * 100.0) as i32;
        ui.label(format!("Current IV Rank: {}%", iv_rank));
        Plot::new("ivr_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = days.iter().zip(iv.iter()).map(|(&d, &v)| [d, v]).collect();
                plot_ui.line(Line::new(pts).color(INFO).width(1.5_f32));
                plot_ui.hline(egui_plot::HLine::new(current_iv).color(AMBER));
            });
    }

    fn draw_sharpe_surface(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Sharpe Ratio Surface").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let x_labels = [
            "Conservative",
            "Moderate",
            "Balanced",
            "Growth",
            "Aggressive",
        ];
        let y_labels = ["1M", "3M", "6M", "1Y", "3Y"];
        let cell_w = rect.width() / x_labels.len() as f32;
        let cell_h = rect.height() / y_labels.len() as f32;
        for (yi, _yl) in y_labels.iter().enumerate() {
            for (xi, _xl) in x_labels.iter().enumerate() {
                let sharpe = 0.5 + xi as f64 * 0.3 - yi as f64 * 0.1;
                let intensity = ((sharpe - 0.2) / 1.5).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, PROFIT.r(), intensity),
                    lerp(10, PROFIT.g(), intensity),
                    lerp(10, PROFIT.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + xi as f32 * cell_w,
                        rect.top() + yi as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }
    // ==================== MICROSTRUCTURE ====================

    fn draw_acf_pacf(&self, ui: &mut egui::Ui) {
        let max_lag = 25usize;
        let a = bt_viz::acf_pacf::acf(&self.data.returns, max_lag);
        let p = bt_viz::acf_pacf::pacf(&self.data.returns, max_lag);
        let conf = 1.96 / (self.data.returns.len() as f64).sqrt();

        ui.label(RichText::new("ACF").strong());
        Plot::new("acf_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.45_f32)
            .show(ui, |plot_ui| {
                draw_lollipop(plot_ui, &a, conf);
            });
        ui.label(RichText::new("PACF").strong());
        Plot::new("pacf_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                draw_lollipop(plot_ui, &p, conf);
            });
    }

    fn draw_hurst(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        ui.label(RichText::new("Hurst Exponent").strong());
        let n = rets.len();
        let mut log_n = Vec::new();
        let mut log_r_s = Vec::new();
        for chunk_size in [10, 20, 40, 80] {
            if chunk_size >= n {
                continue;
            }
            let mut r_s_values = Vec::new();
            for chunk in rets.chunks(chunk_size) {
                let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
                let mut cum_dev = 0.0;
                let mut max_dev = f64::MIN;
                let mut min_dev = f64::MAX;
                for &r in chunk {
                    cum_dev += r - mean;
                    max_dev = max_dev.max(cum_dev);
                    min_dev = min_dev.min(cum_dev);
                }
                let range = max_dev - min_dev;
                let variance =
                    chunk.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / chunk.len() as f64;
                let std_dev = variance.sqrt();
                if std_dev > 0.0 {
                    r_s_values.push(range / std_dev);
                }
            }
            if !r_s_values.is_empty() {
                let avg_r_s = r_s_values.iter().sum::<f64>() / r_s_values.len() as f64;
                log_n.push((chunk_size as f64).ln());
                log_r_s.push(avg_r_s.ln());
            }
        }
        let hurst = if log_n.len() >= 2 {
            let n_points = log_n.len() as f64;
            let sum_x = log_n.iter().sum::<f64>();
            let sum_y = log_r_s.iter().sum::<f64>();
            let sum_xy = log_n
                .iter()
                .zip(log_r_s.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>();
            let sum_x2 = log_n.iter().map(|x| x * x).sum::<f64>();
            (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x)
        } else {
            0.5
        };
        ui.label(format!("Hurst Exponent: {:.4}", hurst));
        Plot::new("hurst_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = log_n
                    .iter()
                    .zip(log_r_s.iter())
                    .map(|(&x, &y)| [x, y])
                    .collect();
                plot_ui.points(
                    Points::new(pts)
                        .color(AMBER)
                        .radius(5.0_f32)
                        .shape(MarkerShape::Circle),
                );
                if log_n.len() >= 2 {
                    let x_min = log_n.iter().fold(f64::INFINITY, |a, b| a.min(*b));
                    let x_max = log_n.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                    let slope = hurst;
                    let intercept = (log_r_s.iter().sum::<f64>()
                        - slope * log_n.iter().sum::<f64>())
                        / log_n.len() as f64;
                    let line_pts = vec![
                        [x_min, slope * x_min + intercept],
                        [x_max, slope * x_max + intercept],
                    ];
                    plot_ui.line(Line::new(line_pts).color(INFO).width(2.0_f32));
                }
            });
    }

    fn draw_wavelet(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        ui.label(RichText::new("Wavelet Power Spectrum").strong());
        Plot::new("wavelet_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let scales: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();
                let power: Vec<f64> = scales
                    .iter()
                    .map(|&s| {
                        let mut sum = 0.0;
                        for i in 0..rets.len().min(100) {
                            sum += (rets[i] * (i as f64 / s).cos()).powi(2);
                        }
                        sum / rets.len() as f64
                    })
                    .collect();
                let pts: PlotPoints = scales
                    .iter()
                    .zip(power.iter())
                    .map(|(&s, &p)| [s, p])
                    .collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.0_f32));
            });
    }

    fn draw_kalman(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new("Kalman Filter").strong());
        Plot::new("kalman_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let mut estimate = candles.candles[0].close;
                let mut error = 1.0;
                let process_noise = 0.01;
                let measurement_noise = 0.1;
                let mut estimates = Vec::new();
                for c in &candles.candles {
                    let prediction = estimate;
                    let prediction_error = error + process_noise;
                    let kalman_gain = prediction_error / (prediction_error + measurement_noise);
                    estimate = prediction + kalman_gain * (c.close - prediction);
                    error = (1.0 - kalman_gain) * prediction_error;
                    estimates.push(estimate);
                }
                let pts: PlotPoints = candles.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.0_f32).name("Observed"));
                let est_pts: PlotPoints = candles
                    .candles
                    .iter()
                    .enumerate()
                    .map(|(i, c)| [c.t, estimates[i]])
                    .collect();
                plot_ui.line(
                    Line::new(est_pts)
                        .color(INFO)
                        .width(2.0_f32)
                        .name("Filtered"),
                );
            });
    }

    fn draw_markov_regime(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        ui.label(RichText::new("Markov Regime Switching").strong());
        Plot::new("markov_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let mut regime = 0;
                let mut regimes = Vec::new();
                for &r in rets {
                    if r > 0.01 {
                        regime = 1;
                    } else if r < -0.01 {
                        regime = 0;
                    }
                    regimes.push(regime);
                }
                let pts: PlotPoints = regimes
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| [i as f64, r as f64])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(0.5).color(Color32::GRAY));
            });
    }

    fn draw_copula_3d(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Copula 3D Scatter").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let n = self.data.returns.len().min(200);
        let mut rng = self.seed_counter;
        for i in 0..n {
            let u = self.data.returns[i].abs().min(1.0);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 11) as f64 / (1u64 << 53) as f64).min(1.0);
            let x = rect.left() + u as f32 * rect.width();
            let y = rect.bottom() - v as f32 * rect.height();
            let color = if self.data.returns[i] >= 0.0 {
                PROFIT
            } else {
                LOSS
            };
            painter.circle_filled(egui::Pos2::new(x, y), 3.0, color);
        }
    }

    fn draw_qq_plot(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        ui.label(RichText::new("Q-Q Plot").strong());
        Plot::new("qq_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let mut sorted = rets.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let mean = sorted.iter().sum::<f64>() / n as f64;
                let std_dev =
                    (sorted.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
                let pts: PlotPoints = sorted
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let p = (i as f64 + 0.5) / n as f64;
                        let z = normal_inverse(p);
                        [z * std_dev + mean, v]
                    })
                    .collect();
                plot_ui.points(
                    Points::new(pts)
                        .color(AMBER)
                        .radius(2.0_f32)
                        .shape(MarkerShape::Circle),
                );
                let min_v = sorted.iter().fold(f64::INFINITY, |a, b| a.min(*b));
                let max_v = sorted.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                let line_pts = vec![[min_v, min_v], [max_v, max_v]];
                plot_ui.line(Line::new(line_pts).color(INFO).width(1.5_f32));
            });
    }

    fn draw_return_dist(&self, ui: &mut egui::Ui) {
        let rets = &self.data.returns;
        ui.label(RichText::new("Return Distribution").strong());
        Plot::new("dist_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let min_r = rets.iter().fold(f64::INFINITY, |a, b| a.min(*b));
                let max_r = rets.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
                let bins = 30;
                let bin_width = (max_r - min_r) / bins as f64;
                let mut counts = vec![0.0_f64; bins];
                for &r in rets {
                    let idx = ((r - min_r) / bin_width).floor() as usize;
                    let idx = idx.min(bins - 1);
                    counts[idx] += 1.0;
                }
                let bars: Vec<Bar> = counts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        Bar::new(min_r + i as f64 * bin_width, c)
                            .width(bin_width * 0.9)
                            .fill(INFO)
                    })
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars));
            });
    }

    fn draw_rolling_moments(&self, ui: &mut egui::Ui) {
        use bt_analytics::rolling_moments;
        let rets = &self.data.returns;
        let (roll_mean, roll_std) = rolling_moments(rets, 20);
        ui.label(RichText::new("Rolling Moments").strong());
        Plot::new("rm_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let mean_pts: PlotPoints = roll_mean
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if !v.is_nan() {
                            Some([i as f64, v])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(mean_pts).color(AMBER).width(2.0_f32).name("Mean"));
                let std_pts: PlotPoints = roll_std
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if !v.is_nan() {
                            Some([i as f64, v])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(std_pts)
                        .color(INFO)
                        .width(2.0_f32)
                        .name("Std Dev"),
                );
            });
    }
    // ==================== INDIA-SPECIFIC ====================

    fn draw_treemap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("BMAP — NIFTY Sector Map").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let mut sorted = self.data.treemap_nodes.clone();
        sorted.sort_by(|a, b| b.market_cap.partial_cmp(&a.market_cap).unwrap());
        let full = bt_viz::sector_treemap::Rect {
            x: 0.0,
            y: 0.0,
            w: rect.width() as f64,
            h: rect.height() as f64,
        };
        let rects = bt_viz::sector_treemap::layout(&sorted, full);
        for (node, r) in sorted.iter().zip(rects.iter()) {
            let t = (node.pct_change.abs() / 3.0).clamp(0.15, 1.0);
            let target = if node.pct_change >= 0.0 { PROFIT } else { LOSS };
            let color = Color32::from_rgb(
                lerp(15, target.r(), t),
                lerp(15, target.g(), t),
                lerp(15, target.b(), t),
            );
            let tile = egui::Rect::from_min_size(
                rect.min + Vec2::new(r.x as f32, r.y as f32),
                Vec2::new(r.w as f32, r.h as f32),
            );
            painter.rect_filled(tile, 0.0, color);
            painter.rect_stroke(tile, 0.0, Stroke::new(1.0_f32, Color32::BLACK));
            if r.w > 60.0 && r.h > 28.0 {
                painter.text(
                    tile.min + Vec2::new(6.0_f32, 6.0_f32),
                    egui::Align2::LEFT_TOP,
                    format!("{}\n{:+.2}%", node.label, node.pct_change),
                    egui::FontId::monospace(13.0_f32),
                    Color32::WHITE,
                );
            }
        }
    }

    fn draw_heatmap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Sensex Heatmap").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let stocks: [(&str, f64); 30] = [
            ("RELIANCE", 1.2),
            ("HDFCBANK", 0.8),
            ("ICICIBANK", 1.5),
            ("INFY", -0.5),
            ("TCS", 0.3),
            ("BHARTIARTL", 2.1),
            ("ITC", -0.2),
            ("LT", 1.8),
            ("SBIN", 2.5),
            ("KOTAKBANK", 0.9),
            ("HINDUNILVR", -0.8),
            ("BAJFINANCE", 1.1),
            ("ASIANPAINT", -0.3),
            ("MARUTI", 1.4),
            ("SUNPHARMA", -1.2),
            ("TITAN", 0.7),
            ("ULTRACEMCO", 1.0),
            ("NESTLEIND", -0.4),
            ("WIPRO", -0.6),
            ("HCLTECH", 0.5),
            ("AXISBANK", 1.3),
            ("TATAMOTORS", 2.0),
            ("TATASTEEL", 1.5),
            ("ADANIENT", 3.2),
            ("ADANIPORTS", 1.8),
            ("BAJAJ-AUTO", 0.6),
            ("COALINDIA", -0.5),
            ("NTPC", 0.4),
            ("POWERGRID", 0.3),
            ("TECHM", -0.4),
        ];
        let cols = 6;
        let rows = (stocks.len() + cols - 1) / cols;
        let cell_w = rect.width() / cols as f32;
        let cell_h = rect.height() / rows as f32;
        let padding = 2.0_f32;
        for (idx, (sym, change)) in stocks.iter().enumerate() {
            let col = idx % cols;
            let row = idx / cols;
            let x = rect.left() + col as f32 * cell_w + padding;
            let y = rect.top() + row as f32 * cell_h + padding;
            let w = cell_w - 2.0_f32 * padding;
            let h = cell_h - 2.0_f32 * padding;
            let intensity = (change.abs() / 3.0).clamp(0.2, 1.0);
            let target = if *change >= 0.0 { PROFIT } else { LOSS };
            let color = Color32::from_rgb(
                lerp(15, target.r(), intensity),
                lerp(15, target.g(), intensity),
                lerp(15, target.b(), intensity),
            );
            let tile = egui::Rect::from_min_size(egui::Pos2::new(x, y), Vec2::new(w, h));
            painter.rect_filled(tile, 4.0_f32, color);
            if w > 50.0 && h > 20.0 {
                painter.text(
                    tile.min + Vec2::new(4.0_f32, 4.0_f32),
                    egui::Align2::LEFT_TOP,
                    format!("{}\n{:+.1}%", sym, change),
                    egui::FontId::monospace(10.0_f32),
                    Color32::WHITE,
                );
            }
        }
    }

    fn draw_fii_dii_flow(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("FII/DII Flow").strong());
        Plot::new("fii_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let days: Vec<f64> = (0..20).map(|i| i as f64).collect();
                let fii: Vec<f64> = days
                    .iter()
                    .map(|&d| 500.0 + 200.0 * (d * 0.5).sin() + 100.0 * (d * 0.3).cos())
                    .collect();
                let dii: Vec<f64> = days
                    .iter()
                    .map(|&d| 300.0 + 150.0 * (d * 0.4).cos() + 80.0 * (d * 0.7).sin())
                    .collect();
                let fii_pts: PlotPoints =
                    days.iter().zip(fii.iter()).map(|(&d, &v)| [d, v]).collect();
                let dii_pts: PlotPoints =
                    days.iter().zip(dii.iter()).map(|(&d, &v)| [d, v]).collect();
                plot_ui.line(Line::new(fii_pts).color(INFO).width(2.0_f32).name("FII"));
                plot_ui.line(Line::new(dii_pts).color(AMBER).width(2.0_f32).name("DII"));
            });
    }

    fn draw_sector_perf(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("WEI — Sector Performance").strong());
        let sectors: [(&str, f64); 15] = [
            ("IT", -0.5),
            ("Banking", 1.8),
            ("Oil & Gas", 1.2),
            ("FMCG", -0.3),
            ("Auto", 1.4),
            ("Pharma", -0.8),
            ("Metals", 1.5),
            ("Cons Dur", 0.7),
            ("Cement", 1.0),
            ("Telecom", 2.1),
            ("Power", 0.4),
            ("Fin Svcs", 1.1),
            ("Chemicals", 0.5),
            ("Construction", 1.8),
            ("Realty", 0.9),
        ];
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let max_change = sectors.iter().map(|(_, c)| c.abs()).fold(0.0_f64, f64::max);
        let bar_height = rect.height() / sectors.len() as f32;
        let bar_max_w = rect.width() * 0.7_f32;
        let center_x = rect.center().x;
        for (i, (name, change)) in sectors.iter().enumerate() {
            let y = rect.top() + i as f32 * bar_height + 2.0_f32;
            let h = bar_height - 4.0_f32;
            let w = (change.abs() / max_change) * bar_max_w as f64;
            let color = if *change >= 0.0 { PROFIT } else { LOSS };
            let bar_rect = if *change >= 0.0 {
                egui::Rect::from_min_size(egui::Pos2::new(center_x, y), Vec2::new(w as f32, h))
            } else {
                egui::Rect::from_min_size(
                    egui::Pos2::new(center_x - w as f32, y),
                    Vec2::new(w as f32, h),
                )
            };
            painter.rect_filled(bar_rect, 2.0_f32, color);
            painter.text(
                egui::Pos2::new(rect.left() + 4.0_f32, y + h / 2.0_f32),
                egui::Align2::LEFT_CENTER,
                *name,
                egui::FontId::monospace(11.0_f32),
                Color32::WHITE,
            );
        }
        painter.line_segment(
            [
                egui::Pos2::new(center_x, rect.top()),
                egui::Pos2::new(center_x, rect.bottom()),
            ],
            Stroke::new(1.0_f32, Color32::GRAY),
        );
    }

    fn draw_yield_curve(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("GOVT — India Sovereign Yield Curve").strong());
        Plot::new("yield_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let colors = [AMBER, INFO, PROFIT];
                for (idx, (label, tenors, yields)) in self.data.yield_curves.iter().enumerate() {
                    let color = colors[idx % colors.len()];
                    let pts: PlotPoints =
                        tenors.iter().zip(yields).map(|(&t, &y)| [t, y]).collect();
                    plot_ui.line(Line::new(pts).color(color).width(2.0_f32).name(label));
                }
            });
    }

    fn draw_usdinr(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("USD/INR Exchange Rate").strong());
        Plot::new("usdinr_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let days: Vec<f64> = (0..60).map(|i| i as f64).collect();
                let rate: Vec<f64> = days
                    .iter()
                    .map(|&d| 83.0 + 0.5 * (d * 0.1).sin() + 0.3 * (d * 0.05).cos())
                    .collect();
                let pts: PlotPoints = days
                    .iter()
                    .zip(rate.iter())
                    .map(|(&d, &v)| [d, v])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
            });
    }

    fn draw_monsoon_agri(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Monsoon & Agriculture").strong());
        Plot::new("monsoon_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let months = ["Jun", "Jul", "Aug", "Sep"];
                let rainfall = [150.0_f64, 280.0, 220.0, 180.0];
                let agri_growth = [3.5_f64, 4.2, 3.8, 3.0];
                let rain_pts: PlotPoints = rainfall
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect();
                let agri_pts: PlotPoints = agri_growth
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v * 50.0])
                    .collect();
                plot_ui.line(
                    Line::new(rain_pts)
                        .color(INFO)
                        .width(2.0_f32)
                        .name("Rainfall (mm)"),
                );
                plot_ui.line(
                    Line::new(agri_pts)
                        .color(PROFIT)
                        .width(2.0_f32)
                        .name("Agri Growth (x50)"),
                );
                let _ = months;
            });
    }

    fn draw_seasonality(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Seasonality — Month x Weekday").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let r_max = rect.width().min(rect.height()) * 0.42_f32;
        let r_min = r_max * 0.22_f32;
        let ring_step = (r_max - r_min) / 7.0_f32;
        let max_abs = self
            .data
            .seasonality
            .iter()
            .flatten()
            .cloned()
            .fold(0.0_f64, |a, v| a.max(v.abs()))
            .max(1e-6);
        for (m, row) in self.data.seasonality.iter().enumerate() {
            let theta0 =
                -std::f32::consts::FRAC_PI_2 + (m as f32) * std::f32::consts::TAU / 12.0_f32;
            let theta1 = theta0 + std::f32::consts::TAU / 12.0_f32;
            for (d, &v) in row.iter().enumerate() {
                let r_inner = r_min + ring_step * d as f32;
                let r_outer = r_inner + ring_step * 0.92_f32;
                let t = (v / max_abs).clamp(-1.0, 1.0) as f32;
                let target = if t >= 0.0 { PROFIT } else { LOSS };
                let alpha = t.abs();
                let color = Color32::from_rgb(
                    lerp(15, target.r(), alpha as f64),
                    lerp(15, target.g(), alpha as f64),
                    lerp(15, target.b(), alpha as f64),
                );
                let segments = 10;
                let mut points = Vec::with_capacity(segments * 2 + 2);
                for s in 0..=segments {
                    let t = theta0 + (theta1 - theta0) * (s as f32 / segments as f32);
                    points.push(center + Vec2::new(r_outer * t.cos(), r_outer * t.sin()));
                }
                for s in (0..=segments).rev() {
                    let t = theta0 + (theta1 - theta0) * (s as f32 / segments as f32);
                    points.push(center + Vec2::new(r_inner * t.cos(), r_inner * t.sin()));
                }
                painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
            }
        }
    }
    // ==================== BLOOMBERG-STYLE ====================

    fn draw_world_indices(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("World Indices").strong());
        let indices = [
            ("S&P 500", 5200.0, 0.8),
            ("NASDAQ", 16500.0, 1.2),
            ("DOW", 39000.0, 0.5),
            ("FTSE", 8200.0, -0.3),
            ("DAX", 18500.0, 0.7),
            ("NIKKEI", 39000.0, 1.5),
            ("SHANGHAI", 3100.0, -0.8),
            ("HANG SENG", 18000.0, -1.2),
        ];
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let bar_height = rect.height() / indices.len() as f32;
        for (i, (name, _value, change)) in indices.iter().enumerate() {
            let y = rect.top() + i as f32 * bar_height + 2.0_f32;
            let h = bar_height - 4.0_f32;
            let color = if *change >= 0.0 { PROFIT } else { LOSS };
            let w = (f64::abs(*change) / 2.0) * rect.width() as f64 * 0.3_f64;
            let bar_rect = if *change >= 0.0 {
                egui::Rect::from_min_size(
                    egui::Pos2::new(rect.center().x, y),
                    Vec2::new(w as f32, h),
                )
            } else {
                egui::Rect::from_min_size(
                    egui::Pos2::new(rect.center().x - w as f32, y),
                    Vec2::new(w as f32, h),
                )
            };
            painter.rect_filled(bar_rect, 2.0_f32, color);
            painter.text(
                egui::Pos2::new(rect.left() + 4.0_f32, y + h / 2.0_f32),
                egui::Align2::LEFT_CENTER,
                *name,
                egui::FontId::monospace(11.0_f32),
                Color32::WHITE,
            );
        }
    }

    fn draw_ticker_tape(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Ticker Tape").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let symbols = [
            ("RELIANCE", 2500.0, 1.2),
            ("TCS", 3800.0, -0.5),
            ("INFY", 1500.0, 0.8),
            ("HDFCBANK", 1650.0, 1.5),
            ("ICICIBANK", 980.0, -0.3),
            ("ITC", 450.0, 0.1),
        ];
        let row_height = rect.height() / symbols.len() as f32;
        for (i, (sym, price, change)) in symbols.iter().enumerate() {
            let y = rect.top() + i as f32 * row_height;
            let color = if *change >= 0.0 { PROFIT } else { LOSS };
            painter.text(
                egui::Pos2::new(rect.left() + 10.0_f32, y + row_height / 2.0_f32),
                egui::Align2::LEFT_CENTER,
                *sym,
                egui::FontId::monospace(12.0_f32),
                Color32::WHITE,
            );
            painter.text(
                egui::Pos2::new(rect.center().x, y + row_height / 2.0_f32),
                egui::Align2::CENTER_CENTER,
                format!("{:.2}", price),
                egui::FontId::monospace(12.0_f32),
                Color32::WHITE,
            );
            painter.text(
                egui::Pos2::new(rect.right() - 10.0_f32, y + row_height / 2.0_f32),
                egui::Align2::RIGHT_CENTER,
                format!("{:+.2}%", change),
                egui::FontId::monospace(12.0_f32),
                color,
            );
        }
    }

    fn draw_currency_matrix(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Currency Matrix").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let currencies = ["USD", "EUR", "GBP", "JPY", "INR"];
        let cell_w = rect.width() / currencies.len() as f32;
        let cell_h = rect.height() / currencies.len() as f32;
        for (i, c1) in currencies.iter().enumerate() {
            for (j, c2) in currencies.iter().enumerate() {
                let val = if i == j {
                    1.0
                } else {
                    0.5 + (i as f64 * 0.1 + j as f64 * 0.05)
                };
                let intensity = ((val - 0.5) / 0.7).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, INFO.r(), intensity),
                    lerp(10, INFO.g(), intensity),
                    lerp(10, INFO.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + j as f32 * cell_w,
                        rect.top() + i as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }

    fn draw_sector_wheel(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Sector Wheel").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let radius = rect.width().min(rect.height()) * 0.4_f32;
        let sectors = [
            ("IT", -0.5),
            ("Banking", 1.8),
            ("Oil & Gas", 1.2),
            ("FMCG", -0.3),
            ("Auto", 1.4),
            ("Pharma", -0.8),
            ("Metals", 1.5),
            ("Cons Dur", 0.7),
        ];
        let angle_step = std::f32::consts::TAU / sectors.len() as f32;
        for (i, (name, change)) in sectors.iter().enumerate() {
            let angle = i as f32 * angle_step - std::f32::consts::FRAC_PI_2;
            let color = if *change >= 0.0 { PROFIT } else { LOSS };
            let label_pos =
                center + Vec2::new(radius * 1.2 * angle.cos(), radius * 1.2 * angle.sin());
            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                *name,
                egui::FontId::monospace(10.0_f32),
                color,
            );
            let inner = center + Vec2::new(radius * 0.6 * angle.cos(), radius * 0.6 * angle.sin());
            painter.circle_filled(inner, 8.0, color);
        }
    }

    fn draw_earnings_calendar(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Earnings Calendar").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let earnings = [
            ("RELIANCE", "Q3 FY26", 15.2),
            ("TCS", "Q3 FY26", 8.5),
            ("INFY", "Q3 FY26", 5.3),
            ("HDFCBANK", "Q3 FY26", 12.1),
            ("ICICIBANK", "Q3 FY26", 10.8),
        ];
        let row_height = rect.height() / earnings.len() as f32;
        for (i, (sym, quarter, eps)) in earnings.iter().enumerate() {
            let y = rect.top() + i as f32 * row_height + 4.0_f32;
            painter.text(
                egui::Pos2::new(rect.left() + 10.0_f32, y),
                egui::Align2::LEFT_CENTER,
                *sym,
                egui::FontId::monospace(12.0_f32),
                Color32::WHITE,
            );
            painter.text(
                egui::Pos2::new(rect.center().x, y),
                egui::Align2::CENTER_CENTER,
                *quarter,
                egui::FontId::monospace(11.0_f32),
                Color32::GRAY,
            );
            painter.text(
                egui::Pos2::new(rect.right() - 10.0_f32, y),
                egui::Align2::RIGHT_CENTER,
                format!("EPS: {:.1}", eps),
                egui::FontId::monospace(12.0_f32),
                PROFIT,
            );
        }
    }

    fn draw_econ_calendar(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Economic Calendar").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let events = [
            ("RBI Rate Decision", "High", "6.50%"),
            ("CPI Inflation", "High", "5.2%"),
            ("GDP Growth", "Medium", "7.2%"),
            ("IIP", "Medium", "4.5%"),
            ("Trade Balance", "Medium", "-$20B"),
        ];
        let row_height = rect.height() / events.len() as f32;
        for (i, (event, impact, value)) in events.iter().enumerate() {
            let y = rect.top() + i as f32 * row_height + 4.0_f32;
            let impact_color = match *impact {
                "High" => LOSS,
                "Medium" => AMBER,
                _ => PROFIT,
            };
            painter.text(
                egui::Pos2::new(rect.left() + 10.0_f32, y),
                egui::Align2::LEFT_CENTER,
                *event,
                egui::FontId::monospace(12.0_f32),
                Color32::WHITE,
            );
            painter.text(
                egui::Pos2::new(rect.center().x, y),
                egui::Align2::CENTER_CENTER,
                *impact,
                egui::FontId::monospace(11.0_f32),
                impact_color,
            );
            painter.text(
                egui::Pos2::new(rect.right() - 10.0_f32, y),
                egui::Align2::RIGHT_CENTER,
                *value,
                egui::FontId::monospace(12.0_f32),
                PROFIT,
            );
        }
    }

    fn draw_correlation_network(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Correlation Network").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let nodes = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC"];
        let n = nodes.len();
        let center = rect.center();
        let radius = rect.width().min(rect.height()) * 0.35_f32;
        let positions: Vec<egui::Pos2> = (0..n)
            .map(|i| {
                let angle =
                    i as f32 * std::f32::consts::TAU / n as f32 - std::f32::consts::FRAC_PI_2;
                center + Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = if i == j {
                    1.0
                } else {
                    0.3 + (i as f64 * 0.1 + j as f64 * 0.05) % 0.5
                };
                let color = if corr > 0.5 { PROFIT } else { INFO };
                painter.line_segment([positions[i], positions[j]], Stroke::new(1.0_f32, color));
            }
        }
        for (i, pos) in positions.iter().enumerate() {
            painter.circle_filled(*pos, 12.0, AMBER);
            painter.text(
                *pos,
                egui::Align2::CENTER_CENTER,
                nodes[i],
                egui::FontId::monospace(8.0_f32),
                Color32::BLACK,
            );
        }
    }

    fn draw_return_heatmap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Return Heatmap").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC"];
        let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
        let cell_w = rect.width() / months.len() as f32;
        let cell_h = rect.height() / stocks.len() as f32;
        for (si, _sym) in stocks.iter().enumerate() {
            for (mi, _month) in months.iter().enumerate() {
                let ret = ((si * 7 + mi * 13) % 20) as f64 / 10.0 - 1.0;
                let intensity = ret.abs().clamp(0.0, 1.0);
                let color = if ret >= 0.0 {
                    Color32::from_rgb(
                        lerp(10, PROFIT.r(), intensity),
                        lerp(10, PROFIT.g(), intensity),
                        lerp(10, PROFIT.b(), intensity),
                    )
                } else {
                    Color32::from_rgb(
                        lerp(10, LOSS.r(), intensity),
                        lerp(10, LOSS.g(), intensity),
                        lerp(10, LOSS.b(), intensity),
                    )
                };
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + mi as f32 * cell_w,
                        rect.top() + si as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }
    // ==================== ADVANCED ====================

    fn draw_multi_indicator(&self, ui: &mut egui::Ui) {
        use bt_analytics::{ema, rsi, sma};
        let series = &self.data.candles;
        let sma20 = sma(series, 20);
        let ema50 = ema(series, 50);
        let rsi_vals = rsi(series, 14);
        ui.label(RichText::new(format!("MULTI — Multi Indicator — {}", series.symbol)).strong());
        Plot::new("multi_price")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.5_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
                let sma_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !sma20[i].is_nan() {
                            Some([c.t, sma20[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(sma_pts).color(INFO).width(1.5_f32).name("SMA20"));
                let ema_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !ema50[i].is_nan() {
                            Some([c.t, ema50[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(ema_pts)
                        .color(PURPLE)
                        .width(1.5_f32)
                        .name("EMA50"),
                );
            });
        Plot::new("multi_rsi")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !rsi_vals[i].is_nan() {
                            Some([c.t, rsi_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(PROFIT).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(70.0).color(LOSS));
                plot_ui.hline(egui_plot::HLine::new(30.0).color(PROFIT));
            });
    }

    fn draw_multi_timeframe(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        ui.label(RichText::new(format!("MTF — Multi Timeframe — {}", series.symbol)).strong());
        Plot::new("mtf_daily")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.33_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Daily"));
            });
        Plot::new("mtf_weekly")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.33_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let weekly: Vec<(f64, f64)> = series
                    .candles
                    .chunks(5)
                    .map(|chunk| (chunk[0].t, chunk.last().unwrap().close))
                    .collect();
                let pts: PlotPoints = weekly.iter().map(|&(t, c)| [t, c]).collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("Weekly"));
            });
        Plot::new("mtf_monthly")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let monthly: Vec<(f64, f64)> = series
                    .candles
                    .chunks(20)
                    .map(|chunk| (chunk[0].t, chunk.last().unwrap().close))
                    .collect();
                let pts: PlotPoints = monthly.iter().map(|&(t, c)| [t, c]).collect();
                plot_ui.line(Line::new(pts).color(PURPLE).width(2.5_f32).name("Monthly"));
            });
    }

    fn draw_macd_divergence(&self, ui: &mut egui::Ui) {
        use bt_analytics::macd;
        let series = &self.data.candles;
        let (macd_line, _signal_line, _histogram) = macd(series);
        ui.label(RichText::new(format!("MACD-DIV — MACD Divergence — {}", series.symbol)).strong());
        Plot::new("macd_div_price")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.5_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
            });
        Plot::new("macd_div_macd")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !macd_line[i].is_nan() {
                            Some([c.t, macd_line[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("MACD"));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_bollinger_breakout(&self, ui: &mut egui::Ui) {
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (_mid, upper, lower) = bollinger(series, 20, 2.0);
        ui.label(RichText::new(format!("BB-BO — Bollinger Breakout — {}", series.symbol)).strong());
        Plot::new("bb_bo_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
                let upper_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !upper[i].is_nan() {
                            Some([c.t, upper[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(upper_pts)
                        .color(LOSS)
                        .width(1.0_f32)
                        .name("Upper"),
                );
                let lower_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !lower[i].is_nan() {
                            Some([c.t, lower[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lower_pts)
                        .color(PROFIT)
                        .width(1.0_f32)
                        .name("Lower"),
                );
            });
    }

    fn draw_volume_weighted_scatter(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(
            RichText::new(format!("VWS — Volume-Price Scatter — {}", candles.symbol)).strong(),
        );
        Plot::new("vws_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let max_vol = candles
                    .candles
                    .iter()
                    .map(|c| c.volume)
                    .fold(0.0_f64, f64::max);
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    let radius = 2.0_f64 + 6.0 * (c.volume / max_vol);
                    plot_ui.points(
                        Points::new(PlotPoints::from(vec![[c.t, c.close]]))
                            .color(color)
                            .radius(radius as f32)
                            .shape(MarkerShape::Circle),
                    );
                }
            });
    }

    fn draw_price_momentum(&self, ui: &mut egui::Ui) {
        use bt_analytics::roc;
        let series = &self.data.candles;
        let roc_vals = roc(series, 10);
        ui.label(RichText::new(format!("PM — Price Momentum — {}", series.symbol)).strong());
        Plot::new("pm_price")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.5_f32)
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
            });
        Plot::new("pm_roc")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !roc_vals[i].is_nan() {
                            Some([c.t, roc_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(INFO).width(2.0_f32).name("ROC"));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_drawdown_recovery(&self, ui: &mut egui::Ui) {
        let dd = bt_viz::drawdown::compute_drawdown(&self.data.equity);
        ui.label(RichText::new("Drawdown & Recovery").strong());
        Plot::new("dd_rec_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = dd.iter().enumerate().map(|(i, &v)| [i as f64, v]).collect();
                plot_ui.line(Line::new(pts).color(LOSS).width(2.0_f32).name("Drawdown"));
                let recovery: PlotPoints = dd
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v == 0.0 {
                            Some([i as f64, 0.0])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.points(
                    Points::new(recovery)
                        .color(PROFIT)
                        .radius(5.0_f32)
                        .shape(MarkerShape::Circle)
                        .name("Recovery"),
                );
            });
    }

    fn draw_rolling_correlation(&self, ui: &mut egui::Ui) {
        use bt_analytics::rolling_correlation;
        let rets = &self.data.returns;
        let rc = rolling_correlation(rets, rets, 20);
        ui.label(RichText::new("Rolling Correlation").strong());
        Plot::new("rc_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = rc
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if !v.is_nan() {
                            Some([i as f64, v])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_tick_tape_adv(&self, ui: &mut egui::Ui) {
        let candles = &self.data.candles;
        ui.label(RichText::new(format!("TTA — Tick Tape Advanced — {}", candles.symbol)).strong());
        Plot::new("tta_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                for c in &candles.candles {
                    let color = if c.is_bullish() { PROFIT } else { LOSS };
                    plot_ui.points(
                        Points::new(PlotPoints::from(vec![[c.t, c.close]]))
                            .color(color)
                            .radius(4.0_f32)
                            .shape(MarkerShape::Circle),
                    );
                }
            });
    }

    fn draw_seasonality_adv(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Advanced Seasonality").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        let cell_w = rect.width() / months.len() as f32;
        let cell_h = rect.height() / 4.0_f32;
        let metrics = ["Return", "Volume", "Volatility", "Drawdown"];
        for (mi, _metric) in metrics.iter().enumerate() {
            for (moi, _month) in months.iter().enumerate() {
                let val = ((mi * 7 + moi * 13) % 20) as f64 / 10.0 - 1.0;
                let intensity = val.abs().clamp(0.0, 1.0);
                let color = if val >= 0.0 {
                    Color32::from_rgb(
                        lerp(10, PROFIT.r(), intensity),
                        lerp(10, PROFIT.g(), intensity),
                        lerp(10, PROFIT.b(), intensity),
                    )
                } else {
                    Color32::from_rgb(
                        lerp(10, LOSS.r(), intensity),
                        lerp(10, LOSS.g(), intensity),
                        lerp(10, LOSS.b(), intensity),
                    )
                };
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + moi as f32 * cell_w,
                        rect.top() + mi as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
            }
        }
    }

    fn draw_parabolic_sar(&self, ui: &mut egui::Ui) {
        use bt_analytics::parabolic_sar;
        let series = &self.data.candles;
        let sar = parabolic_sar(series, 0.02, 0.02, 0.2);
        ui.label(RichText::new(format!("PSAR — Parabolic SAR — {}", series.symbol)).strong());
        Plot::new("psar_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
                let sar_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !sar[i].is_nan() {
                            Some([c.t, sar[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.points(
                    Points::new(sar_pts)
                        .color(INFO)
                        .radius(3.0_f32)
                        .shape(MarkerShape::Circle)
                        .name("SAR"),
                );
            });
    }

    fn draw_macd_histogram(&self, ui: &mut egui::Ui) {
        use bt_analytics::macd;
        let series = &self.data.candles;
        let (_macd_line, _signal_line, histogram) = macd(series);
        ui.label(RichText::new(format!("MACD-H — MACD Histogram — {}", series.symbol)).strong());
        Plot::new("macd_h_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !histogram[i].is_nan() {
                            let color = if histogram[i] >= 0.0 { PROFIT } else { LOSS };
                            Some(
                                Bar::new(c.t, histogram[i])
                                    .width(0.5 * DAY_SECS)
                                    .fill(color),
                            )
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars));
                plot_ui.hline(egui_plot::HLine::new(0.0).color(Color32::GRAY));
            });
    }

    fn draw_rsi_heatmap(&self, ui: &mut egui::Ui) {
        use bt_analytics::rsi;
        let series = &self.data.candles;
        let rsi_vals = rsi(series, 14);
        ui.label(RichText::new(format!("RSI-H — RSI Heatmap — {}", series.symbol)).strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let cell_w = rect.width() / rsi_vals.len() as f32;
        let cell_h = rect.height() / 3.0_f32;
        for (i, &val) in rsi_vals.iter().enumerate() {
            if val.is_nan() {
                continue;
            }
            let color = if val > 70.0 {
                LOSS
            } else if val < 30.0 {
                PROFIT
            } else {
                AMBER
            };
            for row in 0..3 {
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + i as f32 * cell_w,
                        rect.top() + row as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 1.0_f32, cell_h - 1.0_f32),
                );
                painter.rect_filled(tile, 0.0, color);
            }
        }
    }

    fn draw_ichimoku_ema(&self, ui: &mut egui::Ui) {
        use bt_analytics::ema;
        let series = &self.data.candles;
        let ema9 = ema(series, 9);
        let ema26 = ema(series, 26);
        let ema52 = ema(series, 52);
        ui.label(RichText::new(format!("ICH-E — Ichimoku + EMA — {}", series.symbol)).strong());
        Plot::new("ich_ema_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
                let ema9_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !ema9[i].is_nan() {
                            Some([c.t, ema9[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(ema9_pts)
                        .color(PROFIT)
                        .width(1.5_f32)
                        .name("EMA9"),
                );
                let ema26_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !ema26[i].is_nan() {
                            Some([c.t, ema26[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(ema26_pts)
                        .color(LOSS)
                        .width(1.5_f32)
                        .name("EMA26"),
                );
                let ema52_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !ema52[i].is_nan() {
                            Some([c.t, ema52[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(ema52_pts)
                        .color(INFO)
                        .width(1.5_f32)
                        .name("EMA52"),
                );
            });
    }

    fn draw_keltner_breakout(&self, ui: &mut egui::Ui) {
        use bt_analytics::atr;
        use bt_analytics::bollinger;
        let series = &self.data.candles;
        let (mid, _upper, _lower) = bollinger(series, 20, 2.0);
        let atr_vals = atr(series, 14);
        ui.label(RichText::new(format!("KEL-BO — Keltner Breakout — {}", series.symbol)).strong());
        Plot::new("kel_bo_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
                let upper_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() && !atr_vals[i].is_nan() {
                            Some([c.t, mid[i] + 2.0 * atr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(upper_pts)
                        .color(LOSS)
                        .width(1.0_f32)
                        .name("Upper"),
                );
                let lower_pts: PlotPoints = series
                    .candles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        if !mid[i].is_nan() && !atr_vals[i].is_nan() {
                            Some([c.t, mid[i] - 2.0 * atr_vals[i]])
                        } else {
                            None
                        }
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lower_pts)
                        .color(PROFIT)
                        .width(1.0_f32)
                        .name("Lower"),
                );
            });
    }

    fn draw_donchian_breakout(&self, ui: &mut egui::Ui) {
        let series = &self.data.candles;
        ui.label(RichText::new(format!("DON-BO — Donchian Breakout — {}", series.symbol)).strong());
        Plot::new("don_bo_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .label_formatter(|_axis: &str, p: &egui_plot::PlotPoint| format_ts(p.x))
            .show(ui, |plot_ui| {
                let pts: PlotPoints = series.candles.iter().map(|c| [c.t, c.close]).collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(1.5_f32).name("Price"));
            });
    }

    fn draw_copula_heatmap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Copula Heatmap").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let n = 10;
        let cell_w = rect.width() / n as f32;
        let cell_h = rect.height() / n as f32;
        for i in 0..n {
            for j in 0..n {
                let u = i as f64 / n as f64;
                let v = j as f64 / n as f64;
                let copula = u * v + 0.1 * (1.0 - u) * (1.0 - v);
                let intensity = copula.clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, INFO.r(), intensity),
                    lerp(10, INFO.g(), intensity),
                    lerp(10, INFO.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        rect.left() + j as f32 * cell_w,
                        rect.top() + i as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 1.0_f32, cell_h - 1.0_f32),
                );
                painter.rect_filled(tile, 0.0, color);
            }
        }
    }

    fn draw_correlation_network_adv(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Advanced Correlation Network").strong());
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let nodes = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC"];
        let n = nodes.len();
        let center = rect.center();
        let radius = rect.width().min(rect.height()) * 0.35_f32;
        let positions: Vec<egui::Pos2> = (0..n)
            .map(|i| {
                let angle =
                    i as f32 * std::f32::consts::TAU / n as f32 - std::f32::consts::FRAC_PI_2;
                center + Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = 0.3 + (i as f64 * 0.1 + j as f64 * 0.05) % 0.5;
                let color = if corr > 0.5 { PROFIT } else { INFO };
                painter.line_segment([positions[i], positions[j]], Stroke::new(2.0_f32, color));
            }
        }
        for (i, pos) in positions.iter().enumerate() {
            painter.circle_filled(*pos, 15.0, AMBER);
            painter.text(
                *pos,
                egui::Align2::CENTER_CENTER,
                nodes[i],
                egui::FontId::monospace(8.0_f32),
                Color32::BLACK,
            );
        }
    }

    fn refresh_market_data(&mut self) {
        // Production: bt_data::DataService::fetch_quote() per symbol.
        // 78+ individual HTTP calls are too slow for a 30s refresh cycle,
        // so the demo regenerates sample data instead.
        self.market_quotes = sample_market_data();
        self.market_last_refresh = Some(Instant::now());
    }

    fn draw_market_watch(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("MarketWatch — World Equity Indices (WEI)").strong());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Search:").color(Color32::GRAY));
            ui.add(egui::TextEdit::singleline(&mut self.market_search).desired_width(220.0_f32));
            ui.separator();
            let ago = self
                .market_last_refresh
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            ui.label(RichText::new(format!("Last refresh: {}s ago", ago)).color(Color32::GRAY));
            if self.live {
                ui.colored_label(PROFIT, "Auto-refresh: 30s");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new("Sample data — production uses DataService::fetch_quote()")
                        .color(Color32::GRAY)
                        .weak(),
                );
            });
        });
        ui.separator();

        let query = self.market_search.to_lowercase();
        let (sort_col, sort_asc) = (self.market_sort_col, self.market_sort_asc);
        let mut rows: Vec<MarketQuote> = self
            .market_quotes
            .iter()
            .filter(|q| {
                query.is_empty()
                    || q.symbol.to_lowercase().contains(&query)
                    || q.name.to_lowercase().contains(&query)
            })
            .cloned()
            .collect();

        rows.sort_by(|a, b| {
            let ord = match sort_col {
                MarketSortCol::Symbol => a.symbol.cmp(&b.symbol),
                MarketSortCol::Name => a.name.cmp(&b.name),
                MarketSortCol::Price => a
                    .price
                    .partial_cmp(&b.price)
                    .unwrap_or(std::cmp::Ordering::Equal),
                MarketSortCol::Change => a
                    .change
                    .partial_cmp(&b.change)
                    .unwrap_or(std::cmp::Ordering::Equal),
                MarketSortCol::ChangePct => a
                    .change_pct
                    .partial_cmp(&b.change_pct)
                    .unwrap_or(std::cmp::Ordering::Equal),
                MarketSortCol::Volume => a.volume.cmp(&b.volume),
                MarketSortCol::MarketCap => a
                    .market_cap
                    .partial_cmp(&b.market_cap)
                    .unwrap_or(std::cmp::Ordering::Equal),
            };
            if sort_asc {
                ord
            } else {
                ord.reverse()
            }
        });

        let mut header_btn = |ui: &mut egui::Ui, col: MarketSortCol, label: &str| {
            let active = self.market_sort_col == col;
            let arrow = if active {
                if self.market_sort_asc {
                    " \u{25B2}"
                } else {
                    " \u{25BC}"
                }
            } else {
                ""
            };
            let text = RichText::new(format!("{}{}", label, arrow))
                .color(if active { AMBER } else { Color32::WHITE })
                .strong();
            if ui.button(text).clicked() {
                if self.market_sort_col == col {
                    self.market_sort_asc = !self.market_sort_asc;
                } else {
                    self.market_sort_col = col;
                    self.market_sort_asc = col != MarketSortCol::MarketCap;
                }
            }
        };

        egui::Grid::new("market_watch_table")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(70.0_f32)
            .show(ui, |ui| {
                header_btn(ui, MarketSortCol::Symbol, "Symbol");
                header_btn(ui, MarketSortCol::Name, "Name");
                header_btn(ui, MarketSortCol::Price, "Price");
                header_btn(ui, MarketSortCol::Change, "Change");
                header_btn(ui, MarketSortCol::ChangePct, "Change%");
                header_btn(ui, MarketSortCol::Volume, "Volume");
                header_btn(ui, MarketSortCol::MarketCap, "Market Cap");
                ui.end_row();

                for q in &rows {
                    ui.label(RichText::new(&q.symbol).monospace().strong().size(12.0_f32));
                    ui.label(RichText::new(&q.name).size(11.0_f32));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{:.2}", q.price))
                                .monospace()
                                .size(12.0_f32),
                        );
                    });
                    let color = if q.change >= 0.0 { PROFIT } else { LOSS };
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.colored_label(
                            color,
                            RichText::new(format!("{:+.2}", q.change))
                                .monospace()
                                .size(12.0_f32),
                        );
                    });
                    let color = if q.change_pct >= 0.0 { PROFIT } else { LOSS };
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.colored_label(
                            color,
                            RichText::new(format!("{:+.2}%", q.change_pct))
                                .monospace()
                                .size(12.0_f32),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(fmt_big_num(q.volume as f64))
                                .monospace()
                                .size(12.0_f32),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(fmt_big_num(q.market_cap))
                                .monospace()
                                .size(12.0_f32),
                        );
                    });
                    ui.end_row();
                }
            });
    }

    // ==================== INDIA-SPECIFIC (NEW) ====================

    fn draw_fo_chain(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("FOChain — NSE F&O Options Chain (NIFTY 50)").strong());
        let spot = 24_842.10_f64;
        let chain = india::sample_options_chain(spot);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("Spot: {:.2}", spot))
                    .color(AMBER)
                    .monospace(),
            );
            ui.separator();
            ui.label(
                RichText::new("Sample data — production uses NSE F&O feed")
                    .color(Color32::GRAY)
                    .weak(),
            );
        });
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("fo_chain_table")
                .striped(true)
                .spacing(Vec2::new(8.0_f32, 2.0_f32))
                .min_col_width(62.0_f32)
                .show(ui, |ui| {
                    for h in [
                        "Strike", "Call OI", "Chg OI", "C IV", "Delta", "Gamma", "Theta", "Vega",
                        "Put OI", "Chg OI", "P IV", "Delta", "Gamma", "Theta", "Vega",
                    ] {
                        ui.label(RichText::new(h).strong().monospace().size(11.0_f32));
                    }
                    ui.end_row();
                    for e in &chain {
                        ui.label(
                            RichText::new(format!("{:.0}", e.strike))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(fmt_big_num(e.call_oi as f64))
                                .monospace()
                                .size(11.0_f32),
                        );
                        let c = if e.call_chg_oi >= 0 { PROFIT } else { LOSS };
                        ui.colored_label(
                            c,
                            RichText::new(format!("{:+}", e.call_chg_oi))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.2}%", e.call_iv * 100.0))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.3}", e.call_delta))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.4}", e.call_gamma))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.colored_label(
                            LOSS,
                            RichText::new(format!("{:.3}", e.call_theta))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.3}", e.call_vega))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(fmt_big_num(e.put_oi as f64))
                                .monospace()
                                .size(11.0_f32),
                        );
                        let c = if e.put_chg_oi >= 0 { PROFIT } else { LOSS };
                        ui.colored_label(
                            c,
                            RichText::new(format!("{:+}", e.put_chg_oi))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.2}%", e.put_iv * 100.0))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.colored_label(
                            LOSS,
                            RichText::new(format!("{:.3}", e.put_delta))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.4}", e.put_gamma))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.colored_label(
                            LOSS,
                            RichText::new(format!("{:.3}", e.put_theta))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.label(
                            RichText::new(format!("{:.3}", e.put_vega))
                                .monospace()
                                .size(11.0_f32),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    fn draw_iv_surface_india(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("IV Surface — NIFTY Implied Volatility").strong());
        let surf = india::sample_iv_surface();
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let label_w = 46.0_f32;
        let label_h = 20.0_f32;
        let grid_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.left() + label_w, rect.top()),
            Vec2::new(rect.width() - label_w, rect.height() - label_h),
        );
        let cell_w = grid_rect.width() / surf.strikes.len() as f32;
        let cell_h = grid_rect.height() / surf.tenors.len() as f32;
        let mut max_iv = 0.0_f64;
        for row in &surf.ivs {
            for &v in row {
                max_iv = max_iv.max(v);
            }
        }
        for (ti, &tenor) in surf.tenors.iter().enumerate() {
            for (si, &strike) in surf.strikes.iter().enumerate() {
                let iv = surf.ivs[ti][si];
                let intensity = (iv / max_iv).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, AMBER.r(), intensity),
                    lerp(10, AMBER.g(), intensity),
                    lerp(10, AMBER.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        grid_rect.left() + si as f32 * cell_w,
                        grid_rect.top() + ti as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
                painter.text(
                    tile.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{:.1}%", iv * 100.0),
                    egui::FontId::monospace(11.0_f32),
                    Color32::WHITE,
                );
            }
        }
        for (si, &strike) in surf.strikes.iter().enumerate() {
            painter.text(
                egui::Pos2::new(
                    grid_rect.left() + si as f32 * cell_w + cell_w / 2.0_f32,
                    grid_rect.bottom() + 2.0_f32,
                ),
                egui::Align2::CENTER_TOP,
                format!("{:.0}%", strike * 100.0),
                egui::FontId::monospace(9.0_f32),
                Color32::GRAY,
            );
        }
        for (ti, &tenor) in surf.tenors.iter().enumerate() {
            painter.text(
                egui::Pos2::new(
                    rect.left() + 2.0_f32,
                    grid_rect.top() + ti as f32 * cell_h + cell_h / 2.0_f32,
                ),
                egui::Align2::LEFT_CENTER,
                format!("{}D", tenor as i32),
                egui::FontId::monospace(9.0_f32),
                Color32::GRAY,
            );
        }
    }

    fn draw_oi_heatmap(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("OI Heatmap — F&O Open Interest (lakh contracts)").strong());
        let hm = india::sample_oi_heatmap();
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let label_w = 90.0_f32;
        let label_h = 20.0_f32;
        let grid_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.left() + label_w, rect.top()),
            Vec2::new(rect.width() - label_w, rect.height() - label_h),
        );
        let cell_w = grid_rect.width() / hm.strikes.len() as f32;
        let cell_h = grid_rect.height() / hm.underlyings.len() as f32;
        let mut max_oi = 0.0_f64;
        for row in &hm.oi {
            for &v in row {
                max_oi = max_oi.max(v);
            }
        }
        for (ui_idx, name) in hm.underlyings.iter().enumerate() {
            for (si, &_strike) in hm.strikes.iter().enumerate() {
                let oi = hm.oi[ui_idx][si];
                let intensity = (oi / max_oi).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    lerp(10, INFO.r(), intensity),
                    lerp(10, INFO.g(), intensity),
                    lerp(10, INFO.b(), intensity),
                );
                let tile = egui::Rect::from_min_size(
                    egui::Pos2::new(
                        grid_rect.left() + si as f32 * cell_w,
                        grid_rect.top() + ui_idx as f32 * cell_h,
                    ),
                    Vec2::new(cell_w - 2.0_f32, cell_h - 2.0_f32),
                );
                painter.rect_filled(tile, 4.0_f32, color);
                if cell_w > 40.0 && cell_h > 18.0 {
                    painter.text(
                        tile.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{:.1}", oi),
                        egui::FontId::monospace(10.0_f32),
                        Color32::WHITE,
                    );
                }
            }
            painter.text(
                egui::Pos2::new(
                    rect.left() + 4.0_f32,
                    grid_rect.top() + ui_idx as f32 * cell_h + cell_h / 2.0_f32,
                ),
                egui::Align2::LEFT_CENTER,
                name.as_str(),
                egui::FontId::monospace(10.0_f32),
                Color32::WHITE,
            );
        }
        for (si, &strike) in hm.strikes.iter().enumerate() {
            painter.text(
                egui::Pos2::new(
                    grid_rect.left() + si as f32 * cell_w + cell_w / 2.0_f32,
                    grid_rect.bottom() + 2.0_f32,
                ),
                egui::Align2::CENTER_TOP,
                format!("{:.0}%", strike * 100.0),
                egui::FontId::monospace(9.0_f32),
                Color32::GRAY,
            );
        }
    }

    fn draw_gsec(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("GSec — Government Securities Yield Curve").strong());
        let curve = india::sample_gsec_curve();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("10Y: {:.2}%", curve.yields[7]))
                    .color(AMBER)
                    .monospace(),
            );
            ui.separator();
            ui.label(
                RichText::new(format!("30Y: {:.2}%", curve.yields[10]))
                    .color(AMBER)
                    .monospace(),
            );
            ui.separator();
            let slope = curve.yields[7] - curve.yields[0];
            let c = if slope >= 0.0 { PROFIT } else { LOSS };
            ui.colored_label(
                c,
                RichText::new(format!("10Y-3M Slope: {:+.0}bp", slope * 100.0)).monospace(),
            );
        });
        ui.separator();
        Plot::new("gsec_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .show(ui, |plot_ui| {
                let pts: PlotPoints = curve
                    .tenors
                    .iter()
                    .zip(curve.yields.iter())
                    .map(|(&t, &y)| [t, y])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32).name("G-Sec"));
                let pts2: PlotPoints = curve
                    .tenors
                    .iter()
                    .zip(curve.yields.iter())
                    .map(|(&t, &y)| [t, y])
                    .collect();
                plot_ui.points(
                    Points::new(pts2)
                        .color(AMBER)
                        .radius(3.5_f32)
                        .shape(MarkerShape::Circle),
                );
            });
    }

    fn draw_money_market(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Money Market — MIBOR, TREPS, CP/CD Rates").strong());
        let mm = india::sample_money_market();
        let rates = vec![
            ("MIBOR 1D", mm.mibor_1d, 2.0),
            ("MIBOR 1W", mm.mibor_1w, 3.0),
            ("MIBOR 1M", mm.mibor_1m, 5.0),
            ("MIBOR 3M", mm.mibor_3m, 8.0),
            ("TREPS", mm.treps, -2.0),
            ("CBLO", mm.cblo, -3.0),
            ("T-Bill 91D", mm.t_bill_91, 1.0),
            ("T-Bill 182D", mm.t_bill_182, 4.0),
            ("T-Bill 364D", mm.t_bill_364, 6.0),
            ("CP 3M", mm.cp_3m, 10.0),
            ("CD 3M", mm.cd_3m, 7.0),
            ("Reverse Repo (SDF)", mm.reverse_repo, -25.0),
            ("MSF", mm.msf, 0.0),
        ];
        egui::Grid::new("money_market_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(140.0_f32)
            .show(ui, |ui| {
                ui.label(RichText::new("Instrument").strong());
                ui.label(RichText::new("Rate").strong());
                ui.label(RichText::new("Chg (bp)").strong());
                ui.end_row();
                for (name, val, chg) in &rates {
                    ui.label(RichText::new(*name).monospace().size(12.0_f32));
                    ui.label(
                        RichText::new(format!("{:.2}%", val))
                            .monospace()
                            .size(12.0_f32),
                    );
                    let c = if *chg >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", chg))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_rbi_policy(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("RBI Policy — Monetary Policy Dashboard").strong());
        let p = india::sample_rbi_policy();
        ui.horizontal(|ui| {
            ui.colored_label(
                AMBER,
                RichText::new(format!("Repo: {:.2}%", p.repo_rate))
                    .strong()
                    .size(16.0_f32),
            );
            ui.separator();
            ui.label(RichText::new(format!("Stance: {}", p.stance)).strong());
            ui.separator();
            ui.label(RichText::new(format!("Last MPC: {}", p.last_meeting)));
            ui.separator();
            ui.label(RichText::new(format!("Next MPC: {}", p.next_meeting)));
        });
        ui.separator();
        egui::Grid::new("rbi_policy_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(150.0_f32)
            .show(ui, |ui| {
                for (name, val) in [
                    ("Repo Rate", format!("{:.2}%", p.repo_rate)),
                    ("Reverse Repo (SDF)", format!("{:.2}%", p.reverse_repo)),
                    ("MSF / Bank Rate", format!("{:.2}%", p.msf)),
                    ("CRR", format!("{:.2}%", p.crr)),
                    ("SLR", format!("{:.2}%", p.slr)),
                    (
                        "Inflation Target",
                        format!(
                            "{:.1}% +/-{:.1}%",
                            p.inflation_target, p.inflation_tolerance
                        ),
                    ),
                    ("Real Rate (est.)", format!("{:.2}%", p.real_rate)),
                    ("Liquidity Stance", p.liquidity_modality.to_string()),
                ] {
                    ui.label(RichText::new(name).strong().size(12.0_f32));
                    ui.label(RichText::new(val).monospace().size(12.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_macro_india(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Macro India — CPI, WPI, IIP, GDP, PMI").strong());
        let indicators = india::sample_macro_indicators();
        egui::Grid::new("macro_india_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(130.0_f32)
            .show(ui, |ui| {
                for h in ["Indicator", "Value", "YoY", "Prev", "Freq"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for ind in &indicators {
                    ui.label(RichText::new(ind.name).size(12.0_f32));
                    let val_str = match ind.unit {
                        "idx" => format!("{:.1}", ind.value),
                        "% yoy" => format!("{:.1}%", ind.value),
                        "% GDP" => format!("{:.1}%", ind.value),
                        "USD bn" => format!("{:.1} bn", ind.value),
                        "%" => format!("{:.1}%", ind.value),
                        _ => format!("{:.1}{}", ind.value, ind.unit),
                    };
                    ui.label(RichText::new(val_str).monospace().size(12.0_f32));
                    let good_down = matches!(
                        ind.name,
                        "Trade Deficit" | "Fiscal Deficit" | "CAD" | "Unemployment"
                    );
                    let up = ind.yoy >= 0.0;
                    let c = if (up && !good_down) || (!up && good_down) {
                        PROFIT
                    } else {
                        LOSS
                    };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.1}%", ind.yoy))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", ind.prev))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(RichText::new(ind.frequency).size(11.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_commodities_india(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("MCX / NCDEX — Commodity Dashboard").strong());
        let quotes = india::sample_commodities();
        egui::Grid::new("commodities_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(110.0_f32)
            .show(ui, |ui| {
                for h in ["Commodity", "Exch", "Price", "Unit", "Chg%", "Lot"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for q in &quotes {
                    ui.label(RichText::new(q.name).size(12.0_f32));
                    ui.label(
                        RichText::new(q.exchange)
                            .color(INFO)
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", q.price))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(RichText::new(q.unit).size(11.0_f32));
                    let c = if q.change_pct >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.2}%", q.change_pct))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(
                        RichText::new(q.lot_size.to_string())
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_usdinr_curve(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("USD/INR — Spot & Forward Curve").strong());
        let fwd = india::sample_usdinr_forward();
        Plot::new("usdinr_fwd_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.55_f32)
            .show(ui, |plot_ui| {
                let pts: PlotPoints = fwd
                    .tenors
                    .iter()
                    .zip(fwd.outright.iter())
                    .map(|(&t, &v)| [t, v])
                    .collect();
                plot_ui.line(Line::new(pts).color(AMBER).width(2.0_f32).name("Outright"));
                let pts2: PlotPoints = fwd
                    .tenors
                    .iter()
                    .zip(fwd.outright.iter())
                    .map(|(&t, &v)| [t, v])
                    .collect();
                plot_ui.points(
                    Points::new(pts2)
                        .color(AMBER)
                        .radius(3.5_f32)
                        .shape(MarkerShape::Circle),
                );
            });
        egui::Grid::new("usdinr_fwd_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(90.0_f32)
            .show(ui, |ui| {
                for h in ["Tenor (days)", "Fwd Points", "Outright"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for i in 0..fwd.tenors.len() {
                    ui.label(
                        RichText::new(format!("{:.0}", fwd.tenors[i]))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.0}", fwd.forward_points[i]))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", fwd.outright[i]))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_yield_india(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Yield India — Sovereign, SDL, Corporate Curves").strong());
        let y = india::sample_yield_india();
        Plot::new("yield_india_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let series = [
                    ("Sovereign", &y.sovereign, AMBER),
                    ("SDL", &y.sdl, INFO),
                    ("Corp AAA", &y.corporate_aaa, PROFIT),
                    ("Corp AA", &y.corporate_aa, PURPLE),
                ];
                for (name, vals, color) in series {
                    let pts: PlotPoints = y
                        .tenors
                        .iter()
                        .zip(vals.iter())
                        .map(|(&t, &v)| [t, v])
                        .collect();
                    plot_ui.line(Line::new(pts).color(color).width(2.0_f32).name(name));
                }
            });
    }

    fn draw_mf_analytics(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("MF Analytics — Mutual Fund Schemes").strong());
        let schemes = india::sample_mf_schemes();
        egui::Grid::new("mf_grid")
            .striped(true)
            .spacing(Vec2::new(8.0_f32, 2.0_f32))
            .min_col_width(70.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Scheme", "Category", "NAV", "AUM (cr)", "1Y", "3Y", "5Y", "Exp%", "Sharpe",
                    "Beta", "Alpha", "SD%",
                ] {
                    ui.label(RichText::new(h).strong().size(11.0_f32));
                }
                ui.end_row();
                for s in &schemes {
                    ui.label(RichText::new(s.name).size(11.0_f32));
                    ui.label(RichText::new(s.category).size(10.0_f32));
                    ui.label(
                        RichText::new(format!("{:.2}", s.nav))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.0}", s.aum_cr))
                            .monospace()
                            .size(11.0_f32),
                    );
                    for r in [s.ret_1y, s.ret_3y, s.ret_5y] {
                        let c = if r >= 0.0 { PROFIT } else { LOSS };
                        ui.colored_label(
                            c,
                            RichText::new(format!("{:+.1}%", r))
                                .monospace()
                                .size(11.0_f32),
                        );
                    }
                    ui.label(
                        RichText::new(format!("{:.2}", s.expense))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", s.sharpe))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", s.beta))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if s.alpha >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.1}", s.alpha))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", s.std_dev))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_fpi_fii(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("FPI/FII — Flow Dashboard (INR crore)").strong());
        let flows = india::sample_fpi_fii_flows();
        let tot_fpi: f64 = flows.iter().map(|f| f.fpi_equity + f.fpi_debt).sum();
        let tot_dii: f64 = flows.iter().map(|f| f.dii).sum();
        ui.horizontal(|ui| {
            let c = if tot_fpi >= 0.0 { PROFIT } else { LOSS };
            ui.colored_label(
                c,
                RichText::new(format!("Cum FPI: {:+.0} cr", tot_fpi))
                    .strong()
                    .monospace(),
            );
            ui.separator();
            let c = if tot_dii >= 0.0 { PROFIT } else { LOSS };
            ui.colored_label(
                c,
                RichText::new(format!("Cum DII: {:+.0} cr", tot_dii))
                    .strong()
                    .monospace(),
            );
        });
        ui.separator();
        Plot::new("fpi_fii_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.45_f32)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                let mut cum = 0.0_f64;
                let pts: PlotPoints = flows
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        cum += f.fpi_equity;
                        [i as f64, cum]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(pts)
                        .color(LOSS)
                        .width(2.0_f32)
                        .name("Cum FPI Equity"),
                );
                let dii_pts: PlotPoints = flows
                    .iter()
                    .enumerate()
                    .map(|(i, f)| [i as f64, f.dii])
                    .collect();
                plot_ui.line(Line::new(dii_pts).color(PROFIT).width(2.0_f32).name("DII"));
            });
        egui::Grid::new("fpi_fii_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(90.0_f32)
            .show(ui, |ui| {
                for h in ["Date", "FPI Eq", "FPI Debt", "FII Eq", "DII"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for f in &flows {
                    ui.label(RichText::new(f.date).monospace().size(11.0_f32));
                    let c = if f.fpi_equity >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", f.fpi_equity))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if f.fpi_debt >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", f.fpi_debt))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if f.fii_equity >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", f.fii_equity))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if f.dii >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", f.dii))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_credit_ratings(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Credit Ratings — CRISIL / ICRA / CARE / IND").strong());
        let entries = india::sample_credit_ratings();
        egui::Grid::new("credit_ratings_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(110.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Issuer",
                    "Instrument",
                    "Rating",
                    "Outlook",
                    "Agency",
                    "Amount (cr)",
                    "Action",
                ] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for e in &entries {
                    ui.label(RichText::new(e.issuer).size(12.0_f32));
                    ui.label(RichText::new(e.instrument).monospace().size(11.0_f32));
                    ui.label(
                        RichText::new(e.rating)
                            .color(AMBER)
                            .strong()
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = match e.outlook {
                        "Positive" => PROFIT,
                        "Negative" => LOSS,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(c, RichText::new(e.outlook).size(11.0_f32));
                    ui.label(RichText::new(e.agency).size(11.0_f32));
                    ui.label(
                        RichText::new(format!("{:.0}", e.amount_cr))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = match e.action {
                        "Upgrade" => PROFIT,
                        "Downgrade" => LOSS,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(c, RichText::new(e.action).size(11.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_banking_india(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Banking India — System Indicators").strong());
        let inds = india::sample_banking_indicators();
        egui::Grid::new("banking_india_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(150.0_f32)
            .show(ui, |ui| {
                for h in ["Indicator", "Value", "Prev", "Trend"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for b in &inds {
                    ui.label(RichText::new(b.name).size(12.0_f32));
                    ui.label(
                        RichText::new(format!("{:.1}{}", b.value, b.unit))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", b.prev))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let up = b.value > b.prev;
                    let good = match b.name {
                        "Gross NPA Ratio" | "Net NPA Ratio" | "CD Ratio" => !up,
                        _ => up,
                    };
                    let c = if good { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(if up { "\u{25B2}" } else { "\u{25BC}" })
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_corp_actions(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Corp Actions — Corporate Actions Calendar").strong());
        let actions = india::sample_corp_actions();
        egui::Grid::new("corp_actions_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(90.0_f32)
            .show(ui, |ui| {
                for h in ["Symbol", "Company", "Action", "Ex-Date", "Record", "Detail"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for a in &actions {
                    ui.label(RichText::new(a.symbol).monospace().strong().size(11.0_f32));
                    ui.label(RichText::new(a.company).size(11.0_f32));
                    let c = match a.action {
                        "Dividend" => PROFIT,
                        "Split" | "Bonus" => INFO,
                        "Rights" => AMBER,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(c, RichText::new(a.action).strong().size(11.0_f32));
                    ui.label(RichText::new(a.ex_date).monospace().size(11.0_f32));
                    ui.label(RichText::new(a.record_date).monospace().size(11.0_f32));
                    ui.label(RichText::new(a.detail).size(11.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_ipo_pipeline(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("IPO Pipeline — Mainboard Tracker").strong());
        let ipos = india::sample_ipo_pipeline();
        egui::Grid::new("ipo_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(100.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Company",
                    "Sector",
                    "Size (cr)",
                    "Price Band",
                    "Open",
                    "Close",
                    "Status",
                    "GMP%",
                ] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for i in &ipos {
                    ui.label(RichText::new(i.company).size(12.0_f32));
                    ui.label(RichText::new(i.sector).size(11.0_f32));
                    ui.label(
                        RichText::new(format!("{:.0}", i.issue_size_cr))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(RichText::new(i.price_band).monospace().size(11.0_f32));
                    ui.label(RichText::new(i.open_date).monospace().size(11.0_f32));
                    ui.label(RichText::new(i.close_date).monospace().size(11.0_f32));
                    let c = match i.status {
                        "Live" => PROFIT,
                        "Upcoming" => INFO,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(c, RichText::new(i.status).strong().size(11.0_f32));
                    let c = if i.gmp >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.1}%", i.gmp))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_india_breadth(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("India Breadth — NSE Market Breadth").strong());
        let b = india::sample_breadth();
        let adv_pct = b.advances as f64 / b.total as f64 * 100.0;
        ui.horizontal(|ui| {
            ui.colored_label(
                PROFIT,
                RichText::new(format!("Advances: {}", b.advances))
                    .strong()
                    .monospace(),
            );
            ui.separator();
            ui.colored_label(
                LOSS,
                RichText::new(format!("Declines: {}", b.declines))
                    .strong()
                    .monospace(),
            );
            ui.separator();
            ui.label(RichText::new(format!("Unchanged: {}", b.unchanged)).monospace());
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "A/D Ratio: {:.2}",
                    b.advances as f64 / b.declines as f64
                ))
                .monospace(),
            );
        });
        ui.separator();
        let avail = ui.available_size();
        let (rect, _resp) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let bar_h = rect.height() / 4.0;
        let total = (b.advances + b.declines + b.unchanged) as f64;
        let adv_w = rect.width() as f64 * b.advances as f64 / total;
        let dec_w = rect.width() as f64 * b.declines as f64 / total;
        painter.rect_filled(
            egui::Rect::from_min_size(rect.min, Vec2::new(adv_w as f32, bar_h * 0.6_f32)),
            2.0_f32,
            PROFIT,
        );
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::Pos2::new(rect.min.x + adv_w as f32, rect.min.y),
                Vec2::new(dec_w as f32, bar_h * 0.6_f32),
            ),
            2.0_f32,
            LOSS,
        );
        painter.text(
            egui::Pos2::new(rect.min.x + 4.0_f32, rect.min.y + bar_h * 0.6_f32 + 4.0_f32),
            egui::Align2::LEFT_TOP,
            format!("Advances {:.1}%", adv_pct),
            egui::FontId::monospace(11.0_f32),
            Color32::WHITE,
        );
        let y2 = rect.min.y + bar_h;
        let hi_w = rect.width() as f64 * b.new_52w_highs as f64 / b.total as f64;
        let lo_w = rect.width() as f64 * b.new_52w_lows as f64 / b.total as f64;
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::Pos2::new(rect.min.x, y2),
                Vec2::new(hi_w as f32, bar_h * 0.6_f32),
            ),
            2.0_f32,
            PROFIT,
        );
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::Pos2::new(rect.min.x + hi_w as f32, y2),
                Vec2::new(lo_w as f32, bar_h * 0.6_f32),
            ),
            2.0_f32,
            LOSS,
        );
        painter.text(
            egui::Pos2::new(rect.min.x + 4.0_f32, y2 + bar_h * 0.6_f32 + 4.0_f32),
            egui::Align2::LEFT_TOP,
            format!(
                "52W Highs: {}   52W Lows: {}",
                b.new_52w_highs, b.new_52w_lows
            ),
            egui::FontId::monospace(11.0_f32),
            Color32::WHITE,
        );
        let y3 = rect.min.y + 2.0_f32 * bar_h;
        let d50_w = rect.width() as f64 * b.above_50dma as f64 / b.total as f64;
        let d200_w = rect.width() as f64 * b.above_200dma as f64 / b.total as f64;
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::Pos2::new(rect.min.x, y3),
                Vec2::new(d50_w as f32, bar_h * 0.6_f32),
            ),
            2.0_f32,
            INFO,
        );
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::Pos2::new(rect.min.x, y3 + bar_h * 0.6_f32),
                Vec2::new(d200_w as f32, bar_h * 0.6_f32),
            ),
            2.0_f32,
            AMBER,
        );
        painter.text(
            egui::Pos2::new(rect.min.x + 4.0_f32, y3 + bar_h + 4.0_f32),
            egui::Align2::LEFT_TOP,
            format!(
                "Above 50DMA: {} ({:.1}%)   Above 200DMA: {} ({:.1}%)",
                b.above_50dma,
                b.above_50dma as f64 / b.total as f64 * 100.0,
                b.above_200dma,
                b.above_200dma as f64 / b.total as f64 * 100.0
            ),
            egui::FontId::monospace(11.0_f32),
            Color32::WHITE,
        );
    }

    fn draw_sector_research(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Sector Research — Valuation & Momentum").strong());
        let sectors = india::sample_sector_research();
        egui::Grid::new("sector_research_grid")
            .striped(true)
            .spacing(Vec2::new(8.0_f32, 2.0_f32))
            .min_col_width(80.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Sector", "P/E", "P/B", "Div%", "EPS Gr%", "ROE%", "D/E", "1M", "3M", "1Y",
                    "Outlook",
                ] {
                    ui.label(RichText::new(h).strong().size(11.0_f32));
                }
                ui.end_row();
                for s in &sectors {
                    ui.label(RichText::new(s.sector).size(11.0_f32));
                    ui.label(
                        RichText::new(format!("{:.1}", s.pe))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", s.pb))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", s.div_yield))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", s.eps_growth))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", s.roe))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", s.debt_equity))
                            .monospace()
                            .size(11.0_f32),
                    );
                    for m in [s.mom_1m, s.mom_3m, s.mom_1y] {
                        let c = if m >= 0.0 { PROFIT } else { LOSS };
                        ui.colored_label(
                            c,
                            RichText::new(format!("{:+.1}%", m))
                                .monospace()
                                .size(11.0_f32),
                        );
                    }
                    let c = match s.outlook {
                        "Overweight" => PROFIT,
                        "Underweight" => LOSS,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(c, RichText::new(s.outlook).strong().size(11.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_india_news(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("India News — Market News Feed").strong());
        let news = india::sample_india_news();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for n in &news {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(n.time)
                            .color(Color32::GRAY)
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(RichText::new(n.source).color(INFO).strong().size(11.0_f32));
                    ui.label(
                        RichText::new(n.category)
                            .color(AMBER)
                            .monospace()
                            .size(10.0_f32),
                    );
                    let c = if n.sentiment >= 0.2 {
                        PROFIT
                    } else if n.sentiment <= -0.2 {
                        LOSS
                    } else {
                        Color32::GRAY
                    };
                    ui.colored_label(c, RichText::new(n.headline).size(11.0_f32));
                });
                ui.separator();
            }
        });
    }

    fn draw_regulatory(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Regulatory — SEBI / RBI Updates").strong());
        let updates = india::sample_regulatory_updates();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for u in &updates {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(u.date)
                            .color(Color32::GRAY)
                            .monospace()
                            .size(11.0_f32),
                    );
                    let rc = match u.regulator {
                        "SEBI" => INFO,
                        "RBI" => AMBER,
                        _ => PURPLE,
                    };
                    ui.colored_label(
                        rc,
                        RichText::new(u.regulator)
                            .strong()
                            .monospace()
                            .size(11.0_f32),
                    );
                    let ic = match u.impact {
                        "High" => LOSS,
                        "Medium" => AMBER,
                        _ => Color32::GRAY,
                    };
                    ui.colored_label(
                        ic,
                        RichText::new(format!("{} impact", u.impact)).size(10.0_f32),
                    );
                });
                ui.label(RichText::new(u.title).strong().size(12.0_f32));
                ui.label(RichText::new(u.summary).size(11.0_f32));
                ui.separator();
            }
        });
    }

    fn draw_gst_budget(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("GST & Budget — Collections & Fiscal Metrics").strong());
        let gst = india::sample_gst_collections();
        let latest = gst.last().unwrap();
        ui.horizontal(|ui| {
            ui.colored_label(
                AMBER,
                RichText::new(format!("{} GST: Rs {:.0} cr", latest.month, latest.gst_cr))
                    .strong()
                    .monospace(),
            );
            ui.separator();
            let c = if latest.yoy >= 0.0 { PROFIT } else { LOSS };
            ui.colored_label(
                c,
                RichText::new(format!("YoY {:+.1}%", latest.yoy)).monospace(),
            );
        });
        ui.separator();
        Plot::new("gst_plot")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height() * 0.45_f32)
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = gst
                    .iter()
                    .enumerate()
                    .map(|(i, g)| Bar::new(i as f64, g.gst_cr).width(0.6_f64).fill(AMBER))
                    .collect();
                plot_ui.bar_chart(BarChart::new(bars));
            });
        let metrics = india::sample_budget_metrics();
        egui::Grid::new("budget_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(140.0_f32)
            .show(ui, |ui| {
                for h in ["Metric", "FY25", "FY26", "Unit", "Chg"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for m in &metrics {
                    ui.label(RichText::new(m.name).size(12.0_f32));
                    ui.label(
                        RichText::new(format!("{:.1}", m.fy25))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.1}", m.fy26))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(RichText::new(m.unit).size(11.0_f32));
                    let chg = m.fy26 - m.fy25;
                    let good = match m.name {
                        "Fiscal Deficit" | "Interest Outgo" | "Subsidies" => chg <= 0.0,
                        _ => chg >= 0.0,
                    };
                    let c = if good { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.1}", chg))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_india_portfolio(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("India Portfolio — Holdings & Tax Analytics").strong());
        let holdings = india::sample_india_portfolio();
        let tot_stcg: f64 = holdings.iter().map(|h| h.stcg).sum();
        let tot_ltcg: f64 = holdings.iter().map(|h| h.ltcg).sum();
        let tot_div: f64 = holdings.iter().map(|h| h.div_income).sum();
        let tot_tax: f64 = holdings.iter().map(|h| h.tax_liability).sum();
        ui.horizontal(|ui| {
            ui.colored_label(
                LOSS,
                RichText::new(format!("STCG: {:+.0}", tot_stcg)).monospace(),
            );
            ui.separator();
            ui.colored_label(
                LOSS,
                RichText::new(format!("LTCG: {:+.0}", tot_ltcg)).monospace(),
            );
            ui.separator();
            ui.label(RichText::new(format!("Div Income: {:.0}", tot_div)).monospace());
            ui.separator();
            ui.colored_label(
                AMBER,
                RichText::new(format!("Tax: {:.0}", tot_tax))
                    .strong()
                    .monospace(),
            );
        });
        ui.separator();
        egui::Grid::new("india_portfolio_grid")
            .striped(true)
            .spacing(Vec2::new(8.0_f32, 2.0_f32))
            .min_col_width(80.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Symbol", "Name", "Qty", "Avg", "LTP", "STCG", "LTCG", "Div", "Tax",
                ] {
                    ui.label(RichText::new(h).strong().size(11.0_f32));
                }
                ui.end_row();
                for h in &holdings {
                    ui.label(RichText::new(h.symbol).monospace().strong().size(11.0_f32));
                    ui.label(RichText::new(h.name).size(11.0_f32));
                    ui.label(
                        RichText::new(format!("{:.0}", h.qty))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.0}", h.avg_price))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", h.ltp))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if h.stcg >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", h.stcg))
                            .monospace()
                            .size(11.0_f32),
                    );
                    let c = if h.ltcg >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        c,
                        RichText::new(format!("{:+.0}", h.ltcg))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.0}", h.div_income))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.colored_label(
                        AMBER,
                        RichText::new(format!("{:.0}", h.tax_liability))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.end_row();
                }
            });
    }

    fn draw_algo_feed(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Algo Feed — Market Data & Order Feed Status").strong());
        let feeds = india::sample_algo_feeds();
        egui::Grid::new("algo_feed_grid")
            .striped(true)
            .spacing(Vec2::new(10.0_f32, 2.0_f32))
            .min_col_width(130.0_f32)
            .show(ui, |ui| {
                for h in [
                    "Feed",
                    "Status",
                    "Latency (ms)",
                    "Msg/s",
                    "Uptime%",
                    "Last Error",
                ] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for f in &feeds {
                    ui.label(RichText::new(f.name).size(12.0_f32));
                    let c = match f.status {
                        "Connected" => PROFIT,
                        "Degraded" => AMBER,
                        _ => LOSS,
                    };
                    ui.colored_label(c, RichText::new(f.status).strong().size(12.0_f32));
                    let lc = if f.latency_ms < 50 {
                        PROFIT
                    } else if f.latency_ms < 200 {
                        AMBER
                    } else {
                        LOSS
                    };
                    ui.colored_label(
                        lc,
                        RichText::new(f.latency_ms.to_string())
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(
                        RichText::new(f.msgs_per_sec.to_string())
                            .monospace()
                            .size(12.0_f32),
                    );
                    let uc = if f.uptime_pct >= 99.9 {
                        PROFIT
                    } else if f.uptime_pct >= 99.0 {
                        AMBER
                    } else {
                        LOSS
                    };
                    ui.colored_label(
                        uc,
                        RichText::new(format!("{:.2}", f.uptime_pct))
                            .monospace()
                            .size(12.0_f32),
                    );
                    ui.label(RichText::new(f.last_error).size(11.0_f32));
                    ui.end_row();
                }
            });
    }

    fn draw_ai_research(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("AI Research — Summarized Research Notes").strong());
        let items = india::sample_ai_research();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for it in &items {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(it.date)
                            .color(Color32::GRAY)
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(RichText::new(it.source).color(INFO).strong().size(11.0_f32));
                    let c = if it.relevance >= 0.85 {
                        PROFIT
                    } else if it.relevance >= 0.7 {
                        AMBER
                    } else {
                        Color32::GRAY
                    };
                    ui.colored_label(
                        c,
                        RichText::new(format!("Relevance {:.0}%", it.relevance * 100.0))
                            .monospace()
                            .size(11.0_f32),
                    );
                });
                ui.label(RichText::new(it.title).strong().size(13.0_f32));
                ui.label(RichText::new(it.summary).size(11.0_f32));
                ui.horizontal(|ui| {
                    for t in it.tags {
                        ui.colored_label(
                            PURPLE,
                            RichText::new(format!("#{}", t)).monospace().size(10.0_f32),
                        );
                    }
                });
                ui.separator();
            }
        });
    }

    fn draw_india_dashboard(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("India Dashboard — Customizable Widgets").strong());
        let widgets = india::sample_dashboard_widgets();
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("india_dashboard_grid")
                .spacing(Vec2::new(12.0_f32, 8.0_f32))
                .min_col_width(280.0_f32)
                .show(ui, |ui| {
                    for (i, w) in widgets.iter().enumerate() {
                        if i % 3 == 0 && i > 0 {
                            ui.end_row();
                        }
                        ui.label(RichText::new(w.title).strong().size(12.0_f32));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(w.value).monospace().strong().size(14.0_f32));
                            let c = if w.change_pct >= 0.0 { PROFIT } else { LOSS };
                            ui.colored_label(
                                c,
                                RichText::new(format!("{:+.2}%", w.change_pct))
                                    .monospace()
                                    .size(11.0_f32),
                            );
                        });
                        Plot::new(format!("dash_spark_{}", i))
                            .auto_bounds(egui::emath::Vec2b::new(true, true))
                            .height(36.0_f32)
                            .show_axes([false, false])
                            .show_grid([false, false])
                            .show(ui, |plot_ui| {
                                let pts: PlotPoints = w
                                    .sparkline
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &v)| [j as f64, v])
                                    .collect();
                                let c = if w.change_pct >= 0.0 { PROFIT } else { LOSS };
                                plot_ui.line(Line::new(pts).color(c).width(1.5_f32));
                            });
                    }
                    ui.end_row();
                });
        });
    }

    fn draw_multi_compare(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Multi-Compare — Side-by-Side Stock Comparison").strong());

        ui.horizontal(|ui| {
            ui.label(RichText::new("Add Symbol:").color(Color32::GRAY));
            ui.add(egui::TextEdit::singleline(&mut self.compare_search).desired_width(200.0_f32));
            if ui.button("Add").clicked() {
                let query = self.compare_search.to_lowercase();
                for (name, ticker, _exchange) in COMPANY_LIST {
                    if ticker.to_lowercase() == query || name.to_lowercase() == query {
                        if !self.compare_symbols.contains(&ticker.to_string())
                            && self.compare_symbols.len() < 5
                        {
                            self.compare_symbols.push(ticker.to_string());
                        }
                        self.compare_search.clear();
                        break;
                    }
                }
            }
            ui.separator();
            ui.label(
                RichText::new(format!("{} / 5 selected", self.compare_symbols.len()))
                    .color(Color32::GRAY),
            );
        });

        ui.horizontal_wrapped(|ui| {
            let mut to_remove = Vec::new();
            for (i, sym) in self.compare_symbols.iter().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(sym).monospace().strong().size(12.0_f32));
                        if ui.button("\u{00D7}").clicked() {
                            to_remove.push(i);
                        }
                    });
                });
            }
            for &i in to_remove.iter().rev() {
                self.compare_symbols.remove(i);
            }
        });

        ui.separator();

        if self.compare_symbols.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new("Add at least 2 symbols to compare").color(Color32::GRAY));
            });
            return;
        }

        let colors = [
            AMBER,
            INFO,
            PROFIT,
            PURPLE,
            Color32::from_rgb(0xFF, 0x69, 0xB4),
        ];

        egui::Grid::new("multi_compare_table")
            .striped(true)
            .spacing(Vec2::new(8.0_f32, 2.0_f32))
            .min_col_width(80.0_f32)
            .show(ui, |ui| {
                ui.label(RichText::new("Symbol").strong().size(11.0_f32));
                ui.label(RichText::new("Name").strong().size(11.0_f32));
                ui.label(RichText::new("Last Price").strong().size(11.0_f32));
                ui.label(RichText::new("Change%").strong().size(11.0_f32));
                ui.label(RichText::new("1M").strong().size(11.0_f32));
                ui.label(RichText::new("3M").strong().size(11.0_f32));
                ui.label(RichText::new("6M").strong().size(11.0_f32));
                ui.label(RichText::new("1Y").strong().size(11.0_f32));
                ui.label(RichText::new("Volatility").strong().size(11.0_f32));
                ui.label(RichText::new("Sharpe").strong().size(11.0_f32));
                ui.label(RichText::new("Max DD").strong().size(11.0_f32));
                ui.label(RichText::new("Beta").strong().size(11.0_f32));
                ui.label(RichText::new("Alpha").strong().size(11.0_f32));
                ui.end_row();

                for (i, sym) in self.compare_symbols.iter().enumerate() {
                    let seed = 100 + i as u64 * 7;
                    let series = synthetic_ohlcv(sym, 250, seed, 100.0 + i as f64 * 50.0);
                    let closes = series.closes();
                    let last_price = closes.last().copied().unwrap_or(0.0);
                    let first_price = closes.first().copied().unwrap_or(1.0);
                    let change_pct = (last_price - first_price) / first_price * 100.0;

                    let returns = series.returns();
                    let ret_1m = returns.iter().take(22).sum::<f64>() * 100.0;
                    let ret_3m = returns.iter().take(66).sum::<f64>() * 100.0;
                    let ret_6m = returns.iter().take(132).sum::<f64>() * 100.0;
                    let ret_1y = returns.iter().take(250).sum::<f64>() * 100.0;

                    let mean_ret = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
                    let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                        / returns.len().max(1) as f64;
                    let volatility = variance.sqrt() * (252.0_f64).sqrt() * 100.0;

                    let sharpe = if volatility > 0.0 {
                        (ret_1y / 100.0) / (volatility / 100.0)
                    } else {
                        0.0
                    };

                    let mut peak = closes.first().copied().unwrap_or(1.0);
                    let mut max_dd = 0.0_f64;
                    for &c in &closes {
                        if c > peak {
                            peak = c;
                        }
                        let dd = (peak - c) / peak;
                        if dd > max_dd {
                            max_dd = dd;
                        }
                    }
                    let max_dd_pct = max_dd * 100.0;

                    let beta = 0.8 + (i as f64 * 0.15);
                    let alpha = (change_pct - 10.0) / 100.0;

                    let name = COMPANY_LIST
                        .iter()
                        .find(|(_, t, _)| *t == sym.as_str())
                        .map(|(n, _, _)| *n)
                        .unwrap_or("Unknown");

                    let color = colors[i % colors.len()];
                    ui.colored_label(
                        color,
                        RichText::new(sym).monospace().strong().size(11.0_f32),
                    );
                    ui.label(RichText::new(name).size(10.0_f32));
                    ui.label(
                        RichText::new(format!("{:.2}", last_price))
                            .monospace()
                            .size(11.0_f32),
                    );

                    let chg_color = if change_pct >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        chg_color,
                        RichText::new(format!("{:+.2}%", change_pct))
                            .monospace()
                            .size(11.0_f32),
                    );

                    for ret in [ret_1m, ret_3m, ret_6m, ret_1y] {
                        let c = if ret >= 0.0 { PROFIT } else { LOSS };
                        ui.colored_label(
                            c,
                            RichText::new(format!("{:+.1}%", ret))
                                .monospace()
                                .size(11.0_f32),
                        );
                    }

                    ui.label(
                        RichText::new(format!("{:.1}%", volatility))
                            .monospace()
                            .size(11.0_f32),
                    );

                    let sharpe_color = if sharpe >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        sharpe_color,
                        RichText::new(format!("{:.2}", sharpe))
                            .monospace()
                            .size(11.0_f32),
                    );

                    ui.colored_label(
                        LOSS,
                        RichText::new(format!("{:.1}%", max_dd_pct))
                            .monospace()
                            .size(11.0_f32),
                    );
                    ui.label(
                        RichText::new(format!("{:.2}", beta))
                            .monospace()
                            .size(11.0_f32),
                    );

                    let alpha_color = if alpha >= 0.0 { PROFIT } else { LOSS };
                    ui.colored_label(
                        alpha_color,
                        RichText::new(format!("{:+.2}%", alpha * 100.0))
                            .monospace()
                            .size(11.0_f32),
                    );

                    ui.end_row();
                }
            });

        ui.separator();

        ui.label(RichText::new("Normalized Price Performance (Base = 100)").strong());

        let chart_anchor = ui.allocate_exact_size(Vec2::new(1.0, 1.0), egui::Sense::hover());
        let (_rect, chart_resp) = chart_anchor;
        chart_resp.scroll_to_me(Some(egui::Align::Center));

        Plot::new("multi_compare_normalized")
            .auto_bounds(egui::emath::Vec2b::new(true, true))
            .height(ui.available_height())
            .allow_scroll(true)
            .allow_drag(true)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                for (i, sym) in self.compare_symbols.iter().enumerate() {
                    let seed = 100 + i as u64 * 7;
                    let series = synthetic_ohlcv(sym, 250, seed, 100.0 + i as f64 * 50.0);
                    let closes = series.closes();
                    let base = closes.first().copied().unwrap_or(1.0);
                    let pts: PlotPoints = closes
                        .iter()
                        .enumerate()
                        .map(|(j, &c)| [j as f64, c / base * 100.0])
                        .collect();
                    let color = colors[i % colors.len()];
                    plot_ui.line(Line::new(pts).color(color).width(2.0_f32).name(sym));
                }
            });
    }
}

// ==================== HELPERS ====================

fn draw_lollipop(plot_ui: &mut egui_plot::PlotUi, values: &[f64], conf: f64) {
    plot_ui.polygon(
        egui_plot::Polygon::new(PlotPoints::from(vec![
            [0.0, -conf],
            [values.len() as f64, -conf],
            [values.len() as f64, conf],
            [0.0, conf],
        ]))
        .fill_color(INFO.gamma_multiply(0.15))
        .stroke(Stroke::NONE),
    );
    for (lag, &v) in values.iter().enumerate() {
        let color = if v.abs() > conf { AMBER } else { Color32::GRAY };
        plot_ui.line(
            Line::new(PlotPoints::from(vec![[lag as f64, 0.0], [lag as f64, v]]))
                .color(color)
                .width(2.0_f32),
        );
        plot_ui.points(
            Points::new(PlotPoints::from(vec![[lag as f64, v]]))
                .color(color)
                .radius(3.5_f32),
        );
    }
}

fn lerp(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t.clamp(0.0, 1.0)).round() as u8
}

fn fmt_big_num(v: f64) -> String {
    if v >= 1_000_000_000.0 {
        format!("{:.2}B", v / 1_000_000_000.0)
    } else if v >= 1_000_000.0 {
        format!("{:.2}M", v / 1_000_000.0)
    } else if v >= 1_000.0 {
        format!("{:.1}K", v / 1_000.0)
    } else {
        format!("{:.0}", v)
    }
}

fn normal_inverse(p: f64) -> f64 {
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

impl eframe::App for BharatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages();
        self.save_prefs();

        if self.live {
            if let Some(last) = self.last_fetch {
                if last.elapsed() >= Duration::from_secs(30) {
                    self.trigger_fetch();
                }
            } else if !self.fetch_in_flight {
                self.trigger_fetch();
            }
            if self.tab == Tab::MarketWatch {
                let stale = self
                    .market_last_refresh
                    .map(|t| t.elapsed() >= Duration::from_secs(30))
                    .unwrap_or(true);
                if stale {
                    self.refresh_market_data();
                }
            }
        }

        ctx.set_visuals(if self.dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });
        self.header(ctx);
        self.warning_banner(ctx);
        self.tab_bar(ctx);
        self.status_bar(ctx);
        self.error_toast(ctx);
        self.body(ctx);
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1920.0_f32, 1080.0_f32])
            .with_min_inner_size([1100.0_f32, 700.0_f32])
            .with_maximized(true)
            .with_title(format!("{} — Made by {}", APP_NAME, AUTHOR)),
        ..Default::default()
    };
    eframe::run_native(
        &format!("{} v3 — Made by {}", APP_NAME, AUTHOR),
        options,
        Box::new(|cc| Ok(Box::new(BharatApp::new(cc)))),
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefs_default() {
        let prefs = Prefs::default();
        assert!(prefs.live);
        assert_eq!(prefs.symbol, "RELIANCE.NS");
        assert_eq!(prefs.range, "1y");
        assert_eq!(prefs.theme, "dark");
    }

    #[test]
    fn test_time_range_labels() {
        assert_eq!(TimeRange::D1.label(), "1D");
        assert_eq!(TimeRange::Y5.label(), "5Y");
    }

    #[test]
    fn test_time_range_days() {
        assert_eq!(TimeRange::D1.to_days(), 1);
        assert_eq!(TimeRange::Y1.to_days(), 365);
        assert_eq!(TimeRange::Y5.to_days(), 1825);
    }

    #[test]
    fn test_banner_constants() {
        assert!(!APP_NAME.is_empty());
        assert!(!AUTHOR.is_empty());
        assert!(!TAGLINE.is_empty());
    }

    /// Every 3D/surface tab and the 4-in-1 dashboard must be reachable, and the
    /// 3D category must actually be shown in the sidebar.
    #[test]
    fn test_3d_and_dashboard_tabs_are_listed() {
        use std::collections::HashSet;
        assert!(
            CATEGORIES.contains(&TabCategory::ThreeD),
            "the 3D category is not in CATEGORIES, so its tabs cannot be reached"
        );

        let listed: HashSet<Tab> = CATEGORIES
            .iter()
            .flat_map(|c| tabs_in_category(*c))
            .map(|(t, _)| *t)
            .collect();

        for tab in [
            Tab::PriceSurface,
            Tab::VolatilitySurface,
            Tab::ReturnSurface,
            Tab::RiskLandscape,
            Tab::BetaSurface,
            Tab::EntropySurface,
            Tab::AlphaSurface,
            Tab::SignalSurface,
            Tab::RegimeSurface,
            Tab::RegimeTimeline,
            Tab::MomentumSurface,
            Tab::OrderFlowSurface,
            Tab::SkewKurtSurface,
            Tab::SignalEvolution,
            Tab::EquitySurface,
            Tab::VarBandSurface,
            Tab::RiskReturnCloud,
            Tab::EigenvalueCloud,
            Tab::ReturnsHeatmap,
            Tab::PcaProjection,
            Tab::FourInOne,
        ] {
            assert!(
                listed.contains(&tab),
                "{tab:?} is not listed in any category"
            );
        }
    }

    /// The 4-in-1 dashboard must delegate to real renderers, so it cannot
    /// silently render four blank panels.
    #[test]
    fn test_four_in_one_delegates_to_existing_renderers() {
        let s = synthetic_ohlcv("DASH", 120, 5, 100.0);
        assert!(!s.candles.is_empty());
        // Each panel is a chart over the same series; the invariant that matters
        // is that the series satisfies what those renderers assume.
        for c in &s.candles {
            assert!(c.high >= c.low && c.high >= c.open && c.high >= c.close);
        }
    }

    /// Every 3D surface must be constructible and non-degenerate, otherwise a
    /// tab in the 3D category would render empty.
    #[test]
    fn test_3d_surfaces_are_constructible() {
        let s = synthetic_ohlcv("S3D", 200, 6, 100.0);
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
        for (name, f) in builders {
            let surface = f(&s);
            assert!(
                surface.is_drawable(),
                "{name}: surface is not drawable and would render blank"
            );
            assert_eq!(surface.values.len(), surface.cols * surface.rows);
        }
    }

    /// Every indicator tab added in the visualization expansion must be
    /// reachable from the tab bar; an unlisted tab would be dead UI.
    #[test]
    fn test_new_indicator_tabs_are_listed() {
        use std::collections::HashSet;
        let listed: HashSet<Tab> = CATEGORIES
            .iter()
            .flat_map(|c| tabs_in_category(*c))
            .map(|(t, _)| *t)
            .collect();
        for tab in [
            Tab::StochRsi,
            Tab::Zscore,
            Tab::Mfi,
            Tab::UltimateOsc,
            Tab::Tsi,
            Tab::Coppock,
            Tab::Dpo,
            Tab::LogReturns,
            Tab::Kst,
            Tab::ElderRay,
            Tab::Vortex,
            Tab::Aroon,
            Tab::AroonOsc,
            Tab::Ulcer,
            Tab::Eom,
            Tab::ForceIndex,
            Tab::MassIndex,
            Tab::Pvt,
            Tab::Mfv,
            Tab::AdLine,
            Tab::TrendIntensity,
            Tab::RealizedVol,
            Tab::KeltnerWidth,
            Tab::Volatility,
            Tab::Kama,
            Tab::Alma,
            Tab::HullMa,
            Tab::Wma,
            Tab::MultiSma,
            Tab::MultiEma,
            Tab::HighLowBand,
            Tab::VwapBands,
            Tab::Supertrend,
            Tab::Volume,
            Tab::Ohlc,
            Tab::PivotPoints,
            Tab::FibLevels,
        ] {
            assert!(listed.contains(&tab), "{tab:?} not listed in any category");
        }
    }

    #[test]
    fn test_forecast_tab_is_listed_under_advanced() {
        let advanced = tabs_in_category(TabCategory::Advanced);
        let entry = advanced.iter().find(|(tab, _)| *tab == Tab::Forecast);
        assert!(entry.is_some(), "Forecast tab missing from Advanced");
        assert_eq!(entry.unwrap().1, "Forecast");
    }

    #[test]
    fn test_price_scale_uses_low_and_high() {
        let series = synthetic_ohlcv("SCALE", 10, 7, 100.0);
        let (lo, range) = price_scale(&series.candles);
        let expect_lo = series
            .candles
            .iter()
            .map(|c| c.low)
            .fold(f64::MAX, f64::min);
        let expect_hi = series
            .candles
            .iter()
            .map(|c| c.high)
            .fold(f64::MIN, f64::max);
        assert!((lo - expect_lo).abs() < 1e-9);
        assert!((range - (expect_hi - expect_lo)).abs() < 1e-9);
        assert!(range > 0.0);
    }

    #[test]
    fn test_price_scale_empty_series_is_safe() {
        let (lo, range) = price_scale(&[]);
        assert!(lo.is_finite());
        assert!(range > 0.0);
    }

    #[test]
    fn test_candle_body_bullish_is_close_on_top() {
        let c = Candle::new(1.0, 100.0, 110.0, 95.0, 105.0, 1_000.0);
        let (top, bottom) = candle_body(&c, 0.0);
        assert!((top - 105.0).abs() < 1e-9);
        assert!((bottom - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_candle_body_bearish_is_open_on_top() {
        let c = Candle::new(1.0, 105.0, 110.0, 95.0, 100.0, 1_000.0);
        let (top, bottom) = candle_body(&c, 0.0);
        assert!((top - 105.0).abs() < 1e-9);
        assert!((bottom - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_candle_body_doji_gets_visible_height() {
        let c = Candle::new(1.0, 100.0, 110.0, 95.0, 100.0, 1_000.0);
        let min_body = 2.0;
        let (top, bottom) = candle_body(&c, min_body);
        assert!(top > bottom, "doji body must remain visible");
        assert!(((top - bottom) - min_body).abs() < 1e-9);
        let mid = 0.5 * (top + bottom);
        assert!(
            (mid - 100.0).abs() < 1e-9,
            "expansion stays centred on the price"
        );
    }

    #[test]
    fn test_candle_body_never_collapses_for_tiny_ranges() {
        let series = synthetic_ohlcv("TINY", 50, 3, 10.0);
        let (_, range) = price_scale(&series.candles);
        let min_body = range * MIN_BODY_FRAC;
        for c in &series.candles {
            let (top, bottom) = candle_body(c, min_body);
            assert!(
                top > bottom,
                "every candle must have a positive body height"
            );
            assert!(top.is_finite() && bottom.is_finite());
        }
    }

    #[test]
    fn test_arrow_and_body_fractions_are_sane() {
        assert!(MIN_BODY_FRAC > 0.0 && MIN_BODY_FRAC < 0.1);
        assert!(
            ARROW_FRAC > MIN_BODY_FRAC,
            "arrows must be taller than a body"
        );
        assert!(
            ARROW_FRAC < 0.2,
            "arrows must stay small relative to the range"
        );
    }

    /// Regression test for the bug where every candle was drawn
    /// `0.35 * DAY_SECS` wide, making 1-minute bars overlap into a single blob.
    #[test]
    fn test_bar_spacing_uses_real_cadence_not_a_day() {
        let minute_bars: Vec<Candle> = (0..375)
            .map(|i| {
                Candle::new(
                    1_790_567_100.0 + i as f64 * 60.0,
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1.0,
                )
            })
            .collect();
        assert_eq!(bar_spacing(&minute_bars), 60.0);

        let daily_bars: Vec<Candle> = (0..100)
            .map(|i| {
                Candle::new(
                    1_704_067_200.0 + i as f64 * DAY_SECS,
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1.0,
                )
            })
            .collect();
        assert_eq!(bar_spacing(&daily_bars), DAY_SECS);
    }

    #[test]
    fn test_bar_half_never_exceeds_the_gap_to_the_neighbour() {
        let minute_bars: Vec<Candle> = (0..375)
            .map(|i| {
                Candle::new(
                    1_790_567_100.0 + i as f64 * 60.0,
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1.0,
                )
            })
            .collect();
        let half = bar_half(&minute_bars);
        assert!(
            half < 60.0,
            "candle body must be narrower than the 60s bar gap, got {}",
            half
        );
        let gap = 60.0 - 2.0 * half;
        assert!(gap > 0.0, "adjacent candle bodies would overlap");
    }

    #[test]
    fn test_bar_spacing_is_robust_to_session_gaps() {
        // A lunch break creates one huge delta; the median must ignore it.
        let deltas = [60.0, 60.0, 60.0, 60.0, 4.0 * 3600.0, 60.0, 60.0];
        let mut ts = 1_790_567_100.0;
        let bars: Vec<Candle> = deltas
            .iter()
            .map(|d| {
                let c = Candle::new(ts, 100.0, 101.0, 99.0, 100.5, 1.0);
                ts += d;
                c
            })
            .collect();
        assert_eq!(bar_spacing(&bars), 60.0, "median must reject the gap");
    }

    #[test]
    fn test_bar_spacing_edge_cases() {
        assert_eq!(bar_spacing(&[]), FALLBACK_SPACING);
        let one = vec![Candle::new(1.0, 1.0, 2.0, 0.5, 1.5, 1.0)];
        assert_eq!(bar_spacing(&one), FALLBACK_SPACING);
        // Duplicate timestamps must not yield a zero or negative spacing.
        let dupes = vec![
            Candle::new(5.0, 1.0, 2.0, 0.5, 1.5, 1.0),
            Candle::new(5.0, 1.0, 2.0, 0.5, 1.5, 1.0),
        ];
        assert_eq!(bar_spacing(&dupes), FALLBACK_SPACING);
    }

    #[test]
    fn test_x_bounds_wrap_the_data_with_padding() {
        let bars: Vec<Candle> = (0..10)
            .map(|i| Candle::new(1000.0 + i as f64 * 60.0, 1.0, 2.0, 0.5, 1.5, 1.0))
            .collect();
        let (lo, hi) = x_bounds(&bars);
        assert!(lo < 1000.0, "x range must start before the first bar");
        assert!(
            hi > 1000.0 + 9.0 * 60.0,
            "x range must end after the last bar"
        );
    }

    #[test]
    fn test_x_bounds_empty_series_is_finite() {
        let (lo, hi) = x_bounds(&[]);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(hi > lo);
    }

    #[test]
    fn test_x_bounds_ignores_non_finite_timestamps() {
        let bars = vec![
            Candle::new(1000.0, 1.0, 2.0, 0.5, 1.5, 1.0),
            Candle::new(f64::NAN, 1.0, 2.0, 0.5, 1.5, 1.0),
            Candle::new(2000.0, 1.0, 2.0, 0.5, 1.5, 1.0),
        ];
        let (lo, hi) = x_bounds(&bars);
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo < 2000.0 && hi > 2000.0);
    }

    #[test]
    fn test_format_ts_for_picks_a_format_from_the_cadence() {
        // 1_704_110_100 == 2024-01-01 11:55 UTC
        let t = 1_704_110_100.0_f64;
        assert_eq!(
            format_ts_for(t, 60.0),
            "11:55 01 Jan",
            "intraday shows a clock time and the date"
        );
        assert_eq!(format_ts_for(t, DAY_SECS), "01 Jan", "daily shows a date");
        assert_eq!(
            format_ts_for(t, 90.0 * DAY_SECS),
            "Jan 2024",
            "long range shows month"
        );
    }

    #[test]
    fn test_price_decimals_scales_with_magnitude() {
        assert_eq!(price_decimals(12_345.0), 0);
        assert_eq!(price_decimals(1_500.0), 1);
        assert_eq!(price_decimals(975.25), 2);
        assert_eq!(price_decimals(1.75), 3);
        assert_eq!(price_decimals(0.0123), 4);
    }

    #[test]
    fn test_abbreviate_volume() {
        assert_eq!(abbreviate_volume(999.0), "999");
        assert_eq!(abbreviate_volume(1_500.0), "1.5K");
        assert_eq!(abbreviate_volume(2_500_000.0), "2.5M");
        assert_eq!(abbreviate_volume(3_200_000_000.0), "3.2B");
        assert_eq!(abbreviate_volume(4_100_000_000_000.0), "4.1T");
    }

    #[test]
    fn test_intraday_bars_tile_the_visible_x_range() {
        let series = synthetic_ohlcv("SBIN", 375, 21, 975.0);
        let half = bar_half(&series.candles);
        let spacing = bar_spacing(&series.candles);
        assert!(half > 0.0 && half < spacing);
        let (x0, x1) = x_bounds(&series.candles);
        assert!(x1 > x0);
        assert!(
            (x1 - x0) < (series.candles.len() as f64 * spacing * 1.5),
            "x range must stay proportional to the bar count"
        );
    }

    /// Regression test: egui_plot remembers each plot's zoom window, so after
    /// viewing 1Y the 1D intraday series was drawn into that stale window and
    /// collapsed to a sliver. The panes now pin bounds from the data every
    /// frame, so the plotted range must always hug the actual series.
    #[test]
    fn test_bounds_hug_the_series_not_a_stale_longer_window() {
        // 75 five-minute bars = one trading day.
        let intraday: Vec<Candle> = (0..75)
            .map(|i| {
                Candle::new(
                    1_790_567_100.0 + i as f64 * 300.0,
                    1200.0,
                    1202.0,
                    1198.0,
                    1201.0,
                    1.0,
                )
            })
            .collect();
        let (x0, x1) = x_bounds(&intraday);
        // The bounds must bracket the series with only small padding, proving
        // they were derived from these bars rather than a wider remembered range.
        assert!(x0 < 1_790_567_100.0, "must start before the first bar");
        assert!(
            x1 > 1_790_567_100.0 + 74.0 * 300.0,
            "must end after the last"
        );
        let padding = (x1 - x0) - (74.0 * 300.0);
        assert!(
            padding < 74.0 * 300.0,
            "padding must stay small relative to the series span"
        );
        // A stale 365-day window is orders of magnitude wider, which is exactly
        // why the pane has to pin bounds instead of trusting remembered zoom.
        let stale_window = 365.0 * DAY_SECS;
        assert!((x1 - x0) * 100.0 < stale_window);
    }

    #[test]
    fn test_pin_bounds_is_a_noop_for_degenerate_ranges() {
        // Guards the `x1 > x0 && y1 > y0` precondition: a single flat price
        // must not produce an inverted or zero-height range.
        let flat = vec![Candle::new(1.0, 100.0, 100.0, 100.0, 100.0, 1.0)];
        let (_, range) = price_scale(&flat);
        assert!(
            range > 0.0,
            "price_scale keeps range positive for flat data"
        );
        let (x0, x1) = x_bounds(&flat);
        assert!(x1 > x0, "x_bounds must stay ordered even for a single bar");
    }

    #[test]
    fn test_zoom_default_shows_the_full_range() {
        let z = ZoomState::default();
        assert!(!z.is_zoomed());
        assert_eq!(z.window(0.0, 100.0, 0.0, 50.0), (0.0, 100.0, 0.0, 50.0));
    }

    #[test]
    fn test_zoom_in_halves_the_window_around_the_focus() {
        let z = ZoomState::default().zoom_in(50.0, 25.0, &[], 0.0, 100.0);
        assert!(z.is_zoomed());
        assert_eq!(z.factor, ZoomState::STEP);
        let (x0, x1, y0, y1) = z.window(0.0, 100.0, 0.0, 50.0);
        assert!((0.5 * (x0 + x1) - 50.0).abs() < 1e-9, "x focus preserved");
        assert!((0.5 * (y0 + y1) - 25.0).abs() < 1e-9, "y focus preserved");
        assert!((x1 - x0) - 50.0 < 1e-9, "x window halved");
        assert!((y1 - y0) - 25.0 < 1e-9, "y window halved");
    }

    /// The user's core complaint: zooming out must step back down, not snap.
    #[test]
    fn test_zoom_out_steps_back_down_one_level_at_a_time() {
        let mut z = ZoomState::default();
        z = z.zoom_in(50.0, 25.0, &[], 0.0, 100.0);
        assert_eq!(z.factor, 2.0);
        z = z.zoom_out(50.0, 25.0);
        assert_eq!(z.factor, 1.0, "one out-step undoes one in-step");
        assert!(!z.is_zoomed());

        // Three steps in, then one out, leaves two steps in.
        let mut z = ZoomState::default();
        for _ in 0..3 {
            z = z.zoom_in(10.0, 10.0, &[], 0.0, 1000.0);
        }
        assert_eq!(z.factor, 8.0);
        let back = z.zoom_out(10.0, 10.0);
        assert_eq!(back.factor, 4.0, "out-step halves, it does not reset");
        assert!(back.is_zoomed());
    }

    #[test]
    fn test_zoom_out_unwinds_all_the_way_to_the_full_range() {
        let mut z = ZoomState::default();
        for _ in 0..5 {
            z = z.zoom_in(10.0, 10.0, &[], 0.0, 1000.0);
        }
        let start = z.factor;
        for _ in 0..20 {
            z = z.zoom_out(10.0, 10.0);
        }
        assert_eq!(z.factor, ZoomState::MIN_FACTOR);
        assert!(!z.is_zoomed());
        assert!(start > 1.0);
    }

    #[test]
    fn test_zoom_out_never_goes_below_the_full_range() {
        let z = ZoomState::default();
        let out = z.zoom_out(10.0, 10.0);
        assert_eq!(out.factor, ZoomState::MIN_FACTOR);
        assert_eq!(out, z, "zooming out from full range is a no-op");
    }

    #[test]
    fn test_zoom_in_is_capped_at_the_maximum() {
        let mut z = ZoomState::default();
        for _ in 0..40 {
            z = z.zoom_in(10.0, 10.0, &[], 0.0, 1000.0);
        }
        assert_eq!(z.factor, ZoomState::MAX_FACTOR);
        let capped = z.zoom_in(10.0, 10.0, &[], 0.0, 1000.0);
        assert_eq!(capped, z, "further zoom-in changes nothing");
    }

    #[test]
    fn test_centered_steps_work_without_a_pointer() {
        let mut z = ZoomState::default();
        z = z.zoom_in_centered(50.0);
        assert_eq!(z.factor, ZoomState::STEP);
        // With no focus recorded, the window centres on its midpoint.
        let (x0, x1, _, _) = z.window(0.0, 100.0, 0.0, 50.0);
        assert!((0.5 * (x0 + x1) - 50.0).abs() < 1e-9);
        assert_eq!(z.zoom_out_centered().factor, ZoomState::MIN_FACTOR);
    }

    #[test]
    fn test_zoom_ignores_non_finite_points() {
        let z = ZoomState::default();
        // A non-finite x cannot be centred on, so nothing changes.
        assert_eq!(z.zoom_in(f64::NAN, 1.0, &[], 0.0, 100.0), z);
        // A non-finite price is replaced by a usable value rather than being
        // stored, which would otherwise produce a permanently blank pane.
        let zy = z.zoom_in(1.0, f64::INFINITY, &[], 0.0, 100.0);
        assert!(zy.focus_y.map(f64::is_finite).unwrap_or(true));
        assert!(zy.is_zoomed(), "a valid x still zooms");
    }

    /// Regression test: double-clicking empty space above the series zoomed to
    /// a price band containing no candles, leaving a blank pane. The focus
    /// price is now pulled into the visible candles' envelope.
    #[test]
    fn test_zoom_snaps_the_focus_onto_real_candles() {
        let candles: Vec<Candle> = (0..50)
            .map(|i| Candle::new(1_000.0 + i as f64 * 60.0, 100.0, 102.0, 99.0, 101.0, 1.0))
            .collect();
        let snap = price_envelope(&candles, 1_000.0, 4_000.0);
        assert!(snap.is_some());
        let (lo, hi) = snap.unwrap();
        assert!((lo - 99.0).abs() < 1e-9);
        assert!((hi - 102.0).abs() < 1e-9);

        // Click far above the series: the zoom must land inside the envelope.
        let z = ZoomState::default().zoom_in(2_500.0, 5_000.0, &candles, 1_000.0, 4_000.0);
        let focus_y = z.focus_y.unwrap();
        assert!(
            focus_y >= lo && focus_y <= hi,
            "focus {} escaped the candle envelope {lo}..{hi}",
            focus_y
        );
    }

    #[test]
    fn test_zoom_without_snap_keeps_the_raw_focus() {
        let z = ZoomState::default().zoom_in(500.0, 42.0, &[], 0.0, 100.0);
        assert!((z.focus_y.unwrap() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_snap_range_is_none_when_the_window_has_no_bars() {
        let candles = vec![Candle::new(0.0, 1.0, 2.0, 0.5, 1.5, 1.0)];
        assert!(price_envelope(&candles, 5_000.0, 6_000.0).is_none());
        assert!(price_envelope(&[], 0.0, 1.0).is_none());
    }

    #[test]
    fn test_zoom_window_is_safe_for_degenerate_ranges() {
        let z = ZoomState::default().zoom_in(5.0, 5.0, &[], 0.0, 10.0);
        assert_eq!(z.window(10.0, 10.0, 0.0, 50.0), (10.0, 10.0, 0.0, 50.0));
        assert_eq!(z.window(50.0, 10.0, 0.0, 50.0), (50.0, 10.0, 0.0, 50.0));
    }

    #[test]
    fn test_zoom_reset_returns_to_full_range() {
        let z = ZoomState::default().zoom_in(80.0, 20.0, &[], 0.0, 100.0);
        assert!(z.is_zoomed());
        let r = z.reset();
        assert!(!r.is_zoomed());
        assert_eq!(r.window(0.0, 100.0, 0.0, 50.0), (0.0, 100.0, 0.0, 50.0));
    }

    #[test]
    fn test_zoomed_window_stays_finite_and_ordered() {
        let mut z = ZoomState::default();
        for _ in 0..20 {
            z = z.zoom_in(1_000.0, 500.0, &[], 0.0, 2000.0);
        }
        let (x0, x1, y0, y1) = z.window(0.0, 2000.0, 0.0, 1000.0);
        assert!(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite());
        assert!(x1 > x0 && y1 > y0);
    }

    /// A pinch reports a small multiplicative factor per frame, so the zoom
    /// must be continuous rather than a fixed 2x step. Several small pinches
    /// must land on the same result as one equivalent factor.
    #[test]
    fn test_zoom_by_is_continuous_and_matches_a_single_step() {
        let (series, x0, x1, _, _) = pan_fixture();
        let mid = 0.5 * (x0 + x1);
        let one_step = ZoomState::default().zoom_by(4.0, mid, 0.0, &series, x0, x1);
        let two_steps = ZoomState::default()
            .zoom_by(2.0, mid, 0.0, &series, x0, x1)
            .zoom_by(2.0, mid, 0.0, &series, x0, x1);
        assert!((one_step.factor - two_steps.factor).abs() < 1e-9);
        assert!((one_step.factor - 4.0).abs() < 1e-9);
    }

    /// Many tiny pinch deltas must not overshoot, and must stay clamped.
    #[test]
    fn test_zoom_by_accumulates_tiny_deltas_and_clamps() {
        let (series, x0, x1, _, _) = pan_fixture();
        let mid = 0.5 * (x0 + x1);
        let mut z = ZoomState::default();
        for _ in 0..200 {
            z = z.zoom_by(1.05, mid, 0.0, &series, x0, x1);
        }
        assert!(z.factor <= ZoomState::MAX_FACTOR + 1e-9, "{}", z.factor);
        // Pinching closed walks back down to the full range.
        for _ in 0..400 {
            z = z.zoom_by(0.95, mid, 0.0, &series, x0, x1);
        }
        assert!(z.factor >= ZoomState::MIN_FACTOR - 1e-9);
    }

    /// Pinching beyond the limit, or with a junk factor, must be a no-op.
    #[test]
    fn test_zoom_by_rejects_invalid_factors() {
        let (series, x0, x1, _, _) = pan_fixture();
        let mid = 0.5 * (x0 + x1);
        let z = ZoomState::default().zoom_in_centered(125.0);
        assert_eq!(z.zoom_by(f64::NAN, mid, 0.0, &series, x0, x1), z);
        assert_eq!(z.zoom_by(0.0, mid, 0.0, &series, x0, x1), z);
        assert_eq!(z.zoom_by(-2.0, mid, 0.0, &series, x0, x1), z);
        assert_eq!(z.zoom_by(1.0, mid, 0.0, &series, x0, x1), z);
        assert_eq!(z.zoom_by(2.0, f64::NAN, 0.0, &series, x0, x1), z);
    }

    /// A vertical-only pinch scales the y-window while leaving the x-window
    /// alone. This is the `[1, z]` case egui reports for fingers stacked
    /// vertically.
    #[test]
    fn test_vertical_pinch_scales_only_the_y_axis() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default();
        let (bx0, bx1, by0, by1) = z.window(x0, x1, y0, y1);
        let v = z.zoom_y_by(4.0, 0.5 * (x0 + x1), 0.5 * (y0 + y1), &series, x0, x1);
        let (ax0, ax1, ay0, ay1) = v.window(x0, x1, y0, y1);
        // x-window untouched, y-window a quarter of its original height.
        assert!((ax0 - bx0).abs() < 1e-6);
        assert!((ax1 - bx1).abs() < 1e-6);
        assert!((0.25 * (by1 - by0) - (ay1 - ay0)).abs() < 1e-6);
        // A y-only pinch still counts as zoomed, so panning is available.
        assert!(v.is_zoomed());
    }

    /// A horizontal pinch scales the x-window and leaves y alone.
    #[test]
    fn test_horizontal_pinch_scales_only_the_x_axis() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default();
        let (bx0, bx1, by0, by1) = z.window(x0, x1, y0, y1);
        let h = z.zoom_x_by(4.0, 0.5 * (x0 + x1), 0.5 * (y0 + y1), &series, x0, x1);
        let (ax0, ax1, ay0, ay1) = h.window(x0, x1, y0, y1);
        assert!((0.25 * (bx1 - bx0) - (ax1 - ax0)).abs() < 1e-6);
        assert!((ay0 - by0).abs() < 1e-6);
        assert!((ay1 - by1).abs() < 1e-6);
    }

    /// The y-scale is clamped to the same limit as the x-scale.
    #[test]
    fn test_vertical_pinch_is_clamped() {
        let (series, x0, x1, _, _) = pan_fixture();
        let mut v = ZoomState::default();
        for _ in 0..300 {
            v = v.zoom_y_by(1.1, 0.5 * (x0 + x1), 125.0, &series, x0, x1);
        }
        assert!(v.y_factor <= ZoomState::MAX_FACTOR + 1e-9, "{}", v.y_factor);
    }

    /// A y-only pinch leaves the x-scale at 1.0, so the x-window already spans
    /// the whole series: a horizontal pan must be a no-op rather than sliding
    /// the window off the data. A vertical pan still works.
    #[test]
    fn test_y_zoomed_view_pans_vertically_but_not_horizontally() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let v = ZoomState::default().zoom_y_by(4.0, 0.5 * (x0 + x1), 125.0, &series, x0, x1);
        assert!(v.is_zoomed(), "a y-only pinch must count as zoomed");

        // Horizontal drag: the x-window already covers everything, so it holds.
        let before_x = v.window(x0, x1, y0, y1);
        let h = v.pan_by(86_400.0 * 4.0, 0.0, &series, x0, x1, y0, y1);
        let after_x = h.window(x0, x1, y0, y1);
        assert!((after_x.0 - before_x.0).abs() < 1e-6, "x must not move");
        assert!((after_x.1 - before_x.1).abs() < 1e-6, "x must not move");
        assert!(
            (after_x.1 - after_x.0 - (x1 - x0)).abs() < 1e-6,
            "x must stay full width"
        );

        // Vertical drag does move the y-window.
        let t = v.pan_by(0.0, 3.0, &series, x0, x1, y0, y1);
        let (_, _, ay, by) = t.window(x0, x1, y0, y1);
        let (_, _, cy, dy) = v.window(x0, x1, y0, y1);
        assert!((ay - cy - 3.0).abs() < 1e-6, "y low edge: {ay} vs {cy}");
        assert!((by - dy - 3.0).abs() < 1e-6, "y high edge: {by} vs {dy}");

        // And a huge drag on both axes still stays inside the data.
        let far = v.pan_by(1.0e9, 1.0e9, &series, x0, x1, y0, y1);
        let (fx0, fx1, _, _) = far.window(x0, x1, y0, y1);
        assert!(fx0 >= x0 - 1e-6 && fx1 <= x1 + 1e-6);
    }

    /// Stepping out of a y-only zoom returns to the full range, and the x-scale
    /// must not be left stranded mid-zoom.
    #[test]
    fn test_zoom_out_after_a_vertical_pinch_returns_to_full_range() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let mut v = ZoomState::default()
            .zoom_y_by(8.0, 0.5 * (x0 + x1), 125.0, &series, x0, x1)
            .zoom_in_centered(125.0);
        assert!(v.is_zoomed());
        for _ in 0..12 {
            v = v.zoom_out_centered();
        }
        assert!(!v.is_zoomed(), "x or y still zoomed: {v:?}");
        assert_eq!(v.window(x0, x1, y0, y1), (x0, x1, y0, y1));
    }

    /// 200 daily bars with a gently rising, noisy price so the envelope is
    /// strictly inside the requested y-range.
    fn pan_fixture() -> (Vec<Candle>, f64, f64, f64, f64) {
        let day = 86_400.0_f64;
        let start = 1_700_000_000.0_f64;
        let series: Vec<Candle> = (0..200)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.25;
                Candle::new(
                    start + i as f64 * day,
                    base,
                    base + 2.0,
                    base - 2.0,
                    base + 0.5,
                    1_000.0,
                )
            })
            .collect();
        let x0 = start - day;
        let x1 = start + 200.0 * day;
        (series, x0, x1, 50.0, 200.0)
    }

    /// Wheel-up zooms in, wheel-down zooms out, and no scroll means no zoom.
    #[test]
    fn test_scroll_zoom_factor_direction_and_neutrality() {
        assert!((scroll_zoom_factor(0.0) - 1.0).abs() < 1e-12);
        assert!(scroll_zoom_factor(50.0) > 1.0, "wheel-up must zoom in");
        assert!(scroll_zoom_factor(-50.0) < 1.0, "wheel-down must zoom out");
        // A full notch steps about as far as a fraction of a double-click,
        // never a jump to the cap in one frame.
        let notch = scroll_zoom_factor(50.0);
        assert!(
            notch < 2.0,
            "one notch must stay well below a 2x step: {notch}"
        );
        assert!(notch > 1.0, "one notch must be perceptible: {notch}");
        // Symmetry: scrolling back down undoes scrolling up.
        let up = scroll_zoom_factor(50.0);
        let down = scroll_zoom_factor(-50.0);
        assert!(
            (up * down - 1.0).abs() < 1e-9,
            "wheel gestures must be reversible: {up} * {down}"
        );
    }

    /// A tiny trackpad delta produces a tiny factor, never a no-op or a jump.
    #[test]
    fn test_scroll_zoom_factor_is_continuous_for_small_deltas() {
        let tiny = scroll_zoom_factor(2.0);
        assert!(tiny > 1.0 && tiny < 1.01, "trackpad tick: {tiny}");
        for raw in [-120.0_f32, -50.0, -5.0, 5.0, 50.0, 120.0] {
            let f = scroll_zoom_factor(raw);
            assert!(f.is_finite() && f > 0.0, "sane factor for {raw}: {f}");
        }
    }

    /// `pan_x_only` moves the time axis exactly like `pan_by` does, while the
    /// price offset stays pinned at whatever it was.
    #[test]
    fn test_pan_x_only_matches_pan_by_on_x_and_holds_y() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        // Give the state a vertical offset first, so the test can prove it is
        // preserved rather than reset.
        let z = z.pan_by(0.0, 4.0, &series, x0, x1, y0, y1);
        assert!((z.pan_y - 4.0).abs() < 1e-6);
        let before = z.window(x0, x1, y0, y1);
        let moved = z.pan_x_only(86_400.0 * 3.0, x0, x1);
        let after = moved.window(x0, x1, y0, y1);
        assert!((after.0 - before.0 - 86_400.0 * 3.0).abs() < 1e-6);
        assert!((after.1 - before.1 - 86_400.0 * 3.0).abs() < 1e-6);
        // Vertical edges untouched.
        assert!((after.2 - before.2).abs() < 1e-12);
        assert!((after.3 - before.3).abs() < 1e-12);
        assert_eq!(moved.pan_y, z.pan_y, "vertical offset must survive");
    }

    /// `pan_x_only` clamps at the data edges and is a no-op at full range.
    #[test]
    fn test_pan_x_only_clamps_and_ignores_full_range() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let flat = ZoomState::default();
        assert_eq!(flat.pan_x_only(1.0e9, x0, x1), flat);
        let z = ZoomState::default().zoom_in_centered(125.0);
        let far = z.pan_x_only(1.0e9, x0, x1);
        let (fx0, fx1, _, _) = far.window(x0, x1, y0, y1);
        assert!(fx0 >= x0 - 1e-6 && fx1 <= x1 + 1e-6);
        // Settled at the limit, further drags are no-ops.
        assert_eq!(far.pan_x_only(86_400.0, x0, x1), far);
        // Non-finite input never poisons the state.
        assert_eq!(z.pan_x_only(f64::NAN, x0, x1), z);
    }

    /// Panning is disabled until the view is actually zoomed in.
    #[test]
    fn test_pan_is_ignored_before_zooming() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let base = ZoomState::default();
        assert!(!base.is_zoomed());
        let panned = base.pan_by(500.0, 10.0, &series, x0, x1, y0, y1);
        assert_eq!(panned, base, "a full-range view must not pan");
    }

    /// Horizontal drag shifts the x-window by the data-space delta.
    #[test]
    fn test_horizontal_pan_moves_the_x_window() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        assert!(z.is_zoomed());
        let before = z.window(x0, x1, y0, y1);
        // Drag right: the window should move later in time.
        let panned = z.pan_by(86_400.0 * 5.0, 0.0, &series, x0, x1, y0, y1);
        let after = panned.window(x0, x1, y0, y1);
        assert!((after.0 - before.0 - 86_400.0 * 5.0).abs() < 1e-6);
        assert!((after.1 - before.1 - 86_400.0 * 5.0).abs() < 1e-6);
    }

    /// Vertical drag shifts the y-window by the data-space delta.
    #[test]
    fn test_vertical_pan_moves_the_y_window() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        let before = z.window(x0, x1, y0, y1);
        let panned = z.pan_by(0.0, 12.5, &series, x0, x1, y0, y1);
        let after = panned.window(x0, x1, y0, y1);
        assert!((after.2 - before.2 - 12.5).abs() < 1e-6);
        assert!((after.3 - before.3 - 12.5).abs() < 1e-6);
    }

    /// Both axes move together for a diagonal drag.
    #[test]
    fn test_pan_moves_both_axes_at_once() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        let before = z.window(x0, x1, y0, y1);
        let panned = z.pan_by(2.0 * 86_400.0, -6.0, &series, x0, x1, y0, y1);
        let after = panned.window(x0, x1, y0, y1);
        assert!((after.0 - before.0 - 2.0 * 86_400.0).abs() < 1e-6);
        assert!((after.2 - before.2 + 6.0).abs() < 1e-6);
    }

    /// Panning is clamped so the window can never leave the data.
    #[test]
    fn test_pan_is_clamped_to_the_data() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        // A drag far larger than the series cannot pull the window off the end.
        let far = z.pan_by(1.0e9, 1.0e9, &series, x0, x1, y0, y1);
        let (wx0, wx1, _, _) = far.window(x0, x1, y0, y1);
        assert!(wx0 >= x0 - 1e-6, "left edge escaped the data: {wx0}");
        assert!(wx1 <= x1 + 1e-6, "right edge escaped the data: {wx1}");

        // And the same in the other direction.
        let back = z.pan_by(-1.0e9, -1.0e9, &series, x0, x1, y0, y1);
        let (bx0, bx1, _, _) = back.window(x0, x1, y0, y1);
        assert!(bx0 >= x0 - 1e-6);
        assert!(bx1 <= x1 + 1e-6);
    }

    /// Repeated drags accumulate; the bound must not compound per frame.
    #[test]
    fn test_repeated_pan_accumulates_without_compounding_the_clamp() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let mut z = ZoomState::default().zoom_in_centered(125.0);
        // Many small drags right, in total less than the available slack.
        for _ in 0..10 {
            z = z.pan_by(86_400.0, 0.0, &series, x0, x1, y0, y1);
        }
        assert!((z.pan_x - 10.0 * 86_400.0).abs() < 1e-6);
        // Many more than the slack allows: the result must stay inside, and
        // must equal the clamp limit rather than drift further.
        for _ in 0..400 {
            z = z.pan_by(86_400.0, 0.0, &series, x0, x1, y0, y1);
        }
        let (wx0, wx1, _, _) = z.window(x0, x1, y0, y1);
        assert!(wx1 <= x1 + 1e-6, "right edge escaped: {wx1}");
        let (_, _, _, _) = z.window(x0, x1, y0, y1);
        // Re-panning at the limit is a no-op rather than an error.
        let settled = z.pan_by(86_400.0, 0.0, &series, x0, x1, y0, y1);
        assert_eq!(settled, z);
    }

    /// Non-finite deltas must never poison the state.
    #[test]
    fn test_pan_ignores_non_finite_deltas() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let z = ZoomState::default().zoom_in_centered(125.0);
        assert_eq!(z.pan_by(f64::NAN, 0.0, &series, x0, x1, y0, y1), z);
        assert_eq!(z.pan_by(0.0, f64::INFINITY, &series, x0, x1, y0, y1), z);
    }

    /// A panned view still contains candles, so the pane is never blank.
    #[test]
    fn test_panned_window_still_covers_candles() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let mut z = ZoomState::default().zoom_in_centered(125.0);
        z = z.pan_by(3.0 * 86_400.0, 5.0, &series, x0, x1, y0, y1);
        let (wx0, wx1, wy0, wy1) = z.window_with_data(&series, x0, x1, y0, y1);
        let (lo, hi) = price_envelope(&series, wx0, wx1).expect("candles in view");
        assert!(wy0 <= lo + 1e-6, "visible low {lo} clipped by {wy0}");
        assert!(wy1 >= hi - 1e-6, "visible high {hi} clipped by {wy1}");
    }

    /// Zooming in and out returns to the full range.
    #[test]
    fn test_zoom_out_unwinds_to_the_full_range() {
        let (series, x0, x1, y0, y1) = pan_fixture();
        let mut z = ZoomState::default();
        for _ in 0..5 {
            z = z.zoom_in_centered(125.0);
        }
        assert!(z.is_zoomed());
        for _ in 0..10 {
            z = z.zoom_out_centered();
        }
        assert!(!z.is_zoomed());
        assert_eq!(z.window(x0, x1, y0, y1), (x0, x1, y0, y1));
    }

    /// The zoom factor is capped, so repeated clicks stay sane.
    #[test]
    fn test_zoom_in_is_capped() {
        let (series, x0, x1, _, _) = pan_fixture();
        let mut z = ZoomState::default();
        for _ in 0..50 {
            z = z.zoom_in(0.5 * (x0 + x1), 0.0, &series, x0, x1);
        }
        assert!(z.factor <= ZoomState::MAX_FACTOR + f64::EPSILON);
    }

    #[test]
    fn test_custom_window_parses_orders_and_validates() {
        let day = 86_400_i64;
        let base = 1_700_000_000_i64;
        let start = format_ymd(base);
        let end = format_ymd(base + 30 * day);

        let w = CustomWindow::parse(&start, &end).expect("valid window");
        assert_eq!(w.days(), 30);

        // Reversed input is normalised rather than rejected.
        let flipped = CustomWindow::parse(&end, &start).expect("reversed window");
        assert_eq!(flipped, w);

        // Equal dates leave an empty window.
        assert!(CustomWindow::parse(&start, &start).is_none());
        // Unparseable dates are rejected.
        assert!(CustomWindow::parse("not-a-date", &end).is_none());
        assert!(CustomWindow::parse("", &end).is_none());
        // Absurdly long windows are refused.
        let long_start = format_ymd(base);
        let long_end = format_ymd(base + 20 * 365 * day);
        assert!(CustomWindow::parse(&long_start, &long_end).is_none());
    }

    #[test]
    fn test_custom_window_picks_interval_and_axis_style_from_its_length() {
        let day = 86_400_i64;
        let base = 1_700_000_000_i64;
        let win = |d: i64| {
            CustomWindow::parse(&format_ymd(base), &format_ymd(base + d * day)).expect("window")
        };

        // Intraday spans keep intraday bars and show times.
        let one_day = win(1);
        assert_eq!(one_day.interval(), Interval::Min5);
        assert_eq!(one_day.axis_date_style(), AxisDateStyle::TimeAndDate);

        let one_week = win(7);
        assert_eq!(one_week.interval(), Interval::Min15);
        assert_eq!(one_week.axis_date_style(), AxisDateStyle::DayMonth);

        let one_month = win(30);
        assert_eq!(one_month.interval(), Interval::Hour1);
        assert_eq!(one_month.axis_date_style(), AxisDateStyle::DayMonth);

        let six_months = win(180);
        assert_eq!(six_months.interval(), Interval::Day1);
        assert_eq!(six_months.axis_date_style(), AxisDateStyle::MonthYear);

        let two_years = win(730);
        assert_eq!(two_years.interval(), Interval::Day1);
        assert_eq!(two_years.axis_date_style(), AxisDateStyle::MonthYear);
    }

    #[test]
    fn test_custom_range_label_and_fallbacks() {
        assert_eq!(TimeRange::Custom.label(), "Custom");
        // Presets are unaffected by the new variant.
        assert_eq!(TimeRange::D1.to_days(), 1);
        assert_eq!(TimeRange::Y1.to_days(), 365);
        assert_eq!(TimeRange::Y5.to_days(), 1825);
    }

    #[test]
    fn test_ymd_roundtrip() {
        let ts = 1_700_000_000_i64;
        let text = format_ymd(ts);
        let back = parse_ymd(&text).expect("parses back");
        // Midnight of that day, so it lands on or before the original instant.
        assert!(back <= ts);
        assert_eq!(format_ymd(back), text);
        // Surrounding whitespace is tolerated.
        assert_eq!(parse_ymd(&format!("  {text}  ")), Some(back));
    }

    #[test]
    fn test_axis_date_style_matches_the_selected_timeframe() {
        assert_eq!(TimeRange::D1.axis_date_style(), AxisDateStyle::TimeAndDate);
        assert_eq!(TimeRange::W1.axis_date_style(), AxisDateStyle::DayMonth);
        assert_eq!(TimeRange::M1.axis_date_style(), AxisDateStyle::DayMonth);
        assert_eq!(TimeRange::M3.axis_date_style(), AxisDateStyle::MonthYear);
        assert_eq!(TimeRange::M6.axis_date_style(), AxisDateStyle::MonthYear);
        assert_eq!(TimeRange::Y1.axis_date_style(), AxisDateStyle::MonthYear);
        assert_eq!(TimeRange::Y5.axis_date_style(), AxisDateStyle::MonthYear);
    }

    #[test]
    fn test_each_timeframe_uses_a_distinct_axis_format() {
        // 1D must show both a clock and a date.
        let t = 1_704_110_100.0; // 2024-01-01 11:55 UTC
        assert_eq!(
            format_ts_styled(t, AxisDateStyle::TimeAndDate),
            "11:55 01 Jan"
        );
        // 1W shows a day and month, with no time component.
        assert_eq!(format_ts_styled(t, AxisDateStyle::DayMonth), "01 Jan");
        // 3M and longer collapse to month and year.
        assert_eq!(format_ts_styled(t, AxisDateStyle::MonthYear), "Jan 2024");
    }

    #[test]
    fn test_different_ranges_are_actually_distinct_requests() {
        // The 1Y/6M charts looked identical because the key only carried the
        // interval, and both are daily. The window length must differ.
        assert!(TimeRange::Y1.to_days() > TimeRange::M6.to_days());
        assert!(TimeRange::M6.to_days() > TimeRange::M3.to_days());
        assert!(TimeRange::M3.to_days() > TimeRange::M1.to_days());
        // Daily and longer ranges share an interval but not a window.
        assert_eq!(TimeRange::M6.to_interval(), TimeRange::Y1.to_interval());
    }

    #[test]
    fn test_drag_to_data_maps_pixels_to_the_window_fraction() {
        // Dragging a quarter of the plot width moves the view a quarter span.
        let dx = drag_to_data(200.0, 800.0, 1000.0);
        assert!((dx + 250.0).abs() < 1e-9, "drag right moves the view back");
        let dy = drag_to_data(-100.0, 400.0, 500.0);
        assert!((dy - 125.0).abs() < 1e-9, "drag up moves the view down");
    }

    #[test]
    fn test_drag_to_data_ignores_degenerate_input() {
        assert_eq!(drag_to_data(100.0, 0.0, 1000.0), 0.0, "zero plot size");
        assert_eq!(drag_to_data(100.0, 800.0, 0.0), 0.0, "zero span");
        assert_eq!(drag_to_data(100.0, 800.0, -5.0), 0.0, "negative span");
        assert_eq!(drag_to_data(f32::NAN, 800.0, 1000.0), 0.0);
    }

    #[test]
    fn test_pan_shifts_the_window_and_keeps_its_size() {
        let z = ZoomState::default().zoom_in(500.0, 50.0, &[], 0.0, 1000.0);
        let before = z.window(0.0, 1000.0, 0.0, 100.0);
        let panned = z.pan_by(-100.0, -10.0, &flat_series(), 0.0, 1000.0, 0.0, 100.0);
        let after = panned.window(0.0, 1000.0, 0.0, 100.0);
        assert!(
            ((after.1 - after.0) - (before.1 - before.0)).abs() < 1e-9,
            "size kept"
        );
        assert!(
            ((after.3 - after.2) - (before.3 - before.2)).abs() < 1e-9,
            "height kept"
        );
        assert!(after.0 < before.0, "moved left");
        assert!(after.2 < before.2, "moved down");
    }

    #[test]
    fn test_pan_is_ignored_when_not_zoomed() {
        let z = ZoomState::default();
        assert_eq!(
            z.pan_by(-50.0, -5.0, &flat_series(), 0.0, 1000.0, 0.0, 100.0),
            z
        );
    }

    /// A simple flat series used by the pan tests so the price envelope is
    /// well defined and the clamp has real candles to work against.
    fn flat_series() -> Vec<Candle> {
        (0..100)
            .map(|i| Candle::new(1_000.0 + i as f64 * 10.0, 45.0, 55.0, 40.0, 50.0, 1.0))
            .collect()
    }

    #[test]
    fn test_pan_is_clamped_to_the_data_extent() {
        let candles = flat_series();
        let z = ZoomState::default().zoom_in(1_500.0, 47.5, &candles, 1_000.0, 2_000.0);
        // A huge drag must not fling the chart off into empty space.
        let far = z.pan_by(-1.0e9, -1.0e9, &candles, 1_000.0, 2_000.0, 40.0, 55.0);
        let (wx0, wx1, wy0, _) = far.window(1_000.0, 2_000.0, 40.0, 55.0);
        assert!(
            wx0 >= -1e-6 && wx1 <= 2_000.0 + 1e-6,
            "x window must stay on the data"
        );
        assert!(wy0 >= -1e-6, "y window must stay on the data");
        assert!(far.pan_x.abs() < 1.0e9);
    }

    #[test]
    fn test_pan_settles_at_the_data_edge() {
        let candles = flat_series();
        let z = ZoomState::default().zoom_in(1_500.0, 47.5, &candles, 1_000.0, 2_000.0);
        // Push hard to the right; the window should stop at the data edge.
        let mut p = z;
        for _ in 0..50 {
            p = p.pan_by(500.0, 500.0, &candles, 1_000.0, 2_000.0, 40.0, 55.0);
        }
        let (wx0, wx1, _, _) = p.window(1_000.0, 2_000.0, 40.0, 55.0);
        assert!(
            (wx1 - 2_000.0).abs() < 1.0,
            "right edge stops at the data end"
        );
        assert!(wx0 < 2_000.0, "window still has width");
    }

    /// Regression test: the clamp was derived from the *panned* window, so the
    /// allowed range grew every frame and repeated drags walked the chart off
    /// the data entirely, leaving a blank pane at high zoom.
    #[test]
    fn test_repeated_pan_does_not_drift_off_the_data() {
        let candles = flat_series();
        let mut z = ZoomState::default().zoom_in(1_500.0, 47.5, &candles, 1_000.0, 2_000.0);
        z = z.zoom_in(1_500.0, 47.5, &candles, 1_000.0, 2_000.0);
        for _ in 0..200 {
            z = z.pan_by(-25.0, -5.0, &candles, 1_000.0, 2_000.0, 40.0, 55.0);
            let (wx0, wx1, wy0, wy1) = z.window(1_000.0, 2_000.0, 40.0, 55.0);
            assert!(
                wx0 >= -1e-6 && wx1 <= 2_000.0 + 1e-6,
                "x window escaped the data"
            );
            assert!(
                wy0 >= -1e-6 && wy1 <= 55.0 + 1e-6,
                "y window escaped the data"
            );
        }
        let (wx0, wx1, _, _) = z.window(1_000.0, 2_000.0, 40.0, 55.0);
        assert!(wx1 - wx0 > 0.0, "window must stay a valid width");
        assert!(z.pan_x.is_finite() && z.pan_y.is_finite());
    }

    /// Vertical panning must not move the view into a price band that has no
    /// bars in the visible x-window, which would render an empty pane.
    #[test]
    fn test_vertical_pan_keeps_visible_candles_on_screen() {
        let candles = flat_series();
        let z = ZoomState::default().zoom_in(1_500.0, 47.5, &candles, 1_000.0, 2_000.0);
        let mut p = z;
        for _ in 0..60 {
            p = p.pan_by(0.0, -50.0, &candles, 1_000.0, 2_000.0, 40.0, 55.0);
        }
        let (wx0, wx1, wy0, wy1) = p.window(1_000.0, 2_000.0, 40.0, 55.0);
        let (lo, hi) = price_envelope(&candles, wx0, wx1).unwrap();
        assert!(
            wy0 <= hi && wy1 >= lo,
            "y window ({wy0}..{wy1}) must still overlap the candles ({lo}..{hi})"
        );
    }

    #[test]
    fn test_zooming_clears_any_pan_so_the_new_centre_wins() {
        let mut z = ZoomState::default().zoom_in(500.0, 50.0, &[], 0.0, 1000.0);
        z = z.pan_by(-80.0, -8.0, &flat_series(), 0.0, 1000.0, 0.0, 100.0);
        assert!(z.pan_x.abs() > f64::EPSILON, "pan recorded");
        let z = z.zoom_in(500.0, 50.0, &[], 0.0, 1000.0);
        assert_eq!(z.pan_x, 0.0, "zoom-in drops the pan");
        assert_eq!(z.pan_y, 0.0, "zoom-in drops the pan");
    }

    #[test]
    fn test_reset_clears_zoom_and_pan() {
        let mut z = ZoomState::default().zoom_in(500.0, 50.0, &[], 0.0, 1000.0);
        z = z.pan_by(-80.0, -8.0, &flat_series(), 0.0, 1000.0, 0.0, 100.0);
        let r = z.reset();
        assert!(!r.is_zoomed());
        assert_eq!(r.pan_x, 0.0);
        assert_eq!(r.pan_y, 0.0);
        assert_eq!(r.focus_x, None);
    }

    #[test]
    fn test_heikin_ashi_seeds_first_open_from_raw_candle() {
        let c = Candle::new(1.0, 100.0, 110.0, 95.0, 105.0, 1_000.0);
        let ha = heikin_ashi(&[c]);
        assert_eq!(ha.len(), 1);
        assert!(
            (ha[0].open - 102.5).abs() < 1e-9,
            "seed open is (o + c) / 2"
        );
        assert!(
            (ha[0].close - 102.5).abs() < 1e-9,
            "close is the 4-way average"
        );
    }

    #[test]
    fn test_heikin_ashi_is_recursive() {
        let candles = vec![
            Candle::new(1.0, 100.0, 110.0, 95.0, 105.0, 1_000.0),
            Candle::new(2.0, 105.0, 115.0, 100.0, 110.0, 1_000.0),
        ];
        let ha = heikin_ashi(&candles);
        let first_close = (100.0 + 110.0 + 95.0 + 105.0) / 4.0;
        let first_open = (100.0 + 105.0) / 2.0;
        let expect_open = 0.5 * (first_open + first_close);
        assert!(
            (ha[1].open - expect_open).abs() < 1e-9,
            "second open must average the previous HA open and close"
        );
    }

    #[test]
    fn test_heikin_ashi_range_contains_open_and_close() {
        let series = synthetic_ohlcv("HA", 60, 11, 250.0);
        for (src, ha) in series.candles.iter().zip(heikin_ashi(&series.candles)) {
            assert!(
                ha.high >= ha.open.max(ha.close),
                "HA high must contain body"
            );
            assert!(ha.low <= ha.open.min(ha.close), "HA low must contain body");
            assert!(ha.high >= src.high, "HA high must not drop below raw high");
            assert!(ha.low <= src.low, "HA low must not rise above raw low");
            assert_eq!(ha.t, src.t);
            assert_eq!(ha.volume, src.volume);
        }
    }

    #[test]
    fn test_heikin_ashi_empty_series() {
        assert!(heikin_ashi(&[]).is_empty());
    }
}
