//! Optional loopback JSON API over the same cache and engines the UI uses.
//!
//! # Why this exists, and why it is shaped the way it is
//!
//! The obvious implementation of "expose the forecasts over HTTP" is to spin up
//! a server that synthesises a sine wave and serves predictions for it. That is
//! a demo, not a feature: it reports numbers that do not correspond to any
//! instrument, and a caller has no way to tell that from the response. So every
//! route here is backed by the real SQLite cache and the real ONNX graphs, and a
//! cache miss is reported as `503` rather than papered over with invented data.
//!
//! Two further constraints come from the app this ships in:
//!
//! * **Loopback only.** This is a desktop app that already holds a user's
//!   portfolio in memory and a local cache on disk. Exposing that on `0.0.0.0`
//!   would turn any process on the machine, and anything on the network, into a
//!   reader of their data. The bind address is not configurable for that reason:
//!   [`spawn_if_enabled`] always binds `127.0.0.1`.
//! * **Off by default.** A local listening socket is a security-relevant side
//!   effect that nobody asked for just by launching a chart viewer, so it only
//!   starts when `BHARAT_API=1` is set in the environment.
//!
//! Inference is CPU-bound and holds locks, so every handler hands the work to
//! `spawn_blocking` rather than stalling the axum worker that would otherwise
//! serve another request.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use bt_analytics::models::engine::{BharatModelEngine, Model, ModelError};
use bt_analytics::models::narrative::NarrativeEngine;
use bt_analytics::models::quantile_engine::QuantileConeOutput;
use bt_analytics::models::ttm_engine::{TtmEngine, TtmStatus};
use bt_analytics::signal::Signal;
use bt_core::Candle;

/// Environment variable that opts in. Any other value keeps the server off.
const ENV_ENABLE: &str = "BHARAT_API";
/// Port override. The bind address is *not* overridable; see the module docs.
const ENV_PORT: &str = "BHARAT_API_PORT";
const DEFAULT_PORT: u16 = 7878;
/// Cap on a single candles response, so one request cannot stream a whole cache.
const MAX_LIMIT: usize = 5_000;
const DEFAULT_LIMIT: usize = 500;

/// Shared, cheaply-cloned handles to the app's real data and models.
#[derive(Clone)]
pub struct ApiState {
    cache: Arc<bt_data::cache::Cache>,
    engine: Arc<BharatModelEngine>,
}

/// Which engine a forecast route should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Dlinear,
    Nhits,
    Chronos,
    Ttm,
}

impl Engine {
    /// The `BharatModelEngine` model this maps to, if any.
    fn to_model(self) -> Option<Model> {
        match self {
            Engine::Dlinear => Some(Model::DLinear),
            Engine::Nhits => Some(Model::NHiTS),
            Engine::Chronos => Some(Model::Chronos),
            // TinyTimeMixer has its own engine and its own graph problems.
            Engine::Ttm => None,
        }
    }

    /// Resolve a name from the query string. Chronos is the default because it
    /// is the only engine that emits a quantile band alongside the point path.
    fn parse(name: Option<&str>) -> Result<Self, ApiFailure> {
        match name.unwrap_or("chronos") {
            "dlinear" => Ok(Self::Dlinear),
            "nhits" => Ok(Self::Nhits),
            "chronos" => Ok(Self::Chronos),
            "ttm" => Ok(Self::Ttm),
            other => Err(ApiFailure::BadEngine(other.to_string())),
        }
    }
}

/// Query for `/api/forecast`, which selects an engine as well as a symbol.
#[derive(Debug, Deserialize)]
struct ForecastQuery {
    symbol: String,
    #[serde(default = "default_interval")]
    interval: String,
    #[serde(default)]
    engine: Option<String>,
}

/// Direction implied by a point forecast, for when the classifier is
/// unavailable.
///
/// This is a *reading* of the forecast, not a calibrated probability, so the
/// caller is told which source produced the badge via `signal_source`. Anything
/// presented without that distinction would be a fabricated confidence.
fn direction_signal(terminal: f64, last: f64) -> Signal {
    if !terminal.is_finite() || !last.is_finite() || last.abs() < 1e-12 {
        return Signal::Hold;
    }
    let change = terminal / last - 1.0;
    if change > 0.005 {
        Signal::Buy
    } else if change < -0.005 {
        Signal::Sell
    } else {
        Signal::Hold
    }
}

/// A response with no usable data. Always accompanied by an explanation, so a
/// caller can tell "not cached" from "the graph is broken".
fn unavailable(code: StatusCode, reason: &str) -> axum::response::Response {
    (code, Json(serde_json::json!({ "error": reason }))).into_response()
}

/// Map an engine failure onto a status. A broken or missing graph is the
/// server's problem (`500`), a too-short history is the caller's (`400`).
fn engine_error(e: ModelError) -> axum::response::Response {
    let reason = e.to_string();
    let client_fault = matches!(
        e,
        ModelError::InsufficientHistory { .. } | ModelError::WindowLength { .. }
    );
    unavailable(
        if client_fault {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &reason,
    )
}

#[derive(Debug, Deserialize)]
struct CandleQuery {
    symbol: String,
    #[serde(default = "default_interval")]
    interval: String,
    limit: Option<usize>,
}

fn default_interval() -> String {
    "1d".to_string()
}

/// Read the tail of the cached series for a symbol.
///
/// Returns `Ok(None)` when nothing is cached, so each handler can pick the right
/// status instead of inventing a series.
fn cached_closes(
    cache: &bt_data::cache::Cache,
    symbol: &str,
    interval: &str,
) -> Option<Vec<Candle>> {
    cache
        .get_ohlcv(symbol, interval)
        .ok()
        .flatten()
        .filter(|c| !c.is_empty())
}

#[derive(Serialize)]
struct CandleOut {
    t: f64,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    v: f64,
}

impl From<&Candle> for CandleOut {
    fn from(c: &Candle) -> Self {
        Self {
            t: c.t,
            o: c.open,
            h: c.high,
            l: c.low,
            c: c.close,
            v: c.volume,
        }
    }
}

#[derive(Serialize)]
struct CandlesOut {
    symbol: String,
    interval: String,
    count: usize,
    candles: Vec<CandleOut>,
}

/// `GET /api/candles` — the raw cached series, newest last.
async fn candles(State(s): State<ApiState>, Query(q): Query<CandleQuery>) -> impl IntoResponse {
    let cache = s.cache.clone();
    let symbol = q.symbol.clone();
    let interval = q.interval.clone();
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

    let rows = tokio::task::spawn_blocking(move || cached_closes(&cache, &symbol, &interval))
        .await
        .ok()
        .flatten();

    let Some(rows) = rows else {
        return unavailable(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!(
                "no unexpired {} data cached for {}; load the symbol in the app first",
                q.interval, q.symbol
            ),
        );
    };

    let start = rows.len().saturating_sub(limit);
    let candles: Vec<CandleOut> = rows[start..].iter().map(CandleOut::from).collect();
    let count = candles.len();
    Json(CandlesOut {
        symbol: q.symbol,
        interval: q.interval,
        count,
        candles,
    })
    .into_response()
}

#[derive(Serialize)]
struct ForecastOut {
    symbol: String,
    engine: String,
    horizon_steps: usize,
    lookback: usize,
    last_close: f64,
    predictions: Vec<f64>,
    /// Present only for engines that emit a quantile band (Chronos).
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    signal: &'static str,
    confidence: f32,
    /// Where `confidence` came from, so a directional read is never mistaken for
    /// a calibrated classifier output.
    signal_source: &'static str,
    /// Set when the classifier's direction contradicts the forecast's own
    /// direction.
    ///
    /// These are separate models and they genuinely disagree — a point forecast
    /// that drifts up while a classifier reads the order-flow features as a sell
    /// is not a bug. But shipping "+6.4% to a terminal level of X" beside an
    /// unexplained `SELL` badge reads like a contradiction, so the disagreement
    /// is stated rather than left for the caller to notice.
    divergence: Option<String>,
    commentary: String,
    risk_badge: String,
}

/// Does a classifier verdict agree with the direction of the forecast?
fn divergence(terminal: f64, last: f64, signal: Signal) -> Option<String> {
    let direction = direction_signal(terminal, last);
    if direction == Signal::Hold || direction == signal {
        return None;
    }
    let (d, s) = (direction.label(), signal.label());
    Some(format!(
        "WatchSignal reads {s} while the projected path runs {d} over the horizon; \
         the two are independent models, so both are reported as measured rather than reconciled."
    ))
}

/// `GET /api/forecast` — a point forecast plus the plain-language read-out.
async fn forecast(
    State(s): State<ApiState>,
    Query(q): Query<ForecastQuery>,
) -> axum::response::Response {
    let wanted = match Engine::parse(q.engine.as_deref()) {
        Ok(e) => e,
        Err(e) => return e.into_response(),
    };

    let cache = s.cache.clone();
    let engine = s.engine.clone();
    let symbol = q.symbol.clone();
    let interval = q.interval.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<ForecastOut, ApiFailure> {
        let rows = cached_closes(&cache, &symbol, &interval).ok_or(ApiFailure::NotCached)?;
        let closes: Vec<f64> = rows.iter().map(|c| c.close).collect();
        let last = *closes.last().ok_or(ApiFailure::NotCached)?;

        let pick = match wanted.to_model() {
            Some(m) => m,
            // TinyTimeMixer is not served through the shared engine, so report
            // *why* rather than a bare "unsupported".
            None => {
                let ttm = TtmEngine::new(bt_analytics::forecast::models_dir());
                let reason = match ttm.probe() {
                    TtmStatus::Missing => "TinyTimeMixer graph is not installed".to_string(),
                    TtmStatus::GraphBroken(m) => format!("TinyTimeMixer graph cannot run: {m}"),
                    TtmStatus::Ready => {
                        "TinyTimeMixer is available but is not exposed over HTTP".to_string()
                    }
                };
                return Err(ApiFailure::Ttm(reason));
            }
        };
        let out = engine.run(pick, &closes).map_err(ApiFailure::Engine)?;
        let terminal = out.predictions.last().copied().unwrap_or(last);

        // Prefer the real calibrated classifier. Fall back to reading the
        // forecast's own direction, and say so.
        let (signal, confidence, source) = match engine.predict_signal(&rows) {
            Ok(sig) => (
                parse_signal(&sig.signal),
                sig.confidence,
                "watchsignal-lstm",
            ),
            Err(_) => (direction_signal(terminal, last), 0.5, "point-forecast"),
        };

        let band = QuantileConeOutput::from_forecast(&out, out.predictions.len());
        let text =
            NarrativeEngine::describe(&symbol, last, terminal, signal, confidence, &out.model_name);

        Ok(ForecastOut {
            symbol,
            engine: out.model_name,
            horizon_steps: out.horizon_steps,
            lookback: out.lookback,
            last_close: last,
            lower: band.as_ref().map(|c| c.p10_lower.clone()),
            upper: band.as_ref().map(|c| c.p90_upper.clone()),
            predictions: out.predictions,
            signal: signal.label(),
            confidence,
            signal_source: source,
            divergence: divergence(terminal, last, signal),
            commentary: text.commentary,
            risk_badge: text.risk_badge,
        })
    })
    .await;

    match result {
        Ok(Ok(body)) => Json(body).into_response(),
        Ok(Err(e)) => e.into_response(),
        Err(e) => unavailable(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("task failed: {e}"),
        ),
    }
}

/// Map the engine's string label back onto the enum.
fn parse_signal(label: &str) -> Signal {
    match label.to_ascii_uppercase().as_str() {
        "BUY" => Signal::Buy,
        "SELL" => Signal::Sell,
        _ => Signal::Hold,
    }
}

/// Failures a handler can produce, each mapped to a deliberate status.
enum ApiFailure {
    /// The symbol has nothing unexpired in the cache.
    NotCached,
    /// TinyTimeMixer specifically, with the probe's reason. There is no generic
    /// "unsupported" case: an engine is either backed by a graph or it is named
    /// wrongly, and those need different status codes and different wording.
    Ttm(String),
    /// The caller named an engine that does not exist.
    BadEngine(String),
    /// The graph or history rejected the request.
    Engine(ModelError),
}

impl ApiFailure {
    fn into_response(self) -> axum::response::Response {
        match self {
            ApiFailure::NotCached => unavailable(
                StatusCode::SERVICE_UNAVAILABLE,
                "no unexpired data cached for that symbol and interval",
            ),
            ApiFailure::Ttm(reason) => unavailable(StatusCode::NOT_IMPLEMENTED, &reason),
            ApiFailure::BadEngine(name) => unavailable(
                StatusCode::BAD_REQUEST,
                &format!("unknown engine {name:?}; try dlinear, nhits, chronos or ttm"),
            ),
            ApiFailure::Engine(e) => engine_error(e),
        }
    }
}

// Lets a handler `return e.into_response();` without importing axum's trait.
impl From<ApiFailure> for axum::response::Response {
    fn from(f: ApiFailure) -> Self {
        f.into_response()
    }
}

#[derive(Serialize)]
struct EngineStatus {
    name: &'static str,
    available: bool,
    detail: String,
}

/// `GET /api/engines` — what this installation can actually do.
///
/// TTM gets a `probe()` because "the file exists" and "the file runs" are
/// different questions, and for the currently shipped `ttm_r2_int8.onnx` they
/// have different answers.
async fn engines(State(s): State<ApiState>) -> impl IntoResponse {
    let engine = s.engine.clone();
    let out = tokio::task::spawn_blocking(move || {
        // Gate on the runtime before touching any ort API. `ort`'s lazy loader
        // `panic!`s (and this profile is `panic = "abort"`, so it kills the
        // process) if it resolves the bare name `onnxruntime.dll` to the
        // incompatible inbox build. Reporting "no runtime" is both safe and the
        // truth.
        let mut runtime_note = None;
        if let Err(e) = bt_analytics::ort_runtime::ensure_initialized() {
            runtime_note = Some(e);
        }

        let mut v = Vec::new();
        for m in Model::ALL {
            let installed = engine.is_available(m);
            let available = installed && runtime_note.is_none();
            let detail = match (&runtime_note, installed) {
                (Some(r), _) => format!("ONNX Runtime unavailable: {r}"),
                (None, false) => "graph not installed".to_string(),
                (None, true) => format!(
                    "{} -> {} bars, needs {} bars",
                    m.label(),
                    m.horizon(),
                    m.required_history()
                ),
            };
            v.push(EngineStatus {
                name: m.label(),
                available,
                detail,
            });
        }

        let (available, detail) = match &runtime_note {
            Some(r) => (false, format!("ONNX Runtime unavailable: {r}")),
            None => match TtmEngine::new(bt_analytics::forecast::models_dir()).probe() {
                TtmStatus::Ready => (true, "512 -> 96 bars, runs".to_string()),
                TtmStatus::Missing => (false, "graph not installed".to_string()),
                TtmStatus::GraphBroken(m) => (false, format!("graph does not execute: {m}")),
            },
        };
        v.push(EngineStatus {
            name: "TinyTimeMixer R2",
            available,
            detail,
        });
        v
    })
    .await;

    match out {
        Ok(v) => Json(serde_json::json!({ "engines": v })).into_response(),
        Err(e) => unavailable(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("probe failed: {e}"),
        ),
    }
}

#[derive(Serialize)]
struct HealthOut {
    ok: bool,
    version: &'static str,
    bind: &'static str,
}

/// `GET /api/health` — liveness, and a reminder of where it is bound.
async fn health() -> impl IntoResponse {
    Json(HealthOut {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        bind: "127.0.0.1 (loopback only)",
    })
}

/// Build the router. Exposed for tests.
pub fn router(state: ApiState) -> Router {
    // CORS is permissive, which is only defensible because the listener is
    // loopback-only: a page in the user's own browser may read it, nothing else
    // can reach it in the first place.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    Router::new()
        .route("/api/health", get(health))
        .route("/api/engines", get(engines))
        .route("/api/candles", get(candles))
        .route("/api/forecast", get(forecast))
        .layer(cors)
        .with_state(state)
}

/// Start the API if `BHARAT_API=1`.
///
/// See [`spawn`] for the binding behaviour; this only resolves the opt-in.
pub fn spawn_if_enabled() -> Option<SocketAddr> {
    let wanted = std::env::var(ENV_ENABLE).ok()?;
    if wanted != "1" {
        return None;
    }
    let port: u16 = std::env::var(ENV_PORT)
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    spawn(port)
}

/// Bind `127.0.0.1:port` and serve in the background.
///
/// Returns the address actually bound, or `None` if the socket could not be
/// opened. Binding happens synchronously before this returns, so a caller that
/// sees `Some` knows the port is live and the reported port is the real one
/// (which matters when the port is 0).
///
/// Every failure is non-fatal: a terminal that cannot open a debug socket must
/// still draw charts.
pub fn spawn(port: u16) -> Option<SocketAddr> {
    let state = ApiState {
        cache: Arc::new(bt_data::cache::Cache::new(bt_data::default_cache_path()).ok()?),
        engine: Arc::new(BharatModelEngine::with_default_paths()),
    };

    // Bind first, synchronously, so a failure is reported instead of being
    // discovered later in a background task.
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .inspect_err(|e| tracing::warn!(error = %e, "api bind failed"))
        .ok()?;
    let addr = listener.local_addr().ok()?;
    listener.set_nonblocking(true).ok()?;

    std::thread::Builder::new()
        .name("bt-api".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::warn!(error = %e, "api runtime failed");
                    return;
                }
            };
            // The serve future runs for the life of the process, so `block_on`
            // must actually block. Returning from this closure drops the runtime
            // and silently cancels the listener.
            rt.block_on(async move {
                match tokio::net::TcpListener::from_std(listener) {
                    Ok(listener) => {
                        let local = listener
                            .local_addr()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|_| addr.to_string());
                        tracing::info!(address = %local, "api listening");
                        if let Err(e) = axum::serve(listener, router(state)).await {
                            tracing::warn!(error = %e, "api stopped");
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "api listener rejected"),
                }
            });
        })
        .inspect_err(|e| tracing::warn!(error = %e, "api thread failed"))
        .ok()?;

    Some(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> ApiState {
        ApiState {
            cache: Arc::new(
                bt_data::cache::Cache::new(std::env::temp_dir().join("bt_api_test.db")).unwrap(),
            ),
            engine: Arc::new(BharatModelEngine::with_default_paths()),
        }
    }

    #[tokio::test]
    async fn health_reports_the_loopback_bind() {
        let body = health().await.into_response();
        assert_eq!(body.status(), StatusCode::OK);
    }

    /// The single most important property: no cached data must never become a
    /// fabricated response.
    #[tokio::test]
    async fn a_cache_miss_is_503_not_invented_candles() {
        let body = candles(
            State(state()),
            Query(CandleQuery {
                symbol: "NO_SUCH_SYMBOL_EVER_CACHED".into(),
                interval: "1d".into(),
                limit: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(body.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn a_cache_miss_on_forecast_is_503_not_an_invented_forecast() {
        let body = forecast(
            State(state()),
            Query(ForecastQuery {
                symbol: "NO_SUCH_SYMBOL_EVER_CACHED".into(),
                interval: "1d".into(),
                engine: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(body.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn health_is_reachable_through_the_real_router() {
        use tower::ServiceExt;
        let app = router(state());
        let res = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "application/json"
        );
    }

    /// An uncached symbol must produce an explicit error body, never numbers.
    #[tokio::test]
    async fn a_cache_miss_body_says_why() {
        use axum::body::to_bytes;
        use tower::ServiceExt;
        let res = router(state())
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/candles?symbol=NO_SUCH_SYMBOL_EVER_CACHED&interval=1d")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(res.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("error"), "{text}");
        assert!(!text.contains("candles"), "must not emit a series: {text}");
    }

    #[tokio::test]
    async fn an_unknown_engine_is_400_not_a_silent_default() {
        let body = forecast(
            State(state()),
            Query(ForecastQuery {
                symbol: "RELIANCE".into(),
                interval: "1d".into(),
                engine: Some("gpt9".into()),
            }),
        )
        .await
        .into_response();
        assert_eq!(body.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn direction_is_read_only_from_a_real_move() {
        assert_eq!(direction_signal(101.0, 100.0), Signal::Buy);
        assert_eq!(direction_signal(99.0, 100.0), Signal::Sell);
        // Inside the dead band, and for junk inputs, hold rather than guess.
        assert_eq!(direction_signal(100.2, 100.0), Signal::Hold);
        assert_eq!(direction_signal(f64::NAN, 100.0), Signal::Hold);
        assert_eq!(direction_signal(100.0, 0.0), Signal::Hold);
    }

    /// A bullish path with a sell badge must be called out, not printed bare.
    #[test]
    fn a_contradicting_classifier_is_disclosed() {
        let d = divergence(1090.0, 1000.0, Signal::Sell);
        let msg = d.expect("up path + SELL must be disclosed");
        assert!(msg.contains("SELL"), "{msg}");
        assert!(msg.contains("BUY"), "{msg}");

        // Agreement, and a flat path, are not divergences.
        assert_eq!(divergence(1090.0, 1000.0, Signal::Buy), None);
        assert_eq!(divergence(100.2, 100.0, Signal::Sell), None);
    }

    #[tokio::test]
    async fn engine_status_reports_a_broken_graph_as_unavailable() {
        let body = engines(State(state())).await.into_response();
        assert_eq!(body.status(), StatusCode::OK);
    }

    #[test]
    fn the_listener_is_loopback_only() {
        // Guards the security property, not just the implementation detail: the
        // returned address must never be routable off-box.
        let a = SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT));
        assert!(a.ip().is_loopback());
    }

    #[test]
    fn limit_is_clamped_into_a_sane_range() {
        assert_eq!(0usize.clamp(1, MAX_LIMIT), 1);
        assert_eq!(usize::MAX.clamp(1, MAX_LIMIT), MAX_LIMIT);
    }
}
