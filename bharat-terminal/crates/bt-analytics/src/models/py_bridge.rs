// crates/bt-analytics/src/models/py_bridge.rs
// Author: Sourish Dey

//! Bridge to the embedded, headless Python forecaster.
//!
//! # What this is for
//!
//! `scripts/predictor.py` is a drift-and-volatility cone over cached closes. It
//! is cheap, needs no trained weights, and is a genuinely different model from
//! the ONNX engines here, so it is worth keeping as a second opinion rather than
//! folding into one of them. The runtime ships as `python_runtime/python.exe`
//! (embeddable CPython 3.11 with numpy), so the app has no external Python
//! dependency and no network access.
//!
//! # Two things the obvious implementation gets wrong
//!
//! **The child needs a bounded lifetime.** `predictor.py` reads all of stdin and
//! exits, so dropping stdin is enough to make it finish - but "enough" is doing
//! real work there. If the script ever blocks, `wait_with_output` waits forever
//! and takes a thread with it; on a 2 GB machine a handful of wedged children is
//! also a memory problem, not just a hung request. [`EmbeddedPyEngine::predict`]
//! therefore enforces a timeout and kills the child on expiry, so a bad run costs
//! one request rather than a wedged process.
//!
//! **Non-positive prices must be rejected before they reach Python.**
//! `predictor.py` takes `log(price)`, and a zero or negative bar yields `-inf`,
//! which propagates into the result and then into `json.dumps` as the bare token
//! `NaN` - not valid JSON, so the caller sees an opaque parse failure instead of
//! "that candle was 0". The check lives here where the error can name the bar.
//!
//! Input validation mirrors the script's own rule of at least 32 bars, so a
//! refusal reads the same on both sides of the pipe.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

/// Minimum bars `predictor.py` accepts.
pub const PY_MIN_BARS: usize = 32;
/// Default ceiling on one child process.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// CREATE_NO_WINDOW: spawn the child without a console window.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Request payload written to the child's stdin.
#[derive(Serialize)]
struct PythonInput<'a> {
    prices: &'a [f32],
}

/// Response payload read back from the child's stdout.
///
/// Every field is optional because the script reports failures as a JSON object
/// carrying only `error`, and a partial success is still worth surfacing rather
/// than discarding.
#[derive(Deserialize, Debug, Clone, Serialize, PartialEq)]
pub struct PythonForecastResult {
    /// `"ok"` on success.
    pub status: Option<String>,
    /// Last close the forecast is anchored to.
    pub last_price: Option<f32>,
    /// Number of bars projected.
    pub horizon: Option<usize>,
    /// Median path.
    pub forecast_p50: Option<Vec<f32>>,
    /// 10th percentile.
    pub lower_bound_p10: Option<Vec<f32>>,
    /// 90th percentile.
    pub upper_bound_p90: Option<Vec<f32>>,
    /// Std-dev of recent log returns, in percent.
    pub volatility_score: Option<f32>,
    /// Present only when the script failed.
    pub error: Option<String>,
}

impl PythonForecastResult {
    /// Whether the child reported a usable result.
    pub fn is_ok(&self) -> bool {
        self.status.as_deref() == Some("ok") && self.forecast_p50.is_some()
    }
}

/// Failures from the embedded runtime.
#[derive(Debug, thiserror::Error)]
pub enum PyBridgeError {
    #[error("Embedded Python predictor requires at least {PY_MIN_BARS} price bars, got {got}")]
    TooFewBars { got: usize },
    #[error("price history contains a non-finite value at index {index}")]
    NonFinitePrice { index: usize },
    #[error("price history contains a non-positive value ({value}) at index {index}")]
    NonPositivePrice { index: usize, value: f32 },
    #[error("Python runtime binary not found at {path}")]
    RuntimeMissing { path: PathBuf },
    #[error("Predictor script not found at {path}")]
    ScriptMissing { path: PathBuf },
    #[error("could not start the Python predictor: {0}")]
    Spawn(String),
    #[error("Python predictor did not finish within {0:?}")]
    Timeout(Duration),
    #[error("Python predictor failed ({status}): {stderr}")]
    Failed { status: String, stderr: String },
    #[error("Python predictor returned malformed output: {0}")]
    BadOutput(String),
    #[error("Python calculation error: {0}")]
    Script(String),
}

/// What the embedded runtime can actually do.
///
/// Worth distinguishing from "the files exist": `python_runtime/` can be present
/// and still be unable to import numpy, because an embeddable CPython resolves
/// imports through the *user site* directory as well as its own. On a developer
/// machine that silently borrows a globally installed numpy and everything
/// works; on a clean install the same code fails at `import numpy`. So the
/// status records not just whether numpy imported, but whether it came from the
/// bundled runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyStatus {
    /// Runtime found, numpy imports, and numpy lives inside `python_runtime/`.
    Ready { numpy: String },
    /// numpy imports, but from outside the bundled runtime - typically the user
    /// site directory. The feature works here and will break on a clean install,
    /// so this is reported rather than passed off as ready.
    ReadyButNotBundled { numpy: String, path: String },
    /// `python.exe` is not installed.
    RuntimeMissing,
    /// numpy is not importable, so `predictor.py` cannot run.
    NumpyMissing(String),
}

impl PyStatus {
    /// Whether a forecast can be served right now.
    pub fn can_forecast(&self) -> bool {
        matches!(self, Self::Ready { .. } | Self::ReadyButNotBundled { .. })
    }
}

/// Self-verification of the embedded runtime, kept next to the bridge.
impl EmbeddedPyEngine {
    /// Run one import check and report what the runtime can do.
    ///
    /// Cheap enough to call from a status endpoint, and it is what `/api/engines`
    /// reports. Cheaper still: the verdict is cached, since nothing about a
    /// forecast request changes the answer.
    pub fn probe(&self) -> PyStatus {
        if let Some(hit) = self.probed.lock().ok().and_then(|g| g.clone()) {
            return hit;
        }
        let verdict = self.probe_uncached();
        if let Ok(mut g) = self.probed.lock() {
            *g = Some(verdict.clone());
        }
        verdict
    }

    fn probe_uncached(&self) -> PyStatus {
        if !self.python_bin.is_file() {
            return PyStatus::RuntimeMissing;
        }
        if !self.script_path.is_file() {
            return PyStatus::NumpyMissing("predictor.py is not installed".into());
        }
        // Report where numpy lives, not just that it imports. Parsed as JSON
        // rather than by splitting on quotes: the path arrives with doubled
        // backslashes, so a raw string compare against a single-backslash root
        // reports every install as not-self-contained even when it is.
        let code =
            "import json,numpy;print(json.dumps({'v':numpy.__version__,'p':numpy.__file__}))";
        match self.run_python(&["-c", code], None) {
            Ok(text) => {
                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        return PyStatus::NumpyMissing(format!("unreadable probe output: {e}"))
                    }
                };
                let str_field = |k: &str| -> String {
                    parsed
                        .get(k)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string()
                };
                let numpy = str_field("v");
                let path = str_field("p");
                if numpy.is_empty() {
                    return PyStatus::NumpyMissing("could not read numpy version".into());
                }
                // The runtime is `<root>/python_runtime/python.exe`, so a bundled
                // numpy must resolve under `<root>/python_runtime`.
                let root = self
                    .python_bin
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                let bundled = !root.is_empty() && path.starts_with(&root);
                if bundled {
                    PyStatus::Ready { numpy }
                } else {
                    PyStatus::ReadyButNotBundled { numpy, path }
                }
            }
            Err(e) => PyStatus::NumpyMissing(e.to_string()),
        }
    }

    /// Spawn the runtime with `args` and return its trimmed stdout.
    ///
    /// Single place where the child is created, so the windowless flag, the
    /// timeout, the isolation flags and the pipe handling cannot drift apart
    /// between the forecast path and the self-check.
    ///
    /// `-E -s` is what makes this an *embedded* runtime rather than a thin
    /// shim. An embeddable CPython otherwise resolves imports through the user
    /// site directory (`%APPDATA%\Python\Python311\site-packages`) and that
    /// directory sits *ahead* of the bundled one on `sys.path`, so a machine with
    /// a different numpy installed would silently run the forecast against the
    /// user's copy - or fail outright on a clean machine. `-E` ignores inherited
    /// environment variables and `-s` drops the user site, leaving only
    /// `python_runtime/`'s own packages, which is what "self-contained" has to
    /// mean for a shipped app.
    fn run_python(&self, args: &[&str], stdin: Option<&[u8]>) -> Result<String, PyBridgeError> {
        let mut cmd = Command::new(&self.python_bin);
        cmd.arg("-E").arg("-s");
        cmd.args(args);
        cmd.stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

        // Without this, every forecast flashes a black console window at the
        // user. It is the whole reason this is a subprocess rather than a thread.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| PyBridgeError::Spawn(e.to_string()))?;
        // Dropping `stdin` closes the pipe, which is what tells the script that
        // the payload is complete. Holding it open would hang the child.
        if let Some(bytes) = stdin {
            let mut pipe = child
                .stdin
                .take()
                .ok_or_else(|| PyBridgeError::Spawn("child had no stdin".into()))?;
            pipe.write_all(bytes)
                .map_err(|e| PyBridgeError::Spawn(format!("could not write payload: {e}")))?;
        }
        let output = wait_with_timeout(&mut child, self.timeout)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(PyBridgeError::Failed {
                status: output
                    .status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into()),
                stderr,
            });
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

/// Stateless handle to the embedded predictor.
///
/// Construction reads nothing from disk, so a machine without the runtime still
/// starts and only fails if this engine is actually used.
pub struct EmbeddedPyEngine {
    python_bin: PathBuf,
    script_path: PathBuf,
    timeout: Duration,
    /// Cached `probe` verdict; the answer cannot change while the process runs.
    probed: Mutex<Option<PyStatus>>,
}

impl Default for EmbeddedPyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddedPyEngine {
    /// Locate the runtime next to the executable, falling back to the project
    /// layout for development builds.
    ///
    /// The candidate order matches `ort_runtime::candidate_paths` and
    /// `forecast::models_dir`: beside the exe first, then up through ancestors
    /// for `target/<profile>/<bin>` layouts, so an installed copy, a portable
    /// copy and a dev build each find their own.
    pub fn new() -> Self {
        let dir = crate::forecast::models_dir()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            python_bin: first_existing(&python_candidates(&dir))
                .unwrap_or_else(|| dir.join("python_runtime").join("python.exe")),
            script_path: first_existing(&script_candidates(&dir))
                .unwrap_or_else(|| dir.join("scripts").join("predictor.py")),
            timeout: DEFAULT_TIMEOUT,
            probed: Mutex::new(None),
        }
    }

    /// Same as [`Self::new`] but rooted at an explicit directory, for tests.
    pub fn rooted_at<P: AsRef<Path>>(dir: P) -> Self {
        let dir = dir.as_ref().to_path_buf();
        Self {
            python_bin: first_existing(&python_candidates(&dir))
                .unwrap_or_else(|| dir.join("python_runtime").join("python.exe")),
            script_path: first_existing(&script_candidates(&dir))
                .unwrap_or_else(|| dir.join("scripts").join("predictor.py")),
            timeout: DEFAULT_TIMEOUT,
            probed: Mutex::new(None),
        }
    }

    /// Override the child timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the script path, keeping the resolved runtime.
    ///
    /// Used to point at a different forecast script without re-deriving the
    /// runtime location.
    pub fn with_script<P: AsRef<Path>>(mut self, script: P) -> Self {
        self.script_path = script.as_ref().to_path_buf();
        self
    }

    /// Whether both the runtime and the script are present.
    pub fn is_available(&self) -> bool {
        self.python_bin.is_file() && self.script_path.is_file()
    }

    /// Resolved runtime path.
    pub fn python_bin(&self) -> &Path {
        &self.python_bin
    }

    /// Resolved script path.
    pub fn script_path(&self) -> &Path {
        &self.script_path
    }

    /// Run the drift-and-volatility cone over `prices`.
    ///
    /// The child is created hidden, given the payload on stdin, and killed if it
    /// exceeds the timeout.
    pub fn predict(&self, prices: &[f32]) -> Result<PythonForecastResult, PyBridgeError> {
        if prices.len() < PY_MIN_BARS {
            return Err(PyBridgeError::TooFewBars { got: prices.len() });
        }
        // `predictor.py` takes log(prices); a bad bar there becomes NaN, which
        // json.dumps emits as an unquoted `NaN` that no JSON parser accepts.
        for (index, &p) in prices.iter().enumerate() {
            if !p.is_finite() {
                return Err(PyBridgeError::NonFinitePrice { index });
            }
            if p <= 0.0 {
                return Err(PyBridgeError::NonPositivePrice { index, value: p });
            }
        }
        if !self.python_bin.is_file() {
            return Err(PyBridgeError::RuntimeMissing {
                path: self.python_bin.clone(),
            });
        }
        if !self.script_path.is_file() {
            return Err(PyBridgeError::ScriptMissing {
                path: self.script_path.clone(),
            });
        }

        let payload = serde_json::to_string(&PythonInput { prices })
            .map_err(|e| PyBridgeError::Spawn(format!("could not encode payload: {e}")))?;

        let stdout = self.run_python(
            &[self.script_path.to_string_lossy().as_ref()],
            Some(payload.as_bytes()),
        )?;

        let res: PythonForecastResult = serde_json::from_str(&stdout).map_err(|e| {
            // Distinguish "the script apologised" from "the script garbled".
            let detail = if e.is_data() {
                format!("{e}: {}", stdout.trim())
            } else {
                e.to_string()
            };
            PyBridgeError::BadOutput(detail)
        })?;

        // The script reports its own failures in-band, with exit code 0.
        if let Some(err) = &res.error {
            return Err(PyBridgeError::Script(err.clone()));
        }
        Ok(res)
    }
}

/// Collect the child's output, killing it if it overruns.
///
/// `wait_with_output` has no timeout of its own and consumes the child, so it
/// cannot be used directly: we need to keep the handle in order to kill it. The
/// pipes are drained on their own threads and the exit status is polled, which
/// also avoids the classic deadlock where a child blocks writing to stderr while
/// the parent blocks waiting for it.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output, PyBridgeError> {
    use std::io::Read;
    let mut stdout_pipe: Option<Box<dyn Read + Send>> = child
        .stdout
        .take()
        .map(|p| Box::new(p) as Box<dyn Read + Send>);
    let mut stderr_pipe: Option<Box<dyn Read + Send>> = child
        .stderr
        .take()
        .map(|p| Box::new(p) as Box<dyn Read + Send>);

    let drain = |pipe: Option<Box<dyn std::io::Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out_handle = drain(stdout_pipe.take());
    let err_handle = drain(stderr_pipe.take());

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Killing closes the pipes, so these finish promptly.
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return Err(PyBridgeError::Timeout(timeout));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(PyBridgeError::Spawn(e.to_string())),
        }
    };

    Ok(std::process::Output {
        status,
        stdout: out_handle.join().unwrap_or_default(),
        stderr: err_handle.join().unwrap_or_default(),
    })
}

/// First existing path from a candidate list.
fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_file()).cloned()
}

/// Candidate runtime paths for a base directory.
fn python_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    out.push(base.join("python_runtime").join("python.exe"));
    for up in 1..=3 {
        if let Some(parent) = base.ancestors().nth(up) {
            out.push(parent.join("python_runtime").join("python.exe"));
        }
    }
    out
}

/// Candidate script paths for a base directory.
fn script_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    out.push(base.join("scripts").join("predictor.py"));
    for up in 1..=3 {
        if let Some(parent) = base.ancestors().nth(up) {
            out.push(parent.join("scripts").join("predictor.py"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> EmbeddedPyEngine {
        EmbeddedPyEngine::rooted_at(crate::forecast::models_dir().parent().unwrap())
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| 2500.0 + i as f32 * 1.5).collect()
    }

    /// The headline contract: five bars, all three paths, ordered.
    #[test]
    fn produces_a_five_step_cone() {
        let res = engine()
            .predict(&ramp(35))
            .expect("embedded predictor should run when installed");
        assert_eq!(res.status.as_deref(), Some("ok"));
        assert_eq!(res.horizon, Some(5));
        assert_eq!(res.forecast_p50.as_ref().unwrap().len(), 5);
        assert_eq!(res.lower_bound_p10.as_ref().unwrap().len(), 5);
        assert_eq!(res.upper_bound_p90.as_ref().unwrap().len(), 5);
        assert!(res.volatility_score.is_some());
        assert!(res.is_ok());
    }

    /// p10 <= p50 <= p90 at every step, or the "cone" is just three lines.
    #[test]
    fn the_cone_is_ordered() {
        let res = engine().predict(&ramp(60)).expect("predict");
        let (p10, p50, p90) = (
            res.lower_bound_p10.as_ref().unwrap(),
            res.forecast_p50.as_ref().unwrap(),
            res.upper_bound_p90.as_ref().unwrap(),
        );
        for i in 0..p50.len() {
            assert!(
                p10[i] <= p50[i] + 1e-3,
                "step {i}: p10 {} > p50 {}",
                p10[i],
                p50[i]
            );
            assert!(
                p90[i] >= p50[i] - 1e-3,
                "step {i}: p90 {} < p50 {}",
                p90[i],
                p50[i]
            );
        }
    }

    /// last_price must be the bar we passed, not something reconstructed.
    #[test]
    fn the_anchor_is_the_last_bar_we_supplied() {
        let prices = ramp(40);
        let res = engine().predict(&prices).expect("predict");
        assert_eq!(res.last_price, Some(prices[prices.len() - 1]));
    }

    #[test]
    fn a_short_history_is_refused_before_starting_python() {
        let err = engine().predict(&ramp(31)).unwrap_err();
        assert!(
            matches!(err, PyBridgeError::TooFewBars { got: 31 }),
            "{err}"
        );
    }

    /// Guards the real trap: `log(0)` would come back as a bare `NaN` token,
    /// which no JSON parser accepts, hiding the actual cause.
    #[test]
    fn non_positive_and_non_finite_prices_are_rejected_by_name() {
        let mut zeroed = ramp(35);
        zeroed[7] = 0.0;
        match engine().predict(&zeroed) {
            Err(PyBridgeError::NonPositivePrice { index, value }) => {
                assert_eq!(index, 7);
                assert_eq!(value, 0.0);
            }
            other => panic!("expected NonPositivePrice at 7, got {other:?}"),
        }

        let mut negative = ramp(35);
        negative[2] = -5.0;
        match engine().predict(&negative) {
            Err(PyBridgeError::NonPositivePrice { index, .. }) => assert_eq!(index, 2),
            other => panic!("expected NonPositivePrice at 2, got {other:?}"),
        }

        let mut nan = ramp(35);
        nan[3] = f32::NAN;
        match engine().predict(&nan) {
            Err(PyBridgeError::NonFinitePrice { index }) => assert_eq!(index, 3),
            other => panic!("expected NonFinitePrice at 3, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_runtime_reports_the_path_it_looked_for() {
        let e = EmbeddedPyEngine::rooted_at("definitely-not-a-project");
        assert!(!e.is_available());
        match e.predict(&ramp(35)) {
            Err(PyBridgeError::RuntimeMissing { path }) => assert!(path.ends_with("python.exe")),
            other => panic!("expected RuntimeMissing, got {other:?}"),
        }
    }

    /// `probe` is what `/api/engines` reports, so it must agree with whether a
    /// forecast can actually run.
    #[test]
    fn the_probe_agrees_with_whether_a_forecast_runs() {
        let e = engine();
        let probe = e.probe();
        let ran = e.predict(&ramp(35)).is_ok();
        assert_eq!(
            probe.can_forecast(),
            ran,
            "probe said {probe:?} but the forecast ran={ran}"
        );
        if let PyStatus::Ready { numpy } | PyStatus::ReadyButNotBundled { numpy, .. } = &probe {
            assert!(!numpy.is_empty(), "no numpy version reported");
        }
    }

    /// The verdict is cached, so a status endpoint cannot spawn a process per hit.
    #[test]
    fn the_probe_is_cached() {
        let e = engine();
        assert_eq!(e.probe(), e.probe());
    }

    /// A runtime that is absent must be reported as absent.
    #[test]
    fn a_missing_runtime_is_reported_by_the_probe() {
        let e = EmbeddedPyEngine::rooted_at("definitely-not-a-project");
        assert_eq!(e.probe(), PyStatus::RuntimeMissing);
        assert!(!e.probe().can_forecast());
    }

    /// numpy is bundled, so the probe must say `Ready` and not
    /// `ReadyButNotBundled`.
    ///
    /// This is the regression test for a real false alarm: the probe output is
    /// JSON, so `numpy.__file__` arrives with doubled backslashes, and comparing
    /// that raw string against a single-backslash runtime root made every
    /// install report "not self-contained" while being exactly that.
    #[test]
    fn a_bundled_runtime_reports_ready_not_a_false_alarm() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        match e.probe() {
            PyStatus::Ready { numpy } => assert!(!numpy.is_empty()),
            PyStatus::ReadyButNotBundled { path, .. } => panic!(
                "numpy at {path} is inside python_runtime/, so this should be Ready; \
                 the path is likely being compared without decoding JSON escapes"
            ),
            other => panic!("expected a working runtime, got {other:?}"),
        }
    }

    /// The case that motivated `probe` at all: an embeddable CPython resolves
    /// imports through the *user site* directory, so `python_runtime/` could be
    /// present, `is_available()` could be true, and `predictor.py` could still
    /// fail on `import numpy` because numpy was never vendored in.
    ///
    /// numpy is now vendored, so an isolated interpreter - one that ignores the
    /// user site, exactly as the bridge now runs it - can import it. This is the
    /// test that would fail if a future change dropped numpy from the runtime, or
    /// if the isolation flags were removed.
    #[test]
    fn the_bundled_runtime_is_self_contained() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let site = e
            .python_bin()
            .parent()
            .expect("runtime has a parent")
            .join("Lib")
            .join("site-packages");
        assert!(
            site.join("numpy").join("__init__.py").is_file(),
            "numpy is not vendored at {}; the bridge would fall back to whatever \
             numpy the user's machine happens to have, or fail on a clean one",
            site.join("numpy").display()
        );

        // And the bridge must actually run isolated, so a machine with its own
        // numpy cannot shadow the bundled copy.
        let res = e.predict(&ramp(35)).expect("predict under isolation");
        assert!(res.is_ok());
    }

    /// A child that never exits must cost one request, not a wedged process.
    ///
    /// Uses the real runtime with a script that reads stdin then spins, which is
    /// the shape of bug this guards: `wait_with_output` would block here forever.
    #[test]
    fn a_hanging_script_is_killed_rather_than_waited_on() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let dir = std::env::temp_dir().join("bt_pybridge_hang");
        let _ = std::fs::create_dir_all(&dir);
        let script = dir.join("predictor.py");
        std::fs::write(
            &script,
            "import sys\nsys.stdin.read()\nwhile True:\n    pass\n",
        )
        .expect("write hang script");

        let configured = e
            .with_script(&script)
            .with_timeout(Duration::from_millis(600));
        let started = std::time::Instant::now();
        let err = configured.predict(&ramp(35)).unwrap_err();
        let elapsed = started.elapsed();

        assert!(matches!(err, PyBridgeError::Timeout(_)), "{err}");
        assert!(
            elapsed < Duration::from_secs(10),
            "waited {elapsed:?}, the timeout did not bound the call"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A script that fails loudly must surface its stderr, not a bare exit code.
    #[test]
    fn a_failing_script_reports_its_stderr() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let dir = std::env::temp_dir().join("bt_pybridge_boom");
        let _ = std::fs::create_dir_all(&dir);
        let script = dir.join("predictor.py");
        std::fs::write(
            &script,
            "import sys\nsys.stderr.write('model file missing\\n')\nsys.exit(3)\n",
        )
        .expect("write failing script");

        let err = e.with_script(&script).predict(&ramp(35)).unwrap_err();
        match err {
            PyBridgeError::Failed { status, stderr } => {
                assert_eq!(status, "3");
                assert!(stderr.contains("model file missing"), "{stderr}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The script reports its own failures in-band, with exit code 0.
    #[test]
    fn an_in_band_error_becomes_a_script_error() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let dir = std::env::temp_dir().join("bt_pybridge_inband");
        let _ = std::fs::create_dir_all(&dir);
        let script = dir.join("predictor.py");
        std::fs::write(
            &script,
            "import sys, json\nsys.stdin.read()\nprint(json.dumps({'error': 'bad series'}))\n",
        )
        .expect("write in-band script");

        match e.with_script(&script).predict(&ramp(35)) {
            Err(PyBridgeError::Script(msg)) => assert_eq!(msg, "bad series"),
            other => panic!("expected Script error, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
