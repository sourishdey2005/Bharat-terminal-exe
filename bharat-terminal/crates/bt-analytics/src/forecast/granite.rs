// crates/bt-analytics/src/forecast/granite.rs
// Author: Sourish Dey

//! IBM Granite TinyTimeMixer R2 (primary engine).
//!
//! Context 512, horizon 96, patch length 64. Inference runs through the
//! `ttm-rs` CLI, which must be on `PATH` alongside `models/ttm-q8.gguf` and
//! `models/config.json`. When the binary or the files are missing — or the
//! subprocess fails or times out — prediction returns an error and the
//! [`crate::forecast::Forecaster`] falls through to the next engine, so the
//! app never hangs or crashes on a missing model.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use super::ForecastError;

pub const TTM_CONTEXT: usize = 512;
pub const TTM_HORIZON: usize = 96;
pub const TTM_PATCH_LENGTH: usize = 64;

/// How long to wait for one `ttm-rs` invocation before killing it.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(60);

pub struct GraniteForecaster {
    gguf_path: String,
    config_path: String,
    cli_available: bool,
}

impl GraniteForecaster {
    pub fn new(gguf_path: &str, config_path: &str) -> Result<Self, ForecastError> {
        if !Path::new(gguf_path).exists() {
            return Err(ForecastError::ModelNotFound(gguf_path.to_string()));
        }
        if !Path::new(config_path).exists() {
            return Err(ForecastError::ModelNotFound(config_path.to_string()));
        }
        let cli_available = find_on_path("ttm-rs").is_some();
        if !cli_available {
            tracing::warn!(
                "ttm-rs CLI not found on PATH; Granite predictions will fall through to the next engine"
            );
        }
        tracing::info!("Granite TTM R2 ready: {}", gguf_path);
        Ok(Self {
            gguf_path: gguf_path.to_string(),
            config_path: config_path.to_string(),
            cli_available,
        })
    }

    /// Whether the `ttm-rs` binary was found at construction time.
    pub fn cli_available(&self) -> bool {
        self.cli_available
    }

    pub fn predict(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        if history.is_empty() {
            return Err(ForecastError::EmptyHistory);
        }
        if !self.cli_available {
            return Err(ForecastError::Granite(
                "ttm-rs CLI not installed; install it or rely on the fallback chain".into(),
            ));
        }
        if history.len() < TTM_PATCH_LENGTH {
            return Err(ForecastError::InsufficientData(
                TTM_PATCH_LENGTH,
                history.len(),
            ));
        }

        let take = history.len().min(TTM_CONTEXT);
        let context: Vec<String> = history[history.len() - take..]
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect();

        let mut child = Command::new("ttm-rs")
            .args([
                "infer",
                "--gguf",
                &self.gguf_path,
                "--config",
                &self.config_path,
                "--data",
                &context.join(","),
                "--horizon",
                &horizon.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| ForecastError::Granite(format!("failed to launch ttm-rs: {}", e)))?;

        // Wait with a timeout so a wedged subprocess can never freeze the app.
        let (tx, rx) = mpsc::channel();
        let mut stdout = child.stdout.take();
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let status = stdout
                .as_mut()
                .map(|s| s.read_to_end(&mut buf).is_ok())
                .unwrap_or(false);
            let _ = tx.send((status, buf));
        });
        let (ok, buf) = rx.recv_timeout(SUBPROCESS_TIMEOUT).map_err(|_| {
            let _ = child.kill();
            ForecastError::Granite("ttm-rs timed out after 60s".into())
        })?;
        let _ = child.wait();
        if !ok {
            return Err(ForecastError::Granite("ttm-rs produced no output".into()));
        }

        let values: Vec<f64> = String::from_utf8_lossy(&buf)
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .take(horizon)
            .collect();
        if values.is_empty() {
            return Err(ForecastError::Granite(
                "ttm-rs returned no parseable values".into(),
            ));
        }
        Ok(values)
    }
}

/// Minimal `PATH` lookup for an executable name (with `.exe` on Windows).
fn find_on_path(name: &str) -> Option<std::path::PathBuf> {
    let exe = if cfg!(windows) {
        format!("{}.exe", name)
    } else {
        name.to_string()
    };
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(&exe);
            candidate.is_file().then_some(candidate)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_files_are_rejected() {
        assert!(GraniteForecaster::new("no/such/model.gguf", "no/such/config.json").is_err());
    }

    #[test]
    fn test_empty_history_is_rejected() {
        // Construction may succeed if the repo ships the files; either way,
        // an empty history must be rejected, never attempted.
        let f = GraniteForecaster::new("models/ttm-q8.gguf", "models/config.json");
        if let Ok(f) = f {
            assert!(f.predict(&[], 10).is_err());
        }
    }

    #[test]
    fn test_short_history_is_rejected_without_spawning() {
        let dir = std::env::temp_dir();
        let gguf = dir.join("bt-test-ttm.gguf");
        let cfg = dir.join("bt-test-ttm-config.json");
        std::fs::write(&gguf, b"stub").unwrap();
        std::fs::write(&cfg, b"{}").unwrap();
        let f = GraniteForecaster::new(gguf.to_str().unwrap(), cfg.to_str().unwrap()).unwrap();
        // 3 points < 64 patch minimum: rejected before any subprocess runs.
        assert!(f.predict(&[1.0, 2.0, 3.0], 5).is_err());
        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(&cfg);
    }
}
