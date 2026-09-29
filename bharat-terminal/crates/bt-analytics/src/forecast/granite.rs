// crates/bt-analytics/src/forecast/granite.rs
// Author: Sourish Dey

//! IBM Granite TinyTimeMixer R2 (primary engine) via the `zsfm` CLI.
//!
//! Context 512, native horizon 96. The verified interface (measured live) is:
//! `zsfm ttm infer --gguf <file> --config <file>` reading
//! `{"context": [...], "horizon": N}` from stdin and printing OpenAI-style
//! JSON with the points at `choices[0].forecast.point`. Anything before the
//! first `{` on stdout (e.g. "Loading model..." chatter) is skipped.
//!
//! Lookup is PATH-resilient: `zsfm` on `PATH` first, then the well-known
//! `%USERPROFILE%\.cargo\bin` / `~/.cargo/bin` locations. Every failure mode
//! (missing binary, dead files, bad exit code, unparseable output, timeout)
//! returns a contextual `ForecastError::Granite` so the
//! [`crate::forecast::Forecaster`] falls through to the next engine — the app
//! never hangs or crashes on a broken model setup.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use super::ForecastError;

pub const TTM_CONTEXT: usize = 512;
pub const TTM_HORIZON: usize = 96;
pub const TTM_PATCH_LENGTH: usize = 64;

/// Generous ceiling: a live run measures ~0.2 s, so this only ever fires on a
/// genuinely wedged subprocess.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(60);

pub struct GraniteForecaster {
    gguf_path: String,
    config_path: String,
    zsfm_bin: PathBuf,
}

impl GraniteForecaster {
    pub fn new(gguf_path: &str, config_path: &str) -> Result<Self, ForecastError> {
        if !Path::new(gguf_path).exists() {
            return Err(ForecastError::ModelNotFound(format!(
                "GGUF not found: {gguf_path}"
            )));
        }
        if !Path::new(config_path).exists() {
            return Err(ForecastError::ModelNotFound(format!(
                "Config not found: {config_path}"
            )));
        }
        let zsfm_bin = Self::locate_zsfm()?;
        tracing::info!(
            "Granite TTM R2 ready: gguf={}, config={}, zsfm={}",
            gguf_path,
            config_path,
            zsfm_bin.display()
        );
        Ok(Self {
            gguf_path: gguf_path.to_string(),
            config_path: config_path.to_string(),
            zsfm_bin,
        })
    }

    /// Absolute path of the resolved CLI, for diagnostics.
    pub fn cli_path(&self) -> &Path {
        &self.zsfm_bin
    }

    /// Find a working `zsfm`: `PATH` first (validated by actually running the
    /// `ttm infer` subcommand path), then the standard cargo-bin locations.
    /// NOTE: `zsfm --version` does not exist, so probing with it would report
    /// a healthy install as missing — probe `ttm infer --help` instead.
    fn locate_zsfm() -> Result<PathBuf, ForecastError> {
        if Command::new("zsfm")
            .args(["ttm", "infer", "--help"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            tracing::debug!("found zsfm on PATH");
            return Ok(PathBuf::from("zsfm"));
        }
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .unwrap_or_default();
        if !home.is_empty() {
            for candidate in [
                format!("{home}\\.cargo\\bin\\zsfm.exe"),
                format!("{home}/.cargo/bin/zsfm"),
            ] {
                let path = PathBuf::from(&candidate);
                if path.is_file()
                    && Command::new(&path)
                        .args(["ttm", "infer", "--help"])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false)
                {
                    tracing::info!("found zsfm at {}", path.display());
                    return Ok(path);
                }
            }
        }
        Err(ForecastError::Granite(
            "zsfm CLI not found on PATH or in ~/.cargo/bin".into(),
        ))
    }

    pub fn predict(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        if history.is_empty() {
            return Err(ForecastError::EmptyHistory);
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
        let request = format!(
            "{{\"context\": [{}], \"horizon\": {}}}",
            context.join(","),
            horizon
        );
        tracing::debug!(
            "Granite call: context_len={}, horizon={}, zsfm={}",
            context.len(),
            horizon,
            self.zsfm_bin.display()
        );

        let mut child = Command::new(&self.zsfm_bin)
            .args([
                "ttm",
                "infer",
                "--gguf",
                &self.gguf_path,
                "--config",
                &self.config_path,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                ForecastError::Granite(format!(
                    "zsfm spawn failed ({}): {e}",
                    self.zsfm_bin.display()
                ))
            })?;
        // Take (not borrow) stdin so the pipe is closed — and EOF delivered —
        // as soon as the write finishes. The CLI reads stdin to EOF before
        // inferring; leaving our end open wedges it forever.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(request.as_bytes()).map_err(|e| {
                let _ = child.kill();
                ForecastError::Granite(format!("zsfm stdin write: {e}"))
            })?;
        } else {
            let _ = child.kill();
            return Err(ForecastError::Granite("failed to open zsfm stdin".into()));
        }

        // Read stdout on a side thread so a wedged child can be killed: a
        // blocking read on the main thread could never time out. The exit
        // status is collected on this thread afterwards; by the time both
        // streams hit EOF the child has necessarily exited.
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut out = Vec::new();
            let mut err = Vec::new();
            let out_ok = stdout
                .as_mut()
                .map(|s| s.read_to_end(&mut out).is_ok())
                .unwrap_or(false);
            let err_text = stderr
                .as_mut()
                .map(|s| {
                    let _ = s.read_to_end(&mut err);
                    String::from_utf8_lossy(&err).into_owned()
                })
                .unwrap_or_default();
            let _ = tx.send((out_ok, out, err_text));
        });
        let (ok, buf, err_text) = rx.recv_timeout(SUBPROCESS_TIMEOUT).map_err(|_| {
            let _ = child.kill();
            ForecastError::Granite("zsfm timed out after 60s".into())
        })?;
        let status = child
            .wait()
            .ok()
            .and_then(|s| if s.success() { Some(()) } else { None });
        if !ok {
            return Err(ForecastError::Granite(format!(
                "zsfm produced no output (stderr: {})",
                truncate(&err_text, 200)
            )));
        }
        if status.is_none() {
            return Err(ForecastError::Granite(format!(
                "zsfm exited with failure (stderr: {})",
                truncate(&err_text, 200)
            )));
        }

        // Skip any log prefix ("Loading model...") up to the JSON payload.
        let stdout = String::from_utf8_lossy(&buf);
        let json_start = stdout.find('{').ok_or_else(|| {
            ForecastError::Granite(format!(
                "no JSON in zsfm output: {}",
                truncate(&stdout, 200)
            ))
        })?;
        let parsed: serde_json::Value = serde_json::from_str(&stdout[json_start..])
            .map_err(|e| ForecastError::Granite(format!("zsfm JSON parse: {e}")))?;
        let values: Vec<f64> = parsed["choices"][0]["forecast"]["point"]
            .as_array()
            .ok_or_else(|| ForecastError::Granite("missing choices[0].forecast.point".into()))?
            .iter()
            .filter_map(|v| v.as_f64())
            .take(horizon)
            .collect();
        if values.is_empty() {
            return Err(ForecastError::Granite("empty forecast array".into()));
        }
        tracing::info!("Granite TTM produced {} forecast points", values.len());
        Ok(values)
    }
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
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
        let dir = std::env::temp_dir();
        let gguf = dir.join("bt-test-ttm.gguf");
        let cfg = dir.join("bt-test-ttm-config.json");
        std::fs::write(&gguf, b"stub").unwrap();
        std::fs::write(&cfg, b"{}").unwrap();
        // Construction needs a working zsfm; without one this is still a
        // valid rejection path, just at an earlier stage.
        match GraniteForecaster::new(gguf.to_str().unwrap(), cfg.to_str().unwrap()) {
            Ok(f) => assert!(f.predict(&[], 10).is_err()),
            Err(_) => {}
        }
        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(&cfg);
    }

    #[test]
    fn test_short_history_is_rejected_without_spawning() {
        let dir = std::env::temp_dir();
        let gguf = dir.join("bt-test-ttm2.gguf");
        let cfg = dir.join("bt-test-ttm2-config.json");
        std::fs::write(&gguf, b"stub").unwrap();
        std::fs::write(&cfg, b"{}").unwrap();
        // 3 points < 64 patch minimum: rejected before any subprocess runs,
        // but only if a zsfm exists to construct with; otherwise construction
        // itself is the (correct) rejection.
        if let Ok(f) = GraniteForecaster::new(gguf.to_str().unwrap(), cfg.to_str().unwrap()) {
            assert!(f.predict(&[1.0, 2.0, 3.0], 5).is_err());
        }
        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(&cfg);
    }

    /// Live end-to-end run against the real CLI and weights. Ignored by
    /// default (needs `zsfm` + `models/`); run explicitly to prove the path:
    /// `cargo test -p bt-analytics -- --ignored granite_live`
    #[test]
    #[ignore]
    fn granite_live_end_to_end() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let f = GraniteForecaster::new(
            root.join("ttm-q8.gguf").to_str().unwrap(),
            root.join("config.json").to_str().unwrap(),
        )
        .expect("needs zsfm + models/");
        let history: Vec<f64> = (0..120)
            .map(|i| 100.0 + i as f64 * 0.5 + 3.0 * ((i as f64 * 0.7).sin()))
            .collect();
        let v = f.predict(&history, 10).expect("live inference");
        assert_eq!(v.len(), 10);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
