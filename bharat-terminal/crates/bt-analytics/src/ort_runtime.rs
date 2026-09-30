// crates/bt-analytics/src/ort_runtime.rs
// Author: Sourish Dey

//! Pinned ONNX Runtime initialization.
//!
//! The `ort` crate with `load-dynamic` dlopens `onnxruntime.dll` at runtime.
//! Letting the OS resolve that name is dangerous: this machine, for example,
//! carries an incompatible 1.17-era inbox build in System32, and loading it
//! crashes the process natively (0xC0000409) instead of returning an error.
//! So every ONNX engine initializes exclusively through
//! [`ensure_initialized`], which points `ort` at the exact 1.28.0 build the
//! `ort-sys` version was generated against. No DLL, no inference — a plain
//! error the fallback chain handles, never a crash.
//!
//! Expected locations, in order:
//! 1. next to the executable (`onnxruntime.dll`, shipped by the installer)
//! 2. `native/onnxruntime.dll` under the executable's ancestors (covers
//!    `target/debug/deps/<test-bin>` finding `<project>/native`)
//! 3. `native/onnxruntime.dll` under the working directory or its ancestors
//!    (developer runs from the project root)

use std::path::PathBuf;
use std::sync::OnceLock;

/// The process-wide result of the one-time load.
///
/// `load_dynamic::init` is not safe to race: two threads calling
/// `ort::init_from` concurrently can end up with different libraries bound to
/// the same global slot, and the loser reports either a bogus version
/// (`BadVersion 1.17.1`) or a missing `OrtGetApiBase`. Every engine is lazily
/// loaded and tests run in parallel, so this has to be serialized rather than
/// merely idempotent.
static RUNTIME: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Load the pinned ONNX Runtime exactly once per process. Returns the DLL path
/// used, or the same error for every later caller.
pub fn ensure_initialized() -> Result<PathBuf, String> {
    RUNTIME.get_or_init(load_pinned).clone()
}

fn load_pinned() -> Result<PathBuf, String> {
    // Pin the search path *before* touching any ort API.
    //
    // `ort`'s lazy `setup_api` resolves the bare name `onnxruntime.dll` through
    // the OS loader, which on this machine finds the 1.17-era inbox copy in
    // System32 and then `panic!`s on the version check. With `panic = "abort"`
    // that is a hard process kill, and it fires from whichever code path happens
    // to be the first to touch the API. `ORT_DYLIB_PATH` is ort's documented
    // override and is honoured by that lazy path, so setting it here closes the
    // hole for every engine, including ones added later.
    if std::env::var_os(ENV_DYLIB_PATH).is_none() {
        if let Some(best) = candidate_paths().into_iter().find(|p| p.is_file()) {
            std::env::set_var(ENV_DYLIB_PATH, best);
        }
    }

    for candidate in candidate_paths() {
        if !candidate.is_file() {
            continue;
        }
        match ort::init_from(&candidate) {
            Ok(env) => {
                env.with_name("bharat-terminal").commit();
                tracing::info!("ONNX Runtime loaded: {}", candidate.display());
                return Ok(candidate);
            }
            Err(e) => {
                tracing::warn!("ONNX Runtime at {} unusable: {e}", candidate.display());
            }
        }
    }
    Err("no usable onnxruntime.dll next to the executable or in native/".into())
}

/// The override ort's dynamic loader reads.
const ENV_DYLIB_PATH: &str = "ORT_DYLIB_PATH";

fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("onnxruntime.dll"));
            // Walk up for cargo layouts: target/debug/deps/<test-bin> must
            // still find <project>/native/onnxruntime.dll.
            let mut ancestor = dir.to_path_buf();
            for _ in 0..4 {
                if let Some(parent) = ancestor.parent() {
                    ancestor = parent.to_path_buf();
                    out.push(ancestor.join("native").join("onnxruntime.dll"));
                } else {
                    break;
                }
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut ancestor = cwd;
        for _ in 0..4 {
            out.push(ancestor.join("native").join("onnxruntime.dll"));
            if let Some(parent) = ancestor.parent() {
                ancestor = parent.to_path_buf();
            } else {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidates_are_absolute_or_relative_but_sane() {
        // Must not panic and must yield at least the exe-anchored candidate.
        let paths = candidate_paths();
        assert!(!paths.is_empty());
        assert!(paths.iter().all(|p| p.ends_with("onnxruntime.dll")));
    }

    /// The pinned runtime must be 1.28, never the System32 shadow.
    ///
    /// This is the invariant that keeps the app off the 1.17-era inbox build
    /// that `ort` would otherwise pick up by bare name and `panic!` on.
    #[test]
    fn the_pinned_runtime_is_the_1_28_build() {
        if let Ok(path) = ensure_initialized() {
            let bytes = std::fs::read(&path).unwrap_or_default();
            let text = String::from_utf8_lossy(&bytes);
            assert!(
                text.contains("1.28"),
                "pinned runtime at {} does not look like 1.28",
                path.display()
            );
        }
    }

    #[test]
    fn the_dylib_override_points_at_an_absolute_existing_file() {
        // `ensure_initialized` installs this; re-run to observe the result.
        let _ = ensure_initialized();
        if let Some(v) = std::env::var_os(ENV_DYLIB_PATH) {
            let p = std::path::PathBuf::from(v);
            assert!(
                p.is_absolute(),
                "relative ORT_DYLIB_PATH re-opens the shadow"
            );
            assert!(
                p.is_file(),
                "ORT_DYLIB_PATH points at a missing file: {p:?}"
            );
        }
    }

    #[test]
    fn concurrent_callers_agree() {
        let handles: Vec<_> = (0..16)
            .map(|_| std::thread::spawn(|| ensure_initialized().map(|p| p.display().to_string())))
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for r in &results {
            assert_eq!(r, &results[0], "callers disagreed: {results:?}");
        }
    }
}
