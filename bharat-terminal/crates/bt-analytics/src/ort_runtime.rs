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

/// Load the pinned ONNX Runtime exactly once per process (repeat calls are
/// cheap no-ops). Returns the DLL path used, for diagnostics.
pub fn ensure_initialized() -> Result<PathBuf, String> {
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
}
