// crates/bt-app/tests/api_server.rs
// Author: Sourish Dey

//! End-to-end checks against a real listening socket.
//!
//! The unit tests in `src/api.rs` drive the router in-process, which proves the
//! handlers but not the plumbing. These tests start the actual server on an
//! ephemeral loopback port and speak HTTP/1.1 to it, which is the only way to
//! catch the failures that unit tests cannot see: a runtime that is dropped the
//! moment the spawning function returns, a bind that silently went to `0.0.0.0`,
//! or a port that was reported but never opened.
//!
//! This is its own integration-test binary, so mutating process-global
//! environment variables here cannot disturb tests in `src/main.rs`.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::sync::OnceLock;
use std::time::Duration;

// The tests need to reach the binary's `api` module, which is private to the
// bin target. Rather than re-exporting internals from `main.rs`, the module source
// is included here so the tests exercise exactly the same code.
//
// `dead_code` is allowed because this build has no `main` to reach
// `spawn_if_enabled` through; in the real binary both are used.
#[allow(dead_code)]
#[path = "../src/api.rs"]
mod api;

/// Start the server once on an ephemeral loopback port.
///
/// Port 0 lets the OS choose, so a stale port or a parallel run cannot make
/// these tests talk to something else.
fn start() -> SocketAddr {
    static ADDR: OnceLock<SocketAddr> = OnceLock::new();
    *ADDR.get_or_init(|| api::spawn(0).expect("server should bind"))
}

/// Issue a GET and return `(status_code, body)`.
fn get(addr: SocketAddr, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
        .unwrap_or_else(|e| panic!("connect to {addr} failed: {e}"));
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let text = String::from_utf8_lossy(&raw).into_owned();

    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or_else(|| panic!("no status code in response: {text}"));
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

#[test]
fn the_server_stays_up_and_answers_after_spawn_returns() {
    let addr = start();

    // If the runtime were dropped when `spawn` returned, this connect would fail.
    let (code, body) = get(addr, "/api/health");
    assert_eq!(code, 200, "health failed: {body}");
    assert!(body.contains("\"ok\":true"), "{body}");
    assert!(body.contains("loopback"), "{body}");
}

#[test]
fn the_listener_is_reachable_on_loopback_only() {
    let addr = start();
    assert!(
        addr.ip().is_loopback(),
        "bound to {} which is not loopback",
        addr.ip()
    );

    // Nothing should be listening on a routable interface. Probing 0.0.0.0 from
    // the host would resolve to a local address, so instead assert the bound
    // address directly and keep the reachability check on 127.0.0.1.
    let _: IpAddr = addr.ip();
    assert_eq!(get(addr, "/api/health").0, 200);
}

#[test]
fn a_cold_start_reports_no_cached_data_rather_than_inventing_it() {
    let addr = start();
    let (code, body) = get(addr, "/api/candles?symbol=ZZZ_NEVER_CACHED&interval=1d");
    assert_eq!(code, 503, "{body}");
    assert!(body.contains("error"), "{body}");
    assert!(
        !body.contains("\"candles\":[{"),
        "invented a series: {body}"
    );
}

#[test]
fn a_forecast_for_an_uncached_symbol_is_503_with_a_reason() {
    let addr = start();
    let (code, body) = get(addr, "/api/forecast?symbol=ZZZ_NEVER_CACHED&interval=1d");
    assert_eq!(code, 503, "{body}");
    assert!(body.contains("error"), "{body}");
    assert!(
        !body.contains("\"predictions\":[{"),
        "invented a forecast: {body}"
    );
}

#[test]
fn an_unknown_engine_is_rejected_rather_than_defaulted() {
    let addr = start();
    let (code, body) = get(addr, "/api/forecast?symbol=ZZZ_NEVER_CACHED&engine=gpt9");
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("gpt9"), "{body}");
}

#[test]
fn engine_status_is_always_answered() {
    let addr = start();
    let (code, body) = get(addr, "/api/engines");
    assert_eq!(code, 200, "{body}");
    // Every engine must carry a verdict and a reason, installed or not.
    assert!(body.contains("TinyTimeMixer"), "{body}");
    assert!(body.contains("\"available\""), "{body}");
}
