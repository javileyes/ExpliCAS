//! WebAssembly entry point for ExpliCAS (frente W · W2).
//!
//! Exposes the SAME stateless JSON wire that `cas_cli eval --format json`
//! and `web/server.py` speak — one contract, three transports (CLI, HTTP,
//! browser). The wasm-bindgen layer is deliberately thin: all mathematics,
//! verification gates, and honesty contracts live in `cas_solver` and below,
//! already validated by the native test suite and compiled unchanged to
//! wasm32 (W1: `make wasm-check`).
//!
//! Options JSON is `EvalRunOptions` (`{"steps": bool, "pretty": bool,
//! "budget": {"preset": "small"|"cli"|"unlimited", "mode": "strict"|
//! "best-effort"}}`); the response is `EngineWireResponse` schema v1 —
//! always valid JSON, even on errors. Session state and the full CLI flag
//! surface (lang/domain/value-domain/numeric-display) are named future
//! rungs, not silent gaps.

use wasm_bindgen::prelude::*;

/// Evaluate one expression against the stateless wire.
///
/// * `expr` — the input exactly as a user would type it in the CLI.
/// * `opts_json` — `EvalRunOptions` JSON; `"{}"` gives defaults
///   (budget preset "cli", steps off).
///
/// Returns the `EngineWireResponse` JSON string (schema v1).
#[wasm_bindgen]
pub fn eval_str_to_wire(expr: &str, opts_json: &str) -> String {
    cas_solver::wire::eval_str_to_wire(expr, opts_json)
}

/// Crate version, so the page can display which engine build it runs.
#[wasm_bindgen]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    // Native tests of the SAME functions the browser calls (the rlib
    // crate-type exists exactly for this): the wire contract is asserted
    // here once and holds for every transport.

    #[test]
    fn wire_eval_basic_arithmetic_is_ok() {
        let wire = super::eval_str_to_wire("2 + 2", "{}");
        assert!(
            wire.contains("\"ok\":true") || wire.contains("\"ok\": true"),
            "{wire}"
        );
        assert!(wire.contains('4'), "{wire}");
    }

    #[test]
    fn wire_eval_honest_decline_is_still_ok_json() {
        // The honesty contract crosses the transport: a non-elementary
        // integral stays an honest residual, never an error blob.
        let wire = super::eval_str_to_wire("integrate(e^(x^2), x)", "{}");
        assert!(
            wire.contains("\"ok\":true") || wire.contains("\"ok\": true"),
            "{wire}"
        );
    }

    #[test]
    fn wire_eval_parse_error_is_valid_json_error() {
        let wire = super::eval_str_to_wire("((", "{}");
        assert!(
            wire.contains("\"ok\":false") || wire.contains("\"ok\": false"),
            "{wire}"
        );
    }

    #[test]
    fn wire_eval_steps_flag_round_trips() {
        let wire = super::eval_str_to_wire("solve(x^2-4=0, x)", r#"{"steps": true}"#);
        assert!(
            wire.contains("\"ok\":true") || wire.contains("\"ok\": true"),
            "{wire}"
        );
    }

    #[test]
    fn engine_version_is_nonempty() {
        assert!(!super::engine_version().is_empty());
    }
}
