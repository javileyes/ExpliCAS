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

    #[test]
    fn full_wire_carries_latex_and_steps() {
        // The W5 contract: the browser gets the SAME rich wire as
        // `cas_cli eval --format json` — latex fields present, steps
        // populated and localized.
        let wire = super::eval_full_wire("solve(x^2-4=0, x)", r#"{"steps": "on", "lang": "es"}"#);
        assert!(wire.contains("\"result_latex\""), "{wire}");
        assert!(wire.contains("\"input_latex\""), "{wire}");
        assert!(wire.contains("\"solve_steps\""), "{wire}");
        let parsed: serde_json::Value = serde_json::from_str(&wire).expect("valid json");
        assert_eq!(parsed["ok"], true, "{wire}");
        assert!(parsed["result_latex"].as_str().is_some(), "{wire}");
    }

    #[test]
    fn full_wire_localizes_steps_en() {
        let es = super::eval_full_wire("dsolve(diff(y,x)=x*y, y, x)", r#"{"steps": "on"}"#);
        assert!(es.contains("Identificar EDO separable"), "{es}");
        let en = super::eval_full_wire(
            "dsolve(diff(y,x)=x*y, y, x)",
            r#"{"steps": "on", "lang": "en"}"#,
        );
        assert!(en.contains("Identify separable ODE"), "{en}");
    }

    #[test]
    fn full_wire_decimal_display_flag_works() {
        let wire = super::eval_full_wire("approx(1/3)", r#"{"numeric_display": "decimal"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&wire).expect("valid json");
        assert_eq!(parsed["ok"], true, "{wire}");
    }

    #[test]
    fn session_hash_references_persist() {
        // The W6 contract: results store as #N and later expressions can
        // reference them — the browser session parity rung.
        let mut session = super::WasmSession::new();
        let first = session.eval("2 + 2", "{}");
        assert!(
            first.contains("\"ok\":true") || first.contains("\"ok\": true"),
            "{first}"
        );
        let second = session.eval("#1 * 10", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&second).expect("valid json");
        assert_eq!(parsed["result"], "40", "{second}");
    }

    #[test]
    fn session_lazy_assignment_persists() {
        let mut session = super::WasmSession::new();
        let assign = session.eval("f(x) := x^2 + 1", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&assign).expect("valid json");
        assert_eq!(parsed["ok"], true, "{assign}");
        let usage = session.eval("f(3)", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&usage).expect("valid json");
        assert_eq!(parsed["result"], "10", "{usage}");
    }

    #[test]
    fn session_clear_resets_references() {
        let mut session = super::WasmSession::new();
        session.eval("5 + 5", "{}");
        session.clear();
        let after = session.eval("#1 + 1", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&after).expect("valid json");
        assert_eq!(parsed["ok"], false, "{after}");
    }

    #[test]
    fn full_wire_error_is_valid_json() {
        let wire = super::eval_full_wire("((", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&wire).expect("valid json");
        assert_eq!(parsed["ok"], false, "{wire}");
    }
}

/// Options accepted by [`eval_full_wire`] — the browser-facing mirror of the
/// CLI flags `web/server.py` forwards. Serde defaults keep `"{}"` valid.
#[derive(serde::Deserialize)]
#[serde(default)]
pub struct FullWireOptions {
    pub steps: String,
    pub lang: String,
    pub domain: String,
    pub inv_trig: String,
    pub complex: String,
    pub numeric_display: String,
    pub time_budget_ms: Option<u64>,
    pub max_chars: usize,
}

impl Default for FullWireOptions {
    fn default() -> Self {
        Self {
            steps: "off".to_string(),
            lang: "es".to_string(),
            domain: "generic".to_string(),
            inv_trig: "strict".to_string(),
            complex: "off".to_string(),
            numeric_display: "exact".to_string(),
            time_budget_ms: None,
            max_chars: 500_000,
        }
    }
}

fn language_of(opts: &FullWireOptions) -> cas_solver_core::eval_option_axes::Language {
    match opts.lang.as_str() {
        "en" => cas_solver_core::eval_option_axes::Language::En,
        _ => cas_solver_core::eval_option_axes::Language::Es,
    }
}

/// Mirror of the CLI's `eval_command_config` defaults, driven by the
/// browser-facing options. `auto_store` distinguishes the stateless entry
/// (off) from the session entry (on: results become `#N`).
fn build_run_config<'a>(
    expr: &'a str,
    opts: &FullWireOptions,
    auto_store: bool,
) -> cas_api_models::EvalSessionRunConfig<'a> {
    cas_api_models::EvalSessionRunConfig {
        expr,
        auto_store,
        max_chars: opts.max_chars,
        time_budget_ms: opts.time_budget_ms,
        steps_mode: match opts.steps.as_str() {
            "on" => cas_api_models::EvalStepsMode::On,
            "compact" => cas_api_models::EvalStepsMode::Compact,
            _ => cas_api_models::EvalStepsMode::Off,
        },
        budget_preset: cas_api_models::EvalBudgetPreset::Standard,
        strict: false,
        domain: match opts.domain.as_str() {
            "strict" => cas_api_models::EvalDomainMode::Strict,
            "assume" => cas_api_models::EvalDomainMode::Assume,
            _ => cas_api_models::EvalDomainMode::Generic,
        },
        context_mode: cas_api_models::EvalContextMode::Auto,
        branch_mode: cas_api_models::EvalBranchMode::Strict,
        expand_policy: cas_api_models::EvalExpandPolicy::Auto,
        complex_mode: cas_api_models::EvalComplexMode::Auto,
        const_fold: cas_api_models::EvalConstFoldMode::Safe,
        value_domain: match opts.complex.as_str() {
            "on" => cas_api_models::EvalValueDomain::Complex,
            _ => cas_api_models::EvalValueDomain::Real,
        },
        complex_branch: cas_api_models::EvalBranchMode::Principal,
        inv_trig: match opts.inv_trig.as_str() {
            "principal" => cas_api_models::EvalInvTrigPolicy::Principal,
            _ => cas_api_models::EvalInvTrigPolicy::Strict,
        },
        assume_scope: cas_api_models::EvalAssumeScope::Real,
        numeric_display: match opts.numeric_display.as_str() {
            "decimal" => cas_api_models::EvalNumericDisplay::Decimal,
            _ => cas_api_models::EvalNumericDisplay::Exact,
        },
    }
}

/// Evaluate one expression against the FULL wire — the same rich JSON the
/// CLI's `eval --format json` emits (input_latex, result_latex, steps,
/// solve_steps, warnings, required_display, stats, timings): what the web UI
/// needs to render LaTeX and narrated steps. Mirrors the CLI's
/// `eval_command_config` defaults; localization follows `lang` ("es"/"en").
#[wasm_bindgen]
pub fn eval_full_wire(expr: &str, opts_json: &str) -> String {
    let opts: FullWireOptions = match serde_json::from_str(opts_json) {
        Ok(o) => o,
        Err(e) => {
            return format!(
                "{{\"ok\":false,\"error\":\"invalid options JSON: {}\"}}",
                e.to_string().replace('"', "'")
            )
        }
    };
    let language = language_of(&opts);
    let config = build_run_config(expr, &opts, false);

    let mut engine = cas_solver::runtime::Engine::new();
    let mut session = cas_solver::runtime::StatelessEvalSession::new(
        cas_solver_core::eval_options::EvalOptions::default(),
    );

    let result = cas_solver::session_api::eval::evaluate_eval_with_session(
        &mut engine,
        &mut session,
        config,
        language,
        |steps, events, ctx, mode| {
            cas_didactic::collect_step_payloads_with_events_localized(
                steps, events, ctx, mode, language,
            )
        },
    );

    match result {
        Ok(wire) => serde_json::to_string(&wire)
            .unwrap_or_else(|e| format!("{{\"ok\":false,\"error\":\"serialize: {e}\"}}")),
        Err(message) => format!(
            "{{\"ok\":false,\"error\":{}}}",
            serde_json::to_string(&message).unwrap_or_else(|_| "\"eval failed\"".to_string())
        ),
    }
}

/// A persistent per-tab session: `#N` references, `:=` assignments and
/// stored results survive across evaluations inside this wasm instance —
/// the last rung of the browser-mode parity (frente W · W6). State lives
/// only in memory: closing the tab is `clear()`.
#[wasm_bindgen]
pub struct WasmSession {
    engine: cas_solver::runtime::Engine,
    state: cas_session::SessionState,
}

impl Default for WasmSession {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmSession {
        WasmSession {
            engine: cas_solver::runtime::Engine::new(),
            state: cas_session::SessionState::new(),
        }
    }

    /// Evaluate with session state (auto-store ON: results become `#N`).
    /// Same options JSON and same FULL wire as [`eval_full_wire`].
    pub fn eval(&mut self, expr: &str, opts_json: &str) -> String {
        let opts: FullWireOptions = match serde_json::from_str(opts_json) {
            Ok(o) => o,
            Err(e) => {
                return format!(
                    "{{\"ok\":false,\"error\":\"invalid options JSON: {}\"}}",
                    e.to_string().replace('"', "'")
                )
            }
        };
        let language = language_of(&opts);
        let config = build_run_config(expr, &opts, true);
        let result = cas_session::eval::evaluate_eval_command_in_memory_with_state(
            &mut self.engine,
            &mut self.state,
            config,
            language,
            |steps, events, ctx, mode| {
                cas_didactic::collect_step_payloads_with_events_localized(
                    steps, events, ctx, mode, language,
                )
            },
        );
        match result {
            Ok(wire) => serde_json::to_string(&wire)
                .unwrap_or_else(|e| format!("{{\"ok\":false,\"error\":\"serialize: {e}\"}}")),
            Err(message) => format!(
                "{{\"ok\":false,\"error\":{}}}",
                serde_json::to_string(&message).unwrap_or_else(|_| "\"eval failed\"".to_string())
            ),
        }
    }

    /// Reset the session (the Limpiar button): fresh engine + empty store.
    pub fn clear(&mut self) {
        self.engine = cas_solver::runtime::Engine::new();
        self.state = cas_session::SessionState::new();
    }
}
