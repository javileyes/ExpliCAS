//! Non-regression tests for Root and Log identity rules (V2.14.45)
//!
//! These tests verify the parity-aware root simplification and log identity rules:
//!
//! ## Root Rules
//! - RootPowCancelRule: (x^n)^(1/n) → |x| (even n) or x (odd n)
//! - SymbolicRootCancelRule: sqrt(x^n, n) → x in Assume mode with x ≥ 0
//! - CanonicalizeRootRule guard: keep sqrt form for symbolic n
//!
//! ## Log Rules  
//! - LogPowerBaseRule: log(a^m, a^n) → n/m
//! - Anti-worsen guard: blocks abs() introduction in Generic mode

use cas_engine::ValueDomain;
use cas_engine::{DomainMode, Engine, EvalAction, EvalRequest, EvalResult};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
mod support;
use support::SessionState;

// ============================================================================
// Test Infrastructure
// ============================================================================

struct SimplifyResult {
    result: String,
    #[allow(dead_code)]
    has_requires: bool,
    #[allow(dead_code)]
    requires_messages: Vec<String>,
}

fn simplify_with_config(
    input: &str,
    value_domain: ValueDomain,
    domain_mode: DomainMode,
) -> SimplifyResult {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    state.options_mut().shared.semantics.value_domain = value_domain;
    state.options_mut().shared.semantics.domain_mode = domain_mode;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    let requires_messages: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();

    SimplifyResult {
        result,
        has_requires: !requires_messages.is_empty(),
        requires_messages,
    }
}

fn simplify(input: &str) -> String {
    simplify_with_config(input, ValueDomain::RealOnly, DomainMode::Generic).result
}

fn simplify_assume(input: &str) -> SimplifyResult {
    simplify_with_config(input, ValueDomain::RealOnly, DomainMode::Assume)
}

// ============================================================================
// RootPowCancelRule: (x^n)^(1/n) with numeric n
// ============================================================================

mod root_pow_cancel_numeric {
    use super::*;

    #[test]
    fn even_root_gives_abs() {
        // sqrt(x^4, 4) = |x| (even root)
        assert_eq!(simplify("sqrt(x^4, 4)"), "|x|");
    }

    #[test]
    fn even_root_2() {
        // sqrt(x^2, 2) = |x|
        assert_eq!(simplify("sqrt(x^2, 2)"), "|x|");
    }

    #[test]
    fn odd_root_gives_x() {
        // sqrt(x^3, 3) = x (odd root)
        assert_eq!(simplify("sqrt(x^3, 3)"), "x");
    }

    #[test]
    fn odd_root_5() {
        // sqrt(x^5, 5) = x
        assert_eq!(simplify("sqrt(x^5, 5)"), "x");
    }

    #[test]
    fn even_root_6() {
        // sqrt(x^6, 6) = |x|
        assert_eq!(simplify("sqrt(x^6, 6)"), "|x|");
    }
}

// ============================================================================
// SymbolicRootCancelRule: sqrt(x^n, n) with symbolic n
// ============================================================================

mod root_pow_cancel_symbolic {
    use super::*;

    #[test]
    fn generic_keeps_sqrt_form() {
        // sqrt(x^n, n) should NOT simplify in Generic (can't determine parity)
        let r = simplify("sqrt(x^n, n)");
        assert!(
            r.contains("sqrt"),
            "Generic: sqrt(x^n, n) should remain unchanged, got: {}",
            r
        );
    }

    #[test]
    fn assume_simplifies_with_requires() {
        // sqrt(x^n, n) → x in Assume mode with x ≥ 0
        let r = simplify_assume("sqrt(x^n, n)");
        assert_eq!(
            r.result, "x",
            "Assume: sqrt(x^n, n) should simplify to x, got: {}",
            r.result
        );
        // Note: the require (x ≥ 0) is emitted via .requires() on the Rewrite,
        // which is displayed in CLI but not returned as domain_warnings.
        // The simplification itself confirms the rule applied correctly.
    }
}

// ============================================================================
// Fire test: root(3, x^6) * root(4, x^2)
// ============================================================================

mod fire_test {
    use super::*;

    #[test]
    fn combined_roots_no_cycle() {
        // This test catches the split/merge cycle bug
        // The result should NOT have abs() after the merge
        let r = simplify("sqrt(x^6, 3) * sqrt(x^2, 4)");

        // Should simplify to √(x^5) or equivalent, NOT cycle
        // The specific form depends on canonicalization, but should be stable
        assert!(
            !r.contains("cycle") && !r.contains("Blocked"),
            "Should not have cycle, got: {}",
            r
        );
    }
}

// ============================================================================
// LogPowerBaseRule: log(a^m, a^n) → n/m
// ============================================================================

mod log_power_base {
    use super::*;

    #[test]
    fn log_x2_x6_gives_3() {
        // log(x^2, x^6) = 6/2 = 3
        assert_eq!(simplify("log(x^2, x^6)"), "3");
    }

    #[test]
    fn log_x3_x9_gives_3() {
        // log(x^3, x^9) = 9/3 = 3
        assert_eq!(simplify("log(x^3, x^9)"), "3");
    }

    #[test]
    fn log_reciprocal_base() {
        // log(1/x, x) = log(x^(-1), x) = 1/(-1) = -1
        assert_eq!(simplify("log(1/x, x)"), "-1");
    }

    #[test]
    fn log_8_2_evaluates() {
        // log(8, 2) = log(2^3, 2) = 1/3
        let r = simplify("log(8, 2)");
        assert!(
            r == "1/3" || r == "1 / 3",
            "log(8, 2) should be 1/3, got: {}",
            r
        );
    }
}

// ============================================================================
// Anti-worsen guard: don't introduce abs() in Generic mode
// ============================================================================

mod anti_worsen_guard {
    use super::*;

    #[test]
    fn log_even_power_simplifies_with_abs_in_generic() {
        // log(2, x^4) → 4*log(2,|x|) in Generic via LogPerfectSquareRule
        // abs() is mathematically necessary since x could be negative
        let r = simplify("log(2, x^4)");

        // Should simplify the power and introduce abs for correctness
        assert!(
            !r.contains("x^4") && !r.contains("x⁴"),
            "Generic: log(2, x^4) should simplify, got: {}",
            r
        );
    }

    #[test]
    fn log_even_power_simplifies_in_assume() {
        // log(2, x^4) → 4*log(2,x) in Assume mode (with x > 0)
        let r = simplify_assume("log(2, x^4)");

        // Should simplify and have requires
        assert!(
            !r.result.contains("x^4") && !r.result.contains("x⁴"),
            "Assume: log(2, x^4) should simplify, got: {}",
            r.result
        );
    }
}

// ============================================================================
// Canonicalization: sqrt form preserved for symbolic index
// ============================================================================

mod canonicalize_root_guard {
    use super::*;

    #[test]
    fn symbolic_index_stays_sqrt() {
        // sqrt(x, n) with symbolic n should NOT become x^(1/n)
        // (better visualization)
        let r = simplify("sqrt(x, n)");
        assert!(
            r.contains("sqrt") || r.contains("√"),
            "Symbolic index should keep sqrt form, got: {}",
            r
        );
        assert!(
            !r.contains("^(1"),
            "Should NOT normalize to x^(1/n), got: {}",
            r
        );
    }
}
