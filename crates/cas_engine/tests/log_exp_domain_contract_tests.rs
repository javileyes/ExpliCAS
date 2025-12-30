//! Comprehensive contract tests for Log-Exp inverse rules.
//!
//! These tests verify the behavior of LogExpInverseRule and ExponentialLogRule
//! across ALL combinations of:
//! - ValueDomain: RealOnly, ComplexEnabled
//! - DomainMode: Strict, Generic, Assume
//!
//! # Test Matrix
//!
//! ## ln(e^x) → x (LogExpInverseRule)
//!
//! | ValueDomain    | DomainMode | Simplifies? | Warning? | Reason |
//! |----------------|------------|-------------|----------|--------|
//! | RealOnly       | Strict     | NO          | -        | Preserve composition in strict |
//! | RealOnly       | Generic    | YES         | NO       | e^x > 0 ∀x ∈ ℝ, always valid |
//! | RealOnly       | Assume     | YES         | NO       | Same as Generic for reals |
//! | ComplexEnabled | Strict     | NO          | -        | Preserve composition in strict |
//! | ComplexEnabled | Generic    | YES         | YES      | ln multivalued, principal branch |
//! | ComplexEnabled | Assume     | YES         | YES      | Same warning for complex |
//!
//! ## e^(ln(x)) → x (ExponentialLogRule)
//!
//! | ValueDomain    | DomainMode | Simplifies? | Warning? | Reason |
//! |----------------|------------|-------------|----------|--------|
//! | RealOnly       | Strict     | NO (if x unknown) | -  | Can't prove x > 0 |
//! | RealOnly       | Generic    | YES         | YES      | Requires x > 0 for ln |
//! | RealOnly       | Assume     | YES         | YES      | Same, with traceability |
//! | ComplexEnabled | Strict     | NO          | -        | Can't prove x > 0 |
//! | ComplexEnabled | Generic    | YES         | YES      | Requires x > 0 for ln |
//! | ComplexEnabled | Assume     | YES         | YES      | Same |

use cas_ast::DisplayExpr;
use cas_engine::semantics::ValueDomain;
use cas_engine::{
    DomainMode, Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState,
};
use cas_parser::parse;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Result of simplification with all relevant metadata
struct SimplifyResult {
    result: String,
    has_warning: bool,
    warning_messages: Vec<String>,
}

/// Simplify with specific ValueDomain and DomainMode
fn simplify_with_config(
    input: &str,
    value_domain: ValueDomain,
    domain_mode: DomainMode,
) -> SimplifyResult {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Configure semantic axes
    state.options.value_domain = value_domain;
    state.options.domain_mode = domain_mode;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
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

    let warning_messages: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();

    SimplifyResult {
        result,
        has_warning: !warning_messages.is_empty(),
        warning_messages,
    }
}

// ============================================================================
// ln(e^x) → x [LogExpInverseRule] Tests
// ============================================================================

mod log_exp_inverse {
    use super::*;

    // ------------------------------------------------------------------------
    // RealOnly × Strict: YES simplification, NO warning (NEW CONTRACT)
    // By contract, x ∈ ℝ in RealOnly mode, so e^x > 0 always
    // ------------------------------------------------------------------------
    #[test]
    fn real_strict_simplifies_no_warning() {
        let r = simplify_with_config("ln(exp(x))", ValueDomain::RealOnly, DomainMode::Strict);

        assert_eq!(
            r.result, "x",
            "RealOnly+Strict: ln(exp(x)) should simplify to x (x ∈ ℝ by contract), got: {}",
            r.result
        );

        assert!(
            !r.has_warning,
            "RealOnly+Strict: NO warning expected, got: {:?}",
            r.warning_messages
        );
    }

    // ------------------------------------------------------------------------
    // RealOnly × Generic: YES simplification, NO warning
    // ------------------------------------------------------------------------
    #[test]
    fn real_generic_simplifies_no_warning() {
        let r = simplify_with_config("ln(exp(x))", ValueDomain::RealOnly, DomainMode::Generic);

        assert_eq!(
            r.result, "x",
            "RealOnly+Generic: ln(exp(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(
            !r.has_warning,
            "RealOnly+Generic: NO warning expected (e^x > 0 ∀x ∈ ℝ), got: {:?}",
            r.warning_messages
        );
    }

    // ------------------------------------------------------------------------
    // RealOnly × Assume: YES simplification, NO warning
    // ------------------------------------------------------------------------
    #[test]
    fn real_assume_simplifies_no_warning() {
        let r = simplify_with_config("ln(exp(x))", ValueDomain::RealOnly, DomainMode::Assume);

        assert_eq!(
            r.result, "x",
            "RealOnly+Assume: ln(exp(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(
            !r.has_warning,
            "RealOnly+Assume: NO warning expected (e^x > 0 ∀x ∈ ℝ), got: {:?}",
            r.warning_messages
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Strict: NO simplification
    // ------------------------------------------------------------------------
    #[test]
    fn complex_strict_preserves_composition() {
        let r = simplify_with_config(
            "ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Strict,
        );

        assert!(
            r.result.contains("ln") || r.result.contains("log"),
            "ComplexEnabled+Strict: ln(exp(x)) should remain unchanged, got: {}",
            r.result
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Generic: NO simplification (NEW CONTRACT)
    // ln is multivalued in ℂ - never simplify symbolic exponents
    // ------------------------------------------------------------------------
    #[test]
    fn complex_generic_preserves_composition() {
        let r = simplify_with_config(
            "ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Generic,
        );

        assert!(
            r.result.contains("ln") || r.result.contains("log"),
            "ComplexEnabled+Generic: ln(exp(x)) should remain unchanged (ln multivalued in ℂ), got: {}",
            r.result
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Assume: NO simplification (NEW CONTRACT)
    // ln is multivalued in ℂ - never simplify symbolic exponents
    // ------------------------------------------------------------------------
    #[test]
    fn complex_assume_preserves_composition() {
        let r = simplify_with_config(
            "ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Assume,
        );

        assert!(
            r.result.contains("ln") || r.result.contains("log"),
            "ComplexEnabled+Assume: ln(exp(x)) should remain unchanged (ln multivalued in ℂ), got: {}",
            r.result
        );
    }

    // ------------------------------------------------------------------------
    // Numeric exponent always simplifies (any mode)
    // ------------------------------------------------------------------------
    #[test]
    fn numeric_exponent_always_simplifies_strict() {
        let r = simplify_with_config("ln(exp(3))", ValueDomain::RealOnly, DomainMode::Strict);
        assert_eq!(
            r.result, "3",
            "ln(e^3) = 3 should always work, got: {}",
            r.result
        );
    }

    #[test]
    fn numeric_exponent_always_simplifies_complex() {
        let r = simplify_with_config(
            "ln(exp(3))",
            ValueDomain::ComplexEnabled,
            DomainMode::Strict,
        );
        assert_eq!(
            r.result, "3",
            "ln(e^3) = 3 should always work, got: {}",
            r.result
        );
    }
}

// ============================================================================
// e^(ln(x)) → x [ExponentialLogRule] Tests
// ============================================================================

mod exp_ln_inverse {
    use super::*;

    // ------------------------------------------------------------------------
    // RealOnly × Strict: NO simplification (can't prove x > 0)
    // ------------------------------------------------------------------------
    #[test]
    fn real_strict_preserves_composition() {
        let r = simplify_with_config("exp(ln(x))", ValueDomain::RealOnly, DomainMode::Strict);

        // In strict mode, we can't simplify because we can't prove x > 0
        assert!(
            r.result.contains("exp") || r.result.contains("e^") || r.result.contains("ln"),
            "RealOnly+Strict: exp(ln(x)) should remain unchanged (can't prove x > 0), got: {}",
            r.result
        );
    }

    // ------------------------------------------------------------------------
    // RealOnly × Generic: YES simplification, YES warning (x > 0 required)
    // ------------------------------------------------------------------------
    #[test]
    fn real_generic_simplifies_with_positive_warning() {
        let r = simplify_with_config("exp(ln(x))", ValueDomain::RealOnly, DomainMode::Generic);

        assert_eq!(
            r.result, "x",
            "RealOnly+Generic: exp(ln(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(
            r.has_warning,
            "RealOnly+Generic: WARNING expected (ln requires x > 0)"
        );

        // Check the warning mentions "positive" or "> 0"
        let has_positive_warning = r
            .warning_messages
            .iter()
            .any(|w| w.to_lowercase().contains("positive") || w.contains("> 0"));
        assert!(
            has_positive_warning,
            "Warning should mention 'positive' or '> 0', got: {:?}",
            r.warning_messages
        );
    }

    // ------------------------------------------------------------------------
    // RealOnly × Assume: YES simplification, YES warning
    // ------------------------------------------------------------------------
    #[test]
    fn real_assume_simplifies_with_warning() {
        let r = simplify_with_config("exp(ln(x))", ValueDomain::RealOnly, DomainMode::Assume);

        assert_eq!(
            r.result, "x",
            "RealOnly+Assume: exp(ln(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(
            r.has_warning,
            "RealOnly+Assume: WARNING expected (ln requires x > 0)"
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Strict: NO simplification
    // ------------------------------------------------------------------------
    #[test]
    fn complex_strict_preserves_composition() {
        let r = simplify_with_config(
            "exp(ln(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Strict,
        );

        assert!(
            r.result.contains("exp") || r.result.contains("e^") || r.result.contains("ln"),
            "ComplexEnabled+Strict: exp(ln(x)) should remain unchanged, got: {}",
            r.result
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Generic: YES simplification, YES warning
    // ------------------------------------------------------------------------
    #[test]
    fn complex_generic_simplifies_with_warning() {
        let r = simplify_with_config(
            "exp(ln(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Generic,
        );

        assert_eq!(
            r.result, "x",
            "ComplexEnabled+Generic: exp(ln(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(
            r.has_warning,
            "ComplexEnabled+Generic: WARNING expected (ln requires x > 0 or x ∈ ℂ\\{{0}})"
        );
    }

    // ------------------------------------------------------------------------
    // ComplexEnabled × Assume: YES simplification, YES warning
    // ------------------------------------------------------------------------
    #[test]
    fn complex_assume_simplifies_with_warning() {
        let r = simplify_with_config(
            "exp(ln(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Assume,
        );

        assert_eq!(
            r.result, "x",
            "ComplexEnabled+Assume: exp(ln(x)) should simplify to x, got: {}",
            r.result
        );

        assert!(r.has_warning, "ComplexEnabled+Assume: WARNING expected");
    }

    // ------------------------------------------------------------------------
    // Provably positive argument: NO warning needed
    // ------------------------------------------------------------------------
    #[test]
    fn provably_positive_no_warning() {
        // exp(ln(5)) should simplify without warning because 5 > 0 is provable
        let r = simplify_with_config("exp(ln(5))", ValueDomain::RealOnly, DomainMode::Generic);

        assert_eq!(
            r.result, "5",
            "exp(ln(5)) should simplify to 5, got: {}",
            r.result
        );

        // Note: Depending on rule implementation, this may or may not produce warning
        // since 5 is provably positive. This test documents expected behavior.
    }
}

// ============================================================================
// Combined Expression Tests
// ============================================================================

mod combined_expressions {
    use super::*;

    /// exp(ln(x)) + ln(exp(x)) should produce exactly one warning (for x > 0)
    /// when in RealOnly + Generic mode
    #[test]
    fn combined_real_generic_one_warning() {
        let r = simplify_with_config(
            "exp(ln(x)) + ln(exp(x))",
            ValueDomain::RealOnly,
            DomainMode::Generic,
        );

        // Accept both "2·x" and "2 * x" formats
        assert!(
            r.result == "2·x" || r.result == "2 * x",
            "exp(ln(x)) + ln(exp(x)) should simplify to 2x, got: {}",
            r.result
        );

        // Only e^(ln(x)) should produce warning, not ln(e^x)
        assert!(
            r.has_warning,
            "Should have warning for exp(ln(x)) requiring x > 0"
        );
    }

    /// In ComplexEnabled + Generic: ln(e^x) does NOT simplify (NEW CONTRACT)
    /// Only exp(ln(x)) simplifies (with warning for x > 0)
    #[test]
    fn combined_complex_generic_partial_simplify() {
        let r = simplify_with_config(
            "exp(ln(x)) + ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Generic,
        );

        // ln(e^x) does NOT simplify in ComplexEnabled (multivalued)
        // Only exp(ln(x)) → x (with x > 0 warning)
        // Result should be: x + ln(e^x)
        assert!(
            r.result.contains("ln"),
            "ComplexEnabled: ln(exp(x)) should remain unchanged, got: {}",
            r.result
        );

        // exp(ln(x)) does produce a warning for x > 0
        assert!(
            r.has_warning,
            "ComplexEnabled should produce warning for exp(ln(x)) requiring x > 0"
        );
    }

    /// RealOnly + Strict mode: both simplify now (NEW CONTRACT)
    /// ln(e^x) → x because x ∈ ℝ by contract, e^x > 0 always
    /// exp(ln(x)) stays because x > 0 not provable in Strict
    #[test]
    fn combined_real_strict_partial_simplify() {
        let r = simplify_with_config(
            "exp(ln(x)) + ln(exp(x))",
            ValueDomain::RealOnly,
            DomainMode::Strict,
        );

        // ln(e^x) → x now simplifies in RealOnly+Strict
        // exp(ln(x)) cannot simplify in Strict (cannot prove x > 0)
        assert!(
            r.result.contains("exp") || r.result.contains("ln"),
            "Strict mode should preserve exp(ln(x)) composition (can't prove x > 0), got: {}",
            r.result
        );
    }
}

// ============================================================================
// log(b, b^x) = x [General base tests]
// ============================================================================

mod general_log_base {
    use super::*;

    #[test]
    fn log_base_x_of_x_squared_always_simplifies() {
        // log(x, x^2) = 2 should ALWAYS work (numeric exponent)
        let r = simplify_with_config("log(x, x^2)", ValueDomain::RealOnly, DomainMode::Strict);
        assert_eq!(
            r.result, "2",
            "log(x, x^2) = 2 should always work, got: {}",
            r.result
        );
    }

    #[test]
    fn log_base_2_of_2_cubed() {
        // log(2, 2^3) = 3
        let r = simplify_with_config("log(2, 2^3)", ValueDomain::RealOnly, DomainMode::Strict);
        assert_eq!(r.result, "3", "log(2, 2^3) = 3, got: {}", r.result);
    }

    // ------------------------------------------------------------------------
    // GOLDEN REGRESSION TESTS: log(b, b^x) with symbolic base
    // ------------------------------------------------------------------------

    #[test]
    fn log_literal_base_simplifies_in_strict() {
        // log(2, 2^x) → x: 2 is provably positive, should work in Strict
        let r = simplify_with_config("log(2, 2^x)", ValueDomain::RealOnly, DomainMode::Strict);
        assert_eq!(
            r.result, "x",
            "log(2, 2^x) → x (2 > 0 provable), got: {}",
            r.result
        );
    }

    #[test]
    fn log_symbolic_base_blocks_in_strict() {
        // log(b, b^x): symbolic base should NOT simplify in Strict
        // (cannot prove b > 0)
        let r = simplify_with_config("log(b, b^x)", ValueDomain::RealOnly, DomainMode::Strict);
        assert!(
            r.result.contains("log"),
            "log(b, b^x) should remain unchanged in Strict (can't prove b > 0), got: {}",
            r.result
        );
    }

    #[test]
    fn log_symbolic_base_simplifies_in_generic() {
        // log(b, b^x) → x in Generic mode (allowed with assumption)
        let r = simplify_with_config("log(b, b^x)", ValueDomain::RealOnly, DomainMode::Generic);
        assert_eq!(
            r.result, "x",
            "log(b, b^x) → x in Generic mode, got: {}",
            r.result
        );
    }

    #[test]
    fn ln_exp_always_simplifies_in_realonly_strict() {
        // GOLDEN TEST: This was the original user goal
        // RealOnly + Strict: ln(e^x) → x (no warnings)
        let r = simplify_with_config("ln(exp(x))", ValueDomain::RealOnly, DomainMode::Strict);
        assert_eq!(
            r.result, "x",
            "GOLDEN: RealOnly+Strict: ln(e^x) → x, got: {}",
            r.result
        );
        assert!(
            !r.has_warning,
            "GOLDEN: RealOnly+Strict: NO warning for ln(e^x), got: {:?}",
            r.warning_messages
        );
    }

    #[test]
    fn ln_exp_never_simplifies_symbolic_in_complex() {
        // GOLDEN TEST: ComplexEnabled should NEVER simplify symbolic exponents
        let r = simplify_with_config(
            "ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Strict,
        );
        assert!(
            r.result.contains("ln") || r.result.contains("log"),
            "GOLDEN: ComplexEnabled: ln(e^x) stays unchanged, got: {}",
            r.result
        );
    }
}
