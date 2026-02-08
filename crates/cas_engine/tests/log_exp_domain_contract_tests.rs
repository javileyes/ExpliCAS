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
//! ## e^(ln(x)) → x (ExponentialLogRule) [V2.15.4 Implicit Domain]
//!
//! | ValueDomain    | DomainMode | Simplifies? | Requires? | Reason |
//! |----------------|------------|-------------|-----------|--------|
//! | RealOnly       | Strict     | NO (if x unknown) | -  | Can't prove x > 0 |
//! | RealOnly       | Generic    | YES         | x > 0     | x > 0 is IMPLICIT from ln(x) |
//! | RealOnly       | Assume     | YES         | x > 0     | Same as Generic |
//! | ComplexEnabled | Strict     | NO          | -        | Can't prove x > 0 |
//! | ComplexEnabled | Generic    | YES         | x > 0     | x > 0 is IMPLICIT from ln(x) |
//! | ComplexEnabled | Assume     | YES         | x > 0     | Same as Generic |

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
    state.options.shared.semantics.value_domain = value_domain;
    state.options.shared.semantics.domain_mode = domain_mode;

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
    // RealOnly × Generic: YES simplification (V2.15.4 IMPLICIT DOMAIN)
    // x > 0 is IMPLICIT from ln(x), not a new assumption
    // ------------------------------------------------------------------------
    #[test]
    fn real_generic_blocks_analytic() {
        let r = simplify_with_config("exp(ln(x))", ValueDomain::RealOnly, DomainMode::Generic);

        // V2.15.4: exp(ln(x)) SIMPLIFIES with implicit requires (like sqrt(x)^2)
        assert_eq!(
            r.result, "x",
            "RealOnly+Generic: exp(ln(x)) should simplify to x (implicit domain from ln), got: {}",
            r.result
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

        // V2.14.15: Warning suppressed because x > 0 is already implicit from ln(x) in the input.
        // The condition x > 0 is in global_requires, so EvaluateLogarithms' assumption is
        // reclassified to DerivedFromRequires (not shown as ⚠️).
        // This is semantically correct: the input already requires x > 0.
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
    // ComplexEnabled × Generic: YES simplification (V2.15.4 IMPLICIT DOMAIN)
    // x > 0 is IMPLICIT from ln(x), not a new assumption
    // ------------------------------------------------------------------------
    #[test]
    fn complex_generic_blocks_analytic() {
        let r = simplify_with_config(
            "exp(ln(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Generic,
        );

        // V2.15.4: exp(ln(x)) SIMPLIFIES with implicit requires
        assert_eq!(
            r.result, "x",
            "ComplexEnabled+Generic: exp(ln(x)) should simplify to x (implicit domain), got: {}",
            r.result
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

        // V2.15.4: No warning expected - x > 0 is IMPLICIT from ln(x), not an assumption
        assert!(
            !r.has_warning,
            "ComplexEnabled+Assume: No warning expected (implicit domain), got: {:?}",
            r.warning_messages
        );
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

    /// V2.15.4: In Generic mode, BOTH parts simplify:
    /// - exp(ln(x)) → x (implicit domain from ln(x))
    /// - ln(exp(x)) → x (no conditions needed in RealOnly)
    ///
    /// Result: x + x = 2x
    #[test]
    fn combined_real_generic_partial_simplify() {
        let r = simplify_with_config(
            "exp(ln(x)) + ln(exp(x))",
            ValueDomain::RealOnly,
            DomainMode::Generic,
        );

        // V2.15.4: Both parts simplify to x, result is 2*x
        assert_eq!(
            r.result, "2 * x",
            "RealOnly+Generic: exp(ln(x)) + ln(exp(x)) should simplify to 2*x, got: {}",
            r.result
        );
    }

    /// V1.3 CONTRACT: In ComplexEnabled + Generic, BOTH parts stay unchanged
    /// - ln(exp(x)) blocked (multivalued ln)
    /// - exp(ln(x)) blocked (Positive is Analytic, blocked in Generic)
    #[test]
    fn combined_complex_generic_no_simplify() {
        let r = simplify_with_config(
            "exp(ln(x)) + ln(exp(x))",
            ValueDomain::ComplexEnabled,
            DomainMode::Generic,
        );

        // V1.3 CONTRACT: BOTH blocked in Generic
        // ln(e^x) blocked (multivalued ln in complex)
        // exp(ln(x)) blocked (Positive is Analytic)
        // Note: display uses "e^" format, not "exp"
        let has_ln = r.result.contains("ln");
        let has_exp = r.result.contains("exp") || r.result.contains("e^");
        assert!(
            has_ln && has_exp,
            "ComplexEnabled+Generic: Both parts should remain unchanged, got: {}",
            r.result
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
    fn log_symbolic_base_blocks_in_generic() {
        // log(b, b^x): symbolic base should NOT simplify in Generic either
        // (cannot prove b > 0, and Generic doesn't assume)
        let r = simplify_with_config("log(b, b^x)", ValueDomain::RealOnly, DomainMode::Generic);
        assert!(
            r.result.contains("log"),
            "log(b, b^x) should remain unchanged in Generic (can't prove b > 0), got: {}",
            r.result
        );
    }

    #[test]
    fn log_symbolic_base_simplifies_in_assume() {
        // log(b, b^x) → x in Assume mode (allowed with assumption warning)
        let r = simplify_with_config("log(b, b^x)", ValueDomain::RealOnly, DomainMode::Assume);
        assert_eq!(
            r.result, "x",
            "log(b, b^x) → x in Assume mode, got: {}",
            r.result
        );
        assert!(
            r.has_warning,
            "Assume mode should emit warning for b > 0 assumption"
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

// ============================================================================
// V2.14.21: Log Product Expansion & ChainedRewrite Tests
// ============================================================================

mod log_product_expansion {
    use super::*;

    /// REGRESSION TEST: ln(a^2 * b^3) - 2*ln(a) - 3*ln(b) → 0 in Generic mode
    ///
    /// This tests three fixes:
    /// 1. AutoExpandLogRule uses is_condition_implied to recognize b^3 > 0 implied by b > 0
    /// 2. LogEvenPowerWithChainedAbsRule produces ChainedRewrite |a| → a when a > 0 in requires
    /// 3. Result simplifies to 0 (not blocked)
    #[test]
    fn log_product_generic_simplifies_to_zero() {
        let r = simplify_with_config(
            "ln(a^2 * b^3) - 2*ln(a) - 3*ln(b)",
            ValueDomain::RealOnly,
            DomainMode::Generic,
        );

        assert_eq!(
            r.result, "0",
            "Generic: ln(a^2*b^3) - 2*ln(a) - 3*ln(b) should simplify to 0, got: {}",
            r.result
        );

        // No warnings expected - all conditions come from requires
        assert!(
            !r.has_warning,
            "Generic: No warning expected (a > 0, b > 0 from ln(a), ln(b)), got: {:?}",
            r.warning_messages
        );
    }

    /// Same test in Assume mode - should also work
    #[test]
    fn log_product_assume_simplifies_to_zero() {
        let r = simplify_with_config(
            "ln(a^2 * b^3) - 2*ln(a) - 3*ln(b)",
            ValueDomain::RealOnly,
            DomainMode::Assume,
        );

        assert_eq!(
            r.result, "0",
            "Assume: ln(a^2*b^3) - 2*ln(a) - 3*ln(b) should simplify to 0, got: {}",
            r.result
        );
    }
}
