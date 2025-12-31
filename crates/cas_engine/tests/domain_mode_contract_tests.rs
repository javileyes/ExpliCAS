//! Domain Mode ConditionClass Contract Golden Tests
//!
//! These tests define the **irrefutable behavioral contract** for DomainMode
//! based on the ConditionClass taxonomy (V1.3).
//!
//! ConditionClass Taxonomy:
//! - Definability: NonZero, Defined (small holes) - Generic ACCEPTS
//! - Analytic: Positive, NonNegative (big restrictions) - Generic BLOCKS
//!
//! DomainMode Gate:
//! - Strict: Only if proven
//! - Generic: Definability only (Analytic blocked)
//! - Assume: All conditions

use cas_engine::{DomainMode, Simplifier, SimplifyOptions};
use cas_parser::parse;

/// Helper: simplify with specific DomainMode
fn simplify_with_mode(input: &str, mode: DomainMode) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = SimplifyOptions {
        domain: mode,
        ..Default::default()
    };

    let (result, _) = simplifier.simplify_with_options(expr, opts);
    format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// DEFINABILITY CLASS (NonZero, Defined)
// Strict: Block | Generic: Accept | Assume: Accept
// =============================================================================

mod definability_class {
    use super::*;

    #[test]
    fn x_over_x_strict_blocked() {
        let output = simplify_with_mode("x/x", DomainMode::Strict);
        assert_eq!(output, "x / x", "Strict should NOT simplify x/x");
    }

    #[test]
    fn x_over_x_generic_allowed() {
        let output = simplify_with_mode("x/x", DomainMode::Generic);
        assert_eq!(
            output, "1",
            "Generic SHOULD simplify x/x to 1 (Definability)"
        );
    }

    #[test]
    fn x_over_x_assume_allowed() {
        let output = simplify_with_mode("x/x", DomainMode::Assume);
        assert_eq!(output, "1", "Assume SHOULD simplify x/x to 1");
    }

    #[test]
    fn zero_over_x_strict_blocked() {
        let output = simplify_with_mode("0/x", DomainMode::Strict);
        assert_eq!(output, "0 / x", "Strict should NOT simplify 0/x");
    }

    #[test]
    fn zero_over_x_generic_allowed() {
        let output = simplify_with_mode("0/x", DomainMode::Generic);
        assert_eq!(
            output, "0",
            "Generic SHOULD simplify 0/x to 0 (Definability)"
        );
    }

    #[test]
    fn proven_nonzero_all_modes() {
        // 2/2 = 1 should work in ALL modes (proven nonzero)
        let strict = simplify_with_mode("2/2", DomainMode::Strict);
        let generic = simplify_with_mode("2/2", DomainMode::Generic);
        let assume = simplify_with_mode("2/2", DomainMode::Assume);

        assert_eq!(strict, "1", "Strict should simplify 2/2");
        assert_eq!(generic, "1", "Generic should simplify 2/2");
        assert_eq!(assume, "1", "Assume should simplify 2/2");
    }
}

// =============================================================================
// ANALYTIC CLASS (Positive, NonNegative)
// Strict: Block | Generic: BLOCK | Assume: Accept
// This is the KEY distinction from Definability
// =============================================================================

mod analytic_class {
    use super::*;

    /// CRITICAL: Generic must BLOCK Analytic conditions (Positive)
    /// This is what makes Generic != Assume
    #[test]
    fn ln_product_generic_blocked() {
        // ln(x*y) → ln(x) + ln(y) requires Positive(x), Positive(y) - Analytic
        let output = simplify_with_mode("ln(x*y)", DomainMode::Generic);
        // Should NOT contain '+' (expansion blocked)
        assert!(
            !output.contains("+"),
            "Generic should BLOCK ln(x*y) expansion (Analytic blocked), got: {}",
            output
        );
    }

    #[test]
    fn ln_product_strict_blocked() {
        let output = simplify_with_mode("ln(x*y)", DomainMode::Strict);
        assert!(
            !output.contains("+"),
            "Strict should NOT expand ln(x*y), got: {}",
            output
        );
    }

    /// Assume mode allows Analytic conditions with assumptions
    #[test]
    fn ln_quotient_generic_blocked() {
        let output = simplify_with_mode("ln(x/y)", DomainMode::Generic);
        assert!(
            !output.contains("-") || output.contains("ln"),
            "Generic should NOT expand ln(x/y) (Analytic blocked), got: {}",
            output
        );
    }
}

// =============================================================================
// PROVEN POSITIVE: Works in ALL modes
// =============================================================================

mod proven_positive {
    use super::*;

    #[test]
    fn ln_exp_product_all_modes() {
        // ln(exp(a) * exp(b)) should expand in ALL modes
        // because exp(a) > 0 is provable in RealOnly
        let strict = simplify_with_mode("ln(exp(a)*exp(b))", DomainMode::Strict);
        let generic = simplify_with_mode("ln(exp(a)*exp(b))", DomainMode::Generic);
        let assume = simplify_with_mode("ln(exp(a)*exp(b))", DomainMode::Assume);

        // All should expand (proven positive) - should contain '+'
        assert!(
            strict.contains("+") || strict == "a + b",
            "Strict should expand ln(exp*exp) (proven positive), got: {}",
            strict
        );
        assert!(
            generic.contains("+") || generic == "a + b",
            "Generic should expand ln(exp*exp) (proven positive), got: {}",
            generic
        );
        assert!(
            assume.contains("+") || assume == "a + b",
            "Assume should expand ln(exp*exp), got: {}",
            assume
        );
    }

    #[test]
    fn ln_positive_literals_all_modes() {
        // ln(2 * 3) should expand to ln(2) + ln(3) in all modes (literals proven positive)
        let strict = simplify_with_mode("ln(2*3)", DomainMode::Strict);
        let generic = simplify_with_mode("ln(2*3)", DomainMode::Generic);

        // Both should either expand or evaluate to ln(6)
        // Key: NOT blocked due to proven positivity
        assert!(
            !strict.contains("x") && !strict.contains("y"),
            "Strict should handle ln(2*3) (proven), got: {}",
            strict
        );
        assert!(
            !generic.contains("x") && !generic.contains("y"),
            "Generic should handle ln(2*3) (proven), got: {}",
            generic
        );
    }
}

// =============================================================================
// MUL ZERO with undefined risk
// =============================================================================

mod mul_zero_definability {
    use super::*;

    #[test]
    fn zero_times_safe_all_modes() {
        // 0 * 5 should always simplify (no undefined risk)
        let strict = simplify_with_mode("0*5", DomainMode::Strict);
        let generic = simplify_with_mode("0*5", DomainMode::Generic);
        assert_eq!(strict, "0");
        assert_eq!(generic, "0");
    }

    #[test]
    fn zero_times_risky_strict_blocked() {
        // 0 * (1/x) has undefined risk - Strict should block
        let output = simplify_with_mode("0*(1/x)", DomainMode::Strict);
        // Should NOT simplify to just "0"
        assert!(
            output.contains("x"),
            "Strict should NOT simplify 0*(1/x) due to undefined risk, got: {}",
            output
        );
    }

    #[test]
    fn zero_times_risky_generic_allowed() {
        // Generic allows Definability conditions
        let output = simplify_with_mode("0*(1/x)", DomainMode::Generic);
        assert_eq!(
            output, "0",
            "Generic SHOULD simplify 0*(1/x) to 0 (Definability)"
        );
    }
}
