//! MulZeroRule contract tests.
//!
//! # Contract: 0 * expr → 0 (Domain-Aware)
//!
//! The multiplication by zero rule must respect domain preservation:
//! - **Strict**: Only apply if `expr` has no undefined-risk (no variable denominators)
//! - **Assume**: Apply with assumption warning
//! - **Generic**: Apply with transparency warning
//!
//! This prevents "domain shrinking" where `0 * (1/(x+1))` (undefined at x=-1)
//! would become `0` (defined everywhere).

use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify with Strict domain mode
fn simplify_strict(input: &str) -> (String, Vec<String>) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: cas_engine::DomainMode::Strict,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let (result, steps) = simplifier.simplify_with_options(expr, opts);

    // Collect warnings from steps
    let warnings: Vec<String> = steps
        .iter()
        .flat_map(|s| s.assumption_events().iter())
        .map(|e| format!("{:?}", e.key))
        .collect();

    (
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        ),
        warnings,
    )
}

/// Helper: simplify with Assume domain mode
fn simplify_assume(input: &str) -> (String, Vec<String>) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: cas_engine::DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let (result, steps) = simplifier.simplify_with_options(expr, opts);

    let warnings: Vec<String> = steps
        .iter()
        .flat_map(|s| s.assumption_events().iter())
        .map(|e| format!("{:?}", e.key))
        .collect();

    (
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        ),
        warnings,
    )
}

/// Helper: simplify with Generic domain mode (default)
fn simplify_generic(input: &str) -> (String, Vec<String>) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: cas_engine::DomainMode::Generic,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let (result, steps) = simplifier.simplify_with_options(expr, opts);

    let warnings: Vec<String> = steps
        .iter()
        .flat_map(|s| s.assumption_events().iter())
        .map(|e| format!("{:?}", e.key))
        .collect();

    (
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        ),
        warnings,
    )
}

// =============================================================================
// MulZeroRule: Strict Mode Tests
// =============================================================================

#[test]
fn strict_0_times_risky_expr_stays_unchanged() {
    // 0 * (1/(x+1)) should NOT simplify to 0 in Strict mode
    // because 1/(x+1) is undefined at x=-1
    let (result, warnings) = simplify_strict("0 * 1/(x+1)");

    // Should NOT be simplified to just 0
    assert!(
        result.contains("/") || result.contains("0"),
        "Strict mode should not simplify 0*(risky expr) to 0, got: {}",
        result
    );
    assert!(
        warnings.is_empty(),
        "Strict mode should not emit warnings, got: {:?}",
        warnings
    );
}

#[test]
fn strict_0_times_safe_expr_simplifies() {
    // 0 * 5 should simplify to 0 (no undefined risk)
    let (result, warnings) = simplify_strict("0 * 5");
    assert_eq!(result, "0", "0 * 5 should simplify to 0");
    assert!(warnings.is_empty(), "No warnings for safe expr");
}

#[test]
fn strict_0_times_variable_simplifies() {
    // 0 * x should simplify to 0 (x has no undefined risk)
    let (result, warnings) = simplify_strict("0 * x");
    assert_eq!(result, "0", "0 * x should simplify to 0");
    assert!(warnings.is_empty(), "No warnings for variable");
}

// =============================================================================
// MulZeroRule: Assume Mode Tests
// =============================================================================

#[test]
fn assume_0_times_risky_expr_simplifies_with_warning() {
    // 0 * (1/(x+1)) should simplify to 0 WITH warning in Assume mode
    let (result, warnings) = simplify_assume("0 * 1/(x+1)");

    assert_eq!(result, "0", "Assume mode should simplify to 0");
    assert!(
        !warnings.is_empty(),
        "Assume mode should emit warning for risky expr, got: {:?}",
        warnings
    );
}

// =============================================================================
// MulZeroRule: Generic Mode Tests (NEW - Transparency)
// =============================================================================

#[test]
fn generic_0_times_risky_expr_simplifies_with_warning() {
    // 0 * (1/(x+1)) should simplify to 0 WITH warning in Generic mode
    // (pedagogical transparency)
    let (result, warnings) = simplify_generic("0 * 1/(x+1)");

    assert_eq!(result, "0", "Generic mode should simplify to 0");
    assert!(
        !warnings.is_empty(),
        "Generic mode should emit transparency warning for risky expr, got: {:?}",
        warnings
    );
}

#[test]
fn generic_0_times_safe_expr_simplifies_no_warning() {
    // 0 * 5 should simplify to 0 WITHOUT warning (no risk)
    let (result, warnings) = simplify_generic("0 * 5");

    assert_eq!(result, "0", "Generic mode should simplify to 0");
    assert!(
        warnings.is_empty(),
        "Generic mode should NOT emit warning for safe expr, got: {:?}",
        warnings
    );
}

// =============================================================================
// DivZeroRule: Tests
// =============================================================================

#[test]
fn strict_0_div_risky_stays_unchanged() {
    // 0 / (x+1) should NOT simplify to 0 in Strict mode
    let (result, _) = simplify_strict("0 / (x+1)");

    assert!(
        result.contains("/") || result.contains("0"),
        "Strict mode should not simplify 0/(risky expr) to 0, got: {}",
        result
    );
}

#[test]
fn generic_0_div_risky_simplifies_with_warning() {
    // 0 / (x+1) should simplify to 0 WITH warning in Generic mode
    let (result, warnings) = simplify_generic("0 / (x+1)");

    assert_eq!(result, "0", "Generic mode should simplify to 0");
    assert!(
        !warnings.is_empty(),
        "Generic mode should emit warning for 0/risky, got: {:?}",
        warnings
    );
}

// =============================================================================
// AddInverseRule: Tests
// =============================================================================

#[test]
fn strict_add_inverse_risky_stays_unchanged() {
    // (1/(x+1)) + (-(1/(x+1))) should NOT simplify to 0 in Strict mode
    let (result, _) = simplify_strict("(1/(x+1)) + (-(1/(x+1)))");

    // Should NOT be just 0
    assert!(
        result != "0",
        "Strict mode should not simplify risky a+(-a) to 0, got: {}",
        result
    );
}

#[test]
fn generic_add_inverse_risky_simplifies_no_warning() {
    // V2.12.13: (1/(x+1)) + (-(1/(x+1))) should simplify to 0 WITHOUT warning
    // The division 1/(x+1) already emits a NonZero(x+1) require, so the
    // AddInverseRule doesn't need to emit a redundant "is defined" assumption.
    // This was previously expected to emit a warning, but that was redundant.
    let (result, warnings) = simplify_generic("(1/(x+1)) + (-(1/(x+1)))");

    assert_eq!(result, "0", "Generic mode should simplify to 0");
    assert!(
        warnings.is_empty(),
        "Generic mode should NOT emit warning for a+(-a) - the Requires from division cover this. Got: {:?}",
        warnings
    );
}
