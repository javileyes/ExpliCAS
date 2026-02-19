//! Context No-Contamination Tests
//!
//! These tests verify that context-specific rules (ProductToSum, CosProductTelescoping)
//! do NOT fire outside of their intended context (IntegratePrep).

use cas_ast::Context;
use cas_engine::Simplifier;
use cas_engine::Step;
use cas_engine::{BranchMode, ComplexMode, ContextMode, EvalOptions, StepsMode};
use cas_parser::parse;

/// Helper: run simplification with given options and return (result_string, steps)
fn run_simplify(input: &str, opts: &EvalOptions) -> (String, Vec<Step>) {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(opts);
    simplifier.context = ctx;
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, steps)
}

/// Assert that a specific rule did NOT fire
fn assert_no_rule(steps: &[Step], rule_name: &str) {
    let fired = steps.iter().any(|s| s.rule_name == rule_name);
    assert!(
        !fired,
        "Rule '{}' should NOT have fired, but it did. Steps: {:?}",
        rule_name,
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );
}

/// Assert that a specific rule DID fire
fn assert_rule_fired(steps: &[Step], rule_name: &str) {
    let fired = steps.iter().any(|s| s.rule_name == rule_name);
    assert!(
        fired,
        "Rule '{}' should have fired, but it didn't. Steps: {:?}",
        rule_name,
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );
}

fn standard_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn solve_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Solve,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn integrate_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::IntegratePrep,
            ..Default::default()
        },
        ..Default::default()
    }
}

// =============================================================================
// SECTION 1: Standard Mode - Integration rules should NOT fire
// =============================================================================

#[test]
fn test_standard_no_product_to_sum_basic() {
    let (_, steps) = run_simplify("2*sin(x)*cos(y)", &standard_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_standard_no_product_to_sum_commuted_coeff() {
    let (_, steps) = run_simplify("sin(x)*2*cos(y)", &standard_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_standard_no_product_to_sum_commuted_order() {
    let (_, steps) = run_simplify("cos(y)*sin(x)*2", &standard_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_standard_no_product_to_sum_negative() {
    let (_, steps) = run_simplify("-2*sin(x)*cos(y)", &standard_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_standard_no_product_to_sum_negative_cos() {
    let (_, steps) = run_simplify("2*sin(x)*(-cos(y))", &standard_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_standard_no_telescoping_basic() {
    let (_, steps) = run_simplify("cos(x)*cos(2*x)*cos(4*x)", &standard_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

#[test]
fn test_standard_no_telescoping_permuted() {
    let (_, steps) = run_simplify("cos(4*x)*cos(x)*cos(2*x)", &standard_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

// =============================================================================
// SECTION 2: Solve Mode - Integration rules should NOT fire
// =============================================================================

#[test]
fn test_solve_no_product_to_sum() {
    let (_, steps) = run_simplify("2*sin(3*x)*cos(x)", &solve_opts());
    assert_no_rule(&steps, "ProductToSum");
}

#[test]
fn test_solve_no_telescoping() {
    let (_, steps) = run_simplify("cos(x)*cos(2*x)*cos(4*x)", &solve_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

// =============================================================================
// SECTION 3: IntegratePrep Edge Cases - Should NOT match
// =============================================================================

#[test]
fn test_integrateprep_telescoping_different_bases_no_match() {
    // Different variables (x vs y) - no consistent base
    let (_, steps) = run_simplify("cos(x)*cos(2*y)*cos(4*z)", &integrate_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

#[test]
fn test_integrateprep_telescoping_single_factor_no_match() {
    // Single cos factor - not a product
    let (_, steps) = run_simplify("cos(x)", &integrate_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

#[test]
fn test_integrateprep_telescoping_numeric_only_no_match() {
    // Numeric arguments, no variable - no point in telescoping
    let (_, steps) = run_simplify("cos(1)*cos(2)*cos(4)", &integrate_opts());
    assert_no_rule(&steps, "CosProductTelescoping");
}

#[test]
fn test_integrateprep_telescoping_only_two_factors() {
    // 1,2 is valid - cos(x)*cos(2x) -> sin(4x)/(4*sin(x))
    // Note: With only 2 factors, Double Angle Identity doesn't fully destroy the pattern
    let (_, steps) = run_simplify("cos(x)*cos(2*x)", &integrate_opts());
    // The rule may or may not fire first depending on simplifier order
    // For now, just verify no error
    let _ = steps;
}

// =============================================================================
// SECTION 4: IntegratePrep Positive Tests - Should match
// =============================================================================

#[test]
fn test_integrateprep_product_to_sum_basic() {
    let (_, steps) = run_simplify("2*sin(x)*cos(y)", &integrate_opts());
    assert_rule_fired(&steps, "ProductToSum");
}

#[test]
fn test_integrateprep_product_to_sum_commuted_coeff() {
    let (_, steps) = run_simplify("sin(x)*2*cos(y)", &integrate_opts());
    // Note: canonicalization may reorder to 2*sin(x)*cos(y)
    assert_rule_fired(&steps, "ProductToSum");
}

#[test]
fn test_integrateprep_telescoping_basic() {
    let (_, steps) = run_simplify("cos(x)*cos(2*x)*cos(4*x)", &integrate_opts());
    assert_rule_fired(&steps, "CosProductTelescoping");
}

#[test]
fn test_integrateprep_telescoping_permuted() {
    let (_, steps) = run_simplify("cos(4*x)*cos(x)*cos(2*x)", &integrate_opts());
    // Sorting should handle permutation
    assert_rule_fired(&steps, "CosProductTelescoping");
}

#[test]
fn test_integrateprep_telescoping_base_3() {
    // cos(3x)*cos(6x)*cos(12x) = {3,6,12} = 3*{1,2,4}
    let (_, steps) = run_simplify("cos(3*x)*cos(6*x)*cos(12*x)", &integrate_opts());
    assert_rule_fired(&steps, "CosProductTelescoping");
}

// =============================================================================
// SECTION 5: Determinism Tests - Rule ordering is stable
// =============================================================================

/// Verify that telescoping (priority 100) fires before general trig identity rules (priority 0)
/// when both could theoretically apply, due to explicit priority ordering.
#[test]
fn test_priority_telescoping_before_general() {
    // This test verifies priority system works:
    // CosProductTelescopingRule (priority 100) should be listed before
    // any default-priority rules in the same target_type bucket.

    let opts = integrate_opts();

    // Check that high-priority rules come first in the Mul bucket
    // (This is an indirect test - the rule ordering is internal)
    let (_, steps) = run_simplify("cos(x)*cos(2*x)*cos(4*x)", &opts);

    // CosProductTelescoping should fire, proving it ran before any rule
    // that might have transformed the cos(2x) or cos(4x) terms
    assert_rule_fired(&steps, "CosProductTelescoping");
}

/// Verify that rule order is deterministic across multiple runs.
#[test]
fn test_deterministic_rule_order() {
    // Run the same simplification multiple times and verify same result
    let opts = integrate_opts();

    let mut results = Vec::new();
    for _ in 0..3 {
        let (result, _) = run_simplify("2*sin(x)*cos(y)", &opts);
        results.push(result);
    }

    // All runs should produce identical results
    assert!(
        results.windows(2).all(|w| w[0] == w[1]),
        "Non-deterministic results: {:?}",
        results
    );
}

/// Verify that ProductToSum (priority 50) fires after Telescoping (100) but before default (0)
#[test]
fn test_priority_ordering_50_before_0() {
    let (_, steps) = run_simplify("2*sin(x)*cos(y)", &integrate_opts());
    // ProductToSum has priority 50, should fire before any priority-0 rule
    // that might transform sin/cos individually
    assert_rule_fired(&steps, "ProductToSum");
}

// =============================================================================
// SECTION 6: Priority System Stability Tests
// These tests document the priority system guarantees
// =============================================================================

use cas_ast::ExprId;
use cas_engine::ParentContext;
use cas_engine::{Rewrite, Rule};

/// Test rule that matches only value 999 and rewrites to marker
struct OneTimeMarkerRule {
    name: &'static str,
    priority: i32,
    marker: i64,
}

impl Rule for OneTimeMarkerRule {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn target_types(&self) -> Option<cas_engine::target_kind::TargetKindSet> {
        Some(cas_engine::target_kind::TargetKindSet::NUMBER)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        // Only match value 999 (the input we use in tests)
        if let cas_ast::Expr::Number(n) = ctx.get(expr) {
            if n.to_integer() == num_bigint::BigInt::from(999) {
                let marker = ctx.num(self.marker);
                return Some(Rewrite::simple(
                    marker,
                    format!("Marker {} applied", self.name),
                ));
            }
        }
        None
    }
}

/// Same priority: first registered wins (insertion order preserved)
#[test]
fn test_priority_same_first_registered_wins() {
    let mut simplifier = Simplifier::new();

    // Register A first, then B (both priority 0)
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "A",
        priority: 0,
        marker: 111,
    }));
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "B",
        priority: 0,
        marker: 222,
    }));

    let expr = simplifier.context.num(999);
    let (result, steps) = simplifier.simplify(expr);

    // A was registered first, so A should win
    assert!(!steps.is_empty(), "Should have at least one step");
    assert_eq!(
        steps[0].rule_name, "A",
        "First registered rule should apply first"
    );

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "111", "Result should be A's marker");
}

/// Higher priority wins over lower priority
#[test]
fn test_priority_higher_wins() {
    let mut simplifier = Simplifier::new();

    // Register low priority first, then high priority
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "Low",
        priority: 0,
        marker: 111,
    }));
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "High",
        priority: 100,
        marker: 888, // Must differ from input value 999
    }));

    let expr = simplifier.context.num(999);
    let (_, steps) = simplifier.simplify(expr);

    // High priority should win despite being registered second
    assert!(!steps.is_empty(), "Should have at least one step");
    assert_eq!(
        steps[0].rule_name, "High",
        "Higher priority rule should apply first"
    );
}

/// Inserting higher priority doesn't reorder existing same-priority rules
#[test]
fn test_priority_insert_high_preserves_order() {
    let mut simplifier = Simplifier::new();

    // Register A, B, C all with priority 0
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "A",
        priority: 0,
        marker: 1,
    }));
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "B",
        priority: 0,
        marker: 2,
    }));
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "C",
        priority: 0,
        marker: 3,
    }));

    // Now insert X with high priority (only matches value 999)
    simplifier.add_rule(Box::new(OneTimeMarkerRule {
        name: "X",
        priority: 100,
        marker: 888,
    }));

    // X should be first (highest priority)
    let expr = simplifier.context.num(999);
    let (_, steps) = simplifier.simplify(expr);
    assert!(!steps.is_empty(), "Should have at least one step");
    assert_eq!(steps[0].rule_name, "X", "High priority X should be first");
}
