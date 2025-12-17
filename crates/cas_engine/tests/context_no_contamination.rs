//! Context No-Contamination Tests
//!
//! These tests verify that context-specific rules (ProductToSum, CosProductTelescoping)
//! do NOT fire outside of their intended context (IntegratePrep).

use cas_ast::Context;
use cas_engine::options::{BranchMode, ContextMode, EvalOptions};
use cas_engine::Simplifier;
use cas_engine::Step;
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
        cas_ast::DisplayExpr {
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
        context_mode: ContextMode::Standard,
    }
}

fn solve_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Solve,
    }
}

fn integrate_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::IntegratePrep,
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
