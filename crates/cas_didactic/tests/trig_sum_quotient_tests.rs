//! Regression tests for SinCosSumQuotientRule
//!
//! Verifies that (sin(A)+sin(B))/(cos(A)+cos(B)) simplifies using sum-to-product
//! identities instead of triple angle expansion.

use cas_ast::Context;
use cas_parser::parse;
use cas_solver::runtime::EvalOptions;
use cas_solver::runtime::Simplifier;
use std::sync::LazyLock;

const SUM_QUOTIENT_EXPR: &str = "(sin(x) + sin(3*x)) / (cos(x) + cos(3*x))";
const REVERSED_SUM_QUOTIENT_EXPR: &str = "(sin(3*x) + sin(x)) / (cos(3*x) + cos(x))";
const DIFF_QUOTIENT_EXPR: &str = "(sin(3*x) - sin(x)) / (cos(3*x) + cos(x))";
const REVERSED_DIFF_QUOTIENT_EXPR: &str = "(sin(x) - sin(3*x)) / (cos(x) + cos(3*x))";

#[derive(Clone)]
struct StepInfo {
    description: String,
    rule_name: String,
}

fn eval_with_steps_uncached(input: &str) -> (String, Vec<StepInfo>) {
    let opts = EvalOptions::default();
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;

    let (result, steps) = simplifier.simplify(expr);

    let result_str = cas_formatter::DisplayExpr {
        context: &simplifier.context,
        id: result,
    }
    .to_string();

    let step_infos: Vec<StepInfo> = steps
        .iter()
        .map(|s| StepInfo {
            description: s.description.to_string(),
            rule_name: s.rule_name.to_string(),
        })
        .collect();

    (result_str, step_infos)
}

static SUM_QUOTIENT_EVAL: LazyLock<(String, Vec<StepInfo>)> =
    LazyLock::new(|| eval_with_steps_uncached(SUM_QUOTIENT_EXPR));
static REVERSED_SUM_QUOTIENT_EVAL: LazyLock<(String, Vec<StepInfo>)> =
    LazyLock::new(|| eval_with_steps_uncached(REVERSED_SUM_QUOTIENT_EXPR));
static DIFF_QUOTIENT_EVAL: LazyLock<(String, Vec<StepInfo>)> =
    LazyLock::new(|| eval_with_steps_uncached(DIFF_QUOTIENT_EXPR));
static REVERSED_DIFF_QUOTIENT_EVAL: LazyLock<(String, Vec<StepInfo>)> =
    LazyLock::new(|| eval_with_steps_uncached(REVERSED_DIFF_QUOTIENT_EXPR));

fn eval_with_steps(input: &str) -> (String, Vec<StepInfo>) {
    match input {
        SUM_QUOTIENT_EXPR => SUM_QUOTIENT_EVAL.clone(),
        REVERSED_SUM_QUOTIENT_EXPR => REVERSED_SUM_QUOTIENT_EVAL.clone(),
        DIFF_QUOTIENT_EXPR => DIFF_QUOTIENT_EVAL.clone(),
        REVERSED_DIFF_QUOTIENT_EXPR => REVERSED_DIFF_QUOTIENT_EVAL.clone(),
        _ => eval_with_steps_uncached(input),
    }
}

#[test]
fn test_sin_cos_sum_quotient_uses_sum_to_product() {
    let (result, steps) = eval_with_steps("(sin(x) + sin(3*x)) / (cos(x) + cos(3*x))");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Should NOT use Triple Angle Identity
    assert!(
        !rule_names.contains(&"Triple Angle Identity"),
        "Should not expand via triple angle; rules: {:?}",
        rule_names
    );

    // Should use Sum-to-Product Quotient
    assert!(
        rule_names.contains(&"Sum-to-Product Quotient"),
        "Should use Sum-to-Product Quotient rule; rules: {:?}",
        rule_names
    );

    // Result should be tan form (sin/cos converted to tan by TrigQuotientRule)
    assert!(
        result.contains("tan"),
        "Result should be tan form; got: {}",
        result
    );
}

#[test]
fn test_sin_cos_sum_quotient_reversed_order() {
    // Same pattern but with reversed order in sums
    let (result, steps) = eval_with_steps("(sin(3*x) + sin(x)) / (cos(3*x) + cos(x))");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Should NOT use Triple Angle Identity
    assert!(
        !rule_names.contains(&"Triple Angle Identity"),
        "Reversed order: should not expand via triple angle; rules: {:?}",
        rule_names
    );

    // Result should be tan form
    assert!(
        result.contains("tan"),
        "Reversed order: result should be tan form; got: {}",
        result
    );
}

#[test]
fn test_didactic_steps_exist() {
    let (result, steps) = eval_with_steps("(sin(x) + sin(3*x)) / (cos(x) + cos(3*x))");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();
    let descriptions: Vec<&str> = steps.iter().map(|s| s.description.as_str()).collect();
    let all_descs = descriptions.join(" | ");

    // Verify didactic steps exist (3 steps from ChainedRewrite)
    assert!(
        all_descs.contains("sin(A)+sin(B)") || all_descs.contains("sin(x)"),
        "Should show sin sum-to-product step; descs: {}",
        all_descs
    );
    assert!(
        all_descs.contains("cos(A)+cos(B)") || all_descs.contains("cos(x)"),
        "Should show cos sum-to-product step; descs: {}",
        all_descs
    );
    assert!(
        all_descs.contains("Cancel"),
        "Should show cancellation step; descs: {}",
        all_descs
    );

    // V2.14.27: Verify NO "Combine Like Terms" step
    // Pre-simplification in build_avg should eliminate this step
    assert!(
        !rule_names.contains(&"Combine Like Terms"),
        "Should NOT have Combine Like Terms step (pre-simplified); rules: {:?}",
        rule_names
    );

    // Verify final result is clean tan(2x)
    assert!(
        result.contains("tan") && result.contains("2"),
        "Final result should be tan(2·x); got: {}",
        result
    );
}

#[test]
fn test_sin_cos_diff_quotient() {
    // This tests the difference pattern: (sin(A)-sin(B))/(cos(A)+cos(B)) → tan((A-B)/2)
    // For A=3x, B=x: half_diff = (3x-x)/2 = x, so result should be tan(x)
    let (result, steps) = eval_with_steps(DIFF_QUOTIENT_EXPR);

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Should use Sum-to-Product Quotient
    assert!(
        rule_names.contains(&"Sum-to-Product Quotient"),
        "Should use Sum-to-Product Quotient rule; rules: {:?}",
        rule_names
    );

    // Should NOT use Recursive Trig Expansion (didactic regression)
    assert!(
        !rule_names.contains(&"Recursive Trig Expansion"),
        "Should NOT use Recursive Trig Expansion; rules: {:?}",
        rule_names
    );

    // CRITICAL: Result must be tan(x), NOT -tan(x) or tan(-x)
    // This is the exact bug that was fixed - binding A,B from numerator order
    assert!(
        result == "tan(x)",
        "Result should be exactly 'tan(x)'; got: '{}'. Sign bug detected!",
        result
    );
}

/// Regression test for sign correctness in sum-to-product difference case
/// This guards against the bug where canonical ordering of half_diff
/// inverted the sign when numerator was sin(A)-sin(B)
#[test]
fn test_sin_diff_sign_correctness() {
    // The positive orientation is already covered by test_sin_cos_diff_quotient.
    // Here we only keep the reversed case, which is the actual sign-regression guard.
    let (result, _) = eval_with_steps(REVERSED_DIFF_QUOTIENT_EXPR);
    // Since (x-3x)/2 = -x, result should be tan(-x) or equivalently -tan(x)
    assert!(
        result == "tan(-x)"
            || result == "-tan(x)"
            || result.contains("-1") && result.contains("tan"),
        "Bug: wrong sign. Expected tan(-x) or -tan(x), got {}",
        result
    );
}
