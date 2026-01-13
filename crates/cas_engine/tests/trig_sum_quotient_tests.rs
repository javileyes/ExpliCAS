//! Regression tests for SinCosSumQuotientRule
//!
//! Verifies that (sin(A)+sin(B))/(cos(A)+cos(B)) simplifies using sum-to-product
//! identities instead of triple angle expansion.

use cas_ast::Context;
use cas_engine::options::EvalOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

struct StepInfo {
    description: String,
    rule_name: String,
}

fn eval_with_steps(input: &str) -> (String, Vec<StepInfo>) {
    let opts = EvalOptions::default();
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;

    let (result, steps) = simplifier.simplify(expr);

    let result_str = cas_ast::DisplayExpr {
        context: &simplifier.context,
        id: result,
    }
    .to_string();

    let step_infos: Vec<StepInfo> = steps
        .iter()
        .map(|s| StepInfo {
            description: s.description.clone(),
            rule_name: s.rule_name.clone(),
        })
        .collect();

    (result_str, step_infos)
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
    let (result, steps) = eval_with_steps("(sin(5*x) - sin(3*x)) / (cos(5*x) + cos(3*x))");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Debug printing
    for step in &steps {
        eprintln!("Step: {} [{}]", step.description, step.rule_name);
    }

    // Should use Sum-to-Product Quotient
    assert!(
        rule_names.contains(&"Sum-to-Product Quotient"),
        "Should use Sum-to-Product Quotient rule; rules: {:?}",
        rule_names
    );

    // Should NOT use Recursive Trig Expansion
    assert!(
        !rule_names.contains(&"Recursive Trig Expansion"),
        "Should NOT use Recursive Trig Expansion; rules: {:?}",
        rule_names
    );

    // Result should be tan(x) - since (5x-3x)/2 = x
    assert!(
        result.contains("tan") && result.contains("x"),
        "Final result should be tan(x); got: {}",
        result
    );
}
