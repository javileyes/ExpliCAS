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

    // Result should contain sin and cos with matching args
    assert!(
        result.contains("sin") && result.contains("cos"),
        "Result should be sin/cos form; got: {}",
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

    // Result should be sin/cos form
    assert!(
        result.contains("sin") && result.contains("cos"),
        "Reversed order: result should be sin/cos form; got: {}",
        result
    );
}

#[test]
fn test_didactic_steps_exist() {
    let (_, steps) = eval_with_steps("(sin(x) + sin(3*x)) / (cos(x) + cos(3*x))");

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
}
