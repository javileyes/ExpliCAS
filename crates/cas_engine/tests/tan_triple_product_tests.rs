//! Tests for TanTripleProductRule
//!
//! Verifies that tan(u)·tan(π/3+u)·tan(π/3-u) simplifies to tan(3u)

use cas_ast::Context;
use cas_engine::options::EvalOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

#[allow(dead_code)]
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
fn test_tan_triple_product_basic() {
    let (result, steps) = eval_with_steps("tan(x) * tan(pi/3 - x) * tan(pi/3 + x)");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Should use Triple Tangent Product rule
    assert!(
        rule_names.contains(&"Triple Tangent Product (π/3)"),
        "Should use Triple Tangent Product rule; rules: {:?}",
        rule_names
    );

    // Should NOT use Tan to Sin/Cos
    assert!(
        !rule_names.contains(&"Tan to Sin/Cos"),
        "Should NOT expand via Tan to Sin/Cos; rules: {:?}",
        rule_names
    );

    // Result should be tan(3*x)
    assert!(
        result.contains("tan") && result.contains("3"),
        "Result should be tan(3·x); got: {}",
        result
    );
}

#[test]
fn test_tan_triple_product_permutation() {
    // Different ordering of factors
    let (result, steps) = eval_with_steps("tan(pi/3 + x) * tan(x) * tan(pi/3 - x)");

    let rule_names: Vec<&str> = steps.iter().map(|s| s.rule_name.as_str()).collect();

    // Should use Triple Tangent Product rule
    assert!(
        rule_names.contains(&"Triple Tangent Product (π/3)"),
        "Permutation: should use Triple Tangent Product rule; rules: {:?}",
        rule_names
    );

    // Result should be tan(3*x)
    assert!(
        result.contains("tan") && result.contains("3"),
        "Permutation: result should be tan(3·x); got: {}",
        result
    );
}
