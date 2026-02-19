// Tests for inverse trig step didactics improvements
// Ensures atan(x) + atan(1/x) identity fires directly without Add Fractions steps

use cas_ast::Context;
use cas_engine::EvalOptions;
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

    let result_str = cas_formatter::DisplayExpr {
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

fn eval(input: &str) -> String {
    eval_with_steps(input).0
}

/// Test that atan(3) + (atan(1/3) - pi/2) simplifies to 0
#[test]
fn atan_relation_simplifies_to_zero() {
    let result = eval("atan(3) + (atan(1/3) - pi/2)");
    assert_eq!(result, "0", "Expected result 0, got: {}", result);
}

/// Test that the atan identity doesn't go through "Add Fractions" rule
/// This is the key regression test for the didactic improvement
#[test]
fn atan_relation_skips_add_fractions() {
    let (result, steps) = eval_with_steps("atan(3) + (atan(1/3) - pi/2)");

    // Verify result
    assert_eq!(result, "0");

    // Verify NO steps from AddFractionsRule (check rule_name directly)
    for step in &steps {
        assert!(
            step.rule_name != "Add Fractions",
            "AddFractionsRule should NOT fire on atan expression. \
             Found step with rule_name='Add Fractions': '{}'",
            step.description
        );
    }

    // Verify that "Inverse Tan Relations" step IS present
    let has_inverse_tan = steps.iter().any(|s| s.rule_name == "Inverse Tan Relations");
    assert!(
        has_inverse_tan,
        "Expected 'Inverse Tan Relations' step, but found: {:?}",
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );
}

/// Test that regular numeric fraction addition still works (regression guard)
#[test]
fn numeric_fraction_addition_still_works() {
    // 1 + 1/2 = 3/2
    let r1 = eval("1 + 1/2");
    assert_eq!(r1, "3/2", "1 + 1/2 should equal 3/2, got: {}", r1);

    // 1/2 + 1/3 = 5/6
    let r2 = eval("1/2 + 1/3");
    assert_eq!(r2, "5/6", "1/2 + 1/3 should equal 5/6, got: {}", r2);
}

/// Test that two fractions with symbolic content also combine
#[test]
fn symbolic_fractions_still_combine() {
    // x/2 + x/3 should combine to 5x/6
    let r = eval("x/2 + x/3");
    assert!(
        r.contains("x") && (r.contains("5") || r.contains("6")),
        "x/2 + x/3 should simplify to 5x/6 or equivalent, got: {}",
        r
    );
}

/// Test that numeric fractions combine even when functions are present in the expression
/// This protects against over-aggressive guards blocking legitimate fraction addition
#[test]
fn fractions_combine_despite_functions_in_expression() {
    // sin(x) + 1/2 + 1/3 should combine fractions to sin(x) + 5/6
    let (result, steps) = eval_with_steps("sin(x) + 1/2 + 1/3");

    // Result should contain sin(x) and 5/6
    assert!(
        result.contains("sin") && result.contains("5") && result.contains("6"),
        "sin(x) + 1/2 + 1/3 should simplify to sin(x) + 5/6 or equivalent, got: {}",
        result
    );

    // Verify that fraction combination DID happen (via CombineLikeTerms or AddFractions)
    let combined_fractions = steps.iter().any(|s| {
        s.description.contains("1/2") && s.description.contains("5/6")
            || s.rule_name.contains("Combine")
            || s.rule_name.contains("Fraction")
    });
    assert!(
        combined_fractions,
        "Fractions 1/2 + 1/3 should be combined despite sin(x) being present. Steps: {:?}",
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );
}
