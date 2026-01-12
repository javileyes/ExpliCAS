// Tests for inverse trig step didactics improvements
// Ensures atan(x) + atan(1/x) identity fires directly without Add Fractions steps

use cas_ast::Context;
use cas_engine::options::EvalOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

fn eval_with_steps(input: &str) -> (String, Vec<String>) {
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

    let step_descs: Vec<String> = steps.iter().map(|s| s.description.clone()).collect();

    (result_str, step_descs)
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

/// Test that the atan identity doesn't go through "Add numeric fractions" steps
/// This is the key regression test for the didactic improvement
#[test]
fn atan_relation_skips_add_fractions() {
    let (result, steps) = eval_with_steps("atan(3) + (atan(1/3) - pi/2)");

    // Verify result
    assert_eq!(result, "0");

    // Verify step descriptions don't include "Add numeric fractions"
    for step in &steps {
        assert!(
            !step.contains("Add numeric fractions"),
            "AddFractionsRule should not trigger on atan expression. \
             Found 'Add numeric fractions' in step: '{}'",
            step
        );
    }

    // Verify that inverse tan relation step IS present
    let has_inverse_tan = steps
        .iter()
        .any(|s| s.contains("arctan") || s.contains("π/2"));
    assert!(
        has_inverse_tan,
        "Expected arctan step, steps were: {:?}",
        steps
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
