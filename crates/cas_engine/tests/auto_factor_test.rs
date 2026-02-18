use cas_engine::rules::algebra::AutomaticFactorRule;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_auto_factor_integration() {
    // 1. Create Simplifier with AutomaticFactorRule enabled
    // IMPORTANT: We must DISABLE the aggressive ExpandRule and ENABLE ConservativeExpandRule

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Expand Polynomial");
    simplifier.disable_rule("Binomial Expansion");
    simplifier.disable_rule("Distributive Property"); // Disable aggressive distribution
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));
    simplifier.add_rule(Box::new(AutomaticFactorRule));

    // x^2 + 2x + 1 should factor to (x+1)^2
    // This reduces complexity from 9 nodes to 5 nodes.
    let input = "x^2 + 2*x + 1";
    let expr = parse(input, &mut simplifier.context).unwrap();

    let (res, _) = simplifier.simplify(expr);

    let s = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("Result with auto_factor: {}", s);

    // Should be (x+1)^2
    assert!(s.contains("x + 1") || s.contains("1 + x"));
    assert!(s.contains("^ 2") || s.contains("^2"));
}

#[test]
fn test_auto_factor_vs_expand_loop() {
    // Test that we don't loop infinitely if both are enabled.
    // x(x+1) -> expand -> x^2+x -> factor -> x(x+1) ...
    // AutomaticFactorRule has a strict complexity check.
    // x^2+x (5 nodes) -> x(x+1) (5 nodes).
    // So AutomaticFactorRule should REJECT x(x+1) because it's not strictly smaller.
    // So loop should be broken.

    use cas_engine::rules::algebra::{AutomaticFactorRule, ExpandRule};
    use cas_engine::Simplifier;

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.add_rule(Box::new(AutomaticFactorRule));
    simplifier.add_rule(Box::new(ExpandRule));

    let input = "x^2 + x";
    let expr = parse(input, &mut simplifier.context).unwrap();

    // This should finish and not hang
    let (res, _) = simplifier.simplify(expr);

    let s = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("Result loop check: {}", s);

    // It should probably stay as x^2 + x because factor rejected it.
    // Or if expand ran first on x(x+1), it becomes x^2+x.
}
