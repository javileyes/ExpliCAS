use cas_ast::DisplayExpr;
use cas_engine::rules::algebra::FactorRule;
use cas_engine::rules::polynomial::CombineLikeTermsRule;
use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn test_factor_rule_integration() {
    let mut simplifier = Simplifier::new();
    // Only add FactorRule and CombineLikeTerms (to clean up if needed)
    // Do NOT add DistributeRule or BinomialExpansionRule as they undo factorization.
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));

    // Test factor(x^2 - 1)
    let expr = parse("factor(x^2 - 1)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let res = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    println!("Result: {}", res);
    // Should be (x-1)(x+1)
    assert!(res.contains("x - 1") || res.contains("-1 + x") || res.contains("x + -1"));
    assert!(res.contains("x + 1") || res.contains("1 + x"));
    assert!(!res.contains("factor"));
}

#[test]
fn test_factor_perfect_square_integration() {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));

    // Test factor(x^2 + 2x + 1)
    let expr = parse("factor(x^2 + 2*x + 1)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let res = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    println!("Result: {}", res);
    // Should be (x+1)^2
    assert!(res.contains("x + 1") || res.contains("1 + x"));
    assert!(res.contains("^ 2") || res.contains("^2"));
    assert!(!res.contains("factor"));
}
