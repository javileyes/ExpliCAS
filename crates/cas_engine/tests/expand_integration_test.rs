use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_expand_rule_integration() {
    let mut simplifier = Simplifier::new();
    cas_engine::rules::algebra::register(&mut simplifier);
    cas_engine::rules::polynomial::register(&mut simplifier); // For CombineLikeTerms
    cas_engine::rules::arithmetic::register(&mut simplifier);
    cas_engine::rules::exponents::register(&mut simplifier);

    // Test expand(a*(b+c))
    let expr = parse("expand(a * (b + c))", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let res = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    println!("Result: {}", res);
    // Should be a*b + a*c
    assert!(res.contains("a * b"));
    assert!(res.contains("a * c"));
    assert!(!res.contains("expand"));
}

#[test]
fn test_expand_binomial_integration() {
    let mut simplifier = Simplifier::new();
    cas_engine::rules::algebra::register(&mut simplifier);
    cas_engine::rules::polynomial::register(&mut simplifier);
    cas_engine::rules::arithmetic::register(&mut simplifier);
    cas_engine::rules::exponents::register(&mut simplifier);

    // Disable polynomial strategy to prevent re-factoring
    simplifier.enable_polynomial_strategy = false;

    // Test expand((x+1)^2)
    let expr = parse("expand((x + 1)^2)", &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let res = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    println!("Result: {}", res);
    // Should be x^2 + 2*x + 1
    assert!(res.contains("x^2"));
    assert!(res.contains("2 * x"));
    assert!(res.contains("1"));
    assert!(!res.contains("expand"));
}
