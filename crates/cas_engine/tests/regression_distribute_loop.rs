use cas_ast::DisplayExpr;
use cas_engine::rules::algebra::AutomaticFactorRule;
use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_engine::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeNegationRule,
};
use cas_engine::rules::polynomial::CombineLikeTermsRule;
use cas_engine::rules::polynomial::DistributeRule;
use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn test_distribute_factor_loop() {
    let mut simplifier = Simplifier::new();
    // Register rules that might conflict
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AutomaticFactorRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));

    // Case 1: Univariate x*(x - 1) -> Should NOT distribute (avoid loop)
    let input1 = "x*(x - 1)";
    let expr1 = parse(input1, &mut simplifier.context).expect("Failed to parse");
    let (result1, steps1) = simplifier.simplify(expr1);
    let res_str1 = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result1
        }
    );
    println!("Result1: {}", res_str1);

    // Check that we did NOT distribute
    let distributed1 = steps1.iter().any(|s| s.description == "Distribute");
    assert!(!distributed1, "Should not distribute univariate x*(x-1)");

    // Case 2: Multivariate x*(y - z) -> Should distribute (enable cancellation)
    let input2 = "x*(y - z)";
    let expr2 = parse(input2, &mut simplifier.context).expect("Failed to parse");
    let (result2, steps2) = simplifier.simplify(expr2);
    let res_str2 = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result2
        }
    );
    println!("Result2: {}", res_str2);

    // Check that we DID distribute
    let distributed2 = steps2.iter().any(|s| s.description == "Distribute");
    assert!(distributed2, "Should distribute multivariate x*(y-z)");

    // If it loops, steps will be large (hit max iterations)
    // If it stabilizes, it will likely be xy - xz (expanded) or x(y-z) (factored)
    // Given DistributeRule is usually aggressive, I expect xy - xz.
}
