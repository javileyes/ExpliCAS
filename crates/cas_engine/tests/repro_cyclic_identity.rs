#![allow(clippy::format_in_format_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(dead_code)]
#![allow(unused_variables)]
use cas_ast::DisplayExpr;
use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_engine::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeNegationRule,
};
use cas_engine::rules::polynomial::CombineLikeTermsRule;
use cas_engine::rules::polynomial::DistributeRule;
use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn test_cyclic_identity() {
    let mut simplifier = Simplifier::new();
    // Register standard rules
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));

    let input = "x*(y - z) + y*(z - x) + z*(x - y)";
    let expr = parse(input, &mut simplifier.context).expect("Failed to parse");

    let (result, steps) = simplifier.simplify(expr);
    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    println!("Result: {}", res_str);
    for step in steps {
        println!(
            "Step: {} -> {}",
            step.description,
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: step.after
                }
            )
        );
    }

    assert_eq!(res_str, "0");
}
