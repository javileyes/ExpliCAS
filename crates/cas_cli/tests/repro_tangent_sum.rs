#![allow(clippy::format_in_format_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(dead_code)]
#![allow(unused_variables)]
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::rules::algebra::{
    AddFractionsRule, CancelCommonFactorsRule, SimplifyFractionRule, SimplifyMulDivRule,
};
use cas_solver::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_solver::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeNegationRule,
};
use cas_solver::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_solver::rules::polynomial::{BinomialExpansionRule, CombineLikeTermsRule, DistributeRule};
use cas_solver::rules::trigonometry::{AngleIdentityRule, TanToSinCosRule};
use cas_solver::Simplifier;

fn create_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(DistributeRule));

    simplifier.add_rule(Box::new(BinomialExpansionRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier
}

#[test]
fn test_tangent_sum() {
    // Use default simplifier with all rules registered for proper angle expansion
    let mut simplifier = Simplifier::with_default_rules();
    // sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))
    let input = "sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let result = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: simplified
        }
    );
    assert_eq!(result, "0", "Tangent Sum failed to simplify to 0");
}
