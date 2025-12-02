use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, MulZeroRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, DistributeRule, BinomialExpansionRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, EvaluatePowerRule, IdentityPowerRule, PowerProductRule, PowerQuotientRule};
use cas_engine::rules::canonicalization::{CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, AddFractionsRule, SimplifyMulDivRule, CancelCommonFactorsRule};
use cas_engine::rules::trigonometry::{AngleIdentityRule, TanToSinCosRule};
use cas_parser::parse;
use cas_ast::DisplayExpr;

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
    let mut simplifier = create_simplifier();
    // sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))
    let input = "sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    let (simplified, _) = simplifier.simplify(expr);
    let result = format!("{}", DisplayExpr { context: &simplifier.context, id: simplified });
    assert_eq!(result, "0", "Tangent Sum failed to simplify to 0");
}
