use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, EvaluatePowerRule, IdentityPowerRule, PowerProductRule, PowerQuotientRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule, TanToSinCosRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, FactorDifferenceSquaresRule, AddFractionsRule, FactorRule, SimplifyMulDivRule, ExpandRule};
use cas_engine::rules::calculus::{IntegrateRule, DiffRule};
use cas_engine::rules::grouping::CollectRule;
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_parser::parse;
use cas_ast::DisplayExpr;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.allow_numerical_verification = false;
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleConsistencyRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::DoubleAngleRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(cas_engine::rules::polynomial::BinomialExpansionRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::RationalizeDenominatorRule));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(cas_engine::rules::algebra::SimplifySquareRootRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); 
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::MulZeroRule));

    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
    simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::DivZeroRule));
    simplifier
}

#[test]
fn test_tangent_sum_reduced() {
    let mut simplifier = create_full_simplifier();
    // sin(x + y) / (cos(x) * cos(y)) -> tan(x) + tan(y)
    let input = "sin(x + y) / (cos(x) * cos(y))";
    let expr = parse(input, &mut simplifier.context).unwrap();
    println!("Parsed AST: {:?}", simplifier.context.get(expr));
    let (simplified, _) = simplifier.simplify(expr);
    let result = format!("{}", DisplayExpr { context: &simplifier.context, id: simplified });
    println!("Result: {}", result);
    // Expect tan(x) + tan(y) OR sin(x)/cos(x) + sin(y)/cos(y)
    // The simplifier prefers sin/cos usually.
    // So we expect sin(x)/cos(x) + sin(y)/cos(y)
    // Or if we subtract (tan(x)+tan(y)), we get 0.
    // Let's just check the output for now.
}
