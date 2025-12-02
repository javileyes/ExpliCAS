use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, EvaluatePowerRule, IdentityPowerRule, PowerProductRule, PowerQuotientRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeDivRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule, TanToSinCosRule, RecursiveTrigExpansionRule, CanonicalizeTrigSquareRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, FactorDifferenceSquaresRule, AddFractionsRule, FactorRule, SimplifyMulDivRule, ExpandRule, PullConstantFromFractionRule};
use cas_engine::rules::calculus::{IntegrateRule, DiffRule};
use cas_engine::rules::grouping::CollectRule;
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_parser::parse;
use cas_ast::DisplayExpr;
use std::collections::HashMap;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.allow_numerical_verification = false;
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleConsistencyRule));
    simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::DoubleAngleRule));
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(cas_engine::rules::exponents::NegativeBasePowerRule));
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
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
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
fn profile_torture_tests() {
    let cases = vec![
        ("Rational Telescoping", "1/(x*(x+1)) + 1/((x+1)*(x+2)) - 2/(x*(x+2))"),
        ("Trig Shift", "sin(x + pi/2) - cos(x)"),
        ("Lagrange Identity", "expand((a^2 + b^2) * (c^2 + d^2) - (a*c + b*d)^2 - (a*d - b*c)^2)"),
        ("Hyperbolic Masquerade", "((e^x + e^(-x))/2)^2 - ((e^x - e^(-x))/2)^2 - 1"),
        ("Tangent Sum", "sin(x + y) / (cos(x) * cos(y)) - (tan(x) + tan(y))"),
        ("Log Sqrt", "ln(sqrt(x^2 + 2*x + 1)) - ln(x + 1)"),
        ("Trig Power", "8 * sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x))"),
        ("Sophie Germain", "expand((x^2 + 2*y^2 + 2*x*y) * (x^2 + 2*y^2 - 2*x*y) - (x^4 + 4*y^4))"),
        ("Angle Compound", "sin(x + y) - (sin(x)*cos(y) + cos(x)*sin(y))"),
        ("Difference Quotient", "((x + h)^3 - x^3) / h - (3*x^2 + 3*x*h + h^2)"),
        ("Polynomial Stress", "(x - 1) * (x + 1) * (x^2 + 1) * (x^4 + 1) - (x^8 - 1)"),
        ("Abs Squared", "abs(x)^2 - x^2"),
        ("Log Chain", "(ln(x) / ln(10)) * (ln(10) / ln(x)) - 1"),
        ("Conjugate", "1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)"),
    ];

    println!("| Test Case | Steps | Result |");
    println!("|---|---|---|");

    for (name, input) in cases {
        let mut simplifier = create_full_simplifier();
        let expr = parse(input, &mut simplifier.context).expect("Parse failed");
        
        let (res, steps) = simplifier.simplify(expr);
        let res_str = format!("{}", DisplayExpr { context: &simplifier.context, id: res });
        
        println!("| {} | {} | {} |", name, steps.len(), res_str);
        
        // Analyze for cycles
        let mut seen = HashMap::new();
        for (i, step) in steps.iter().enumerate() {
            let s = format!("{}", DisplayExpr { context: &simplifier.context, id: step.after });
            if let Some(_prev_idx) = seen.insert(s.clone(), i) {
                // Cycle detected? Not necessarily, could be same state reached via different path.
                // But if we see the same state multiple times, it's interesting.
                // println!("  [Cycle/Repeat] State at step {} seen at step {}: {}", i, prev_idx, s);
            }
        }
        
        // Analyze rule usage
        let mut rule_counts = HashMap::new();
        for step in steps.iter() {
            *rule_counts.entry(step.rule_name.clone()).or_insert(0) += 1;
        }
        // println!("  Top rules: {:?}", rule_counts);
    }
}
