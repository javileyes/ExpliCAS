// GCD Gauntlet - Round 8 Torture Tests
// Tests for polynomial GCD correctness and efficiency

use cas_solver::rules::algebra::{
    AddFractionsRule, ExpandRule, FactorDifferenceSquaresRule, FactorRule,
    PullConstantFromFractionRule, SimplifyFractionRule, SimplifyMulDivRule,
};
use cas_solver::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use cas_solver::rules::calculus::{DiffRule, IntegrateRule};
use cas_solver::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeDivRule, CanonicalizeMulRule, CanonicalizeNegationRule,
    CanonicalizeRootRule,
};
use cas_solver::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_solver::rules::functions::EvaluateAbsRule;
use cas_solver::rules::grouping::CollectRule;
use cas_solver::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use cas_solver::rules::number_theory::NumberTheoryRule;
use cas_solver::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule, DistributeRule};
use cas_solver::rules::trigonometry::{
    CanonicalizeTrigSquareRule, EvaluateTrigRule, PythagoreanIdentityRule,
    RecursiveTrigExpansionRule, TanToSinCosRule,
};
use cas_solver::Simplifier;

use cas_ast::Expr;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

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
    simplifier.add_rule(Box::new(cas_solver::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(cas_solver::rules::trigonometry::AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(
        cas_solver::rules::trigonometry::AngleConsistencyRule,
    ));
    simplifier.add_rule(Box::new(cas_solver::rules::trigonometry::DoubleAngleRule));
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
    simplifier.add_rule(Box::new(
        cas_solver::rules::exponents::NegativeBasePowerRule,
    ));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(
        cas_solver::rules::polynomial::BinomialExpansionRule,
    ));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(
        cas_solver::rules::algebra::RationalizeDenominatorRule,
    ));
    simplifier.add_rule(Box::new(
        cas_solver::rules::algebra::CancelCommonFactorsRule,
    ));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    simplifier.add_rule(Box::new(FactorDifferenceSquaresRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(cas_solver::rules::arithmetic::MulZeroRule));
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
    simplifier.add_rule(Box::new(cas_solver::rules::arithmetic::DivZeroRule));
    simplifier
}

#[test]
fn test_gcd_41_non_monic() {
    // 41. El Clásico No-Mónico (Non-Monic Common Factor)
    // gcd(2*x^2 + 7*x + 3, 2*x^2 + 5*x + 2) = 2*x + 1
    // Polynomials:
    // A = (2x + 1)(x + 3)
    // B = (2x + 1)(x + 2)
    // GCD = 2x + 1 (or monic: x + 0.5)

    let mut simplifier = create_full_simplifier();

    // Test via fraction simplification
    let input = parse(
        "(2*x^2 + 7*x + 3) / (2*x^2 + 5*x + 2)",
        &mut simplifier.context,
    )
    .unwrap();
    let (result, _) = simplifier.simplify(input);

    // After canceling (2x+1), should get (x+3)/(x+2)
    let expected = parse("(x + 3) / (x + 2)", &mut simplifier.context).unwrap();
    assert!(
        simplifier.are_equivalent(result, expected),
        "Non-monic GCD failed.\nGot: {}\nExpected: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        },
        DisplayExpr {
            context: &simplifier.context,
            id: expected
        }
    );
}

#[test]
fn test_gcd_42_coprime() {
    // 42. El Primos Relativos (Coprime Check)
    // gcd(x^3 + x + 1, x^2 + x + 1) = 1
    // These polynomials share no common roots

    let mut simplifier = create_full_simplifier();

    // If coprime, fraction shouldn't simplify
    let input = parse("(x^3 + x + 1) / (x^2 + x + 1)", &mut simplifier.context).unwrap();
    let (result, _) = simplifier.simplify(input);

    // Should remain a fraction (no common factors to cancel)
    // If it simplified to something other than a fraction, GCD is wrong
    if let Expr::Div(_, _) = simplifier.context.get(result) {
        // Good - it's still a fraction
    } else {
        panic!("Coprime test failed: fraction simplified when it shouldn't have");
    }
}

#[test]
fn test_gcd_43_high_degree_sparse() {
    // 43. La Pesadilla Dispersa (High Degree Sparse)
    // gcd(x^100 - 1, x^25 - 1) = x^25 - 1
    // Uses property: gcd(x^m - 1, x^n - 1) = x^gcd(m,n) - 1
    // gcd(100, 25) = 25

    let mut simplifier = create_full_simplifier();

    // Test via fraction
    let input = parse("(x^100 - 1) / (x^25 - 1)", &mut simplifier.context).unwrap();
    let (result, _) = simplifier.simplify(input);

    // After canceling x^25 - 1, should get: (x^75 + x^50 + x^25 + 1)
    // This is x^100 - 1 divided by x^25 - 1
    // Actually: (x^100-1)/(x^25-1) = Sum_{k=0}^{3} x^{25k} = 1 + x^25 + x^50 + x^75
    let expected = parse("1 + x^25 + x^50 + x^75", &mut simplifier.context).unwrap();
    assert!(
        simplifier.are_equivalent(result, expected),
        "Sparse high-degree GCD failed.\nGot: {}\nExpected: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        },
        DisplayExpr {
            context: &simplifier.context,
            id: expected
        }
    );
}

#[test]
fn test_gcd_44_coefficient_swell() {
    // 44. El Constructor de Explosiones (Coefficient Swell Test)
    // gcd((x^3 + 3*x + 5)*(x^2 - 2), (x^3 + 3*x + 5)*(x^2 + 3*x + 1))
    // Common factor: x^3 + 3*x + 5
    // Remaining factors: (x^2 - 2) and (x^2 + 3*x + 1) are coprime

    let mut simplifier = create_full_simplifier();

    // Expand and then simplify the fraction
    let input = parse(
        "expand((x^3 + 3*x + 5)*(x^2 - 2)) / expand((x^3 + 3*x + 5)*(x^2 + 3*x + 1))",
        &mut simplifier.context,
    )
    .unwrap();
    let (result, _) = simplifier.simplify(input);

    // After canceling (x^3 + 3*x + 5), should get (x^2 - 2)/(x^2 + 3*x + 1)
    let expected = parse("(x^2 - 2) / (x^2 + 3*x + 1)", &mut simplifier.context).unwrap();
    assert!(
        simplifier.are_equivalent(result, expected),
        "Coefficient swell test failed.\nGot: {}\nExpected: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        },
        DisplayExpr {
            context: &simplifier.context,
            id: expected
        }
    );
}

#[test]
fn test_gcd_45_fraction_reduction() {
    // 45. La Simplificación Definitiva (Fraction Reduction)
    // (x^4 - 1) / (x^3 - x^2 + x - 1) => x + 1
    //
    // Numerator: x^4 - 1 = (x^2 - 1)(x^2 + 1) = (x-1)(x+1)(x^2+1)
    // Denominator: x^3 - x^2 + x - 1 = x^2(x-1) + 1(x-1) = (x-1)(x^2+1)
    // GCD: (x-1)(x^2+1) = x^3 - x^2 + x - 1
    // Result: (x+1)

    let mut simplifier = create_full_simplifier();

    let input = parse("(x^4 - 1) / (x^3 - x^2 + x - 1)", &mut simplifier.context).unwrap();
    let (result, _) = simplifier.simplify(input);

    let expected = parse("x + 1", &mut simplifier.context).unwrap();
    assert!(
        simplifier.are_equivalent(result, expected),
        "Fraction reduction test failed.\nGot: {}\nExpected: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        },
        DisplayExpr {
            context: &simplifier.context,
            id: expected
        }
    );
}
