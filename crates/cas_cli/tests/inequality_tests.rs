use cas_ast::{BoundType, Equation, RelOp, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::rules::algebra::SimplifyFractionRule;
use cas_solver::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use cas_solver::rules::canonicalization::{
    CanonicalizeAddRule, CanonicalizeMulRule, CanonicalizeNegationRule, CanonicalizeRootRule,
};
use cas_solver::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, ProductPowerRule,
};
use cas_solver::rules::functions::EvaluateAbsRule;
use cas_solver::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_solver::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule, DistributeRule};
use cas_solver::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_solver::solve;
use cas_solver::Simplifier;

fn create_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier
}

#[test]
fn test_inequality_simple() {
    // 2x < 10 -> x < 5
    let mut simplifier = create_simplifier();
    let lhs_str = "2 * x";
    let rhs_str = "10";
    let op = RelOp::Lt;

    let lhs = parse(lhs_str, &mut simplifier.context).unwrap();
    let rhs = parse(rhs_str, &mut simplifier.context).unwrap();
    let eq = Equation { lhs, rhs, op };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        // x < 5 -> (-inf, 5)
        assert_eq!(interval.max_type, BoundType::Open);
        let (final_max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: final_max
                }
            ),
            "5"
        );
    } else {
        panic!("Expected Continuous solution");
    }
}

#[test]
fn test_inequality_flip_negative_coeff() {
    // -3x >= 9 -> x <= -3
    let mut simplifier = create_simplifier();
    let lhs_str = "-3 * x";
    let rhs_str = "9";
    let op = RelOp::Geq;

    let lhs = parse(lhs_str, &mut simplifier.context).unwrap();
    let rhs = parse(rhs_str, &mut simplifier.context).unwrap();
    let eq = Equation { lhs, rhs, op };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        // x <= -3 -> (-inf, -3]
        assert_eq!(interval.max_type, BoundType::Closed);
        let (final_max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: final_max
                }
            ),
            "-3"
        );
    } else {
        panic!("Expected Continuous solution");
    }
}

#[test]
fn test_abs_inequality_complex() {
    // |x - 1| < 2
    // Case 1: x - 1 < 2 -> x < 3
    // Case 2: x - 1 > -2 -> x > -1
    // Result: x < 3, x > -1

    let mut simplifier = create_simplifier();
    let lhs_str = "|x - 1|";
    let rhs_str = "2";
    let op = RelOp::Lt;
    let expected_min = "-1";
    let expected_max = "3";

    let lhs = parse(lhs_str, &mut simplifier.context).unwrap();
    let rhs = parse(rhs_str, &mut simplifier.context).unwrap();
    let eq = Equation { lhs, rhs, op };

    let (result, _) = solve(&eq, "x", &mut simplifier).expect("Failed to solve");

    if let SolutionSet::Continuous(interval) = result {
        // (-1, 3)
        let (min, _) = simplifier.simplify(interval.min);
        let (max, _) = simplifier.simplify(interval.max);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: min
                }
            ),
            expected_min
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: max
                }
            ),
            expected_max
        );
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}
