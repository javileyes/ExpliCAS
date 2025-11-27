use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule};
use cas_parser::parse;
use cas_ast::{Equation, RelOp, SolutionSet, BoundType};
use cas_engine::solver::solve;

fn create_full_simplifier() -> Simplifier {
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
    simplifier.add_rule(Box::new(ZeroOnePowerRule));
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
    let simplifier = create_full_simplifier();
    let lhs = parse("2 * x").unwrap();
    let rhs = parse("10").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Lt };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Continuous(interval) = result {
        // x < 5 -> (-inf, 5)
        assert_eq!(interval.max_type, BoundType::Open);
        let (final_max, _) = simplifier.simplify(interval.max.clone());
        assert_eq!(format!("{}", final_max), "5");
    } else {
        panic!("Expected Continuous solution");
    }
}

#[test]
fn test_inequality_flip_negative_coeff() {
    // -3x >= 9 -> x <= -3
    let simplifier = create_full_simplifier();
    let lhs = parse("-3 * x").unwrap();
    let rhs = parse("9").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Geq };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Continuous(interval) = result {
        // x <= -3 -> (-inf, -3]
        assert_eq!(interval.max_type, BoundType::Closed);
        let (final_max, _) = simplifier.simplify(interval.max.clone());
        assert_eq!(format!("{}", final_max), "-3");
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
    
    let simplifier = create_full_simplifier();
    let lhs = parse("|x - 1|").unwrap();
    let rhs = parse("2").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Lt };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Continuous(interval) = result {
        // (-1, 3)
        let (final_min, _) = simplifier.simplify(interval.min.clone());
        let (final_max, _) = simplifier.simplify(interval.max.clone());
        assert_eq!(format!("{}", final_min), "-1");
        assert_eq!(format!("{}", final_max), "3");
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}
