use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule}; // FactorDifferenceSquaresRule might not be public or exported
use cas_parser::parse;
use cas_ast::{Equation, RelOp, SolutionSet, BoundType, Expr};
use cas_engine::solver::solve;
use std::rc::Rc;
use num_traits::Zero; // Import Zero trait

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
    simplifier.add_rule(Box::new(cas_engine::rules::polynomial::BinomialExpansionRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); 
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::MulZeroRule));
    simplifier
}

fn assert_equivalent(s: &Simplifier, expr1: Rc<Expr>, expr2: Rc<Expr>) {
    let (sim1, _) = s.simplify(expr1.clone());
    let (sim2, _) = s.simplify(expr2.clone());
    // Check if sim1 == sim2 directly first
    if sim1 == sim2 {
        return;
    }
    // Try simplifying difference
    let diff = Expr::sub(sim1.clone(), sim2.clone());
    let (sim_diff, _) = s.simplify(diff);
    // Check if difference is 0
    if let Expr::Number(n) = sim_diff.as_ref() {
        if n.is_zero() {
            return;
        }
    }
    
    panic!("Expressions not equivalent.\nExpr1: {}\nSim1: {}\nExpr2: {}\nSim2: {}\nDiff: {}", 
           expr1, sim1, expr2, sim2, sim_diff);
}

// --- Level 1: Algebraic and Rational Simplification ---

#[test]
fn test_rational_simplification_invisible() {
    // simplify((x^3 - 1) / (x - 1)) -> x^2 + x + 1
    // Requires difference of cubics or polynomial division.
    let simplifier = create_full_simplifier();
    let input = parse("(x^3 - 1) / (x - 1)").unwrap();
    let expected = parse("x^2 + x + 1").unwrap();
    assert_equivalent(&simplifier, input, expected);
}

#[test]
fn test_exponential_expansion() {
    // expand((x + 1)^5)
    let simplifier = create_full_simplifier();
    let input = parse("(x + 1)^5").unwrap();
    let expected = parse("x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1").unwrap();
    assert_equivalent(&simplifier, input, expected);
}

#[test]
fn test_nested_fraction() {
    // simplify((1 + 1/x) / (1 - 1/x)) -> (x + 1) / (x - 1)
    let simplifier = create_full_simplifier();
    let input = parse("(1 + 1/x) / (1 - 1/x)").unwrap();
    let expected = parse("(x + 1) / (x - 1)").unwrap();
    assert_equivalent(&simplifier, input, expected);
}

// --- Level 2: Transcendental and Properties ---

#[test]
fn test_trig_identity_hidden() {
    // equiv(sin(x)^4 - cos(x)^4, sin(x)^2 - cos(x)^2) -> True
    let simplifier = create_full_simplifier();
    let expr1 = parse("sin(x)^4 - cos(x)^4").unwrap();
    let expr2 = parse("sin(x)^2 - cos(x)^2").unwrap();
    assert_equivalent(&simplifier, expr1, expr2);
}

#[test]
fn test_log_power_trap() {
    // simplify(ln(e^(x^2 + 1))) -> x^2 + 1
    let simplifier = create_full_simplifier();
    let input = parse("ln(e^(x^2 + 1))").unwrap();
    let expected = parse("x^2 + 1").unwrap();
    assert_equivalent(&simplifier, input, expected);
}

#[test]
fn test_log_cancellation() {
    // simplify(e^(ln(x) + ln(y))) -> x * y
    let simplifier = create_full_simplifier();
    let input = parse("e^(ln(x) + ln(y))").unwrap();
    let expected = parse("x * y").unwrap();
    assert_equivalent(&simplifier, input, expected);
}

// --- Level 3: The Solver ---

#[test]
fn test_hidden_quadratic_solve() {
    // solve e^(2*x) - 3*e^x + 2 = 0
    // u = e^x -> u^2 - 3u + 2 = 0 -> (u-2)(u-1)=0 -> u=2, u=1
    // e^x = 2 -> x = ln(2)
    // e^x = 1 -> x = 0
    let simplifier = create_full_simplifier();
    let lhs = parse("e^(2*x) - 3*e^x + 2").unwrap();
    let rhs = parse("0").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Eq };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Discrete(solutions) = result {
        // Expect 2 solutions
        assert_eq!(solutions.len(), 2, "Expected 2 solutions, got {:?}", solutions);
        // Check for 0 and ln(2)
        let has_zero = solutions.iter().any(|s| format!("{}", s) == "0");
        let has_ln2 = solutions.iter().any(|s| format!("{}", s) == "ln(2)");
        assert!(has_zero, "Missing solution x=0");
        assert!(has_ln2, "Missing solution x=ln(2)");
    } else {
        panic!("Expected Discrete solution, got {:?}", result);
    }
}

#[test]
fn test_nested_abs_solve() {
    // solve ||x - 1| - 2| = 1
    // |x-1| - 2 = 1  OR  |x-1| - 2 = -1
    // |x-1| = 3      OR  |x-1| = 1
    // x-1=3 -> x=4   OR  x-1=1 -> x=2
    // x-1=-3 -> x=-2 OR  x-1=-1 -> x=0
    // Solutions: 4, -2, 2, 0
    let simplifier = create_full_simplifier();
    let lhs = parse("||x - 1| - 2|").unwrap();
    let rhs = parse("1").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Eq };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Discrete(solutions) = result {
        assert_eq!(solutions.len(), 4, "Expected 4 solutions, got {:?}", solutions);
        let s_strs: Vec<String> = solutions.iter().map(|s| format!("{}", s)).collect();
        assert!(s_strs.contains(&"4".to_string()));
        assert!(s_strs.contains(&"-2".to_string()));
        assert!(s_strs.contains(&"2".to_string()));
        assert!(s_strs.contains(&"0".to_string()));
    } else {
        panic!("Expected Discrete solution, got {:?}", result);
    }
}

#[test]
fn test_rational_inequality_signs() {
    // solve (x - 1) / (x + 2) >= 0
    // Critical points: 1 (zero), -2 (pole)
    // Intervals: (-inf, -2), (-2, 1), (1, inf)
    // Test -3: (-4)/(-1) = 4 > 0 -> True
    // Test 0: (-1)/(2) = -0.5 < 0 -> False
    // Test 2: (1)/(4) = 0.25 > 0 -> True
    // Result: (-inf, -2) U [1, inf)
    // Note: -2 is OPEN because it's a pole. 1 is CLOSED because >= 0.
    
    let simplifier = create_full_simplifier();
    let lhs = parse("(x - 1) / (x + 2)").unwrap();
    let rhs = parse("0").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Geq };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Union(intervals) = result {
        assert_eq!(intervals.len(), 2);
        // (-inf, -2)
        let i1 = &intervals[0];
        let (min1, _) = simplifier.simplify(i1.min.clone());
        let (max1, _) = simplifier.simplify(i1.max.clone());
        assert_eq!(format!("{}", min1), "-1 * infinity");
        assert_eq!(format!("{}", max1), "-2");
        assert_eq!(i1.max_type, BoundType::Open, "Pole at -2 should be Open");
        
        // [1, inf)
        let i2 = &intervals[1];
        let (min2, _) = simplifier.simplify(i2.min.clone());
        let (max2, _) = simplifier.simplify(i2.max.clone());
        assert_eq!(format!("{}", min2), "1");
        assert_eq!(format!("{}", max2), "infinity");
        assert_eq!(i2.min_type, BoundType::Closed, "Zero at 1 should be Closed for >=");
    } else {
        panic!("Expected Union solution, got {:?}", result);
    }
}

#[test]
fn test_quadratic_abs_inequality() {
    // solve |x^2 - 1| < 3
    // -3 < x^2 - 1 < 3
    // x^2 - 1 > -3 -> x^2 > -2 (Always true)
    // x^2 - 1 < 3 -> x^2 < 4 -> (-2, 2)
    // Intersection: (-2, 2)
    
    let simplifier = create_full_simplifier();
    let lhs = parse("|x^2 - 1|").unwrap();
    let rhs = parse("3").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Lt };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Continuous(interval) = result {
        let (min, _) = simplifier.simplify(interval.min.clone());
        let (max, _) = simplifier.simplify(interval.max.clone());
        assert_eq!(format!("{}", min), "-2");
        assert_eq!(format!("{}", max), "2");
    } else {
        panic!("Expected Continuous solution, got {:?}", result);
    }
}
