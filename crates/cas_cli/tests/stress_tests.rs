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
use cas_ast::{Equation, RelOp, SolutionSet};
use cas_engine::solver::solve;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    // Canonicalization first
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    
    // Evaluation
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    
    // Exponents
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(ZeroOnePowerRule));
    
    // Polynomials
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    
    // Algebra
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    
    // Arithmetic Cleanup
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    
    simplifier
}

#[test]
fn test_deeply_nested_simplification() {
    // ((x+1)^2 + (x-1)^2) / (x^2 + 1)
    // Numerator: (x^2 + 2x + 1) + (x^2 - 2x + 1) = 2x^2 + 2 = 2(x^2 + 1)
    // Result: 2
    
    let simplifier = create_full_simplifier();
    // Note: We need DistributeRule to expand (x+1)^2, but currently DistributeRule might only handle a*(b+c).
    // Power expansion (a+b)^2 is not yet implemented as a rule! 
    // This test is expected to fail or return unsimplified result if we don't have "ExpandPowerRule".
    // Let's see what happens.
    
    let input = "((x + 1)^2 + (x - 1)^2) / (x^2 + 1)";
    let expr = parse(input).unwrap();
    let (res, _) = simplifier.simplify(expr);
    
    // If it fails to simplify fully, it might remain as input.
    // If we want it to pass, we might need to implement ExpandPowerRule.
    // For stress testing, let's assert the ideal result and see.
    assert_eq!(format!("{}", res), "2");
}

#[test]
fn test_mixed_transcendental() {
    // sin(ln(exp(x))) -> sin(x)
    let simplifier = create_full_simplifier();
    let input = "sin(ln(exp(x)))";
    let expr = parse(input).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(format!("{}", res), "sin(x)");
}

#[test]
fn test_rational_simplification() {
    // (x^2 - 1) / (x - 1) -> x + 1
    let simplifier = create_full_simplifier();
    let input = "(x^2 - 1) / (x - 1)";
    let expr = parse(input).unwrap();
    let (res, _) = simplifier.simplify(expr);
    assert_eq!(format!("{}", res), "1 + x");
}

#[test]
fn test_quadratic_solver() {
    // solve(x^2 - 4 = 0, x) -> x = 2 (or x = -2, solver might pick one or return set)
    // Our solver currently isolates. x^2 = 4 -> x = 4^(1/2) -> x = 2.
    // It usually returns the principal root.
    
    let simplifier = create_full_simplifier();
    let lhs = parse("x^2 - 4").unwrap();
    let rhs = parse("0").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Eq };
    
    // Pre-simplify: x^2 - 4 = 0.
    // Solver needs to move 4.
    // Our solver handles "Sub(l, r) = RHS". If l has var (x^2), it adds r to RHS.
    // So x^2 = 4.
    // Then Pow(b, e) = RHS. b = RHS^(1/e). x = 4^(1/2).
    // Then simplify 4^(1/2) -> 2.
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Discrete(solutions) = result {
        assert!(!solutions.is_empty());
        assert!(!solutions.is_empty());
        let found = solutions.iter().any(|res_rhs| {
            let (final_rhs, _) = simplifier.simplify(res_rhs.clone());
            format!("{}", final_rhs) == "2"
        });
        assert!(found, "Expected solution '2' not found");
    } else {
        panic!("Expected Discrete solution");
    }
}

#[test]
fn test_exponential_solver() {
    // solve(exp(2*x) - 1 = 0, x) -> x = 0
    let simplifier = create_full_simplifier();
    let lhs = parse("exp(2 * x) - 1").unwrap();
    let rhs = parse("0").unwrap();
    let eq = Equation { lhs, rhs, op: RelOp::Eq };
    
    let (result, _) = solve(&eq, "x", &simplifier).expect("Failed to solve");
    
    if let SolutionSet::Discrete(solutions) = result {
        assert!(!solutions.is_empty());
        assert!(!solutions.is_empty());
        let found = solutions.iter().any(|res_rhs| {
            let (final_rhs, _) = simplifier.simplify(res_rhs.clone());
            format!("{}", final_rhs) == "0"
        });
        assert!(found, "Expected solution '0' not found");
    } else {
        panic!("Expected Discrete solution");
    }
}
