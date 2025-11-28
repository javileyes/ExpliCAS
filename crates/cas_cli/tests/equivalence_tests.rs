use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, MulZeroRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule, AssociativityRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule, AngleIdentityRule, TanToSinCosRule, DoubleAngleRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, ExpandRule};
use cas_ast::{Equation, RelOp, SolutionSet, BoundType, Expr, Context, ExprId, DisplayExpr};
use cas_engine::solver::solve;
use num_traits::Zero;
use cas_parser::parse;

fn create_full_simplifier() -> Simplifier {
    let mut simplifier = Simplifier::new();
    // Canonicalization
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(AssociativityRule));
    
    // Evaluation
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
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
    simplifier.add_rule(Box::new(ExpandRule));
    
    // Arithmetic
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    
    simplifier
}

#[test]
fn test_trig_identity_equivalence() {
    let mut simplifier = create_full_simplifier();
    let e1 = parse("sin(x)^2 + cos(x)^2", &mut simplifier.context).unwrap();
    let e2 = parse("1", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_polynomial_equivalence() {
    let mut simplifier = create_full_simplifier();
    // 2(x+1) equiv 2x+2
    let e1 = parse("2 * (x + 1)", &mut simplifier.context).unwrap();
    let e2 = parse("2 * x + 2", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_not_equivalent() {
    let mut simplifier = create_full_simplifier();
    let input = parse("x", &mut simplifier.context).unwrap();
    let expected = parse("x + 1", &mut simplifier.context).unwrap();
    assert!(!simplifier.are_equivalent(input, expected));
}

#[test]
fn test_complex_equivalence() {
    let mut simplifier = create_full_simplifier();
    // ((x+1)^2 + (x-1)^2) / (x^2 + 1) equiv 2
    let e1 = parse("((x + 1)^2 + (x - 1)^2) / (x^2 + 1)", &mut simplifier.context).unwrap();
    let e2 = parse("2", &mut simplifier.context).unwrap();
    
    // Note: This relies on the same logic as stress tests.
    // If stress test passed, this should pass.
    assert!(simplifier.are_equivalent(e1, e2));
}

fn assert_equivalent(s: &mut Simplifier, expr1: ExprId, expr2: ExprId) {
    let (sim1, _) = s.simplify(expr1);
    let (sim2, _) = s.simplify(expr2);
    
    if s.are_equivalent(sim1, sim2) {
        return;
    }
    
    let diff = s.context.add(Expr::Sub(sim1, sim2));
    let (sim_diff, _) = s.simplify(diff);
    
    if let Expr::Number(n) = s.context.get(sim_diff) {
        if n.is_zero() {
            return;
        }
    }
    
    panic!("Expressions not equivalent.\nExpr1: {}\nSim1: {}\nExpr2: {}\nSim2: {}\nDiff: {}", 
           DisplayExpr { context: &s.context, id: expr1 },
           DisplayExpr { context: &s.context, id: sim1 },
           DisplayExpr { context: &s.context, id: expr2 },
           DisplayExpr { context: &s.context, id: sim2 },
           DisplayExpr { context: &s.context, id: sim_diff });
}

#[test]
fn test_commutativity_equivalence() {
    let mut simplifier = create_full_simplifier();
    // x + y equiv y + x
    let e1 = parse("x + y", &mut simplifier.context).unwrap();
    let e2 = parse("y + x", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
    
    // x * y equiv y * x
    let e1 = parse("x * y", &mut simplifier.context).unwrap();
    let e2 = parse("y * x", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_associativity_equivalence() {
    let mut simplifier = create_full_simplifier();
    // (x + y) + z equiv x + (y + z)
    let e1 = parse("(x + y) + z", &mut simplifier.context).unwrap();
    let e2 = parse("x + (y + z)", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_exponent_equivalence() {
    let mut simplifier = create_full_simplifier();
    // (x^2)^3 equiv x^6
    let e1 = parse("(x^2)^3", &mut simplifier.context).unwrap();
    let e2 = parse("x^6", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_log_exp_equivalence() {
    let mut simplifier = create_full_simplifier();
    // exp(ln(x)) equiv x
    let e1 = parse("exp(ln(x))", &mut simplifier.context).unwrap();
    let e2 = parse("x", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_zero_property_equivalence() {
    let mut simplifier = create_full_simplifier();
    // x * 0 equiv 0
    let e1 = parse("x * 0", &mut simplifier.context).unwrap();
    let e2 = parse("0", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_deep_associativity() {
    let mut simplifier = create_full_simplifier();
    // ((a + b) + c) + d equiv a + (b + (c + d))
    let e1 = parse("((a + b) + c) + d", &mut simplifier.context).unwrap();
    let e2 = parse("a + (b + (c + d))", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_mixed_distributive_associativity() {
    let mut simplifier = create_full_simplifier();
    // 2*x + 2*y + 2*z equiv 2*(x + y + z)
    // Note: 2*(x+y+z) parses as 2 * ((x+y)+z) usually.
    // DistributeRule should handle this.
    let e1 = parse("2*x + 2*y + 2*z", &mut simplifier.context).unwrap();
    let e2 = parse("2 * (x + y + z)", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_trig_associativity() {
    let mut simplifier = create_full_simplifier();
    // sin(x)^2 + (cos(x)^2 + 1) equiv 2
    // This requires:
    // 1. Flattening: sin^2 + cos^2 + 1
    // 2. Sorting: cos^2 + sin^2 + 1 (if cos < sin)
    // 3. Identity: 1 + 1 -> 2
    let e1 = parse("sin(x)^2 + (cos(x)^2 + 1)", &mut simplifier.context).unwrap();
    let e2 = parse("2", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_numeric_verification() {
    let mut simplifier = create_full_simplifier();
    // sin(x + y) equiv sin(x)cos(y) + cos(x)sin(y)
    // We now have AngleIdentityRule, so this should pass symbolically.
    // But we keep it here to ensure it passes (and numeric check is a fallback).
    let e1 = parse("sin(x + y)", &mut simplifier.context).unwrap();
    let e2 = parse("sin(x)*cos(y) + cos(x)*sin(y)", &mut simplifier.context).unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}
