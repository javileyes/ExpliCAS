use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, MulZeroRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule, DistributeRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule, AssociativityRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule};
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
    
    // Arithmetic
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    
    simplifier
}

#[test]
fn test_trig_identity_equivalence() {
    let simplifier = create_full_simplifier();
    let e1 = parse("sin(x)^2 + cos(x)^2").unwrap();
    let e2 = parse("1").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_polynomial_equivalence() {
    let simplifier = create_full_simplifier();
    // 2(x+1) equiv 2x+2
    let e1 = parse("2 * (x + 1)").unwrap();
    let e2 = parse("2 * x + 2").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_not_equivalent() {
    let simplifier = create_full_simplifier();
    let e1 = parse("x").unwrap();
    let e2 = parse("x + 1").unwrap();
    assert!(!simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_complex_equivalence() {
    let simplifier = create_full_simplifier();
    // ((x+1)^2 + (x-1)^2) / (x^2 + 1) equiv 2
    let e1 = parse("((x + 1)^2 + (x - 1)^2) / (x^2 + 1)").unwrap();
    let e2 = parse("2").unwrap();
    
    // Note: This relies on the same logic as stress tests.
    // If stress test passed, this should pass.
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_commutativity_equivalence() {
    let simplifier = create_full_simplifier();
    // x + y equiv y + x
    let e1 = parse("x + y").unwrap();
    let e2 = parse("y + x").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
    
    // x * y equiv y * x
    let e1 = parse("x * y").unwrap();
    let e2 = parse("y * x").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_associativity_equivalence() {
    let simplifier = create_full_simplifier();
    // (x + y) + z equiv x + (y + z)
    let e1 = parse("(x + y) + z").unwrap();
    let e2 = parse("x + (y + z)").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_exponent_equivalence() {
    let simplifier = create_full_simplifier();
    // (x^2)^3 equiv x^6
    let e1 = parse("(x^2)^3").unwrap();
    let e2 = parse("x^6").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_log_exp_equivalence() {
    let simplifier = create_full_simplifier();
    // exp(ln(x)) equiv x
    let e1 = parse("exp(ln(x))").unwrap();
    let e2 = parse("x").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_zero_property_equivalence() {
    let simplifier = create_full_simplifier();
    // x * 0 equiv 0
    let e1 = parse("x * 0").unwrap();
    let e2 = parse("0").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_deep_associativity() {
    let simplifier = create_full_simplifier();
    // ((a + b) + c) + d equiv a + (b + (c + d))
    let e1 = parse("((a + b) + c) + d").unwrap();
    let e2 = parse("a + (b + (c + d))").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_mixed_distributive_associativity() {
    let simplifier = create_full_simplifier();
    // 2*x + 2*y + 2*z equiv 2*(x + y + z)
    // Note: 2*(x+y+z) parses as 2 * ((x+y)+z) usually.
    // DistributeRule should handle this.
    let e1 = parse("2*x + 2*y + 2*z").unwrap();
    let e2 = parse("2 * (x + y + z)").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}

#[test]
fn test_trig_associativity() {
    let simplifier = create_full_simplifier();
    // sin(x)^2 + (cos(x)^2 + 1) equiv 2
    // This requires:
    // 1. Flattening: sin^2 + cos^2 + 1
    // 2. Sorting: cos^2 + sin^2 + 1 (if cos < sin)
    // 3. Identity: 1 + 1 -> 2
    let e1 = parse("sin(x)^2 + (cos(x)^2 + 1)").unwrap();
    let e2 = parse("2").unwrap();
    assert!(simplifier.are_equivalent(e1, e2));
}
