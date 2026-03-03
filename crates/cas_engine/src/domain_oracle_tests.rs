use crate::domain::DomainMode;
use crate::domain_facts::{DomainOracle, FactStrength, Predicate};
use crate::domain_oracle::StandardOracle;
use crate::semantics::ValueDomain;
use cas_ast::Context;

#[test]
fn test_oracle_nonzero_constant() {
    let mut ctx = Context::new();
    let two = ctx.num(2);

    let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

    // 2 is provably non-zero
    assert_eq!(oracle.query(&Predicate::NonZero(two)), FactStrength::Proven);
    assert!(oracle.allows(&Predicate::NonZero(two)).allow);
}

#[test]
fn test_oracle_nonzero_zero() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);

    let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

    // 0 is provably zero (disproven nonzero)
    assert_eq!(
        oracle.query(&Predicate::NonZero(zero)),
        FactStrength::Disproven
    );
    // Even in Assume mode, disproven is denied
    assert!(!oracle.allows(&Predicate::NonZero(zero)).allow);
}

#[test]
fn test_oracle_nonzero_variable_strict() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

    // x is unknown nonzero
    assert_eq!(oracle.query(&Predicate::NonZero(x)), FactStrength::Unknown);
    // Strict blocks unknown
    assert!(!oracle.allows(&Predicate::NonZero(x)).allow);
}

#[test]
fn test_oracle_nonzero_variable_generic() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let oracle = StandardOracle::new(&ctx, DomainMode::Generic, ValueDomain::RealOnly);

    // x is unknown nonzero
    assert_eq!(oracle.query(&Predicate::NonZero(x)), FactStrength::Unknown);
    // Generic allows NonZero (Definability)
    assert!(oracle.allows(&Predicate::NonZero(x)).allow);
}

#[test]
fn test_oracle_positive_constant() {
    let mut ctx = Context::new();
    let three = ctx.num(3);

    let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

    // 3 is provably positive
    assert_eq!(
        oracle.query(&Predicate::Positive(three)),
        FactStrength::Proven
    );
    assert!(oracle.allows(&Predicate::Positive(three)).allow);
}

#[test]
fn test_oracle_positive_variable_generic() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let oracle = StandardOracle::new(&ctx, DomainMode::Generic, ValueDomain::RealOnly);

    // x is unknown positive
    assert_eq!(oracle.query(&Predicate::Positive(x)), FactStrength::Unknown);
    // Generic blocks Positive (Analytic)
    assert!(!oracle.allows(&Predicate::Positive(x)).allow);
}

#[test]
fn test_oracle_positive_variable_assume() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

    // x is unknown positive
    assert_eq!(oracle.query(&Predicate::Positive(x)), FactStrength::Unknown);
    // Assume allows everything
    assert!(oracle.allows(&Predicate::Positive(x)).allow);
}

#[test]
fn test_oracle_nonnegative_zero() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);

    let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

    // 0 is provably non-negative
    assert_eq!(
        oracle.query(&Predicate::NonNegative(zero)),
        FactStrength::Proven
    );
    assert!(oracle.allows(&Predicate::NonNegative(zero)).allow);
}

#[test]
fn test_oracle_defined_always_unknown() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

    // Defined is always Unknown (no dedicated prover)
    assert_eq!(oracle.query(&Predicate::Defined(x)), FactStrength::Unknown);
}

#[test]
fn test_oracle_parity_with_can_cancel_factor() {
    // Exhaustive parity test: oracle.allows() must produce the same
    // allow/deny as can_cancel_factor() for all (mode, proof) combos
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let zero = ctx.num(0);
    let x = ctx.var("x");

    for mode in [DomainMode::Strict, DomainMode::Generic, DomainMode::Assume] {
        let oracle = StandardOracle::new(&ctx, mode, ValueDomain::RealOnly);

        // Proven nonzero (constant 2)
        let oracle_decision = oracle.allows(&Predicate::NonZero(two));
        let proof = crate::helpers::prove_nonzero(&ctx, two);
        let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
        assert_eq!(
            oracle_decision.allow, legacy_decision.allow,
            "Mismatch for mode={:?}, expr=2",
            mode
        );

        // Disproven nonzero (constant 0)
        let oracle_decision = oracle.allows(&Predicate::NonZero(zero));
        let proof = crate::helpers::prove_nonzero(&ctx, zero);
        let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
        assert_eq!(
            oracle_decision.allow, legacy_decision.allow,
            "Mismatch for mode={:?}, expr=0",
            mode
        );

        // Unknown nonzero (variable x)
        let oracle_decision = oracle.allows(&Predicate::NonZero(x));
        let proof = crate::helpers::prove_nonzero(&ctx, x);
        let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
        assert_eq!(
            oracle_decision.allow, legacy_decision.allow,
            "Mismatch for mode={:?}, expr=x",
            mode
        );
    }
}
