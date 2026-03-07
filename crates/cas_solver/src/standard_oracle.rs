//! Standard Domain Oracle implementation for solver facade.
//!
//! This mirrors engine behavior while keeping oracle surface owned by
//! `cas_solver` during migration.

mod hint;
mod runtime;

pub use hint::oracle_allows_with_hint;
pub use runtime::StandardOracle;

#[cfg(test)]
mod tests {
    use super::*;
    use cas_solver_core::domain_facts_model::Predicate;
    use cas_solver_core::domain_oracle_model::DomainOracle;

    #[test]
    fn strict_unknown_nonzero_is_blocked() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(x));
        assert!(!decision.allow);
    }

    #[test]
    fn assume_unknown_nonzero_is_allowed() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(x));
        assert!(decision.allow);
    }

    #[test]
    fn strict_proven_nonzero_is_allowed() {
        let mut ctx = cas_ast::Context::new();
        let two = ctx.num(2);
        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(two));
        assert!(decision.allow);
        assert!(decision.assumption.is_none());
    }
}
