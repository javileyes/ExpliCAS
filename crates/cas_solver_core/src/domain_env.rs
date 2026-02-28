//! Generic domain environment for solver contexts.
//!
//! `cas_engine` (or any runtime crate) can provide its own required-domain
//! storage type by implementing [`RequiredDomainSet`].

use cas_ast::{ConditionSet, ExprId};

/// Contract required by solver domain environments.
pub trait RequiredDomainSet {
    fn contains_positive(&self, expr: ExprId) -> bool;
    fn contains_nonnegative(&self, expr: ExprId) -> bool;
    fn contains_nonzero(&self, expr: ExprId) -> bool;
    fn to_condition_set(&self) -> ConditionSet;
}

/// Solver domain environment (per recursion level).
#[derive(Debug, Clone, Default)]
pub struct SolveDomainEnv<R> {
    /// Required conditions inferred from equation structure.
    pub required: R,
}

impl<R> SolveDomainEnv<R>
where
    R: RequiredDomainSet + Default,
{
    /// Create a new empty environment.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R> SolveDomainEnv<R>
where
    R: RequiredDomainSet,
{
    /// Check if a Positive condition is already in the required set.
    pub fn has_positive(&self, expr: ExprId) -> bool {
        self.required.contains_positive(expr)
    }

    /// Check if a NonNegative condition is already in the required set.
    pub fn has_nonnegative(&self, expr: ExprId) -> bool {
        self.required.contains_nonnegative(expr)
    }

    /// Check if a NonZero condition is already in the required set.
    pub fn has_nonzero(&self, expr: ExprId) -> bool {
        self.required.contains_nonzero(expr)
    }

    /// Convert required conditions to a condition set for guard composition.
    pub fn required_as_condition_set(&self) -> ConditionSet {
        self.required.to_condition_set()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use cas_ast::{ConditionPredicate, ConditionSet, Context, ExprId};

    use super::{RequiredDomainSet, SolveDomainEnv};

    #[derive(Debug, Clone, Default)]
    struct DummyRequiredSet {
        positive: HashSet<ExprId>,
        nonnegative: HashSet<ExprId>,
        nonzero: HashSet<ExprId>,
    }

    impl RequiredDomainSet for DummyRequiredSet {
        fn contains_positive(&self, expr: ExprId) -> bool {
            self.positive.contains(&expr)
        }

        fn contains_nonnegative(&self, expr: ExprId) -> bool {
            self.nonnegative.contains(&expr)
        }

        fn contains_nonzero(&self, expr: ExprId) -> bool {
            self.nonzero.contains(&expr)
        }

        fn to_condition_set(&self) -> ConditionSet {
            let mut out = ConditionSet::empty();
            for expr in self.positive.iter().copied() {
                out.push(ConditionPredicate::Positive(expr));
            }
            for expr in self.nonnegative.iter().copied() {
                out.push(ConditionPredicate::NonNegative(expr));
            }
            for expr in self.nonzero.iter().copied() {
                out.push(ConditionPredicate::NonZero(expr));
            }
            out
        }
    }

    #[test]
    fn solve_domain_env_queries_required_flags() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let mut required = DummyRequiredSet::default();
        required.positive.insert(x);
        required.nonzero.insert(x);

        let env = SolveDomainEnv { required };
        assert!(env.has_positive(x));
        assert!(!env.has_nonnegative(x));
        assert!(env.has_nonzero(x));
    }
}
