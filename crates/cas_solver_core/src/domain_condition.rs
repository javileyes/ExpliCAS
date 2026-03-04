//! Implicit-domain condition model shared by engine/solver layers.
//!
//! This module owns:
//! - condition vocabulary (`ImplicitCondition`)
//! - display policy (`RequiresDisplayLevel`)
//! - inferred condition set container (`ImplicitDomain`)

use cas_ast::{Context, ExprId};
use std::collections::HashSet;

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x > 0 (from ln(x) or log(x))
    Positive(ExprId),
    /// x ≠ 0 (from 1/x or Div(_, x))
    NonZero(ExprId),
}

impl ImplicitCondition {
    /// Human-readable display for REPL/UI.
    pub fn display(&self, ctx: &Context) -> String {
        use cas_formatter::DisplayExpr;

        match self {
            Self::NonNegative(e) => format!(
                "{} ≥ 0",
                DisplayExpr {
                    context: ctx,
                    id: *e
                }
            ),
            Self::Positive(e) => format!(
                "{} > 0",
                DisplayExpr {
                    context: ctx,
                    id: *e
                }
            ),
            Self::NonZero(e) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: ctx,
                    id: *e
                }
            ),
        }
    }

    /// Check if this condition is trivial (always true or on a constant expression).
    pub fn is_trivial(&self, ctx: &Context) -> bool {
        let expr = match self {
            Self::NonNegative(e) | Self::Positive(e) | Self::NonZero(e) => *e,
        };

        // Fully numeric expressions are trivial in this context.
        if !cas_math::expr_predicates::contains_variable(ctx, expr) {
            return true;
        }

        // x^2 >= 0-like predicates are always true and not useful to display.
        if let Self::NonNegative(e) = self {
            if cas_math::expr_predicates::is_always_nonnegative_expr(ctx, *e) {
                return true;
            }
        }

        false
    }

    /// Check if this condition's witness survives in the output expression.
    pub fn witness_survives_in(&self, ctx: &Context, output: ExprId) -> bool {
        use cas_math::expr_witness::{witness_survives, WitnessKind};

        match self {
            Self::NonNegative(e) => witness_survives(ctx, *e, output, WitnessKind::Sqrt),
            Self::Positive(e) => witness_survives(ctx, *e, output, WitnessKind::Log),
            Self::NonZero(e) => witness_survives(ctx, *e, output, WitnessKind::Division),
        }
    }
}

/// Display level for required conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequiresDisplayLevel {
    /// Show only essential requires (witness consumed or from equation derivation)
    #[default]
    Essential,
    /// Show all requires including implicit ones (witness survives)
    All,
}

/// Filter required conditions based on display level and witness survival.
pub fn filter_requires_for_display<'a>(
    requires: &'a [ImplicitCondition],
    ctx: &Context,
    result: ExprId,
    level: RequiresDisplayLevel,
) -> Vec<&'a ImplicitCondition> {
    requires
        .iter()
        .filter(|cond| {
            if level == RequiresDisplayLevel::All {
                return true;
            }
            !cond.witness_survives_in(ctx, result)
        })
        .collect()
}

/// Set of implicit conditions inferred from an expression.
#[derive(Debug, Clone, Default)]
pub struct ImplicitDomain {
    conditions: HashSet<ImplicitCondition>,
}

impl ImplicitDomain {
    /// Create an empty implicit domain.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if domain is empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Check if an expression has an implicit NonNegative constraint.
    pub fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.conditions
            .contains(&ImplicitCondition::NonNegative(expr))
    }

    /// Check if an expression has an implicit Positive constraint.
    pub fn contains_positive(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::Positive(expr))
    }

    /// Check if an expression has an implicit NonZero constraint.
    pub fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::NonZero(expr))
    }

    /// Add a NonNegative condition.
    pub fn add_nonnegative(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonNegative(expr));
    }

    /// Add a Positive condition.
    pub fn add_positive(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::Positive(expr));
    }

    /// Add a NonZero condition.
    pub fn add_nonzero(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonZero(expr));
    }

    /// Get all conditions (for iteration/comparison).
    pub fn conditions(&self) -> &HashSet<ImplicitCondition> {
        &self.conditions
    }

    /// Get mutable access to conditions.
    pub fn conditions_mut(&mut self) -> &mut HashSet<ImplicitCondition> {
        &mut self.conditions
    }

    /// Check if this domain is a superset of another (contains all its conditions).
    pub fn contains_all(&self, other: &ImplicitDomain) -> bool {
        other.conditions.is_subset(&self.conditions)
    }

    /// Get conditions that are in self but not in other (dropped conditions).
    pub fn dropped_from<'a>(&'a self, other: &'a ImplicitDomain) -> Vec<&'a ImplicitCondition> {
        self.conditions.difference(&other.conditions).collect()
    }

    /// Merge conditions from another domain into this one.
    pub fn extend(&mut self, other: &ImplicitDomain) {
        for cond in &other.conditions {
            self.conditions.insert(cond.clone());
        }
    }

    /// Convert to a ConditionSet for solver use.
    pub fn to_condition_set(&self) -> cas_ast::ConditionSet {
        let predicates: Vec<cas_ast::ConditionPredicate> =
            self.conditions.iter().map(|c| c.into()).collect();
        cas_ast::ConditionSet::from_predicates(predicates)
    }
}

impl crate::domain_env::RequiredDomainSet for ImplicitDomain {
    fn contains_positive(&self, expr: ExprId) -> bool {
        self.contains_positive(expr)
    }

    fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.contains_nonnegative(expr)
    }

    fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.contains_nonzero(expr)
    }

    fn to_condition_set(&self) -> cas_ast::ConditionSet {
        self.to_condition_set()
    }
}

impl From<&ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: &ImplicitCondition) -> Self {
        match cond {
            ImplicitCondition::NonNegative(e) => cas_ast::ConditionPredicate::NonNegative(*e),
            ImplicitCondition::Positive(e) => cas_ast::ConditionPredicate::Positive(*e),
            ImplicitCondition::NonZero(e) => cas_ast::ConditionPredicate::NonZero(*e),
        }
    }
}

impl From<ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: ImplicitCondition) -> Self {
        (&cond).into()
    }
}

impl TryFrom<&cas_ast::ConditionPredicate> for ImplicitCondition {
    type Error = ();

    fn try_from(pred: &cas_ast::ConditionPredicate) -> Result<Self, Self::Error> {
        match pred {
            cas_ast::ConditionPredicate::NonNegative(e) => Ok(ImplicitCondition::NonNegative(*e)),
            cas_ast::ConditionPredicate::Positive(e) => Ok(ImplicitCondition::Positive(*e)),
            cas_ast::ConditionPredicate::NonZero(e) => Ok(ImplicitCondition::NonZero(*e)),
            _ => Err(()),
        }
    }
}
