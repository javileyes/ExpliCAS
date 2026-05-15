//! Implicit-domain condition model shared by engine/solver layers.
//!
//! This module owns:
//! - condition vocabulary (`ImplicitCondition`)
//! - display policy (`RequiresDisplayLevel`)
//! - inferred condition set container (`ImplicitDomain`)

use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::extract_abs_argument_view;
use num_rational::BigRational;
use std::collections::HashSet;

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x ≥ c, for real-domain lower bounds such as acosh(x), where c = 1.
    LowerBound(ExprId, BigRational),
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
            Self::NonNegative(e) => {
                if let Some(bound) = display_unit_interval_nonnegative(ctx, *e) {
                    bound
                } else {
                    format!(
                        "{} ≥ 0",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                }
            }
            Self::LowerBound(e, lower) => format!(
                "{} ≥ {}",
                DisplayExpr {
                    context: ctx,
                    id: *e
                },
                lower
            ),
            Self::Positive(e) => {
                if let Some(arg) = extract_abs_argument_view(ctx, *e) {
                    format!(
                        "{} ≠ 0",
                        DisplayExpr {
                            context: ctx,
                            id: arg
                        }
                    )
                } else {
                    format!(
                        "{} > 0",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                }
            }
            Self::NonZero(e) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: ctx,
                    id: extract_abs_argument_view(ctx, *e).unwrap_or(*e)
                }
            ),
        }
    }

    /// Check if this condition is trivial (always true or on a constant expression).
    pub fn is_trivial(&self, ctx: &Context) -> bool {
        let expr = match self {
            Self::NonNegative(e)
            | Self::LowerBound(e, _)
            | Self::Positive(e)
            | Self::NonZero(e) => *e,
        };

        if matches!(self, Self::NonZero(_))
            && cas_math::prove_nonzero::prove_nonzero_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                |_ctx, _expr| cas_math::tri_proof::TriProof::Unknown,
                |_ctx, _expr| None,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::NonZero(_))
            && cas_math::prove_sign::prove_positive_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::Positive(_))
            && cas_math::prove_sign::prove_positive_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if matches!(self, Self::NonNegative(_))
            && cas_math::prove_sign::prove_nonnegative_depth_with(
                ctx,
                expr,
                crate::predicate_proofs::DEFAULT_PROOF_DEPTH,
                true,
                |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
            )
            .is_proven()
        {
            return true;
        }

        if let Self::LowerBound(_, lower) = self {
            if let Expr::Number(n) = ctx.get(expr) {
                return n >= lower;
            }
        }

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
            Self::LowerBound(_, _) => false,
            Self::Positive(e) => witness_survives(ctx, *e, output, WitnessKind::Log),
            Self::NonZero(e) => witness_survives(ctx, *e, output, WitnessKind::Division),
        }
    }
}

fn display_unit_interval_nonnegative(ctx: &Context, expr: ExprId) -> Option<String> {
    use cas_formatter::DisplayExpr;

    let base = unit_interval_base(ctx, expr)?;
    if !cas_math::expr_predicates::contains_variable(ctx, base) {
        return None;
    }

    if let Some(denominator) = reciprocal_denominator(ctx, base) {
        return Some(format!(
            "{} ≤ -1 or {} ≥ 1",
            DisplayExpr {
                context: ctx,
                id: denominator
            },
            DisplayExpr {
                context: ctx,
                id: denominator
            }
        ));
    }

    Some(format!(
        "-1 ≤ {} ≤ 1",
        DisplayExpr {
            context: ctx,
            id: base
        }
    ))
}

fn unit_interval_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => squared_base(ctx, *right),
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_squared_base(ctx, *right)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_squared_base(ctx, *left)
        }
        _ => None,
    }
}

fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *exponent, 2) {
        Some(*base)
    } else {
        None
    }
}

fn negated_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => squared_base(ctx, *inner),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => squared_base(ctx, *right),
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => squared_base(ctx, *left),
        _ => None,
    }
}

fn reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) if is_integer_literal(ctx, *numerator, 1) => {
            Some(*denominator)
        }
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, -1) => Some(*base),
        _ => None,
    }
}

fn is_integer_literal(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n == &BigRational::from_integer(value.into()))
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

    /// Add a lower-bound condition.
    pub fn add_lower_bound(&mut self, expr: ExprId, lower: BigRational) {
        self.conditions
            .insert(ImplicitCondition::LowerBound(expr, lower));
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
            ImplicitCondition::LowerBound(e, lower) => cas_ast::ConditionPredicate::LowerBound {
                expr: *e,
                lower: lower.clone(),
            },
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
            cas_ast::ConditionPredicate::LowerBound { expr, lower } => {
                Ok(ImplicitCondition::LowerBound(*expr, lower.clone()))
            }
            cas_ast::ConditionPredicate::Positive(e) => Ok(ImplicitCondition::Positive(*e)),
            cas_ast::ConditionPredicate::NonZero(e) => Ok(ImplicitCondition::NonZero(*e)),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImplicitCondition;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn nonzero_exponential_condition_is_trivial() {
        let mut ctx = Context::new();
        let exp_x = parse("exp(x)", &mut ctx).expect("parse exp(x)");

        assert!(ImplicitCondition::NonZero(exp_x).is_trivial(&ctx));
    }

    #[test]
    fn nonzero_strictly_positive_quadratic_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("x^2+1", &mut ctx).expect("parse x^2+1");

        assert!(ImplicitCondition::NonZero(expr).is_trivial(&ctx));
    }

    #[test]
    fn positive_strictly_positive_quadratic_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("x^2+1", &mut ctx).expect("parse x^2+1");

        assert!(ImplicitCondition::Positive(expr).is_trivial(&ctx));
    }

    #[test]
    fn nonnegative_shifted_square_plus_constant_condition_is_trivial() {
        let mut ctx = Context::new();
        let expr = parse("(2*x+1)^2+3", &mut ctx).expect("parse shifted square");

        assert!(ImplicitCondition::NonNegative(expr).is_trivial(&ctx));
    }

    #[test]
    fn positive_abs_displays_as_nonzero_inner_expression() {
        let mut ctx = Context::new();
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        assert_eq!(ImplicitCondition::Positive(abs_x).display(&ctx), "x ≠ 0");
    }

    #[test]
    fn nonzero_abs_displays_as_nonzero_inner_expression() {
        let mut ctx = Context::new();
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        assert_eq!(ImplicitCondition::NonZero(abs_x).display(&ctx), "x ≠ 0");
    }
}
