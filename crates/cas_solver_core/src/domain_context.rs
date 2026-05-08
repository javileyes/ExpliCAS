//! Domain condition implication context shared by engine/solver layers.

use crate::domain_condition::ImplicitCondition;
use crate::domain_normalization::conditions_equivalent;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, is_nonnegative_shift_from_source, is_odd_power_of, is_positive_multiple_of,
    is_positive_power_of_base, lower_bound_and_nonzero_implies_reciprocal_positive,
    lower_bound_implies_nonnegative, lower_bound_implies_positive,
    nonzero_implies_reciprocal_even_power_positive,
    positive_abs_quotient_implies_negated_arg_positive,
};

/// Context for domain condition tracking during step processing.
#[derive(Debug, Clone, Default)]
pub struct DomainContext {
    /// Conditions inferred from the original input expression.
    pub global_requires: Vec<ImplicitCondition>,
    /// Conditions introduced by previous steps (accumulated).
    pub introduced_requires: Vec<ImplicitCondition>,
}

impl DomainContext {
    /// Create a new DomainContext with global requires from the input expression.
    pub fn new(global_requires: Vec<ImplicitCondition>) -> Self {
        Self {
            global_requires,
            introduced_requires: Vec::new(),
        }
    }

    /// Check if a condition is implied by the known requires (global ∪ introduced).
    ///
    /// Implication rules:
    /// - Exact polynomial equivalence
    /// - x > 0 is implied by x^(odd positive) > 0
    /// - x ≠ 0 is implied by x > 0
    pub fn is_condition_implied(&self, ctx: &Context, cond: &ImplicitCondition) -> bool {
        let all_known: Vec<_> = self
            .global_requires
            .iter()
            .chain(self.introduced_requires.iter())
            .collect();

        for known in all_known {
            if conditions_equivalent(ctx, cond, known) {
                return true;
            }

            match (cond, known) {
                (ImplicitCondition::NonZero(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                (ImplicitCondition::Positive(target), ImplicitCondition::NonZero(source)) => {
                    if nonzero_implies_reciprocal_even_power_positive(ctx, *target, *source) {
                        return true;
                    }
                }
                (ImplicitCondition::Positive(target), ImplicitCondition::Positive(source)) => {
                    if is_positive_multiple_of(ctx, *source, *target) {
                        return true;
                    }
                    if is_odd_power_of(ctx, *source, *target) {
                        return true;
                    }
                    if is_positive_power_of_base(ctx, *target, *source) {
                        return true;
                    }
                    if is_nonnegative_shift_from_source(ctx, *target, *source) {
                        return true;
                    }
                    if positive_abs_quotient_implies_negated_arg_positive(ctx, *source, *target) {
                        return true;
                    }
                }
                (
                    ImplicitCondition::Positive(target),
                    ImplicitCondition::LowerBound(source, lower),
                ) => {
                    if lower_bound_implies_positive(ctx, *target, *source, lower) {
                        return true;
                    }
                    let known_nonzeros = self.known_nonzero_exprs(ctx);
                    if lower_bound_and_nonzero_implies_reciprocal_positive(
                        ctx,
                        *target,
                        *source,
                        lower,
                        &known_nonzeros,
                    ) {
                        return true;
                    }
                }
                (ImplicitCondition::NonNegative(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source)
                        || is_positive_multiple_of(ctx, *source, *target)
                        || is_nonnegative_shift_from_source(ctx, *target, *source)
                        || positive_abs_quotient_implies_negated_arg_positive(ctx, *source, *target)
                    {
                        return true;
                    }
                }
                (ImplicitCondition::NonNegative(target), ImplicitCondition::NonZero(source)) => {
                    if nonzero_implies_reciprocal_even_power_positive(ctx, *target, *source) {
                        return true;
                    }
                }
                (
                    ImplicitCondition::NonNegative(target),
                    ImplicitCondition::LowerBound(source, lower),
                ) => {
                    if lower_bound_implies_nonnegative(ctx, *target, *source, lower) {
                        return true;
                    }
                    let known_nonzeros = self.known_nonzero_exprs(ctx);
                    if lower_bound_and_nonzero_implies_reciprocal_positive(
                        ctx,
                        *target,
                        *source,
                        lower,
                        &known_nonzeros,
                    ) {
                        return true;
                    }
                }
                (
                    ImplicitCondition::NonNegative(target),
                    ImplicitCondition::NonNegative(source),
                ) => {
                    if is_positive_multiple_of(ctx, *source, *target) {
                        return true;
                    }
                    if is_nonnegative_shift_from_source(ctx, *target, *source) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn known_nonzero_exprs(&self, ctx: &Context) -> Vec<ExprId> {
        let mut out = Vec::new();
        for cond in self
            .global_requires
            .iter()
            .chain(self.introduced_requires.iter())
        {
            if let ImplicitCondition::NonZero(expr) = cond {
                collect_nonzero_witnesses(ctx, *expr, &mut out);
            }
        }
        out
    }

    /// Add a newly introduced condition (from a step that introduces constraints).
    pub fn add_introduced(&mut self, cond: ImplicitCondition) {
        self.introduced_requires.push(cond);
    }
}

fn collect_nonzero_witnesses(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    out.push(expr);

    match ctx.get(expr) {
        Expr::Neg(inner) => collect_nonzero_witnesses(ctx, *inner, out),
        Expr::Mul(left, right) if ctx.is_mul_commutative(expr) => {
            collect_nonzero_witnesses(ctx, *left, out);
            collect_nonzero_witnesses(ctx, *right, out);
        }
        Expr::Pow(base, exp) => {
            let Expr::Number(n) = ctx.get(*exp) else {
                return;
            };
            if n.is_integer() && n.to_integer() > 0.into() {
                collect_nonzero_witnesses(ctx, *base, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn shifted_nonnegative_lower_bound_implies_weaker_nonnegative_target() {
        let mut ctx = Context::new();
        let source = parse("x - 1", &mut ctx).expect("parse source");
        let target = parse("x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::NonNegative(source)]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(target)));
    }

    #[test]
    fn shifted_positive_lower_bound_implies_weaker_positive_target() {
        let mut ctx = Context::new();
        let source = parse("2*x - 1", &mut ctx).expect("parse source");
        let target = parse("2*x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::Positive(source)]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(target)));
    }

    #[test]
    fn explicit_lower_bound_implies_abs_safe_nonnegative_target() {
        let mut ctx = Context::new();
        let source = parse("2*x + 1", &mut ctx).expect("parse source");
        let target = parse("x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::LowerBound(
            source,
            num_rational::BigRational::from_integer(1.into()),
        )]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(target)));
    }

    #[test]
    fn explicit_negative_orientation_lower_bound_implies_nonpositive_target() {
        let mut ctx = Context::new();
        let source = parse("1 - 2*x", &mut ctx).expect("parse source");
        let neg_target = parse("-x", &mut ctx).expect("parse negated target");
        let dc = DomainContext::new(vec![ImplicitCondition::LowerBound(
            source,
            num_rational::BigRational::from_integer(1.into()),
        )]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(neg_target)));
    }

    #[test]
    fn explicit_nonunit_lower_bound_implies_shifted_targets() {
        let mut ctx = Context::new();
        let source = parse("x", &mut ctx).expect("parse source");
        let weak_target = parse("x - 2", &mut ctx).expect("parse weak target");
        let strict_target = parse("x - 1", &mut ctx).expect("parse strict target");
        let too_strong_target = parse("x - 3", &mut ctx).expect("parse too strong target");
        let dc = DomainContext::new(vec![ImplicitCondition::LowerBound(
            source,
            num_rational::BigRational::from_integer(2.into()),
        )]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(weak_target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(strict_target)));
        assert!(!dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(too_strong_target)));
    }

    #[test]
    fn lower_bound_plus_nonzero_implies_reciprocal_sign() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let denominator = parse("x - 1", &mut ctx).expect("parse denominator");
        let reciprocal_target = parse("-1/(x - 1)", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(denominator),
        ]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(reciprocal_target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(reciprocal_target)));
    }

    #[test]
    fn lower_bound_without_nonzero_does_not_imply_reciprocal_sign() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let reciprocal_target = parse("-1/(x - 1)", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::LowerBound(
            source,
            num_rational::BigRational::from_integer(1.into()),
        )]);

        assert!(!dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(reciprocal_target)));
    }

    #[test]
    fn nonzero_implies_even_power_reciprocal_positive() {
        let mut ctx = Context::new();
        let denominator = parse("x - 1", &mut ctx).expect("parse denominator");
        let reciprocal_target = parse("1/(x - 1)^2", &mut ctx).expect("parse target");
        let negative_control = parse("-1/(x - 1)^2", &mut ctx).expect("parse negative control");
        let dc = DomainContext::new(vec![ImplicitCondition::NonZero(denominator)]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(reciprocal_target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(reciprocal_target)));
        assert!(!dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(negative_control)));
    }

    #[test]
    fn lower_bound_plus_nonzero_implies_odd_power_reciprocal_sign() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let denominator = parse("x - 1", &mut ctx).expect("parse denominator");
        let reciprocal_target = parse("-1/(x - 1)^3", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(denominator),
        ]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(reciprocal_target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(reciprocal_target)));
    }

    #[test]
    fn lower_bound_plus_nonzeros_implies_product_reciprocal_sign() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let first = parse("x - 1", &mut ctx).expect("parse first");
        let second = parse("x - 2", &mut ctx).expect("parse second");
        let product_denominator = parse("(x - 1)*(x - 2)", &mut ctx).expect("parse denominator");
        let positive_product = parse("1/((x - 1)*(x - 2))", &mut ctx).expect("parse positive");
        let normalized_positive_product =
            parse("1/((x - 2)*(x - 1))", &mut ctx).expect("parse normalized positive");
        let negative_product = parse("-1/((x - 1)*(2 - x))", &mut ctx).expect("parse negative");
        let dc = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(first),
            ImplicitCondition::NonZero(second),
        ]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(positive_product)));
        assert!(dc.is_condition_implied(
            &ctx,
            &ImplicitCondition::Positive(normalized_positive_product)
        ));
        assert!(dc.is_condition_implied(
            &ctx,
            &ImplicitCondition::NonNegative(normalized_positive_product)
        ));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(negative_product)));

        let dc_from_product_nonzero = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(product_denominator),
        ]);
        assert!(dc_from_product_nonzero
            .is_condition_implied(&ctx, &ImplicitCondition::Positive(positive_product)));
    }

    #[test]
    fn lower_bound_plus_nonzeros_implies_mixed_power_product_reciprocal_sign() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let first = parse("x - 1", &mut ctx).expect("parse first");
        let second = parse("x - 2", &mut ctx).expect("parse second");
        let negative_product =
            parse("-1/((x - 1)^2*(x - 2))", &mut ctx).expect("parse negative product");
        let positive_product =
            parse("1/((x - 1)^2*(2 - x))", &mut ctx).expect("parse positive product");
        let dc = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(first),
            ImplicitCondition::NonZero(second),
        ]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(negative_product)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(negative_product)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(positive_product)));
    }

    #[test]
    fn product_reciprocal_requires_all_factor_nonzero_witnesses() {
        let mut ctx = Context::new();
        let source = parse("3 - 2*x", &mut ctx).expect("parse source");
        let first = parse("x - 1", &mut ctx).expect("parse first");
        let positive_product = parse("1/((x - 1)*(x - 2))", &mut ctx).expect("parse product");
        let dc = DomainContext::new(vec![
            ImplicitCondition::LowerBound(
                source,
                num_rational::BigRational::from_integer(1.into()),
            ),
            ImplicitCondition::NonZero(first),
        ]);

        assert!(!dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(positive_product)));
    }

    #[test]
    fn positive_negated_abs_quotient_implies_negative_argument_domain() {
        let mut ctx = Context::new();
        let source = parse("-abs(x)/x", &mut ctx).expect("parse source");
        let target = parse("-x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::Positive(source)]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(target)));
    }

    #[test]
    fn positive_scaled_fact_implies_unscaled_positive_and_nonnegative_targets() {
        let mut ctx = Context::new();
        let source = parse("-x/2", &mut ctx).expect("parse source");
        let target = parse("-x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::Positive(source)]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(target)));
        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(target)));
    }

    #[test]
    fn explicit_lower_bound_implies_strict_positive_when_margin_is_positive() {
        let mut ctx = Context::new();
        let source = parse("x", &mut ctx).expect("parse source");
        let target = parse("x", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::LowerBound(
            source,
            num_rational::BigRational::from_integer(1.into()),
        )]);

        assert!(dc.is_condition_implied(&ctx, &ImplicitCondition::Positive(target)));
    }

    #[test]
    fn earlier_lower_bound_does_not_imply_later_nonnegative_target() {
        let mut ctx = Context::new();
        let source = parse("x", &mut ctx).expect("parse source");
        let target = parse("x - 1", &mut ctx).expect("parse target");
        let dc = DomainContext::new(vec![ImplicitCondition::NonNegative(source)]);

        assert!(!dc.is_condition_implied(&ctx, &ImplicitCondition::NonNegative(target)));
    }
}
