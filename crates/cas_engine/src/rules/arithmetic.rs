use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::arithmetic_cancel_support::{
    try_rewrite_add_inverse_zero_expr, try_rewrite_sub_self_zero_expr,
};
use cas_math::arithmetic_rule_support::{
    try_rewrite_add_zero_expr, try_rewrite_combine_constants_expr, try_rewrite_mul_one_expr,
    try_rewrite_normalize_mul_neg_expr, try_rewrite_simplify_numeric_exponents_expr,
};
use cas_math::arithmetic_zero_support::{match_div_zero_numerator_pattern, match_mul_zero_pattern};

define_rule!(
    AddZeroRule,
    "Identity Property of Addition",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_add_zero_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_mul_one_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// MulZeroRule: 0*e → 0
// Domain Mode Policy: 0*e → 0 changes the domain of definition if e can be undefined.
// Uses ConditionClass taxonomy:
// - Strict: only apply if other factor has no undefined risk
// - Generic: apply with Defined(e) assumption (Definability class)
// - Assume: apply with Defined(e) assumption
define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        let pattern = match_mul_zero_pattern(ctx, expr)?;
        let other = pattern.other;
        let has_risk = crate::collect::has_undefined_risk(ctx, other);
        let allowed = cas_math::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            has_risk,
        );

        if !allowed {
            return None; // Strict mode: don't simplify if has risk
        }

        // Build assumption events if has risk and allowed
        let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if has_risk {
            smallvec::smallvec![crate::assumptions::AssumptionEvent::defined(ctx, other)]
        } else {
            smallvec::SmallVec::new()
        };

        let description = if pattern.zero_on_lhs {
            "0 * x = 0".to_string()
        } else {
            "x * 0 = 0".to_string()
        };


        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc(description).assume_all(assumption_events))
    }
);

// DivZeroRule: 0/d → 0
// Domain Mode Policy: 0/d → 0 changes the domain of definition if d can be 0.
// Uses unified DomainOracle via oracle_allows_with_hint:
// - Strict: only apply if prove_nonzero(d) == Proven
// - Generic: apply with NonZero(d) assumption (Definability class)
// - Assume: apply with NonZero(d) assumption
define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain::Proof;
        use crate::domain_facts::Predicate;

        let pattern = match_div_zero_numerator_pattern(ctx, expr)?;
        let den = pattern.denominator;

        // Special case: 0/0 → undefined (all modes)
        if pattern.denominator_is_literal_zero {
            let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
            return Some(Rewrite::new(undef).desc("0/0 is undefined"));
        }

        // Use unified oracle for NonZero condition (Definability class)
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Zero Property of Division",
        );

        if !decision.allow {
            return None; // Strict mode: don't simplify if not proven
        }

        // Build assumption events if needed
        let den_proof = crate::helpers::prove_nonzero(ctx, den);
        let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if decision.assumption.is_some() && den_proof != Proof::Proven {
            smallvec::smallvec![crate::assumptions::AssumptionEvent::nonzero(ctx, den)]
        } else {
            smallvec::SmallVec::new()
        };

        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc("0 / d = 0").assume_all(assumption_events))
    }
);

define_rule!(
    CombineConstantsRule,
    "Combine Constants",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_combine_constants_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_add_zero() {
        let mut ctx = Context::new();
        let rule = AddZeroRule;
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Add(x, zero));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x"
        );
    }

    #[test]
    fn test_mul_one() {
        let mut ctx = Context::new();
        let rule = MulOneRule;
        let one = ctx.num(1);
        let y = ctx.var("y");
        let expr = ctx.add(Expr::Mul(one, y));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "y"
        );
    }

    #[test]
    fn test_combine_constants() {
        let mut ctx = Context::new();
        let rule = CombineConstantsRule;
        let two = ctx.num(2);
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Add(two, three));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "5"
        );
    }
}

// =============================================================================
// SubSelfToZeroRule: a - a = 0 (Short-circuit)
// =============================================================================
//
// V2.14.45: This rule MUST fire before expansion rules like TanToSinCosRule.
// Without this, tan(3x) - tan(3x) would expand both tans and fail to cancel.
// Uses priority 500 to ensure it runs first.
//
// Domain Policy: Same as AddInverseRule - check for undefined subexpressions.
// Uses compare_expr for structural equality (handles tan(3x) == tan(3·x)).
// =============================================================================
define_rule!(
    SubSelfToZeroRule,
    "Subtraction Self-Cancel",
    priority: 500, // High priority: before any expansion rules
    |ctx, expr, parent_ctx| {
        let rewrite = try_rewrite_sub_self_zero_expr(ctx, expr)?;
        let allow =
            cas_math::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                crate::collect::has_undefined_risk(ctx, rewrite.inner),
            );
        if !allow {
            return None;
        }

        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// AddInverseRule: a + (-a) = 0
// Domain Mode Policy: Like other cancellation rules, we must respect domain_mode
// because if `a` can be undefined (e.g., x/(x+1) when x=-1), then a + (-a)
// is undefined, not 0.
// - Strict: only if `a` contains no potentially-undefined subexpressions (no variable denominator)
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
//
// V2.12.13: REMOVED redundant "is defined" assumption event.
// The individual Div operations already emit NonZero(denominator) as Requires.
// Showing "a is defined" here is redundant and confusing.
define_rule!(AddInverseRule, "Add Inverse", |ctx, expr, parent_ctx| {
    let rewrite = try_rewrite_add_inverse_zero_expr(ctx, expr)?;
    let allow =
        cas_math::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            crate::collect::has_undefined_risk(ctx, rewrite.inner),
        );
    if !allow {
        return None;
    }

    // V2.12.13: No assumption events - the division conditions are already
    // tracked as Requires from the original Div operations.
    // Adding "a is defined" here is redundant and clutters the output.
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// Simplify sums of fractions in exponents: x^(1/2 + 1/3) → x^(5/6)
// This makes the fraction sum visible as a step in the timeline.
define_rule!(
    SimplifyNumericExponentsRule,
    "Sum Exponents",
    |ctx, expr| {
        let rewrite = try_rewrite_simplify_numeric_exponents_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// =============================================================================
// NormalizeMulNegRule: Lift Neg out of Mul for canonical form
// =============================================================================
//
// Canonical form: Neg should be at the TOP of Mul, not buried inside.
// This unlocks cancellations in Add like: a*(-b) + (-a)*b → Neg(a*b) + Neg(a*b) → -2*a*b
//
// Rewrites:
// - Mul(Neg(a), b) → Neg(Mul(a, b))
// - Mul(a, Neg(b)) → Neg(Mul(a, b))
// - Mul(Neg(a), Neg(b)) → Mul(a, b)  (double negation cancels)
//
// This is idempotent and always reduces complexity.
// =============================================================================
define_rule!(
    NormalizeMulNegRule,
    "Normalize Negation in Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_normalize_mul_neg_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    // High-priority short-circuit rules first
    simplifier.add_rule(Box::new(SubSelfToZeroRule)); // priority 500: before expansion

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(NormalizeMulNegRule)); // Lift Neg out of Mul for canonical form
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(SimplifyNumericExponentsRule));
    simplifier.add_rule(Box::new(AddInverseRule));
}

#[cfg(test)]
mod importance_tests {
    use super::*;
    use crate::rule::SimpleRule;
    use crate::step::ImportanceLevel;

    #[test]
    fn test_mul_one_rule_importance() {
        let rule = MulOneRule;
        assert_eq!(
            rule.importance(),
            ImportanceLevel::Low,
            "MulOneRule should have Low importance (hidden in normal mode)"
        );
    }

    #[test]
    fn test_add_zero_rule_importance() {
        let rule = AddZeroRule;
        assert_eq!(
            rule.importance(),
            ImportanceLevel::Low,
            "AddZeroRule should have Low importance (hidden in normal mode)"
        );
    }
}
