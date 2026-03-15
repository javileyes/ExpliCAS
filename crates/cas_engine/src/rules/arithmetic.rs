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

fn canonicalize_nested_integer_powers(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let rebuilt = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Div(lhs, rhs))
        }
        Expr::Pow(base, exp) => {
            let base = canonicalize_nested_integer_powers(ctx, base);
            let exp = canonicalize_nested_integer_powers(ctx, exp);
            let pow = ctx.add(Expr::Pow(base, exp));
            cas_math::rational_canonicalization_support::try_rewrite_nested_pow_canonical_expr(
                ctx, pow,
            )
            .map(|rewrite| rewrite.rewritten)
            .unwrap_or(pow)
        }
        Expr::Neg(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(name, args) => {
            let args = args
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Function(name, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Hold(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Hold(inner))
        }
        Expr::SessionRef(id) => ctx.add(Expr::SessionRef(id)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => expr,
    };

    if rebuilt == expr {
        expr
    } else {
        rebuilt
    }
}

fn collect_add_terms(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    out: &mut Vec<cas_ast::ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_add_terms(ctx, *lhs, out);
            collect_add_terms(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn exprs_equal_up_to_add_term_order(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_add_terms(ctx, lhs, &mut lhs_terms);
    collect_add_terms(ctx, rhs, &mut rhs_terms);
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs_terms: Vec<_> = lhs_terms
        .into_iter()
        .map(|term| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                }
            )
        })
        .collect();
    let mut rhs_terms: Vec<_> = rhs_terms
        .into_iter()
        .map(|term| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                }
            )
        })
        .collect();

    lhs_terms.sort();
    rhs_terms.sort();
    lhs_terms == rhs_terms
}

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
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        let pattern = match_mul_zero_pattern(ctx, expr)?;
        let other = pattern.other;
        let has_risk = crate::collect::has_undefined_risk(ctx, other);
        let allowed = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            has_risk,
        );

        if !allowed {
            return None; // Strict mode: don't simplify if has risk
        }

        // Build assumption events if has risk and allowed
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if has_risk {
            smallvec::smallvec![crate::AssumptionEvent::defined(ctx, other)]
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
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Proof;
        use crate::Predicate;

        let pattern = match_div_zero_numerator_pattern(ctx, expr)?;
        let den = pattern.denominator;

        // Special case: 0/0 → undefined (all modes)
        if pattern.denominator_is_literal_zero {
            let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
            return Some(Rewrite::new(undef).desc("0/0 is undefined"));
        }

        // Use unified oracle for NonZero condition (Definability class)
        let decision = crate::oracle_allows_with_hint(
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
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if decision.assumption.is_some() && den_proof != Proof::Proven {
            smallvec::smallvec![crate::AssumptionEvent::nonzero(ctx, den)]
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
            cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                crate::collect::has_undefined_risk(ctx, rewrite.inner),
            );
        if !allow {
            return None;
        }

        Some(Rewrite::new(rewrite.rewritten).desc("a - a = 0"))
    }
);

define_rule!(
    SubtractExpandedSumDiffCubesQuotientRule,
    "Subtract Expanded Sum/Difference of Cubes Quotient",
    priority: 500,
    |ctx, expr, parent_ctx| {
        use crate::{ImplicitCondition, Predicate};

        let (lhs, rhs) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
                (_, Expr::Neg(inner)) => (*lhs, *inner),
                (Expr::Neg(inner), _) => (*rhs, *inner),
                _ => return None,
            },
            _ => return None,
        };

        let (num, den) = match ctx.get(lhs) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Subtract Expanded Sum/Difference of Cubes Quotient",
        );
        if !decision.allow {
            return None;
        }

        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            ctx, num, den, false,
        )?;

        let cancelled = canonicalize_nested_integer_powers(ctx, plan.cancelled_result);
        let rhs = canonicalize_nested_integer_powers(ctx, rhs);
        if !(cas_math::expr_domain::exprs_equivalent(ctx, cancelled, rhs)
            || exprs_equal_up_to_add_term_order(ctx, cancelled, rhs))
        {
            return None;
        }

        Some(
            Rewrite::new(ctx.num(0))
                .desc("((a^3 ± b^3)/(a ± b)) - expanded quotient = 0")
                .requires(ImplicitCondition::NonZero(den)),
        )
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
        cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
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
    Some(Rewrite::new(rewrite.rewritten).desc("a + (-a) = 0"))
});

#[cfg(test)]
mod tests {
    use super::{
        canonicalize_nested_integer_powers, exprs_equal_up_to_add_term_order, SubSelfToZeroRule,
        SubtractExpandedSumDiffCubesQuotientRule,
    };
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use crate::DomainMode;
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn subtraction_self_cancel_rule_matches_abs_sub_mirror_in_generic() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = SubSelfToZeroRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn subtract_expanded_sum_diff_cubes_quotient_rule_matches_trig_square_cube_residual() {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - ((sin(u)^2)^2 + (sin(u)^2) + 1)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let (lhs, rhs) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
                (_, Expr::Neg(inner)) => (*lhs, *inner),
                (Expr::Neg(inner), _) => (*rhs, *inner),
                _ => panic!("unexpected add form"),
            },
            other => panic!("unexpected root: {other:?}"),
        };
        let (num, den) = match ctx.get(lhs) {
            Expr::Div(num, den) => (*num, *den),
            other => panic!("unexpected lhs: {other:?}"),
        };
        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            &mut ctx, num, den, false,
        )
        .unwrap_or_else(|| panic!("plan"));
        let cancelled = canonicalize_nested_integer_powers(&mut ctx, plan.cancelled_result);
        let rhs = canonicalize_nested_integer_powers(&mut ctx, rhs);
        assert!(
            cas_math::expr_domain::exprs_equivalent(&ctx, cancelled, rhs)
                || exprs_equal_up_to_add_term_order(&ctx, cancelled, rhs),
            "cancelled={} rhs={}",
            DisplayExpr {
                context: &ctx,
                id: cancelled
            },
            DisplayExpr {
                context: &ctx,
                id: rhs
            }
        );

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = SubtractExpandedSumDiffCubesQuotientRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }
}

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
    simplifier.add_rule(Box::new(SubtractExpandedSumDiffCubesQuotientRule));

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(NormalizeMulNegRule)); // Lift Neg out of Mul for canonical form
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(SimplifyNumericExponentsRule));
    simplifier.add_rule(Box::new(AddInverseRule));
}
