use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::number_theory_support::{
    dispatch_number_theory_call, try_rewrite_consecutive_factorial_ratio_expr,
    NumberTheoryDispatch, NumberTheorySimpleRewrite,
};
use cas_math::poly_gcd_dispatch::{compute_poly_gcd_unified_with, pre_evaluate_for_gcd_with};
use cas_math::poly_gcd_mode::GcdGoal;

define_rule!(
    ConsecutiveFactorialRatioRule,
    "Consecutive Factorial Ratio",
    Some(crate::target_kind::TargetKindSet::DIV),
    solve_safety: crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
    |ctx, expr, _parent_ctx| {
        let rewrite = try_rewrite_consecutive_factorial_ratio_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Cancel consecutive factorials")
                .local(expr, rewrite.rewritten)
                .requires(crate::ImplicitCondition::NonNegative(
                    rewrite.factorial_arg_requires_nonnegative,
                )),
        )
    }
);

define_rule!(NumberTheoryRule, "Number Theory Operations", |ctx, expr| {
    let (result, desc) = match dispatch_number_theory_call(ctx, expr)? {
        NumberTheoryDispatch::Simple(simple) => {
            (simple.result(), render_number_theory_desc(ctx, simple))
        }
        NumberTheoryDispatch::PolyGcd { lhs, rhs, mode } => {
            let (result, desc) = compute_poly_gcd_unified_with(
                ctx,
                lhs,
                rhs,
                GcdGoal::UserPolyGcd,
                mode,
                None,
                None,
                |core_ctx, id| {
                    pre_evaluate_for_gcd_with(core_ctx, id, crate::expand::eval_expand_off)
                },
                cas_formatter::render_expr,
            );
            (cas_ast::hold::wrap_hold(ctx, result), desc)
        }
    };

    Some(Rewrite::new(result).desc(desc))
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ConsecutiveFactorialRatioRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));
}

fn render_number_theory_desc(ctx: &cas_ast::Context, call: NumberTheorySimpleRewrite) -> String {
    match call {
        NumberTheorySimpleRewrite::Unary { name, arg, .. } => {
            format!("{}({})", name, cas_formatter::render_expr(ctx, arg))
        }
        NumberTheorySimpleRewrite::Binary { name, lhs, rhs, .. } => {
            format!(
                "{}({}, {})",
                name,
                cas_formatter::render_expr(ctx, lhs),
                cas_formatter::render_expr(ctx, rhs)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn consecutive_factorial_ratio_rule_simplifies_and_requires_factorial_domain_argument() {
        let mut ctx = Context::new();
        let expr = parse("(n + 1)! / n!", &mut ctx).expect("parse");
        let expected = parse("n + 1", &mut ctx).expect("expected");
        let expected_arg = parse("n", &mut ctx).expect("arg");

        let rewrite = ConsecutiveFactorialRatioRule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("rewrite");

        assert_eq!(
            compare_expr(&ctx, rewrite.new_expr, expected),
            std::cmp::Ordering::Equal
        );
        assert!(rewrite.required_conditions.iter().any(|cond| {
            matches!(
                cond,
                crate::ImplicitCondition::NonNegative(arg)
                    if compare_expr(&ctx, *arg, expected_arg) == std::cmp::Ordering::Equal
            )
        }));
    }

    #[test]
    fn consecutive_factorial_ratio_rule_rejects_nonconsecutive_ratio() {
        let mut ctx = Context::new();
        let expr = parse("(n + 2)! / n!", &mut ctx).expect("parse");
        assert!(ConsecutiveFactorialRatioRule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root()
            )
            .is_none());
    }
}
