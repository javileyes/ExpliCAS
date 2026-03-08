use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::number_theory_support::{
    dispatch_number_theory_call, NumberTheoryDispatch, NumberTheorySimpleRewrite,
};
use cas_math::poly_gcd_dispatch::{compute_poly_gcd_unified_with, pre_evaluate_for_gcd_with};
use cas_math::poly_gcd_mode::GcdGoal;

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
