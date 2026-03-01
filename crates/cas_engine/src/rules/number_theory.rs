use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::number_theory_support::rewrite_number_theory_call_user_poly_gcd_with_expand_eval;

define_rule!(NumberTheoryRule, "Number Theory Operations", |ctx, expr| {
    let (result, desc) = rewrite_number_theory_call_user_poly_gcd_with_expand_eval(
        ctx,
        expr,
        cas_formatter::render_expr,
        crate::rules::algebra::poly_runtime::eval_expand_off,
    )?;

    Some(Rewrite::new(result).desc(desc))
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
