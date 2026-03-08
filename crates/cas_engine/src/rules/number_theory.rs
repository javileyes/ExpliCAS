use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::number_theory_support::rewrite_number_theory_call_with_expand_eval;
use cas_math::poly_gcd_mode::GcdGoal;

define_rule!(NumberTheoryRule, "Number Theory Operations", |ctx, expr| {
    let (result, desc) = rewrite_number_theory_call_with_expand_eval(
        ctx,
        expr,
        GcdGoal::UserPolyGcd,
        cas_formatter::render_expr,
        crate::expand::eval_expand_off,
    )?;

    Some(Rewrite::new(result).desc(desc))
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
