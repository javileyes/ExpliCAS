use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::number_theory_support::{
    dispatch_number_theory_call, render_number_theory_desc_with, NumberTheoryDispatch,
};

define_rule!(NumberTheoryRule, "Number Theory Operations", |ctx, expr| {
    match dispatch_number_theory_call(ctx, expr)? {
        NumberTheoryDispatch::Simple(simple) => {
            let desc = render_number_theory_desc_with(simple, |id| {
                format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
            });
            Some(Rewrite::new(simple.result()).desc(desc))
        }
        NumberTheoryDispatch::PolyGcd { lhs, rhs, mode } => {
            use crate::rules::algebra::poly_gcd::compute_poly_gcd_unified;
            use cas_math::poly_gcd_mode::GcdGoal;

            let (result, desc) = compute_poly_gcd_unified(
                ctx,
                lhs,
                rhs,
                GcdGoal::UserPolyGcd,
                mode,
                None, // modp_preset
                None, // modp_main_var
            );

            // Wrap in __hold to prevent further simplification
            let held = cas_ast::hold::wrap_hold(ctx, result);
            Some(Rewrite::new(held).desc(desc))
        }
    }
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NumberTheoryRule));
}
