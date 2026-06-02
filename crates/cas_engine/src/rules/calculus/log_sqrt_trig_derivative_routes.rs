//! Direct diff routes for log/sqrt trigonometric derivative families.
//!
//! The presentation helpers own the compact formulas. This route owns their
//! priority in the diff shortcut chain and returns conditions as one boundary.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::log_reciprocal_trig_sqrt_derivative_presentation::ln_reciprocal_trig_sqrt_derivative_presentation;
use super::log_shifted_tan_sqrt_derivative_presentation::ln_constant_shifted_tan_sqrt_derivative_presentation;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn log_sqrt_trig_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        log_sqrt_trig_derivative_route(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

pub(super) fn log_sqrt_trig_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    ln_reciprocal_trig_sqrt_derivative_presentation(ctx, target, var_name)
        .or_else(|| ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::log_sqrt_trig_derivative_route;
    use crate::rule::Rewrite;
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn rendered_required_conditions(ctx: &Context, rewrite: &Rewrite) -> Vec<String> {
        rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(ctx))
            .collect()
    }

    #[test]
    fn log_sqrt_trig_rewrite_preserves_finalized_conditions() {
        let mut ctx = Context::new();
        let target = parse("ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1)))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = super::log_sqrt_trig_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "3 / (2 * sqrt(3 * x + 1) * cos(sqrt(3 * x + 1)))"
        );
        assert_eq!(
            rendered_required_conditions(&ctx, &rewrite),
            vec![
                "x > -1/3",
                "cos(sqrt(3 * x + 1)) ≠ 0",
                "tan(sqrt(3 * x + 1)) + sec(sqrt(3 * x + 1)) > 0"
            ]
        );
    }

    #[test]
    fn log_sqrt_trig_route_preserves_reciprocal_trig_conditions() {
        let mut ctx = Context::new();
        let target = parse("ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1)))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            log_sqrt_trig_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "3 / (2 * sqrt(3 * x + 1) * cos(sqrt(3 * x + 1)))"
        );
        assert_eq!(required_conditions.len(), 3);
        assert_eq!(required_conditions[0].display(&ctx), "x > -1/3");
    }

    #[test]
    fn log_sqrt_trig_route_preserves_shifted_tan_conditions() {
        let mut ctx = Context::new();
        let target = parse("ln(1+tan(sqrt(2*x+3)))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            log_sqrt_trig_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (sqrt(2 * x + 3) * cos(sqrt(2 * x + 3))^2 * (tan(sqrt(2 * x + 3)) + 1))"
        );
        assert_eq!(required_conditions.len(), 3);
        assert_eq!(required_conditions[0].display(&ctx), "x > -3/2");
    }
}
