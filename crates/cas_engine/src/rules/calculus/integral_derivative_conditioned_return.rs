//! Conditioned return policy for integration-backed derivative shortcuts.
//!
//! Routing modules decide which integrand form is valid. This module owns the
//! final `hold(...)` wrapping and condition collection for the public
//! `diff(integrate(...), x)` shortcut result.

use super::integration_conditions::IntegrationRequiredConditions;
use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::ImplicitCondition;
use cas_ast::{Context, ExprId};

pub(super) fn conditioned_integral_derivative_shortcut_result(
    ctx: &mut Context,
    compact: ExprId,
    integrate_call: &NamedVarCall,
) -> (ExprId, Vec<ImplicitCondition>) {
    let required_conditions = IntegrationRequiredConditions::from_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )
    .into_implicit_conditions()
    .collect();

    (cas_ast::hold::wrap_hold(ctx, compact), required_conditions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn conditioned_result_wraps_compact_and_collects_conditions() {
        let mut ctx = Context::new();
        let compact = parse("sin(2*x+1)", &mut ctx).unwrap();
        let integrate_call = NamedVarCall {
            target: compact,
            var_name: "x".to_string(),
        };

        let (held, conditions) =
            conditioned_integral_derivative_shortcut_result(&mut ctx, compact, &integrate_call);

        assert!(hold::is_hold(&ctx, held));
        assert!(conditions.is_empty());
    }
}
