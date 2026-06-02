//! Direct trig-affine route for integration-backed derivative shortcuts.
//!
//! This route is the final verified fallback in the `diff(integrate(...), x)`
//! shortcut gate. It accepts only source integrands containing direct
//! sine/cosine calls with affine arguments and verifies the source integrand
//! through the symbolic integrator before returning it.

use super::direct_trig_affine_integrand_presentation::expr_contains_direct_trig_with_affine_arg;
use super::integration_antiderivative_verification::verified_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) fn direct_trig_affine_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if !expr_contains_direct_trig_with_affine_arg(ctx, target, var_name) {
        return None;
    }

    verified_integrand_target(ctx, target, var_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn accepts_verified_direct_trig_affine_source() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)", &mut ctx).unwrap();

        assert_eq!(
            direct_trig_affine_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_direct_trig_nonlinear_source() {
        let mut ctx = Context::new();
        let target = parse("sin(x^2)", &mut ctx).unwrap();

        assert_eq!(
            direct_trig_affine_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
