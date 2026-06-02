//! Source-side held routes for integration-backed derivative shortcuts.
//!
//! This module owns route 11 from `integral_derivative_shortcut_presentation`.
//! The matched route returns a held compact/source integrand for presentation;
//! it does not perform an additional integrator verification.

use super::integral_derivative_shortcut_return_policy::held_inverse_sqrt_compact_or_source_integrand_target;
use super::inverse_sqrt_product_integrand_preservation::arcsin_inverse_sqrt_product_integrand_for_calculus_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn source_held_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if !arcsin_inverse_sqrt_product_integrand_for_calculus_presentation(ctx, target, var_name) {
        return None;
    }

    Some(held_inverse_sqrt_compact_or_source_integrand_target(
        ctx, target,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn accepts_arcsin_inverse_sqrt_product_as_held_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/(sqrt(x)*sqrt(1-x))", &mut ctx).unwrap();

        let held = source_held_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
    }

    #[test]
    fn rejects_source_outside_held_source_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            source_held_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
