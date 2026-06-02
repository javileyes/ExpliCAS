//! Held-presentation routes for integration-backed derivative shortcuts.
//!
//! This module preserves the route order for routes 11-13 from
//! `integral_derivative_shortcut_presentation`. They intentionally return held
//! source or compact/source integrands so the caller can collect required
//! integration conditions from the original source.

use super::integral_derivative_held_source_routes::source_held_integral_derivative_shortcut;
use super::integral_derivative_verified_held_inverse_sqrt_routes::verified_held_inverse_sqrt_integral_derivative_shortcut;
use super::integral_derivative_verified_held_polynomial_substitution_routes::verified_held_polynomial_substitution_integral_derivative_shortcut;
use cas_ast::{Context, ExprId};

pub(super) fn held_presentation_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(held_target) = source_held_integral_derivative_shortcut(ctx, target, var_name) {
        return Some(held_target);
    }

    if let Some(held_target) =
        verified_held_inverse_sqrt_integral_derivative_shortcut(ctx, target, var_name)
    {
        return Some(held_target);
    }

    if let Some(held_target) =
        verified_held_polynomial_substitution_integral_derivative_shortcut(ctx, target, var_name)
    {
        return Some(held_target);
    }

    None
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

        let held = held_presentation_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
    }

    #[test]
    fn accepts_affine_sqrt_product_derivative_as_verified_held_integrand() {
        let mut ctx = Context::new();
        let target = parse("1/sqrt(x)", &mut ctx).unwrap();

        let held = held_presentation_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
    }

    #[test]
    fn accepts_polynomial_substitution_as_verified_held_source_integrand() {
        let mut ctx = Context::new();
        let target = parse("2*x/sqrt(1+x^4)", &mut ctx).unwrap();

        let held = held_presentation_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
        assert_eq!(hold::unwrap_internal_hold(&ctx, held), target);
    }

    #[test]
    fn rejects_source_outside_held_presentation_group() {
        let mut ctx = Context::new();
        let target = parse("sin(x)", &mut ctx).unwrap();

        assert_eq!(
            held_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
