//! Arctan-polynomial source-side route for derivative shortcuts.
//!
//! This module owns route 6 from `integral_derivative_shortcut_presentation`.
//! The matched route returns the original integrand for presentation; it does
//! not perform an additional integrator verification.

use super::arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
use cas_ast::{Context, ExprId};

pub(super) fn arctan_polynomial_source_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    polynomial_times_arctan_affine_integrand_for_diff_shortcut(ctx, target, var_name)
        .then_some(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn accepts_polynomial_times_arctan_affine_source() {
        let mut ctx = Context::new();
        let target = parse("x^2*arctan(1-x)", &mut ctx).unwrap();

        assert_eq!(
            arctan_polynomial_source_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_non_arctan_polynomial_source() {
        let mut ctx = Context::new();
        let target = parse("x^2*sin(1-x)", &mut ctx).unwrap();

        assert_eq!(
            arctan_polynomial_source_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}
