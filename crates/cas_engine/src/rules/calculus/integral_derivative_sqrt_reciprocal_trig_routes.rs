//! Sqrt-chain reciprocal-trig-product route for derivative shortcuts.
//!
//! This module owns only the source-side acceptance policy for the
//! `diff(integrate(...), x)` shortcut gate. The matched route returns the
//! original integrand for presentation; it does not perform an additional
//! integrator verification.

use super::sqrt_reciprocal_trig_product_integrand_presentation::sqrt_reciprocal_trig_product_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) enum SqrtReciprocalTrigProductIntegralDerivativeRoute {
    NoMatch,
    AcceptedSource(ExprId),
}

impl SqrtReciprocalTrigProductIntegralDerivativeRoute {
    pub(super) fn into_source_target(self) -> Option<ExprId> {
        match self {
            SqrtReciprocalTrigProductIntegralDerivativeRoute::AcceptedSource(source_target) => {
                Some(source_target)
            }
            SqrtReciprocalTrigProductIntegralDerivativeRoute::NoMatch => None,
        }
    }
}

pub(super) fn sqrt_reciprocal_trig_product_integral_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtReciprocalTrigProductIntegralDerivativeRoute {
    if sqrt_reciprocal_trig_product_integrand_target(ctx, target, var_name) {
        return SqrtReciprocalTrigProductIntegralDerivativeRoute::AcceptedSource(target);
    }

    SqrtReciprocalTrigProductIntegralDerivativeRoute::NoMatch
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn accepts_sqrt_chain_reciprocal_trig_product_source() {
        let mut ctx = Context::new();
        let target = parse("sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        assert!(matches!(
            sqrt_reciprocal_trig_product_integral_derivative_route(&mut ctx, target, "x"),
            SqrtReciprocalTrigProductIntegralDerivativeRoute::AcceptedSource(source)
                if source == target
        ));
    }

    #[test]
    fn rejects_non_sqrt_reciprocal_trig_product_source() {
        let mut ctx = Context::new();
        let target = parse("sec(x)*tan(x)", &mut ctx).unwrap();

        assert!(matches!(
            sqrt_reciprocal_trig_product_integral_derivative_route(&mut ctx, target, "x"),
            SqrtReciprocalTrigProductIntegralDerivativeRoute::NoMatch
        ));
    }
}
