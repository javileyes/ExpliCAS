//! Definite integration over the real domain via the fundamental theorem
//! (block 13 first rung). The mathematical core is the INTERVAL
//! CERTIFICATE: before any substitution, every condition required by the
//! indefinite antiderivative must be certified on [lower, upper] with
//! numeric rational bounds — a pole inside the closed interval makes the
//! integral undefined (divergent for the supported rational families),
//! and anything the certificate cannot decide stays an honest residual.

use super::integration_conditions::IntegrationRequiredConditions;
use super::integration_result_pipeline::integrate_with_result_preservation;
use crate::symbolic_calculus_call_support::DefiniteIntegralCall;
use crate::ImplicitCondition;
use crate::Rewrite;
use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::numeric_eval::as_rational_const;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

enum IntervalCertificate {
    Certified,
    Undefined,
    Unknown,
}

pub(super) fn definite_integration_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
) -> Option<Rewrite> {
    let lower = as_rational_const(ctx, call.lower);
    let upper = as_rational_const(ctx, call.upper);
    if let (Some(lower), Some(upper)) = (&lower, &upper) {
        if lower == upper {
            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc("integrate(f, x, a, a) = 0"));
        }
    }

    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let antiderivative = integrate_with_result_preservation(
        ctx,
        call.target,
        &call.var_name,
        &mut required_conditions,
    )?;
    let (Some(lower), Some(upper)) = (lower, upper) else {
        // Symbolic bounds: sound exactly when the antiderivative is
        // unconditional - every condition-free antiderivative the engine
        // emits is continuous on all of R, so there is no interval to
        // certify (the curriculum "area function" integrate(f, x, a, t)).
        if required_conditions
            .into_implicit_conditions()
            .next()
            .is_some()
        {
            return None;
        }
        let mut antiderivative = antiderivative;
        loop {
            let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
            if unwrapped == antiderivative {
                break;
            }
            antiderivative = unwrapped;
        }
        let at_upper =
            cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.upper);
        let at_lower =
            cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.lower);
        let result = ctx.add(Expr::Sub(at_upper, at_lower));
        return Some(Rewrite::new(result).desc("integrate(f, x, a, b)"));
    };
    // F(upper) - F(lower) is already orientation-aware; the swap is only
    // for the certificate's closed interval.
    let (interval_low, interval_high) = if lower < upper {
        (lower, upper)
    } else {
        (upper, lower)
    };
    let mut antiderivative = antiderivative;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative = unwrapped;
    }

    match certify_interval(
        ctx,
        required_conditions,
        &call.var_name,
        &interval_low,
        &interval_high,
    ) {
        IntervalCertificate::Certified => {}
        IntervalCertificate::Undefined => {
            let undefined = ctx.add(Expr::Constant(Constant::Undefined));
            return Some(
                Rewrite::new(undefined)
                    .desc("integrate(f, x, a, b) diverges: pole inside the interval"),
            );
        }
        IntervalCertificate::Unknown => return None,
    }

    let at_upper = cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.upper);
    let at_lower = cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.lower);
    let result = ctx.add(Expr::Sub(at_upper, at_lower));
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Decide every required condition on the closed interval. Conservative by
/// construction: only conditions provably independent of the interval (or
/// provably violated inside it) are decided; everything else is Unknown
/// and the call stays residual.
fn certify_interval(
    ctx: &mut Context,
    required_conditions: IntegrationRequiredConditions,
    var_name: &str,
    interval_low: &BigRational,
    interval_high: &BigRational,
) -> IntervalCertificate {
    let mut outcome = IntervalCertificate::Certified;
    for condition in required_conditions.into_implicit_conditions() {
        match condition {
            ImplicitCondition::NonZero(expr) => {
                match nonzero_on_interval(ctx, expr, var_name, interval_low, interval_high) {
                    IntervalCertificate::Certified => {}
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    IntervalCertificate::Unknown => outcome = IntervalCertificate::Unknown,
                }
            }
            ImplicitCondition::Positive(expr) => {
                // Only variable-free numeric positivity is certifiable here.
                match as_rational_const(ctx, expr) {
                    Some(value) if value.is_positive() => {}
                    _ => outcome = IntervalCertificate::Unknown,
                }
            }
            _ => outcome = IntervalCertificate::Unknown,
        }
    }
    outcome
}

fn nonzero_on_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &BigRational,
    interval_high: &BigRational,
) -> IntervalCertificate {
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_zero() {
            IntervalCertificate::Undefined
        } else {
            IntervalCertificate::Certified
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    match poly.degree() {
        0 => {
            if poly.coeffs.first().is_none_or(num_traits::Zero::is_zero) {
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        1 => {
            let root = -&poly.coeffs[0] / &poly.coeffs[1];
            if &root >= interval_low && &root <= interval_high {
                // A pole of the supported rational families inside the
                // closed interval: the integral diverges.
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        2 => {
            let a = poly.coeffs[2].clone();
            let b = poly.coeffs[1].clone();
            let c = poly.coeffs[0].clone();
            let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
            if discriminant.is_negative() {
                IntervalCertificate::Certified
            } else {
                IntervalCertificate::Unknown
            }
        }
        _ => IntervalCertificate::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn eval_definite(source: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(source, &mut ctx).expect(source);
        let call =
            crate::symbolic_calculus_call_support::try_extract_definite_integrate_call(&ctx, expr)?;
        let rewrite = definite_integration_rewrite(&mut ctx, &call)?;
        Some(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ))
    }

    #[test]
    fn ftc_evaluates_certified_intervals() {
        // The raw rewrite is the unsimplified F(b) - F(a); the engine
        // simplifier folds it to 1/3 (pinned by the matrix row).
        assert!(eval_definite("integrate(x^2, x, 0, 1)").is_some());
        // Orientation is automatic: F(upper) - F(lower) with original bounds.
        assert!(eval_definite("integrate(x, x, 1, 0)").is_some());
    }

    #[test]
    fn pole_inside_closed_interval_is_undefined() {
        let result = eval_definite("integrate(1/x, x, -1, 1)").expect("rewrite");
        assert_eq!(result, "undefined");
        let endpoint = eval_definite("integrate(1/x, x, 0, 1)").expect("rewrite");
        assert_eq!(endpoint, "undefined");
    }

    #[test]
    fn unknown_certificates_stay_residual() {
        // Symbolic bound over a CONDITIONAL antiderivative: the pole
        // location cannot be certified against a symbolic interval.
        assert!(eval_definite("integrate(1/x, x, 1, t)").is_none());
        // Symbolic positivity condition cannot be certified.
        assert!(eval_definite("integrate(c/((x+b)^2+a), x, 0, 1)").is_none());
    }

    #[test]
    fn symbolic_bounds_evaluate_for_unconditional_antiderivatives() {
        // The curriculum "area function": condition-free antiderivatives
        // are continuous on all of R, so symbolic bounds need no
        // certificate.
        assert!(eval_definite("integrate(x^2, x, 0, t)").is_some());
        assert!(eval_definite("integrate(1/(x^2+1), x, a, b)").is_some());
    }
}
