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
use cas_math::limit_types::{Approach, LimitOptions};
use cas_math::numeric_eval::as_rational_const;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

enum IntervalCertificate {
    Certified,
    /// Every obstruction is a root exactly AT a boundary endpoint of the
    /// sorted interval: the integral may converge as an improper one,
    /// decided by one-sided limits of the antiderivative.
    BoundaryTouch {
        lower: bool,
        upper: bool,
    },
    Undefined,
    Unknown,
}

enum DefiniteBound {
    Finite(Endpoint),
    PosInfinity,
    NegInfinity,
    Symbolic,
}

/// Exact interval endpoint of the form `rational + pi_multiple * pi`,
/// covering both rational bounds and the exam-standard rational multiples
/// of pi. Comparisons are exact whenever the pi parts agree (in
/// particular for two pi-pure values, which is how trig zeros at
/// k*pi/2 compare against pi-multiple bounds); mixed comparisons fall
/// back to the rational pi enclosure and refuse when undecidable.
#[derive(Clone, PartialEq)]
struct Endpoint {
    rational: BigRational,
    pi_multiple: BigRational,
}

impl Endpoint {
    fn from_rational(value: BigRational) -> Self {
        Endpoint {
            rational: value,
            pi_multiple: BigRational::from_integer(0.into()),
        }
    }

    fn from_pi_multiple(multiple: BigRational) -> Self {
        Endpoint {
            rational: BigRational::from_integer(0.into()),
            pi_multiple: multiple,
        }
    }

    fn enclosure(&self) -> (BigRational, BigRational) {
        let (pi_low, pi_high) = pi_enclosure();
        if self.pi_multiple >= BigRational::from_integer(0.into()) {
            (
                &self.rational + &self.pi_multiple * pi_low,
                &self.rational + &self.pi_multiple * pi_high,
            )
        } else {
            (
                &self.rational + &self.pi_multiple * pi_high,
                &self.rational + &self.pi_multiple * pi_low,
            )
        }
    }

    fn try_cmp(&self, other: &Endpoint) -> Option<std::cmp::Ordering> {
        if self.pi_multiple == other.pi_multiple {
            return Some(self.rational.cmp(&other.rational));
        }
        let (self_low, self_high) = self.enclosure();
        let (other_low, other_high) = other.enclosure();
        if self_high < other_low {
            return Some(std::cmp::Ordering::Less);
        }
        if self_low > other_high {
            return Some(std::cmp::Ordering::Greater);
        }
        None
    }
}

/// Recognize rational multiples of pi: pi, q*pi, pi/n and negations.
fn pi_multiple_of(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Pi) => Some(BigRational::from_integer(1.into())),
        Expr::Neg(inner) => pi_multiple_of(ctx, *inner).map(|value| -value),
        Expr::Mul(l, r) => {
            if let Some(scale) = as_rational_const(ctx, *l) {
                return pi_multiple_of(ctx, *r).map(|value| scale * value);
            }
            if let Some(scale) = as_rational_const(ctx, *r) {
                return pi_multiple_of(ctx, *l).map(|value| scale * value);
            }
            None
        }
        Expr::Div(numerator, denominator) => {
            let divisor = as_rational_const(ctx, *denominator)?;
            if divisor.is_zero() {
                return None;
            }
            pi_multiple_of(ctx, *numerator).map(|value| value / divisor)
        }
        _ => None,
    }
}

fn classify_bound(ctx: &Context, bound: ExprId) -> DefiniteBound {
    if let Some(value) = as_rational_const(ctx, bound) {
        return DefiniteBound::Finite(Endpoint::from_rational(value));
    }
    if let Some(multiple) = pi_multiple_of(ctx, bound) {
        return DefiniteBound::Finite(Endpoint::from_pi_multiple(multiple));
    }
    match ctx.get(bound) {
        Expr::Constant(Constant::Infinity) => DefiniteBound::PosInfinity,
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            DefiniteBound::NegInfinity
        }
        _ => DefiniteBound::Symbolic,
    }
}

/// A rational point strictly inside the (possibly pi-valued) interval,
/// via the enclosures; None when the enclosures overlap.
fn interval_rational_probe(low: &Endpoint, high: &Endpoint) -> Option<BigRational> {
    let (_, low_high) = low.enclosure();
    let (high_low, _) = high.enclosure();
    if low_high < high_low {
        Some((low_high + high_low) / BigRational::from_integer(2.into()))
    } else {
        None
    }
}

enum RootPosition {
    Outside,
    AtLower,
    AtUpper,
    Inside,
}

/// Position of a rational root relative to the closed interval; None
/// when undecidable.
fn root_position(low: &Endpoint, high: &Endpoint, value: &BigRational) -> Option<RootPosition> {
    let point = Endpoint::from_rational(value.clone());
    match point.try_cmp(low)? {
        std::cmp::Ordering::Less => return Some(RootPosition::Outside),
        std::cmp::Ordering::Equal => return Some(RootPosition::AtLower),
        std::cmp::Ordering::Greater => {}
    }
    match point.try_cmp(high)? {
        std::cmp::Ordering::Greater => Some(RootPosition::Outside),
        std::cmp::Ordering::Equal => Some(RootPosition::AtUpper),
        std::cmp::Ordering::Less => Some(RootPosition::Inside),
    }
}

pub(super) fn definite_integration_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
) -> Option<Rewrite> {
    let lower_bound = classify_bound(ctx, call.lower);
    let upper_bound = classify_bound(ctx, call.upper);
    if let (DefiniteBound::Finite(lower), DefiniteBound::Finite(upper)) =
        (&lower_bound, &upper_bound)
    {
        if lower == upper {
            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc("integrate(f, x, a, a) = 0"));
        }
    }

    let (antiderivative, conditions) = resolve_indefinite_for_definite(ctx, call)?;

    if matches!(
        lower_bound,
        DefiniteBound::PosInfinity | DefiniteBound::NegInfinity
    ) || matches!(
        upper_bound,
        DefiniteBound::PosInfinity | DefiniteBound::NegInfinity
    ) {
        return improper_integration_rewrite(
            ctx,
            call,
            antiderivative,
            conditions,
            &lower_bound,
            &upper_bound,
        );
    }

    let (DefiniteBound::Finite(lower), DefiniteBound::Finite(upper)) = (lower_bound, upper_bound)
    else {
        // Symbolic bounds: sound exactly when the antiderivative is
        // unconditional - every condition-free antiderivative the engine
        // emits is continuous on all of R, so there is no interval to
        // certify (the curriculum "area function" integrate(f, x, a, t)).
        if !conditions.is_empty() {
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
    let (interval_low, interval_high) = match lower.try_cmp(&upper) {
        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => (lower, upper),
        Some(std::cmp::Ordering::Greater) => (upper, lower),
        None => return None,
    };
    let mut antiderivative = antiderivative;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative = unwrapped;
    }

    match combine_certificates(
        certify_interval(
            ctx,
            &conditions,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
        integrand_risks_certified(
            ctx,
            call.target,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
    ) {
        IntervalCertificate::Certified => {}
        IntervalCertificate::BoundaryTouch {
            lower: touch_low,
            upper: touch_high,
        } => {
            return boundary_touch_evaluation(
                ctx,
                call,
                antiderivative,
                &interval_low,
                &interval_high,
                touch_low,
                touch_high,
            );
        }
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

/// Boundary-touched endpoints: the obstruction sits exactly at an
/// endpoint of the sorted interval, so the boundary value is the
/// ONE-SIDED LIMIT of the antiderivative approaching from inside the
/// interval (curriculum improper integrals like the unit-interval
/// natural-log integral evaluating to -1). Finite limits converge;
/// signed infinities report honest
/// divergence; unresolved limits stay residual.
#[allow(clippy::too_many_arguments)]
fn boundary_touch_evaluation(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    antiderivative: ExprId,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
    touch_low: bool,
    touch_high: bool,
) -> Option<Rewrite> {
    let bound_value = |ctx: &mut Context, bound_expr: ExprId| -> Option<ExprId> {
        let endpoint = match classify_bound(ctx, bound_expr) {
            DefiniteBound::Finite(endpoint) => endpoint,
            _ => return None,
        };
        let touched =
            (touch_low && endpoint == *interval_low) || (touch_high && endpoint == *interval_high);
        if !touched {
            return Some(cas_ast::substitute_expr_by_id(
                ctx,
                antiderivative,
                call.var_expr,
                bound_expr,
            ));
        }
        // Approach from inside the sorted interval.
        let side = if endpoint == *interval_low {
            cas_math::limit_types::FiniteLimitSide::Right
        } else {
            cas_math::limit_types::FiniteLimitSide::Left
        };
        let opts = LimitOptions::default();
        let mut budget = crate::budget::Budget::preset_cli();
        let outcome = crate::limits::limit(
            ctx,
            antiderivative,
            call.var_expr,
            Approach::FiniteOneSided(bound_expr, side),
            &opts,
            &mut budget,
        )
        .ok()?;
        if outcome.warning.is_some() || expr_contains_limit_call(ctx, outcome.expr) {
            return None;
        }
        if matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined)) {
            return None;
        }
        Some(outcome.expr)
    };

    let upper_value = bound_value(ctx, call.upper)?;
    let lower_value = bound_value(ctx, call.lower)?;

    let upper_sign = infinite_sign(ctx, upper_value);
    let lower_sign = infinite_sign(ctx, lower_value);
    let result = match (upper_sign, lower_sign) {
        (Some(_), Some(_)) => return None, // infinity - infinity: indeterminate
        (Some(sign), None) => build_signed_infinity(ctx, sign),
        (None, Some(sign)) => build_signed_infinity(ctx, -sign),
        (None, None) => ctx.add(Expr::Sub(upper_value, lower_value)),
    };
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Improper integrals: at an infinite bound the boundary value is the
/// LIMIT of the antiderivative (never a substitution - the symbolic path
/// used to leak the infinity constant into F, producing forms like
/// arctan(infinity)). Divergence to +-infinity is reported as the honest
/// infinite value; indeterminate combinations stay residual.
fn improper_integration_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    antiderivative: ExprId,
    conditions: Vec<ImplicitCondition>,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    match combine_certificates(
        certify_unbounded_interval(ctx, &conditions, &call.var_name, lower_bound, upper_bound),
        integrand_risks_certified_unbounded(
            ctx,
            call.target,
            &call.var_name,
            lower_bound,
            upper_bound,
        ),
    ) {
        IntervalCertificate::Certified => {}
        // Touches at the finite endpoint of an unbounded interval need
        // mixed one-sided/at-infinity evaluation: next rung, residual.
        IntervalCertificate::BoundaryTouch { .. } => return None,
        IntervalCertificate::Undefined => {
            let undefined = ctx.add(Expr::Constant(Constant::Undefined));
            return Some(
                Rewrite::new(undefined)
                    .desc("integrate(f, x, a, b) diverges: pole inside the interval"),
            );
        }
        IntervalCertificate::Unknown => return None,
    }

    let mut antiderivative = antiderivative;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative = unwrapped;
    }

    let upper_value = boundary_value(ctx, antiderivative, call.var_expr, call.upper, upper_bound)?;
    let lower_value = boundary_value(ctx, antiderivative, call.var_expr, call.lower, lower_bound)?;

    let upper_sign = infinite_sign(ctx, upper_value);
    let lower_sign = infinite_sign(ctx, lower_value);
    let result = match (upper_sign, lower_sign) {
        (Some(_), Some(_)) => return None, // infinity - infinity: indeterminate
        (Some(sign), None) => build_signed_infinity(ctx, sign),
        (None, Some(sign)) => build_signed_infinity(ctx, -sign),
        (None, None) => ctx.add(Expr::Sub(upper_value, lower_value)),
    };
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Boundary value of the antiderivative: substitution at finite or
/// symbolic-free bounds, the limits engine at infinite ones. None when
/// the limit is unresolved or unsafe (honest residual).
fn boundary_value(
    ctx: &mut Context,
    antiderivative: ExprId,
    var_expr: ExprId,
    bound_expr: ExprId,
    bound: &DefiniteBound,
) -> Option<ExprId> {
    let approach = match bound {
        DefiniteBound::Finite(_) => {
            return Some(cas_ast::substitute_expr_by_id(
                ctx,
                antiderivative,
                var_expr,
                bound_expr,
            ));
        }
        DefiniteBound::PosInfinity => Approach::PosInfinity,
        DefiniteBound::NegInfinity => Approach::NegInfinity,
        DefiniteBound::Symbolic => return None,
    };
    let opts = LimitOptions::default();
    let mut budget = crate::budget::Budget::preset_cli();
    let outcome =
        crate::limits::limit(ctx, antiderivative, var_expr, approach, &opts, &mut budget).ok()?;
    if outcome.warning.is_some() || expr_contains_limit_call(ctx, outcome.expr) {
        return None;
    }
    if matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined)) {
        return None;
    }
    Some(outcome.expr)
}

fn expr_contains_limit_call(ctx: &mut Context, expr: ExprId) -> bool {
    let limit_symbol = ctx.intern_symbol("limit");
    expr_contains_call_to(ctx, expr, limit_symbol)
}

fn expr_contains_call_to(ctx: &Context, expr: ExprId, target: cas_ast::symbol::SymbolId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            *fn_id == target
                || args
                    .iter()
                    .any(|arg| expr_contains_call_to(ctx, *arg, target))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_call_to(ctx, *l, target) || expr_contains_call_to(ctx, *r, target)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_call_to(ctx, *inner, target),
        _ => false,
    }
}

fn infinite_sign(ctx: &Context, expr: ExprId) -> Option<i32> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => Some(1),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(-1)
        }
        _ => None,
    }
}

fn build_signed_infinity(ctx: &mut Context, sign: i32) -> ExprId {
    let infinity = ctx.add(Expr::Constant(Constant::Infinity));
    if sign >= 0 {
        infinity
    } else {
        ctx.add(Expr::Neg(infinity))
    }
}

/// Certificate over a (half-)infinite interval: linear poles must lie
/// strictly outside the unbounded closed interval.
fn certify_unbounded_interval(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    let mut outcome = IntervalCertificate::Certified;
    for condition in conditions {
        match condition {
            ImplicitCondition::NonZero(expr) => {
                // cos/sin have zeros in every unbounded interval.
                if trig_condition_target(ctx, *expr, var_name) {
                    return IntervalCertificate::Undefined;
                }
                match nonzero_on_unbounded_interval(ctx, *expr, var_name, lower_bound, upper_bound)
                {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            ImplicitCondition::Positive(expr) | ImplicitCondition::NonNegative(expr) => {
                match globally_positive(ctx, *expr, var_name) {
                    true => {}
                    false => outcome = IntervalCertificate::Unknown,
                }
            }
            _ => outcome = IntervalCertificate::Unknown,
        }
    }
    outcome
}

fn trig_condition_target(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Sin)
            ) && matches!(ctx.get(args[0]),
                Expr::Variable(sym) if ctx.sym_name(*sym) == var_name)
        }
        _ => false,
    }
}

/// Positivity on all of R: variable-free positive numerics, or quadratics
/// with negative discriminant and positive leading coefficient.
fn globally_positive(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    if let Some(value) = as_rational_const(ctx, expr) {
        return value.is_positive();
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 {
        return false;
    }
    let a = poly.coeffs[2].clone();
    let b = poly.coeffs[1].clone();
    let c = poly.coeffs[0].clone();
    let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
    discriminant.is_negative() && a.is_positive()
}

fn nonzero_on_unbounded_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
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
            if poly.coeffs.first().is_none_or(Zero::is_zero) {
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        1 => {
            let root = -&poly.coeffs[0] / &poly.coeffs[1];
            let point = Endpoint::from_rational(root);
            let above_lower = match lower_bound {
                DefiniteBound::NegInfinity => true,
                DefiniteBound::Finite(lower) => match point.try_cmp(lower) {
                    Some(std::cmp::Ordering::Less) => false,
                    Some(_) => true,
                    None => return IntervalCertificate::Unknown,
                },
                _ => return IntervalCertificate::Unknown,
            };
            let below_upper = match upper_bound {
                DefiniteBound::PosInfinity => true,
                DefiniteBound::Finite(upper) => match point.try_cmp(upper) {
                    Some(std::cmp::Ordering::Greater) => false,
                    Some(_) => true,
                    None => return IntervalCertificate::Unknown,
                },
                _ => return IntervalCertificate::Unknown,
            };
            if above_lower && below_upper {
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

/// Decide every required condition on the closed interval. Conservative by
/// construction: only conditions provably independent of the interval (or
/// provably violated inside it) are decided; everything else is Unknown
/// and the call stays residual.
/// Resolve the indefinite antiderivative for a definite call, mirroring
/// the indefinite rule's route order: the derivative-cofactor route
/// first (it owns u'/sqrt(u)-style shapes the standard pipeline
/// declines), then the standard pipeline.
fn resolve_indefinite_for_definite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    if let Some((result, condition)) =
        super::integration_derivative_cofactor_routes::polynomial_trig_reciprocal_derivative_root_gate_route(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, vec![condition]));
    }
    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let antiderivative = integrate_with_result_preservation(
        ctx,
        call.target,
        &call.var_name,
        &mut required_conditions,
    )?;
    Some((
        antiderivative,
        required_conditions.into_implicit_conditions().collect(),
    ))
}

fn certify_interval(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let mut outcome = IntervalCertificate::Certified;
    for condition in conditions {
        match condition {
            ImplicitCondition::NonZero(expr) => {
                match nonzero_on_interval(ctx, *expr, var_name, interval_low, interval_high) {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            ImplicitCondition::Positive(expr) | ImplicitCondition::NonNegative(expr) => {
                match positive_on_interval(ctx, *expr, var_name, interval_low, interval_high) {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            _ => outcome = IntervalCertificate::Unknown,
        }
    }
    outcome
}

fn combine_certificates(
    first: IntervalCertificate,
    second: IntervalCertificate,
) -> IntervalCertificate {
    use IntervalCertificate::*;
    match (first, second) {
        (Undefined, _) | (_, Undefined) => Undefined,
        (Unknown, _) | (_, Unknown) => Unknown,
        (BoundaryTouch { lower: a, upper: b }, BoundaryTouch { lower: c, upper: d }) => {
            BoundaryTouch {
                lower: a || c,
                upper: b || d,
            }
        }
        (touch @ BoundaryTouch { .. }, Certified) | (Certified, touch @ BoundaryTouch { .. }) => {
            touch
        }
        (Certified, Certified) => Certified,
    }
}

/// SELF-CONTAINED risk scan of the integrand: the condition collectors
/// are not guaranteed complete (adversarial review found ln-denominator
/// conditions systematically absent), so certification additionally
/// requires every risky subterm of the integrand itself to be certified
/// on the interval - denominators factor by factor, ln arguments
/// positive AND away from 1 (ln(u) = 0 at u = 1), fractional-power bases
/// positive, trig denominators via the pi enclosure. Anything the scan
/// cannot decide refuses certification.
fn integrand_risks_certified(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    scan_expr_risks(ctx, expr, var_name, &mut |ctx, risk| match risk {
        RiskKind::DenominatorNonZero(factor) => {
            nonzero_on_interval(ctx, factor, var_name, interval_low, interval_high)
        }
        RiskKind::MustBePositive(arg) => {
            positive_on_interval(ctx, arg, var_name, interval_low, interval_high)
        }
        RiskKind::DefinedOnUnitInterval(arg) => {
            unit_interval_certificate(ctx, arg, var_name, interval_low, interval_high)
        }
    })
}

/// -1 <= u <= 1 over the closed interval: certify 1 - u >= 0 and
/// 1 + u >= 0, where an endpoint TOUCH (u = +-1 exactly at a bound) is
/// still certified because arcsin/arccos are defined there.
fn unit_interval_certificate(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let one = ctx.num(1);
    let upper_slack = ctx.add(Expr::Sub(one, arg));
    let one = ctx.num(1);
    let lower_slack = ctx.add(Expr::Add(one, arg));
    let mut outcome = IntervalCertificate::Certified;
    for slack in [upper_slack, lower_slack] {
        let cert = match positive_on_interval(ctx, slack, var_name, interval_low, interval_high) {
            IntervalCertificate::BoundaryTouch { .. } => IntervalCertificate::Certified,
            other => other,
        };
        outcome = combine_certificates(outcome, cert);
    }
    outcome
}

fn integrand_risks_certified_unbounded(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    scan_expr_risks(ctx, expr, var_name, &mut |ctx, risk| match risk {
        RiskKind::DenominatorNonZero(factor) => {
            if trig_condition_target(ctx, factor, var_name) {
                return IntervalCertificate::Undefined;
            }
            nonzero_on_unbounded_interval(ctx, factor, var_name, lower_bound, upper_bound)
        }
        RiskKind::MustBePositive(arg) => {
            positive_on_unbounded_interval(ctx, arg, var_name, lower_bound, upper_bound)
        }
        // arcsin/arccos cannot stay within [-1, 1] on an unbounded
        // interval for any nonconstant argument the scan certifies today.
        RiskKind::DefinedOnUnitInterval(_) => IntervalCertificate::Unknown,
    })
}

enum RiskKind {
    DenominatorNonZero(ExprId),
    MustBePositive(ExprId),
    /// arcsin/arccos argument: defined on the CLOSED unit interval, so
    /// endpoint touches certify (only the derivative is singular there).
    DefinedOnUnitInterval(ExprId),
}

fn scan_expr_risks(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
) -> IntervalCertificate {
    let node = ctx.get(expr).clone();
    let mut outcome = IntervalCertificate::Certified;
    let merge = |certificate: IntervalCertificate, outcome: &mut IntervalCertificate| {
        *outcome = combine_certificates(
            std::mem::replace(outcome, IntervalCertificate::Certified),
            certificate,
        );
    };
    match node {
        Expr::Div(numerator, denominator) => {
            merge(
                certify_denominator_factors(ctx, denominator, var_name, certify),
                &mut outcome,
            );
            merge(
                scan_expr_risks(ctx, numerator, var_name, certify),
                &mut outcome,
            );
            merge(
                scan_expr_risks(ctx, denominator, var_name, certify),
                &mut outcome,
            );
        }
        Expr::Pow(base, exponent) => {
            let exponent_value = as_rational_const(ctx, exponent);
            match exponent_value {
                Some(value) if value.is_integer() && value.is_positive() => {}
                Some(value) if value.is_integer() => {
                    merge(
                        certify_denominator_factors(ctx, base, var_name, certify),
                        &mut outcome,
                    );
                }
                Some(_) => {
                    // Fractional exponent: real-domain base positivity.
                    merge(certify(ctx, RiskKind::MustBePositive(base)), &mut outcome);
                }
                None => {
                    // Variable exponent: total and positive only for a
                    // positive constant base (e included).
                    let base_safe = matches!(ctx.get(base), Expr::Constant(Constant::E))
                        || as_rational_const(ctx, base).is_some_and(|value| value.is_positive());
                    if !base_safe {
                        merge(IntervalCertificate::Unknown, &mut outcome);
                    }
                    merge(
                        scan_expr_risks(ctx, exponent, var_name, certify),
                        &mut outcome,
                    );
                }
            }
            merge(scan_expr_risks(ctx, base, var_name, certify), &mut outcome);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            merge(scan_expr_risks(ctx, l, var_name, certify), &mut outcome);
            merge(scan_expr_risks(ctx, r, var_name, certify), &mut outcome);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            merge(scan_expr_risks(ctx, inner, var_name, certify), &mut outcome);
        }
        Expr::Function(fn_id, args) => {
            let recurse_args =
                |ctx: &mut Context,
                 certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
                 outcome: &mut IntervalCertificate| {
                    for arg in &args {
                        let cert = scan_expr_risks(ctx, *arg, var_name, certify);
                        *outcome = combine_certificates(
                            std::mem::replace(outcome, IntervalCertificate::Certified),
                            cert,
                        );
                    }
                };
            match ctx.builtin_of(fn_id) {
                Some(
                    cas_ast::BuiltinFn::Sin
                    | cas_ast::BuiltinFn::Cos
                    | cas_ast::BuiltinFn::Exp
                    | cas_ast::BuiltinFn::Arctan
                    | cas_ast::BuiltinFn::Atan
                    | cas_ast::BuiltinFn::Sinh
                    | cas_ast::BuiltinFn::Cosh
                    | cas_ast::BuiltinFn::Tanh
                    | cas_ast::BuiltinFn::Abs,
                ) => recurse_args(ctx, certify, &mut outcome),
                Some(
                    cas_ast::BuiltinFn::Arcsin
                    | cas_ast::BuiltinFn::Asin
                    | cas_ast::BuiltinFn::Arccos
                    | cas_ast::BuiltinFn::Acos,
                ) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::DefinedOnUnitInterval(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Ln) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::MustBePositive(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Sqrt) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::MustBePositive(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Tan) if args.len() == 1 => {
                    // tan's poles are cos zeros.
                    let cos_arg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![args[0]]);
                    merge(
                        certify(ctx, RiskKind::DenominatorNonZero(cos_arg)),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                _ => merge(IntervalCertificate::Unknown, &mut outcome),
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
        _ => merge(IntervalCertificate::Unknown, &mut outcome),
    }
    outcome
}

/// Certify a denominator factor by factor: polynomials by root location,
/// integer powers by their base, exp-like factors are never zero,
/// ln(u) = 0 exactly at u = 1 (certified via u - 1 nonzero plus the ln
/// domain), trig factors via the pi enclosure.
#[allow(clippy::only_used_in_recursion)]
fn certify_denominator_factors(
    ctx: &mut Context,
    denominator: ExprId,
    var_name: &str,
    certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
) -> IntervalCertificate {
    let node = ctx.get(denominator).clone();
    match node {
        Expr::Mul(l, r) => combine_certificates(
            certify_denominator_factors(ctx, l, var_name, certify),
            certify_denominator_factors(ctx, r, var_name, certify),
        ),
        Expr::Neg(inner) | Expr::Hold(inner) => {
            certify_denominator_factors(ctx, inner, var_name, certify)
        }
        Expr::Pow(base, exponent) => {
            // A positive constant base (e included) is never zero,
            // whatever the exponent.
            let base_never_zero = matches!(ctx.get(base), Expr::Constant(Constant::E))
                || as_rational_const(ctx, base).is_some_and(|value| value.is_positive());
            if base_never_zero {
                return IntervalCertificate::Certified;
            }
            match as_rational_const(ctx, exponent) {
                Some(value) if !value.is_zero() => {
                    certify_denominator_factors(ctx, base, var_name, certify)
                }
                _ => IntervalCertificate::Unknown,
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
            Some(cas_ast::BuiltinFn::Exp) => IntervalCertificate::Certified,
            Some(cas_ast::BuiltinFn::Ln) => {
                let one = ctx.num(1);
                let shifted = ctx.add(Expr::Sub(args[0], one));
                combine_certificates(
                    certify(ctx, RiskKind::MustBePositive(args[0])),
                    certify(ctx, RiskKind::DenominatorNonZero(shifted)),
                )
            }
            Some(cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Sin) => {
                certify(ctx, RiskKind::DenominatorNonZero(denominator))
            }
            Some(cas_ast::BuiltinFn::Sqrt) => {
                // sqrt(u) = 0 exactly at u = 0; strict positivity of u
                // certifies both the domain and the nonzero denominator.
                certify(ctx, RiskKind::MustBePositive(args[0]))
            }
            _ => IntervalCertificate::Unknown,
        },
        _ => certify(ctx, RiskKind::DenominatorNonZero(denominator)),
    }
}

/// Strict positivity of a polynomial condition on the closed interval:
/// variable-free numerics decide directly; otherwise every rational root
/// must lie strictly outside [low, high], the non-root residual must have
/// no real roots (negative discriminant or constant), and a sign probe at
/// an interior root-free point confirms the sign. Roots touching or
/// inside the interval are conservatively Unknown (the boundary case may
/// be a convergent improper integral, which this rung does not decide).
fn positive_on_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_positive() {
            IntervalCertificate::Certified
        } else {
            IntervalCertificate::Unknown
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    if poly.is_zero() {
        return IntervalCertificate::Unknown;
    }

    let mut residual_has_real_roots_ruled_out = false;
    let mut touches = IntervalCertificate::Certified;
    let factors = poly.factor_rational_roots();
    for factor in &factors {
        match factor.degree() {
            0 => {}
            1 => {
                let root = -&factor.coeffs[0] / &factor.coeffs[1];
                match root_position(interval_low, interval_high, &root) {
                    Some(RootPosition::Outside) => {}
                    Some(RootPosition::AtLower) => {
                        touches = combine_certificates(
                            touches,
                            IntervalCertificate::BoundaryTouch {
                                lower: true,
                                upper: false,
                            },
                        );
                    }
                    Some(RootPosition::AtUpper) => {
                        touches = combine_certificates(
                            touches,
                            IntervalCertificate::BoundaryTouch {
                                lower: false,
                                upper: true,
                            },
                        );
                    }
                    _ => return IntervalCertificate::Unknown,
                }
            }
            2 => {
                let a = factor.coeffs[2].clone();
                let b = factor.coeffs[1].clone();
                let c = factor.coeffs[0].clone();
                let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
                if !discriminant.is_negative() {
                    // Irrational real roots could lie anywhere.
                    return IntervalCertificate::Unknown;
                }
                residual_has_real_roots_ruled_out = true;
            }
            _ => return IntervalCertificate::Unknown,
        }
    }
    let _ = residual_has_real_roots_ruled_out;

    // No root strictly inside: the sign is constant in the interior;
    // probe a rational point strictly inside.
    let Some(probe) = interval_rational_probe(interval_low, interval_high) else {
        return IntervalCertificate::Unknown;
    };
    if poly.eval(&probe).is_positive() {
        touches
    } else {
        IntervalCertificate::Unknown
    }
}

/// Positivity of a polynomial on a (half-)infinite interval: globally
/// positive quadratics certify directly; otherwise every rational root
/// must lie strictly outside, quadratic residuals must have no real
/// roots, infinite tails must point positive (leading-coefficient sign),
/// and a probe inside confirms.
fn positive_on_unbounded_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    if globally_positive(ctx, expr, var_name) {
        return IntervalCertificate::Certified;
    }
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_positive() {
            IntervalCertificate::Certified
        } else {
            IntervalCertificate::Unknown
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    if poly.is_zero() {
        return IntervalCertificate::Unknown;
    }

    let inside = |root: &BigRational| -> bool {
        let point = Endpoint::from_rational(root.clone());
        let above_lower = match lower_bound {
            DefiniteBound::NegInfinity => true,
            DefiniteBound::Finite(lower) => {
                !matches!(point.try_cmp(lower), Some(std::cmp::Ordering::Less))
            }
            _ => return true, // conservador: trátalo como dentro
        };
        let below_upper = match upper_bound {
            DefiniteBound::PosInfinity => true,
            DefiniteBound::Finite(upper) => {
                !matches!(point.try_cmp(upper), Some(std::cmp::Ordering::Greater))
            }
            _ => return true,
        };
        above_lower && below_upper
    };
    for factor in poly.factor_rational_roots() {
        match factor.degree() {
            0 => {}
            1 => {
                let root = -&factor.coeffs[0] / &factor.coeffs[1];
                if inside(&root) {
                    return IntervalCertificate::Unknown;
                }
            }
            2 => {
                let a = factor.coeffs[2].clone();
                let b = factor.coeffs[1].clone();
                let c = factor.coeffs[0].clone();
                let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
                if !discriminant.is_negative() {
                    return IntervalCertificate::Unknown;
                }
            }
            _ => return IntervalCertificate::Unknown,
        }
    }

    // Infinite tails must be positive.
    let leading = poly.leading_coeff();
    if matches!(upper_bound, DefiniteBound::PosInfinity) && !leading.is_positive() {
        return IntervalCertificate::Unknown;
    }
    if matches!(lower_bound, DefiniteBound::NegInfinity) {
        let degree_even = poly.degree() % 2 == 0;
        let tail_positive = if degree_even {
            leading.is_positive()
        } else {
            leading.is_negative()
        };
        if !tail_positive {
            return IntervalCertificate::Unknown;
        }
    }

    // Probe a point inside the certified root-free region (the enclosure
    // bound is rational even for pi-valued endpoints).
    let probe_point = match (lower_bound, upper_bound) {
        (DefiniteBound::Finite(lower), _) => {
            lower.enclosure().1 + BigRational::from_integer(1.into())
        }
        (_, DefiniteBound::Finite(upper)) => {
            upper.enclosure().0 - BigRational::from_integer(1.into())
        }
        _ => BigRational::from_integer(0.into()),
    };
    if poly.eval(&probe_point).is_positive() {
        IntervalCertificate::Certified
    } else {
        IntervalCertificate::Unknown
    }
}

/// Rational enclosure of pi, tight enough for textbook bounds.
fn pi_enclosure() -> (BigRational, BigRational) {
    let denom = num_bigint::BigInt::from(100_000_000_000_000u64);
    (
        BigRational::new(
            num_bigint::BigInt::from(314_159_265_358_979u64),
            denom.clone(),
        ),
        BigRational::new(num_bigint::BigInt::from(314_159_265_358_980u64), denom),
    )
}

/// Zeros of cos (odd multiples of pi/2) or sin (integer multiples of pi)
/// against the closed rational interval, via the pi enclosure: every zero
/// enclosure disjoint from [low, high] certifies; an enclosure fully
/// inside is a pole; overlap with the boundary stays Unknown.
fn trig_nonzero_on_interval(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> Option<IntervalCertificate> {
    let builtin = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let inner_is_var = matches!(ctx.get(args[0]),
                Expr::Variable(sym) if ctx.sym_name(*sym) == var_name);
            if !inner_is_var {
                return None;
            }
            ctx.builtin_of(*fn_id)?
        }
        _ => return None,
    };
    let half = BigRational::new(1.into(), 2.into());
    // Zeros at multiplier * pi with multiplier in the arithmetic
    // progression below.
    let (start, step) = match builtin {
        cas_ast::BuiltinFn::Cos => (half, BigRational::from_integer(1.into())),
        cas_ast::BuiltinFn::Sin => (
            BigRational::from_integer(0.into()),
            BigRational::from_integer(1.into()),
        ),
        _ => return None,
    };
    let (pi_low, pi_high) = pi_enclosure();

    // Multiplier window covering the interval generously.
    let approx_low = &interval_low.enclosure().0 / &pi_high - BigRational::from_integer(2.into());
    let approx_high = &interval_high.enclosure().1 / &pi_low + BigRational::from_integer(2.into());
    let k_low = approx_low.floor().to_integer();
    let k_high = approx_high.ceil().to_integer();

    let mut k = k_low;
    while k <= k_high {
        let multiplier = &start + &step * BigRational::from_integer(k.clone());
        // The zero is exactly multiplier * pi: pi-pure, so comparisons
        // against pi-multiple bounds are exact rational comparisons.
        let zero = Endpoint::from_pi_multiple(multiplier);
        let before_interval = matches!(zero.try_cmp(interval_low), Some(std::cmp::Ordering::Less));
        let after_interval = matches!(
            zero.try_cmp(interval_high),
            Some(std::cmp::Ordering::Greater)
        );
        if !(before_interval || after_interval) {
            let strictly_inside =
                matches!(
                    zero.try_cmp(interval_low),
                    Some(std::cmp::Ordering::Greater)
                ) && matches!(zero.try_cmp(interval_high), Some(std::cmp::Ordering::Less));
            if strictly_inside {
                return Some(IntervalCertificate::Undefined);
            }
            if matches!(zero.try_cmp(interval_low), Some(std::cmp::Ordering::Equal)) {
                return Some(IntervalCertificate::BoundaryTouch {
                    lower: true,
                    upper: false,
                });
            }
            if matches!(zero.try_cmp(interval_high), Some(std::cmp::Ordering::Equal)) {
                return Some(IntervalCertificate::BoundaryTouch {
                    lower: false,
                    upper: true,
                });
            }
            return Some(IntervalCertificate::Unknown);
        }
        k += 1;
    }
    Some(IntervalCertificate::Certified)
}

fn nonzero_on_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    if let Some(certificate) =
        trig_nonzero_on_interval(ctx, expr, var_name, interval_low, interval_high)
    {
        return certificate;
    }
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
            match root_position(interval_low, interval_high, &root) {
                // A pole strictly inside the interval: divergent.
                Some(RootPosition::Inside) => IntervalCertificate::Undefined,
                Some(RootPosition::Outside) => IntervalCertificate::Certified,
                Some(RootPosition::AtLower) => IntervalCertificate::BoundaryTouch {
                    lower: true,
                    upper: false,
                },
                Some(RootPosition::AtUpper) => IntervalCertificate::BoundaryTouch {
                    lower: false,
                    upper: true,
                },
                None => IntervalCertificate::Unknown,
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

    pub(super) fn eval_definite(source: &str) -> Option<String> {
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
        // An endpoint pole is a boundary touch: the one-sided limit of
        // ln|x| reports the honest signed divergence instead.
        let endpoint = eval_definite("integrate(1/x, x, 0, 1)").expect("rewrite");
        assert_eq!(endpoint, "infinity");
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

#[cfg(test)]
mod improper_tests {
    use super::tests::eval_definite;

    #[test]
    fn improper_integrals_converge_via_limits() {
        assert!(eval_definite("integrate(1/x^2, x, 1, infinity)").is_some());
        assert!(eval_definite("integrate(1/(x^2+1), x, -infinity, infinity)").is_some());
    }

    #[test]
    fn improper_divergence_reports_infinity() {
        let result = eval_definite("integrate(x^2, x, 0, infinity)").expect("rewrite");
        assert_eq!(result, "infinity");
    }

    #[test]
    fn improper_pole_inside_unbounded_interval_is_undefined() {
        let result = eval_definite("integrate(1/x^2, x, -1, infinity)").expect("rewrite");
        assert_eq!(result, "undefined");
    }

    #[test]
    fn logarithmic_divergence_reports_signed_infinity() {
        // F = ln|x| now resolves at both infinities, so the divergent
        // half-line integrals report their honest signed value.
        assert_eq!(
            eval_definite("integrate(1/x, x, 1, infinity)").as_deref(),
            Some("infinity")
        );
        assert_eq!(
            eval_definite("integrate(1/x, x, -infinity, -1)").as_deref(),
            Some("-infinity")
        );
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::tests::eval_definite;

    #[test]
    fn cofactor_route_antiderivatives_evaluate_definitely() {
        assert!(eval_definite("integrate(x/sqrt(x^2+1), x, 0, 1)").is_some());
    }

    #[test]
    fn polynomial_positivity_certifies_away_from_roots() {
        // Positive(1-x^2) with roots +-1 strictly outside [0, 1/2].
        assert!(eval_definite("integrate(x/sqrt(1-x^2), x, 0, 1/2)").is_some());
        // Root on the boundary: now a boundary touch that converges via
        // the one-sided limit of the antiderivative.
        assert!(eval_definite("integrate(x/sqrt(1-x^2), x, 0, 1)").is_some());
    }

    #[test]
    fn trig_nonzero_certificate_locates_cosine_zeros() {
        // [0, 1] is inside (-pi/2, pi/2): certified. (The raw unit
        // context sees the simplifier-normalized 1/cos^2 form.)
        assert!(eval_definite("integrate(1/cos(x)^2, x, 0, 1)").is_some());
        // pi/2 inside [0, 2]: pole, undefined.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, 2)").as_deref(),
            Some("undefined")
        );
        // Unbounded interval always contains cosine zeros.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, infinity)").as_deref(),
            Some("undefined")
        );
    }
}

#[cfg(test)]
mod pi_bound_tests {
    use super::tests::eval_definite;

    #[test]
    fn pi_multiple_bounds_certify_exactly_against_trig_zeros() {
        // [0, pi/4] inside (-pi/2, pi/2): exact pi-pure comparison.
        assert!(eval_definite("integrate(1/cos(x)^2, x, 0, pi/4)").is_some());
        // pi/2 strictly inside [0, 3pi/4]: undefined, also exactly.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, 3*pi/4)").as_deref(),
            Some("undefined")
        );
    }

    #[test]
    fn pi_endpoints_compose_with_polynomial_certificates() {
        // Positive(x^2+1)-style risks certify against the pi enclosure.
        assert!(eval_definite("integrate(x/(x^2+1), x, 0, pi)").is_some());
    }

    #[test]
    fn mixed_undecidable_endpoints_stay_residual() {
        // A bound inside the pi enclosure of pi/2 cannot be decided and
        // must refuse rather than guess.
        assert!(
            eval_definite("integrate(1/cos(x)^2, x, 0, 15707963267948966/10000000000000000)")
                .is_none()
        );
    }
}

#[cfg(test)]
mod boundary_touch_tests {
    use super::tests::eval_definite;

    #[test]
    fn bounded_inverse_trig_integrands_certify_on_the_unit_interval() {
        // arcsin/arccos are defined on the CLOSED unit interval: the
        // endpoint touch at x = 1 certifies, out-of-domain refuses.
        assert!(eval_definite("integrate(x*arcsin(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(arccos(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(arcsin(x), x, -1, 1)").is_some());
        assert!(eval_definite("integrate(arcsin(x), x, 0, 2)").is_none());
        assert!(eval_definite("integrate(x*arcsin(x), x, -2, 0)").is_none());
    }

    #[test]
    fn boundary_touch_fractional_power_atoms_evaluate() {
        // Positive fractional powers touching x = 0: the antiderivative's
        // one-sided limit resolves via the power atom + product/scale rules.
        assert!(eval_definite("integrate(x^(1/2), x, 0, 4)").is_some());
        assert!(eval_definite("integrate(x^(1/3), x, 0, 8)").is_some());
        assert!(eval_definite("integrate(x^(3/2), x, 0, 1)").is_some());
    }

    #[test]
    fn boundary_convergent_improper_integrals_evaluate() {
        // The textbook trio: F's one-sided limit at the touched endpoint.
        assert!(eval_definite("integrate(ln(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(x*(1-x^2)^(-1/2), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(x^(-1/2), x, 0, 1)").is_some());
    }

    #[test]
    fn boundary_divergence_reports_signed_infinity() {
        assert_eq!(
            eval_definite("integrate(1/x^2, x, 0, 1)").as_deref(),
            Some("infinity")
        );
    }

    #[test]
    fn interior_roots_still_refuse_or_diverge() {
        assert_eq!(
            eval_definite("integrate(1/x, x, -1, 1)").as_deref(),
            Some("undefined")
        );
    }
}
