use crate::calculus_domain_support::{
    known_positive_constant_exceeds_one, real_domain_is_empty_for_static_expr,
};
use crate::infinity_support::{mk_infinity, InfSign};
use crate::limit_types::{
    Approach, FiniteLimitSide, LimitEvalOutcome, LimitOptions, PreSimplifyMode,
};
use crate::perfect_square_support::rational_sqrt;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::polynomial::Polynomial;
use crate::root_forms::{extract_square_root_base, rational_cbrt_exact, rational_nth_root};
use crate::trig_eval_table_support::lookup_trig_or_inverse;
use crate::trig_values::TrigValue;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Check if an expression depends on a specific variable id.
///
/// Uses iterative traversal to avoid recursion limits on deep trees.
pub fn depends_on(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];

    while let Some(current) = stack.pop() {
        if current == var {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
            Expr::Matrix { .. } | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Parse a power expression with integer exponent.
///
/// Returns `(base, n)` if `expr` is `base^n` where `n` is an integer literal.
pub fn parse_pow_int(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let n = crate::expr_extract::extract_i64_integer(ctx, *exp)?;
            Some((*base, n))
        }
        _ => None,
    }
}

/// Create a residual limit expression: `limit(expr, var, approach_symbol)`.
pub fn mk_limit(ctx: &mut Context, expr: ExprId, var: ExprId, approach: InfSign) -> ExprId {
    let approach_sym = match approach {
        InfSign::Pos => ctx.add(Expr::Constant(Constant::Infinity)),
        InfSign::Neg => {
            let inf = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(inf))
        }
    };
    ctx.call("limit", vec![expr, var, approach_sym])
}

/// Create a residual limit expression from a typed approach.
pub fn mk_limit_for_approach(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> ExprId {
    match approach {
        Approach::PosInfinity => mk_limit(ctx, expr, var, InfSign::Pos),
        Approach::NegInfinity => mk_limit(ctx, expr, var, InfSign::Neg),
        Approach::Finite(point) => ctx.call("limit", vec![expr, var, point]),
        Approach::FiniteOneSided(point, side) => {
            let side_marker = ctx.var(side.marker());
            ctx.call("limit", vec![expr, var, point, side_marker])
        }
    }
}

/// Determine resulting infinity sign from approach sign and exponent parity.
pub fn limit_sign(approach: InfSign, power: i64) -> InfSign {
    match approach {
        InfSign::Pos => InfSign::Pos,
        InfSign::Neg => {
            if power % 2 == 0 {
                InfSign::Pos // (-∞)^even = +∞
            } else {
                InfSign::Neg // (-∞)^odd = -∞
            }
        }
    }
}

/// Create infinity with appropriate sign.
pub fn mk_inf(ctx: &mut Context, sign: InfSign) -> ExprId {
    mk_infinity(ctx, sign)
}

/// Rule 1: Constant - lim c = c (if `expr` doesn't depend on `var`).
pub fn apply_constant_rule(ctx: &Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !depends_on(ctx, expr, var) {
        Some(expr)
    } else {
        None
    }
}

const LIMIT_STATIC_DOMAIN_PROOF_DEPTH: usize = 8;
const LIMIT_STATIC_DOMAIN_SCAN_DEPTH: usize = 24;

fn apply_static_empty_real_domain_rule(
    ctx: &mut Context,
    expr: ExprId,
    _var: ExprId,
) -> Option<ExprId> {
    if !real_domain_is_empty_for_static_expr(
        ctx,
        expr,
        LIMIT_STATIC_DOMAIN_PROOF_DEPTH,
        LIMIT_STATIC_DOMAIN_SCAN_DEPTH,
    ) {
        return None;
    }

    Some(ctx.add(Expr::Constant(Constant::Undefined)))
}

fn apply_finite_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();

    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    let value = poly.eval(&point_value);
    Some(ctx.add(Expr::Number(value)))
}

fn apply_finite_rational_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

/// Finite-point 0/0 limit of `(scale*sqrt(a x + b) + k) / den(x)` where
/// the radical numerator vanishes at the point (so the form is 0/0):
/// rationalize by the conjugate. (scale sqrt + k)(scale sqrt - k) =
/// scale^2 (a x + b) - k^2 is a polynomial, so the quotient becomes
/// [scale^2(ax+b) - k^2] / [den(x) (scale sqrt + k - wait, conjugate)].
/// The polynomial part is a removable rational hole and the conjugate
/// is continuous (nonzero) at the point. Covers (sqrt(x)-2)/(x-4)=1/4
/// and (sqrt(x+1)-2)/(x-3)=1/4.
fn apply_finite_radical_conjugate_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point).clone() else {
        return None;
    };
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Numerator = scale * sqrt(linear) + const (const var-free numeric).
    let (scale, radicand, additive) = split_scaled_sqrt_plus_constant(ctx, num, &var_name)?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, &var_name).ok()?;
    if radicand_poly.degree() != 1 {
        return None;
    }
    let radicand_at_point = radicand_poly.eval(&point_value);
    let sqrt_at_point = rational_sqrt(&radicand_at_point)?;
    // 0/0 numerator: scale*sqrt(point) + const == 0.
    if (&scale * &sqrt_at_point + &additive) != BigRational::from_integer(BigInt::from(0)) {
        return None;
    }

    let denominator_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    // 0/0 denominator.
    if !denominator_poly.eval(&point_value).is_zero() {
        return None;
    }

    // num2 = scale^2 (a x + b) - const^2 (the rationalized numerator).
    let scale_sq = &scale * &scale;
    let num2 = Polynomial::new(
        vec![
            &scale_sq * radicand_poly.coeffs.first()? - &additive * &additive,
            &scale_sq * radicand_poly.coeffs.get(1)?,
        ],
        var_name.clone(),
    );
    // Removable rational part num2/den.
    let rational_part = finite_rational_polynomial_value(&num2, &denominator_poly, &point_value)?;
    // Conjugate scale*sqrt - const, continuous and nonzero at the point.
    let conjugate_value = &scale * &sqrt_at_point - &additive;
    if conjugate_value.is_zero() {
        return None;
    }
    Some(ctx.add(Expr::Number(rational_part / conjugate_value)))
}

/// 0/0 finite-point limits of `(s1 sqrt(L1) + s2 sqrt(L2)) / den` with two linear
/// radicands, resolved by the conjugate `s1 sqrt(L1) - s2 sqrt(L2)`: the product
/// is the polynomial `s1^2 L1 - s2^2 L2`, so the numerator's radical cancels and
/// the limit is `[s1^2 L1 - s2^2 L2 over den]_pt / (s1 sqrt(L1(pt)) - s2 sqrt(L2(pt)))`.
/// The single-sqrt-plus-constant case is owned by the rule above; this one is its
/// sqrt-MINUS-sqrt complement. Resolves `(sqrt(1+x)-sqrt(1-x))/x -> 1`,
/// `(sqrt(4+x)-sqrt(4-x))/x -> 1/4`. Gated to a genuine 0/0 with rational radical
/// values at the point and a nonzero conjugate; irrational sqrt values and
/// degenerate conjugates decline.
fn apply_finite_radical_difference_conjugate_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point).clone() else {
        return None;
    };
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (s1, l1, s2, l2) = split_scaled_sqrt_difference(ctx, num, &var_name)?;
    let l1_poly = Polynomial::from_expr(ctx, l1, &var_name).ok()?;
    let l2_poly = Polynomial::from_expr(ctx, l2, &var_name).ok()?;
    if l1_poly.degree() != 1 || l2_poly.degree() != 1 {
        return None;
    }

    let sqrt1 = rational_sqrt(&l1_poly.eval(&point_value))?;
    let sqrt2 = rational_sqrt(&l2_poly.eval(&point_value))?;
    // 0/0 numerator: s1*sqrt(L1(pt)) + s2*sqrt(L2(pt)) == 0.
    if (&s1 * &sqrt1 + &s2 * &sqrt2) != BigRational::from_integer(BigInt::from(0)) {
        return None;
    }

    let denominator_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    if !denominator_poly.eval(&point_value).is_zero() {
        return None;
    }

    // num2 = s1^2 L1 - s2^2 L2 (the rationalized numerator, a linear polynomial).
    let s1_sq = &s1 * &s1;
    let s2_sq = &s2 * &s2;
    let num2 = Polynomial::new(
        vec![
            &s1_sq * l1_poly.coeffs.first()? - &s2_sq * l2_poly.coeffs.first()?,
            &s1_sq * l1_poly.coeffs.get(1)? - &s2_sq * l2_poly.coeffs.get(1)?,
        ],
        var_name.clone(),
    );
    let rational_part = finite_rational_polynomial_value(&num2, &denominator_poly, &point_value)?;
    // Conjugate s1*sqrt(L1) - s2*sqrt(L2), continuous and nonzero at the point.
    let conjugate_value = &s1 * &sqrt1 - &s2 * &sqrt2;
    if conjugate_value.is_zero() {
        return None;
    }
    Some(ctx.add(Expr::Number(rational_part / conjugate_value)))
}

/// Flatten an additive tree into signed leaf terms (handles Add, Sub,
/// and unary Neg).
fn collect_signed_add_terms(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            collect_signed_add_terms(ctx, l, positive, terms);
            collect_signed_add_terms(ctx, r, positive, terms);
        }
        Expr::Sub(l, r) => {
            collect_signed_add_terms(ctx, l, positive, terms);
            collect_signed_add_terms(ctx, r, !positive, terms);
        }
        Expr::Neg(inner) => collect_signed_add_terms(ctx, inner, !positive, terms),
        _ => terms.push((expr, positive)),
    }
}

/// Decompose `scale * sqrt(radicand) + const` (in any additive order)
/// into (scale, radicand, const). Exactly one square-root term; the
/// rest must be a single var-free numeric constant.
fn split_scaled_sqrt_plus_constant(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, expr, true, &mut terms);
    let mut sqrt_part: Option<(BigRational, ExprId)> = None;
    let mut constant = BigRational::from_integer(BigInt::from(0));
    for (term, positive) in terms {
        if let Some((scale, radicand)) = scaled_square_root_base(ctx, term) {
            if sqrt_part.is_some() {
                return None;
            }
            let signed = if positive { scale } else { -scale };
            sqrt_part = Some((signed, radicand));
        } else {
            let value = numeric_limit_value(ctx, term)?;
            if positive {
                constant += value;
            } else {
                constant -= value;
            }
        }
    }
    let (scale, radicand) = sqrt_part?;
    // The radicand must actually depend on the variable.
    if !crate::expr_predicates::contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    Some((scale, radicand, constant))
}

/// Split `s1*sqrt(L1) + s2*sqrt(L2)` into `(s1, L1, s2, L2)`: EXACTLY two scaled
/// square-root terms (signs folded into the scales), each radicand depending on
/// the variable. Any non-sqrt term (a constant, a bare polynomial) declines, so
/// this is the sqrt-MINUS-sqrt complement of `split_scaled_sqrt_plus_constant`.
fn split_scaled_sqrt_difference(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BigRational, ExprId)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, expr, true, &mut terms);
    if terms.len() != 2 {
        return None;
    }
    let mut sqrts: Vec<(BigRational, ExprId)> = Vec::new();
    for (term, positive) in terms {
        let (scale, radicand) = scaled_square_root_base(ctx, term)?;
        let signed = if positive { scale } else { -scale };
        sqrts.push((signed, radicand));
    }
    let (s1, l1) = sqrts[0].clone();
    let (s2, l2) = sqrts[1].clone();
    if !crate::expr_predicates::contains_named_var(ctx, l1, var_name)
        || !crate::expr_predicates::contains_named_var(ctx, l2, var_name)
    {
        return None;
    }
    Some((s1, l1, s2, l2))
}

fn finite_rational_polynomial_value(
    numerator: &Polynomial,
    denominator: &Polynomial,
    point: &BigRational,
) -> Option<BigRational> {
    let mut numerator = numerator.clone();
    let mut denominator = denominator.clone();
    let max_derivative_steps = numerator.degree().max(denominator.degree()) + 1;

    for _ in 0..=max_derivative_steps {
        let numerator_value = numerator.eval(point);
        let denominator_value = denominator.eval(point);
        if !denominator_value.is_zero() {
            return Some(numerator_value / denominator_value);
        }
        if !numerator_value.is_zero() || (numerator.is_zero() && denominator.is_zero()) {
            return None;
        }

        numerator = numerator.derivative();
        denominator = denominator.derivative();
    }

    None
}

fn finite_polynomial_local_order_and_derivative(
    polynomial: &Polynomial,
    point: &BigRational,
) -> Option<(usize, BigRational)> {
    let mut current = polynomial.clone();
    for order in 0..=polynomial.degree() {
        let value = current.eval(point);
        if !value.is_zero() {
            return Some((order, value));
        }
        current = current.derivative();
    }
    None
}

fn finite_local_tail_sign(
    derivative_value: &BigRational,
    order: usize,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if derivative_value.is_zero() {
        return None;
    }
    let positive = derivative_value.is_positive()
        == (side == FiniteLimitSide::Right || order.is_multiple_of(2));
    Some(if positive { InfSign::Pos } else { InfSign::Neg })
}

fn finite_endpoint_argument_zero_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if let Ok(polynomial) = Polynomial::from_expr(ctx, expr, var_name) {
        let (order, derivative) =
            finite_polynomial_local_order_and_derivative(&polynomial, point_value)?;
        if order == 0 {
            return None;
        }
        return finite_local_tail_sign(&derivative, order, side);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(point_value);
    if denominator_value.is_zero() {
        return None;
    }

    let (numerator_order, numerator_derivative) =
        finite_polynomial_local_order_and_derivative(&numerator, point_value)?;
    if numerator_order == 0 {
        return None;
    }

    let numerator_tail = finite_local_tail_sign(&numerator_derivative, numerator_order, side)?;
    let denominator_tail = if denominator_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if numerator_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_local_tail_positive_on_both_sides(
    derivative_value: &BigRational,
    order: usize,
) -> Option<bool> {
    Some(
        finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Left)? == InfSign::Pos
            && finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Right)?
                == InfSign::Pos,
    )
}

fn finite_local_tail_negative_on_both_sides(
    derivative_value: &BigRational,
    order: usize,
) -> Option<bool> {
    Some(
        finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Left)? == InfSign::Neg
            && finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Right)?
                == InfSign::Neg,
    )
}

fn finite_endpoint_argument_zero_tail_positive_on_both_sides(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
) -> Option<bool> {
    Some(
        finite_endpoint_argument_zero_tail_sign(
            ctx,
            expr,
            var_name,
            point_value,
            FiniteLimitSide::Left,
        )? == InfSign::Pos
            && finite_endpoint_argument_zero_tail_sign(
                ctx,
                expr,
                var_name,
                point_value,
                FiniteLimitSide::Right,
            )? == InfSign::Pos,
    )
}

fn finite_endpoint_argument_zero_tail_negative_on_both_sides(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
) -> Option<bool> {
    Some(
        finite_endpoint_argument_zero_tail_sign(
            ctx,
            expr,
            var_name,
            point_value,
            FiniteLimitSide::Left,
        )? == InfSign::Neg
            && finite_endpoint_argument_zero_tail_sign(
                ctx,
                expr,
                var_name,
                point_value,
                FiniteLimitSide::Right,
            )? == InfSign::Neg,
    )
}

fn finite_endpoint_unit_base_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if let Ok(polynomial) = Polynomial::from_expr(ctx, expr, var_name) {
        let unit_gap = polynomial.sub(&Polynomial::one(var_name.to_string()));
        let (order, derivative) =
            finite_polynomial_local_order_and_derivative(&unit_gap, point_value)?;
        if order == 0 {
            return None;
        }
        return finite_local_tail_sign(&derivative, order, side);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(point_value);
    if denominator_value.is_zero() {
        return None;
    }

    let unit_gap_numerator = numerator.sub(&denominator);
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&unit_gap_numerator, point_value)?;
    if gap_order == 0 {
        return None;
    }

    let gap_tail = finite_local_tail_sign(&gap_derivative, gap_order, side)?;
    let denominator_tail = if denominator_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if gap_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn apply_finite_one_sided_rational_polynomial_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let (den_order, den_derivative) =
        finite_polynomial_local_order_and_derivative(&denominator, &point_value)?;
    if den_order == 0 {
        return None;
    }

    let Some((num_order, num_derivative)) =
        finite_polynomial_local_order_and_derivative(&numerator, &point_value)
    else {
        return Some(ctx.num(0));
    };
    if num_order >= den_order {
        return None;
    }

    let num_sign = finite_local_tail_sign(&num_derivative, num_order, side)?;
    let den_sign = finite_local_tail_sign(&den_derivative, den_order, side)?;
    Some(signed_abs_ratio_infinity(ctx, num_sign, den_sign))
}

fn apply_finite_bilateral_rational_polynomial_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_rational_polynomial_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_rational_polynomial_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn matching_finite_bilateral_one_sided_result(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, left),
        infinity_sign_of_expr(ctx, right),
    ) {
        (Some(left_sign), Some(right_sign)) if left_sign == right_sign => return Some(left),
        _ => {}
    }

    match (ctx.get(left), ctx.get(right)) {
        (Expr::Number(left_value), Expr::Number(right_value)) if left_value == right_value => {
            Some(left)
        }
        _ if left == right => Some(left),
        _ => None,
    }
}

fn apply_finite_one_sided_abs_polynomial_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, num)?;
    let abs_arg_poly = Polynomial::from_expr(ctx, abs_arg, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let (abs_order, abs_derivative) =
        finite_polynomial_local_order_and_derivative(&abs_arg_poly, &point_value)?;
    let (den_order, den_derivative) =
        finite_polynomial_local_order_and_derivative(&denominator, &point_value)?;
    if den_order == 0 {
        return None;
    }

    let numerator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let den_tail = finite_local_tail_sign(&den_derivative, den_order, side)?;

    if abs_order < den_order {
        return Some(signed_abs_ratio_infinity(ctx, numerator_tail, den_tail));
    }
    if abs_order > den_order {
        return Some(ctx.num(0));
    }

    let magnitude = abs_scale.abs() * abs_derivative.abs() / den_derivative.abs();
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        den_tail,
    ))
}

fn apply_finite_bilateral_abs_polynomial_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_abs_polynomial_ratio_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_abs_polynomial_ratio_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn apply_finite_one_sided_sign_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Sign) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let (order, derivative) =
        finite_polynomial_local_order_and_derivative(&argument, &point_value)?;
    let tail_sign = finite_local_tail_sign(&derivative, order, side)?;
    Some(signed_unit_limit(ctx, tail_sign))
}

fn apply_finite_bilateral_sign_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_sign_polynomial_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_sign_polynomial_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn scaled_sine_argument(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Sin) => {
            Some((BigRational::one(), args[0]))
        }
        Expr::Neg(inner) => {
            let (scale, argument) = scaled_sine_argument(ctx, inner)?;
            Some((-scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument) = scaled_sine_argument(ctx, rhs)?;
                return Some((scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument) = scaled_sine_argument(ctx, lhs)?;
                return Some((scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_zero_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && (ctx.is_builtin(fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(fn_id, BuiltinFn::Cos)) =>
        {
            let builtin = if ctx.is_builtin(fn_id, BuiltinFn::Sin) {
                BuiltinFn::Sin
            } else {
                BuiltinFn::Cos
            };
            Some((builtin, BigRational::one(), args[0]))
        }
        Expr::Neg(inner) => {
            let (builtin, scale, argument) = scaled_trig_zero_argument(ctx, inner)?;
            Some((builtin, -scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (builtin, inner_scale, argument) = scaled_trig_zero_argument(ctx, rhs)?;
                return Some((builtin, scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (builtin, inner_scale, argument) = scaled_trig_zero_argument(ctx, lhs)?;
                return Some((builtin, scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_zero_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BigRational, ExprId, usize)> {
    if let Some((builtin, scale, argument)) = scaled_trig_zero_argument(ctx, expr) {
        return Some((builtin, scale, argument, 1));
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (builtin, scale, argument) = scaled_trig_zero_argument(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((builtin, power_scale, argument, exponent))
}

fn scaled_reciprocal_trig_zero_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(fn_id, BuiltinFn::Csc) {
                Some((BigRational::one(), BuiltinFn::Sin, args[0]))
            } else if ctx.is_builtin(fn_id, BuiltinFn::Sec) {
                Some((BigRational::one(), BuiltinFn::Cos, args[0]))
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let (scale, builtin, argument) = scaled_reciprocal_trig_zero_argument(ctx, inner)?;
            Some((-scale, builtin, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, builtin, argument) =
                    scaled_reciprocal_trig_zero_argument(ctx, rhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, builtin, argument) =
                    scaled_reciprocal_trig_zero_argument(ctx, lhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_reciprocal_trig_zero_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, usize)> {
    if let Some((scale, builtin, argument)) = scaled_reciprocal_trig_zero_argument(ctx, expr) {
        if scale.is_zero() {
            return None;
        }
        return Some((scale, builtin, argument, 1));
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, builtin, argument, exponent) =
            scaled_reciprocal_trig_zero_power_argument(ctx, inner)?;
        return Some((-scale, builtin, argument, exponent));
    }
    if let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() {
        if let Some(scale) = constant_rational_value(ctx, lhs) {
            let (inner_scale, builtin, argument, exponent) =
                scaled_reciprocal_trig_zero_power_argument(ctx, rhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((scale, builtin, argument, exponent));
        }
        if let Some(scale) = constant_rational_value(ctx, rhs) {
            let (inner_scale, builtin, argument, exponent) =
                scaled_reciprocal_trig_zero_power_argument(ctx, lhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((scale, builtin, argument, exponent));
        }
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (scale, builtin, argument) = scaled_reciprocal_trig_zero_argument(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((power_scale, builtin, argument, exponent))
}

fn scaled_tan_cot_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(fn_id, BuiltinFn::Tan) {
                Some((BigRational::one(), BuiltinFn::Tan, args[0]))
            } else if ctx.is_builtin(fn_id, BuiltinFn::Cot) {
                Some((BigRational::one(), BuiltinFn::Cot, args[0]))
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let (scale, builtin, argument) = scaled_tan_cot_argument(ctx, inner)?;
            Some((-scale, builtin, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, builtin, argument) = scaled_tan_cot_argument(ctx, rhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, builtin, argument) = scaled_tan_cot_argument(ctx, lhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_ratio_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, BuiltinFn, ExprId, usize)> {
    if let Some((
        scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
    )) = trig_ratio_source_components(ctx, expr)
    {
        if scale.is_zero() {
            return None;
        }
        return Some((
            scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            1,
        ));
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (
            scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            exponent,
        ) = scaled_trig_ratio_power_argument(ctx, inner)?;
        return Some((
            -scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            exponent,
        ));
    }
    if let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() {
        if let Some(scale) = constant_rational_value(ctx, lhs) {
            let (
                inner_scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ) = scaled_trig_ratio_power_argument(ctx, rhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((
                scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ));
        }
        if let Some(scale) = constant_rational_value(ctx, rhs) {
            let (
                inner_scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ) = scaled_trig_ratio_power_argument(ctx, lhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((
                scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ));
        }
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (scale, numerator_builtin, numerator_argument, denominator_builtin, denominator_argument) =
        trig_ratio_source_components(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((
        power_scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
        exponent,
    ))
}

fn finite_trig_power_pole_components(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, BigRational, ExprId, usize)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let numerator = constant_rational_value(ctx, num)?;
        if numerator.is_zero() {
            return None;
        }

        let (builtin, den_scale, argument, exponent) = scaled_trig_zero_power_argument(ctx, den)?;
        return Some((numerator, builtin, den_scale, argument, exponent));
    }

    let (numerator, builtin, argument, exponent) =
        scaled_reciprocal_trig_zero_power_argument(ctx, expr)?;
    Some((numerator, builtin, BigRational::one(), argument, exponent))
}

fn tan_cot_ratio_builtins(source_builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match source_builtin {
        BuiltinFn::Tan => Some((BuiltinFn::Sin, BuiltinFn::Cos)),
        BuiltinFn::Cot => Some((BuiltinFn::Cos, BuiltinFn::Sin)),
        _ => None,
    }
}

fn trig_ratio_source_components(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, BuiltinFn, ExprId)> {
    if let Some((scale, source_builtin, argument)) = scaled_tan_cot_argument(ctx, expr) {
        let (numerator_builtin, denominator_builtin) = tan_cot_ratio_builtins(source_builtin)?;
        return Some((
            scale,
            numerator_builtin,
            argument,
            denominator_builtin,
            argument,
        ));
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let (numerator_builtin, numerator_scale, numerator_argument) =
        scaled_trig_zero_argument(ctx, numerator)?;
    let (denominator_builtin, denominator_scale, denominator_argument) =
        scaled_trig_zero_argument(ctx, denominator)?;
    if denominator_scale.is_zero() {
        return None;
    }

    Some((
        numerator_scale / denominator_scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
    ))
}

fn rational_tail_sign(value: &BigRational) -> InfSign {
    if value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

fn multiply_tail_signs(lhs: InfSign, rhs: InfSign) -> InfSign {
    if lhs == rhs {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

fn structurally_equal_expr(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    lhs == rhs || cas_ast::ordering::compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
}

fn finite_argument_tail_after_limit(
    ctx: &mut Context,
    argument: ExprId,
    argument_limit: ExprId,
) -> ExprId {
    match ctx.get(argument).clone() {
        Expr::Add(lhs, rhs) => {
            if structurally_equal_expr(ctx, lhs, argument_limit) {
                return rhs;
            }
            if structurally_equal_expr(ctx, rhs, argument_limit) {
                return lhs;
            }
        }
        Expr::Sub(lhs, rhs) => {
            if structurally_equal_expr(ctx, lhs, argument_limit) {
                return ctx.add(Expr::Neg(rhs));
            }
            if structurally_equal_expr(ctx, rhs, argument_limit) {
                return lhs;
            }
        }
        _ => {}
    }

    let neg_limit = ctx.add(Expr::Neg(argument_limit));
    ctx.add(Expr::Add(argument, neg_limit))
}

fn finite_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    var: ExprId,
    point: ExprId,
    point_value: Option<&BigRational>,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();

    match builtin {
        BuiltinFn::Sin => {
            let Some(point_value) = point_value else {
                return finite_direct_variable_special_trig_zero_tail_sign(
                    ctx, builtin, argument, var, point, side,
                );
            };
            let argument = match Polynomial::from_expr(ctx, argument, &var_name) {
                Ok(argument) => argument,
                Err(_) => {
                    let local = FiniteTrigZeroTailLocal {
                        var,
                        point,
                        point_value,
                        side,
                        var_name: &var_name,
                    };
                    return finite_table_trig_zero_tail_sign(ctx, builtin, argument, &local);
                }
            };
            let (argument_order, argument_derivative) =
                finite_polynomial_local_order_and_derivative(&argument, point_value)?;
            if argument_order == 0 {
                return None;
            }
            finite_local_tail_sign(&argument_derivative, argument_order, side)
        }
        BuiltinFn::Cos => {
            if point_value.is_none() {
                if let Some(tail_sign) = finite_direct_variable_special_trig_zero_tail_sign(
                    ctx, builtin, argument, var, point, side,
                ) {
                    return Some(tail_sign);
                }
            }

            let point_value = point_value?;
            let local = FiniteTrigZeroTailLocal {
                var,
                point,
                point_value,
                side,
                var_name: &var_name,
            };
            finite_table_trig_zero_tail_sign(ctx, builtin, argument, &local)
        }
        _ => None,
    }
}

struct FiniteTrigZeroTailLocal<'a> {
    var: ExprId,
    point: ExprId,
    point_value: &'a BigRational,
    side: FiniteLimitSide,
    var_name: &'a str,
}

fn finite_table_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    local: &FiniteTrigZeroTailLocal<'_>,
) -> Option<InfSign> {
    let argument_limit = try_limit_rules_at_finite(ctx, argument, local.var, local.point)?;
    let value_at_limit = finite_total_real_unary_trig_table_result(ctx, builtin, argument_limit)?;
    if !constant_rational_value(ctx, value_at_limit)?.is_zero() {
        return None;
    }

    let derivative_factor = match builtin {
        BuiltinFn::Sin => {
            let cos_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Cos, argument_limit)?;
            constant_rational_value(ctx, cos_at_limit)?
        }
        BuiltinFn::Cos => {
            let sin_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Sin, argument_limit)?;
            -constant_rational_value(ctx, sin_at_limit)?
        }
        _ => return None,
    };
    if derivative_factor.is_zero() {
        return None;
    }

    let tail_expr = finite_argument_tail_after_limit(ctx, argument, argument_limit);
    let tail = Polynomial::from_expr(ctx, tail_expr, local.var_name).ok()?;
    let (tail_order, tail_derivative) =
        finite_polynomial_local_order_and_derivative(&tail, local.point_value)?;
    if tail_order == 0 {
        return None;
    }

    let tail_sign = finite_local_tail_sign(&tail_derivative, tail_order, local.side)?;
    let derivative_sign = if derivative_factor.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if derivative_sign == tail_sign {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_special_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if !structurally_equal_expr(ctx, argument, var) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, argument, var, point)?;
    let derivative_sign =
        finite_direct_variable_trig_zero_derivative_sign(ctx, builtin, argument_limit)?;
    let point_tail_sign = match side {
        FiniteLimitSide::Left => InfSign::Neg,
        FiniteLimitSide::Right => InfSign::Pos,
    };
    Some(if derivative_sign == point_tail_sign {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_trig_zero_derivative_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<InfSign> {
    if let Some(sign) =
        finite_direct_variable_rational_pi_trig_zero_derivative_sign(ctx, builtin, argument_limit)
    {
        return Some(sign);
    }

    let value_at_limit = finite_total_real_unary_trig_table_result(ctx, builtin, argument_limit)?;
    if !constant_rational_value(ctx, value_at_limit)?.is_zero() {
        return None;
    }

    let derivative_value = match builtin {
        BuiltinFn::Sin => {
            let cos_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Cos, argument_limit)?;
            constant_rational_value(ctx, cos_at_limit)?
        }
        BuiltinFn::Cos => {
            let sin_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Sin, argument_limit)?;
            -constant_rational_value(ctx, sin_at_limit)?
        }
        _ => return None,
    };
    if derivative_value.is_zero() {
        return None;
    }

    Some(if derivative_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_rational_pi_trig_zero_derivative_sign(
    ctx: &Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<InfSign> {
    let k = extract_rational_pi_multiple(ctx, argument_limit)?;
    match builtin {
        BuiltinFn::Sin if k.is_integer() => integer_parity_cos_sign(&k),
        BuiltinFn::Cos => {
            let sin_sign = half_integer_sin_sign(&k)?;
            Some(match sin_sign {
                InfSign::Pos => InfSign::Neg,
                InfSign::Neg => InfSign::Pos,
            })
        }
        _ => None,
    }
}

fn integer_parity_cos_sign(k: &BigRational) -> Option<InfSign> {
    if !k.is_integer() {
        return None;
    }
    let n = k.to_integer();
    let two = BigInt::from(2);
    if (&n % &two).is_zero() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn half_integer_sin_sign(k: &BigRational) -> Option<InfSign> {
    if k.denom() != &BigInt::from(2) {
        return None;
    }

    let four = BigInt::from(4);
    let mut rem = k.numer() % &four;
    if rem.is_negative() {
        rem += &four;
    }

    if rem == BigInt::from(1) {
        Some(InfSign::Pos)
    } else if rem == BigInt::from(3) {
        Some(InfSign::Neg)
    } else {
        None
    }
}

fn apply_finite_one_sided_trig_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let point_value = match ctx.get(point) {
        Expr::Number(point_value) => Some(point_value.clone()),
        _ => None,
    };

    let (numerator, builtin, den_scale, argument, exponent) =
        finite_trig_power_pole_components(ctx, expr)?;
    let argument_tail = finite_trig_zero_tail_sign(
        ctx,
        builtin,
        argument,
        var,
        point,
        point_value.as_ref(),
        side,
    )?;
    let scale_tail = if den_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let den_tail = if exponent.is_multiple_of(2) || scale_tail == argument_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let numerator_tail = if numerator.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(signed_abs_ratio_infinity(ctx, numerator_tail, den_tail))
}

fn apply_finite_bilateral_trig_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_trig_power_pole_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_trig_power_pole_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn apply_finite_one_sided_trig_ratio_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let point_value = match ctx.get(point) {
        Expr::Number(point_value) => Some(point_value.clone()),
        _ => None,
    };

    let (
        scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
        exponent,
    ) = scaled_trig_ratio_power_argument(ctx, expr)?;
    let numerator_argument_limit = try_limit_rules_at_finite(ctx, numerator_argument, var, point)?;
    let denominator_argument_limit =
        try_limit_rules_at_finite(ctx, denominator_argument, var, point)?;

    let denominator_at_limit = finite_total_real_unary_trig_table_result(
        ctx,
        denominator_builtin,
        denominator_argument_limit,
    )?;
    if !constant_rational_value(ctx, denominator_at_limit)?.is_zero() {
        return None;
    }

    let numerator_at_limit = finite_total_real_unary_trig_table_result(
        ctx,
        numerator_builtin,
        numerator_argument_limit,
    )?;
    let numerator_value = constant_rational_value(ctx, numerator_at_limit)?;
    if numerator_value.is_zero() {
        return None;
    }

    let denominator_tail = finite_trig_zero_tail_sign(
        ctx,
        denominator_builtin,
        denominator_argument,
        var,
        point,
        point_value.as_ref(),
        side,
    )?;
    let denominator_tail = if exponent.is_multiple_of(2) {
        InfSign::Pos
    } else {
        denominator_tail
    };

    let scale_tail = rational_tail_sign(&scale);
    let numerator_tail = if exponent.is_multiple_of(2) || numerator_value.is_positive() {
        scale_tail
    } else {
        multiply_tail_signs(scale_tail, InfSign::Neg)
    };

    Some(signed_abs_ratio_infinity(
        ctx,
        numerator_tail,
        denominator_tail,
    ))
}

fn apply_finite_bilateral_trig_ratio_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_trig_ratio_power_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_trig_ratio_power_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

#[derive(Clone)]
enum UnitLogBase {
    Natural,
    Fixed(BigRational),
    UnitBoundary(InfSign),
}

fn finite_unit_log_base(
    ctx: &mut Context,
    base_expr: ExprId,
    var: ExprId,
    point: ExprId,
    var_name: &str,
    point_value: &BigRational,
    unit_boundary_side: Option<FiniteLimitSide>,
) -> Option<UnitLogBase> {
    let base = if let Some(base) = constant_rational_value(ctx, base_expr) {
        base
    } else if let Ok(base_poly) = Polynomial::from_expr(ctx, base_expr, var_name) {
        base_poly.eval(point_value)
    } else {
        let base_limit = try_limit_rules_at_finite(ctx, base_expr, var, point)?;
        constant_rational_value(ctx, base_limit)?
    };

    if !base.is_positive() {
        return None;
    }
    if base == rational_one() {
        let side = unit_boundary_side?;
        let base_tail =
            finite_endpoint_unit_base_tail_sign(ctx, base_expr, var_name, point_value, side)?;
        return Some(UnitLogBase::UnitBoundary(base_tail));
    }
    Some(UnitLogBase::Fixed(base))
}

fn scaled_unit_log_argument(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    var_name: &str,
    point_value: &BigRational,
    unit_boundary_side: Option<FiniteLimitSide>,
) -> Option<(BigRational, ExprId, UnitLogBase)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id)? {
            BuiltinFn::Ln => Some((BigRational::one(), args[0], UnitLogBase::Natural)),
            BuiltinFn::Log2 => Some((
                BigRational::one(),
                args[0],
                UnitLogBase::Fixed(BigRational::from_integer(BigInt::from(2))),
            )),
            BuiltinFn::Log10 => Some((
                BigRational::one(),
                args[0],
                UnitLogBase::Fixed(BigRational::from_integer(BigInt::from(10))),
            )),
            _ => None,
        },
        Expr::Function(fn_id, args) if args.len() == 2 && ctx.is_builtin(fn_id, BuiltinFn::Log) => {
            let base = finite_unit_log_base(
                ctx,
                args[0],
                var,
                point,
                var_name,
                point_value,
                unit_boundary_side,
            )?;
            Some((BigRational::one(), args[1], base))
        }
        Expr::Neg(inner) => {
            let (scale, argument, base) = scaled_unit_log_argument(
                ctx,
                inner,
                var,
                point,
                var_name,
                point_value,
                unit_boundary_side,
            )?;
            Some((-scale, argument, base))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument, base) = scaled_unit_log_argument(
                    ctx,
                    rhs,
                    var,
                    point,
                    var_name,
                    point_value,
                    unit_boundary_side,
                )?;
                return Some((scale * inner_scale, argument, base));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument, base) = scaled_unit_log_argument(
                    ctx,
                    lhs,
                    var,
                    point,
                    var_name,
                    point_value,
                    unit_boundary_side,
                )?;
                return Some((scale * inner_scale, argument, base));
            }
            None
        }
        _ => None,
    }
}

fn finite_log_unit_quotient_result(
    ctx: &mut Context,
    value: BigRational,
    base: UnitLogBase,
) -> ExprId {
    match base {
        UnitLogBase::Natural => ctx.add(Expr::Number(value)),
        UnitLogBase::Fixed(base) => {
            if value.is_zero() {
                return ctx.add(Expr::Number(value));
            }
            let numerator = ctx.add(Expr::Number(value));
            let base_expr = ctx.add(Expr::Number(base));
            let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
            ctx.add(Expr::Div(numerator, ln_base))
        }
        UnitLogBase::UnitBoundary(_) => {
            unreachable!("unit-boundary log bases are only valid for endpoint limits")
        }
    }
}

fn unit_log_base_tail_coeff(base: &UnitLogBase) -> BigRational {
    match base {
        UnitLogBase::Natural => rational_one(),
        UnitLogBase::Fixed(base) if base > &rational_one() => rational_one(),
        UnitLogBase::Fixed(_) => -rational_one(),
        UnitLogBase::UnitBoundary(InfSign::Pos) => rational_one(),
        UnitLogBase::UnitBoundary(InfSign::Neg) => -rational_one(),
    }
}

fn apply_finite_one_sided_log_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();

    let (scale, log_argument, base) =
        scaled_unit_log_argument(ctx, expr, var, point, &var_name, &point_value, Some(side))?;
    if scale.is_zero() {
        return None;
    }

    if finite_endpoint_argument_zero_tail_sign(ctx, log_argument, &var_name, &point_value, side)?
        != InfSign::Pos
    {
        return None;
    }

    let total_scale = scale * unit_log_base_tail_coeff(&base);
    scale_infinity(ctx, &total_scale, InfSign::Neg)
}

fn apply_finite_bilateral_log_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_log_endpoint_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_log_endpoint_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn apply_finite_one_sided_sqrt_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let radicand = extract_square_root_base(ctx, expr)?;
    if finite_endpoint_argument_zero_tail_sign(ctx, radicand, var_name, &point_value, side)?
        != InfSign::Pos
    {
        return None;
    }

    Some(ctx.num(0))
}

fn apply_finite_bilateral_sqrt_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let radicand = extract_square_root_base(ctx, expr)?;
    if !finite_endpoint_argument_zero_tail_positive_on_both_sides(
        ctx,
        radicand,
        var_name,
        &point_value,
    )? {
        return None;
    }

    Some(ctx.num(0))
}

fn apply_finite_one_sided_acosh_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    if argument.eval(&point_value) != rational_one() {
        return None;
    }

    let endpoint_gap = argument.sub(&Polynomial::one(var_name));
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if finite_local_tail_sign(&gap_derivative, gap_order, side)? != InfSign::Pos {
        return None;
    }

    Some(ctx.num(0))
}

fn apply_finite_acosh_polynomial_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    if argument.eval(&point_value) != rational_one() {
        return None;
    }

    let endpoint_gap = argument.sub(&Polynomial::one(var_name));
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if !finite_local_tail_positive_on_both_sides(&gap_derivative, gap_order)? {
        return None;
    }

    Some(ctx.num(0))
}

#[derive(Clone, Copy)]
enum InverseTrigEndpoint {
    Lower,
    Upper,
}

fn finite_inverse_trig_endpoint_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    endpoint: InverseTrigEndpoint,
) -> Option<ExprId> {
    match (builtin, endpoint) {
        (BuiltinFn::Asin | BuiltinFn::Arcsin, InverseTrigEndpoint::Lower) => {
            let pi_over_two = TrigValue::PiDiv(2).to_expr(ctx);
            Some(ctx.add(Expr::Neg(pi_over_two)))
        }
        (BuiltinFn::Asin | BuiltinFn::Arcsin, InverseTrigEndpoint::Upper) => {
            Some(TrigValue::PiDiv(2).to_expr(ctx))
        }
        (BuiltinFn::Acos | BuiltinFn::Arccos, InverseTrigEndpoint::Lower) => {
            Some(ctx.add(Expr::Constant(Constant::Pi)))
        }
        (BuiltinFn::Acos | BuiltinFn::Arccos, InverseTrigEndpoint::Upper) => Some(ctx.num(0)),
        _ => None,
    }
}

fn finite_inverse_trig_endpoint_gap(
    argument: &Polynomial,
    var_name: &str,
    point_value: &BigRational,
) -> Option<(Polynomial, InverseTrigEndpoint)> {
    let one = rational_one();
    let argument_value = argument.eval(point_value);
    if argument_value == one {
        Some((
            Polynomial::one(var_name.to_string()).sub(argument),
            InverseTrigEndpoint::Upper,
        ))
    } else if argument_value == -one {
        Some((
            argument.add(&Polynomial::one(var_name.to_string())),
            InverseTrigEndpoint::Lower,
        ))
    } else {
        None
    }
}

fn apply_finite_one_sided_atanh_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    let (endpoint_gap, endpoint) =
        finite_inverse_trig_endpoint_gap(&argument, &var_name, &point_value)?;
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if finite_local_tail_sign(&gap_derivative, gap_order, side)? != InfSign::Pos {
        return None;
    }

    Some(match endpoint {
        InverseTrigEndpoint::Lower => mk_infinity(ctx, InfSign::Neg),
        InverseTrigEndpoint::Upper => mk_infinity(ctx, InfSign::Pos),
    })
}

fn apply_finite_inverse_trig_polynomial_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    let (endpoint_gap, endpoint) =
        finite_inverse_trig_endpoint_gap(&argument, &var_name, &point_value)?;
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if !finite_local_tail_positive_on_both_sides(&gap_derivative, gap_order)? {
        return None;
    }

    finite_inverse_trig_endpoint_result(ctx, builtin, endpoint)
}

fn apply_finite_one_sided_inverse_trig_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    let (endpoint_gap, endpoint) =
        finite_inverse_trig_endpoint_gap(&argument, &var_name, &point_value)?;
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if finite_local_tail_sign(&gap_derivative, gap_order, side)? != InfSign::Pos {
        return None;
    }

    finite_inverse_trig_endpoint_result(ctx, builtin, endpoint)
}

fn exp_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Exp) => {
            Some(args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => Some(exp),
        _ => None,
    }
}

fn scaled_exp_zero_offset_argument(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, rhs) => {
            Some((BigRational::one(), exp_argument(ctx, lhs)?))
        }
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, lhs) => {
            Some((-BigRational::one(), exp_argument(ctx, rhs)?))
        }
        Expr::Add(lhs, rhs) => {
            if constant_rational_value(ctx, rhs).is_some_and(|value| value == -BigRational::one()) {
                return Some((BigRational::one(), exp_argument(ctx, lhs)?));
            }
            if constant_rational_value(ctx, lhs).is_some_and(|value| value == -BigRational::one()) {
                return Some((BigRational::one(), exp_argument(ctx, rhs)?));
            }
            None
        }
        Expr::Neg(inner) => {
            let (scale, argument) = scaled_exp_zero_offset_argument(ctx, inner)?;
            Some((-scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument) = scaled_exp_zero_offset_argument(ctx, rhs)?;
                return Some((scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument) = scaled_exp_zero_offset_argument(ctx, lhs)?;
                return Some((scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

fn apply_finite_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    scaled_argument: fn(&Context, ExprId) -> Option<(BigRational, ExprId)>,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, zero_argument) = scaled_argument(ctx, num)?;
    let argument = Polynomial::from_expr(ctx, zero_argument, &var_name).ok()?;
    if !argument.eval(&point_value).is_zero() {
        return None;
    }

    let scale_poly = Polynomial::new(vec![scale], var_name.to_string());
    let numerator = argument.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

fn apply_finite_sine_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    apply_finite_zero_quotient_rule(ctx, expr, var, point, scaled_sine_argument)
}

fn apply_finite_exp_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    apply_finite_zero_quotient_rule(ctx, expr, var, point, scaled_exp_zero_offset_argument)
}

/// `(a^(g) - 1) / h(x) -> ln(a) * lim(g/h)` as `var -> point`, the
/// derivative-of-`a^x`-at-0 family: `(2^x - 1)/x = ln 2`, `(3^x - 1)/x = ln 3`,
/// `(2^(3x) - 1)/x = 3 ln 2`. The base `a` must be a positive rational != 1
/// (the natural base `e^x - 1` is left to apply_finite_exp_zero_quotient_rule,
/// which gives the cleaner 1 instead of ln(e)). Since `a^g ~ 1 + g ln a` as
/// `g -> 0`, the numerator's first-order equivalent is `g ln a`, and the limit
/// is `ln(a)` times the rational limit of `g/h`.
fn apply_finite_general_exp_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, num)?;
    if !base.is_positive() || base.is_one() {
        return None;
    }
    let exponent_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
    // a^g -> 1 (so a^g - 1 -> 0) requires the exponent to vanish at the point.
    if !exponent_poly.eval(&point_value).is_zero() {
        return None;
    }
    let scale_poly = Polynomial::new(vec![scale], var_name.clone());
    let numerator = exponent_poly.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let ratio = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    // 0 * ln(a) = 0 (g vanishes faster than h); fold it rather than emit 0*ln(a).
    if ratio.is_zero() {
        return Some(ctx.num(0));
    }
    let base_expr = ctx.add(Expr::Number(base));
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
    if ratio.is_one() {
        return Some(ln_base);
    }
    let ratio_expr = ctx.add(Expr::Number(ratio));
    Some(ctx.add(Expr::Mul(ratio_expr, ln_base)))
}

/// `(c1 (a^g - 1)) / (c2 (b^h - 1))` at 0 -> the ratio of first-order
/// coefficients. Since `a^g - 1 ~ g ln a` for `g -> 0`, the numerator is
/// `~ c1 g'(0) ln a * x` and the denominator `~ c2 h'(0) ln b * x`, so the
/// limit is `(c1 g'(0) ln a) / (c2 h'(0) ln b)`. Resolves the classic
/// `(3^x - 1)/(2^x - 1) = ln 3 / ln 2`. Both sides must be a numeric-base
/// `a^x - 1` form vanishing at 0.
fn apply_finite_general_exp_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // (rational coefficient, base) of the first-order term scale*slope*ln(base).
    let first_order = |ctx: &mut Context, side: ExprId| -> Option<(BigRational, BigRational)> {
        let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, side)?;
        if !base.is_positive() || base.is_one() {
            return None;
        }
        let exp_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
        // The exponent must vanish at 0 (so a^g -> 1) and have a linear term.
        if !exp_poly.eval(&BigRational::zero()).is_zero() {
            return None;
        }
        let slope = exp_poly.coeffs.get(1).cloned()?;
        if slope.is_zero() {
            return None;
        }
        Some((scale * slope, base))
    };
    let (num_coeff, num_base) = first_order(ctx, num)?;
    let (den_coeff, den_base) = first_order(ctx, den)?;
    // result = (num_coeff / den_coeff) * ln(num_base) / ln(den_base).
    let rational = &num_coeff / &den_coeff;
    // Equal bases: ln(a)/ln(a) = 1, so the limit is the bare rational ratio.
    if num_base == den_base {
        return Some(ctx.add(Expr::Number(rational)));
    }
    let num_base_expr = ctx.add(Expr::Number(num_base));
    let num_log = ctx.call_builtin(BuiltinFn::Ln, vec![num_base_expr]);
    let den_base_expr = ctx.add(Expr::Number(den_base));
    let den_log = ctx.call_builtin(BuiltinFn::Ln, vec![den_base_expr]);
    let log_ratio = ctx.add(Expr::Div(num_log, den_log));
    if rational.is_one() {
        Some(log_ratio)
    } else {
        let rational_expr = ctx.add(Expr::Number(rational));
        Some(ctx.add(Expr::Mul(rational_expr, log_ratio)))
    }
}

/// Recognize `scale * (a^(g) - 1)` with `a` a numeric base; returns
/// `(scale, a, g)`. Handles the Sub and Add(-1) offset forms, a numeric
/// scale factor, and a negation.
fn scaled_general_power_zero_offset(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, rhs) => {
            let (base, exponent) = numeric_base_power(ctx, lhs)?;
            Some((rational_one(), base, exponent))
        }
        // 1 - a^g = -(a^g - 1).
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, lhs) => {
            let (base, exponent) = numeric_base_power(ctx, rhs)?;
            Some((-rational_one(), base, exponent))
        }
        Expr::Add(lhs, rhs) => {
            if constant_rational_value(ctx, rhs).is_some_and(|v| v == -rational_one()) {
                let (base, exponent) = numeric_base_power(ctx, lhs)?;
                return Some((rational_one(), base, exponent));
            }
            if constant_rational_value(ctx, lhs).is_some_and(|v| v == -rational_one()) {
                let (base, exponent) = numeric_base_power(ctx, rhs)?;
                return Some((rational_one(), base, exponent));
            }
            None
        }
        Expr::Neg(inner) => {
            let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, inner)?;
            Some((-scale, base, exponent))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, base, exponent) = scaled_general_power_zero_offset(ctx, rhs)?;
                return Some((scale * inner_scale, base, exponent));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, base, exponent) = scaled_general_power_zero_offset(ctx, lhs)?;
                return Some((scale * inner_scale, base, exponent));
            }
            None
        }
        _ => None,
    }
}

/// `a^(exponent)` with `a` a numeric rational base; returns `(a, exponent)`.
fn numeric_base_power(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let base_value = constant_rational_value(ctx, *base)?;
            Some((base_value, *exponent))
        }
        _ => None,
    }
}

/// An exponential atom `base^(g)`: returns `(Some(rational_base), g)` for a
/// numeric base != 1, or `(None, g)` for the natural base e (ln(e) = 1). Both
/// `Pow(E, g)` and `exp(g)` are recognized.
fn exponential_base_and_exponent(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Option<BigRational>, ExprId)> {
    use num_traits::One;
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Exp) =>
        {
            Some((None, args[0]))
        }
        Expr::Pow(base, exponent) => {
            if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
                return Some((None, *exponent));
            }
            let base_value = constant_rational_value(ctx, *base)?;
            if !base_value.is_positive() || base_value.is_one() {
                return None;
            }
            Some((Some(base_value), *exponent))
        }
        _ => None,
    }
}

/// Accumulate `expr` (under overall `sign`) as a linear combination of
/// exponentials `c * a^(g)` (a a positive rational != 1, or e) plus constants,
/// reading off the value at 0 and the first derivative at 0. Each exponential
/// requires `g` a polynomial with `g(0) = 0` and degree >= 1, so `a^g -> 1`
/// and the derivative contribution is `c * g'(0) * ln(a)` (or `c * g'(0)` for
/// the base e). Returns None for any term outside the class.
#[allow(clippy::too_many_arguments)]
fn accumulate_exp_combination(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    sign: &BigRational,
    value: &mut BigRational,
    log_terms: &mut Vec<(BigRational, BigRational)>,
    const_deriv: &mut BigRational,
    saw_exp: &mut bool,
) -> Option<()> {
    use num_traits::Zero;
    // A pure constant contributes to the value, nothing to the derivative.
    if let Some(c) = constant_rational_value(ctx, expr) {
        *value += sign * &c;
        return Some(());
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let neg = -sign.clone();
            accumulate_exp_combination(
                ctx,
                inner,
                var_name,
                &neg,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Add(a, b) => {
            accumulate_exp_combination(
                ctx,
                a,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )?;
            accumulate_exp_combination(
                ctx,
                b,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Sub(a, b) => {
            accumulate_exp_combination(
                ctx,
                a,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )?;
            let neg = -sign.clone();
            accumulate_exp_combination(
                ctx,
                b,
                var_name,
                &neg,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Mul(a, b) => {
            // Exactly one factor must be an x-free rational scale.
            if let Some(scale) = constant_rational_value(ctx, a) {
                let s = sign * &scale;
                return accumulate_exp_combination(
                    ctx,
                    b,
                    var_name,
                    &s,
                    value,
                    log_terms,
                    const_deriv,
                    saw_exp,
                );
            }
            if let Some(scale) = constant_rational_value(ctx, b) {
                let s = sign * &scale;
                return accumulate_exp_combination(
                    ctx,
                    a,
                    var_name,
                    &s,
                    value,
                    log_terms,
                    const_deriv,
                    saw_exp,
                );
            }
            None
        }
        _ => {
            let (base_opt, g) = exponential_base_and_exponent(ctx, expr)?;
            let g_poly = Polynomial::from_expr(ctx, g, var_name).ok()?;
            if !g_poly.eval(&BigRational::zero()).is_zero() || g_poly.degree() < 1 {
                return None;
            }
            let slope = g_poly
                .coeffs
                .get(1)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            *saw_exp = true;
            // a^(g(0)) = a^0 = 1 contributes to the value at 0.
            *value += sign.clone();
            let coeff = sign * &slope;
            match base_opt {
                Some(base) => {
                    if !coeff.is_zero() {
                        log_terms.push((coeff, base));
                    }
                }
                None => *const_deriv += coeff, // ln(e) = 1
            }
            Some(())
        }
    }
}

/// `(c0 a^(g0) + c1 a^(g1) + ...) / h` at 0 where the numerator is a linear
/// combination of exponentials that vanishes at 0 and `h` is a polynomial
/// vanishing to first order: the limit is the ratio of first derivatives
/// `N'(0) / h'(0) = (sum c_i g_i'(0) ln a_i) / h'(0)`. Resolves the difference
/// of general-base exponentials `(a^x - b^x)/x -> ln(a) - ln(b)`, which the
/// single-power rule and the rational Taylor engine cannot reach (ln a is
/// transcendental).
/// `(c0 a^g0 + ...) / (d0 b^h0 + ...)` at 0 where BOTH sides are exponential
/// combinations vanishing at 0: the limit is the ratio of first derivatives
/// `N'(0)/D'(0)`. Resolves `(2^x - 3^x)/(5^x - 7^x) -> (ln 2 - ln 3)/(ln 5 -
/// ln 7)`, the two-sided sibling of (a^x-b^x)/x and (a^x-1)/(b^x-1).
fn apply_finite_exp_combination_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let (num_log, num_const) = exp_combination_first_derivative(ctx, num, &var_name)?;
    let (den_log, den_const) = exp_combination_first_derivative(ctx, den, &var_name)?;
    // D'(0) must be nonzero for the ratio to be the limit. A combination of
    // logs is exactly 0 only on cancellation; a clearly-nonzero float magnitude
    // proves it is safe (a true zero evaluates to exactly 0).
    if log_combination_float(&den_log, &den_const).abs() < 1e-9 {
        return None;
    }
    // N'(0) = 0 over a nonzero D'(0): the numerator vanishes faster, limit 0.
    if log_combination_float(&num_log, &num_const).abs() < 1e-12 {
        return Some(ctx.num(0));
    }
    let num_deriv = build_log_combination_expr(ctx, num_log, num_const);
    let den_deriv = build_log_combination_expr(ctx, den_log, den_const);
    Some(ctx.add(Expr::Div(num_deriv, den_deriv)))
}

/// Run the exponential-combination accumulator and return its first-derivative
/// `(log terms, constant)` only when the expression genuinely vanishes at 0
/// over a real exponential combination.
fn exp_combination_first_derivative(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(Vec<(BigRational, BigRational)>, BigRational)> {
    use num_traits::Zero;
    let mut value = BigRational::zero();
    let mut log_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut const_deriv = BigRational::zero();
    let mut saw_exp = false;
    accumulate_exp_combination(
        ctx,
        expr,
        var_name,
        &rational_one(),
        &mut value,
        &mut log_terms,
        &mut const_deriv,
        &mut saw_exp,
    )?;
    if !saw_exp || !value.is_zero() {
        return None;
    }
    Some((log_terms, const_deriv))
}

/// Numeric value of `const + sum coeff_i ln(base_i)`, for a nonzero check.
fn log_combination_float(log_terms: &[(BigRational, BigRational)], constant: &BigRational) -> f64 {
    use num_traits::ToPrimitive;
    let mut total = constant.to_f64().unwrap_or(0.0);
    for (coeff, base) in log_terms {
        let c = coeff.to_f64().unwrap_or(0.0);
        let b = base.to_f64().unwrap_or(1.0);
        if b > 0.0 {
            total += c * b.ln();
        }
    }
    total
}

/// Build the expression `const + sum coeff_i ln(base_i)`.
fn build_log_combination_expr(
    ctx: &mut Context,
    log_terms: Vec<(BigRational, BigRational)>,
    constant: BigRational,
) -> ExprId {
    use num_traits::{One, Zero};
    let mut result: Option<ExprId> = None;
    if !constant.is_zero() {
        result = Some(ctx.add(Expr::Number(constant)));
    }
    for (coeff, base) in log_terms {
        if coeff.is_zero() {
            continue;
        }
        let base_expr = ctx.add(Expr::Number(base));
        let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
        let term = if coeff.is_one() {
            ln_base
        } else if coeff == -rational_one() {
            ctx.add(Expr::Neg(ln_base))
        } else {
            let coeff_expr = ctx.add(Expr::Number(coeff));
            ctx.add(Expr::Mul(coeff_expr, ln_base))
        };
        result = Some(match result {
            Some(acc) => ctx.add(Expr::Add(acc, term)),
            None => term,
        });
    }
    result.unwrap_or_else(|| ctx.num(0))
}

fn apply_finite_exp_linear_combination_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let mut value = BigRational::zero();
    let mut log_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut const_deriv = BigRational::zero();
    let mut saw_exp = false;
    accumulate_exp_combination(
        ctx,
        num,
        &var_name,
        &rational_one(),
        &mut value,
        &mut log_terms,
        &mut const_deriv,
        &mut saw_exp,
    )?;
    // A genuine 0/0 over a real exponential combination.
    if !saw_exp || !value.is_zero() {
        return None;
    }
    // Denominator: a polynomial vanishing to exactly first order at 0.
    let den_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    if !den_poly.eval(&BigRational::zero()).is_zero() {
        return None;
    }
    let den_slope = den_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if den_slope.is_zero() {
        return None;
    }

    // result = (const_deriv + sum coeff_i ln(base_i)) / den_slope.
    let mut result: Option<ExprId> = None;
    let scaled_const = &const_deriv / &den_slope;
    if !scaled_const.is_zero() {
        result = Some(ctx.add(Expr::Number(scaled_const)));
    }
    for (coeff, base) in log_terms {
        let scaled = &coeff / &den_slope;
        if scaled.is_zero() {
            continue;
        }
        let base_expr = ctx.add(Expr::Number(base));
        let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
        let term = if scaled.is_one() {
            ln_base
        } else if scaled == -rational_one() {
            ctx.add(Expr::Neg(ln_base))
        } else {
            let coeff_expr = ctx.add(Expr::Number(scaled));
            ctx.add(Expr::Mul(coeff_expr, ln_base))
        };
        result = Some(match result {
            Some(acc) => ctx.add(Expr::Add(acc, term)),
            None => term,
        });
    }
    Some(result.unwrap_or_else(|| ctx.num(0)))
}

fn apply_finite_log_unit_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, log_argument, base) =
        scaled_unit_log_argument(ctx, num, var, point, &var_name, &point_value, None)?;
    let argument = Polynomial::from_expr(ctx, log_argument, &var_name).ok()?;
    if argument.eval(&point_value) != BigRational::one() {
        return None;
    }

    let unit_offset = argument.sub(&Polynomial::one(var_name.clone()));
    let scale_poly = Polynomial::new(vec![scale], var_name.clone());
    let numerator = unit_offset.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(finite_log_unit_quotient_result(ctx, value, base))
}

/// Functions `f` with `f(u) ~ u` as `u -> 0` (Taylor leading term exactly
/// `u`, i.e. `f(u)/u -> 1`). These are the first-order equivalent
/// infinitesimals. Cos/cosh are EXCLUDED: they tend to 1, not 0.
fn is_first_order_zero_atom(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Sin
            | BuiltinFn::Tan
            | BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Sinh
            | BuiltinFn::Tanh
    )
}

/// Extracts the first-order equivalent polynomial of `expr` as `var -> point`,
/// for use as a numerator/denominator in a 0/0 quotient. The equivalent
/// infinitesimal theorem makes `lim(num/den) = lim(equiv_num/equiv_den)` for
/// PRODUCTS and QUOTIENTS of these atoms; it is invalid inside a sum/difference
/// where the leading terms cancel, so a top-level `Add`/`Sub` of atoms declines.
///
/// Recognized shapes, in order:
/// - an exact polynomial in the variable (e.g. `x`, `x^2`, `3*x`),
/// - `exp(u) - 1 ~ u` (matched before the generic sum decline),
/// - `f(u) ~ u` for `f` a first-order zero atom, gated on `u -> 0` at the point,
/// - `-g ~ -equiv(g)`,
/// - `a * b ~ equiv(a) * equiv(b)`.
///
/// Everything else (notably a top-level `Add`/`Sub` of atoms, and `cos`/`cosh`)
/// declines, keeping `(1 - cos x)/x^2`, `(sin x - x)/x^3` honestly residual.
fn first_order_equivalent_poly(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
) -> Option<Polynomial> {
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) {
        return Some(poly);
    }
    // `exp(u) - 1` is syntactically a Sub, so it must be recognized as an atom
    // BEFORE the generic Add/Sub decline below.
    if let Some((scale, exponent)) = scaled_exp_zero_offset_argument(ctx, expr) {
        let exponent_poly = Polynomial::from_expr(ctx, exponent, var_name).ok()?;
        if !exponent_poly.eval(point_value).is_zero() {
            return None;
        }
        let scale_poly = Polynomial::new(vec![scale], var_name.to_string());
        return Some(exponent_poly.mul(&scale_poly));
    }
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(fn_id)?;
            if !is_first_order_zero_atom(builtin) {
                return None;
            }
            let argument_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
            // The equivalent `f(u) ~ u` only holds where `u -> 0`. Without this
            // guard, `sin(x)/x` at pi would wrongly resolve to 1 instead of 0.
            if !argument_poly.eval(point_value).is_zero() {
                return None;
            }
            Some(argument_poly)
        }
        Expr::Neg(inner) => {
            let inner_poly = first_order_equivalent_poly(ctx, inner, var_name, point_value)?;
            let minus_one = Polynomial::new(vec![-BigRational::one()], var_name.to_string());
            Some(inner_poly.mul(&minus_one))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_poly = first_order_equivalent_poly(ctx, lhs, var_name, point_value)?;
            let rhs_poly = first_order_equivalent_poly(ctx, rhs, var_name, point_value)?;
            Some(lhs_poly.mul(&rhs_poly))
        }
        _ => None,
    }
}

/// 0/0 finite-point limits resolved by replacing every factor of the numerator
/// and denominator with its first-order equivalent infinitesimal, then taking
/// the polynomial ratio. Generalizes the sine/exp small-angle rules to handle
/// inversion (`x/sin(x) -> 1`), composition (`sin(3x)/sin(5x) -> 3/5`), and the
/// missing equivalents (`tan/asin/arctan/sinh/tanh`).
///
/// Runs AFTER the sine/exp/log rules, so it only fires on the cases they leave
/// residual (a non-polynomial denominator, or a numerator with no existing
/// recognizer). Only acts on a genuine 0/0 form: both equivalents must vanish
/// at the point.
fn apply_finite_equivalent_infinitesimal_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = first_order_equivalent_poly(ctx, num, &var_name, &point_value)?;
    let denominator = first_order_equivalent_poly(ctx, den, &var_name, &point_value)?;
    // Genuine 0/0 only: a non-vanishing denominator is a continuous case owned
    // by ordinary substitution, and a non-vanishing numerator over a vanishing
    // denominator is a pole that must stay residual.
    if !numerator.eval(&point_value).is_zero() || !denominator.eval(&point_value).is_zero() {
        return None;
    }
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

fn is_finite_total_real_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Cos
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Tanh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asinh
            | BuiltinFn::Cbrt
            | BuiltinFn::Abs
    )
}

fn is_finite_positive_domain_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 | BuiltinFn::Sqrt
    )
}

fn is_finite_partial_domain_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Acos
            | BuiltinFn::Arccos
            | BuiltinFn::Atanh
            | BuiltinFn::Acosh
    )
}

fn is_finite_domain_checked_trig_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Tan | BuiltinFn::Sec | BuiltinFn::Csc | BuiltinFn::Cot
    )
}

fn finite_total_real_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> ExprId {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if matches!(builtin, BuiltinFn::Cbrt) {
            if let Some(root) = rational_cbrt_exact(&argument_value) {
                return ctx.add(Expr::Number(root));
            }
            let value_expr = ctx.add(Expr::Number(argument_value));
            return ctx.call_builtin(BuiltinFn::Cbrt, vec![value_expr]);
        }
        if matches!(builtin, BuiltinFn::Abs) {
            return ctx.add(Expr::Number(argument_value.abs()));
        }
        if matches!(
            builtin,
            BuiltinFn::Sin
                | BuiltinFn::Sinh
                | BuiltinFn::Tanh
                | BuiltinFn::Atan
                | BuiltinFn::Arctan
                | BuiltinFn::Asinh
        ) && argument_value.is_zero()
        {
            return ctx.num(0);
        }
        if matches!(builtin, BuiltinFn::Exp | BuiltinFn::Cos | BuiltinFn::Cosh)
            && argument_value.is_zero()
        {
            return ctx.num(1);
        }

        let value_expr = ctx.add(Expr::Number(argument_value));
        if let Some(exact_result) =
            finite_total_real_unary_trig_table_result(ctx, builtin, value_expr)
        {
            return exact_result;
        }
        return ctx.call_builtin(builtin, vec![value_expr]);
    }

    if let Some(exact_result) =
        finite_total_real_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return exact_result;
    }

    // Saturate a growing function of an unbounded argument: sinh(inf) -> inf,
    // cosh(inf) -> inf, tanh(inf) -> 1, exp(-inf) -> 0, etc. Without this the
    // composition leaks an unfolded sinh(inf), which downstream `0 * value`
    // wrongly reads as bounded (x * sinh(1/x^2) -> 0 instead of +inf).
    let candidate = ctx.call_builtin(builtin, vec![argument_limit]);
    crate::infinity_support::fold_infinity_saturation(ctx, candidate)
}

fn finite_total_real_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Abs => finite_abs_exact_expr_result(ctx, argument_limit),
        BuiltinFn::Exp => finite_exp_exact_expr_result(ctx, argument_limit),
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Atan | BuiltinFn::Arctan => {
            finite_total_real_unary_trig_table_result(ctx, builtin, argument_limit)
        }
        _ => None,
    }
}

fn finite_total_real_unary_trig_table_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Atan | BuiltinFn::Arctan
    ) {
        return None;
    }

    lookup_trig_or_inverse(ctx, builtin.name(), argument)
        .map(|hit| trig_table_value_to_limit_expr(ctx, hit.value))
}

fn trig_table_value_to_limit_expr(ctx: &mut Context, value: &TrigValue) -> ExprId {
    match value {
        TrigValue::Fraction(numerator, 1) => ctx.num(*numerator),
        _ => value.to_expr(ctx),
    }
}

fn finite_abs_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    if finite_expr_proven_positive(ctx, argument_limit) {
        return Some(argument_limit);
    }

    let Expr::Neg(inner) = ctx.get(argument_limit).clone() else {
        return None;
    };
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

fn finite_exp_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let inner = args[0];
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

fn finite_positive_domain_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_positive_domain_unary_exact_numeric_result(ctx, builtin, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(builtin, vec![value_expr]));
    }

    if let Some(exact_result) =
        finite_positive_domain_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return Some(exact_result);
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(builtin, vec![argument_limit]))
}

fn finite_positive_domain_unary_exact_numeric_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_value: &BigRational,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Sqrt => rational_sqrt(argument_value).map(|root| ctx.add(Expr::Number(root))),
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 if argument_value.is_one() => {
            Some(ctx.num(0))
        }
        BuiltinFn::Log2 => {
            let base = BigRational::from_integer(BigInt::from(2));
            finite_exact_rational_log_result(ctx, &base, argument_value)
        }
        BuiltinFn::Log10 => {
            let base = BigRational::from_integer(BigInt::from(10));
            finite_exact_rational_log_result(ctx, &base, argument_value)
        }
        _ => None,
    }
}

fn finite_positive_domain_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Ln => finite_ln_exact_expr_result(ctx, argument_limit),
        _ => None,
    }
}

fn finite_partial_domain_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !finite_partial_domain_argument_is_strictly_interior(builtin, &argument_value) {
            return None;
        }
        if let Some(exact_result) =
            finite_partial_domain_unary_exact_numeric_result(ctx, builtin, &argument_value)
        {
            return Some(exact_result);
        }

        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(builtin, vec![value_expr]));
    }

    if !finite_partial_domain_expr_is_strictly_interior(ctx, builtin, argument_limit) {
        return None;
    }
    if let Some(exact_result) =
        finite_partial_domain_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return Some(exact_result);
    }
    Some(ctx.call_builtin(builtin, vec![argument_limit]))
}

fn finite_partial_domain_argument_is_strictly_interior(
    builtin: BuiltinFn,
    argument_value: &BigRational,
) -> bool {
    let one = rational_one();
    let neg_one = -one.clone();
    match builtin {
        BuiltinFn::Asin
        | BuiltinFn::Arcsin
        | BuiltinFn::Acos
        | BuiltinFn::Arccos
        | BuiltinFn::Atanh => argument_value > &neg_one && argument_value < &one,
        BuiltinFn::Acosh => argument_value > &one,
        _ => false,
    }
}

fn finite_partial_domain_expr_is_strictly_interior(
    ctx: &Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> bool {
    match builtin {
        BuiltinFn::Asin
        | BuiltinFn::Arcsin
        | BuiltinFn::Acos
        | BuiltinFn::Arccos
        | BuiltinFn::Atanh => finite_expr_proven_abs_less_than_one(ctx, argument_limit),
        BuiltinFn::Acosh => finite_expr_proven_greater_than_one(ctx, argument_limit),
        _ => false,
    }
}

fn finite_expr_proven_abs_less_than_one(ctx: &Context, expr: ExprId) -> bool {
    if let Some(value) = numeric_limit_value(ctx, expr) {
        return finite_partial_domain_argument_is_strictly_interior(BuiltinFn::Atanh, &value);
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        return finite_expr_proven_abs_less_than_one(ctx, *inner);
    }

    let Some(radicand) = extract_square_root_base(ctx, expr) else {
        return false;
    };
    numeric_limit_value(ctx, radicand).is_some_and(|radicand_value| {
        !radicand_value.is_negative() && radicand_value < rational_one()
    })
}

fn finite_expr_proven_greater_than_one(ctx: &Context, expr: ExprId) -> bool {
    if numeric_limit_value(ctx, expr).is_some_and(|value| value > rational_one()) {
        return true;
    }

    let Some(radicand) = extract_square_root_base(ctx, expr) else {
        return false;
    };
    numeric_limit_value(ctx, radicand).is_some_and(|radicand_value| radicand_value > rational_one())
}

fn finite_partial_domain_unary_exact_numeric_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_value: &BigRational,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Atanh if argument_value.is_zero() => {
            Some(ctx.num(0))
        }
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos => {
            let argument_expr = ctx.add(Expr::Number(argument_value.clone()));
            lookup_trig_or_inverse(ctx, builtin.name(), argument_expr)
                .map(|hit| trig_table_value_to_limit_expr(ctx, hit.value))
        }
        _ => None,
    }
}

fn finite_partial_domain_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return None;
    }

    lookup_trig_or_inverse(ctx, builtin.name(), argument)
        .map(|hit| trig_table_value_to_limit_expr(ctx, hit.value))
}

fn finite_domain_checked_trig_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if argument_value.is_zero() {
            return match builtin {
                BuiltinFn::Tan => Some(ctx.num(0)),
                BuiltinFn::Sec => Some(ctx.num(1)),
                _ => None,
            };
        }

        let argument_expr = ctx.add(Expr::Number(argument_value));
        return finite_domain_checked_trig_unary_table_result(ctx, builtin, argument_expr);
    }

    finite_domain_checked_trig_unary_table_result(ctx, builtin, argument_limit)
}

fn finite_domain_checked_trig_unary_table_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Tan | BuiltinFn::Sec | BuiltinFn::Csc | BuiltinFn::Cot
    ) {
        return None;
    }

    let hit = lookup_trig_or_inverse(ctx, builtin.name(), argument)?;
    if matches!(*hit.value, TrigValue::Undefined) {
        return None;
    }
    Some(trig_table_value_to_limit_expr(ctx, hit.value))
}

fn finite_ln_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    if matches!(ctx.get(argument_limit), Expr::Constant(Constant::E)) {
        return Some(ctx.num(1));
    }

    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Exp) && args.len() == 1 {
        Some(args[0])
    } else {
        None
    }
}

fn finite_log_base_limit_is_valid(ctx: &Context, base_limit: ExprId) -> bool {
    let Some(base_value) = numeric_limit_value(ctx, base_limit) else {
        return false;
    };
    base_value.is_positive() && base_value != rational_one()
}

fn finite_log_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if !finite_log_base_limit_is_valid(ctx, base_limit) {
        return None;
    }
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_log_exact_numeric_result(ctx, base_limit, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(BuiltinFn::Log, vec![base_limit, value_expr]));
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(BuiltinFn::Log, vec![base_limit, argument_limit]))
}

fn finite_log_exact_numeric_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_value: &BigRational,
) -> Option<ExprId> {
    if argument_value.is_one() {
        return Some(ctx.num(0));
    }

    let base_value = numeric_limit_value(ctx, base_limit)?;
    if base_value == *argument_value {
        return Some(ctx.num(1));
    }

    finite_exact_rational_log_result(ctx, &base_value, argument_value)
}

const FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT: u64 = 32;
const FINITE_LOG_EXACT_RATIONAL_NUMERATOR_LIMIT: i64 = 32;
const FINITE_LOG_EXACT_RATIONAL_DENOMINATOR_LIMIT: i64 = 8;

fn finite_exact_rational_log_result(
    ctx: &mut Context,
    base_value: &BigRational,
    argument_value: &BigRational,
) -> Option<ExprId> {
    let exponent = exact_rational_log_result(base_value, argument_value)?;
    Some(ctx.add(Expr::Number(exponent)))
}

fn exact_rational_log_result(
    base_value: &BigRational,
    argument_value: &BigRational,
) -> Option<BigRational> {
    if !base_value.is_positive()
        || base_value.is_one()
        || !argument_value.is_positive()
        || argument_value.is_one()
    {
        return None;
    }

    for denominator in 1..=FINITE_LOG_EXACT_RATIONAL_DENOMINATOR_LIMIT {
        let argument_power = rational_pow_nonnegative(argument_value, denominator as u64);
        for numerator in 1..=FINITE_LOG_EXACT_RATIONAL_NUMERATOR_LIMIT {
            let base_power = rational_pow_nonnegative(base_value, numerator as u64);
            if argument_power == base_power {
                return Some(BigRational::new(
                    BigInt::from(numerator),
                    BigInt::from(denominator),
                ));
            }
            if argument_power == BigRational::one() / base_power {
                return Some(BigRational::new(
                    BigInt::from(-numerator),
                    BigInt::from(denominator),
                ));
            }
        }
    }

    None
}

fn rational_pow_nonnegative(base: &BigRational, exponent: u64) -> BigRational {
    let mut result = BigRational::one();
    let mut factor = base.clone();
    let mut remaining = exponent;

    while remaining > 0 {
        if remaining % 2 == 1 {
            result *= factor.clone();
        }
        remaining /= 2;
        if remaining > 0 {
            factor = factor.clone() * factor;
        }
    }

    result
}

fn finite_sqrt_even_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 2 != 0 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, base_limit)?;
    let radicand_value = numeric_limit_value(ctx, radicand)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let half_exponent = exponent.unsigned_abs() / 2;
    if half_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, half_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_cbrt_multiple_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 3 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Cbrt) || args.len() != 1 {
        return None;
    }

    let radicand_value = numeric_limit_value(ctx, args[0])?;
    if exponent <= 0 && radicand_value.is_zero() {
        return None;
    }

    let reduced_exponent = exponent.unsigned_abs() / 3;
    if reduced_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, reduced_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_integer_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if let Some(result) = finite_cbrt_multiple_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    let base_nonzero = finite_denominator_proven_nonzero(ctx, base_limit);
    if exponent <= 0 && !base_nonzero {
        return None;
    }

    if exponent == 0 {
        return Some(ctx.num(1));
    }

    if let Some(result) = finite_sqrt_even_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    if let Some(base_value) = numeric_limit_value(ctx, base_limit) {
        let abs_exponent = exponent.unsigned_abs();
        if abs_exponent <= FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
            let mut value = rational_pow_nonnegative(&base_value, abs_exponent);
            if exponent < 0 {
                if value.is_zero() {
                    return None;
                }
                value = BigRational::one() / value;
            }
            return Some(ctx.add(Expr::Number(value)));
        }
    }

    let exponent_expr = if exponent > 0 {
        ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            exponent,
        ))))
    } else {
        let positive_exponent = exponent.checked_neg()?;
        let positive_exponent_expr = ctx.add(Expr::Number(BigRational::from_integer(
            BigInt::from(positive_exponent),
        )));
        let denominator = if positive_exponent == 1 {
            base_limit
        } else {
            ctx.add(Expr::Pow(base_limit, positive_exponent_expr))
        };
        let one = ctx.num(1);
        return Some(ctx.add(Expr::Div(one, denominator)));
    };

    Some(ctx.add(Expr::Pow(base_limit, exponent_expr)))
}

fn apply_finite_integer_power_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let (base_expr, exponent) = parse_pow_int(ctx, expr)?;
    let base_limit = try_limit_rules_at_finite(ctx, base_expr, var, point)?;
    finite_integer_power_result(ctx, base_limit, exponent)
}

fn apply_finite_elementary_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let (builtin, argument_expr) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            (ctx.builtin_of(fn_id)?, args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (BuiltinFn::Exp, exp)
        }
        _ => return None,
    };
    if !matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Cos
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Tanh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asinh
            | BuiltinFn::Cbrt
            | BuiltinFn::Abs
            | BuiltinFn::Ln
            | BuiltinFn::Log2
            | BuiltinFn::Log10
            | BuiltinFn::Sqrt
            | BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Acos
            | BuiltinFn::Arccos
            | BuiltinFn::Atanh
            | BuiltinFn::Acosh
            | BuiltinFn::Tan
            | BuiltinFn::Sec
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, argument_expr, var_name).ok()?;
    let argument_value = argument.eval(&point_value);
    if is_finite_total_real_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return Some(finite_total_real_unary_result(ctx, builtin, argument_limit));
    }
    if is_finite_positive_domain_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_positive_domain_unary_result(ctx, builtin, argument_limit);
    }
    if is_finite_partial_domain_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_partial_domain_unary_result(ctx, builtin, argument_limit);
    }
    if is_finite_domain_checked_trig_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_domain_checked_trig_unary_result(ctx, builtin, argument_limit);
    }

    None
}

fn pow_one_third_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    match ctx.get(*exp) {
        Expr::Number(value) if *value.numer() == 1.into() && *value.denom() == 3.into() => {
            Some(*base)
        }
        Expr::Div(num, den) => {
            let (Expr::Number(num_value), Expr::Number(den_value)) = (ctx.get(*num), ctx.get(*den))
            else {
                return None;
            };
            if num_value.is_one() && den_value.is_integer() && *den_value.numer() == 3.into() {
                return Some(*base);
            }
            None
        }
        _ => None,
    }
}

fn apply_finite_cube_root_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let argument_expr = pow_one_third_argument(ctx, expr)?;
    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    Some(finite_total_real_unary_result(
        ctx,
        BuiltinFn::Cbrt,
        argument_limit,
    ))
}

fn apply_finite_square_root_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Pow(_, _)) {
        return None;
    }

    let argument_expr = extract_square_root_base(ctx, expr)?;
    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    finite_positive_domain_unary_result(ctx, BuiltinFn::Sqrt, argument_limit)
}

fn apply_finite_total_real_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let (builtin, argument_expr) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            (ctx.builtin_of(fn_id)?, args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (BuiltinFn::Exp, exp)
        }
        _ => return None,
    };
    if !is_finite_total_real_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    // sin/cos oscillate at +-infinity: their limit does not exist there.
    // Decline instead of leaking an unfolded sin(infinity)/cos(infinity)
    // atom (the saturation fold cleans up exp/atan/tanh/cosh but cannot
    // fold an oscillating outer). The odd-pole sibling already declines via
    // the one-sided saturator's "fold changed it" gate; this is the
    // even-pole bilateral path (cos(1/x^2)).
    if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos)
        && infinity_sign_of_expr(ctx, argument_limit).is_some()
    {
        return None;
    }
    Some(finite_total_real_unary_result(ctx, builtin, argument_limit))
}

fn apply_finite_positive_domain_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_positive_domain_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_positive_domain_unary_result(ctx, builtin, argument_limit)
}

fn apply_finite_partial_domain_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_partial_domain_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_partial_domain_unary_result(ctx, builtin, argument_limit)
}

fn apply_finite_domain_checked_trig_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_domain_checked_trig_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_domain_checked_trig_unary_result(ctx, builtin, argument_limit)
}

fn apply_finite_binary_log_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Log) || args.len() != 2 {
        return None;
    }

    let base_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    let argument_limit = try_limit_rules_at_finite(ctx, args[1], var, point)?;
    finite_log_result(ctx, base_limit, argument_limit)
}

fn try_limit_rules_at_finite(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if let Some(result) = apply_static_empty_real_domain_rule(ctx, expr, var) {
        return Some(result);
    }
    if let Some(result) = apply_constant_rule(ctx, expr, var) {
        return Some(result);
    }
    if expr == var && !depends_on(ctx, point, var) {
        return Some(point);
    }
    if let Some(result) = apply_finite_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_rational_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_radical_difference_conjugate_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_radical_conjugate_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_bilateral_rational_polynomial_pole_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_abs_polynomial_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_sign_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_trig_power_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_trig_ratio_power_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_sine_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_general_exp_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_general_exp_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_combination_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_log_unit_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_equivalent_infinitesimal_quotient_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_taylor_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_linear_combination_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_log_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_elementary_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_acosh_polynomial_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_inverse_trig_polynomial_endpoint_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_sqrt_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_square_root_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_cube_root_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_integer_power_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_positive_domain_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_partial_domain_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_domain_checked_trig_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_binary_log_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_total_real_unary_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_squeeze_bounded_product_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_to_infinity_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_even_saturating_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_lhopital_nonzero_point_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            Some(finite_add_result(ctx, lhs_limit, rhs_limit))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            if let Some(result) = finite_sub_result(ctx, lhs_limit, rhs_limit) {
                return Some(result);
            }
            // finite_sub_result declined the indeterminate same-sign ∞ - ∞: combine `lhs - rhs`
            // over a common denominator and retry the limit of the single fraction. The engine
            // evaluates `(x² - sin²x)/(x²·sin²x) -> 1/3`, recovering the value the operand-wise
            // split could not. The combined form is a `Div` (not a `Sub`), so this does not loop.
            let combined = combine_difference_over_common_denominator(ctx, lhs, rhs)?;
            try_limit_rules_at_finite(ctx, combined, var, point)
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            finite_mul_result(ctx, lhs_limit, rhs_limit)
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_finite(ctx, num, var, point)?;
            let den_limit = try_limit_rules_at_finite(ctx, den, var, point)?;
            finite_div_result(ctx, num_limit, den_limit)
        }
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_finite(ctx, inner, var, point)?;
            Some(finite_neg_result(ctx, inner_limit))
        }
        _ => None,
    }
}

/// Squeeze theorem at a finite point: a product converges to 0 when at
/// least one factor tends to 0 and every other factor stays bounded near
/// the point. This is the only path by which `x * sin(1/x) -> 0`
/// resolves, because `sin(1/x)` itself oscillates and has no limit.
///
/// Footprint-minimal by design: the rule fires only when at least one
/// factor is bounded WITHOUT a resolvable limit (a genuine oscillator
/// like `sin(1/x)`). When every factor has a limit, the generic Mul
/// branch already handles the product, so this rule defers (returns
/// None) and no existing result moves. Honesty is preserved because a
/// bare `sin(1/x)` is not a product (declines here) and a product with
/// no infinitesimal factor (`2*sin(1/x)`) also declines.
fn apply_finite_squeeze_bounded_product_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }
    let mut has_infinitesimal = false;
    let mut has_bounded_oscillator = false;
    for factor in collect_mul_factors(ctx, expr) {
        match classify_squeeze_factor(ctx, factor, var, point)? {
            SqueezeFactorClass::Infinitesimal => has_infinitesimal = true,
            SqueezeFactorClass::BoundedOscillator => has_bounded_oscillator = true,
            SqueezeFactorClass::FiniteLimit => {}
        }
    }
    // Need a genuine 0 factor (otherwise the product oscillates) AND a
    // bounded-but-limitless factor (otherwise the generic path owns it).
    (has_infinitesimal && has_bounded_oscillator)
        .then(|| ctx.add(Expr::Number(BigRational::zero())))
}

/// `cosh(g(x)) -> +infinity` bilaterally when the inner argument diverges to
/// infinity on BOTH sides, regardless of sign. cosh is even, so the odd pole
/// `cosh(1/x)` (inner -> -inf on the left, +inf on the right) still saturates
/// to the same +infinity on each side — a case the one-sided composition rule
/// resolves per side but the bilateral evaluator left residual because the two
/// sides reach +inf from opposite inner signs. The even outer makes the
/// bilateral limit well-defined; oscillating or sign-flipping outers are not
/// admitted (only cosh, whose fold is sign-independent).
fn apply_finite_bilateral_even_saturating_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Cosh)) {
        return None;
    }
    let inner = args[0];
    // The inner must diverge to infinity on BOTH sides (either sign).
    one_sided_inner_infinity_sign(ctx, inner, var, point, FiniteLimitSide::Left)?;
    one_sided_inner_infinity_sign(ctx, inner, var, point, FiniteLimitSide::Right)?;
    // cosh is even: cosh(+-inf) = +inf, so the bilateral value is +inf.
    saturate_outer_at_infinity(
        ctx,
        |ctx, inf| ctx.add(Expr::Function(fn_id, vec![inf])),
        InfSign::Pos,
    )
}

enum SqueezeFactorClass {
    /// Resolves to 0.
    Infinitesimal,
    /// Resolves to a finite nonzero value (bounded cofactor).
    FiniteLimit,
    /// No limit, but globally bounded near the point (e.g. sin(1/x)).
    BoundedOscillator,
}

fn classify_squeeze_factor(
    ctx: &mut Context,
    factor: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<SqueezeFactorClass> {
    if let Some(value) = try_limit_rules_at_finite(ctx, factor, var, point) {
        // A divergent factor makes the product 0 * infinity indeterminate.
        if limit_value_infinite_sign(ctx, value).is_some() {
            return None;
        }
        let is_zero =
            crate::numeric_eval::as_rational_const(ctx, value).is_some_and(|v| v.is_zero());
        return Some(if is_zero {
            SqueezeFactorClass::Infinitesimal
        } else {
            SqueezeFactorClass::FiniteLimit
        });
    }
    // No limit: admissible only as a globally bounded oscillator.
    is_globally_bounded_near_finite_point(ctx, factor, var)
        .then_some(SqueezeFactorClass::BoundedOscillator)
}

fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match ctx.get(node) {
            Expr::Mul(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            _ => factors.push(node),
        }
    }
    factors
}

/// True when `expr` has a globally bounded range near the point, so it
/// can act as a squeeze cofactor even with no limit. The bounded outer
/// functions (sin/cos/atan/arctan/tanh) saturate regardless of how their
/// argument behaves; the argument is gated to a rational function of the
/// variable, which is real on a two-sided punctured neighbourhood (a
/// real rational function is never complex where defined). That gate
/// excludes domain-restricted arguments like `ln(x)` or `sqrt(x)` whose
/// one-sided undefinedness would break the bilateral bound.
fn is_globally_bounded_near_finite_point(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return false;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    is_globally_bounded_near_finite_point_inner(ctx, expr, &var_name)
}

fn is_globally_bounded_near_finite_point_inner(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Atan
                        | BuiltinFn::Arctan
                        | BuiltinFn::Tanh,
                )
            ) && argument_is_real_rational_function(ctx, args[0], var_name)
        }
        Expr::Neg(inner) => is_globally_bounded_near_finite_point_inner(ctx, *inner, var_name),
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_globally_bounded_near_finite_point_inner(ctx, *lhs, var_name)
                && is_globally_bounded_near_finite_point_inner(ctx, *rhs, var_name)
        }
        Expr::Div(num, den) => {
            // A nonzero constant denominator keeps the quotient bounded;
            // constant_rational_value is Some only for var-free constants.
            is_globally_bounded_near_finite_point_inner(ctx, *num, var_name)
                && constant_rational_value(ctx, *den).is_some_and(|value| !value.is_zero())
        }
        _ => false,
    }
}

/// A rational function of the variable (polynomial, or a ratio of two
/// polynomials such as `1/x`). Real wherever defined, hence bounded near
/// any point for a saturating outer function.
fn argument_is_real_rational_function(ctx: &Context, arg: ExprId, var_name: &str) -> bool {
    if Polynomial::from_expr(ctx, arg, var_name).is_ok() {
        return true;
    }
    if let Expr::Div(num, den) = ctx.get(arg) {
        // The denominator must be a NONZERO polynomial. An identically-zero
        // denominator (1/(x - x) = 1/0) makes the quotient undefined on the
        // WHOLE punctured neighbourhood, so sin/cos/... of it is nowhere
        // defined and has no limit - it must not count as bounded. A
        // denominator that merely vanishes at isolated points (1/x) is fine:
        // the quotient is defined on the punctured neighbourhood.
        return Polynomial::from_expr(ctx, *num, var_name).is_ok()
            && Polynomial::from_expr(ctx, *den, var_name)
                .is_ok_and(|den_poly| !den_poly.is_zero());
    }
    false
}

/// Highest Taylor order tracked by the higher-order 0/0 quotient rule.
const TAYLOR_QUOTIENT_MAX_ORDER: usize = 12;

/// Higher-order 0/0 limits at a finite point via Taylor series:
/// `(1 - cos x)/x^2 -> 1/2`, `(sin x - x)/x^3 -> -1/6`,
/// `(e^x - 1 - x)/x^2 -> 1/2`. Both numerator and denominator are expanded
/// to a bounded order; the limit is the ratio of the lowest-order
/// coefficients when the numerator does not vanish slower than the
/// denominator. Coefficients up to the denominator's order are EXACT
/// (truncation only drops higher orders), so the value is exact.
///
/// Runs after the first-order equivalent engine, which owns the simple
/// `sin x / x` cases; this rule resolves the cancellation cases that need
/// the second/third Taylor term.
fn apply_finite_taylor_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;
    // Only the point 0 (the Taylor series are expanded at 0).
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let num_series = taylor_at_zero(ctx, num, &var_name, TAYLOR_QUOTIENT_MAX_ORDER)?;
    let den_series = taylor_at_zero(ctx, den, &var_name, TAYLOR_QUOTIENT_MAX_ORDER)?;
    let num_low = lowest_nonzero_order(&num_series.coeffs);
    let den_low = lowest_nonzero_order(&den_series.coeffs)?;
    // Genuine 0/0 only: the denominator must vanish at 0.
    if den_low == 0 {
        return None;
    }
    let den_coeff = den_series.coeffs[den_low].clone();
    match num_low {
        // Numerator is identically 0 up to the tracked order: 0 / (x^d) -> 0
        // only if it truly vanishes to order > den_low. We only know that when
        // den_low <= tracked order, which it is here; a higher-order numerator
        // term cannot change a 0 leading behaviour against x^den_low.
        None => Some(ctx.num(0)),
        Some(m) if m > den_low => Some(ctx.num(0)),
        Some(m) if m == den_low => {
            let value = &num_series.coeffs[m] / &den_coeff;
            Some(ctx.add(Expr::Number(value)))
        }
        // m < den_low: numerator vanishes slower -> divergent (DNE bilateral).
        Some(_) => None,
    }
}

/// The lowest index with a nonzero coefficient, or None when all are zero.
fn lowest_nonzero_order(coeffs: &[BigRational]) -> Option<usize> {
    coeffs.iter().position(|c| !c.is_zero())
}

thread_local! {
    /// Re-entry depth of the L'Hôpital rule. The rule evaluates the limit of the
    /// numerator and denominator by recursing into the finite cascade, which can
    /// land back here; this caps that re-entry.
    static LHOPITAL_REENTRY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Maximum L'Hôpital re-entries AND successive differentiations. Four covers the
/// university repertoire (a quartic-order 0/0 needs four passes) while bounding
/// the cost and any pathological recursion.
const MAX_LHOPITAL_DEPTH: usize = 4;

/// Recursively replace any constant subexpression with its literal value. The
/// symbolic differentiator emits unfolded arithmetic — notably exponents like
/// `(x-1)^(2-1)` — that `Polynomial::from_expr` and the continuous-limit rules
/// reject; folding `2-1` to `1` (via `as_rational_const`, which evaluates
/// arithmetic, unlike literal-only matchers) makes the differentiated form
/// consumable. Structure with the variable is preserved.
fn fold_constant_subexprs(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(value) = crate::numeric_eval::as_rational_const(ctx, expr) {
        return ctx.add(Expr::Number(value));
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let l2 = fold_constant_subexprs(ctx, l);
            let r2 = fold_constant_subexprs(ctx, r);
            ctx.add(Expr::Add(l2, r2))
        }
        Expr::Sub(l, r) => {
            let l2 = fold_constant_subexprs(ctx, l);
            let r2 = fold_constant_subexprs(ctx, r);
            ctx.add(Expr::Sub(l2, r2))
        }
        Expr::Mul(l, r) => {
            let l2 = fold_constant_subexprs(ctx, l);
            let r2 = fold_constant_subexprs(ctx, r);
            ctx.add(Expr::Mul(l2, r2))
        }
        Expr::Div(l, r) => {
            let l2 = fold_constant_subexprs(ctx, l);
            let r2 = fold_constant_subexprs(ctx, r);
            ctx.add(Expr::Div(l2, r2))
        }
        Expr::Neg(inner) => {
            let i2 = fold_constant_subexprs(ctx, inner);
            ctx.add(Expr::Neg(i2))
        }
        Expr::Pow(base, exp) => {
            let base2 = fold_constant_subexprs(ctx, base);
            let exp2 = fold_constant_subexprs(ctx, exp);
            // x^1 -> x so the polynomial recognizer sees a clean degree.
            if crate::numeric_eval::as_rational_const(ctx, exp2).is_some_and(|e| e.is_one()) {
                return base2;
            }
            ctx.add(Expr::Pow(base2, exp2))
        }
        Expr::Function(fn_id, args) => {
            let args2: Vec<ExprId> = args
                .iter()
                .map(|a| fold_constant_subexprs(ctx, *a))
                .collect();
            ctx.add(Expr::Function(fn_id, args2))
        }
        _ => expr,
    }
}

/// The rational value of a fully evaluated limit result, if it is a plain number.
fn limit_result_rational(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        _ => None,
    }
}

/// L'Hôpital's rule for a genuine 0/0 quotient at a finite NON-ZERO point.
///
/// The point 0 is owned by the equivalent-infinitesimal and Maclaurin-Taylor
/// rules above (which also carry the educational small-angle narration), so this
/// rule deliberately declines there: its job is the gap they cannot reach, a 0/0
/// whose vanishing happens at a shifted point (`sin(x)/(x-pi)` at `pi`,
/// `tan(x)/sin(x)` at `pi`, `(1-cos(x-1))/(x-1)^2` at `1`). It differentiates the
/// numerator and denominator and re-evaluates the quotient's limit, repeating
/// while the form stays 0/0.
///
/// Soundness: L'Hôpital concludes `lim f/g = lim f'/g'` ONLY when the latter
/// exists. We therefore evaluate `lim f'` and `lim g'` through the full limit
/// machinery and act only on definite finite values: if either fails to resolve,
/// or the denominator's limit stays 0 while the numerator's does not (a pole), we
/// decline and the form remains an honest residual. The point-vanishing of both
/// parts is verified before differentiating, so non-0/0 quotients are left to the
/// ordinary substitution rules.
fn apply_finite_lhopital_nonzero_point_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    // Point 0 is owned by the at-zero rules; only act on a shifted point.
    if crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num0, den0) = ctx.get(expr).clone() else {
        return None;
    };

    let depth = LHOPITAL_REENTRY_DEPTH.with(|d| d.get());
    if depth >= MAX_LHOPITAL_DEPTH {
        return None;
    }
    LHOPITAL_REENTRY_DEPTH.with(|d| d.set(depth + 1));
    let result = lhopital_evaluate(ctx, num0, den0, var, &var_name, point);
    LHOPITAL_REENTRY_DEPTH.with(|d| d.set(depth));
    result
}

fn lhopital_evaluate(
    ctx: &mut Context,
    mut num: ExprId,
    mut den: ExprId,
    var: ExprId,
    var_name: &str,
    point: ExprId,
) -> Option<ExprId> {
    for applied in 0..=MAX_LHOPITAL_DEPTH {
        // The differentiator emits unexpanded products (`2·(x-1)`); expand so the
        // polynomial/continuous rules recognize them when their limit is taken.
        // The differentiator emits unfolded exponents (`(x-1)^(2-1)`), which the
        // polynomial recognizer rejects; fold them so the limit of each part can
        // be taken by the ordinary rules.
        // The differentiator leaves arithmetic in exponents (`(x-1)^(2-1)`) that
        // the polynomial recognizer rejects; fold every constant subexpression to
        // a literal so the ordinary rules can take each part's limit.
        num = fold_constant_subexprs(ctx, num);
        den = fold_constant_subexprs(ctx, den);
        let limit_num = try_limit_rules_at_finite(ctx, num, var, point)?;
        let limit_den = try_limit_rules_at_finite(ctx, den, var, point)?;
        let num_value = limit_result_rational(ctx, limit_num)?;
        let den_value = limit_result_rational(ctx, limit_den)?;

        let num_zero = num_value.is_zero();
        let den_zero = den_value.is_zero();

        if den_zero && num_zero {
            // Genuine 0/0: differentiate both and apply L'Hôpital once more.
            if applied >= MAX_LHOPITAL_DEPTH {
                return None;
            }
            num = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx, num, var_name,
            )?;
            den = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx, den, var_name,
            )?;
            continue;
        }
        if den_zero {
            // Numerator does not vanish while the denominator does: a pole. Stay
            // an honest residual (the bilateral limit diverges or is signed-DNE).
            return None;
        }
        if applied == 0 {
            // Not a 0/0 to begin with: ordinary substitution owns this, not us.
            return None;
        }
        // Denominator's limit is nonzero after >=1 application: lim f/g = num/den.
        return Some(ctx.add(Expr::Number(num_value / den_value)));
    }
    None
}

/// Taylor coefficients of `expr` at 0 as a polynomial truncated to `order`,
/// for expressions built from polynomials, the standard analytic functions
/// (sin, cos, tan, exp, sinh, cosh, atan, asin, ln), and their sums,
/// products, integer powers, and compositions with a zero-at-0 argument.
/// Public Maclaurin expansion (expansion point `0`) of `expr` in `var_name`, truncated to
/// total degree `order` (inclusive), returned as an expression. `None` when the summand is
/// outside the supported analytic family — the same coverage the limit engine relies on
/// (polynomials, `exp`/`sin`/`cos`/`sinh`/`cosh`/`tan`/`atan`/`asin`/`ln`, their sums,
/// products, integer powers, and compositions with a series vanishing at 0). Expansion
/// points other than 0 are NOT handled here; the caller must restrict to `point = 0`.
pub fn taylor_series_at_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    order: usize,
) -> Option<ExprId> {
    let series = taylor_at_zero_with_rational(ctx, expr, var_name, order)?;
    Some(series.to_expr(ctx))
}

/// Taylor series of `expr` around a constant `point` to `order`, built directly from the
/// definition `Σ_{k=0}^{order} f^(k)(point)/k! · (var − point)^k` by repeated differentiation and
/// substitution. Handles a general expansion point (use [`taylor_series_at_zero_expr`] for the
/// Maclaurin case, whose analytic engine gives nicer closed forms). `None` if a needed
/// derivative is unavailable.
pub fn taylor_series_at_point_expr(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    point: ExprId,
    order: usize,
) -> Option<ExprId> {
    use num_bigint::BigInt;
    use num_traits::One;

    let var_id = ctx.var(var_name);
    let mut derivative = expr;
    let mut factorial = BigInt::one();
    let mut sum: Option<ExprId> = None;
    for k in 0..=order {
        // f^(k)(point) / k!
        let value = cas_ast::substitute_expr_by_id(ctx, derivative, var_id, point);
        let factorial_expr = ctx.add(Expr::Number(BigRational::from_integer(factorial.clone())));
        let coefficient = ctx.add(Expr::Div(value, factorial_expr));
        // · (var − point)^k
        let term = if k == 0 {
            coefficient
        } else {
            let shifted = ctx.add(Expr::Sub(var_id, point));
            let exponent = ctx.num(k as i64);
            let power = ctx.add(Expr::Pow(shifted, exponent));
            ctx.add(Expr::Mul(coefficient, power))
        };
        sum = Some(match sum {
            None => term,
            Some(acc) => ctx.add(Expr::Add(acc, term)),
        });
        if k < order {
            derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx, derivative, var_name,
            )?;
            factorial *= BigInt::from(k + 1);
        }
    }
    sum
}

/// Power-series reciprocal `1/den` to `order`, via the standard recurrence
/// `r_0 = 1/d_0`, `r_k = -(1/d_0)·Σ_{i=1}^{k} d_i·r_{k-i}`. Requires `den(0) ≠ 0`
/// (returns `None` otherwise — a pole at 0 has no Maclaurin expansion).
fn reciprocal_series(den: &Polynomial, order: usize, var_name: &str) -> Option<Polynomial> {
    use num_traits::{One, Zero};
    let d0 = den.coeffs.first().cloned().filter(|c| !c.is_zero())?;
    let mut r = vec![BigRational::zero(); order + 1];
    r[0] = BigRational::one() / d0.clone();
    for k in 1..=order {
        let mut acc = BigRational::zero();
        for i in 1..=k {
            let di = den.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            acc += di * r[k - i].clone();
        }
        r[k] = -acc / d0.clone();
    }
    Some(Polynomial::new(r, var_name.to_string()))
}

/// Maclaurin expansion extended to RATIONAL summands: the analytic `taylor_at_zero`
/// plus quotients `num/den` and negative integer powers `base^(-m)` whose denominator
/// is non-zero at 0 (so the function is analytic there). Kept SEPARATE from
/// `taylor_at_zero` so the limit evaluator's series path is unaffected — only the public
/// `taylor()`/`series()` command sees the rational extension.
fn taylor_at_zero_with_rational(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    order: usize,
) -> Option<Polynomial> {
    if let Some(series) = taylor_at_zero(ctx, expr, var_name, order) {
        return Some(series);
    }
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let num_series = taylor_at_zero_with_rational(ctx, num, var_name, order)?;
            let den_series = taylor_at_zero_with_rational(ctx, den, var_name, order)?;
            let recip = reciprocal_series(&den_series, order, var_name)?;
            Some(truncate_polynomial(
                &num_series.mul(&recip),
                order,
                var_name,
            ))
        }
        Expr::Pow(base, exponent) => {
            // base^(negative integer) = 1 / base^|n|.
            let exp_value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !exp_value.is_integer() || !exp_value.is_negative() {
                return None;
            }
            let m: u32 = (-exp_value.to_integer()).try_into().ok()?;
            let base_series = taylor_at_zero_with_rational(ctx, base, var_name, order)?;
            let mut den_pow = Polynomial::one(var_name.to_string());
            for _ in 0..m {
                den_pow = truncate_polynomial(&den_pow.mul(&base_series), order, var_name);
            }
            reciprocal_series(&den_pow, order, var_name)
        }
        _ => None,
    }
}

fn taylor_at_zero(ctx: &Context, expr: ExprId, var_name: &str, order: usize) -> Option<Polynomial> {
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) {
        return Some(truncate_polynomial(&poly, order, var_name));
    }
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => Some(
            taylor_at_zero(ctx, lhs, var_name, order)?
                .add(&taylor_at_zero(ctx, rhs, var_name, order)?),
        ),
        Expr::Sub(lhs, rhs) => Some(
            taylor_at_zero(ctx, lhs, var_name, order)?
                .sub(&taylor_at_zero(ctx, rhs, var_name, order)?),
        ),
        Expr::Neg(inner) => {
            let minus_one = Polynomial::new(vec![-rational_one()], var_name.to_string());
            Some(taylor_at_zero(ctx, inner, var_name, order)?.mul(&minus_one))
        }
        Expr::Mul(lhs, rhs) => {
            let l = taylor_at_zero(ctx, lhs, var_name, order)?;
            let r = taylor_at_zero(ctx, rhs, var_name, order)?;
            Some(truncate_polynomial(&l.mul(&r), order, var_name))
        }
        Expr::Pow(base, exponent) => {
            // e^arg, or [series]^n for a non-negative integer n.
            if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
                let inner = taylor_at_zero(ctx, exponent, var_name, order)?;
                return compose_standard_series(BuiltinFn::Exp, &inner, order, var_name);
            }
            let exp_value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !exp_value.is_integer() || exp_value.is_negative() {
                return None;
            }
            let n: u32 = exp_value.to_integer().try_into().ok()?;
            let base_series = taylor_at_zero(ctx, base, var_name, order)?;
            let mut acc = Polynomial::one(var_name.to_string());
            for _ in 0..n {
                acc = truncate_polynomial(&acc.mul(&base_series), order, var_name);
            }
            Some(acc)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(fn_id)?;
            let inner = taylor_at_zero(ctx, args[0], var_name, order)?;
            compose_standard_series(builtin, &inner, order, var_name)
        }
        _ => None,
    }
}

/// Compose a standard analytic function with the already-expanded inner
/// series. The inner series must satisfy the function's expansion point
/// (0 for everything except ln, which needs 1).
fn compose_standard_series(
    builtin: BuiltinFn,
    inner: &Polynomial,
    order: usize,
    var_name: &str,
) -> Option<Polynomial> {
    use num_traits::Zero;
    let zero = BigRational::zero();
    let inner_const = inner
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| zero.clone());

    if matches!(builtin, BuiltinFn::Ln) {
        // ln(arg) = ln(1 + (arg - 1)); the argument must tend to 1 at 0.
        if inner_const != rational_one() {
            return None;
        }
        let shifted = inner.sub(&Polynomial::new(vec![rational_one()], var_name.to_string()));
        let coeffs = standard_taylor_coeffs(BuiltinFn::Ln, order)?;
        return Some(compose_with_zero_inner(&coeffs, &shifted, order, var_name));
    }

    // Every other supported function expands at 0, so the inner series must
    // vanish at 0 for the composition to use the standard series.
    if !inner_const.is_zero() {
        return None;
    }
    let coeffs = standard_taylor_coeffs(builtin, order)?;
    Some(compose_with_zero_inner(&coeffs, inner, order, var_name))
}

/// Horner evaluation of `sum_k coeffs[k] * inner^k` truncated to `order`,
/// valid when `inner(0) = 0` (so `inner^k` has order >= k).
fn compose_with_zero_inner(
    coeffs: &[BigRational],
    inner: &Polynomial,
    order: usize,
    var_name: &str,
) -> Polynomial {
    let mut acc = Polynomial::new(vec![], var_name.to_string());
    for c in coeffs.iter().rev() {
        let c_poly = Polynomial::new(vec![c.clone()], var_name.to_string());
        acc = truncate_polynomial(&acc.mul(inner), order, var_name).add(&c_poly);
    }
    truncate_polynomial(&acc, order, var_name)
}

/// Coefficients `[c_0 .. c_order]` of the standard analytic functions at 0
/// (Ln is the series of `ln(1 + u)`); the i-th coefficient is generated as a
/// pure function of the index i.
fn standard_taylor_coeffs(builtin: BuiltinFn, order: usize) -> Option<Vec<BigRational>> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};
    let factorial =
        |k: usize| -> BigInt { (2..=k).fold(BigInt::one(), |acc, i| acc * BigInt::from(i)) };
    // (-1)^k as a BigInt.
    let alternating = |k: usize| -> BigInt {
        if k.is_multiple_of(2) {
            BigInt::one()
        } else {
            -BigInt::one()
        }
    };
    let zero = BigRational::zero();
    // Tan has no closed per-index formula; divide sin by cos.
    if matches!(builtin, BuiltinFn::Tan) {
        let sin = standard_taylor_coeffs(BuiltinFn::Sin, order)?;
        let cos = standard_taylor_coeffs(BuiltinFn::Cos, order)?;
        return power_series_divide(&sin, &cos, order);
    }
    let coeff = |i: usize| -> BigRational {
        let odd = !i.is_multiple_of(2);
        match builtin {
            BuiltinFn::Exp => BigRational::new(BigInt::one(), factorial(i)),
            BuiltinFn::Sin if odd => BigRational::new(alternating(i / 2), factorial(i)),
            BuiltinFn::Sinh if odd => BigRational::new(BigInt::one(), factorial(i)),
            BuiltinFn::Cos if !odd => BigRational::new(alternating(i / 2), factorial(i)),
            BuiltinFn::Cosh if !odd => BigRational::new(BigInt::one(), factorial(i)),
            BuiltinFn::Atan | BuiltinFn::Arctan if odd => {
                BigRational::new(alternating(i / 2), BigInt::from(i))
            }
            BuiltinFn::Asin | BuiltinFn::Arcsin if odd => {
                // c_(2k+1) = (2k)! / (4^k (k!)^2 (2k+1)), i = 2k+1.
                let k = i / 2;
                let denominator =
                    BigInt::from(4).pow(k as u32) * factorial(k).pow(2) * BigInt::from(i);
                BigRational::new(factorial(2 * k), denominator)
            }
            // ln(1 + u) = sum_{k>=1} (-1)^(k+1) u^k / k.
            BuiltinFn::Ln if i >= 1 => BigRational::new(alternating(i + 1), BigInt::from(i)),
            _ => zero.clone(),
        }
    };
    // Reject unsupported builtins (every supported one is matched above).
    if !matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Sinh
            | BuiltinFn::Cos
            | BuiltinFn::Cosh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Ln
    ) {
        return None;
    }
    Some((0..=order).map(coeff).collect())
}

/// Divide two power series `num / den` (den_0 != 0) to `order` terms.
fn power_series_divide(
    num: &[BigRational],
    den: &[BigRational],
    order: usize,
) -> Option<Vec<BigRational>> {
    use num_traits::Zero;
    let den0 = den.first()?;
    if den0.is_zero() {
        return None;
    }
    let mut q = vec![BigRational::zero(); order + 1];
    for n in 0..=order {
        let mut acc = num.get(n).cloned().unwrap_or_else(BigRational::zero);
        for (k, item) in den.iter().enumerate().take(n + 1).skip(1) {
            acc -= item * &q[n - k];
        }
        q[n] = acc / den0;
    }
    Some(q)
}

/// Keep only the coefficients up to `order` (drop higher-degree terms).
fn truncate_polynomial(poly: &Polynomial, order: usize, var_name: &str) -> Polynomial {
    let coeffs: Vec<BigRational> = poly.coeffs.iter().take(order + 1).cloned().collect();
    Polynomial::new(coeffs, var_name.to_string())
}

/// Resolve |var - point| by the approach side anywhere in the
/// expression (|u| = u from the right, |u| = -u from the left), then
/// re-run the one-sided chain: antiderivatives emit ln|.| forms whose
/// absolute value is sign-determined at the endpoint.
fn apply_finite_one_sided_abs_resolution(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    let rewritten = resolve_abs_shifts_for_side(ctx, expr, var, point, side);
    if rewritten == expr {
        return None;
    }
    try_limit_rules_at_finite_one_sided(ctx, rewritten, var, point, side)
}

fn resolve_abs_shifts_for_side(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> ExprId {
    let node = ctx.get(expr).clone();
    match node {
        Expr::Function(fn_id, args) => {
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Abs))
                && is_var_shift(ctx, args[0], var, point)
            {
                return match side {
                    FiniteLimitSide::Right => args[0],
                    FiniteLimitSide::Left => ctx.add(Expr::Neg(args[0])),
                };
            }
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|arg| resolve_abs_shifts_for_side(ctx, *arg, var, point, side))
                .collect();
            if new_args == args {
                expr
            } else {
                ctx.add(Expr::Function(fn_id, new_args))
            }
        }
        Expr::Add(l, r) => {
            let nl = resolve_abs_shifts_for_side(ctx, l, var, point, side);
            let nr = resolve_abs_shifts_for_side(ctx, r, var, point, side);
            if nl == l && nr == r {
                expr
            } else {
                ctx.add(Expr::Add(nl, nr))
            }
        }
        Expr::Sub(l, r) => {
            let nl = resolve_abs_shifts_for_side(ctx, l, var, point, side);
            let nr = resolve_abs_shifts_for_side(ctx, r, var, point, side);
            if nl == l && nr == r {
                expr
            } else {
                ctx.add(Expr::Sub(nl, nr))
            }
        }
        Expr::Mul(l, r) => {
            let nl = resolve_abs_shifts_for_side(ctx, l, var, point, side);
            let nr = resolve_abs_shifts_for_side(ctx, r, var, point, side);
            if nl == l && nr == r {
                expr
            } else {
                ctx.add(Expr::Mul(nl, nr))
            }
        }
        Expr::Div(l, r) => {
            let nl = resolve_abs_shifts_for_side(ctx, l, var, point, side);
            let nr = resolve_abs_shifts_for_side(ctx, r, var, point, side);
            if nl == l && nr == r {
                expr
            } else {
                ctx.add(Expr::Div(nl, nr))
            }
        }
        Expr::Neg(inner) => {
            let ni = resolve_abs_shifts_for_side(ctx, inner, var, point, side);
            if ni == inner {
                expr
            } else {
                ctx.add(Expr::Neg(ni))
            }
        }
        _ => expr,
    }
}

/// One-sided composition: scaling by constants, additive combination/// One-sided composition: scaling by constants, additive combination
/// with infinity awareness (infinity - infinity refuses), and the
/// power-log dominance u^p * ln(u)^q -> 0 (p > 0) as u -> 0+ with
/// u = var - point. Children resolve recursively through the full
/// one-sided chain, so the existing endpoint atoms compose.
/// The signed infinity of the one-sided limit of `inner`, or None when
/// it is finite or undecidable.
fn one_sided_inner_infinity_sign(
    ctx: &mut Context,
    inner: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    let value = try_limit_rules_at_finite_one_sided(ctx, inner, var, point, side)?;
    infinity_sign_of_expr(ctx, value)
}

/// Build `outer(+-inf)` and resolve it via the saturation fold; returns
/// the folded value only when the fold actually changed it (so
/// oscillating outers like sin/cos, which do not fold, decline).
fn saturate_outer_at_infinity(
    ctx: &mut Context,
    build_outer: impl FnOnce(&mut Context, ExprId) -> ExprId,
    sign: InfSign,
) -> Option<ExprId> {
    let inf = mk_infinity(ctx, sign);
    let candidate = build_outer(ctx, inf);
    let folded = crate::infinity_support::fold_infinity_saturation(ctx, candidate);
    (folded != candidate).then_some(folded)
}

fn apply_finite_one_sided_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let value = try_limit_rules_at_finite_one_sided(ctx, inner, var, point, side)?;
            Some(negate_limit_value(ctx, value))
        }
        Expr::Sub(left, right) => {
            let negated = ctx.add(Expr::Neg(right));
            let as_sum = ctx.add(Expr::Add(left, negated));
            apply_finite_one_sided_composition_rule(ctx, as_sum, var, point, side)
        }
        Expr::Add(left, right) => {
            let left_value = try_limit_rules_at_finite_one_sided(ctx, left, var, point, side)?;
            let right_value = try_limit_rules_at_finite_one_sided(ctx, right, var, point, side)?;
            combine_limit_sum(ctx, left_value, right_value)
        }
        Expr::Pow(base, exponent) => {
            // e^(g(x)) where g -> +-inf one-sided: e^(+inf)=inf, e^(-inf)=0
            // via the saturation fold (composition with a known inner
            // divergence; the bilateral case is handled at the eval layer).
            if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
                if let Some(sign) = one_sided_inner_infinity_sign(ctx, exponent, var, point, side) {
                    return saturate_outer_at_infinity(
                        ctx,
                        |ctx, inf| ctx.add(Expr::Pow(base, inf)),
                        sign,
                    );
                }
                return None;
            }
            // (var - point)^q -> 0 from the right for rational q > 0
            // (fractional powers included: the x^(3/2) endpoint atom).
            if !matches!(side, FiniteLimitSide::Right) {
                return None;
            }
            if !is_var_shift(ctx, base, var, point) {
                return None;
            }
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            value.is_positive().then(|| ctx.num(0))
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            // f(g(x)) where g -> +-inf one-sided and f saturates
            // (arctan/tanh/exp/ln/sqrt/sinh/cosh). Oscillating functions
            // do not fold, so the saturation check returns None for them.
            let arg = args[0];
            let sign = one_sided_inner_infinity_sign(ctx, arg, var, point, side)?;
            saturate_outer_at_infinity(
                ctx,
                |ctx, inf| ctx.add(Expr::Function(fn_id, vec![inf])),
                sign,
            )
        }
        Expr::Mul(left, right) => {
            if let Some(value) = power_log_dominance_zero_limit(ctx, left, right, var, point, side)
            {
                return Some(value);
            }
            if let Some(scale) = crate::numeric_eval::as_rational_const(ctx, left) {
                let value = try_limit_rules_at_finite_one_sided(ctx, right, var, point, side)?;
                return scale_limit_value(ctx, value, &scale);
            }
            if let Some(scale) = crate::numeric_eval::as_rational_const(ctx, right) {
                let value = try_limit_rules_at_finite_one_sided(ctx, left, var, point, side)?;
                return scale_limit_value(ctx, value, &scale);
            }
            let left_value = try_limit_rules_at_finite_one_sided(ctx, left, var, point, side)?;
            let right_value = try_limit_rules_at_finite_one_sided(ctx, right, var, point, side)?;
            combine_limit_product(ctx, left_value, right_value)
        }
        Expr::Div(numerator, denominator) => {
            // Unsimplified antiderivatives reach this chain as f / c
            // (e.g. x^(1/3 + 1) / (1/3 + 1) from the power rule).
            let scale = crate::numeric_eval::as_rational_const(ctx, denominator)?;
            if scale.is_zero() {
                return None;
            }
            let value = try_limit_rules_at_finite_one_sided(ctx, numerator, var, point, side)?;
            scale_limit_value(ctx, value, &scale.recip())
        }
        _ => None,
    }
}

fn combine_limit_product(ctx: &mut Context, left: ExprId, right: ExprId) -> Option<ExprId> {
    let left_sign = limit_value_infinite_sign(ctx, left);
    let right_sign = limit_value_infinite_sign(ctx, right);
    match (left_sign, right_sign) {
        (None, None) => {
            let left_const = crate::numeric_eval::as_rational_const(ctx, left);
            let right_const = crate::numeric_eval::as_rational_const(ctx, right);
            // Both factors are finite, so 0 * finite = 0 even when the other
            // factor is a non-rational symbolic value (e.g. -pi/2). The
            // indeterminate 0 * infinity case never reaches here; it is
            // resolved by the (Some, None) arms below.
            if left_const.as_ref().is_some_and(BigRational::is_zero)
                || right_const.as_ref().is_some_and(BigRational::is_zero)
            {
                return Some(ctx.add(Expr::Number(BigRational::zero())));
            }
            if let (Some(a), Some(b)) = (left_const, right_const) {
                return Some(ctx.add(Expr::Number(a * b)));
            }
            Some(ctx.add(Expr::Mul(left, right)))
        }
        // An infinite factor needs a NUMERIC nonzero cofactor to decide
        // the sign; zero times infinity is indeterminate.
        (Some(sign), None) | (None, Some(sign)) => {
            let finite = if left_sign.is_some() { right } else { left };
            let value = crate::numeric_eval::as_rational_const(ctx, finite)?;
            if value.is_zero() {
                return None;
            }
            let product_sign = if value.is_positive() { sign } else { -sign };
            Some(if product_sign > 0 {
                ctx.add(Expr::Constant(Constant::Infinity))
            } else {
                let infinity = ctx.add(Expr::Constant(Constant::Infinity));
                ctx.add(Expr::Neg(infinity))
            })
        }
        (Some(a), Some(b)) => Some(if a * b > 0 {
            ctx.add(Expr::Constant(Constant::Infinity))
        } else {
            let infinity = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(infinity))
        }),
    }
}

fn limit_value_infinite_sign(ctx: &Context, value: ExprId) -> Option<i32> {
    match ctx.get(value) {
        Expr::Constant(Constant::Infinity) => Some(1),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(-1)
        }
        _ => None,
    }
}

fn negate_limit_value(ctx: &mut Context, value: ExprId) -> ExprId {
    match limit_value_infinite_sign(ctx, value) {
        Some(1) => {
            let infinity = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(infinity))
        }
        Some(_) => ctx.add(Expr::Constant(Constant::Infinity)),
        None => ctx.add(Expr::Neg(value)),
    }
}

fn combine_limit_sum(ctx: &mut Context, left: ExprId, right: ExprId) -> Option<ExprId> {
    match (
        limit_value_infinite_sign(ctx, left),
        limit_value_infinite_sign(ctx, right),
    ) {
        (None, None) => {
            if let (Some(a), Some(b)) = (
                crate::numeric_eval::as_rational_const(ctx, left),
                crate::numeric_eval::as_rational_const(ctx, right),
            ) {
                return Some(ctx.add(Expr::Number(a + b)));
            }
            Some(ctx.add(Expr::Add(left, right)))
        }
        (Some(sign), None) | (None, Some(sign)) => Some(if sign > 0 {
            ctx.add(Expr::Constant(Constant::Infinity))
        } else {
            let infinity = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(infinity))
        }),
        (Some(a), Some(b)) if a == b => Some(if a > 0 {
            ctx.add(Expr::Constant(Constant::Infinity))
        } else {
            let infinity = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(infinity))
        }),
        // infinity - infinity: indeterminate.
        _ => None,
    }
}

fn scale_limit_value(ctx: &mut Context, value: ExprId, scale: &BigRational) -> Option<ExprId> {
    if scale.is_zero() {
        // 0 * infinity is indeterminate; plain zero scaling is exact.
        return if limit_value_infinite_sign(ctx, value).is_some() {
            None
        } else {
            Some(ctx.num(0))
        };
    }
    match limit_value_infinite_sign(ctx, value) {
        Some(sign) => {
            let scaled_sign = if scale.is_positive() { sign } else { -sign };
            Some(if scaled_sign > 0 {
                ctx.add(Expr::Constant(Constant::Infinity))
            } else {
                let infinity = ctx.add(Expr::Constant(Constant::Infinity));
                ctx.add(Expr::Neg(infinity))
            })
        }
        None => {
            if let Some(inner) = crate::numeric_eval::as_rational_const(ctx, value) {
                return Some(ctx.add(Expr::Number(scale * inner)));
            }
            let scale_expr = ctx.add(Expr::Number(scale.clone()));
            Some(ctx.add(Expr::Mul(scale_expr, value)))
        }
    }
}

/// u^p * ln(u)^q -> 0 as u -> 0 from the side where u > 0, for p > 0 and
/// q >= 1, with u = var - point (or var itself at point 0).
fn power_log_dominance_zero_limit(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    // u must approach 0 from the positive side: right of the point.
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    let (power_factor, log_factor) =
        if one_sided_log_power_of_shift(ctx, right, var, point).is_some() {
            (left, right)
        } else if one_sided_log_power_of_shift(ctx, left, var, point).is_some() {
            (right, left)
        } else {
            return None;
        };
    one_sided_log_power_of_shift(ctx, log_factor, var, point)?;
    let exponent = one_sided_positive_power_of_shift(ctx, power_factor, var, point)?;
    if exponent.is_positive() {
        Some(ctx.num(0))
    } else {
        None
    }
}

/// `sum_i c_i (var-point)^{a_i} P_i(ln(var-point)) -> 0` as `var -> point+`,
/// where every additive term carries a STRICTLY POSITIVE total power of
/// `(var-point)` and otherwise only a polynomial in `ln(var-point)` and
/// var-free constants. A positive power dominates any polynomial in the
/// logarithm, so each term -> 0 and the sum -> 0.
///
/// This generalizes `power_log_dominance_zero_limit` (a single
/// `u^p * ln(u)^q` product) to the antiderivatives of `x^a ln(x)^b`, e.g.
/// `int ln(x)^2 dx = x(ln(x)^2 - 2 ln(x) + 2)` and
/// `int ln(x)/sqrt(x) dx = 2 sqrt(x) ln(x) - 4 sqrt(x)`, whose lower
/// endpoint touches 0 and whose boundary value the definite integrator
/// needs as a one-sided limit.
fn apply_finite_one_sided_power_log_polynomial_zero(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    // ln(var - point) is real only to the right of the point.
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    if !depends_on(ctx, expr, var) {
        return None;
    }
    power_log_polynomial_sum_to_zero(ctx, expr, var, point).then(|| ctx.num(0))
}

/// Every additive term of `expr` is power-log dominated to zero.
fn power_log_polynomial_sum_to_zero(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            power_log_polynomial_sum_to_zero(ctx, lhs, var, point)
                && power_log_polynomial_sum_to_zero(ctx, rhs, var, point)
        }
        Expr::Neg(inner) => power_log_polynomial_sum_to_zero(ctx, inner, var, point),
        _ => power_log_term_dominated_to_zero(ctx, expr, var, point),
    }
}

/// A single multiplicative term tends to 0 from the right: its factors are
/// powers of `(var-point)`, polynomials in `ln(var-point)`, and var-free
/// constants, and the total `(var-point)` exponent is strictly positive.
fn power_log_term_dominated_to_zero(
    ctx: &Context,
    term: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let mut total_power = BigRational::zero();
    let mut saw_power = false;
    for factor in collect_mul_factors(ctx, term) {
        if let Some(exponent) = shift_power_exponent(ctx, factor, var, point) {
            total_power += exponent;
            saw_power = true;
        } else if is_var_shift_log_polynomial(ctx, factor, var, point) {
            // ln-polynomial growth is dominated by any positive power.
        } else if depends_on(ctx, factor, var) {
            // A factor that is neither a (var-point) power nor a log
            // polynomial (e.g. sin, exp, a foreign variable) is unclassified.
            return false;
        }
        // A var-free constant factor neither vanishes nor adds power.
    }
    saw_power && total_power.is_positive()
}

/// The exponent of a `(var-point)` power factor: the bare shift (1), its
/// sqrt (1/2), or `(var-point)^p` for rational p. None when the factor is
/// not such a power.
fn shift_power_exponent(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    if is_var_shift(ctx, expr, var, point) {
        return Some(rational_one());
    }
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_var_shift(ctx, *base, var, point) => {
            crate::numeric_eval::as_rational_const(ctx, *exp)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sqrt))
                && is_var_shift(ctx, args[0], var, point) =>
        {
            Some(BigRational::new(1.into(), 2.into()))
        }
        _ => None,
    }
}

/// `expr` is a polynomial in `ln(var-point)`: var-free constants, the bare
/// `ln(var-point)`, its non-negative integer powers, and their sums and
/// products. Restricted to integer powers because `ln(var-point) < 0` near
/// the point makes fractional powers leave the reals.
fn is_var_shift_log_polynomial(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> bool {
    if !depends_on(ctx, expr, var) {
        return true;
    }
    let is_ln_of_shift = |candidate: ExprId| -> bool {
        matches!(ctx.get(candidate), Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln))
                && is_var_shift(ctx, args[0], var, point))
    };
    if is_ln_of_shift(expr) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_ln_of_shift(*base) => {
            crate::numeric_eval::as_rational_const(ctx, *exp)
                .is_some_and(|value| value.is_integer() && !value.is_negative())
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_var_shift_log_polynomial(ctx, *lhs, var, point)
                && is_var_shift_log_polynomial(ctx, *rhs, var, point)
        }
        Expr::Neg(inner) => is_var_shift_log_polynomial(ctx, *inner, var, point),
        _ => false,
    }
}

/// Recognize (var - point)^p with rational p (var itself when point = 0),
/// including the sqrt form; returns p.
fn one_sided_positive_power_of_shift(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    if is_var_shift(ctx, expr, var, point) {
        return Some(rational_one());
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) if is_var_shift(ctx, base, var, point) => {
            crate::numeric_eval::as_rational_const(ctx, exp)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt))
                && is_var_shift(ctx, args[0], var, point) =>
        {
            Some(BigRational::new(1.into(), 2.into()))
        }
        _ => None,
    }
}

/// Recognize ln(var - point)^q (q >= 1 rational; q = 1 for the bare ln).
fn one_sided_log_power_of_shift(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    let is_ln_of_shift = |ctx: &Context, candidate: ExprId| -> bool {
        matches!(ctx.get(candidate), Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln))
                && is_var_shift(ctx, args[0], var, point))
    };
    if is_ln_of_shift(ctx, expr) {
        return Some(rational_one());
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) if is_ln_of_shift(ctx, base) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exp)?;
            (value >= rational_one()).then_some(value)
        }
        _ => None,
    }
}

/// var - point structurally: the bare var at point 0, or Sub(var, point)
/// up to numeric equality of the point.
fn is_var_shift(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> bool {
    let point_value = crate::numeric_eval::as_rational_const(ctx, point);
    if expr == var {
        return matches!(point_value, Some(value) if value.is_zero());
    }
    match ctx.get(expr) {
        Expr::Sub(l, r) => {
            *l == var
                && match (crate::numeric_eval::as_rational_const(ctx, *r), point_value) {
                    (Some(a), Some(b)) => a == b,
                    _ => *r == point,
                }
        }
        _ => false,
    }
}

fn try_limit_rules_at_finite_one_sided(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if let Some(result) = try_limit_rules_at_finite(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_one_sided_power_log_polynomial_zero(ctx, expr, var, point, side)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_zero_base_power_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_one_sided_rational_polynomial_pole_rule(ctx, expr, var, point, side)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_trig_power_pole_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_one_sided_trig_ratio_power_pole_rule(ctx, expr, var, point, side)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_log_endpoint_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_sqrt_endpoint_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_acosh_endpoint_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_composition_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_abs_resolution(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_sided_atanh_endpoint_rule(ctx, expr, var, point, side) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_one_sided_inverse_trig_endpoint_rule(ctx, expr, var, point, side)
    {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_one_sided_abs_polynomial_ratio_rule(ctx, expr, var, point, side)
    {
        return Some(result);
    }
    apply_finite_one_sided_sign_polynomial_rule(ctx, expr, var, point, side)
}

/// Rule 2: Variable - lim x = ±∞ based on approach sign.
pub fn apply_variable_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if expr != var {
        return None;
    }
    Some(mk_infinity(ctx, approach))
}

/// Rule 3: Power - lim x^n for integer n.
///
/// - n > 0: ±∞ (sign depends on approach and parity)
/// - n = 0: 1
/// - n < 0: 0
pub fn apply_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let (base, n) = parse_pow_int(ctx, expr)?;

    // Base must be exactly the limit variable
    if base != var {
        return None;
    }

    if n == 0 {
        return Some(ctx.num(1));
    }
    if n < 0 {
        return Some(ctx.num(0));
    }

    let sign = limit_sign(approach, n);
    Some(mk_infinity(ctx, sign))
}

/// Rational-exponent power at infinity: `lim_{x->+∞} x^q` for a non-integer
/// rational `q`.  The bare integer case is owned by [`apply_power_rule`] (which
/// also resolves the `x->-∞` parity); this rule covers fractional exponents,
/// where the antiderivatives of fractional-power integrands surface
/// (`∫x^(-3/2) = -2/√x`, etc.):
///
/// * `q > 0` → `+∞`
/// * `q < 0` → `0`
///
/// For `x->-∞` the base is negative and `x^q` with non-integer `q` is not real,
/// so we decline and leave the limit as an honest residual.
pub fn apply_rational_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };

    // Base must be exactly the limit variable.
    if base != var {
        return None;
    }

    // Integer exponents belong to `apply_power_rule`, which also handles the
    // sign parity for `x->-∞`.
    let q = crate::numeric_eval::as_rational_const(ctx, exp)?;
    if q.is_integer() {
        return None;
    }

    // A non-integer power of a negative magnitude is not real-valued.
    if approach == InfSign::Neg {
        return None;
    }

    if q.is_negative() {
        Some(ctx.num(0))
    } else {
        Some(mk_infinity(ctx, InfSign::Pos))
    }
}

/// Rule 4: Reciprocal power - lim c/x^n = 0 for n > 0 and c independent of x.
pub fn apply_reciprocal_power_rule(ctx: &mut Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Numerator must be constant wrt variable.
    if depends_on(ctx, num, var) {
        return None;
    }

    // Denominator must be x^n with n > 0, or plain x.
    let power = if den == var {
        1
    } else if let Some((base, n)) = parse_pow_int(ctx, den) {
        if base != var || n <= 0 {
            return None;
        }
        n
    } else {
        return None;
    };

    if power > 0 {
        Some(ctx.num(0))
    } else {
        None
    }
}

fn infinity_sign_of_expr(ctx: &Context, expr: ExprId) -> Option<InfSign> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(InfSign::Neg)
        }
        _ => None,
    }
}

fn neg_inf_sign(sign: InfSign) -> InfSign {
    match sign {
        InfSign::Pos => InfSign::Neg,
        InfSign::Neg => InfSign::Pos,
    }
}

fn negate_limit_result(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(sign) = infinity_sign_of_expr(ctx, expr) {
        return mk_infinity(ctx, neg_inf_sign(sign));
    }

    match ctx.get(expr).clone() {
        Expr::Number(value) => ctx.add(Expr::Number(-value)),
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn numeric_limit_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Neg(inner) => numeric_limit_value(ctx, *inner).map(|value| -value),
        Expr::Add(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? + numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Sub(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? - numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Mul(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? * numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Div(num, den) => {
            let den_value = numeric_limit_value(ctx, *den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(numeric_limit_value(ctx, *num)? / den_value)
        }
        _ => None,
    }
}

fn finite_numeric_expr(ctx: &mut Context, value: BigRational) -> ExprId {
    ctx.add(Expr::Number(value))
}

fn finite_limit_is_numeric_zero(ctx: &Context, expr: ExprId) -> bool {
    numeric_limit_value(ctx, expr).is_some_and(|value| value.is_zero())
}

fn finite_limit_is_numeric_one(ctx: &Context, expr: ExprId) -> bool {
    numeric_limit_value(ctx, expr).is_some_and(|value| value.is_one())
}

fn finite_add_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> ExprId {
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return finite_numeric_expr(ctx, lhs_value + rhs_value);
    }
    if finite_limit_is_numeric_zero(ctx, lhs) {
        return rhs;
    }
    if finite_limit_is_numeric_zero(ctx, rhs) {
        return lhs;
    }
    ctx.add(Expr::Add(lhs, rhs))
}

/// Combine `lhs - rhs` over a common denominator into a single fraction
/// `(num_l·den_r - num_r·den_l) / (den_l·den_r)`, treating a non-fraction term as denominator 1.
/// `None` when NEITHER side is a fraction (combining a pure polynomial difference yields no new
/// structure). Used to resolve a `∞ - ∞` finite limit: `1/sin²x - 1/x²` becomes
/// `(x² - sin²x)/(x²·sin²x)`, which the limit engine evaluates to `1/3`.
fn combine_difference_over_common_denominator(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
    let lhs_frac = match ctx.get(lhs) {
        Expr::Div(n, d) => Some((*n, *d)),
        _ => None,
    };
    let rhs_frac = match ctx.get(rhs) {
        Expr::Div(n, d) => Some((*n, *d)),
        _ => None,
    };
    if lhs_frac.is_none() && rhs_frac.is_none() {
        return None;
    }
    let one = ctx.num(1);
    let (num_l, den_l) = lhs_frac.unwrap_or((lhs, one));
    let (num_r, den_r) = rhs_frac.unwrap_or((rhs, one));
    let t1 = ctx.add(Expr::Mul(num_l, den_r));
    let t2 = ctx.add(Expr::Mul(num_r, den_l));
    let num = ctx.add(Expr::Sub(t1, t2));
    let den = ctx.add(Expr::Mul(den_l, den_r));
    Some(ctx.add(Expr::Div(num, den)))
}

fn finite_sub_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    // (±inf) - (±inf) of the SAME sign is INDETERMINATE, not 0: the `lhs == rhs` shortcut below
    // would otherwise collapse two equal interned `Constant(Infinity)` values to 0 (so
    // `lim 1/sin^2 x - 1/x^2` returned 0 instead of 1/3). Decline so the limit stays an honest
    // residual rather than a wrong value — mirroring the `0 * infinity` guard in
    // `finite_mul_result`, and matching how the engine already refuses `ln(x) - ln(x)` as inf - inf
    // (a genuine `f - f = 0` with f -> inf, like `1/x^2 - 1/x^2`, also declines). Opposite-sign
    // infinities (`+inf - (-inf) = +inf`) and `inf - finite` are DETERMINATE and fall through.
    if let (Some(lhs_sign), Some(rhs_sign)) = (
        limit_value_infinite_sign(ctx, lhs),
        limit_value_infinite_sign(ctx, rhs),
    ) {
        if lhs_sign == rhs_sign {
            return None;
        }
    }
    if lhs == rhs {
        return Some(ctx.num(0));
    }
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return Some(finite_numeric_expr(ctx, lhs_value - rhs_value));
    }
    if finite_limit_is_numeric_zero(ctx, rhs) {
        return Some(lhs);
    }
    if finite_limit_is_numeric_zero(ctx, lhs) {
        return Some(negate_limit_result(ctx, rhs));
    }
    Some(ctx.add(Expr::Sub(lhs, rhs)))
}

fn finite_mul_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return Some(finite_numeric_expr(ctx, lhs_value * rhs_value));
    }
    if finite_limit_is_numeric_zero(ctx, lhs) || finite_limit_is_numeric_zero(ctx, rhs) {
        // 0 * infinity is INDETERMINATE, not 0: decline rather than collapse a
        // divergent cofactor (x * sinh(1/x^2) -> +inf, not 0). The cofactor's
        // resolved limit is a saturated infinity here because the composition
        // rule now folds sinh(inf)/cosh(inf)/... to a bare infinity. Returning
        // None keeps the limit an honest residual instead of a wrong value.
        if limit_value_infinite_sign(ctx, lhs).is_some()
            || limit_value_infinite_sign(ctx, rhs).is_some()
        {
            return None;
        }
        return Some(ctx.num(0));
    }
    if finite_limit_is_numeric_one(ctx, lhs) {
        return Some(rhs);
    }
    if finite_limit_is_numeric_one(ctx, rhs) {
        return Some(lhs);
    }
    Some(ctx.add(Expr::Mul(lhs, rhs)))
}

fn finite_div_result(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    if !finite_denominator_proven_nonzero(ctx, den) {
        return None;
    }
    if num == den {
        return Some(ctx.num(1));
    }
    if let (Some(num_value), Some(den_value)) =
        (numeric_limit_value(ctx, num), numeric_limit_value(ctx, den))
    {
        if den_value.is_zero() {
            return None;
        }
        return Some(finite_numeric_expr(ctx, num_value / den_value));
    }
    if finite_limit_is_numeric_zero(ctx, num) {
        return Some(ctx.num(0));
    }
    if finite_limit_is_numeric_one(ctx, den) {
        return Some(num);
    }
    Some(ctx.add(Expr::Div(num, den)))
}

fn finite_neg_result(ctx: &mut Context, inner: ExprId) -> ExprId {
    if let Some(value) = numeric_limit_value(ctx, inner) {
        return finite_numeric_expr(ctx, -value);
    }
    negate_limit_result(ctx, inner)
}

fn finite_expr_proven_positive(ctx: &Context, expr: ExprId) -> bool {
    crate::prove_sign::prove_positive_depth_with(ctx, expr, 4, true, |_, _, _| {
        crate::tri_proof::TriProof::Unknown
    })
    .is_proven()
}

fn finite_denominator_proven_nonzero(ctx: &Context, expr: ExprId) -> bool {
    if numeric_limit_value(ctx, expr).is_some_and(|value| !value.is_zero()) {
        return true;
    }
    if finite_expr_proven_positive(ctx, expr) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => finite_expr_proven_positive(ctx, *inner),
        _ => false,
    }
}

fn scale_infinity(ctx: &mut Context, scale: &BigRational, sign: InfSign) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }

    let result_sign = if scale.is_positive() {
        sign
    } else {
        neg_inf_sign(sign)
    };
    Some(mk_infinity(ctx, result_sign))
}

#[derive(Debug, Clone)]
struct PolynomialGrowthInfo {
    degree: u32,
    leading_coeff: BigRational,
}

fn polynomial_growth_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() {
        return None;
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx).constant_value()?;
    Some(PolynomialGrowthInfo {
        degree,
        leading_coeff,
    })
}

fn negate_polynomial_growth_info(mut growth: PolynomialGrowthInfo) -> PolynomialGrowthInfo {
    growth.leading_coeff = -growth.leading_coeff;
    growth
}

fn scale_polynomial_growth_info(
    mut growth: PolynomialGrowthInfo,
    scale: BigRational,
) -> Option<PolynomialGrowthInfo> {
    if scale.is_zero() {
        return None;
    }
    growth.leading_coeff *= scale;
    Some(growth)
}

fn polynomial_growth_info_with_bounded_additive_noise(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<PolynomialGrowthInfo> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(growth);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach) {
                    return Some(growth);
                }
            }
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach) {
                    return Some(growth);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach) {
                    return Some(growth);
                }
            }
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach) {
                    return Some(negate_polynomial_growth_info(growth));
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)?,
                    scale,
                );
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)?,
                    scale,
                );
            }
            None
        }
        Expr::Neg(inner) => Some(negate_polynomial_growth_info(
            polynomial_growth_info_with_bounded_additive_noise(ctx, inner, var, approach)?,
        )),
        _ => None,
    }
}

fn scaled_square_root_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((BigRational::from_integer(BigInt::from(1)), radicand));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, rhs).map(|radicand| (scale, radicand));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, lhs).map(|radicand| (scale, radicand));
            }
            None
        }
        Expr::Neg(inner) => {
            scaled_square_root_base(ctx, inner).map(|(scale, radicand)| (-scale, radicand))
        }
        _ => None,
    }
}

fn scaled_abs_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Abs)) =>
        {
            Some((rational_one(), args[0]))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return scaled_abs_base(ctx, rhs)
                    .map(|(inner_scale, arg)| (scale * inner_scale, arg));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return scaled_abs_base(ctx, lhs)
                    .map(|(inner_scale, arg)| (scale * inner_scale, arg));
            }
            None
        }
        Expr::Neg(inner) => scaled_abs_base(ctx, inner).map(|(scale, arg)| (-scale, arg)),
        _ => None,
    }
}

/// Limit of `sqrt(P) - sqrt(Q)` at +-infinity for polynomials P, Q with
/// the SAME positive leading coefficient and the same degree n in
/// {1, 2}. The conjugate expansion gives sqrt(P) = sqrt(a) x^(n/2) +
/// (b_P/(2 sqrt(a))) x^(n/2 - 1) + ..., so the difference is finite when
/// n <= 2: degree-1 radicands always cancel to 0, degree-2 radicands
/// give (b_P - b_Q)/(2 sqrt(a)). Covers sqrt(x+1) - sqrt(x) = 0 and
/// sqrt(x^2+x) - sqrt(x^2-x) = 1. Higher degrees and mismatched leading
/// terms decline.
fn sqrt_minus_sqrt_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(l, r) => match ctx.get(r).clone() {
            Expr::Neg(inner) => (l, inner),
            _ => match ctx.get(l).clone() {
                Expr::Neg(inner) => (inner, r),
                _ => return None,
            },
        },
        _ => return None,
    };
    let one = BigRational::from_integer(BigInt::from(1));
    let (left_scale, left_radicand) = scaled_square_root_base(ctx, left)?;
    let (right_scale, right_radicand) = scaled_square_root_base(ctx, right)?;
    if left_scale != one || right_scale != one {
        return None;
    }
    let p = Polynomial::from_expr(ctx, left_radicand, &var_name).ok()?;
    let q = Polynomial::from_expr(ctx, right_radicand, &var_name).ok()?;
    let degree = p.degree();
    if degree != q.degree() || degree == 0 || degree > 2 {
        return None;
    }
    let a = p.coeffs.get(degree)?.clone();
    if !a.is_positive() || q.coeffs.get(degree) != Some(&a) {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    if degree == 1 {
        // sqrt(linear) - sqrt(linear): both -> +inf, difference -> 0.
        // Only defined as x -> +inf (radicand goes negative at -inf).
        if approach != InfSign::Pos {
            return None;
        }
        return Some(ctx.add(Expr::Number(zero)));
    }
    // degree == 2: (b_P - b_Q)/(2 sqrt(a)), sign flips at -inf.
    let sqrt_a = rational_sqrt(&a)?;
    let b_p = p.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let b_q = q.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let two = BigRational::from_integer(BigInt::from(2));
    let mut value = (&b_p - &b_q) / (&two * &sqrt_a);
    if approach == InfSign::Neg {
        value = -value;
    }
    Some(ctx.add(Expr::Number(value)))
}

/// Limit of `sqrt(a x^2 + b x + c) - (d x + e)` (or the reverse) at
/// +-infinity via the conjugate / leading-term expansion. With sqrt(a)
/// rational and the leading terms cancelling, the limit is finite:
/// sqrt(a x^2 + b x + c) = sqrt(a)|x| + b/(2 sqrt(a)) sign(x) + o(1).
/// Covers the classic sqrt(x^2+x) - x = 1/2, sqrt(x^2+1) - x = 0,
/// x - sqrt(x^2-x) = 1/2. Diverging cases (leading terms differ) decline
/// here and fall through to the polynomial-dominance rules.
fn sqrt_quadratic_minus_linear_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(l, r) => match ctx.get(r).clone() {
            Expr::Neg(inner) => (l, inner),
            _ => match ctx.get(l).clone() {
                Expr::Neg(inner) => (r, inner),
                _ => return None,
            },
        },
        _ => return None,
    };
    sqrt_quadratic_minus_linear_oriented(ctx, left, right, true, &var_name, approach).or_else(
        || sqrt_quadratic_minus_linear_oriented(ctx, right, left, false, &var_name, approach),
    )
}

/// One orientation: `sqrt_side - linear_side` when `sqrt_first`, else
/// `linear_side - sqrt_side`. Returns the finite limit or None.
fn sqrt_quadratic_minus_linear_oriented(
    ctx: &mut Context,
    sqrt_side: ExprId,
    linear_side: ExprId,
    sqrt_first: bool,
    var_name: &str,
    approach: InfSign,
) -> Option<ExprId> {
    let (sqrt_scale, radicand) = scaled_square_root_base(ctx, sqrt_side)?;
    if !sqrt_scale.is_positive() {
        return None;
    }
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    if radicand_poly.degree() != 2 {
        return None;
    }
    let a = radicand_poly.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }
    let b = radicand_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(|| BigRational::from_integer(BigInt::from(0)));
    let sqrt_a = rational_sqrt(&a)?;

    let linear_poly = Polynomial::from_expr(ctx, linear_side, var_name).ok()?;
    if linear_poly.degree() > 1 {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    let d = linear_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(|| zero.clone());
    let e = linear_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| zero.clone());

    let two = BigRational::from_integer(BigInt::from(2));
    // sqrt(a x^2 + b x + c) ~ sqrt(a)|x| + sign(x) b/(2 sqrt(a)).
    let (sqrt_leading, b_constant) = match approach {
        InfSign::Pos => (&sqrt_scale * &sqrt_a, &sqrt_scale * &b / (&two * &sqrt_a)),
        InfSign::Neg => (
            -(&sqrt_scale * &sqrt_a),
            -(&sqrt_scale * &b) / (&two * &sqrt_a),
        ),
    };
    let (leading, constant) = if sqrt_first {
        (&sqrt_leading - &d, &b_constant - &e)
    } else {
        (&d - &sqrt_leading, &e - &b_constant)
    };
    // Finite limit only when the leading terms cancel exactly.
    if !leading.is_zero() {
        return None;
    }
    Some(ctx.add(Expr::Number(constant)))
}

/// Limit of `factor * (radical difference)` at +infinity when the difference is
/// a conjugate form that DECAYS to 0 — the genuine `0 * inf` form the
/// multiplicative rule leaves residual (e.g. `x (sqrt(x^2+1) - x)`). Rationalizing
/// the difference by its conjugate turns it into `N(x) / (sqrt + ...)`, whose
/// leading asymptotic `K x^p` multiplies the factor's leading `c x^q`; the limit
/// is the coefficient `c K` when `p + q == 0`, is `0` when `p + q < 0`, and the
/// divergent case `p + q > 0` declines to the dominance rules. Covers the classic
/// `x (sqrt(x^2+1) - x) = 1/2`, `x (sqrt(x^2+a) - x) = a/2`,
/// `x (sqrt(x^2+2x) - x - 1) = -1/2`, and `sqrt(x) (sqrt(x+1) - sqrt(x)) = 1/2`.
/// Only defined as `x -> +inf` (radicands and sqrt factors require `x > 0`);
/// the `-inf` side declines and stays honest.
fn radical_conjugate_product_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (a, b) = match ctx.get(expr).clone() {
        Expr::Mul(a, b) => (a, b),
        _ => return None,
    };
    radical_conjugate_product_oriented(ctx, a, b, &var_name)
        .or_else(|| radical_conjugate_product_oriented(ctx, b, a, &var_name))
}

/// One orientation: `factor * diff`. Combines the factor's leading term with the
/// decaying difference's leading term; finite only when the exponents sum to 0.
fn radical_conjugate_product_oriented(
    ctx: &mut Context,
    factor: ExprId,
    diff: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (factor_coeff, factor_exp) = factor_leading_term_at_pos_inf(ctx, factor, var_name)?;
    let (diff_coeff, diff_exp) = radical_difference_asymptotic_at_pos_inf(ctx, diff, var_name)?;
    let exp = &factor_exp + &diff_exp;
    let zero = BigRational::from_integer(BigInt::from(0));
    if exp > zero {
        return None; // product diverges; leave to the dominance rules.
    }
    if exp < zero {
        return Some(ctx.add(Expr::Number(zero))); // factor cannot outrun the decay.
    }
    Some(ctx.add(Expr::Number(factor_coeff * diff_coeff)))
}

/// Asymptotic leading term `coeff * x^exp` (`coeff != 0`) of `factor` as
/// `x -> +inf`, for a polynomial factor or a `scale * sqrt(polynomial)` factor
/// with a rational leading square root. `exp` is rational (a sqrt factor halves
/// the degree). Declines anything else.
fn factor_leading_term_at_pos_inf(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) {
        let deg = poly.degree();
        let lead = poly.coeffs.get(deg)?.clone();
        if lead.is_zero() {
            return None;
        }
        return Some((lead, BigRational::from_integer(BigInt::from(deg as i64))));
    }
    let (scale, radicand) = scaled_square_root_base(ctx, expr)?;
    if !scale.is_positive() {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let deg = poly.degree();
    let lead = poly.coeffs.get(deg)?.clone();
    let sqrt_lead = rational_sqrt(&lead)?;
    Some((
        scale * sqrt_lead,
        BigRational::new(BigInt::from(deg as i64), BigInt::from(2)),
    ))
}

/// Asymptotic leading term `coeff * x^exp` of a conjugate radical difference as
/// `x -> +inf`, when the leading radical terms cancel so the difference decays.
/// Flattens the additive terms and partitions them into `scale*sqrt(polynomial)`
/// terms and a polynomial remainder. One sqrt term + linear remainder is the
/// `sqrt(quadratic) - linear` form (any orientation, any split linear tail); two
/// sqrt terms with zero remainder is `sqrt(P) - sqrt(Q)`. None otherwise.
fn radical_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut sqrt_terms: Vec<(BigRational, ExprId)> = Vec::new(); // (signed scale, radicand)
    let mut remainder: Vec<BigRational> = Vec::new(); // polynomial remainder coeffs
    for (term, positive) in terms {
        if let Some((scale, radicand)) = scaled_square_root_base(ctx, term) {
            sqrt_terms.push((if positive { scale } else { -scale }, radicand));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    match sqrt_terms.len() {
        1 => sqrt_quadratic_plus_remainder_asymptotic(ctx, &sqrt_terms[0], &remainder, var_name),
        2 => {
            if remainder.iter().any(|c| !c.is_zero()) {
                return None; // a sqrt(P)-sqrt(Q) difference carries no polynomial tail here
            }
            two_sqrt_difference_asymptotic(ctx, &sqrt_terms[0], &sqrt_terms[1], var_name)
        }
        _ => None,
    }
}

/// Asymptotic of `s*sqrt(Q) + R` with `Q` quadratic and `R` the (linear)
/// polynomial remainder, via the conjugate `s*sqrt(Q) - R`. The leading radical
/// term must cancel `R`'s linear term (`s*sqrt(a) + r1 == 0`); then the
/// rationalized numerator `s^2 Q - R^2` over the conjugate sum `~ 2 s sqrt(a) x`
/// gives the decay rate — a nonzero constant, or `K/x` when the constant cancels.
fn sqrt_quadratic_plus_remainder_asymptotic(
    ctx: &Context,
    sqrt_term: &(BigRational, ExprId),
    remainder: &[BigRational],
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (scale, radicand) = sqrt_term;
    let q = Polynomial::from_expr(ctx, *radicand, var_name).ok()?;
    if q.degree() != 2 {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    let a = q.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }
    let b = q.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let c = q.coeffs.first().cloned().unwrap_or_else(|| zero.clone());
    // Remainder must be at most linear.
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let sqrt_a = rational_sqrt(&a)?;
    let lead_sqrt = scale * &sqrt_a; // leading coeff of s*sqrt(Q)
                                     // Leading cancellation: s*sqrt(a) + r1 == 0.
    if &lead_sqrt + &r1 != zero {
        return None;
    }
    let den_lead = &lead_sqrt - &r1; // conjugate sum leading coeff = 2 s sqrt(a) != 0
    if den_lead.is_zero() {
        return None;
    }
    // N = s^2 Q - R^2 = (s^2 b - 2 r1 r0) x + (s^2 c - r0^2); x^2 cancels.
    let two = BigRational::from_integer(BigInt::from(2));
    let scale_sq = scale * scale;
    let n1 = &scale_sq * &b - &two * &r1 * &r0;
    let n0 = &scale_sq * &c - &r0 * &r0;
    if !n1.is_zero() {
        Some((n1 / den_lead, zero)) // difference -> nonzero constant
    } else if !n0.is_zero() {
        Some((n0 / den_lead, BigRational::from_integer(BigInt::from(-1)))) // difference ~ K/x
    } else {
        None // difference asymptotically 0; decline (degenerate input)
    }
}

/// Asymptotic of `s1*sqrt(P) + s2*sqrt(Q)` (a sqrt-sqrt difference) via the
/// conjugate `s1*sqrt(P) - s2*sqrt(Q)`, requiring equal degree `n` in `{1,2}`
/// and a leading cancellation `s1*sqrt(lead_P) + s2*sqrt(lead_Q) == 0`. The decay
/// rate is the leading term of `s1^2 P - s2^2 Q` over `~ 2 s1 sqrt(lead_P) x^(n/2)`.
fn two_sqrt_difference_asymptotic(
    ctx: &Context,
    left: &(BigRational, ExprId),
    right: &(BigRational, ExprId),
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (scale_l, rad_l) = left;
    let (scale_r, rad_r) = right;
    let p = Polynomial::from_expr(ctx, *rad_l, var_name).ok()?;
    let q = Polynomial::from_expr(ctx, *rad_r, var_name).ok()?;
    let n = p.degree();
    if n == 0 || n > 2 || q.degree() != n {
        return None;
    }
    let lead_p = p.coeffs.get(n)?.clone();
    let lead_q = q.coeffs.get(n)?.clone();
    if !lead_p.is_positive() || !lead_q.is_positive() {
        return None;
    }
    let sqrt_lead_p = rational_sqrt(&lead_p)?;
    let sqrt_lead_q = rational_sqrt(&lead_q)?;
    let zero = BigRational::from_integer(BigInt::from(0));
    // Leading cancellation: s1*sqrt(lead_P) + s2*sqrt(lead_Q) == 0.
    if scale_l * &sqrt_lead_p + scale_r * &sqrt_lead_q != zero {
        return None;
    }
    let den_lead = scale_l * &sqrt_lead_p - scale_r * &sqrt_lead_q; // = 2 s1 sqrt(lead_P)
    if den_lead.is_zero() {
        return None;
    }
    // N = s1^2 P - s2^2 Q; the x^n term cancels, take the next nonzero term.
    let scale_l_sq = scale_l * scale_l;
    let scale_r_sq = scale_r * scale_r;
    let mut n_lead = zero.clone();
    let mut n_deg: i64 = -1;
    for k in (0..n).rev() {
        let pk = p.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let qk = q.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let coeff_k = &scale_l_sq * &pk - &scale_r_sq * &qk;
        if !coeff_k.is_zero() {
            n_lead = coeff_k;
            n_deg = k as i64;
            break;
        }
    }
    if n_deg < 0 {
        return None; // difference asymptotically 0; decline
    }
    let n_half = BigRational::new(BigInt::from(n as i64), BigInt::from(2));
    let exp = BigRational::from_integer(BigInt::from(n_deg)) - n_half;
    Some((n_lead / den_lead, exp))
}

/// Cube-root radicand of `cbrt(P)` or `P^(1/3)`.
fn cube_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cbrt)) =>
        {
            Some(args[0])
        }
        // `P^(1/3)` -- the exponent may be a folded `Number` or an unevaluated
        // `1 / 3` quotient, so compare its numeric value.
        Expr::Pow(base, exp) => {
            let third = BigRational::new(BigInt::from(1), BigInt::from(3));
            if numeric_limit_value(ctx, *exp) == Some(third) {
                Some(*base)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `scale * cbrt(P)` in any numeric-scaled / negated form -> (scale, radicand).
fn scaled_cube_root_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Some(radicand) = cube_root_base(ctx, expr) {
        return Some((rational_one(), radicand));
    }
    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return cube_root_base(ctx, rhs).map(|radicand| (scale, radicand));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return cube_root_base(ctx, lhs).map(|radicand| (scale, radicand));
            }
            None
        }
        Expr::Neg(inner) => {
            scaled_cube_root_base(ctx, inner).map(|(scale, radicand)| (-scale, radicand))
        }
        _ => None,
    }
}

/// Asymptotic leading term `coeff * x^exp` of a cube-root conjugate difference
/// `s*cbrt(P) + R` (with `R` the polynomial remainder) as `x -> +inf`, when the
/// leading terms cancel so it decays. Rationalizing by `A^2 + A L + L^2` (with
/// `A = s*cbrt(P)`, `L = -R`): `A - L = (A^3 - L^3)/(A^2 + A L + L^2) =
/// (s^3 P - L^3)/(~ 3 d^2 x^2)`, where the leading cancellation forces
/// `s*cbrt(a) = d` (`a` the leading coeff of the cubic `P`, `d` the slope of L).
fn cbrt_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut cbrt_part: Option<(BigRational, ExprId)> = None;
    let mut remainder: Vec<BigRational> = Vec::new();
    for (term, positive) in terms {
        if let Some((scale, radicand)) = scaled_cube_root_base(ctx, term) {
            if cbrt_part.is_some() {
                return None; // only a single cube-root term is supported
            }
            cbrt_part = Some((if positive { scale } else { -scale }, radicand));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    let (scale, radicand) = cbrt_part?;
    let p = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    if p.degree() != 3 {
        return None;
    }
    let a = p.coeffs.get(3)?.clone();
    if !a.is_positive() {
        return None;
    }
    // Remainder must be at most linear.
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let cbrt_a = rational_cbrt_exact(&a)?;
    let lead = &scale * &cbrt_a; // leading coeff of s*cbrt(P)
                                 // Leading cancellation: s*cbrt(a) + r1 == 0.
    if &lead + &r1 != zero {
        return None;
    }
    // L = -R = d x + e with d = -r1 = lead (nonzero), e = -r0.
    let d = lead;
    if d.is_zero() {
        return None;
    }
    let e = -r0;
    // N = s^3 P - L^3 = s^3 P - (d x + e)^3; the x^3 term cancels, so N has
    // degree <= 2 and N / (3 d^2 x^2) gives the decay: N's x^2 term -> a nonzero
    // constant (exp 0), its x term -> K/x (exp -1), its constant -> K/x^2 (exp -2).
    let three = BigRational::from_integer(BigInt::from(3));
    let scale_cubed = &scale * &scale * &scale;
    let c2 = p.coeffs.get(2).cloned().unwrap_or_else(|| zero.clone());
    let c1 = p.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let c0 = p.coeffs.first().cloned().unwrap_or_else(|| zero.clone());
    let n2 = &scale_cubed * &c2 - &three * &d * &d * &e;
    let n1 = &scale_cubed * &c1 - &three * &d * &e * &e;
    let n0 = &scale_cubed * &c0 - &e * &e * &e;
    let den_lead = &three * &d * &d; // conjugate sum ~ 3 d^2 x^2
    if !n2.is_zero() {
        Some((n2 / den_lead, zero)) // difference -> nonzero constant
    } else if !n1.is_zero() {
        Some((n1 / den_lead, BigRational::from_integer(BigInt::from(-1)))) // ~ K/x
    } else if !n0.is_zero() {
        Some((n0 / den_lead, BigRational::from_integer(BigInt::from(-2)))) // ~ K/x^2
    } else {
        None // difference asymptotically 0; decline
    }
}

/// Limit at +infinity of `[factor *] (cube-root conjugate difference)` -- the
/// cube-root companion of `radical_conjugate_product_limit_at_infinity`. Handles
/// the bare difference (`cbrt(x^3+x^2) - x = 1/3`) and the `0 * inf` product
/// (`x^2 (cbrt(x^3+1) - x) = 1/3`). Only fires toward +inf and declines the
/// divergent (exponent-sum > 0) case to the dominance rules.
fn cbrt_conjugate_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let zero = BigRational::from_integer(BigInt::from(0));

    // Bare difference (no growing factor).
    if let Some((coeff, exp)) = cbrt_difference_asymptotic_at_pos_inf(ctx, expr, &var_name) {
        if exp > zero {
            return None;
        }
        if exp < zero {
            return Some(ctx.add(Expr::Number(zero)));
        }
        return Some(ctx.add(Expr::Number(coeff)));
    }

    // factor * difference, in either order.
    if let Expr::Mul(a, b) = ctx.get(expr).clone() {
        return cbrt_conjugate_product_oriented(ctx, a, b, &var_name)
            .or_else(|| cbrt_conjugate_product_oriented(ctx, b, a, &var_name));
    }
    None
}

fn cbrt_conjugate_product_oriented(
    ctx: &mut Context,
    factor: ExprId,
    diff: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (factor_coeff, factor_exp) = factor_leading_term_at_pos_inf(ctx, factor, var_name)?;
    let (diff_coeff, diff_exp) = cbrt_difference_asymptotic_at_pos_inf(ctx, diff, var_name)?;
    let exp = &factor_exp + &diff_exp;
    let zero = BigRational::from_integer(BigInt::from(0));
    if exp > zero {
        return None; // product diverges; leave to the dominance rules.
    }
    if exp < zero {
        return Some(ctx.add(Expr::Number(zero)));
    }
    Some(ctx.add(Expr::Number(factor_coeff * diff_coeff)))
}

/// `r^m` for a rational `r` and small `m`.
fn pow_rational(r: &BigRational, m: u32) -> BigRational {
    let mut acc = BigRational::from_integer(BigInt::from(1));
    for _ in 0..m {
        acc *= r;
    }
    acc
}

/// Binomial coefficient `C(n, k)` as a rational.
fn binomial_rational(n: u32, k: u32) -> BigRational {
    let mut acc = BigRational::from_integer(BigInt::from(1));
    for i in 0..k {
        acc *= BigRational::new(BigInt::from((n - i) as i64), BigInt::from((i + 1) as i64));
    }
    acc
}

/// `scale * (P)^(1/n)` -> (scale, radicand P, n) for an integer `n >= 2`.
fn scaled_nth_root_pow_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId, u32)> {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            let value = numeric_limit_value(ctx, exp)?;
            if *value.numer() != BigInt::from(1) {
                return None;
            }
            let n = value.denom().to_u32()?;
            if n < 2 {
                return None;
            }
            Some((rational_one(), base, n))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return scaled_nth_root_pow_base(ctx, rhs).map(|(s, p, n)| (scale * s, p, n));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return scaled_nth_root_pow_base(ctx, lhs).map(|(s, p, n)| (scale * s, p, n));
            }
            None
        }
        Expr::Neg(inner) => scaled_nth_root_pow_base(ctx, inner).map(|(s, p, n)| (-s, p, n)),
        _ => None,
    }
}

/// Asymptotic leading term `coeff * x^exp` of a general `n`-th-root conjugate
/// difference `s*(P)^(1/n) + R` (R the polynomial remainder, P degree n) as
/// `x -> +inf`, when the leading terms cancel. Rationalizing by the n-term
/// conjugate `a^(n-1)+...+b^(n-1)` (leading `n d^(n-1) x^(n-1)`) gives the decay
/// from `N = s^n P - L^n` (L = -R, degree <= n-1 once `x^n` cancels). Generalizes
/// the sqrt (n=2) and cbrt (n=3) rules to the Pow form `(P)^(1/n)`.
fn nth_root_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut root_part: Option<(BigRational, ExprId, u32)> = None; // (signed scale, P, n)
    let mut remainder: Vec<BigRational> = Vec::new();
    for (term, positive) in terms {
        if let Some((scale, radicand, n)) = scaled_nth_root_pow_base(ctx, term) {
            if root_part.is_some() {
                return None;
            }
            root_part = Some((if positive { scale } else { -scale }, radicand, n));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    let (scale, radicand, n) = root_part?;
    let p = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let n_usize = n as usize;
    if p.degree() != n_usize {
        return None;
    }
    let a = p.coeffs.get(n_usize)?.clone();
    if !a.is_positive() {
        return None;
    }
    // L = -R must be at most linear (R linear).
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let root_a = rational_nth_root(&a, n)?;
    let lead = &scale * &root_a; // leading coeff of s*(P)^(1/n)
                                 // Leading cancellation: s*a^(1/n) + r1 == 0.
    if &lead + &r1 != zero {
        return None;
    }
    let d = lead; // = -r1 (slope of L), nonzero since a > 0
    if d.is_zero() {
        return None;
    }
    let e = -r0;
    // N = s^n P - L^n with L = d x + e; the x^n term cancels. Read N's coefficients
    // directly: N_k = s^n P_k - C(n,k) d^k e^(n-k), for k from n-1 down to 0.
    let scale_n = pow_rational(&scale, n);
    let mut n_lead = zero.clone();
    let mut n_deg: i64 = -1;
    for k in (0..n_usize).rev() {
        let pk = p.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let l_k = binomial_rational(n, k as u32)
            * pow_rational(&d, k as u32)
            * pow_rational(&e, n - k as u32);
        let coeff_k = &scale_n * &pk - &l_k;
        if !coeff_k.is_zero() {
            n_lead = coeff_k;
            n_deg = k as i64;
            break;
        }
    }
    if n_deg < 0 {
        return None;
    }
    let den_lead = BigRational::from_integer(BigInt::from(n as i64)) * pow_rational(&d, n - 1);
    let exp = BigRational::from_integer(BigInt::from(n_deg))
        - BigRational::from_integer(BigInt::from((n - 1) as i64));
    Some((n_lead / den_lead, exp))
}

/// Limit at +infinity of `[factor *] ((P)^(1/n) - L)` for a general integer
/// `n >= 2`, the Pow-form companion of the sqrt and cbrt conjugate rules. Runs
/// after them, so it only newly resolves the higher roots they do not own.
fn nth_root_conjugate_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let zero = BigRational::from_integer(BigInt::from(0));

    if let Some((coeff, exp)) = nth_root_difference_asymptotic_at_pos_inf(ctx, expr, &var_name) {
        if exp > zero {
            return None;
        }
        if exp < zero {
            return Some(ctx.add(Expr::Number(zero)));
        }
        return Some(ctx.add(Expr::Number(coeff)));
    }

    if let Expr::Mul(a, b) = ctx.get(expr).clone() {
        return nth_root_conjugate_product_oriented(ctx, a, b, &var_name)
            .or_else(|| nth_root_conjugate_product_oriented(ctx, b, a, &var_name));
    }
    None
}

fn nth_root_conjugate_product_oriented(
    ctx: &mut Context,
    factor: ExprId,
    diff: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (factor_coeff, factor_exp) = factor_leading_term_at_pos_inf(ctx, factor, var_name)?;
    let (diff_coeff, diff_exp) = nth_root_difference_asymptotic_at_pos_inf(ctx, diff, var_name)?;
    let exp = &factor_exp + &diff_exp;
    let zero = BigRational::from_integer(BigInt::from(0));
    if exp > zero {
        return None;
    }
    if exp < zero {
        return Some(ctx.add(Expr::Number(zero)));
    }
    Some(ctx.add(Expr::Number(factor_coeff * diff_coeff)))
}

fn sqrt_polynomial_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (sqrt_scale, radicand) = scaled_square_root_base(ctx, num)?;
    let radicand_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || den_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if den_growth.degree != sqrt_degree {
        return None;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        let mut ratio = sqrt_scale * sqrt_leading_coeff / den_growth.leading_coeff;
        if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
            ratio = -ratio;
        }
        return Some(ctx.add(Expr::Number(ratio)));
    }

    let leading_coeff = ctx.add(Expr::Number(radicand_growth.leading_coeff));
    let sqrt_leading_coeff = ctx.call_builtin(BuiltinFn::Sqrt, vec![leading_coeff]);
    let denominator_abs = den_growth.leading_coeff.abs();
    let scale_abs = sqrt_scale.abs();
    let one = BigRational::from_integer(BigInt::from(1));
    let scaled_sqrt = if scale_abs == one {
        sqrt_leading_coeff
    } else {
        let multiplier = ctx.add(Expr::Number(scale_abs));
        ctx.add(Expr::Mul(multiplier, sqrt_leading_coeff))
    };
    let unsigned_result = if denominator_abs == one {
        scaled_sqrt
    } else {
        let denominator = ctx.add(Expr::Number(denominator_abs));
        ctx.add(Expr::Div(scaled_sqrt, denominator))
    };

    let flips_at_neg_infinity = approach == InfSign::Neg && sqrt_degree % 2 == 1;
    let needs_negation =
        sqrt_scale.is_negative() ^ den_growth.leading_coeff.is_negative() ^ flips_at_neg_infinity;
    if needs_negation {
        Some(ctx.add(Expr::Neg(unsigned_result)))
    } else {
        Some(unsigned_result)
    }
}

fn rationalized_surd_product(
    ctx: &mut Context,
    coeff: BigRational,
    radicand: BigRational,
) -> ExprId {
    if coeff.is_zero() {
        return ctx.add(Expr::Number(coeff));
    }

    let sqrt_radicand = ctx.add(Expr::Number(radicand));
    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_radicand]);
    let abs_coeff = coeff.abs();
    let one_int = BigInt::from(1);

    let numerator = if abs_coeff.numer() == &one_int {
        sqrt_expr
    } else {
        let multiplier = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.numer().clone(),
        )));
        ctx.add(Expr::Mul(multiplier, sqrt_expr))
    };

    let unsigned = if abs_coeff.denom() == &one_int {
        numerator
    } else {
        let denominator = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.denom().clone(),
        )));
        ctx.add(Expr::Div(numerator, denominator))
    };

    if coeff.is_negative() {
        ctx.add(Expr::Neg(unsigned))
    } else {
        unsigned
    }
}

fn polynomial_sqrt_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let radicand = extract_square_root_base(ctx, den)?;
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let radicand_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var, approach)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || num_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if num_growth.degree != sqrt_degree {
        return None;
    }

    let mut signed_num_lc = num_growth.leading_coeff;
    if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
        signed_num_lc = -signed_num_lc;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        return Some(ctx.add(Expr::Number(signed_num_lc / sqrt_leading_coeff)));
    }

    let coeff = signed_num_lc / radicand_growth.leading_coeff.clone();
    Some(rationalized_surd_product(
        ctx,
        coeff,
        radicand_growth.leading_coeff,
    ))
}

fn signed_abs_ratio_result(
    ctx: &mut Context,
    magnitude: BigRational,
    numerator_tail: InfSign,
    denominator_tail: InfSign,
) -> ExprId {
    let signed = if numerator_tail == denominator_tail {
        magnitude
    } else {
        -magnitude
    };
    ctx.add(Expr::Number(signed))
}

fn signed_abs_ratio_infinity(
    ctx: &mut Context,
    numerator_tail: InfSign,
    denominator_tail: InfSign,
) -> ExprId {
    let sign = if numerator_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    mk_infinity(ctx, sign)
}

fn abs_polynomial_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, num)?;
    let abs_arg_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, abs_arg, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;

    let numerator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let denominator_tail =
        limit_growth_sign(&den_growth.leading_coeff, den_growth.degree, approach);

    if abs_arg_growth.degree < den_growth.degree {
        return Some(ctx.num(0));
    }
    if abs_arg_growth.degree > den_growth.degree {
        return Some(signed_abs_ratio_infinity(
            ctx,
            numerator_tail,
            denominator_tail,
        ));
    }

    let magnitude =
        abs_scale.abs() * abs_arg_growth.leading_coeff.abs() / den_growth.leading_coeff.abs();
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        denominator_tail,
    ))
}

fn polynomial_abs_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, den)?;
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let abs_arg_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, abs_arg, var, approach)?;

    let numerator_tail = limit_growth_sign(&num_growth.leading_coeff, num_growth.degree, approach);
    let denominator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };

    if num_growth.degree < abs_arg_growth.degree {
        return Some(ctx.num(0));
    }
    if num_growth.degree > abs_arg_growth.degree {
        return Some(signed_abs_ratio_infinity(
            ctx,
            numerator_tail,
            denominator_tail,
        ));
    }

    let magnitude =
        num_growth.leading_coeff.abs() / (abs_scale.abs() * abs_arg_growth.leading_coeff.abs());
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        denominator_tail,
    ))
}

fn combine_add_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, lhs),
        infinity_sign_of_expr(ctx, rhs),
    ) {
        (Some(left), Some(right)) if left == right => Some(mk_infinity(ctx, left)),
        (Some(_), Some(_)) => None,
        (Some(sign), None) | (None, Some(sign)) => Some(mk_infinity(ctx, sign)),
        (None, None) => {
            if let (Some(lhs_value), Some(rhs_value)) =
                (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
            {
                return Some(ctx.add(Expr::Number(lhs_value + rhs_value)));
            }
            Some(ctx.add(Expr::Add(lhs, rhs)))
        }
    }
}

fn combine_sub_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, lhs),
        infinity_sign_of_expr(ctx, rhs),
    ) {
        (Some(left), Some(right)) if left == right => None,
        (Some(left), Some(right)) if left != right => Some(mk_infinity(ctx, left)),
        (Some(sign), None) => Some(mk_infinity(ctx, sign)),
        (None, Some(sign)) => Some(mk_infinity(ctx, neg_inf_sign(sign))),
        (None, None) => {
            if let (Some(lhs_value), Some(rhs_value)) =
                (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
            {
                return Some(ctx.add(Expr::Number(lhs_value - rhs_value)));
            }
            Some(ctx.add(Expr::Sub(lhs, rhs)))
        }
        _ => None,
    }
}

fn combine_mul_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    let lhs_inf = infinity_sign_of_expr(ctx, lhs);
    let rhs_inf = infinity_sign_of_expr(ctx, rhs);

    match (lhs_inf, rhs_inf) {
        (Some(left), Some(right)) => {
            let sign = if left == right {
                InfSign::Pos
            } else {
                InfSign::Neg
            };
            return Some(mk_infinity(ctx, sign));
        }
        (Some(sign), None) => {
            let scale = numeric_limit_value(ctx, rhs)?;
            return scale_infinity(ctx, &scale, sign);
        }
        (None, Some(sign)) => {
            let scale = numeric_limit_value(ctx, lhs)?;
            return scale_infinity(ctx, &scale, sign);
        }
        (None, None) => {}
    }

    // Both sides are resolved FINITE limit values; a symbolic factor
    // (pi/2 from arctan, e, ...) multiplies exactly like the additive
    // combiner already composes symbolic sums. Rational factors fold
    // through scale_limit_value so 0 * finite collapses to 0.
    if let Some(lhs_value) = numeric_limit_value(ctx, lhs) {
        return scale_limit_value(ctx, rhs, &lhs_value);
    }
    if let Some(rhs_value) = numeric_limit_value(ctx, rhs) {
        return scale_limit_value(ctx, lhs, &rhs_value);
    }
    Some(ctx.add(Expr::Mul(lhs, rhs)))
}

fn combine_div_limit_results(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    let num_inf = infinity_sign_of_expr(ctx, num);
    let den_inf = infinity_sign_of_expr(ctx, den);

    match (num_inf, den_inf) {
        (Some(_), Some(_)) => return None,
        (Some(sign), None) => {
            let den_value = numeric_limit_value(ctx, den)?;
            if den_value.is_zero() {
                return None;
            }
            return scale_infinity(
                ctx,
                &(BigRational::from_integer(BigInt::from(1)) / den_value),
                sign,
            );
        }
        (None, Some(_)) => {
            numeric_limit_value(ctx, num)?;
            return Some(ctx.num(0));
        }
        (None, None) => {}
    }

    let den_value = numeric_limit_value(ctx, den)?;
    if den_value.is_zero() {
        return None;
    }
    match numeric_limit_value(ctx, num) {
        Some(num_value) => Some(ctx.add(Expr::Number(num_value / den_value))),
        // Finite symbolic numerator over a nonzero rational divides
        // exactly; symbolic denominators stay refused (sign unknown).
        None => Some(ctx.add(Expr::Div(num, den))),
    }
}

fn limit_growth_sign(leading_coeff: &BigRational, degree: u32, approach: InfSign) -> InfSign {
    let coeff_positive = leading_coeff.is_positive();
    let power_positive = match approach {
        InfSign::Pos => true,
        InfSign::Neg => degree.is_multiple_of(2),
    };

    if coeff_positive == power_positive {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

fn linear_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let growth = polynomial_growth_info(ctx, arg, var)?;
    if growth.degree != 1 {
        return None;
    }
    Some(limit_growth_sign(
        &growth.leading_coeff,
        growth.degree,
        approach,
    ))
}

fn polynomial_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let growth = polynomial_growth_info(ctx, arg, var)?;
    Some(limit_growth_sign(
        &growth.leading_coeff,
        growth.degree,
        approach,
    ))
}

fn polynomial_or_constant_growth_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    polynomial_growth_info(ctx, expr, var).or_else(|| {
        let leading_coeff = constant_rational_value(ctx, expr)?;
        if leading_coeff.is_zero() {
            return None;
        }
        Some(PolynomialGrowthInfo {
            degree: 0,
            leading_coeff,
        })
    })
}

fn rational_polynomial_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree <= den_growth.degree {
        return None;
    }

    let degree_delta = num_growth.degree - den_growth.degree;
    let leading_ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    Some(limit_growth_sign(&leading_ratio, degree_delta, approach))
}

fn rational_polynomial_argument_zero_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree >= den_growth.degree {
        return None;
    }

    let degree_delta = den_growth.degree - num_growth.degree;
    let leading_ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    Some(limit_growth_sign(&leading_ratio, degree_delta, approach))
}

fn rational_polynomial_argument_finite_tail_value(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
) -> Option<BigRational> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree != den_growth.degree {
        return None;
    }

    Some(num_growth.leading_coeff / den_growth.leading_coeff)
}

fn unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    polynomial_argument_tail_sign(ctx, arg, var, approach)
        .or_else(|| rational_polynomial_argument_tail_sign(ctx, arg, var, approach))
        .or_else(|| abs_unbounded_argument_tail_sign(ctx, arg, var, approach))
        .or_else(|| radical_unbounded_argument_tail_sign(ctx, arg, var, approach))
}

/// sqrt(inner) and inner^q (rational q > 0) tend to +infinity whenever
/// the inner argument is unbounded toward +infinity, which lets
/// saturating compositions like arctan(sqrt(x)) resolve.
fn radical_unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(arg) {
        Expr::Neg(inner) => match unbounded_argument_tail_sign(ctx, *inner, var, approach)? {
            InfSign::Pos => Some(InfSign::Neg),
            InfSign::Neg => Some(InfSign::Pos),
        },
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sqrt)) =>
        {
            (unbounded_argument_tail_sign(ctx, args[0], var, approach)? == InfSign::Pos)
                .then_some(InfSign::Pos)
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, *exponent)?;
            if !value.is_positive() || value.is_integer() {
                return None;
            }
            (unbounded_argument_tail_sign(ctx, *base, var, approach)? == InfSign::Pos)
                .then_some(InfSign::Pos)
        }
        _ => None,
    }
}

/// |inner| tends to +infinity whenever the inner argument is unbounded
/// toward either sign, which lets compositions like ln(|x|) resolve at
/// both infinities (antiderivatives emit ln|.| forms).
fn abs_unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(arg) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs)) =>
        {
            unbounded_argument_tail_sign(ctx, args[0], var, approach).map(|_| InfSign::Pos)
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ScaledPolynomialExpTailInfo {
    coeff: BigRational,
    tail: InfSign,
}

fn linear_exp_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            linear_argument_tail_sign(ctx, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            linear_argument_tail_sign(ctx, exp, var, approach)
        }
        _ => None,
    }
}

fn polynomial_exp_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            polynomial_argument_tail_sign(ctx, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            polynomial_argument_tail_sign(ctx, exp, var, approach)
        }
        _ => None,
    }
}

fn scaled_polynomial_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledPolynomialExpTailInfo> {
    if let Some(tail) = polynomial_exp_tail_sign(ctx, expr, var, approach) {
        return Some(ScaledPolynomialExpTailInfo {
            coeff: BigRational::from_integer(BigInt::from(1)),
            tail,
        });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_polynomial_exp_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
                {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
                {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

fn nonzero_scaled_polynomial_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledPolynomialExpTailInfo> {
    let info = scaled_polynomial_exp_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

fn polynomial_or_numeric_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(limit_growth_sign(
            &growth.leading_coeff,
            growth.degree,
            approach,
        ));
    }

    let value = numeric_limit_value(ctx, expr)?;
    if value.is_zero() {
        None
    } else if value.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn rational_one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
}

fn constant_rational_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Number(value) => Some(value),
        Expr::Neg(inner) => Some(-constant_rational_value(ctx, inner)?),
        Expr::Div(num, den) => {
            let den_value = constant_rational_value(ctx, den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(constant_rational_value(ctx, num)? / den_value)
        }
        _ => None,
    }
}

fn is_rational_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value == &rational_one())
}

fn exact_named_log_base_tail_sign(ctx: &Context, base: ExprId) -> Option<InfSign> {
    if known_positive_constant_exceeds_one(ctx, base) {
        return Some(InfSign::Pos);
    }

    if let Expr::Div(num, den) = ctx.get(base).clone() {
        if is_rational_one(ctx, num) {
            return match exact_named_log_base_tail_sign(ctx, den)? {
                InfSign::Pos => Some(InfSign::Neg),
                InfSign::Neg => Some(InfSign::Pos),
            };
        }
    }

    if let Expr::Pow(pow_base, exp) = ctx.get(base).clone() {
        let base_sign = exact_named_log_base_tail_sign(ctx, pow_base)?;
        let exponent = crate::expr_extract::extract_i64_integer(ctx, exp)?;
        if exponent == 0 {
            return None;
        }
        return match (base_sign, exponent.is_positive()) {
            (InfSign::Pos, true) | (InfSign::Neg, false) => Some(InfSign::Pos),
            (InfSign::Pos, false) | (InfSign::Neg, true) => Some(InfSign::Neg),
        };
    }

    None
}

fn log_base_tail_coeff_from_sign(sign: InfSign) -> BigRational {
    match sign {
        InfSign::Pos => rational_one(),
        InfSign::Neg => -rational_one(),
    }
}

fn positive_log_base_tail_coeff(ctx: &Context, base: ExprId) -> Option<BigRational> {
    if let Some(sign) = exact_named_log_base_tail_sign(ctx, base) {
        return Some(log_base_tail_coeff_from_sign(sign));
    }

    let base_value = constant_rational_value(ctx, base)?;
    let one = rational_one();
    if !base_value.is_positive() || base_value == one {
        return None;
    }
    if base_value > one {
        Some(one)
    } else {
        Some(-one)
    }
}

fn log_argument_tail_coeff(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<BigRational> {
    if unbounded_argument_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos) {
        return Some(rational_one());
    }

    if rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos) {
        return Some(-rational_one());
    }

    None
}

fn subpolynomial_tail_coeff(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id)?;
            match (builtin, args.as_slice()) {
                (BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10, [arg]) => {
                    log_argument_tail_coeff(ctx, *arg, var, approach)
                }
                (BuiltinFn::Acosh, [arg])
                    if unbounded_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    Some(rational_one())
                }
                (BuiltinFn::Sqrt, [arg])
                    if linear_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    Some(rational_one())
                }
                (BuiltinFn::Cbrt | BuiltinFn::Asinh, [arg]) => {
                    match linear_argument_tail_sign(ctx, *arg, var, approach)? {
                        InfSign::Pos => Some(rational_one()),
                        InfSign::Neg => Some(-rational_one()),
                    }
                }
                (BuiltinFn::Log, [base, arg]) => {
                    let arg_coeff = log_argument_tail_coeff(ctx, *arg, var, approach)?;
                    Some(positive_log_base_tail_coeff(ctx, *base)? * arg_coeff)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ScaledSubpolynomialTailInfo {
    coeff: BigRational,
}

fn scaled_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    if let Some(coeff) = subpolynomial_tail_coeff(ctx, expr, var, approach) {
        return Some(ScaledSubpolynomialTailInfo { coeff });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_subpolynomial_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) = scaled_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) = scaled_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

fn nonzero_scaled_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    let info = scaled_subpolynomial_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

fn subpolynomial_tail_sign(info: &ScaledSubpolynomialTailInfo) -> Option<InfSign> {
    if info.coeff.is_zero() {
        None
    } else if info.coeff.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn signed_pi_over_two(ctx: &mut Context, sign: InfSign) -> ExprId {
    let pi_over_two = TrigValue::PiDiv(2).to_expr(ctx);
    match sign {
        InfSign::Pos => pi_over_two,
        InfSign::Neg => ctx.add(Expr::Neg(pi_over_two)),
    }
}

fn signed_unit_limit(ctx: &mut Context, sign: InfSign) -> ExprId {
    match sign {
        InfSign::Pos => ctx.num(1),
        InfSign::Neg => ctx.num(-1),
    }
}

fn unary_log_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(builtin, BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) {
        return None;
    }

    let coeff = log_argument_tail_coeff(ctx, arg, var, approach)?;
    scale_infinity(ctx, &coeff, InfSign::Pos)
}

fn elementary_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(limit) = unary_log_argument_limit_at_infinity(ctx, builtin, arg, var, approach) {
        return Some(limit);
    }

    let arg_tail = match builtin {
        // Saturating maps (and exp's signed tails) only consume the
        // tail SIGN, so any certified unbounded inner is sound.
        BuiltinFn::Acosh
        | BuiltinFn::Abs
        | BuiltinFn::Exp
        | BuiltinFn::Atan
        | BuiltinFn::Arctan
        | BuiltinFn::Tanh => unbounded_argument_tail_sign(ctx, arg, var, approach)?,
        BuiltinFn::Cbrt | BuiltinFn::Asinh | BuiltinFn::Sinh | BuiltinFn::Cosh => {
            polynomial_argument_tail_sign(ctx, arg, var, approach)?
        }
        _ => linear_argument_tail_sign(ctx, arg, var, approach)?,
    };
    match (builtin, arg_tail) {
        (BuiltinFn::Sqrt | BuiltinFn::Acosh, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Abs, _) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Atan | BuiltinFn::Arctan, tail) => Some(signed_pi_over_two(ctx, tail)),
        (BuiltinFn::Tanh, tail) => Some(signed_unit_limit(ctx, tail)),
        (BuiltinFn::Sinh, tail) => Some(mk_infinity(ctx, tail)),
        (BuiltinFn::Cosh, _) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Cbrt | BuiltinFn::Asinh, tail) => Some(mk_infinity(ctx, tail)),
        (BuiltinFn::Exp, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Exp, InfSign::Neg) => Some(ctx.num(0)),
        _ => None,
    }
}

fn binary_log_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let arg_coeff = log_argument_tail_coeff(ctx, arg, var, approach)?;
    let base_coeff = positive_log_base_tail_coeff(ctx, base)?;
    scale_infinity(ctx, &(base_coeff * arg_coeff), InfSign::Pos)
}

fn unary_linear_exp_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Ln
            | BuiltinFn::Log2
            | BuiltinFn::Log10
            | BuiltinFn::Sqrt
            | BuiltinFn::Cbrt
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Tanh
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Asinh
            | BuiltinFn::Acosh
    ) {
        return None;
    }

    let exp_tail = linear_exp_tail_sign(ctx, arg, var, approach)?;
    match builtin {
        BuiltinFn::Sqrt | BuiltinFn::Cbrt | BuiltinFn::Asinh if exp_tail == InfSign::Neg => {
            Some(ctx.num(0))
        }
        BuiltinFn::Atan | BuiltinFn::Arctan if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Atan | BuiltinFn::Arctan => Some(signed_pi_over_two(ctx, InfSign::Pos)),
        BuiltinFn::Tanh if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Tanh => Some(ctx.num(1)),
        BuiltinFn::Sinh if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Sinh => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Cosh if exp_tail == InfSign::Neg => Some(ctx.num(1)),
        BuiltinFn::Cosh => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Sqrt | BuiltinFn::Cbrt | BuiltinFn::Asinh => {
            Some(mk_infinity(ctx, InfSign::Pos))
        }
        BuiltinFn::Acosh if exp_tail == InfSign::Pos => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Acosh => None,
        _ => Some(mk_infinity(ctx, exp_tail)),
    }
}

fn binary_log_linear_exp_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let exp_tail = linear_exp_tail_sign(ctx, arg, var, approach)?;
    let base_coeff = positive_log_base_tail_coeff(ctx, base)?;
    scale_infinity(ctx, &base_coeff, exp_tail)
}

fn unary_elementary_function_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(result) = elementary_argument_limit_at_infinity(ctx, builtin, arg, var, approach) {
        return Some(result);
    }

    unary_linear_exp_argument_limit_at_infinity(ctx, builtin, arg, var, approach)
}

fn unary_log_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if !matches!(builtin, BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_positive_domain_unary_result(ctx, builtin, value_expr)
}

fn unary_sqrt_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Sqrt {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_positive_domain_unary_result(ctx, BuiltinFn::Sqrt, value_expr)
}

fn unary_sqrt_zero_tail_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Sqrt {
        return None;
    }

    if rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach)? != InfSign::Pos {
        return None;
    }

    Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))))
}

fn unary_cbrt_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Cbrt {
        return None;
    }

    if let Some(tail) = rational_polynomial_argument_tail_sign(ctx, arg, var, approach) {
        return Some(mk_infinity(ctx, tail));
    }

    if let Some(value) = rational_polynomial_argument_finite_tail_value(ctx, arg, var) {
        let value_expr = ctx.add(Expr::Number(value));
        return Some(finite_total_real_unary_result(
            ctx,
            BuiltinFn::Cbrt,
            value_expr,
        ));
    }

    rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach)?;
    Some(ctx.num(0))
}

fn unary_acosh_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Acosh {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if value <= rational_one() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_partial_domain_unary_result(ctx, BuiltinFn::Acosh, value_expr)
}

fn binary_log_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, base, var) {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_log_result(ctx, base, value_expr)
}

/// Elementary argument limits as `x -> ±∞`.
///
/// Logs and inverse hyperbolic cosine accept polynomial and rational-polynomial
/// arguments only when leading-term analysis proves a positive unbounded real
/// tail. Logs also accept rational-polynomial arguments with positive finite or
/// zero tails through explicit degree/leading-coefficient analysis. Square
/// roots accept positive rational-polynomial finite tails and path-compatible
/// positive zero tails. Cube roots, inverse hyperbolic sine, arctangent,
/// hyperbolic tangent, hyperbolic sine, and hyperbolic cosine accept signed
/// polynomial tails because they are total over the reals; cube roots also
/// accept signed rational-polynomial unbounded, finite, and zero tails. Inverse hyperbolic
/// cosine also accepts rational-polynomial finite tails only when the
/// leading-coefficient ratio lands strictly inside its real domain (`tail > 1`).
/// Other elementary families remain linear-argument only, except for already
/// documented `exp(linear)` compositions.
pub fn elementary_function_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id)?;
            match (builtin, args.as_slice()) {
                (BuiltinFn::Log, [base, arg]) => {
                    if let Some(result) =
                        binary_log_argument_limit_at_infinity(ctx, *base, *arg, var, approach)
                    {
                        return Some(result);
                    }
                    if let Some(result) =
                        binary_log_finite_rational_argument_limit_at_infinity(ctx, *base, *arg, var)
                    {
                        return Some(result);
                    }

                    binary_log_linear_exp_argument_limit_at_infinity(
                        ctx, *base, *arg, var, approach,
                    )
                }
                (_, [arg]) => {
                    if let Some(result) = unary_log_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_sqrt_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_sqrt_zero_tail_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var, approach,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_cbrt_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var, approach,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_acosh_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    unary_elementary_function_limit_at_infinity(ctx, builtin, *arg, var, approach)
                }
                _ => None,
            }
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            elementary_argument_limit_at_infinity(ctx, BuiltinFn::Exp, exp, var, approach)
        }
        _ => None,
    }
}

/// `ln(P(x)) - ln(Q(x))` at +inf collapses the indeterminate inf - inf to
/// `ln(lim P/Q)`: a finite `ln(lc_P/lc_Q)` when the degrees match, `+inf`
/// when P outgrows Q, and `-inf` when Q outgrows P. P and Q must be
/// polynomials tending to +inf (positive leading coefficient) so each
/// logarithm is eventually real. Runs before the generic Sub handling,
/// which would otherwise return the unresolved inf - inf.
fn log_difference_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Sub(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };
    let p = bare_natural_log_argument(ctx, lhs)?;
    let q = bare_natural_log_argument(ctx, rhs)?;
    let p_growth = polynomial_growth_info(ctx, p, var)?;
    let q_growth = polynomial_growth_info(ctx, q, var)?;
    // Both arguments must tend to +inf so the logarithms are eventually real.
    if !p_growth.leading_coeff.is_positive() || !q_growth.leading_coeff.is_positive() {
        return None;
    }
    use std::cmp::Ordering;
    match p_growth.degree.cmp(&q_growth.degree) {
        Ordering::Equal => {
            let ratio = p_growth.leading_coeff / q_growth.leading_coeff;
            // ln(1) = 0; the raw limit output is not fully simplified, so fold
            // the unit ratio here rather than emit a bare ln(1).
            if ratio.is_one() {
                return Some(ctx.num(0));
            }
            let ratio_expr = ctx.add(Expr::Number(ratio));
            Some(ctx.call_builtin(BuiltinFn::Ln, vec![ratio_expr]))
        }
        Ordering::Greater => Some(mk_infinity(ctx, InfSign::Pos)),
        Ordering::Less => Some(mk_infinity(ctx, InfSign::Neg)),
    }
}

/// The argument `P` of a bare `ln(P)` (natural logarithm, single argument).
fn bare_natural_log_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln)) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

/// `lim_{x->+inf} ln(sum c_i b_i^(s_i x)) / (slope x) = ln(max b_i^s_i) / slope`.
/// The dominant exponential sets the growth: with `B = max effective base`,
/// `ln(sum) = ln(B^x (c_dom + o(1))) = x ln B + O(1)`, so the quotient tends to
/// `ln(B) / slope`. Resolves `ln(2^x + 3^x)/x = ln 3`,
/// `ln(2^x + 3^x + 5^x)/x = ln 5`, `ln(5^x - 3^x)/x = ln 5`. Rational bases
/// with integer exponent slopes (so the effective base is rational and
/// comparable exactly); e-bases and mixed sums decline.
fn log_exp_sum_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{One, Signed, Zero};
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let arg = bare_natural_log_argument(ctx, num)?;
    // den = slope * x: a degree-1 polynomial with no constant term.
    let den_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    if den_poly.degree() != 1 || !den_poly.coeffs.first().is_some_and(|c| c.is_zero()) {
        return None;
    }
    let den_slope = den_poly.coeffs.get(1).cloned()?;
    if !den_slope.is_positive() {
        return None;
    }
    let mut terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, arg, &var_name, &rational_one(), &mut terms)?;
    if terms.is_empty() {
        return None;
    }
    let max_base = terms.iter().map(|(_, b)| b.clone()).max()?;
    // The dominant exponential must actually grow.
    if max_base <= BigRational::one() {
        return None;
    }
    // The dominant terms' coefficients must sum positive, so the sum -> +inf
    // and its logarithm is defined.
    let dominant_sum: BigRational = terms
        .iter()
        .filter(|(_, b)| *b == max_base)
        .map(|(c, _)| c.clone())
        .sum();
    if !dominant_sum.is_positive() {
        return None;
    }
    let base_expr = ctx.add(Expr::Number(max_base));
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
    if den_slope.is_one() {
        Some(ln_base)
    } else {
        let inv = BigRational::one() / den_slope;
        let coeff = ctx.add(Expr::Number(inv));
        Some(ctx.add(Expr::Mul(coeff, ln_base)))
    }
}

/// Accumulate `expr` (under `sign`) as a sum of rational-base exponentials
/// `c * b^(s x)` recorded as `(c, effective base b^s)`, plus constants
/// (effective base 1). Returns None for any term outside the class.
fn collect_rational_exp_terms(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    sign: &BigRational,
    terms: &mut Vec<(BigRational, BigRational)>,
) -> Option<()> {
    use num_traits::{One, Signed, Zero};
    if let Some(c) = constant_rational_value(ctx, expr) {
        terms.push((sign * &c, BigRational::one()));
        return Some(());
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let neg = -sign.clone();
            collect_rational_exp_terms(ctx, inner, var_name, &neg, terms)
        }
        Expr::Add(a, b) => {
            collect_rational_exp_terms(ctx, a, var_name, sign, terms)?;
            collect_rational_exp_terms(ctx, b, var_name, sign, terms)
        }
        Expr::Sub(a, b) => {
            collect_rational_exp_terms(ctx, a, var_name, sign, terms)?;
            let neg = -sign.clone();
            collect_rational_exp_terms(ctx, b, var_name, &neg, terms)
        }
        Expr::Mul(a, b) => {
            if let Some(scale) = constant_rational_value(ctx, a) {
                let s = sign * &scale;
                return collect_rational_exp_terms(ctx, b, var_name, &s, terms);
            }
            if let Some(scale) = constant_rational_value(ctx, b) {
                let s = sign * &scale;
                return collect_rational_exp_terms(ctx, a, var_name, &s, terms);
            }
            None
        }
        _ => {
            // b^(s x): rational base b > 0, exponent a positive-integer multiple
            // of x, so the effective base b^s is a comparable rational.
            let (base, exponent) = numeric_base_power(ctx, expr)?;
            if !base.is_positive() {
                return None;
            }
            let exp_poly = Polynomial::from_expr(ctx, exponent, var_name).ok()?;
            if exp_poly.degree() != 1 || !exp_poly.coeffs.first().is_some_and(|c| c.is_zero()) {
                return None;
            }
            let slope = exp_poly.coeffs.get(1).cloned()?;
            if !slope.is_integer() || !slope.is_positive() {
                return None;
            }
            let s = u32::try_from(slope.to_integer()).ok()?;
            if s > 64 {
                return None;
            }
            let effective_base = num_traits::pow::pow(base, s as usize);
            terms.push((sign.clone(), effective_base));
            Some(())
        }
    }
}

pub fn additive_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_add_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_sub_limit_results(ctx, lhs_limit, rhs_limit)
        }
        _ => None,
    }
}

pub fn multiplicative_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_infinity(ctx, inner, var, approach)?;
            Some(negate_limit_result(ctx, inner_limit))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_mul_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_infinity(ctx, num, var, approach)?;
            let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
            combine_div_limit_results(ctx, num_limit, den_limit)
        }
        _ => None,
    }
}

fn is_eventually_nonzero_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if constant_rational_value(ctx, expr).is_some_and(|value| !value.is_zero()) {
        return true;
    }
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_argument_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    if rational_polynomial_argument_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    if rational_polynomial_argument_zero_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    rational_polynomial_argument_finite_tail_value(ctx, expr, var)
        .is_some_and(|value| !value.is_zero())
}

fn is_eventually_positive_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if constant_rational_value(ctx, expr).is_some_and(|value| value.is_positive()) {
        return true;
    }
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_argument_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    if rational_polynomial_argument_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    if rational_polynomial_argument_zero_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    rational_polynomial_argument_finite_tail_value(ctx, expr, var)
        .is_some_and(|value| value.is_positive())
}

fn is_eventually_real_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_growth_info(ctx, expr, var).is_some() {
        return true;
    }

    match ctx.get(expr) {
        Expr::Variable(_) | Expr::Number(_) => true,
        Expr::Constant(Constant::I | Constant::Infinity | Constant::Undefined) => false,
        Expr::Neg(inner) | Expr::Hold(inner) => {
            is_eventually_real_expr_at_infinity(ctx, *inner, var, approach)
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_eventually_real_expr_at_infinity(ctx, *lhs, var, approach)
                && is_eventually_real_expr_at_infinity(ctx, *rhs, var, approach)
        }
        Expr::Div(num, den) => {
            is_eventually_real_expr_at_infinity(ctx, *num, var, approach)
                && is_eventually_real_expr_at_infinity(ctx, *den, var, approach)
                && is_eventually_nonzero_expr_at_infinity(ctx, *den, var, approach)
        }
        Expr::Pow(base, exp) => {
            if let Some((_pow_base, power)) = parse_pow_int(ctx, expr) {
                if power < 0 && !is_eventually_nonzero_expr_at_infinity(ctx, *base, var, approach) {
                    return false;
                }
                return is_eventually_real_expr_at_infinity(ctx, *base, var, approach);
            }
            if constant_rational_value(ctx, *exp).is_some_and(|value| {
                value.denom() == &BigInt::from(2) && value.numer() == &BigInt::from(1)
            }) {
                return is_eventually_positive_expr_at_infinity(ctx, *base, var, approach);
            }
            false
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = args[0];
            match ctx.builtin_of(*fn_id) {
                Some(
                    BuiltinFn::Exp
                    | BuiltinFn::Sin
                    | BuiltinFn::Cos
                    | BuiltinFn::Atan
                    | BuiltinFn::Arctan
                    | BuiltinFn::Tanh
                    | BuiltinFn::Sinh
                    | BuiltinFn::Cosh
                    | BuiltinFn::Asinh
                    | BuiltinFn::Cbrt
                    | BuiltinFn::Abs,
                ) => is_eventually_real_expr_at_infinity(ctx, arg, var, approach),
                Some(BuiltinFn::Sqrt | BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) => {
                    is_eventually_positive_expr_at_infinity(ctx, arg, var, approach)
                }
                Some(BuiltinFn::Acosh) => {
                    unbounded_argument_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos)
                        || rational_polynomial_argument_finite_tail_value(ctx, arg, var)
                            .is_some_and(|value| value > rational_one())
                        || constant_rational_value(ctx, arg)
                            .is_some_and(|value| value > rational_one())
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn is_bounded_elementary_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if !depends_on(ctx, expr, var) {
        return is_eventually_real_expr_at_infinity(ctx, expr, var, approach);
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Atan
                        | BuiltinFn::Arctan
                        | BuiltinFn::Tanh,
                )
            ) && is_eventually_real_expr_at_infinity(ctx, args[0], var, approach)
        }
        Expr::Neg(inner) => is_bounded_elementary_expr_at_infinity(ctx, *inner, var, approach),
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_bounded_elementary_expr_at_infinity(ctx, *lhs, var, approach)
                && is_bounded_elementary_expr_at_infinity(ctx, *rhs, var, approach)
        }
        Expr::Div(num, den) => {
            is_bounded_elementary_expr_at_infinity(ctx, *num, var, approach)
                && !depends_on(ctx, *den, var)
                && constant_rational_value(ctx, *den).is_some_and(|value| !value.is_zero())
        }
        _ => false,
    }
}

/// Conservative bounded-over-divergent rule as `x -> ±∞`.
///
/// This is intentionally narrow: only real-domain globally bounded elementary
/// numerators (`sin`/`cos`/`arctan`/`tanh`, plus finite arithmetic combinations
/// of bounded pieces and constants) are accepted, and the denominator must
/// already be proven divergent by the existing infinity rules.
pub fn bounded_elementary_over_divergent_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    if !depends_on(ctx, num, var)
        || !is_bounded_elementary_expr_at_infinity(ctx, num, var, approach)
    {
        return None;
    }

    let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
    infinity_sign_of_expr(ctx, den_limit)?;
    Some(ctx.num(0))
}

fn bounded_elementary_times_decaying_exp_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach)
        {
            return Some(ctx.num(0));
        }
    }

    if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach)
        {
            return Some(ctx.num(0));
        }
    }

    None
}

/// Dominance rule for polynomial-argument exponentials against polynomial growth as `x -> ±∞`.
///
/// This is intentionally narrow: only `exp(p(x))`/`e^(p(x))` with optional
/// numeric scaling is compared with polynomials whose relevant leading
/// coefficient is numeric. Parameterized exponent tails and nested
/// exponentials remain residual.
pub fn exponential_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &(-info.coeff), InfSign::Pos);
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, den, var, approach)
            {
                if den_info.tail == InfSign::Pos {
                    polynomial_or_numeric_tail_sign(ctx, num, var, approach)?;
                    return Some(ctx.num(0));
                }

                if let Some(num_sign) = polynomial_or_numeric_tail_sign(ctx, num, var, approach) {
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, num, var, approach)
            {
                if num_info.tail != InfSign::Pos {
                    return None;
                }
                let den_sign = polynomial_or_numeric_tail_sign(ctx, den, var, approach)?;
                return scale_infinity(ctx, &num_info.coeff, den_sign);
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, rhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            if let Some(info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, lhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            None
        }
        _ => None,
    }
}

/// Dominance rule for domain-safe subpolynomial tails against polynomial growth.
///
/// Logs and inverse hyperbolic cosine accept polynomial and rational-polynomial
/// arguments only when the argument tends to `+∞`, which keeps the real-domain
/// tail explicit. Square roots remain linear-only here; their polynomial
/// arguments can carry polynomial-scale growth. Cube roots and inverse
/// hyperbolic sine are total-real, so their signed linear tails are preserved.
/// Unsupported nonlinear arguments and
/// subpolynomial-vs-subpolynomial comparisons remain residual.
pub fn subpolynomial_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            if scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, neg_inf_sign(sign)));
            }
            if scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Div(num, den) => {
            if scaled_subpolynomial_tail_info(ctx, num, var, approach).is_some() {
                polynomial_growth_info(ctx, den, var)?;
                return Some(ctx.num(0));
            }

            if let Some(den_info) = nonzero_scaled_subpolynomial_tail_info(ctx, den, var, approach)
            {
                let growth = polynomial_growth_info(ctx, num, var)?;
                let num_sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return scale_infinity(
                    ctx,
                    &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                    num_sign,
                );
            }
            None
        }
        _ => None,
    }
}

/// Any positive power of the variable dominates any power of the logarithm:
/// `c ln(x)^a / x^b -> 0` and `c x^b / ln(x)^a -> sign(c) * inf`, for a >= 1
/// integer and b > 0 rational (fractional included). This generalizes the
/// subpolynomial/polynomial rule (single `ln(x)` over an INTEGER power) to
/// higher log powers (`ln(x)^2 / x`) and fractional powers (`ln(x)/sqrt(x)`).
/// Only meaningful as `var -> +inf` (the logarithm needs `x > 0`).
fn polylog_power_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // c ln(x)^a / x^b -> 0 (both coefficients nonzero).
    if let (Some((num_coeff, _)), Some((den_coeff, _))) = (
        constant_times_log_power(ctx, num, var, approach),
        positive_power_tail(ctx, den, var),
    ) {
        if !num_coeff.is_zero() && !den_coeff.is_zero() {
            return Some(ctx.num(0));
        }
    }
    // c x^b / (c' ln(x)^a) -> sign(c / c') * inf.
    if let (Some((num_coeff, _)), Some((den_coeff, _))) = (
        positive_power_tail(ctx, num, var),
        constant_times_log_power(ctx, den, var, approach),
    ) {
        if !num_coeff.is_zero() && !den_coeff.is_zero() {
            let sign = if (num_coeff / den_coeff).is_positive() {
                InfSign::Pos
            } else {
                InfSign::Neg
            };
            return Some(mk_infinity(ctx, sign));
        }
    }
    None
}

/// `c * ln(x)^a` (a >= 1 integer, c != 0) with the logarithm's argument
/// tending to +inf; returns (c, a). Recognizes the bare log, its integer
/// powers, a numeric scale, and a negation.
fn constant_times_log_power(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<(BigRational, i64)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (c, a) = constant_times_log_power(ctx, inner, var, approach)?;
            Some((-c, a))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                let (c, a) = constant_times_log_power(ctx, rhs, var, approach)?;
                return Some((scale * c, a));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                let (c, a) = constant_times_log_power(ctx, lhs, var, approach)?;
                return Some((scale * c, a));
            }
            None
        }
        Expr::Pow(base, exponent) if is_unbounded_log(ctx, base, var, approach) => {
            let exp = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if exp.is_integer() && exp.is_positive() {
                Some((rational_one(), exp.to_integer().try_into().ok()?))
            } else {
                None
            }
        }
        _ if is_unbounded_log(ctx, expr, var, approach) => Some((rational_one(), 1)),
        _ => None,
    }
}

/// `ln(arg)` / `log2(arg)` / `log10(arg)` with `arg -> +inf`.
fn is_unbounded_log(ctx: &Context, expr: ExprId, var: ExprId, approach: InfSign) -> bool {
    matches!(ctx.get(expr), Expr::Function(fn_id, args)
        if args.len() == 1
            && matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10)
            )
            && log_argument_tail_coeff(ctx, args[0], var, approach).is_some_and(|c| c.is_positive()))
}

/// `c * x^b` with `b > 0` rational (fractional included): the bare variable,
/// its rational powers, `sqrt(var)`, and a numeric scale. Returns (c, b).
fn positive_power_tail(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<(BigRational, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (c, b) = positive_power_tail(ctx, inner, var)?;
            Some((-c, b))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                let (c, b) = positive_power_tail(ctx, rhs, var)?;
                return Some((scale * c, b));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                let (c, b) = positive_power_tail(ctx, lhs, var)?;
                return Some((scale * c, b));
            }
            None
        }
        Expr::Pow(base, exponent) if base == var => {
            let exp = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            exp.is_positive().then_some((rational_one(), exp))
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt))
                && args[0] == var =>
        {
            Some((
                rational_one(),
                BigRational::new(BigInt::from(1), BigInt::from(2)),
            ))
        }
        _ if expr == var => Some((rational_one(), rational_one())),
        _ => None,
    }
}

fn scaled_exp_subpoly_product_limit(
    ctx: &mut Context,
    exp_info: ScaledPolynomialExpTailInfo,
    subpoly_info: ScaledSubpolynomialTailInfo,
) -> Option<ExprId> {
    if exp_info.coeff.is_zero() || subpoly_info.coeff.is_zero() || exp_info.tail == InfSign::Neg {
        return Some(ctx.num(0));
    }

    scale_infinity(ctx, &(exp_info.coeff * subpoly_info.coeff), InfSign::Pos)
}

/// Dominance rule for polynomial-argument exponentials against domain-safe subpolynomial tails.
///
/// Both sides are accepted only through existing tail analyzers: exponential
/// exponents must have a non-parametric polynomial tail; logarithms and inverse
/// hyperbolic cosine may carry positive polynomial tails; square-root arguments
/// must tend to `+∞` on a linear real tail; cube roots and inverse hyperbolic
/// sine may carry either signed linear tail. Parametric or nested exponential
/// tails remain residual.
pub fn exponential_subpolynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &(-exp_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if let Some(subpoly_info) = scaled_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if let Some(subpoly_info) = scaled_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, den, var, approach)
            {
                if let Some(num_info) = scaled_subpolynomial_tail_info(ctx, num, var, approach) {
                    let Some(num_sign) = subpolynomial_tail_sign(&num_info) else {
                        return Some(ctx.num(0));
                    };
                    if den_info.tail == InfSign::Pos {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = scaled_polynomial_exp_tail_info(ctx, num, var, approach) {
                if let Some(den_info) =
                    nonzero_scaled_subpolynomial_tail_info(ctx, den, var, approach)
                {
                    if num_info.coeff.is_zero() || num_info.tail == InfSign::Neg {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(ctx, &(num_info.coeff / den_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        _ => None,
    }
}

/// Polynomial limit rule for `P(x)` as `x -> ±∞`.
///
/// This handles polynomial expressions whose leading coefficient in `var` is a
/// numeric constant. Symbolic leading coefficients remain unresolved because
/// their sign is not known under the conservative limit policy.
pub fn polynomial_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() {
        return Some(ctx.num(0));
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx);
    let leading_value = leading_coeff.constant_value()?;
    let sign = limit_growth_sign(&leading_value, degree, approach);
    Some(mk_infinity(ctx, sign))
}

/// Rational polynomial limit rule for `P(x)/Q(x)` as `x -> ±∞`.
///
/// Compares polynomial degrees in `var`:
/// - `deg(P) < deg(Q) -> 0`
/// - `deg(P) = deg(Q) -> lc(P)/lc(Q)` when both leading coefficients are numeric
/// - `deg(P) > deg(Q) -> ±∞` according to leading coefficient sign and parity
pub fn rational_poly_limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    // Match Div(num, den)
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Get variable name
    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    // Conservative budget for polynomial conversion
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    // Convert numerator and denominator to polynomials
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Get variable index in polynomial
    // If var not in poly, it's constant wrt var (degree 0)
    let var_idx_num = p_num.var_index(var_name);
    let var_idx_den = p_den.var_index(var_name);

    // If neither contains the variable, constant rule handles it
    if var_idx_num.is_none() && var_idx_den.is_none() {
        return None; // Let constant rule handle it
    }

    // Check for zero denominator polynomial
    if p_den.is_zero() {
        return None; // Division by zero - don't handle here
    }

    // Get degrees
    let deg_p = var_idx_num.map(|idx| p_num.degree_in(idx)).unwrap_or(0);
    let deg_q = var_idx_den.map(|idx| p_den.degree_in(idx)).unwrap_or(0);

    // Get leading coefficients
    let lc_p = var_idx_num
        .map(|idx| p_num.leading_coeff_in(idx))
        .unwrap_or_else(|| p_num.clone());
    let lc_q = var_idx_den
        .map(|idx| p_den.leading_coeff_in(idx))
        .unwrap_or_else(|| p_den.clone());

    // Both leading coefficients must be numeric constants
    let lc_p_val = lc_p.constant_value()?;
    let lc_q_val = lc_q.constant_value()?;

    // Case 1: deg(P) < deg(Q) -> 0
    if deg_p < deg_q {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // Case 2: deg(P) = deg(Q) -> lc(P)/lc(Q)
    if deg_p == deg_q {
        let ratio = &lc_p_val / &lc_q_val;
        return Some(ctx.add(Expr::Number(ratio)));
    }

    // Case 3: deg(P) > deg(Q) -> ±∞
    // Sign = sign(lc_p/lc_q) * sign(x^k) where k = deg_p - deg_q
    let k = deg_p - deg_q;
    let ratio = &lc_p_val / &lc_q_val;
    let sign = limit_growth_sign(&ratio, k, approach);
    Some(mk_infinity(ctx, sign))
}

/// Try all limit-at-infinity rules in conservative order.
///
/// Order:
/// 1. Constant
/// 2. Variable
/// 3. Power
/// 4. Reciprocal power
/// 5. Elementary exact-argument functions
/// 6. Additive combination
/// 7. Determinate multiplicative combination
/// 8. Bounded trig over divergent denominator
/// 9. Square-root polynomial ratio with matching growth
/// 10. Polynomial over square-root polynomial with matching growth
/// 11. Absolute-value polynomial ratio with matching growth
/// 12. Polynomial over absolute-value polynomial with matching growth
/// 13. Exact exponential-vs-polynomial dominance
/// 14. Exact subpolynomial-vs-polynomial dominance
/// 15. Exact exponential-vs-subpolynomial dominance
/// 16. Polynomial
/// 17. Rational polynomial
pub fn try_limit_rules_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(r) = apply_static_empty_real_domain_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = apply_constant_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = apply_variable_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_rational_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_reciprocal_power_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = one_to_infinity_power_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = general_base_exponential_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = inf_to_zero_power_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = elementary_function_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = log_exp_sum_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = log_difference_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = product_log_unit_argument_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = additive_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = multiplicative_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = bounded_elementary_over_divergent_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) =
        bounded_elementary_times_decaying_exp_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = sqrt_quadratic_minus_linear_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = sqrt_minus_sqrt_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = radical_conjugate_product_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = cbrt_conjugate_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = nth_root_conjugate_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = sqrt_polynomial_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_sqrt_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = abs_polynomial_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_abs_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exponential_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = subpolynomial_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = polylog_power_dominance_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exponential_subpolynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = polynomial_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = bounded_noise_rational_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = rational_difference_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exp_sum_quotient_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = general_exp_vs_polynomial_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_times_decaying_exponential_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    rational_poly_limit(ctx, expr, var, approach)
}

/// A polynomial times a DECAYING general-base exponential at infinity tends to
/// 0: the exponential decay outruns any polynomial growth. Covers
/// `x * 2^(-x) -> 0` (at +inf) and `x * 2^x -> 0` (at -inf). The product is the
/// 0 * inf form the generic Mul branch leaves residual; here the 0 factor is a
/// genuine exponential, so the product is 0 regardless of the polynomial sign.
fn polynomial_times_decaying_exponential_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::Zero;
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };
    for (poly_cand, exp_cand) in [(lhs, rhs), (rhs, lhs)] {
        let is_polynomial =
            Polynomial::from_expr(ctx, poly_cand, &var_name).is_ok_and(|p| p.degree() >= 1);
        if !is_polynomial || numeric_base_power(ctx, exp_cand).is_none() {
            continue;
        }
        // The exponential factor must DECAY (limit 0) so it dominates the poly.
        let exp_limit = general_base_exponential_limit_at_infinity(ctx, exp_cand, var, approach)?;
        if crate::numeric_eval::as_rational_const(ctx, exp_limit).is_some_and(|v| v.is_zero()) {
            return Some(ctx.num(0));
        }
    }
    None
}

/// A quotient of a general-base exponential sum against a polynomial at
/// infinity: an exponential `b^x` with `b > 1` beats any polynomial, so
/// `(sum b_i^x) / poly -> +-inf` and `poly / (sum b_i^x) -> 0`. The engine
/// already settles the natural base; this covers `2^x / x^2 -> +inf`,
/// `x^10 / 2^x -> 0`. Reuses the effective-base parser for the exponential
/// side and the polynomial reader for the other.
fn general_exp_vs_polynomial_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // Numerator is an exponential sum, denominator a polynomial: exp wins.
    if let Some(num_sign) = exp_sum_dominant_sign(ctx, num, &var_name) {
        if let Some(den_sign) = positive_degree_polynomial_tail_sign(ctx, den, &var_name) {
            let positive = num_sign == den_sign;
            return Some(mk_infinity(
                ctx,
                if positive { InfSign::Pos } else { InfSign::Neg },
            ));
        }
    }
    // Numerator is a polynomial, denominator an exponential sum: poly loses.
    if positive_degree_polynomial_tail_sign(ctx, num, &var_name).is_some()
        && exp_sum_dominant_sign(ctx, den, &var_name).is_some()
    {
        return Some(ctx.num(0));
    }
    None
}

/// The sign of the dominant term of a sum of rational-base exponentials at
/// `+inf`, but only when a genuinely growing base (`> 1`) dominates with a
/// nonzero coefficient. None when the expression is not such a sum.
fn exp_sum_dominant_sign(ctx: &Context, expr: ExprId, var_name: &str) -> Option<i32> {
    use num_traits::{One, Signed, Zero};
    let mut terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, expr, var_name, &rational_one(), &mut terms)?;
    let one = BigRational::one();
    let max_base = terms.iter().map(|(_, b)| b.clone()).max()?;
    if max_base <= one {
        return None;
    }
    let dominant: BigRational = terms
        .iter()
        .filter(|(_, b)| *b == max_base)
        .map(|(c, _)| c.clone())
        .sum();
    if dominant.is_zero() {
        return None;
    }
    Some(if dominant.is_positive() { 1 } else { -1 })
}

/// The sign of a positive-degree polynomial as `x -> +inf` (the sign of its
/// leading coefficient, since `x^deg -> +inf`). None for a non-polynomial or a
/// constant.
fn positive_degree_polynomial_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<i32> {
    use num_traits::Signed;
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() < 1 {
        return None;
    }
    let leading = poly.coeffs.get(poly.degree())?;
    Some(if leading.is_positive() { 1 } else { -1 })
}

/// A quotient of sums of rational-base exponentials at infinity, decided by
/// the dominant base: `(sum c_i b_i^x) / (sum d_j e_j^x)` tends to the ratio
/// of the dominant coefficients when the top bases match, to `+-inf` when the
/// numerator's dominant base is larger, and to 0 when it is smaller. Reuses the
/// effective-base term parser. Resolves `3^x/2^x -> +inf`,
/// `(2^x+3^x)/3^x -> 1`, `(3^x-2^x)/(3^x+2^x) -> 1`.
fn exp_sum_quotient_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{Signed, Zero};
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let mut num_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut den_terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, num, &var_name, &rational_one(), &mut num_terms)?;
    collect_rational_exp_terms(ctx, den, &var_name, &rational_one(), &mut den_terms)?;
    // Require at least one genuinely growing exponential on each side so this
    // is an exponential quotient, not a rational one (owned upstream).
    let one = BigRational::one();
    if !num_terms.iter().any(|(_, b)| *b > one) || !den_terms.iter().any(|(_, b)| *b > one) {
        return None;
    }
    let num_max = num_terms.iter().map(|(_, b)| b.clone()).max()?;
    let den_max = den_terms.iter().map(|(_, b)| b.clone()).max()?;
    let dominant_sum = |terms: &[(BigRational, BigRational)], base: &BigRational| -> BigRational {
        terms
            .iter()
            .filter(|(_, b)| b == base)
            .map(|(c, _)| c.clone())
            .sum()
    };
    let num_dom = dominant_sum(&num_terms, &num_max);
    let den_dom = dominant_sum(&den_terms, &den_max);
    // A cancelled dominant coefficient leaves a lower true dominant - too subtle
    // here, so decline.
    if den_dom.is_zero() {
        return None;
    }
    use std::cmp::Ordering;
    match num_max.cmp(&den_max) {
        Ordering::Less => Some(ctx.num(0)),
        Ordering::Equal => {
            let ratio = &num_dom / &den_dom;
            Some(ctx.add(Expr::Number(ratio)))
        }
        Ordering::Greater => {
            if num_dom.is_zero() {
                return None;
            }
            let positive = num_dom.is_positive() == den_dom.is_positive();
            Some(mk_infinity(
                ctx,
                if positive { InfSign::Pos } else { InfSign::Neg },
            ))
        }
    }
}

/// The exponential indeterminate form `1^inf` at infinity: when the base
/// tends to 1 and the exponent diverges, `base^exp = exp(exp * ln(base))` and,
/// since `ln(1 + h) ~ h` for `h = base - 1 -> 0`, the limit is
/// `exp(lim exp * (base - 1))`. Resolves `(1 + a/x)^x -> e^a`,
/// `(1 + 1/x)^(2x) -> e^2`, `(1 + 1/x^2)^x -> 1`, `((2x+1)/(2x-1))^x -> e`.
/// Gated on a unit base limit and a divergent exponent, so `2^x`, `x^x`, and
/// any base that does not tend to 1 are left untouched.
fn one_to_infinity_power_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::One;
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    // Both the base and the exponent must carry the variable for an
    // indeterminate 1^inf; a constant base or exponent is a different form.
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    // The base must tend to exactly 1.
    let base_limit = try_limit_rules_at_infinity(ctx, base, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, base_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    // The exponent must diverge (otherwise 1^finite = 1 is the continuous case).
    let exp_limit = try_limit_rules_at_infinity(ctx, exp, var, approach)?;
    limit_value_infinite_sign(ctx, exp_limit)?;
    // L = lim exp * (base - 1), rationalized so the quotient limit can read it.
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(base, one));
    let product = ctx.add(Expr::Mul(exp, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let l_limit = try_limit_rules_at_infinity(ctx, combined, var, approach)?;
    // exp(L): e^finite, e^(+inf) = inf, e^(-inf) = 0.
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            // Fold the degenerate finite exponents: e^0 = 1, e^1 = e.
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                use num_traits::{One, Zero};
                if value.is_zero() {
                    return Some(ctx.num(1));
                }
                if value.is_one() {
                    return Some(ctx.add(Expr::Constant(Constant::E)));
                }
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// The `inf * 0` form `g(x) * ln(f(x))` at infinity when `f -> 1` (so
/// `ln f -> 0`) and `g` diverges. Since `ln(1 + h) ~ h` for `h = f - 1 -> 0`,
/// the limit is `lim g * (f - 1)` -- the same `L` the 1^inf power rule computes,
/// exposed for the standalone product. Resolves `x ln(1 + a/x) -> a`,
/// `x ln((x+1)/x) -> 1`, `x^2 ln(1 + 1/x^2) -> 1`, `x ln(1 - 1/x) -> -1`. The
/// `f -> 1` gate keeps it off `x ln(x)` (`f -> inf`) and `ln(2) x` (`f` constant),
/// and the divergent-cofactor gate leaves the continuous `finite * 0` case alone.
fn product_log_unit_argument_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::One;
    let Expr::Mul(a, b) = ctx.get(expr).clone() else {
        return None;
    };
    let natural_log_arg = |ctx: &Context, e: ExprId| -> Option<ExprId> {
        if let Expr::Function(fn_id, args) = ctx.get(e) {
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
                return Some(args[0]);
            }
        }
        None
    };
    let (cofactor, ln_arg) = if let Some(arg) = natural_log_arg(ctx, a) {
        (b, arg)
    } else if let Some(arg) = natural_log_arg(ctx, b) {
        (a, arg)
    } else {
        return None;
    };
    if !depends_on(ctx, cofactor, var) || !depends_on(ctx, ln_arg, var) {
        return None;
    }
    // The log argument must tend to exactly 1 (so ln -> 0): a genuine inf*0.
    let arg_limit = try_limit_rules_at_infinity(ctx, ln_arg, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, arg_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    // The cofactor must diverge (otherwise finite * 0 = 0 is the continuous case).
    let cofactor_limit = try_limit_rules_at_infinity(ctx, cofactor, var, approach)?;
    limit_value_infinite_sign(ctx, cofactor_limit)?;
    // L = lim g * (f - 1), rationalized so the quotient limit can read it.
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(ln_arg, one));
    let product = ctx.add(Expr::Mul(cofactor, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    try_limit_rules_at_infinity(ctx, combined, var, approach)
}

/// The `0^0` form `x^g -> exp(lim g * ln x)` as `x -> 0+`. The base is the bare
/// variable, which is positive on the RIGHT of 0, so `x^g = exp(g ln x)` is
/// real and the limit is `exp(lim g ln x)`. Resolves the canonical `x^x -> 1`
/// (lim x ln x = 0). Fires only on the right side at 0 with a bare-variable
/// base; a two-sided `x^x` is complex on the left and stays residual.
fn apply_finite_zero_base_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    // The base is the bare variable: x -> 0+ through positive values, so ln(x)
    // is real. A non-variable base could approach 0 with an unknown sign.
    if base != var || !depends_on(ctx, exp, var) {
        return None;
    }
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![var]);
    let product = ctx.add(Expr::Mul(exp, ln_base));
    let l_limit = try_limit_rules_at_finite_one_sided(ctx, product, var, point, side)?;
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                if value.is_zero() {
                    return Some(ctx.num(1));
                }
                if value.is_one() {
                    return Some(ctx.add(Expr::Constant(Constant::E)));
                }
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// The `1^inf` form at a FINITE point: `(1 + x)^(1/x) -> e` and friends. Same
/// identity as the infinity case - `exp(lim exp*(base-1))` via `ln(1+h) ~ h` -
/// but the exponent (e.g. `1/x` at 0) has no signed bilateral limit, so the
/// gate is on the PRODUCT instead: a unit base limit and a NONZERO product
/// limit `L = lim exp*(base-1)` (a nonzero L forces the exponent to diverge,
/// so the form is genuinely indeterminate; a continuous `1^finite` gives
/// `L = 0` and is left to ordinary substitution).
fn apply_finite_one_to_infinity_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    let base_limit = try_limit_rules_at_finite(ctx, base, var, point)?;
    if !crate::numeric_eval::as_rational_const(ctx, base_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(base, one));
    let product = ctx.add(Expr::Mul(exp, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let l_limit = try_limit_rules_at_finite(ctx, combined, var, point)?;
    // A zero product limit is the continuous 1^finite case (1^c = 1), owned by
    // ordinary substitution; only fire on the genuinely indeterminate form.
    if crate::numeric_eval::as_rational_const(ctx, l_limit).is_some_and(|v| v.is_zero()) {
        return None;
    }
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if crate::numeric_eval::as_rational_const(ctx, l_limit).is_some_and(|v| v.is_one()) {
                return Some(ctx.add(Expr::Constant(Constant::E)));
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// A general-base exponential `b^(s x)` at infinity (`b` a positive rational):
/// `b^x -> +inf` for `b > 1`, `-> 0` for `0 < b < 1`, `-> 1` for `b = 1`, with
/// the sign of the exponent slope and the approach direction combined. The
/// engine already grows `e^x`; this covers the rational bases (`2^x -> inf`),
/// which also lets a sum like `2^x + 3^x` diverge and feed the inf^0 rule.
fn general_base_exponential_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{One, Signed, Zero};
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let (base, exponent) = numeric_base_power(ctx, expr)?;
    if !base.is_positive() {
        return None;
    }
    if base.is_one() {
        return Some(ctx.num(1));
    }
    // The exponent must be linear in x; its sign at the approach decides growth.
    let exp_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
    if exp_poly.degree() != 1 {
        return None;
    }
    let slope = exp_poly.coeffs.get(1).cloned()?;
    if slope.is_zero() {
        return None;
    }
    // The exponent tends to +inf when its slope and the approach agree in sign.
    let exponent_to_pos_inf = slope.is_positive() == matches!(approach, InfSign::Pos);
    let base_gt_one = base > BigRational::one();
    if base_gt_one == exponent_to_pos_inf {
        Some(mk_infinity(ctx, InfSign::Pos))
    } else {
        Some(ctx.num(0))
    }
}

/// The `inf^0` form at infinity: when the base diverges to `+inf` and the
/// exponent tends to 0, `base^exp = exp(exp * ln base)` and the limit is
/// `exp(lim exp * ln base)`. With the log-of-exponential-sum dominance rule
/// feeding the inner limit, this resolves `(2^x + 3^x)^(1/x) = e^(ln 3) = 3`.
/// A positive-infinite base keeps ln real; `x^(1/x) = 1` falls out too.
fn inf_to_zero_power_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::Zero;
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    // base -> +inf keeps it positive (ln real); exp -> 0 makes it inf^0.
    let base_limit = try_limit_rules_at_infinity(ctx, base, var, approach)?;
    if limit_value_infinite_sign(ctx, base_limit) != Some(1) {
        return None;
    }
    let exp_limit = try_limit_rules_at_infinity(ctx, exp, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, exp_limit).is_some_and(|v| v.is_zero()) {
        return None;
    }
    // L = lim exp * ln(base), rationalized into a single fraction and
    // presimplified (drops the unit factor the rationalizer leaves) so the
    // log-of-exponential-sum dominance rule sees a BARE ln in the numerator.
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let product = ctx.add(Expr::Mul(exp, ln_base));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let combined = presimplify_safe_for_limit(ctx, combined);
    let l_limit = try_limit_rules_at_infinity(ctx, combined, var, approach)?;
    exp_of_limit_value(ctx, l_limit)
}

/// `exp(L)` for a resolved inner limit L, folding the cases an exponential
/// indeterminate form produces: e^(+inf)=inf, e^(-inf)=0, e^0=1, e^1=e,
/// e^(ln c)=c, otherwise e^L.
fn exp_of_limit_value(ctx: &mut Context, l_limit: ExprId) -> Option<ExprId> {
    use num_traits::{One, Zero};
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                if value.is_zero() {
                    return Some(ctx.num(1));
                }
                if value.is_one() {
                    return Some(ctx.add(Expr::Constant(Constant::E)));
                }
            }
            // e^(ln c) = c.
            if let Some(arg) = bare_natural_log_argument(ctx, l_limit) {
                return Some(arg);
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// Combine an expression into a single fraction `(numerator, denominator)` of
/// polynomial-buildable parts, putting Add/Sub over a common denominator and
/// flattening Mul/Div. Lets a downstream polynomial reader expand and cancel
/// (e.g. `x * ((1 + 1/x) - 1)` rationalizes to `(x * (x + 1 - x)) / x`).
/// Non-rational atoms (functions, irrational powers) stay opaque over 1, so the
/// reader's own preconditions reject them.
fn rationalize_to_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    let one = |ctx: &mut Context| ctx.num(1);
    match ctx.get(expr).clone() {
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            let subtract = matches!(ctx.get(expr), Expr::Sub(_, _));
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            let left = ctx.add(Expr::Mul(na, db));
            let right = ctx.add(Expr::Mul(nb, da));
            let num = if subtract {
                ctx.add(Expr::Sub(left, right))
            } else {
                ctx.add(Expr::Add(left, right))
            };
            let den = ctx.add(Expr::Mul(da, db));
            (num, den)
        }
        Expr::Mul(a, b) => {
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            (ctx.add(Expr::Mul(na, nb)), ctx.add(Expr::Mul(da, db)))
        }
        Expr::Div(a, b) => {
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            // (na/da) / (nb/db) = (na db) / (da nb).
            (ctx.add(Expr::Mul(na, db)), ctx.add(Expr::Mul(da, nb)))
        }
        Expr::Neg(inner) => {
            let (n, d) = rationalize_to_fraction(ctx, inner);
            (ctx.add(Expr::Neg(n)), d)
        }
        _ => {
            let d = one(ctx);
            (expr, d)
        }
    }
}

/// An additive combination of rational functions at infinity that the
/// per-term additive rule leaves as `inf - inf` (e.g. `(x^2+1)/(x+1) - x`):
/// place the terms over a common denominator and reuse the rational quotient
/// limit. Runs only after the structural additive rules decline, so a sum of
/// terms that each have a finite limit keeps its existing trace. Non-rational
/// operands (sqrt, sin, ...) make the multipoly conversion fail, so the
/// conjugate/elementary paths keep their forms.
fn rational_difference_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let (lhs, rhs, subtract) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs, true),
        Expr::Add(lhs, rhs) => (lhs, rhs, false),
        _ => return None,
    };
    let (a_num, a_den) = rational_numerator_denominator(ctx, lhs);
    let (b_num, b_den) = rational_numerator_denominator(ctx, rhs);
    // (a_num/a_den) +/- (b_num/b_den) = (a_num b_den +/- b_num a_den)/(a_den b_den).
    let left_cross = ctx.add(Expr::Mul(a_num, b_den));
    let right_cross = ctx.add(Expr::Mul(b_num, a_den));
    let numerator = if subtract {
        ctx.add(Expr::Sub(left_cross, right_cross))
    } else {
        ctx.add(Expr::Add(left_cross, right_cross))
    };
    let denominator = ctx.add(Expr::Mul(a_den, b_den));
    let combined = ctx.add(Expr::Div(numerator, denominator));
    rational_poly_limit(ctx, combined, var, approach)
}

/// `(numerator, denominator)` of an expression viewed as a fraction: a Div
/// splits, anything else is over 1.
fn rational_numerator_denominator(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        let one = ctx.num(1);
        (expr, one)
    }
}

/// A rational quotient at infinity where one or both sides are a polynomial
/// plus BOUNDED additive noise (e.g. `(x + sin x)/x -> 1`): the polynomial
/// part sets the growth, the bounded noise is dominated. Reuses
/// polynomial_growth_info_with_bounded_additive_noise and compares degrees
/// exactly as the pure rational rule does. Gated to fire only when at least
/// one side is NOT a pure polynomial, so rational_poly_limit keeps ownership
/// of the exact poly/poly case.
fn bounded_noise_rational_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    if polynomial_growth_info(ctx, num, var).is_some()
        && polynomial_growth_info(ctx, den, var).is_some()
    {
        return None;
    }
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;
    if den_growth.leading_coeff.is_zero() {
        return None;
    }
    if num_growth.degree < den_growth.degree {
        return Some(ctx.num(0));
    }
    let ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    if num_growth.degree == den_growth.degree {
        return Some(ctx.add(Expr::Number(ratio)));
    }
    let sign = limit_growth_sign(&ratio, num_growth.degree - den_growth.degree, approach);
    Some(mk_infinity(ctx, sign))
}

const FINITE_POINT_LIMIT_UNSUPPORTED_WARNING: &str =
    "Finite point limits are not supported safely yet";
const FINITE_EMPTY_PUNCTURED_REAL_NEIGHBORHOOD_WARNING_DETAIL: &str =
    "real-domain condition holds only at the approach point; no punctured real neighbourhood is available";

struct FiniteResidualPoint {
    var_name: String,
    point_value: BigRational,
}

fn finite_residual_point(ctx: &Context, var: ExprId, point: ExprId) -> Option<FiniteResidualPoint> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    Some(FiniteResidualPoint {
        var_name: ctx.sym_name(*var_symbol).to_string(),
        point_value: point_value.clone(),
    })
}

fn finite_single_function_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    Some((ctx.builtin_of(fn_id)?, args[0]))
}

fn finite_polynomial_tail_negative_on_both_sides(
    polynomial: &Polynomial,
    point_value: &BigRational,
) -> Option<bool> {
    let (order, derivative) =
        finite_polynomial_local_order_and_derivative(polynomial, point_value)?;
    finite_local_tail_negative_on_both_sides(&derivative, order)
}

fn finite_residual_has_empty_punctured_sqrt_domain(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let Some(finite_point) = finite_residual_point(ctx, var, point) else {
        return false;
    };
    let Some(radicand) = extract_square_root_base(ctx, expr) else {
        return false;
    };

    finite_endpoint_argument_zero_tail_negative_on_both_sides(
        ctx,
        radicand,
        &finite_point.var_name,
        &finite_point.point_value,
    )
    .unwrap_or(false)
}

fn finite_residual_has_empty_punctured_acosh_domain(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let Some(finite_point) = finite_residual_point(ctx, var, point) else {
        return false;
    };
    let Some((builtin, argument_expr)) = finite_single_function_arg(ctx, expr) else {
        return false;
    };
    if builtin != BuiltinFn::Acosh {
        return false;
    }

    let Ok(argument) = Polynomial::from_expr(ctx, argument_expr, &finite_point.var_name) else {
        return false;
    };
    if argument.eval(&finite_point.point_value) != rational_one() {
        return false;
    }
    let endpoint_gap = argument.sub(&Polynomial::one(finite_point.var_name));

    finite_polynomial_tail_negative_on_both_sides(&endpoint_gap, &finite_point.point_value)
        .unwrap_or(false)
}

fn finite_residual_has_empty_punctured_inverse_trig_domain(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let Some(finite_point) = finite_residual_point(ctx, var, point) else {
        return false;
    };
    let Some((builtin, argument_expr)) = finite_single_function_arg(ctx, expr) else {
        return false;
    };
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return false;
    }

    let Ok(argument) = Polynomial::from_expr(ctx, argument_expr, &finite_point.var_name) else {
        return false;
    };
    let Some((endpoint_gap, _endpoint)) = finite_inverse_trig_endpoint_gap(
        &argument,
        &finite_point.var_name,
        &finite_point.point_value,
    ) else {
        return false;
    };

    finite_polynomial_tail_negative_on_both_sides(&endpoint_gap, &finite_point.point_value)
        .unwrap_or(false)
}

fn finite_residual_warning(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> String {
    if finite_residual_has_empty_punctured_sqrt_domain(ctx, expr, var, point)
        || finite_residual_has_empty_punctured_acosh_domain(ctx, expr, var, point)
        || finite_residual_has_empty_punctured_inverse_trig_domain(ctx, expr, var, point)
    {
        format!(
            "{FINITE_POINT_LIMIT_UNSUPPORTED_WARNING}; {FINITE_EMPTY_PUNCTURED_REAL_NEIGHBORHOOD_WARNING_DETAIL}"
        )
    } else {
        FINITE_POINT_LIMIT_UNSUPPORTED_WARNING.to_string()
    }
}

/// Evaluate a limit at infinity using conservative rules.
///
/// This runs optional safe pre-simplification, applies direct rules in order,
/// and otherwise returns a residual `limit(...)` expression.
pub fn eval_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    opts: &LimitOptions,
) -> LimitEvalOutcome {
    let simplified_expr = match opts.presimplify {
        PreSimplifyMode::Off => expr,
        PreSimplifyMode::Safe => presimplify_safe_for_limit(ctx, expr),
    };

    if let Approach::Finite(point) = approach {
        if let Some(result_expr) = try_limit_rules_at_finite(ctx, simplified_expr, var, point) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }
    if let Approach::FiniteOneSided(point, side) = approach {
        if let Some(result_expr) =
            try_limit_rules_at_finite_one_sided(ctx, simplified_expr, var, point, side)
        {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    if let Some(sign) = approach.inf_sign() {
        if let Some(result_expr) = try_limit_rules_at_infinity(ctx, simplified_expr, var, sign) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
        // Reciprocal-substitution fallback: `lim_{x→±∞} g(x) = lim_{u→0±} g(1/u)` (exact — `u = 1/x`
        // approaches 0 from the matching side). Only reached when the direct ∞ rules declined, so it
        // can RESOLVE a previously-residual limit but never overrides a direct result.
        if let Some(result_expr) =
            try_limit_at_infinity_by_reciprocal_substitution(ctx, simplified_expr, var, sign)
        {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    let residual = mk_limit_for_approach(ctx, simplified_expr, var, approach);
    let warning = match approach {
        Approach::Finite(point) => finite_residual_warning(ctx, simplified_expr, var, point),
        Approach::FiniteOneSided(_, _) => {
            "One-sided finite point limits are not supported safely for this expression yet"
                .to_string()
        }
        Approach::PosInfinity | Approach::NegInfinity => {
            "Could not determine limit safely".to_string()
        }
    };
    LimitEvalOutcome {
        expr: residual,
        warning: Some(warning),
    }
}

/// Evaluate `lim_{x→±∞} expr` by the exact change of variable `u = 1/x`: the limit equals
/// `lim_{u→0±} expr[x ↦ 1/u]` (`u → 0⁺` for `x → +∞`, `u → 0⁻` for `x → −∞`). Reusing the SAME
/// variable for `u` avoids introducing a fresh symbol. Closes notable forms the direct ∞ rules miss,
/// e.g. `x·sin(1/x) → 1` and `(1 + a/x)^x → e^a`. Sound because the substitution is an exact identity
/// and the finite one-sided evaluator it delegates to is itself sound.
fn try_limit_at_infinity_by_reciprocal_substitution(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    sign: InfSign,
) -> Option<ExprId> {
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, var));
    let substituted = cas_ast::substitute_expr_by_id(ctx, expr, var, reciprocal);
    // The substitution leaves nested reciprocals and unit factors (`1/(1/x)`, `(1·sin(x))/x`);
    // normalize so the finite evaluator sees the reduced form (`x·sin(1/x)` → `sin(x)/x`). The
    // finite one-sided evaluator re-checks the domain, so canonical normalization is safe here.
    // The substitution leaves nested reciprocals and unit factors (`1/(1/x)`, `sin(x)·(1/x)`) that
    // the cas_math normalizers do not reduce; clean them so the finite evaluator sees `sin(x)/x`.
    let substituted = reduce_reciprocal_substitution_artifacts(ctx, substituted);
    let zero = ctx.num(0);
    let side = match sign {
        InfSign::Pos => FiniteLimitSide::Right,
        InfSign::Neg => FiniteLimitSide::Left,
    };
    try_limit_rules_at_finite_one_sided(ctx, substituted, var, zero, side)
}

/// Multiply two expressions, dropping a unit factor (`1·e → e`).
fn mul_drop_unit(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_one(ctx, a) {
        return b;
    }
    if expr_is_one(ctx, b) {
        return a;
    }
    ctx.add(Expr::Mul(a, b))
}

/// Bottom-up cleanup of the artifacts that `x ↦ 1/x` substitution introduces: nested reciprocals
/// (`a/(b/c) → (a·c)/b`, so `1/(1/x) → x`), products by a reciprocal (`a·(1/d) → a/d`), and unit
/// `Mul` factors (`1·e → e`). Purely structural and value-preserving, so it does not affect the
/// limit; it only puts the substituted expression in the shape the finite evaluator recognises.
fn reduce_reciprocal_substitution_artifacts(ctx: &mut Context, expr: ExprId) -> ExprId {
    let rebuilt = match *ctx.get(expr) {
        Expr::Add(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Mul(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Mul(a, b))
        }
        Expr::Div(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(a) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            ctx.add(Expr::Neg(a))
        }
        Expr::Function(fn_id, ref args) => {
            let args: Vec<ExprId> = args.clone();
            let reduced: Vec<ExprId> = args
                .into_iter()
                .map(|arg| reduce_reciprocal_substitution_artifacts(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, reduced))
        }
        _ => expr,
    };
    // Local rewrites on the rebuilt (children-reduced) node.
    match *ctx.get(rebuilt) {
        // a / (b/c) = (a·c)/b  →  in particular 1/(1/x) = x.
        Expr::Div(num, den) => {
            if let Expr::Div(inner_num, inner_den) = *ctx.get(den) {
                let new_num = mul_drop_unit(ctx, num, inner_den);
                if expr_is_one(ctx, inner_num) {
                    return new_num; // (a·c)/1
                }
                return ctx.add(Expr::Div(new_num, inner_num));
            }
            rebuilt
        }
        // a·(1/d) = a/d, and drop unit factors.
        Expr::Mul(a, b) => {
            if expr_is_one(ctx, a) {
                return b;
            }
            if expr_is_one(ctx, b) {
                return a;
            }
            if let Expr::Div(bn, bd) = *ctx.get(b) {
                if expr_is_one(ctx, bn) {
                    return ctx.add(Expr::Div(a, bd));
                }
            }
            if let Expr::Div(an, ad) = *ctx.get(a) {
                if expr_is_one(ctx, an) {
                    return ctx.add(Expr::Div(b, ad));
                }
            }
            rebuilt
        }
        _ => rebuilt,
    }
}

const PRESIMPLIFY_MAX_DEPTH: usize = 500;

fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Zero;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn expr_is_one(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::One;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn apply_safe_add_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, b) {
        return a;
    }
    if expr_is_zero(ctx, a) {
        return b;
    }

    if let Expr::Neg(neg_inner) = ctx.get(b) {
        if *neg_inner == a {
            return ctx.num(0);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(a) {
        if *neg_inner == b {
            return ctx.num(0);
        }
    }

    ctx.add(Expr::Add(a, b))
}

fn apply_safe_sub_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, b) {
        return a;
    }
    if a == b {
        return ctx.num(0);
    }
    ctx.add(Expr::Sub(a, b))
}

fn apply_safe_mul_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, a) || expr_is_zero(ctx, b) {
        return ctx.num(0);
    }
    if expr_is_one(ctx, b) {
        return a;
    }
    if expr_is_one(ctx, a) {
        return b;
    }
    ctx.add(Expr::Mul(a, b))
}

fn presimplify_recursive(ctx: &mut Context, expr: ExprId, depth: usize) -> ExprId {
    if depth > PRESIMPLIFY_MAX_DEPTH {
        return expr;
    }

    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_add_rules(ctx, a2, b2)
        }
        Expr::Sub(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_sub_rules(ctx, a2, b2)
        }
        Expr::Mul(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_mul_rules(ctx, a2, b2)
        }
        Expr::Neg(a) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            if let Expr::Neg(inner) = ctx.get(a2) {
                return *inner;
            }
            ctx.add(Expr::Neg(a2))
        }
        Expr::Div(num, den) => {
            let num2 = presimplify_recursive(ctx, num, depth + 1);
            let den2 = presimplify_recursive(ctx, den, depth + 1);
            ctx.add(Expr::Div(num2, den2))
        }
        Expr::Pow(base, exp) => {
            let base2 = presimplify_recursive(ctx, base, depth + 1);
            let exp2 = presimplify_recursive(ctx, exp, depth + 1);
            ctx.add(Expr::Pow(base2, exp2))
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                new_args.push(presimplify_recursive(ctx, arg, depth + 1));
            }
            ctx.add(Expr::Function(name, new_args))
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => expr,
        Expr::Hold(inner) => {
            let inner2 = presimplify_recursive(ctx, inner, depth + 1);
            ctx.add(Expr::Hold(inner2))
        }
        Expr::Matrix { .. } | Expr::SessionRef(_) => expr,
    }
}

/// Safe pre-simplification for limit evaluation.
///
/// This is an allowlist-only pass and intentionally excludes transforms that
/// require domain assumptions (for example, `a/a -> 1` or `a^0 -> 1`).
pub fn presimplify_safe_for_limit(ctx: &mut Context, expr: ExprId) -> ExprId {
    presimplify_recursive(ctx, expr, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    #[test]
    fn combine_difference_over_common_denominator_builds_single_fraction() {
        use std::collections::HashMap;
        // 1/x - 1/y -> (y - x)/(x*y); check the combined fraction's value at x=2, y=3 is 1/6.
        let mut ctx = Context::new();
        let lhs = parse_expr(&mut ctx, "1/x");
        let rhs = parse_expr(&mut ctx, "1/y");
        let combined = combine_difference_over_common_denominator(&mut ctx, lhs, rhs)
            .expect("two fractions combine");
        assert!(
            matches!(ctx.get(combined), Expr::Div(_, _)),
            "combined is a single fraction"
        );
        let mut map = HashMap::new();
        map.insert("x".to_string(), 2.0);
        map.insert("y".to_string(), 3.0);
        let v = crate::evaluator_f64::eval_f64(&ctx, combined, &map).expect("foldable");
        assert!((v - 1.0 / 6.0).abs() < 1e-12, "1/2 - 1/3 = 1/6, got {v}");

        // Neither side is a fraction -> declines (no new structure to gain).
        let a = parse_expr(&mut ctx, "x^2");
        let b = parse_expr(&mut ctx, "x");
        assert!(combine_difference_over_common_denominator(&mut ctx, a, b).is_none());
    }

    #[test]
    fn finite_sub_result_declines_same_sign_infinity_difference() {
        // (+inf) - (+inf) is INDETERMINATE: must NOT collapse to 0 (the `lhs == rhs` shortcut on
        // equal interned infinities was the `lim 1/sin^2 x - 1/x^2 = 0` wrong-answer).
        let mut ctx = Context::new();
        let pos_inf = ctx.add(Expr::Constant(Constant::Infinity));
        assert_eq!(
            finite_sub_result(&mut ctx, pos_inf, pos_inf),
            None,
            "(+inf) - (+inf) must decline, not return 0"
        );
        // -inf - -inf also indeterminate.
        let inf_a = ctx.add(Expr::Constant(Constant::Infinity));
        let neg_inf = ctx.add(Expr::Neg(inf_a));
        let inf_b = ctx.add(Expr::Constant(Constant::Infinity));
        let neg_inf_b = ctx.add(Expr::Neg(inf_b));
        assert_eq!(
            finite_sub_result(&mut ctx, neg_inf, neg_inf_b),
            None,
            "(-inf) - (-inf) must decline"
        );
        // DETERMINATE cases must still resolve (not decline):
        //   +inf - (-inf) = +inf, +inf - finite = +inf, finite - finite = value.
        assert!(
            finite_sub_result(&mut ctx, pos_inf, neg_inf).is_some(),
            "(+inf) - (-inf) is determinate (+inf), must resolve"
        );
        let five = ctx.num(5);
        assert!(
            finite_sub_result(&mut ctx, pos_inf, five).is_some(),
            "(+inf) - 5 is determinate, must resolve"
        );
        let seven = ctx.num(7);
        let three = ctx.num(3);
        let diff = finite_sub_result(&mut ctx, seven, three).expect("7 - 3 resolves");
        assert!(
            matches!(ctx.get(diff), Expr::Number(n) if *n == num_rational::BigRational::from_integer(4.into()))
        );
    }

    fn display_expr(ctx: &Context, expr: ExprId) -> String {
        DisplayExpr {
            context: ctx,
            id: expr,
        }
        .to_string()
    }

    fn assert_rational_taylor(src: &str, order: usize, expected: &[(i64, i64)]) {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, src);
        let poly = taylor_at_zero_with_rational(&ctx, expr, "x", order)
            .unwrap_or_else(|| panic!("{src} should expand"));
        for (k, (num, den)) in expected.iter().enumerate() {
            let coeff = poly
                .coeffs
                .get(k)
                .cloned()
                .unwrap_or_else(|| BigRational::new(0.into(), 1.into()));
            assert_eq!(
                coeff,
                BigRational::new((*num).into(), (*den).into()),
                "{src}: coefficient of x^{k}"
            );
        }
    }

    #[test]
    fn rational_taylor_matches_known_geometric_series() {
        assert_rational_taylor("1/(1-x)", 4, &[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]);
        assert_rational_taylor("1/(1+x)", 4, &[(1, 1), (-1, 1), (1, 1), (-1, 1), (1, 1)]);
        assert_rational_taylor("1/(1+x^2)", 4, &[(1, 1), (0, 1), (-1, 1), (0, 1), (1, 1)]);
        assert_rational_taylor("1/(2-x)", 3, &[(1, 2), (1, 4), (1, 8), (1, 16)]);
        assert_rational_taylor("1/(1-x)^2", 3, &[(1, 1), (2, 1), (3, 1), (4, 1)]);
        assert_rational_taylor("x/(1-x)", 4, &[(0, 1), (1, 1), (1, 1), (1, 1), (1, 1)]);
    }

    #[test]
    fn rational_taylor_declines_pole_at_zero() {
        // 1/x and 1/(x - 1 + 1) style poles at 0 have no Maclaurin expansion.
        for src in ["1/x", "1/x^2", "(1+x)/x"] {
            let mut ctx = Context::new();
            let expr = parse_expr(&mut ctx, src);
            assert!(
                taylor_at_zero_with_rational(&ctx, expr, "x", 4).is_none(),
                "{src} has a pole at 0 and must not expand"
            );
        }
    }

    fn assert_number_expr(ctx: &Context, expr: ExprId, numerator: i64, denominator: i64) {
        let Expr::Number(value) = ctx.get(expr) else {
            panic!("expected exact rational expression");
        };
        assert_eq!(
            value,
            &BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
        );
    }

    fn assert_ratio_over_ln_base(
        ctx: &Context,
        expr: ExprId,
        numerator: i64,
        denominator: i64,
        base_numerator: i64,
        base_denominator: i64,
    ) {
        let Expr::Div(num_expr, den_expr) = ctx.get(expr).clone() else {
            panic!(
                "expected quotient over ln(base), got {}",
                display_expr(ctx, expr)
            );
        };
        assert_number_expr(ctx, num_expr, numerator, denominator);

        let Expr::Function(fn_id, args) = ctx.get(den_expr).clone() else {
            panic!(
                "expected ln(base) denominator, got {}",
                display_expr(ctx, den_expr)
            );
        };
        assert!(ctx.is_builtin(fn_id, BuiltinFn::Ln));
        assert_eq!(args.len(), 1);
        assert_number_expr(ctx, args[0], base_numerator, base_denominator);
    }

    #[test]
    fn depends_on_detects_simple_variable() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x + 1");
        let x = parse_expr(&mut ctx, "x");
        assert!(depends_on(&ctx, expr, x));
    }

    #[test]
    fn depends_on_rejects_constant_expression() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 + pi");
        let x = parse_expr(&mut ctx, "x");
        assert!(!depends_on(&ctx, expr, x));
    }

    #[test]
    fn parse_pow_int_extracts_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");
        let (_, n) = parse_pow_int(&ctx, expr).expect("power");
        assert_eq!(n, 3);
    }

    #[test]
    fn limit_sign_handles_neg_infinity_parity() {
        assert_eq!(limit_sign(InfSign::Pos, 7), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 2), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 3), InfSign::Neg);
    }

    #[test]
    fn mk_limit_builds_limit_call_with_signed_infinity_symbol() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2");
        let var = parse_expr(&mut ctx, "x");
        let lim = mk_limit(&mut ctx, expr, var, InfSign::Neg);

        let Expr::Function(_fn_id, args) = ctx.get(lim) else {
            panic!("expected limit function call");
        };
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], expr);
        assert_eq!(args[1], var);

        let approach = args[2];
        match ctx.get(approach) {
            Expr::Neg(inner) => {
                assert!(matches!(
                    ctx.get(*inner),
                    Expr::Constant(Constant::Infinity)
                ));
            }
            _ => panic!("expected negative infinity argument"),
        }
    }

    #[test]
    fn one_sided_composition_saturates_inner_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, side, expected) in [
            ("e^(1/x)", FiniteLimitSide::Right, "infinity"),
            ("e^(1/x)", FiniteLimitSide::Left, "0"),
            ("atan(1/x)", FiniteLimitSide::Right, "pi / 2"),
            ("atan(1/x)", FiniteLimitSide::Left, "-pi / 2"),
            ("tanh(1/x)", FiniteLimitSide::Right, "1"),
            ("e^(-1/x)", FiniteLimitSide::Right, "0"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, side)
                .unwrap_or_else(|| panic!("must resolve: {source} {side:?}"));
            // The one-sided rule returns f(+-inf); the eval layer folds it,
            // but fold_infinity_saturation already runs inside the helper.
            assert_eq!(display_expr(&ctx, out), expected, "{source} {side:?}");
        }
    }

    #[test]
    fn one_sided_product_of_zero_and_finite_collapses_to_zero() {
        // e^(1/x) -> 0 and atan(1/x) -> -pi/2 from the left, so their product
        // is 0 * (-pi/2) = 0. combine_limit_product must fold the zero factor
        // instead of emitting the un-normalized product `-0 * pi/2`.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let expr = parse_expr(&mut ctx, "e^(1/x) * atan(1/x)");
        let out =
            try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, FiniteLimitSide::Left)
                .expect("0 * finite must resolve");
        assert_eq!(display_expr(&ctx, out), "0");
    }

    #[test]
    fn one_sided_composition_declines_oscillating_outers() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        // sin/cos at infinity do not converge; the saturation fold leaves
        // them symbolic, so the rule must decline (stay residual).
        for source in ["sin(1/x)", "cos(1/x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                try_limit_rules_at_finite_one_sided(
                    &mut ctx,
                    expr,
                    x,
                    zero,
                    FiniteLimitSide::Right
                )
                .is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn bilateral_even_cosh_over_pole_saturates_to_infinity() {
        // cosh is even, so cosh(1/x) -> +inf from both sides even though the
        // inner pole 1/x diverges with opposite signs; shifted and scaled
        // poles behave the same.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point) in [
            ("cosh(1/x)", "0"),
            ("cosh(3/x)", "0"),
            ("cosh(1/(x - 2))", "2"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point_expr = parse_expr(&mut ctx, point);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_expr)
                .unwrap_or_else(|| panic!("cosh pole must saturate: {source} at {point}"));
            assert_eq!(display_expr(&ctx, out), "infinity", "{source} at {point}");
        }
    }

    #[test]
    fn bilateral_even_cosh_rule_declines_non_cosh_and_non_pole() {
        // Odd outers (sinh: sides disagree), oscillating inners (cosh(sin(1/x))
        // has no inner infinity), and a convergent inner (cosh(x): inner -> 0)
        // must not be folded to +inf by this rule.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in ["sinh(1/x)", "cosh(sin(1/x))", "cosh(x)", "cos(1/x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_bilateral_even_saturating_pole_rule(&mut ctx, expr, x, zero).is_none(),
                "even-cosh pole rule must decline: {source}"
            );
        }
    }

    #[test]
    fn oscillating_outer_over_even_pole_stays_residual() {
        // sin/cos of an even pole (inner -> +inf both sides) oscillate, so the
        // limit does not exist. The composition rule must decline rather than
        // leak an unfolded sin(infinity)/cos(infinity) atom. Finite arguments
        // and saturating outers (atan/cosh) are unaffected.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in ["cos(1/x^2)", "sin(1/x^2)", "cos(1/x^4)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_total_real_unary_composition_rule(&mut ctx, expr, x, zero).is_none(),
                "oscillating outer over an even pole must decline: {source}"
            );
        }
        // A saturating sibling still resolves (raw atan(infinity), which the
        // eval layer folds to pi/2 - foldable, unlike the oscillating atoms),
        // and a finite argument folds cleanly here.
        let atan_pole = parse_expr(&mut ctx, "atan(1/x^2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, atan_pole, x, zero).is_some(),
            "atan over an even pole must still resolve"
        );
        let cos_finite = parse_expr(&mut ctx, "cos(x^2)");
        let cos_finite_out = try_limit_rules_at_finite(&mut ctx, cos_finite, x, zero)
            .expect("cos of a finite-argument limit must resolve");
        assert_eq!(display_expr(&ctx, cos_finite_out), "1");
    }

    #[test]
    fn finite_squeeze_bounded_product_collapses_to_zero() {
        // Squeeze theorem: an infinitesimal times a bounded oscillator
        // tends to 0 even though the oscillator itself has no limit.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point) in [
            ("x * sin(1/x)", "0"),
            ("x^2 * cos(1/x)", "0"),
            ("sin(x) * sin(1/x)", "0"),
            ("x * sin(1/x^2)", "0"),
            ("-x * sin(1/x)", "0"),
            ("3 * x * sin(1/x) * cos(1/x)", "0"),
            ("(x - 2) * sin(1/(x - 2))", "2"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point_expr = parse_expr(&mut ctx, point);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_expr)
                .unwrap_or_else(|| panic!("squeeze must resolve: {source} at {point}"));
            assert_eq!(display_expr(&ctx, out), "0", "{source} at {point}");
        }
    }

    #[test]
    fn finite_squeeze_declines_unsound_shapes() {
        // Each of these must stay residual: a bare oscillator (no
        // infinitesimal), a scaled oscillator (no infinitesimal), an
        // unbounded outer (tan), a divergent cofactor (1/x), and a
        // domain-restricted argument (ln/sqrt are not two-sided).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in [
            "sin(1/x)",
            "2 * sin(1/x)",
            "x * tan(1/x)",
            "x * sin(ln(x))",
            "x * sin(1/sqrt(x))",
            // Identically-zero denominator: sin(1/(x - x)) = sin(1/0) is
            // undefined on the whole neighbourhood, so it is NOT a bounded
            // oscillator and the product has no limit.
            "x * sin(1/(x - x))",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_squeeze_bounded_product_rule(&mut ctx, expr, x, zero).is_none(),
                "squeeze must decline: {source}"
            );
        }
    }

    #[test]
    fn finite_zero_times_unbounded_function_stays_residual() {
        // SOUNDNESS: 0 * infinity is indeterminate. An infinitesimal times an
        // UNBOUNDED function (sinh/cosh/exp of an argument that diverges) must
        // NOT collapse to 0 - the divergent cofactor can dominate
        // (x * sinh(1/x^2) -> +inf). These stay honest residuals.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in [
            "x * sinh(1/x^2)",
            "x * cosh(1/x^2)",
            "x * exp(1/x^2)",
            "x * cosh(1/x)",
            "2 * x * sinh(1/x^2)",
            "x^2 * sinh(1/x^2)",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                try_limit_rules_at_finite(&mut ctx, expr, x, zero).is_none(),
                "0 * unbounded must stay residual: {source}"
            );
        }
    }

    #[test]
    fn finite_unbounded_function_saturates_and_bounded_product_collapses() {
        // The saturation fold makes a growing function of a divergent argument
        // resolve (sinh(1/x^2) -> inf, tanh(1/x^2) -> 1, exp(-1/x^2) -> 0), and
        // a genuinely DECAYING or BOUNDED cofactor still collapses the product.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("sinh(1/x^2)", "infinity"),
            ("cosh(1/x^2)", "infinity"),
            ("tanh(1/x^2)", "1"),
            ("x * exp(-1/x^2)", "0"),
            ("x * tanh(1/x^2)", "0"),
            ("x * sin(1/x^2)", "0"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_radical_conjugate_resolves_removable_root_quotients() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src, num, den) in [
            ("(sqrt(x) - 2)/(x - 4)", "4", 1, 4),
            ("(sqrt(x) - 3)/(x - 9)", "9", 1, 6),
            ("(sqrt(x + 1) - 2)/(x - 3)", "3", 1, 4),
            ("(2*sqrt(x) - 4)/(x - 4)", "4", 1, 2),
            ("(sqrt(2*x + 1) - 3)/(x - 4)", "4", 1, 3),
            ("(sqrt(x) - 2)/(x^2 - 16)", "4", 1, 32),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            let out = apply_finite_radical_conjugate_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn finite_radical_conjugate_declines_non_zero_over_zero_and_irrational() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src) in [
            // numerator nonzero at the point (pole, not removable):
            ("(sqrt(x) - 2)/(x - 1)", "1"),
            // irrational radical value at the point:
            ("(sqrt(x) - 1)/(x - 2)", "2"),
            // not a 0/0 form (denominator nonzero):
            ("(sqrt(x) - 2)/(x - 9)", "9"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            assert!(
                apply_finite_radical_conjugate_rule(&mut ctx, expr, x, point).is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn finite_radical_difference_conjugate_resolves_sqrt_minus_sqrt() {
        // (s1 sqrt(L1) + s2 sqrt(L2))/den at a 0/0 point, rationalized by the
        // conjugate. Values cross-checked numerically (mpmath dps 40).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src, num, den) in [
            ("(sqrt(1+x) - sqrt(1-x))/x", "0", 1, 1),
            ("(sqrt(1+2*x) - sqrt(1-2*x))/x", "0", 2, 1),
            ("(sqrt(4+x) - sqrt(4-x))/x", "0", 1, 2),
            ("(sqrt(x+3) - sqrt(2*x+2))/(x-1)", "1", -1, 4),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            let out = apply_finite_radical_difference_conjugate_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn finite_radical_difference_conjugate_declines_pole_irrational_and_nonlinear() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src) in [
            // numerator nonzero at the point (sqrt(2) - 1 != 0): a pole.
            ("(sqrt(2+x) - sqrt(1-x))/x", "0"),
            // a SUM of square roots (both positive): not 0/0 at the point.
            ("(sqrt(1+x) + sqrt(1-x))/x", "0"),
            // a nonlinear radicand is out of scope (1 + x^2).
            ("(sqrt(1+x) - sqrt(1+x^2))/x", "0"),
            // irrational radical value at the point (sqrt(2)).
            ("(sqrt(2+x) - sqrt(2-x))/x", "0"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            assert!(
                apply_finite_radical_difference_conjugate_rule(&mut ctx, expr, x, point).is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn finite_rational_polynomial_limit_resolves_exact_removable_holes_only() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "1");

        let simple_hole = parse_expr(&mut ctx, "(x^2 - 1)/(x - 1)");
        let simple_hole_out = try_limit_rules_at_finite(&mut ctx, simple_hole, x, point)
            .expect("expected exact removable rational-polynomial limit");
        let Expr::Number(value) = ctx.get(simple_hole_out) else {
            panic!("expected exact numeric removable rational-polynomial limit");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));

        let higher_numerator_multiplicity = parse_expr(&mut ctx, "(x - 1)^2/(x - 1)");
        let higher_numerator_out =
            try_limit_rules_at_finite(&mut ctx, higher_numerator_multiplicity, x, point)
                .expect("expected removable zero limit");
        let Expr::Number(value) = ctx.get(higher_numerator_out) else {
            panic!("expected exact zero removable rational-polynomial limit");
        };
        assert_eq!(value, &BigRational::zero());

        let finite_pole = parse_expr(&mut ctx, "(x - 1)/(x - 1)^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, finite_pole, x, point).is_none(),
            "odd-order finite poles must remain residual because the two-sided limit diverges differently"
        );

        let even_positive_pole = parse_expr(&mut ctx, "2/(x - 1)^2");
        let even_positive_out = try_limit_rules_at_finite(&mut ctx, even_positive_pole, x, point)
            .expect("expected bilateral even-order rational pole");
        assert_eq!(display_expr(&ctx, even_positive_out), "infinity");

        let even_negative_pole = parse_expr(&mut ctx, "-2/(x - 1)^2");
        let even_negative_out = try_limit_rules_at_finite(&mut ctx, even_negative_pole, x, point)
            .expect("expected negative bilateral even-order rational pole");
        assert_eq!(display_expr(&ctx, even_negative_out), "-infinity");
    }

    #[test]
    fn finite_one_sided_limits_resolve_orientation_and_simple_poles() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let abs_ratio = parse_expr(&mut ctx, "abs(x)/x");
        let right_abs = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            abs_ratio,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided abs orientation limit");
        assert_number_expr(&ctx, right_abs, 1, 1);

        let left_abs = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            abs_ratio,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided abs orientation limit");
        assert_number_expr(&ctx, left_abs, -1, 1);

        let sign_right = parse_expr(&mut ctx, "sign(x)");
        let sign_right_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sign_right,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided sign orientation limit");
        assert_number_expr(&ctx, sign_right_out, 1, 1);

        let sign_left = parse_expr(&mut ctx, "sign(x)");
        let sign_left_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sign_left,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided sign orientation limit");
        assert_number_expr(&ctx, sign_left_out, -1, 1);

        let point_one = parse_expr(&mut ctx, "1");
        let sign_quadratic_left = parse_expr(&mut ctx, "sign(x^2 - 1)");
        let sign_quadratic_left_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sign_quadratic_left,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided quadratic sign orientation limit");
        assert_number_expr(&ctx, sign_quadratic_left_out, -1, 1);

        let sign_even_left = parse_expr(&mut ctx, "sign(x^2)");
        let sign_even_left_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sign_even_left,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected even-order sign orientation limit");
        assert_number_expr(&ctx, sign_even_left_out, 1, 1);

        let reciprocal = parse_expr(&mut ctx, "1/x");
        let right_pole = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            reciprocal,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided rational pole");
        assert_eq!(display_expr(&ctx, right_pole), "infinity");

        let left_pole = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            reciprocal,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided rational pole");
        assert_eq!(display_expr(&ctx, left_pole), "-infinity");

        let ln_right_endpoint = parse_expr(&mut ctx, "ln(x)");
        let ln_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            ln_right_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided log endpoint");
        assert_eq!(display_expr(&ctx, ln_right_endpoint_out), "-infinity");

        let ln_left_endpoint = parse_expr(&mut ctx, "ln(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                ln_left_endpoint,
                x,
                point_zero,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "wrong-side log endpoint must remain residual"
        );

        let ln_neg_left_endpoint = parse_expr(&mut ctx, "ln(-x)");
        let ln_neg_left_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            ln_neg_left_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided positive-tail log endpoint");
        assert_eq!(display_expr(&ctx, ln_neg_left_endpoint_out), "-infinity");

        let log2_right_endpoint = parse_expr(&mut ctx, "log2(x)");
        let log2_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            log2_right_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected fixed-base log endpoint");
        assert_eq!(display_expr(&ctx, log2_right_endpoint_out), "-infinity");

        let reciprocal_base_log = parse_expr(&mut ctx, "log(1/2, x)");
        let reciprocal_base_log_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            reciprocal_base_log,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected reciprocal-base log endpoint");
        assert_eq!(display_expr(&ctx, reciprocal_base_log_out), "infinity");

        let point_one = parse_expr(&mut ctx, "1");
        let unit_boundary_base_above = parse_expr(&mut ctx, "log(x, (x - 1)/(x + 3))");
        let unit_boundary_base_above_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            unit_boundary_base_above,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .expect("expected unit-boundary base log endpoint from above");
        assert_eq!(
            display_expr(&ctx, unit_boundary_base_above_out),
            "-infinity"
        );

        let unit_boundary_base_below = parse_expr(&mut ctx, "log(2 - x, (x - 1)/(x + 3))");
        let unit_boundary_base_below_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            unit_boundary_base_below,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .expect("expected unit-boundary base log endpoint from below");
        assert_eq!(display_expr(&ctx, unit_boundary_base_below_out), "infinity");

        let unit_boundary_wrong_side = parse_expr(&mut ctx, "log(x, (x - 1)/(x + 3))");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                unit_boundary_wrong_side,
                x,
                point_one,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "unit-boundary base log endpoint must still reject wrong-side arguments"
        );

        let sqrt_right_endpoint = parse_expr(&mut ctx, "sqrt(x)");
        let sqrt_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sqrt_right_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided sqrt endpoint");
        assert_eq!(display_expr(&ctx, sqrt_right_endpoint_out), "0");

        let sqrt_left_endpoint = parse_expr(&mut ctx, "sqrt(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                sqrt_left_endpoint,
                x,
                point_zero,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "wrong-side sqrt endpoint must remain residual"
        );

        let sqrt_neg_left_endpoint = parse_expr(&mut ctx, "sqrt(-x)");
        let sqrt_neg_left_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sqrt_neg_left_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided sqrt endpoint");
        assert_eq!(display_expr(&ctx, sqrt_neg_left_endpoint_out), "0");

        let sqrt_even_endpoint = parse_expr(&mut ctx, "sqrt(x^2)");
        let sqrt_even_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sqrt_even_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected even-order sqrt endpoint");
        assert_eq!(display_expr(&ctx, sqrt_even_endpoint_out), "0");

        let sqrt_abs_endpoint = parse_expr(&mut ctx, "sqrt(abs(x))");
        let sqrt_abs_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sqrt_abs_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("abs resolves by approach side: |x| = x from the right");
        assert_eq!(display_expr(&ctx, sqrt_abs_out), "0");

        let point_one = parse_expr(&mut ctx, "1");
        let acosh_right_endpoint = parse_expr(&mut ctx, "acosh(x)");
        let acosh_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acosh_right_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided acosh lower-bound endpoint");
        assert_eq!(display_expr(&ctx, acosh_right_endpoint_out), "0");

        let acosh_left_endpoint = parse_expr(&mut ctx, "acosh(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                acosh_left_endpoint,
                x,
                point_one,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "wrong-side acosh lower-bound endpoint must remain residual"
        );

        let acosh_neg_orientation_endpoint = parse_expr(&mut ctx, "acosh(2 - x)");
        let acosh_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acosh_neg_orientation_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided negative-orientation acosh lower-bound endpoint");
        assert_eq!(display_expr(&ctx, acosh_neg_orientation_endpoint_out), "0");

        let acosh_even_endpoint = parse_expr(&mut ctx, "acosh(1 + x^2)");
        let acosh_even_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acosh_even_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected even-order acosh lower-bound endpoint");
        assert_eq!(display_expr(&ctx, acosh_even_endpoint_out), "0");

        let acosh_sqrt_endpoint = parse_expr(&mut ctx, "acosh(sqrt(x))");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                acosh_sqrt_endpoint,
                x,
                point_one,
                FiniteLimitSide::Right,
            )
            .is_none(),
            "non-polynomial acosh endpoint remains residual for a later policy"
        );

        let acos_upper_left_endpoint = parse_expr(&mut ctx, "acos(x)");
        let acos_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_upper_left_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided inverse-trig upper endpoint");
        assert_eq!(display_expr(&ctx, acos_upper_left_endpoint_out), "0");

        let asin_upper_left_endpoint = parse_expr(&mut ctx, "asin(x)");
        let asin_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            asin_upper_left_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided arcsin upper endpoint");
        assert_eq!(display_expr(&ctx, asin_upper_left_endpoint_out), "pi / 2");

        let acos_upper_right_endpoint = parse_expr(&mut ctx, "acos(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                acos_upper_right_endpoint,
                x,
                point_one,
                FiniteLimitSide::Right,
            )
            .is_none(),
            "wrong-side inverse-trig upper endpoint must remain residual"
        );

        let acos_upper_neg_orientation_endpoint = parse_expr(&mut ctx, "acos(2 - x)");
        let acos_upper_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_upper_neg_orientation_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided negative-orientation inverse-trig upper endpoint");
        assert_eq!(
            display_expr(&ctx, acos_upper_neg_orientation_endpoint_out),
            "0"
        );

        let point_minus_one = parse_expr(&mut ctx, "-1");
        let acos_lower_right_endpoint = parse_expr(&mut ctx, "acos(x)");
        let acos_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_lower_right_endpoint,
            x,
            point_minus_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided inverse-trig lower endpoint");
        assert_eq!(display_expr(&ctx, acos_lower_right_endpoint_out), "pi");

        let asin_lower_right_endpoint = parse_expr(&mut ctx, "asin(x)");
        let asin_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            asin_lower_right_endpoint,
            x,
            point_minus_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided arcsin lower endpoint");
        assert_eq!(display_expr(&ctx, asin_lower_right_endpoint_out), "-pi / 2");

        let asin_lower_left_endpoint = parse_expr(&mut ctx, "asin(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                asin_lower_left_endpoint,
                x,
                point_minus_one,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "wrong-side inverse-trig lower endpoint must remain residual"
        );

        let acos_above_domain_endpoint = parse_expr(&mut ctx, "acos(1 + x^2)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                acos_above_domain_endpoint,
                x,
                point_zero,
                FiniteLimitSide::Right,
            )
            .is_none(),
            "empty-domain one-sided inverse-trig endpoint must remain residual"
        );

        let atanh_upper_left_endpoint = parse_expr(&mut ctx, "atanh(x)");
        let atanh_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_upper_left_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided atanh upper endpoint");
        assert_eq!(
            display_expr(&ctx, atanh_upper_left_endpoint_out),
            "infinity"
        );

        let atanh_upper_right_endpoint = parse_expr(&mut ctx, "atanh(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                atanh_upper_right_endpoint,
                x,
                point_one,
                FiniteLimitSide::Right,
            )
            .is_none(),
            "wrong-side atanh upper endpoint must remain residual"
        );

        let atanh_upper_neg_orientation_endpoint = parse_expr(&mut ctx, "atanh(2 - x)");
        let atanh_upper_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_upper_neg_orientation_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided negative-orientation atanh upper endpoint");
        assert_eq!(
            display_expr(&ctx, atanh_upper_neg_orientation_endpoint_out),
            "infinity"
        );

        let atanh_lower_right_endpoint = parse_expr(&mut ctx, "atanh(x)");
        let atanh_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_lower_right_endpoint,
            x,
            point_minus_one,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided atanh lower endpoint");
        assert_eq!(
            display_expr(&ctx, atanh_lower_right_endpoint_out),
            "-infinity"
        );

        let atanh_lower_left_endpoint = parse_expr(&mut ctx, "atanh(x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                atanh_lower_left_endpoint,
                x,
                point_minus_one,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "wrong-side atanh lower endpoint must remain residual"
        );

        let atanh_above_domain_endpoint = parse_expr(&mut ctx, "atanh(1 + x^2)");
        let atanh_above_domain_endpoint_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_above_domain_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected empty-domain one-sided atanh endpoint to be undefined");
        assert_eq!(
            display_expr(&ctx, atanh_above_domain_endpoint_out),
            "undefined"
        );

        let acos_sqrt_endpoint = parse_expr(&mut ctx, "acos(sqrt(x))");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                acos_sqrt_endpoint,
                x,
                point_one,
                FiniteLimitSide::Left,
            )
            .is_none(),
            "non-polynomial inverse-trig endpoint remains residual for a later policy"
        );
    }

    #[test]
    fn finite_bilateral_abs_polynomial_ratio_resolves_only_matching_sides() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let bilateral_abs_even_pole = parse_expr(&mut ctx, "abs(x)/x^2");
        let bilateral_abs_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, bilateral_abs_even_pole, x, point_zero)
                .expect("expected matching bilateral abs polynomial-ratio pole");
        assert_eq!(display_expr(&ctx, bilateral_abs_even_pole_out), "infinity");

        let abs_orientation_jump = parse_expr(&mut ctx, "abs(x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, abs_orientation_jump, x, point_zero).is_none(),
            "bilateral abs orientation jump must remain residual when one-sided limits differ"
        );

        let sign_even_orientation = parse_expr(&mut ctx, "sign(x^2)");
        let sign_even_orientation_out =
            try_limit_rules_at_finite(&mut ctx, sign_even_orientation, x, point_zero)
                .expect("expected matching bilateral sign polynomial limit");
        assert_number_expr(&ctx, sign_even_orientation_out, 1, 1);

        let sign_orientation_jump = parse_expr(&mut ctx, "sign(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sign_orientation_jump, x, point_zero).is_none(),
            "bilateral sign orientation jump must remain residual when one-sided limits differ"
        );

        let point_one = parse_expr(&mut ctx, "1");
        let shifted_abs_even_pole = parse_expr(&mut ctx, "abs(x - 1)/(x - 1)^2");
        let shifted_abs_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, shifted_abs_even_pole, x, point_one)
                .expect("expected shifted matching bilateral abs polynomial-ratio pole");
        assert_eq!(display_expr(&ctx, shifted_abs_even_pole_out), "infinity");
    }

    #[test]
    fn finite_bilateral_trig_power_poles_resolve_only_matching_sides() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let even_sine_pole = parse_expr(&mut ctx, "1/sin(x)^2");
        let even_sine_pole_out = try_limit_rules_at_finite(&mut ctx, even_sine_pole, x, point_zero)
            .expect("expected bilateral even-order sine pole");
        assert_eq!(display_expr(&ctx, even_sine_pole_out), "infinity");

        let negative_scaled_sine_pole = parse_expr(&mut ctx, "-1/sin(2*x)^2");
        let negative_scaled_sine_pole_out =
            try_limit_rules_at_finite(&mut ctx, negative_scaled_sine_pole, x, point_zero)
                .expect("expected negative bilateral even-order sine pole");
        assert_eq!(
            display_expr(&ctx, negative_scaled_sine_pole_out),
            "-infinity"
        );

        let first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, first_order_sine_pole, x, point_zero).is_none(),
            "bilateral first-order sine pole must remain residual when one-sided limits differ"
        );

        let reciprocal_sine_even_pole = parse_expr(&mut ctx, "csc(x + pi)^2");
        let reciprocal_sine_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, reciprocal_sine_even_pole, x, point_zero)
                .expect("expected bilateral even-order reciprocal sine pole");
        assert_eq!(
            display_expr(&ctx, reciprocal_sine_even_pole_out),
            "infinity"
        );

        let reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "csc(x + pi)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, reciprocal_sine_first_order_pole, x, point_zero)
                .is_none(),
            "bilateral first-order reciprocal sine pole must remain residual"
        );

        let right_reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "csc(x + pi)");
        let right_reciprocal_sine_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            right_reciprocal_sine_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided first-order reciprocal sine pole");
        assert_eq!(
            display_expr(&ctx, right_reciprocal_sine_first_order_pole_out),
            "-infinity"
        );

        let scaled_right_reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "-2*csc(x + pi)");
        let scaled_right_reciprocal_sine_first_order_pole_out =
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                scaled_right_reciprocal_sine_first_order_pole,
                x,
                point_zero,
                FiniteLimitSide::Right,
            )
            .expect("expected scaled right-sided first-order reciprocal sine pole");
        assert_eq!(
            display_expr(&ctx, scaled_right_reciprocal_sine_first_order_pole_out),
            "infinity"
        );

        let negative_scaled_reciprocal_sine_even_pole = parse_expr(&mut ctx, "-3*csc(x + pi)^2");
        let negative_scaled_reciprocal_sine_even_pole_out = try_limit_rules_at_finite(
            &mut ctx,
            negative_scaled_reciprocal_sine_even_pole,
            x,
            point_zero,
        )
        .expect("expected negative scaled bilateral even-order reciprocal sine pole");
        assert_eq!(
            display_expr(&ctx, negative_scaled_reciprocal_sine_even_pole_out),
            "-infinity"
        );

        let right_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
        let right_first_order_sine_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            right_first_order_sine_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided first-order sine pole");
        assert_eq!(
            display_expr(&ctx, right_first_order_sine_pole_out),
            "infinity"
        );

        let left_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
        let left_first_order_sine_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            left_first_order_sine_pole,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided first-order sine pole");
        assert_eq!(
            display_expr(&ctx, left_first_order_sine_pole_out),
            "-infinity"
        );

        let point_one = parse_expr(&mut ctx, "1");
        let shifted_sine_pole = parse_expr(&mut ctx, "1/sin(x - 1)^2");
        let shifted_sine_pole_out =
            try_limit_rules_at_finite(&mut ctx, shifted_sine_pole, x, point_one)
                .expect("expected shifted bilateral even-order sine pole");
        assert_eq!(display_expr(&ctx, shifted_sine_pole_out), "infinity");

        let even_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)^2");
        let even_cosine_pole_out =
            try_limit_rules_at_finite(&mut ctx, even_cosine_pole, x, point_zero)
                .expect("expected bilateral even-order cosine pole at a tabulated zero");
        assert_eq!(display_expr(&ctx, even_cosine_pole_out), "infinity");

        let negative_scaled_cosine_pole = parse_expr(&mut ctx, "-1/cos(pi/2 + 2*x)^2");
        let negative_scaled_cosine_pole_out =
            try_limit_rules_at_finite(&mut ctx, negative_scaled_cosine_pole, x, point_zero)
                .expect("expected negative bilateral even-order cosine pole");
        assert_eq!(
            display_expr(&ctx, negative_scaled_cosine_pole_out),
            "-infinity"
        );

        let first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, first_order_cosine_pole, x, point_zero).is_none(),
            "bilateral first-order cosine pole must remain residual when one-sided limits differ"
        );

        let reciprocal_cosine_even_pole = parse_expr(&mut ctx, "sec(pi/2 + x)^2");
        let reciprocal_cosine_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, reciprocal_cosine_even_pole, x, point_zero)
                .expect("expected bilateral even-order reciprocal cosine pole");
        assert_eq!(
            display_expr(&ctx, reciprocal_cosine_even_pole_out),
            "infinity"
        );

        let scaled_right_reciprocal_cosine_first_order_pole =
            parse_expr(&mut ctx, "2*sec(pi/2 + x)");
        let scaled_right_reciprocal_cosine_first_order_pole_out =
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                scaled_right_reciprocal_cosine_first_order_pole,
                x,
                point_zero,
                FiniteLimitSide::Right,
            )
            .expect("expected scaled right-sided first-order reciprocal cosine pole");
        assert_eq!(
            display_expr(&ctx, scaled_right_reciprocal_cosine_first_order_pole_out),
            "-infinity"
        );

        let right_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
        let right_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            right_first_order_cosine_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided first-order cosine pole");
        assert_eq!(
            display_expr(&ctx, right_first_order_cosine_pole_out),
            "-infinity"
        );

        let left_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
        let left_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            left_first_order_cosine_pole,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided first-order cosine pole");
        assert_eq!(
            display_expr(&ctx, left_first_order_cosine_pole_out),
            "infinity"
        );

        let point_pi_over_two = parse_expr(&mut ctx, "pi/2");
        let direct_special_point_cosine_pole = parse_expr(&mut ctx, "1/cos(x)^2");
        let direct_special_point_cosine_pole_out = try_limit_rules_at_finite(
            &mut ctx,
            direct_special_point_cosine_pole,
            x,
            point_pi_over_two,
        )
        .expect("expected bilateral even-order cosine pole at direct special-angle point");
        assert_eq!(
            display_expr(&ctx, direct_special_point_cosine_pole_out),
            "infinity"
        );

        let direct_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(x)");
        assert!(
            try_limit_rules_at_finite(
                &mut ctx,
                direct_first_order_cosine_pole,
                x,
                point_pi_over_two
            )
            .is_none(),
            "direct bilateral first-order cosine pole must remain residual"
        );

        let direct_right_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(x)");
        let direct_right_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            direct_right_first_order_cosine_pole,
            x,
            point_pi_over_two,
            FiniteLimitSide::Right,
        )
        .expect("expected direct right-sided first-order cosine pole at special-angle point");
        assert_eq!(
            display_expr(&ctx, direct_right_first_order_cosine_pole_out),
            "-infinity"
        );

        let point_two_pi = parse_expr(&mut ctx, "2*pi");
        let direct_rational_pi_sine_pole = parse_expr(&mut ctx, "1/sin(x)^2");
        let direct_rational_pi_sine_pole_out =
            try_limit_rules_at_finite(&mut ctx, direct_rational_pi_sine_pole, x, point_two_pi)
                .expect("expected bilateral even-order sine pole at rational-pi point");
        assert_eq!(
            display_expr(&ctx, direct_rational_pi_sine_pole_out),
            "infinity"
        );

        let direct_rational_pi_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
        assert!(
            try_limit_rules_at_finite(
                &mut ctx,
                direct_rational_pi_first_order_sine_pole,
                x,
                point_two_pi
            )
            .is_none(),
            "direct bilateral first-order sine pole at rational-pi point must remain residual"
        );

        let right_tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
        let right_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            right_tangent_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided first-order tangent pole");
        assert_eq!(
            display_expr(&ctx, right_tangent_first_order_pole_out),
            "-infinity"
        );

        let left_tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
        let left_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            left_tangent_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .expect("expected left-sided first-order tangent pole");
        assert_eq!(
            display_expr(&ctx, left_tangent_first_order_pole_out),
            "infinity"
        );

        let tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, tangent_first_order_pole, x, point_zero).is_none(),
            "bilateral first-order tangent pole must remain residual"
        );

        let tangent_even_pole = parse_expr(&mut ctx, "tan(pi/2 + x)^2");
        let tangent_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, tangent_even_pole, x, point_zero)
                .expect("expected bilateral even-order tangent pole");
        assert_eq!(display_expr(&ctx, tangent_even_pole_out), "infinity");

        let right_cotangent_first_order_pole = parse_expr(&mut ctx, "cot(x + pi)");
        let right_cotangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            right_cotangent_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided first-order cotangent pole");
        assert_eq!(
            display_expr(&ctx, right_cotangent_first_order_pole_out),
            "infinity"
        );

        let explicit_right_tangent_first_order_pole =
            parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x)");
        let explicit_right_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            explicit_right_tangent_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided explicit sine/cosine tangent-ratio pole");
        assert_eq!(
            display_expr(&ctx, explicit_right_tangent_first_order_pole_out),
            "-infinity"
        );

        let explicit_tangent_first_order_pole = parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, explicit_tangent_first_order_pole, x, point_zero)
                .is_none(),
            "explicit bilateral first-order tangent-ratio pole must remain residual"
        );

        let explicit_tangent_even_pole = parse_expr(&mut ctx, "(sin(pi/2 + x)/cos(pi/2 + x))^2");
        let explicit_tangent_even_pole_out =
            try_limit_rules_at_finite(&mut ctx, explicit_tangent_even_pole, x, point_zero)
                .expect("expected bilateral even-order explicit tangent-ratio pole");
        assert_eq!(
            display_expr(&ctx, explicit_tangent_even_pole_out),
            "infinity"
        );

        let explicit_right_cotangent_first_order_pole =
            parse_expr(&mut ctx, "cos(x + pi)/sin(x + pi)");
        let explicit_right_cotangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            explicit_right_cotangent_first_order_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected right-sided explicit cosine/sine cotangent-ratio pole");
        assert_eq!(
            display_expr(&ctx, explicit_right_cotangent_first_order_pole_out),
            "infinity"
        );

        let cross_argument_explicit_tangent_pole =
            parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 - x)");
        let cross_argument_explicit_tangent_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            cross_argument_explicit_tangent_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected cross-argument explicit tangent-ratio pole");
        assert_eq!(
            display_expr(&ctx, cross_argument_explicit_tangent_pole_out),
            "infinity"
        );

        let noisy_denominator_explicit_tangent_pole =
            parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x + 0)");
        let noisy_denominator_explicit_tangent_pole_out = try_limit_rules_at_finite_one_sided(
            &mut ctx,
            noisy_denominator_explicit_tangent_pole,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .expect("expected explicit tangent-ratio pole with harmless denominator noise");
        assert_eq!(
            display_expr(&ctx, noisy_denominator_explicit_tangent_pole_out),
            "-infinity"
        );

        let zero_numerator_explicit_trig_ratio = parse_expr(&mut ctx, "sin(x)/cos(pi/2 - x)");
        assert!(
            try_limit_rules_at_finite_one_sided(
                &mut ctx,
                zero_numerator_explicit_trig_ratio,
                x,
                point_zero,
                FiniteLimitSide::Right
            )
            .is_none(),
            "explicit trig ratio with zero numerator limit must not be treated as a pole"
        );

        let point_three_pi_over_two = parse_expr(&mut ctx, "3*pi/2");
        let direct_rational_pi_cosine_pole = parse_expr(&mut ctx, "1/cos(x)^2");
        let direct_rational_pi_cosine_pole_out = try_limit_rules_at_finite(
            &mut ctx,
            direct_rational_pi_cosine_pole,
            x,
            point_three_pi_over_two,
        )
        .expect("expected bilateral even-order cosine pole at rational-pi point");
        assert_eq!(
            display_expr(&ctx, direct_rational_pi_cosine_pole_out),
            "infinity"
        );
    }

    #[test]
    fn finite_sine_zero_quotient_limit_uses_removable_polynomial_ratio() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let cases = [
            ("sin(x)/x", 1, 1),
            ("sin(2*x)/x", 2, 1),
            ("3*sin(2*x)/(5*x)", 6, 5),
            ("sin(x^2)/x", 0, 1),
        ];

        for (input, expected_num, expected_den) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
                .unwrap_or_else(|| panic!("expected finite sine quotient limit for {input}"));
            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected exact rational sine quotient limit for {input}");
            };
            assert_eq!(
                value,
                &BigRational::new(BigInt::from(expected_num), BigInt::from(expected_den))
            );
        }

        let point_one = parse_expr(&mut ctx, "1");
        let shifted = parse_expr(&mut ctx, "sin(x - 1)/(x^2 - 1)");
        let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
            .expect("expected shifted finite sine quotient limit");
        let Expr::Number(value) = ctx.get(shifted_out) else {
            panic!("expected exact shifted sine quotient limit");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

        let nonzero_argument = parse_expr(&mut ctx, "sin(x + 1)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, nonzero_argument, x, point_zero).is_none(),
            "sine quotient rule must only apply when the sine argument tends to zero"
        );

        let finite_pole = parse_expr(&mut ctx, "sin(x)/x^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
            "sine quotient rule must not promote finite poles"
        );
    }

    #[test]
    fn finite_equivalent_infinitesimal_quotient_resolves_ratios() {
        // Ratio of first-order equivalent infinitesimals: inversion,
        // composition, the missing atoms (tan/asin/arctan/sinh/tanh),
        // exp, sign, and a Sub INSIDE an atom argument (which is exact).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("x / sin(x)", "1"),
            ("sin(3*x) / sin(5*x)", "3/5"),
            ("sin(x) / sin(2*x)", "1/2"),
            ("tan(x) / x", "1"),
            ("asin(x) / x", "1"),
            ("arctan(x) / x", "1"),
            ("sinh(x) / x", "1"),
            ("tanh(x) / x", "1"),
            ("sin(-3*x) / sin(x)", "-3"),
            ("tan(2*x) / sin(3*x)", "2/3"),
            ("(exp(x) - 1) / sin(x)", "1"),
            ("sin(x^2 - x) / x", "-1"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero).unwrap_or_else(|| {
                panic!("equivalent-infinitesimal quotient must resolve: {source}")
            });
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_equivalent_infinitesimal_quotient_declines_unsound_shapes() {
        // Each must stay residual: higher-order (cos / cubic Taylor) and
        // sum-cancellation forms (first-order equivalents are invalid inside
        // a difference), cos (not a zero atom), a finite pole, and an atom
        // whose argument does NOT tend to 0 at the point.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in [
            "(1 - cos(x)) / x^2",
            "(sin(x) - x) / x^3",
            "(tan(x) - x) / x^3",
            "cos(x) / sin(x)",
            "sin(x) / x^2",
            "sin(x + 1) / x",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_equivalent_infinitesimal_quotient_rule(&mut ctx, expr, x, zero)
                    .is_none(),
                "equivalent-infinitesimal quotient must decline: {source}"
            );
        }
    }

    #[test]
    fn power_log_polynomial_dominance_resolves_antiderivative_endpoints() {
        // The one-sided limits of x^a ln(x)^b antiderivatives at 0+, which the
        // definite integrator needs to certify int_0^1 ln(x)^2 = 2 etc.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in [
            "x * (ln(x)^2 - 2*ln(x) + 2)",
            "x^2 * (2*ln(x) - 1)",
            "2*sqrt(x)*ln(x) - 4*sqrt(x)",
            "x * (ln(x) - 1)",
            "x^(3/2) * ln(x)^3",
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_finite_one_sided(
                &mut ctx,
                expr,
                x,
                zero,
                FiniteLimitSide::Right,
            )
            .unwrap_or_else(|| panic!("power-log dominance must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), "0", "{source}");
        }
    }

    #[test]
    fn power_log_polynomial_dominance_declines_non_vanishing() {
        // Each must NOT be folded to 0: no positive power (pure log diverges),
        // a bare constant term (tends to that constant), a negative power
        // (diverges), and a non-power/non-log factor.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in ["ln(x)^2", "x * ln(x) + 5", "ln(x) / x", "sin(x) * ln(x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_one_sided_power_log_polynomial_zero(
                    &mut ctx,
                    expr,
                    x,
                    zero,
                    FiniteLimitSide::Right
                )
                .is_none(),
                "power-log dominance must decline: {source}"
            );
        }
    }

    #[test]
    fn finite_exp_zero_quotient_limit_uses_removable_polynomial_ratio() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let cases = [
            ("(exp(x)-1)/x", 1, 1),
            ("(exp(2*x)-1)/x", 2, 1),
            ("3*(exp(2*x)-1)/(5*x)", 6, 5),
            ("(exp(x^2)-1)/x", 0, 1),
            ("(1-exp(2*x))/x", -2, 1),
        ];

        for (input, expected_num, expected_den) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
                .unwrap_or_else(|| panic!("expected finite exp quotient limit for {input}"));
            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected exact rational exp quotient limit for {input}");
            };
            assert_eq!(
                value,
                &BigRational::new(BigInt::from(expected_num), BigInt::from(expected_den))
            );
        }

        let point_one = parse_expr(&mut ctx, "1");
        let shifted = parse_expr(&mut ctx, "(exp(x - 1) - 1)/(x^2 - 1)");
        let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
            .expect("expected shifted finite exp quotient limit");
        let Expr::Number(value) = ctx.get(shifted_out) else {
            panic!("expected exact shifted exp quotient limit");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

        let nonzero_argument = parse_expr(&mut ctx, "(exp(x + 1) - 1)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, nonzero_argument, x, point_zero).is_none(),
            "exp quotient rule must only apply when the exponent tends to zero"
        );

        let finite_pole = parse_expr(&mut ctx, "(exp(x)-1)/x^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
            "exp quotient rule must not promote finite poles"
        );
    }

    #[test]
    fn finite_exp_combination_ratio_yields_ratio_of_derivatives() {
        // (sum exp)/(sum exp) at 0 -> N'(0)/D'(0), a ratio of log combinations.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(2^x-3^x)/(5^x-7^x)", "(ln(2) - ln(3)) / (ln(5) - ln(7))"),
            ("(2^x-3^x)/(2^x-5^x)", "(ln(2) - ln(3)) / (ln(2) - ln(5))"),
            ("(2^x-2^x)/(5^x-7^x)", "0"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("exp combination ratio must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_exp_combination_ratio_declines_cancelled_denominator() {
        // A denominator whose first derivative cancels to 0 - trivially
        // (5^x-5^x) or via a log identity (ln6 = ln2 + ln3) - is a higher-order
        // form and must stay residual, not divide by zero.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in ["(2^x-3^x)/(5^x-5^x)", "(2^x-3^x)/(6^x-2^x-3^x+1)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero).is_none(),
                "exp combination ratio must decline a cancelled denominator: {source}"
            );
        }
    }

    #[test]
    fn finite_general_exp_ratio_yields_ratio_of_logs() {
        // (a^x - 1)/(b^x - 1) -> ln(a)/ln(b): ratio of first-order coefficients.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(3^x-1)/(2^x-1)", "ln(3) / ln(2)"),
            ("(2^x-1)/(3^x-1)", "ln(2) / ln(3)"),
            ("(2^(2*x)-1)/(2^x-1)", "2"),
            ("(3^x-1)/(3^x-1)", "1"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("exp ratio must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_general_exp_ratio_declines_non_exp_denominator() {
        // A non-(b^x-1) denominator (sin, x) and a non-zero point decline.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let one = parse_expr(&mut ctx, "1");
        for source in ["(2^x-1)/sin(x)", "(2^x-1)/x", "(exp(x)-1)/(2^x-1)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, zero).is_none(),
                "exp ratio must decline: {source}"
            );
        }
        let expr = parse_expr(&mut ctx, "(3^x-1)/(2^x-1)");
        assert!(
            apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, one).is_none(),
            "exp ratio is the form at 0"
        );
    }

    #[test]
    fn finite_general_exp_zero_quotient_yields_log_of_base() {
        // (a^g - 1)/h -> ln(a) lim(g/h): the derivative of a^x at 0.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(2^x - 1)/x", "ln(2)"),
            ("(3^x - 1)/x", "ln(3)"),
            ("(2^(3*x) - 1)/x", "3 * ln(2)"),
            ("(2^x - 1)/(2*x)", "1/2 * ln(2)"),
            ("(10^x - 1)/x", "ln(10)"),
            ("(1 - 5^x)/x", "-ln(5)"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("general exp quotient must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_general_exp_zero_quotient_declines_e_unit_base_and_poles() {
        // Base e is left to the exp rule; base 1 has no log; a finite pole and a
        // non-vanishing exponent stay residual via this rule.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for source in ["(1^x - 1)/x", "(2^x - 1)/x^2", "(2^(x+1) - 1)/x"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_general_exp_zero_quotient_rule(&mut ctx, expr, x, zero).is_none(),
                "general exp quotient must decline: {source}"
            );
        }
        // e^x - 1 is NOT matched here (base E is not a numeric rational base).
        let exp_form = parse_expr(&mut ctx, "(exp(x) - 1)/x");
        assert!(
            apply_finite_general_exp_zero_quotient_rule(&mut ctx, exp_form, x, zero).is_none(),
            "natural base is left to the exp rule"
        );
    }

    #[test]
    fn finite_exp_linear_combination_yields_difference_of_logs() {
        // (a^x - b^x)/x -> ln(a) - ln(b): the derivative of a difference of
        // general-base exponentials at 0, where ln(a) is transcendental.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(2^x - 3^x)/x", "ln(2) - ln(3)"),
            ("(3^x - 2^x)/x", "ln(3) - ln(2)"),
            ("(2^(3*x) - 3^x)/x", "3 * ln(2) - ln(3)"),
            ("(5*2^x - 5*3^x)/x", "5 * ln(2) - 5 * ln(3)"),
            ("(exp(x) - 2^x)/x", "-ln(2) + 1"),
            ("(2*exp(x) - 2^x - 1)/x", "-ln(2) + 2"),
            ("(2^x + 3^x - 2)/x", "ln(2) + ln(3)"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("exp combination must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_exp_linear_combination_declines_non_class_and_non_indeterminate() {
        // Honest declines: a non-vanishing numerator (not 0/0), a higher-order
        // denominator, a foreign-variable exponent, and a bare oscillation.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let one = parse_expr(&mut ctx, "1");
        for source in [
            "(2^x + 3^x)/x",   // numerator -> 2, not 0/0
            "2^x/x",           // numerator -> 1, not 0/0
            "(2^x - 3^x)/x^2", // denominator vanishes to second order
            "(2^x - 3^y)/x",   // foreign-variable exponent is not a polynomial in x
            "sin(1/x)",        // not a quotient of this class (honesty list)
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, zero)
                    .is_none(),
                "exp combination must decline: {source}"
            );
        }
        // Only defined at the origin (a^x -> 1 needs the exponent at 0).
        let expr = parse_expr(&mut ctx, "(2^x - 3^x)/x");
        assert!(
            apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, one).is_none(),
            "exp combination is only defined at the origin"
        );
    }

    #[test]
    fn finite_taylor_quotient_resolves_higher_order_zero_over_zero() {
        // Both sides vanish at 0; the limit is the ratio of leading Taylor
        // coefficients once the numerator's order matches the denominator's.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(1 - cos(x))/x^2", "1/2"),
            ("(sin(x) - x)/x^3", "-1/6"),
            ("(x - sin(x))/x^3", "1/6"),
            ("(tan(x) - x)/x^3", "1/3"),
            ("(exp(x) - 1 - x)/x^2", "1/2"),
            ("(cosh(x) - 1)/x^2", "1/2"),
            ("(sinh(x) - x)/x^3", "1/6"),
            ("(1 - cos(2*x))/x^2", "2"),
            ("(ln(1+x) - x)/x^2", "-1/2"),
            ("(arctan(x) - x)/x^3", "-1/3"),
            ("(arcsin(x) - x)/x^3", "1/6"),
            ("(1 - cos(x))/(x*sin(x))", "1/2"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = apply_finite_taylor_quotient_rule(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("taylor quotient must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_taylor_quotient_declines_non_vanishing_and_unsupported() {
        // Honest declines: a numerator that does not out-vanish the denominator
        // (m < d), an oscillation whose argument does not tend to 0, a constant
        // numerator (den vanishes alone), and a nonzero approach point.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let one = parse_expr(&mut ctx, "1");
        for source in [
            "(1 - cos(x))/x^3", // numerator order 2 < denominator order 3
            "sin(1/x)/x",       // argument 1/x does not tend to 0 (honesty list)
            "cos(x)/x",         // numerator does not vanish at 0
            "x/sin(1/x)",       // unsupported inner series
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_taylor_quotient_rule(&mut ctx, expr, x, zero).is_none(),
                "taylor quotient must decline: {source}"
            );
        }
        // A nonzero approach point is out of scope for the at-zero series.
        let expr = parse_expr(&mut ctx, "(1 - cos(x))/x^2");
        assert!(
            apply_finite_taylor_quotient_rule(&mut ctx, expr, x, one).is_none(),
            "taylor quotient is only defined at the origin"
        );
    }

    #[test]
    fn finite_lhopital_nonzero_point_resolves_shifted_zero_over_zero() {
        // 0/0 forms whose vanishing happens at a non-zero point: the at-zero
        // equivalent/Taylor rules cannot reach them, so L'Hôpital differentiates
        // and re-evaluates until the form is no longer 0/0. Values cross-checked
        // numerically (mpmath dps 40).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src, expected) in [
            ("sin(x)/(x-pi)", "pi", "-1"),
            ("(x-pi)/sin(x)", "pi", "-1"),
            ("tan(x)/sin(x)", "pi", "-1"),
            ("cos(x)/(x-pi/2)", "pi/2", "-1"),
            ("(1 - cos(x-1))/(x-1)^2", "1", "1/2"), // two applications
            ("(sin(x-1) - (x-1))/(x-1)^3", "1", "-1/6"), // three applications
            ("sin(x-3)/(x^2-9)", "3", "1/6"),
            ("ln(x)/(x-1)", "1", "1"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            let out = apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("L'Hôpital must resolve {source} at {point_src}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source} at {point_src}");
        }
    }

    #[test]
    fn finite_lhopital_declines_poles_non_quotients_and_origin() {
        // Honest declines: a pole (numerator does not vanish while the
        // denominator does), an even-order pole that diverges bilaterally, a
        // non-0/0 quotient owned by ordinary substitution, an oscillation, and a
        // symbolic (non-rational) value which we leave residual rather than guess.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, point_src) in [
            ("1/(x-1)", "1"),               // simple pole, not 0/0
            ("1/sin(x)", "pi"),             // pole at a sine zero
            ("sin(x)/(x-pi)^2", "pi"),      // odd/even -> 1/(x-pi) blows up
            ("sin(1/(x-1))", "1"),          // inner oscillates, no limit
            ("cos(x)/cos(x)", "pi"),        // not 0/0 (continuous, owned elsewhere)
            ("(cos(x)-cos(2))/(x-2)", "2"), // value -sin(2) is not rational
        ] {
            let expr = parse_expr(&mut ctx, source);
            let point = parse_expr(&mut ctx, point_src);
            assert!(
                apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, point)
                    .is_none(),
                "L'Hôpital must decline {source} at {point_src}"
            );
        }
        // The origin is owned by the equivalent-infinitesimal / Taylor rules
        // (with their small-angle narration); L'Hôpital declines there.
        let zero = parse_expr(&mut ctx, "0");
        let expr = parse_expr(&mut ctx, "sin(x)/x");
        assert!(
            apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, zero).is_none(),
            "L'Hôpital declines at the origin"
        );
    }

    #[test]
    fn finite_log_unit_quotient_limit_uses_removable_polynomial_ratio() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let cases = [
            ("ln(1+x)/x", 1, 1),
            ("ln(1+2*x)/x", 2, 1),
            ("3*ln(1+2*x)/(5*x)", 6, 5),
            ("ln(1+x^2)/x", 0, 1),
            ("-ln(1+2*x)/x", -2, 1),
        ];

        for (input, expected_num, expected_den) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
                .unwrap_or_else(|| panic!("expected finite log unit quotient limit for {input}"));
            assert_number_expr(&ctx, out, expected_num, expected_den);
        }

        let fixed_base_cases = [
            ("log2(1+2*x)/x", 2, 1, 2, 1),
            ("3*log10(1+2*x)/(5*x)", 6, 5, 10, 1),
            ("log(3, 1+2*x)/x", 2, 1, 3, 1),
            ("5*log(1/2, 1+2*x)/(2*x)", 5, 1, 1, 2),
        ];

        for (input, expected_num, expected_den, expected_base_num, expected_base_den) in
            fixed_base_cases
        {
            let expr = parse_expr(&mut ctx, input);
            let out =
                try_limit_rules_at_finite(&mut ctx, expr, x, point_zero).unwrap_or_else(|| {
                    panic!("expected fixed-base log unit quotient limit for {input}")
                });
            assert_ratio_over_ln_base(
                &ctx,
                out,
                expected_num,
                expected_den,
                expected_base_num,
                expected_base_den,
            );
        }

        let variable_base_cases = [
            ("log(x+2, 1+2*x)/x", 2, 1, 2, 1),
            ("log(x+1/4, 1+2*x)/x", 2, 1, 1, 4),
            ("5*log(x+3/2, 1+2*x)/(2*x)", 5, 1, 3, 2),
            ("log(exp(x)+2, 1+2*x)/x", 2, 1, 3, 1),
            ("log(sin(x)+2, 1+2*x)/x", 2, 1, 2, 1),
            ("log(sqrt(x+4)+1, 1+2*x)/x", 2, 1, 3, 1),
        ];

        for (input, expected_num, expected_den, expected_base_num, expected_base_den) in
            variable_base_cases
        {
            let expr = parse_expr(&mut ctx, input);
            let out =
                try_limit_rules_at_finite(&mut ctx, expr, x, point_zero).unwrap_or_else(|| {
                    panic!("expected variable-base log unit quotient limit for {input}")
                });
            assert_ratio_over_ln_base(
                &ctx,
                out,
                expected_num,
                expected_den,
                expected_base_num,
                expected_base_den,
            );
        }

        let fixed_zero = parse_expr(&mut ctx, "log10(1+x^2)/x");
        let fixed_zero_out = try_limit_rules_at_finite(&mut ctx, fixed_zero, x, point_zero)
            .expect("expected zero fixed-base log unit quotient limit");
        assert_number_expr(&ctx, fixed_zero_out, 0, 1);

        let fixed_nonunit_argument = parse_expr(&mut ctx, "log2(2 + x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, fixed_nonunit_argument, x, point_zero).is_none(),
            "fixed-base log quotient rule must only apply when the log argument tends to one"
        );

        let fixed_finite_pole = parse_expr(&mut ctx, "log10(1+x)/x^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, fixed_finite_pole, x, point_zero).is_none(),
            "fixed-base log quotient rule must not promote finite poles"
        );

        let variable_base_one_log = parse_expr(&mut ctx, "log(x+1, 1+2*x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, variable_base_one_log, x, point_zero).is_none(),
            "binary log quotient rule must not promote variable-base quotients whose base tends to one"
        );

        let variable_base_zero_log = parse_expr(&mut ctx, "log(x, 1+2*x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, variable_base_zero_log, x, point_zero).is_none(),
            "binary log quotient rule must not promote variable-base quotients whose base tends to zero"
        );

        let variable_base_negative_log = parse_expr(&mut ctx, "log(x-1, 1+2*x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, variable_base_negative_log, x, point_zero)
                .is_none(),
            "binary log quotient rule must not promote variable-base quotients whose base tends negative"
        );

        let non_rational_variable_base_log = parse_expr(&mut ctx, "log(ln(x+3), 1+2*x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, non_rational_variable_base_log, x, point_zero)
                .is_none(),
            "binary log quotient rule must keep non-rational resolved variable bases residual"
        );

        let unit_base_log = parse_expr(&mut ctx, "log(1, 1+x)/x");
        let unit_base_log_out = try_limit_rules_at_finite(&mut ctx, unit_base_log, x, point_zero)
            .expect("constant-base log with base one has empty real domain");
        assert_eq!(
            display_expr(&ctx, unit_base_log_out),
            "undefined",
            "binary log quotient rule must reject base one as an empty real domain"
        );

        let negative_base_log = parse_expr(&mut ctx, "log(-2, 1+x)/x");
        let negative_base_log_out =
            try_limit_rules_at_finite(&mut ctx, negative_base_log, x, point_zero)
                .expect("constant-base log with negative base has empty real domain");
        assert_eq!(
            display_expr(&ctx, negative_base_log_out),
            "undefined",
            "binary log quotient rule must reject negative bases as an empty real domain"
        );

        let binary_nonunit_argument = parse_expr(&mut ctx, "log(3, 2+x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, binary_nonunit_argument, x, point_zero).is_none(),
            "binary log quotient rule must only apply when the log argument tends to one"
        );

        let point_one = parse_expr(&mut ctx, "1");
        let shifted = parse_expr(&mut ctx, "ln(x)/(x - 1)");
        let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
            .expect("expected shifted finite log unit quotient limit");
        let Expr::Number(value) = ctx.get(shifted_out) else {
            panic!("expected exact shifted log unit quotient limit");
        };
        assert_eq!(value, &BigRational::one());

        let fixed_shifted = parse_expr(&mut ctx, "log2(x)/(x - 1)");
        let fixed_shifted_out = try_limit_rules_at_finite(&mut ctx, fixed_shifted, x, point_one)
            .expect("expected shifted fixed-base log unit quotient limit");
        assert_ratio_over_ln_base(&ctx, fixed_shifted_out, 1, 1, 2, 1);

        let quadratic_argument = parse_expr(&mut ctx, "ln(x^2)/(x - 1)");
        let quadratic_out = try_limit_rules_at_finite(&mut ctx, quadratic_argument, x, point_one)
            .expect("expected quadratic log unit quotient limit");
        let Expr::Number(value) = ctx.get(quadratic_out) else {
            panic!("expected exact quadratic log unit quotient limit");
        };
        assert_eq!(value, &BigRational::from_integer(BigInt::from(2)));

        let nonunit_argument = parse_expr(&mut ctx, "ln(2 + x)/x");
        assert!(
            try_limit_rules_at_finite(&mut ctx, nonunit_argument, x, point_zero).is_none(),
            "log quotient rule must only apply when the log argument tends to one"
        );

        let finite_pole = parse_expr(&mut ctx, "ln(1+x)/x^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
            "log quotient rule must not promote finite poles"
        );
    }

    #[test]
    fn finite_elementary_polynomial_limit_handles_total_real_functions() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "-1");

        let cases = [
            ("exp(x^2 + 1)", BuiltinFn::Exp),
            ("sin(x^2 + 1)", BuiltinFn::Sin),
            ("cos(x^2 + 1)", BuiltinFn::Cos),
            ("sinh(x^2 + 1)", BuiltinFn::Sinh),
            ("cosh(x^2 + 1)", BuiltinFn::Cosh),
            ("tanh(x^2 + 1)", BuiltinFn::Tanh),
            ("atan(x^2 + 1)", BuiltinFn::Atan),
            ("arctan(x^2 + 1)", BuiltinFn::Arctan),
            ("asinh(x^2 + 1)", BuiltinFn::Asinh),
            ("cbrt(x^2 + 1)", BuiltinFn::Cbrt),
        ];

        for (input, expected_builtin) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

            let Expr::Function(fn_id, args) = ctx.get(out).clone() else {
                panic!("expected function output for {input}");
            };
            assert_eq!(ctx.builtin_of(fn_id), Some(expected_builtin));
            assert_eq!(args.len(), 1);

            let Expr::Number(value) = ctx.get(args[0]) else {
                panic!("expected numeric function argument for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(2.into()));
        }
    }

    #[test]
    fn finite_elementary_polynomial_limit_evaluates_zero_special_values() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");

        let cases = [
            ("exp(x)", 1),
            ("sin(x)", 0),
            ("cos(x)", 1),
            ("sinh(x)", 0),
            ("cosh(x)", 1),
            ("tanh(x)", 0),
            ("atan(x)", 0),
            ("arctan(x)", 0),
            ("asinh(x)", 0),
            ("cbrt(x)", 0),
            ("abs(x)", 0),
        ];

        for (input, expected) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected numeric special value for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(expected.into()));
        }
    }

    #[test]
    fn finite_abs_polynomial_limit_evaluates_exact_rational_absolute_value() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");

        let cases = [("abs(x^2 - 1)", 1), ("abs(x - 2)", 2)];

        for (input, expected) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite abs polynomial limit for {input}"));

            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected numeric absolute value for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(expected.into()));
        }
    }

    #[test]
    fn finite_real_cube_root_limit_evaluates_exact_and_symbolic_values() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let point_neg_two = parse_expr(&mut ctx, "-2");
        let exact_builtin = parse_expr(&mut ctx, "cbrt(x^3)");
        let exact_builtin_out =
            try_limit_rules_at_finite(&mut ctx, exact_builtin, x, point_neg_two)
                .expect("expected exact finite cbrt limit");
        let Expr::Number(value) = ctx.get(exact_builtin_out) else {
            panic!("expected exact cbrt limit to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer((-2).into()));

        let point_one = parse_expr(&mut ctx, "1");
        let exact_power = parse_expr(&mut ctx, "(x^2 - 9)^(1/3)");
        let exact_power_out = try_limit_rules_at_finite(&mut ctx, exact_power, x, point_one)
            .expect("expected exact finite one-third power limit");
        let Expr::Number(value) = ctx.get(exact_power_out) else {
            panic!("expected exact one-third power limit to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer((-2).into()));

        let sqrt_power = parse_expr(&mut ctx, "(2*x + 3)^(1/2)");
        let sqrt_power_out = try_limit_rules_at_finite(&mut ctx, sqrt_power, x, point_one)
            .expect("expected finite square-root power limit");
        assert_eq!(display_expr(&ctx, sqrt_power_out), "sqrt(5)");

        let sqrt_power_endpoint = parse_expr(&mut ctx, "x^(1/2)");
        let point_zero = parse_expr(&mut ctx, "0");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sqrt_power_endpoint, x, point_zero).is_none(),
            "finite square-root power endpoint must remain residual"
        );

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let symbolic_builtin = parse_expr(&mut ctx, "cbrt(x^2 + 1)");
        let symbolic_builtin_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_builtin, x, point_neg_one)
                .expect("expected symbolic finite cbrt limit");
        assert_eq!(display_expr(&ctx, symbolic_builtin_out), "cbrt(2)");
    }

    #[test]
    fn finite_total_real_unary_composition_limit_reuses_resolved_sublimits() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let point_zero = parse_expr(&mut ctx, "0");
        let nested_trig = parse_expr(&mut ctx, "cos(sin(x))");
        let nested_trig_out = try_limit_rules_at_finite(&mut ctx, nested_trig, x, point_zero)
            .expect("expected nested total-real trig finite limit");
        let Expr::Number(value) = ctx.get(nested_trig_out) else {
            panic!("expected cos(sin(x)) at 0 to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer(1.into()));

        let point_neg_two = parse_expr(&mut ctx, "-2");
        let sin_sqrt = parse_expr(&mut ctx, "sin(sqrt(x^2 + 1))");
        let sin_sqrt_out = try_limit_rules_at_finite(&mut ctx, sin_sqrt, x, point_neg_two)
            .expect("expected total-real unary composition over safe sqrt sublimit");
        assert_eq!(display_expr(&ctx, sin_sqrt_out), "sin(sqrt(5))");

        let sin_special_angle = parse_expr(&mut ctx, "sin(x + pi/6)");
        let sin_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, sin_special_angle, x, point_zero)
                .expect("expected sin over exact special-angle sublimit");
        assert_eq!(display_expr(&ctx, sin_special_angle_out), "1 / 2");

        let cos_special_angle = parse_expr(&mut ctx, "cos(x + pi/3)");
        let cos_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, cos_special_angle, x, point_zero)
                .expect("expected cos over exact special-angle sublimit");
        assert_eq!(display_expr(&ctx, cos_special_angle_out), "1 / 2");

        let arctan_special_input = parse_expr(&mut ctx, "arctan(x + 1)");
        let arctan_special_input_out =
            try_limit_rules_at_finite(&mut ctx, arctan_special_input, x, point_zero)
                .expect("expected arctan over exact table input sublimit");
        assert_eq!(display_expr(&ctx, arctan_special_input_out), "pi / 4");

        let arctan_sqrt_special_input = parse_expr(&mut ctx, "arctan(sqrt(x + 3))");
        let arctan_sqrt_special_input_out =
            try_limit_rules_at_finite(&mut ctx, arctan_sqrt_special_input, x, point_zero)
                .expect("expected arctan over exact radical table input sublimit");
        assert_eq!(display_expr(&ctx, arctan_sqrt_special_input_out), "pi / 3");

        let exp_abs = parse_expr(&mut ctx, "exp(abs(x))");
        let exp_abs_out = try_limit_rules_at_finite(&mut ctx, exp_abs, x, point_neg_two)
            .expect("expected exp over resolved abs sublimit");
        let Expr::Function(fn_id, args) = ctx.get(exp_abs_out).clone() else {
            panic!("expected exp(abs(x)) finite limit to remain an exp function");
        };
        assert_eq!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp));
        assert_eq!(args.len(), 1);
        let Expr::Number(value) = ctx.get(args[0]) else {
            panic!("expected exp argument to be exact numeric absolute value");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));

        let point_eight = parse_expr(&mut ctx, "8");
        let sin_cbrt = parse_expr(&mut ctx, "sin(cbrt(x))");
        let sin_cbrt_out = try_limit_rules_at_finite(&mut ctx, sin_cbrt, x, point_eight)
            .expect("expected total-real unary composition over exact cbrt sublimit");
        assert_eq!(display_expr(&ctx, sin_cbrt_out), "sin(2)");
    }

    #[test]
    fn finite_arithmetic_composition_folds_safe_numeric_and_structural_results() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let numeric_sum = parse_expr(&mut ctx, "abs(x) + 1");
        let numeric_sum_out = try_limit_rules_at_finite(&mut ctx, numeric_sum, x, point_neg_two)
            .expect("expected safe numeric finite sum");
        let Expr::Number(value) = ctx.get(numeric_sum_out) else {
            panic!("expected exact numeric finite sum");
        };
        assert_eq!(value, &BigRational::from_integer(3.into()));

        let structural_zero = parse_expr(&mut ctx, "sqrt(x^2 + 1) - sqrt(x^2 + 1)");
        let structural_zero_out =
            try_limit_rules_at_finite(&mut ctx, structural_zero, x, point_neg_two)
                .expect("expected safe structural zero finite difference");
        let Expr::Number(value) = ctx.get(structural_zero_out) else {
            panic!("expected structural zero finite difference to fold");
        };
        assert_eq!(value, &BigRational::zero());

        let zero_quotient = parse_expr(&mut ctx, "(sqrt(x^2 + 1) - sqrt(x^2 + 1))/(abs(x) + 1)");
        let zero_quotient_out =
            try_limit_rules_at_finite(&mut ctx, zero_quotient, x, point_neg_two)
                .expect("expected safe zero quotient finite limit");
        let Expr::Number(value) = ctx.get(zero_quotient_out) else {
            panic!("expected safe zero quotient to fold");
        };
        assert_eq!(value, &BigRational::zero());

        let symbolic_sum = parse_expr(&mut ctx, "sqrt(x^2 + 1) + ln(x + 5)");
        let symbolic_sum_out = try_limit_rules_at_finite(&mut ctx, symbolic_sum, x, point_neg_two)
            .expect("expected safe symbolic finite sum");
        assert_eq!(display_expr(&ctx, symbolic_sum_out), "ln(3) + sqrt(5)");

        let unsafe_zero_product = parse_expr(&mut ctx, "0 * sqrt(x)");
        let point_zero = parse_expr(&mut ctx, "0");
        assert!(
            try_limit_rules_at_finite(&mut ctx, unsafe_zero_product, x, point_zero).is_none(),
            "zero product must not hide an unresolved finite sublimit"
        );
    }

    #[test]
    fn finite_positive_domain_unary_composition_requires_positive_sublimit() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let ln_sqrt = parse_expr(&mut ctx, "ln(sqrt(x^2 + 1))");
        let ln_sqrt_out = try_limit_rules_at_finite(&mut ctx, ln_sqrt, x, point_neg_two)
            .expect("expected ln over proven-positive sqrt sublimit");
        assert_eq!(display_expr(&ctx, ln_sqrt_out), "ln(sqrt(5))");

        let sqrt_abs_shift = parse_expr(&mut ctx, "sqrt(abs(x) + 1)");
        let sqrt_abs_shift_out =
            try_limit_rules_at_finite(&mut ctx, sqrt_abs_shift, x, point_neg_two)
                .expect("expected sqrt over positive arithmetic sublimit");
        assert_eq!(display_expr(&ctx, sqrt_abs_shift_out), "sqrt(3)");

        let ln_abs = parse_expr(&mut ctx, "ln(abs(x))");
        let ln_abs_out = try_limit_rules_at_finite(&mut ctx, ln_abs, x, point_neg_two)
            .expect("expected ln over positive abs sublimit");
        assert_eq!(display_expr(&ctx, ln_abs_out), "ln(2)");

        let log2_poly = parse_expr(&mut ctx, "log2(x^2 + 1)");
        let log2_poly_out = try_limit_rules_at_finite(&mut ctx, log2_poly, x, point_neg_two)
            .expect("expected log2 over positive polynomial argument");
        assert_eq!(display_expr(&ctx, log2_poly_out), "log2(5)");

        let log10_sqrt = parse_expr(&mut ctx, "log10(sqrt(x^2 + 1))");
        let log10_sqrt_out = try_limit_rules_at_finite(&mut ctx, log10_sqrt, x, point_neg_two)
            .expect("expected log10 over proven-positive sqrt sublimit");
        assert_eq!(display_expr(&ctx, log10_sqrt_out), "log10(sqrt(5))");

        let log2_abs = parse_expr(&mut ctx, "log2(abs(x))");
        let log2_abs_out = try_limit_rules_at_finite(&mut ctx, log2_abs, x, point_neg_two)
            .expect("expected log2 over positive abs sublimit");
        assert_eq!(display_expr(&ctx, log2_abs_out), "1");

        let point_zero = parse_expr(&mut ctx, "0");
        let sqrt_perfect_square_poly = parse_expr(&mut ctx, "sqrt(x^2 + 4*x + 4)");
        let sqrt_perfect_square_poly_out =
            try_limit_rules_at_finite(&mut ctx, sqrt_perfect_square_poly, x, point_zero)
                .expect("expected exact sqrt over positive rational square sublimit");
        assert_eq!(display_expr(&ctx, sqrt_perfect_square_poly_out), "2");

        let ln_one = parse_expr(&mut ctx, "ln(x^2 + 1)");
        let ln_one_out = try_limit_rules_at_finite(&mut ctx, ln_one, x, point_zero)
            .expect("expected exact ln(1) finite limit");
        assert_eq!(display_expr(&ctx, ln_one_out), "0");

        let point_e = parse_expr(&mut ctx, "e");
        let ln_e = parse_expr(&mut ctx, "ln(x)");
        let ln_e_out = try_limit_rules_at_finite(&mut ctx, ln_e, x, point_e)
            .expect("expected exact ln(e) finite limit");
        assert_eq!(display_expr(&ctx, ln_e_out), "1");

        let log2_one = parse_expr(&mut ctx, "log2(x^2 + 1)");
        let log2_one_out = try_limit_rules_at_finite(&mut ctx, log2_one, x, point_zero)
            .expect("expected exact log2(1) finite limit");
        assert_eq!(display_expr(&ctx, log2_one_out), "0");

        let log10_one = parse_expr(&mut ctx, "log10(x^2 + 1)");
        let log10_one_out = try_limit_rules_at_finite(&mut ctx, log10_one, x, point_zero)
            .expect("expected exact log10(1) finite limit");
        assert_eq!(display_expr(&ctx, log10_one_out), "0");

        let point_two = parse_expr(&mut ctx, "2");
        let log2_exact_power = parse_expr(&mut ctx, "log2(x^2 + 4)");
        let log2_exact_power_out =
            try_limit_rules_at_finite(&mut ctx, log2_exact_power, x, point_two)
                .expect("expected exact integer log2 finite limit");
        assert_eq!(display_expr(&ctx, log2_exact_power_out), "3");

        let log10_exact_power = parse_expr(&mut ctx, "log10(x^2 + 96)");
        let log10_exact_power_out =
            try_limit_rules_at_finite(&mut ctx, log10_exact_power, x, point_two)
                .expect("expected exact integer log10 finite limit");
        assert_eq!(display_expr(&ctx, log10_exact_power_out), "2");

        let exp_ln_abs = parse_expr(&mut ctx, "exp(ln(abs(x)))");
        let exp_ln_abs_out = try_limit_rules_at_finite(&mut ctx, exp_ln_abs, x, point_neg_two)
            .expect("expected exact exp(ln(g)) finite limit when g is positive");
        assert_eq!(display_expr(&ctx, exp_ln_abs_out), "2");

        let ln_exp_abs = parse_expr(&mut ctx, "ln(exp(abs(x)))");
        let ln_exp_abs_out = try_limit_rules_at_finite(&mut ctx, ln_exp_abs, x, point_neg_two)
            .expect("expected exact ln(exp(g)) finite limit");
        assert_eq!(display_expr(&ctx, ln_exp_abs_out), "2");

        let abs_sqrt = parse_expr(&mut ctx, "abs(sqrt(x^2 + 1))");
        let abs_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_sqrt, x, point_neg_two)
            .expect("expected exact abs over positive sqrt finite limit");
        assert_eq!(display_expr(&ctx, abs_sqrt_out), "sqrt(5)");

        let abs_neg_sqrt = parse_expr(&mut ctx, "abs(-sqrt(x^2 + 1))");
        let abs_neg_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_neg_sqrt, x, point_neg_two)
            .expect("expected exact abs over negative positive-sqrt finite limit");
        assert_eq!(display_expr(&ctx, abs_neg_sqrt_out), "sqrt(5)");

        let exp_ln_abs_zero = parse_expr(&mut ctx, "exp(ln(abs(x)))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, exp_ln_abs_zero, x, point_zero).is_none(),
            "exp(ln(abs(x))) at zero must remain residual"
        );

        let sqrt_abs_zero = parse_expr(&mut ctx, "sqrt(abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sqrt_abs_zero, x, point_zero).is_none(),
            "sqrt over zero sublimit must remain residual"
        );

        let ln_sin_zero = parse_expr(&mut ctx, "ln(sin(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, ln_sin_zero, x, point_zero).is_none(),
            "ln over zero sublimit must remain residual"
        );

        let log10_abs_zero = parse_expr(&mut ctx, "log10(abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log10_abs_zero, x, point_zero).is_none(),
            "log10 over zero sublimit must remain residual"
        );
    }

    #[test]
    fn finite_partial_domain_unary_composition_requires_strict_interior_sublimit() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let arcsin_half = parse_expr(&mut ctx, "arcsin(x/2)");
        let arcsin_half_out = try_limit_rules_at_finite(&mut ctx, arcsin_half, x, point_zero)
            .expect("expected arcsin over strict interior numeric sublimit");
        assert_eq!(display_expr(&ctx, arcsin_half_out), "0");

        let atanh_half = parse_expr(&mut ctx, "atanh(x/2)");
        let atanh_half_out = try_limit_rules_at_finite(&mut ctx, atanh_half, x, point_zero)
            .expect("expected atanh over strict interior numeric sublimit");
        assert_eq!(display_expr(&ctx, atanh_half_out), "0");

        let acos_half = parse_expr(&mut ctx, "acos(x/2)");
        let acos_half_out = try_limit_rules_at_finite(&mut ctx, acos_half, x, point_zero)
            .expect("expected acos over strict interior numeric sublimit");
        assert_eq!(display_expr(&ctx, acos_half_out), "pi / 2");

        let arcsin_shifted_half = parse_expr(&mut ctx, "arcsin(x/2 + 1/2)");
        let arcsin_shifted_half_out =
            try_limit_rules_at_finite(&mut ctx, arcsin_shifted_half, x, point_zero)
                .expect("expected arcsin exact-table hit over strict interior sublimit");
        assert_eq!(display_expr(&ctx, arcsin_shifted_half_out), "pi / 6");

        let arccos_shifted_half = parse_expr(&mut ctx, "arccos(x/2 + 1/2)");
        let arccos_shifted_half_out =
            try_limit_rules_at_finite(&mut ctx, arccos_shifted_half, x, point_zero)
                .expect("expected arccos exact-table hit over strict interior sublimit");
        assert_eq!(display_expr(&ctx, arccos_shifted_half_out), "pi / 3");

        let acosh_abs_shift = parse_expr(&mut ctx, "acosh(abs(x) + 2)");
        let acosh_abs_shift_out =
            try_limit_rules_at_finite(&mut ctx, acosh_abs_shift, x, point_zero)
                .expect("expected acosh over strict interior numeric sublimit");
        assert_eq!(display_expr(&ctx, acosh_abs_shift_out), "acosh(2)");

        let point_one = parse_expr(&mut ctx, "1");
        let acosh_sqrt_affine = parse_expr(&mut ctx, "acosh(sqrt(2*x + 3))");
        let acosh_sqrt_affine_out =
            try_limit_rules_at_finite(&mut ctx, acosh_sqrt_affine, x, point_one)
                .expect("expected acosh over strict interior square-root sublimit");
        assert_eq!(display_expr(&ctx, acosh_sqrt_affine_out), "acosh(sqrt(5))");

        let acosh_sqrt_endpoint = parse_expr(&mut ctx, "acosh(sqrt(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acosh_sqrt_endpoint, x, point_one).is_none(),
            "acosh sqrt endpoint must remain residual"
        );

        let point_neg_five_four = parse_expr(&mut ctx, "-5/4");
        let atanh_sqrt_non_square = parse_expr(&mut ctx, "atanh(sqrt(2*x + 3))");
        let atanh_sqrt_non_square_out =
            try_limit_rules_at_finite(&mut ctx, atanh_sqrt_non_square, x, point_neg_five_four)
                .expect("expected atanh over strict interior square-root sublimit");
        assert_eq!(
            display_expr(&ctx, atanh_sqrt_non_square_out),
            "atanh(sqrt(1/2))"
        );

        let atanh_neg_sqrt_non_square = parse_expr(&mut ctx, "atanh(-sqrt(2*x + 3))");
        let atanh_neg_sqrt_non_square_out =
            try_limit_rules_at_finite(&mut ctx, atanh_neg_sqrt_non_square, x, point_neg_five_four)
                .expect("expected atanh over negated strict interior square-root sublimit");
        assert_eq!(
            display_expr(&ctx, atanh_neg_sqrt_non_square_out),
            "atanh(-sqrt(1/2))"
        );

        let arcsin_sqrt_non_square = parse_expr(&mut ctx, "arcsin(sqrt(2*x + 3))");
        let arcsin_sqrt_non_square_out =
            try_limit_rules_at_finite(&mut ctx, arcsin_sqrt_non_square, x, point_neg_five_four)
                .expect("expected arcsin over strict interior square-root sublimit");
        assert_eq!(display_expr(&ctx, arcsin_sqrt_non_square_out), "pi / 4");

        let acos_sqrt_non_square = parse_expr(&mut ctx, "acos(sqrt(2*x + 3))");
        let acos_sqrt_non_square_out =
            try_limit_rules_at_finite(&mut ctx, acos_sqrt_non_square, x, point_neg_five_four)
                .expect("expected acos over strict interior square-root sublimit");
        assert_eq!(display_expr(&ctx, acos_sqrt_non_square_out), "pi / 4");

        let atanh_sqrt_endpoint = parse_expr(&mut ctx, "atanh(sqrt(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, atanh_sqrt_endpoint, x, point_one).is_none(),
            "atanh sqrt endpoint must remain residual"
        );

        let point_two = parse_expr(&mut ctx, "2");
        let arcsin_endpoint = parse_expr(&mut ctx, "arcsin(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, arcsin_endpoint, x, point_one).is_none(),
            "arcsin endpoint must remain residual"
        );
        let acos_endpoint = parse_expr(&mut ctx, "acos(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acos_endpoint, x, point_one).is_none(),
            "acos one-sided-only endpoint must remain residual for bilateral limits"
        );
        let acos_even_gap_endpoint = parse_expr(&mut ctx, "acos(1 - x^2)");
        let acos_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acos_even_gap_endpoint, x, point_zero)
                .expect("expected bilateral inverse-trig upper endpoint");
        assert_eq!(display_expr(&ctx, acos_even_gap_endpoint_out), "0");

        let arcsin_even_gap_endpoint = parse_expr(&mut ctx, "arcsin(1 - x^2)");
        let arcsin_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, arcsin_even_gap_endpoint, x, point_zero)
                .expect("expected bilateral arcsin upper endpoint");
        assert_eq!(display_expr(&ctx, arcsin_even_gap_endpoint_out), "pi / 2");

        let acos_shifted_even_gap_endpoint = parse_expr(&mut ctx, "acos(1 - (x - 2)^2)");
        let acos_shifted_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acos_shifted_even_gap_endpoint, x, point_two)
                .expect("expected shifted bilateral inverse-trig upper endpoint");
        assert_eq!(display_expr(&ctx, acos_shifted_even_gap_endpoint_out), "0");

        let acos_odd_gap_endpoint = parse_expr(&mut ctx, "acos(1 - x^3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acos_odd_gap_endpoint, x, point_zero).is_none(),
            "one-sided-only inverse-trig upper endpoint must remain residual"
        );

        let acos_above_domain_endpoint = parse_expr(&mut ctx, "acos(1 + x^2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acos_above_domain_endpoint, x, point_zero)
                .is_none(),
            "empty-punctured-domain inverse-trig upper endpoint must remain residual"
        );

        let acos_lower_even_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + x^2)");
        let acos_lower_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acos_lower_even_gap_endpoint, x, point_zero)
                .expect("expected bilateral inverse-trig lower endpoint");
        assert_eq!(display_expr(&ctx, acos_lower_even_gap_endpoint_out), "pi");

        let arcsin_lower_even_gap_endpoint = parse_expr(&mut ctx, "arcsin(-1 + x^2)");
        let arcsin_lower_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, arcsin_lower_even_gap_endpoint, x, point_zero)
                .expect("expected bilateral arcsin lower endpoint");
        assert_eq!(
            display_expr(&ctx, arcsin_lower_even_gap_endpoint_out),
            "-pi / 2"
        );

        let acos_shifted_lower_even_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + (x - 2)^2)");
        let acos_shifted_lower_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acos_shifted_lower_even_gap_endpoint, x, point_two)
                .expect("expected shifted bilateral inverse-trig lower endpoint");
        assert_eq!(
            display_expr(&ctx, acos_shifted_lower_even_gap_endpoint_out),
            "pi"
        );

        let acos_lower_odd_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + x^3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acos_lower_odd_gap_endpoint, x, point_zero)
                .is_none(),
            "one-sided-only inverse-trig lower endpoint must remain residual"
        );

        let acos_below_domain_endpoint = parse_expr(&mut ctx, "acos(-1 - x^2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acos_below_domain_endpoint, x, point_zero)
                .is_none(),
            "empty-punctured-domain inverse-trig lower endpoint must remain residual"
        );
        let arcsin_out_of_domain = parse_expr(&mut ctx, "arcsin(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, arcsin_out_of_domain, x, point_two).is_none(),
            "arcsin out-of-domain point must remain residual"
        );
        let atanh_endpoint = parse_expr(&mut ctx, "atanh(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, atanh_endpoint, x, point_one).is_none(),
            "atanh endpoint must remain residual"
        );
        let acosh_endpoint = parse_expr(&mut ctx, "acosh(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acosh_endpoint, x, point_one).is_none(),
            "acosh endpoint must remain residual"
        );

        let acosh_even_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + x^2)");
        let acosh_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acosh_even_gap_endpoint, x, point_zero)
                .expect("expected bilateral acosh lower-bound endpoint");
        assert_eq!(display_expr(&ctx, acosh_even_gap_endpoint_out), "0");

        let acosh_shifted_even_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + (x - 2)^2)");
        let acosh_shifted_even_gap_endpoint_out =
            try_limit_rules_at_finite(&mut ctx, acosh_shifted_even_gap_endpoint, x, point_two)
                .expect("expected shifted bilateral acosh lower-bound endpoint");
        assert_eq!(display_expr(&ctx, acosh_shifted_even_gap_endpoint_out), "0");

        let acosh_odd_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + x^3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acosh_odd_gap_endpoint, x, point_zero).is_none(),
            "one-sided-only acosh polynomial endpoint must remain residual"
        );

        let acosh_negative_gap_endpoint = parse_expr(&mut ctx, "acosh(1 - x^2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, acosh_negative_gap_endpoint, x, point_zero)
                .is_none(),
            "empty-punctured-domain acosh endpoint must remain residual"
        );
    }

    #[test]
    fn finite_bilateral_sqrt_endpoint_requires_positive_tail_on_both_sides() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let even_gap = parse_expr(&mut ctx, "sqrt(x^2)");
        let even_gap_out = try_limit_rules_at_finite(&mut ctx, even_gap, x, point_zero)
            .expect("expected bilateral sqrt endpoint over positive even gap");
        assert_eq!(display_expr(&ctx, even_gap_out), "0");

        let point_one = parse_expr(&mut ctx, "1");
        let shifted_even_gap = parse_expr(&mut ctx, "sqrt((x - 1)^2)");
        let shifted_even_gap_out =
            try_limit_rules_at_finite(&mut ctx, shifted_even_gap, x, point_one)
                .expect("expected shifted bilateral sqrt endpoint over positive even gap");
        assert_eq!(display_expr(&ctx, shifted_even_gap_out), "0");

        let one_sided_only = parse_expr(&mut ctx, "sqrt(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, one_sided_only, x, point_zero).is_none(),
            "sqrt(x) at a finite endpoint must remain residual for bilateral limits"
        );

        let odd_gap = parse_expr(&mut ctx, "sqrt(x^3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, odd_gap, x, point_zero).is_none(),
            "sqrt over an odd local tail must remain residual for bilateral limits"
        );

        let log_even_gap = parse_expr(&mut ctx, "ln(x^2)");
        let log_even_gap_out = try_limit_rules_at_finite(&mut ctx, log_even_gap, x, point_zero)
            .expect("expected bilateral log endpoint over positive even gap");
        assert_eq!(display_expr(&ctx, log_even_gap_out), "-infinity");

        let reciprocal_base_log_even_gap = parse_expr(&mut ctx, "log(1/2, x^2)");
        let reciprocal_base_log_even_gap_out =
            try_limit_rules_at_finite(&mut ctx, reciprocal_base_log_even_gap, x, point_zero)
                .expect("expected reciprocal-base bilateral log endpoint over positive even gap");
        assert_eq!(
            display_expr(&ctx, reciprocal_base_log_even_gap_out),
            "infinity"
        );

        let log_one_sided_only = parse_expr(&mut ctx, "ln(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_one_sided_only, x, point_zero).is_none(),
            "ln(x) at a finite endpoint must remain residual for bilateral limits"
        );

        let log_empty_punctured = parse_expr(&mut ctx, "ln(-x^2)");
        let log_empty_punctured_out =
            try_limit_rules_at_finite(&mut ctx, log_empty_punctured, x, point_zero)
                .expect("log over an empty real domain should be undefined");
        assert_eq!(
            display_expr(&ctx, log_empty_punctured_out),
            "undefined",
            "log over an empty real domain must be undefined"
        );
    }

    #[test]
    fn finite_residual_warning_marks_empty_punctured_domains() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let empty_punctured = parse_expr(&mut ctx, "sqrt(-x^2)");
        let outcome = eval_limit_at_infinity(
            &mut ctx,
            empty_punctured,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        assert_eq!(
            display_expr(&ctx, outcome.expr),
            "limit(sqrt(-(x^2)), x, 0)"
        );
        let warning = outcome.warning.expect("expected residual warning");
        assert!(warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(warning.contains("no punctured real neighbourhood"));

        let empty_punctured_acosh = parse_expr(&mut ctx, "acosh(1 - x^2)");
        let acosh_outcome = eval_limit_at_infinity(
            &mut ctx,
            empty_punctured_acosh,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        assert_eq!(
            display_expr(&ctx, acosh_outcome.expr),
            "limit(acosh(1 - x^2), x, 0)"
        );
        let acosh_warning = acosh_outcome.warning.expect("expected residual warning");
        assert!(acosh_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(acosh_warning.contains("no punctured real neighbourhood"));

        let empty_punctured_inverse_trig = parse_expr(&mut ctx, "acos(1 + x^2)");
        let inverse_trig_outcome = eval_limit_at_infinity(
            &mut ctx,
            empty_punctured_inverse_trig,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        assert_eq!(
            display_expr(&ctx, inverse_trig_outcome.expr),
            "limit(acos(x^2 + 1), x, 0)"
        );
        let inverse_trig_warning = inverse_trig_outcome
            .warning
            .expect("expected residual warning");
        assert!(inverse_trig_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(inverse_trig_warning.contains("no punctured real neighbourhood"));

        let empty_punctured_inverse_trig_lower = parse_expr(&mut ctx, "acos(-1 - x^2)");
        let inverse_trig_lower_outcome = eval_limit_at_infinity(
            &mut ctx,
            empty_punctured_inverse_trig_lower,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        let inverse_trig_lower_warning = inverse_trig_lower_outcome
            .warning
            .expect("expected residual warning");
        assert!(inverse_trig_lower_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(inverse_trig_lower_warning.contains("no punctured real neighbourhood"));

        let one_sided_only = parse_expr(&mut ctx, "sqrt(x^3)");
        let one_sided_outcome = eval_limit_at_infinity(
            &mut ctx,
            one_sided_only,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        let one_sided_warning = one_sided_outcome
            .warning
            .expect("expected generic residual warning");
        assert!(one_sided_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(!one_sided_warning.contains("no punctured real neighbourhood"));

        let one_sided_only_acosh = parse_expr(&mut ctx, "acosh(1 + x^3)");
        let one_sided_acosh_outcome = eval_limit_at_infinity(
            &mut ctx,
            one_sided_only_acosh,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        let one_sided_acosh_warning = one_sided_acosh_outcome
            .warning
            .expect("expected generic residual warning");
        assert!(one_sided_acosh_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(!one_sided_acosh_warning.contains("no punctured real neighbourhood"));

        let one_sided_only_inverse_trig = parse_expr(&mut ctx, "acos(1 - x^3)");
        let one_sided_inverse_trig_outcome = eval_limit_at_infinity(
            &mut ctx,
            one_sided_only_inverse_trig,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        let one_sided_inverse_trig_warning = one_sided_inverse_trig_outcome
            .warning
            .expect("expected generic residual warning");
        assert!(one_sided_inverse_trig_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
        assert!(!one_sided_inverse_trig_warning.contains("no punctured real neighbourhood"));

        let positive_even_gap = parse_expr(&mut ctx, "sqrt(x^2)");
        let resolved_outcome = eval_limit_at_infinity(
            &mut ctx,
            positive_even_gap,
            x,
            Approach::Finite(point_zero),
            &LimitOptions::default(),
        );
        assert_eq!(display_expr(&ctx, resolved_outcome.expr), "0");
        assert!(resolved_outcome.warning.is_none());
    }

    #[test]
    fn finite_domain_checked_trig_unary_composition_accepts_defined_table_sublimits() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_zero = parse_expr(&mut ctx, "0");

        let tan_half = parse_expr(&mut ctx, "tan(x/2)");
        let tan_half_out = try_limit_rules_at_finite(&mut ctx, tan_half, x, point_zero)
            .expect("expected tan over exact zero sublimit");
        assert_eq!(display_expr(&ctx, tan_half_out), "0");

        let sec_square = parse_expr(&mut ctx, "sec(x^2)");
        let sec_square_out = try_limit_rules_at_finite(&mut ctx, sec_square, x, point_zero)
            .expect("expected sec over exact zero sublimit");
        assert_eq!(display_expr(&ctx, sec_square_out), "1");

        let tan_sin = parse_expr(&mut ctx, "tan(sin(x))");
        let tan_sin_out = try_limit_rules_at_finite(&mut ctx, tan_sin, x, point_zero)
            .expect("expected tan over resolved zero sin sublimit");
        assert_eq!(display_expr(&ctx, tan_sin_out), "0");

        let sec_abs = parse_expr(&mut ctx, "sec(abs(x))");
        let sec_abs_out = try_limit_rules_at_finite(&mut ctx, sec_abs, x, point_zero)
            .expect("expected sec over resolved zero abs sublimit");
        assert_eq!(display_expr(&ctx, sec_abs_out), "1");

        let tan_special_angle = parse_expr(&mut ctx, "tan(x + pi/4)");
        let tan_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, tan_special_angle, x, point_zero)
                .expect("expected tan over defined special-angle sublimit");
        assert_eq!(display_expr(&ctx, tan_special_angle_out), "1");

        let sec_special_angle = parse_expr(&mut ctx, "sec(x + pi/3)");
        let sec_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, sec_special_angle, x, point_zero)
                .expect("expected sec over defined special-angle sublimit");
        assert_eq!(display_expr(&ctx, sec_special_angle_out), "2");

        let csc_special_angle = parse_expr(&mut ctx, "csc(x + pi/6)");
        let csc_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, csc_special_angle, x, point_zero)
                .expect("expected csc over defined special-angle sublimit");
        assert_eq!(display_expr(&ctx, csc_special_angle_out), "2");

        let cot_special_angle = parse_expr(&mut ctx, "cot(x + pi/4)");
        let cot_special_angle_out =
            try_limit_rules_at_finite(&mut ctx, cot_special_angle, x, point_zero)
                .expect("expected cot over defined special-angle sublimit");
        assert_eq!(display_expr(&ctx, cot_special_angle_out), "1");

        let tan_pole_angle = parse_expr(&mut ctx, "tan(x + pi/2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, tan_pole_angle, x, point_zero).is_none(),
            "tan at a table-undefined pole must remain residual"
        );

        let sec_pole_angle = parse_expr(&mut ctx, "sec(x + pi/2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sec_pole_angle, x, point_zero).is_none(),
            "sec at a table-undefined pole must remain residual"
        );

        let csc_pole_angle = parse_expr(&mut ctx, "csc(x + pi)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, csc_pole_angle, x, point_zero).is_none(),
            "csc at a table-undefined pole must remain residual"
        );

        let cot_pole_angle = parse_expr(&mut ctx, "cot(x + pi)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, cot_pole_angle, x, point_zero).is_none(),
            "cot at a table-undefined pole must remain residual"
        );

        let point_one = parse_expr(&mut ctx, "1");
        let tan_nonzero = parse_expr(&mut ctx, "tan(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, tan_nonzero, x, point_one).is_none(),
            "tan at nonzero rational sublimit must remain residual without pole proof"
        );
        let sec_nonzero = parse_expr(&mut ctx, "sec(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sec_nonzero, x, point_one).is_none(),
            "sec at nonzero rational sublimit must remain residual without pole proof"
        );
        let csc_zero = parse_expr(&mut ctx, "csc(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, csc_zero, x, point_zero).is_none(),
            "csc at zero must remain residual"
        );
        let cot_zero = parse_expr(&mut ctx, "cot(x)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, cot_zero, x, point_zero).is_none(),
            "cot at zero must remain residual"
        );
    }

    #[test]
    fn finite_binary_log_composition_requires_valid_base_and_positive_sublimits() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let binary_log_poly = parse_expr(&mut ctx, "log(2, x^2 + 1)");
        let binary_log_poly_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_poly, x, point_neg_two)
                .expect("expected constant-base log over positive polynomial argument");
        assert_eq!(display_expr(&ctx, binary_log_poly_out), "log(2, 5)");

        let binary_log_sqrt = parse_expr(&mut ctx, "log(1/2, sqrt(x^2 + 1))");
        let binary_log_sqrt_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_sqrt, x, point_neg_two)
                .expect("expected constant-base log over proven-positive sqrt sublimit");
        assert_eq!(
            display_expr(&ctx, binary_log_sqrt_out),
            "log(1 / 2, sqrt(5))"
        );

        let binary_log_abs = parse_expr(&mut ctx, "log(2, abs(x))");
        let binary_log_abs_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_abs, x, point_neg_two)
                .expect("expected constant-base log over positive abs sublimit");
        assert_eq!(display_expr(&ctx, binary_log_abs_out), "1");

        let point_zero = parse_expr(&mut ctx, "0");
        let binary_log_arg_one = parse_expr(&mut ctx, "log(2, x^2 + 1)");
        let binary_log_arg_one_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_arg_one, x, point_zero)
                .expect("expected exact binary log of one finite limit");
        assert_eq!(display_expr(&ctx, binary_log_arg_one_out), "0");

        let point_two = parse_expr(&mut ctx, "2");
        let binary_log_integer_power = parse_expr(&mut ctx, "log(2, x^2 + 4)");
        let binary_log_integer_power_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_integer_power, x, point_two)
                .expect("expected exact integer binary log finite limit");
        assert_eq!(display_expr(&ctx, binary_log_integer_power_out), "3");

        let binary_log_negative_integer_power = parse_expr(&mut ctx, "log(1/2, x^2 + 4)");
        let binary_log_negative_integer_power_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_negative_integer_power, x, point_two)
                .expect("expected exact negative-integer binary log finite limit");
        assert_eq!(
            display_expr(&ctx, binary_log_negative_integer_power_out),
            "-3"
        );

        let binary_log_fractional_power = parse_expr(&mut ctx, "log(4, x^2 + 4)");
        let binary_log_fractional_power_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_fractional_power, x, point_two)
                .expect("expected exact rational-exponent binary log finite limit");
        assert_eq!(display_expr(&ctx, binary_log_fractional_power_out), "3/2");

        let binary_log_negative_fractional_power = parse_expr(&mut ctx, "log(1/4, x^2 + 4)");
        let binary_log_negative_fractional_power_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_negative_fractional_power, x, point_two)
                .expect("expected exact negative rational-exponent binary log finite limit");
        assert_eq!(
            display_expr(&ctx, binary_log_negative_fractional_power_out),
            "-3/2"
        );

        let binary_log_two_thirds = parse_expr(&mut ctx, "log(27, x^2 + 5)");
        let binary_log_two_thirds_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_two_thirds, x, point_two)
                .expect("expected exact two-thirds binary log finite limit");
        assert_eq!(display_expr(&ctx, binary_log_two_thirds_out), "2/3");

        let binary_log_not_exact = parse_expr(&mut ctx, "log(2, x^2 + 1)");
        let binary_log_not_exact_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_not_exact, x, point_two)
                .expect("expected safe binary log finite limit without exact rational fold");
        assert_eq!(display_expr(&ctx, binary_log_not_exact_out), "log(2, 5)");

        let variable_base_log_poly = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 1)");
        let variable_base_log_poly_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_poly, x, point_neg_two)
                .expect("expected log over safe finite base and argument sublimits");
        assert_eq!(display_expr(&ctx, variable_base_log_poly_out), "log(7, 5)");

        let variable_base_log_sqrt = parse_expr(&mut ctx, "log(x^2 + 3, sqrt(x^2 + 1))");
        let variable_base_log_sqrt_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_sqrt, x, point_neg_two)
                .expect("expected log over safe finite base and positive sqrt argument sublimit");
        assert_eq!(
            display_expr(&ctx, variable_base_log_sqrt_out),
            "log(7, sqrt(5))"
        );

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let variable_base_log_same = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 3)");
        let variable_base_log_same_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_same, x, point_neg_one)
                .expect("expected exact binary log with equal finite base and argument");
        assert_eq!(display_expr(&ctx, variable_base_log_same_out), "1");

        let binary_log_abs_zero = parse_expr(&mut ctx, "log(2, abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, binary_log_abs_zero, x, point_zero).is_none(),
            "constant-base log over zero sublimit must remain residual"
        );

        let log_base_one = parse_expr(&mut ctx, "log(1, x^2 + 1)");
        let log_base_one_out = try_limit_rules_at_finite(&mut ctx, log_base_one, x, point_neg_two)
            .expect("constant-base log with base one has empty real domain");
        assert_eq!(display_expr(&ctx, log_base_one_out), "undefined");

        let log_negative_base = parse_expr(&mut ctx, "log(-2, x^2 + 1)");
        let log_negative_base_out =
            try_limit_rules_at_finite(&mut ctx, log_negative_base, x, point_neg_two)
                .expect("constant-base log with negative base has empty real domain");
        assert_eq!(display_expr(&ctx, log_negative_base_out), "undefined");

        let log_variable_base_one = parse_expr(&mut ctx, "log(x^2 - 3, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_variable_base_one, x, point_neg_two).is_none(),
            "variable-base log with base sublimit one must remain residual"
        );

        let log_variable_base_zero = parse_expr(&mut ctx, "log(x^2 - 4, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_variable_base_zero, x, point_neg_two).is_none(),
            "variable-base log with zero base sublimit must remain residual"
        );
    }

    #[test]
    fn finite_integer_power_composition_requires_safe_sublimit_and_nonzero_base_when_needed() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let numeric_positive_power = parse_expr(&mut ctx, "(abs(x) + 1)^2");
        let numeric_positive_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_positive_power, x, point_neg_two)
                .expect("expected integer power over exact numeric sublimit");
        let Expr::Number(value) = ctx.get(numeric_positive_power_out) else {
            panic!("expected exact integer power to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(9.into()));

        let symbolic_positive_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^2");
        let symbolic_positive_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_positive_power, x, point_neg_two)
                .expect("expected integer power over safe symbolic sublimit");
        let Expr::Number(value) = ctx.get(symbolic_positive_power_out) else {
            panic!("expected even power over exact sqrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(5.into()));

        let symbolic_odd_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^3");
        let symbolic_odd_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_odd_power, x, point_neg_two)
                .expect("expected odd integer power over safe symbolic sublimit");
        assert_eq!(display_expr(&ctx, symbolic_odd_power_out), "sqrt(5)^3");

        let numeric_negative_power = parse_expr(&mut ctx, "(abs(x) + 1)^(-2)");
        let numeric_negative_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_negative_power, x, point_neg_two)
                .expect("expected negative integer power over nonzero numeric sublimit");
        let Expr::Number(value) = ctx.get(numeric_negative_power_out) else {
            panic!("expected exact negative integer power to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(9)));

        let symbolic_negative_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-1)");
        let symbolic_negative_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_negative_power, x, point_neg_two)
                .expect("expected negative integer power over proven nonzero symbolic sublimit");
        assert_eq!(
            display_expr(&ctx, symbolic_negative_power_out),
            "1 / sqrt(5)"
        );

        let symbolic_negative_square_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-2)");
        let symbolic_negative_square_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_negative_square_power, x, point_neg_two)
                .expect("expected negative even power over exact sqrt sublimit to fold");
        let Expr::Number(value) = ctx.get(symbolic_negative_square_power_out) else {
            panic!("expected negative even power over exact sqrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(5)));

        let point_zero = parse_expr(&mut ctx, "0");
        let affine_root_negative_square_power = parse_expr(&mut ctx, "(sqrt(x + 4))^(-2)");
        let affine_root_negative_square_power_out =
            try_limit_rules_at_finite(&mut ctx, affine_root_negative_square_power, x, point_zero)
                .expect("expected negative even power over positive affine sqrt sublimit to fold");
        let Expr::Number(value) = ctx.get(affine_root_negative_square_power_out) else {
            panic!("expected negative even power over affine sqrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(4)));

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let cbrt_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^3");
        let cbrt_cube_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_cube_power, x, point_neg_one)
                .expect("expected cube power over exact cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_cube_power_out) else {
            panic!("expected cube power over exact cbrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));

        let cbrt_square_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^2");
        let cbrt_square_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_square_power, x, point_neg_one)
                .expect("expected non-multiple cbrt power to remain explicit");
        assert_eq!(display_expr(&ctx, cbrt_square_power_out), "cbrt(2)^2");

        let cbrt_negative_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^(-3)");
        let cbrt_negative_cube_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_negative_cube_power, x, point_neg_one)
                .expect("expected negative cube power over exact nonzero cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_negative_cube_power_out) else {
            panic!("expected negative cube power over exact cbrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

        let cbrt_zero_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^0");
        let cbrt_zero_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_zero_power, x, point_neg_one)
                .expect("expected zero power over nonzero cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_zero_power_out) else {
            panic!("expected zero power over nonzero cbrt sublimit to fold to one");
        };
        assert_eq!(value, &BigRational::one());

        let numeric_zero_power = parse_expr(&mut ctx, "(abs(x) + 1)^0");
        let numeric_zero_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_zero_power, x, point_neg_two)
                .expect("expected zero power over nonzero sublimit");
        let Expr::Number(value) = ctx.get(numeric_zero_power_out) else {
            panic!("expected safe zero power to fold to one");
        };
        assert_eq!(value, &BigRational::one());

        let zero_base_negative_power = parse_expr(&mut ctx, "(abs(x) - 2)^(-1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_base_negative_power, x, point_neg_two)
                .is_none(),
            "negative integer power over zero sublimit must remain residual"
        );

        let zero_base_zero_power = parse_expr(&mut ctx, "abs(x)^0");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_base_zero_power, x, point_zero).is_none(),
            "zero power over zero sublimit must remain residual"
        );

        let zero_cbrt_base_negative_power = parse_expr(&mut ctx, "cbrt(x)^(-3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_cbrt_base_negative_power, x, point_zero)
                .is_none(),
            "negative cube power over zero cbrt sublimit must remain residual"
        );

        let unresolved_base_power = parse_expr(&mut ctx, "sqrt(x)^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, unresolved_base_power, x, point_zero).is_none(),
            "integer power must not hide an unresolved finite base sublimit"
        );

        let unresolved_base_negative_power = parse_expr(&mut ctx, "sqrt(x)^(-2)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, unresolved_base_negative_power, x, point_zero)
                .is_none(),
            "negative integer power must not hide an unresolved finite base sublimit"
        );
    }

    #[test]
    fn finite_total_real_unary_composition_rejects_unresolved_inner_limit() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");
        let expr = parse_expr(&mut ctx, "sin(sign(x))");

        assert!(
            try_limit_rules_at_finite(&mut ctx, expr, x, point).is_none(),
            "outer total-real function must not hide unresolved discontinuous inner limit"
        );
    }

    #[test]
    fn rational_poly_limit_handles_equal_and_higher_degree_cases() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let equal = parse_expr(&mut ctx, "(3*x^2 + 1)/(6*x^2 - 5)");
        let higher = parse_expr(&mut ctx, "(2*x^3)/(x^2+1)");

        let equal_out = rational_poly_limit(&mut ctx, equal, x, InfSign::Pos).expect("equal");
        let higher_out = rational_poly_limit(&mut ctx, higher, x, InfSign::Neg).expect("higher");

        assert!(matches!(ctx.get(equal_out), Expr::Number(_)));
        assert!(matches!(ctx.get(higher_out), Expr::Neg(_)));
    }

    #[test]
    fn rational_poly_limit_rejects_non_polynomial_and_symbolic_leading_coeff() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let non_poly = parse_expr(&mut ctx, "sin(x)/x");
        let symbolic_lc = parse_expr(&mut ctx, "(y*x^2)/x^2");

        let out1 = rational_poly_limit(&mut ctx, non_poly, x, InfSign::Pos);
        let out2 = rational_poly_limit(&mut ctx, symbolic_lc, x, InfSign::Pos);

        assert!(out1.is_none());
        assert!(out2.is_none());
    }

    #[test]
    fn sqrt_quadratic_minus_linear_limit_resolves_finite_cancellations() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, num, den) in [
            ("sqrt(x^2 + x) - x", 1, 2),
            ("sqrt(x^2 + 1) - x", 0, 1),
            ("x - sqrt(x^2 - x)", 1, 2),
            ("sqrt(x^2 + 3*x) - x", 3, 2),
            ("sqrt(4*x^2 + x) - 2*x", 1, 4),
            ("sqrt(x^2 + x + 1) - x", 1, 2),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out =
                sqrt_quadratic_minus_linear_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn sqrt_quadratic_minus_linear_limit_declines_divergent_and_irrational() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Leading terms that do not cancel diverge; irrational sqrt(a)
        // and non-quadratic radicands have no rational closed form here.
        for source in [
            "sqrt(x^2 + 1) - 2*x",
            "sqrt(2*x^2 + x) - x",
            "sqrt(x^2 + 1) + x",
            "sqrt(x^3 + 1) - x",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                sqrt_quadratic_minus_linear_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn sqrt_minus_sqrt_limit_resolves_matching_leading_radicands() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, num, den) in [
            ("sqrt(x + 1) - sqrt(x)", 0, 1),
            ("sqrt(x^2 + x) - sqrt(x^2 - x)", 1, 1),
            ("sqrt(x^2 + 1) - sqrt(x^2 - 1)", 0, 1),
            ("sqrt(x^2 + 3*x) - sqrt(x^2 + x)", 1, 1),
            ("sqrt(4*x^2 + x) - sqrt(4*x^2 - x)", 1, 2),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = sqrt_minus_sqrt_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn sqrt_minus_sqrt_limit_declines_mismatched_and_high_degree() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Different degrees, irrational leading sqrt, degree > 2
        // (divergent), and additive (non-difference) forms decline.
        for source in [
            "sqrt(x^2 + x) - sqrt(x - 1)",
            "sqrt(2*x^2 + x) - sqrt(2*x^2 - x)",
            "sqrt(x^3 + x) - sqrt(x^3 - x)",
            "sqrt(x^2 + 1) + sqrt(x^2 - 1)",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                sqrt_minus_sqrt_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn radical_conjugate_product_resolves_zero_times_infinity_forms() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // factor * (radical difference that decays) -> finite. The decay rate
        // times the factor's growth lands a rational constant.
        for (source, num, den) in [
            ("x*(sqrt(x^2 + 1) - x)", 1, 2),
            ("x*(sqrt(x^2 + 4) - x)", 2, 1),
            ("x*(sqrt(x^2 - 1) - x)", -1, 2),
            ("x*(sqrt(x^2 + 2*x) - x - 1)", -1, 2),
            ("(sqrt(x^2 + 1) - x)*x", 1, 2),
            ("2*x*(sqrt(x^2 + 1) - x)", 1, 1),
            ("x*(sqrt(9*x^2 + 1) - 3*x)", 1, 6),
            ("x*(2*x - sqrt(4*x^2 + 1))", -1, 4),
            ("sqrt(x)*(sqrt(x + 1) - sqrt(x))", 1, 2),
            ("(sqrt(x + 1) - sqrt(x))*sqrt(x)", 1, 2),
            ("sqrt(x)*(sqrt(x + 2) - sqrt(x))", 1, 1),
            ("x*(sqrt(x^2 + x + 1) - sqrt(x^2 + x))", 1, 2),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = radical_conjugate_product_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn radical_conjugate_product_declines_divergent_irrational_and_neg_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Divergent products (the difference tends to a nonzero constant or the
        // factor outruns the decay), an irrational factor sqrt, a non-cancelling
        // additive form, and a non-radical second factor all decline.
        for source in [
            "x*(sqrt(x^2 + x) - x)",             // difference -> 1/2, product -> +inf
            "x^2*(sqrt(x^2 + 1) - x)",           // factor outruns the 1/x decay
            "sqrt(2*x)*(sqrt(x + 1) - sqrt(x))", // irrational leading sqrt(2)
            "x*(sqrt(x^2 + 1) + x)",             // additive: no leading cancellation
            "x*sin(x)",                          // second factor is not a radical difference
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                radical_conjugate_product_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "must decline: {source}"
            );
        }
        // The finite +inf forms are undefined toward -inf (radicands/sqrt factors
        // need x > 0), so the rule stays honest there.
        let neg_form = parse_expr(&mut ctx, "x*(sqrt(x^2 + 1) - x)");
        assert!(
            radical_conjugate_product_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg)
                .is_none(),
            "must decline toward -inf"
        );
    }

    #[test]
    fn cbrt_conjugate_resolves_bare_and_zero_times_infinity_forms() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Bare cube-root conjugate differences and their 0*inf products resolve
        // to rationals via the a^2+ab+b^2 rationalization (~ 3 x^2 denominator).
        for (source, num, den) in [
            ("cbrt(x^3 + x^2) - x", 1, 3),
            ("cbrt(x^3 + 2*x^2) - x", 2, 3),
            ("cbrt(x^3 + 1) - x", 0, 1),
            ("cbrt(x^3 + x) - x", 0, 1),
            ("x^2*(cbrt(x^3 + 1) - x)", 1, 3),
            ("(cbrt(x^3 + 1) - x)*x^2", 1, 3),
            ("x*(cbrt(x^3 + 1) - x)", 0, 1),
            ("(x^3 + x^2)^(1/3) - x", 1, 3),
            ("x^2*((x^3 + 1)^(1/3) - x)", 1, 3),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = cbrt_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn cbrt_conjugate_declines_divergent_irrational_and_neg_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Divergent products, an irrational cube root, a non-cubic radicand, and
        // a non-cube-root factor all decline.
        for source in [
            "x*(cbrt(x^3 + 3*x^2) - x)", // difference -> 1, product -> +inf
            "x^2*(cbrt(x^3 + x) - x)",   // factor outruns the 1/x decay
            "cbrt(2*x^3 + x^2) - x",     // irrational cbrt(2) leading
            "cbrt(x^2 + x) - x",         // radicand not a cubic
            "x*sin(x)",                  // second factor is not a cube-root difference
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                cbrt_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "must decline: {source}"
            );
        }
        // Defined only toward +inf in this rule; the -inf side stays honest.
        let neg_form = parse_expr(&mut ctx, "cbrt(x^3 + x^2) - x");
        assert!(
            cbrt_conjugate_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg).is_none(),
            "must decline toward -inf"
        );
    }

    #[test]
    fn nth_root_conjugate_resolves_general_root_forms() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // (P)^(1/n) - L conjugate differences and 0*inf products for n >= 4, via
        // the n-term conjugate ~ n d^(n-1) x^(n-1). a/n for the surviving tail.
        for (source, num, den) in [
            ("(x^4 + x^3)^(1/4) - x", 1, 4),
            ("(x^4 + 2*x^3)^(1/4) - x", 1, 2),
            ("(x^5 + x^4)^(1/5) - x", 1, 5),
            ("(16*x^4 + x^3)^(1/4) - 2*x", 1, 32),
            ("(x^4 + x^3)^(1/4) - x - 1", -3, 4),
            ("x^3*((x^4 + 1)^(1/4) - x)", 1, 4),
            ("((x^4 + 1)^(1/4) - x)*x^3", 1, 4),
            ("(x^4 + x^2)^(1/4) - x", 0, 1),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = nth_root_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_number_expr(&ctx, out, num, den);
        }
    }

    #[test]
    fn nth_root_conjugate_declines_irrational_divergent_and_neg_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in [
            "(2*x^4 + x^3)^(1/4) - x",   // irrational 2^(1/4) leading
            "x^4*((x^4 + 1)^(1/4) - x)", // factor outruns the 1/x^3 decay
            "(x^3 + x^2)^(1/4) - x",     // radicand degree != n
            "x*sin(x)",                  // not an nth-root difference
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                nth_root_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "must decline: {source}"
            );
        }
        let neg_form = parse_expr(&mut ctx, "(x^4 + x^3)^(1/4) - x");
        assert!(
            nth_root_conjugate_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg).is_none(),
            "must decline toward -inf"
        );
    }

    #[test]
    fn sqrt_polynomial_ratio_limit_at_infinity_handles_matching_growth() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let pos = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
        let neg = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
        let scaled = parse_expr(&mut ctx, "sqrt(4*x^2 + 1)/(2*x)");
        let even_den = parse_expr(&mut ctx, "sqrt(x^4 + 1)/x^2");
        let irrational_coeff = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/x");
        let scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(3*x)");
        let neg_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(-3*x)");
        let noisy_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + x + 1)/(3*x + 1)");
        let bounded_noise_surd_den = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1)");
        let bounded_noise_surd_noisy_den =
            parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
        let scaled_bounded_noise_surd_noisy_den =
            parse_expr(&mut ctx, "5*sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
        let bounded_noise_surd_scaled_noisy_den = parse_expr(
            &mut ctx,
            "sqrt((3*x + 1)^2 + sin(x))/(2*(2*x + 1 + cos(x)))",
        );

        let pos_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
        let neg_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
        let scaled_out = sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled, x, InfSign::Pos)
            .expect("scaled");
        let even_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, even_den, x, InfSign::Neg)
                .expect("even denominator degree");
        let irrational_coeff_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, irrational_coeff, x, InfSign::Pos)
                .expect("irrational leading coefficient");
        let scaled_surd_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled_surd_den, x, InfSign::Pos)
                .expect("scaled surd denominator");
        let neg_scaled_surd_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg_scaled_surd_den, x, InfSign::Neg)
                .expect("negative scaled surd denominator");
        let noisy_scaled_surd_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            noisy_scaled_surd_den,
            x,
            InfSign::Pos,
        )
        .expect("noisy scaled surd denominator at +inf");
        let noisy_scaled_surd_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            noisy_scaled_surd_den,
            x,
            InfSign::Neg,
        )
        .expect("noisy scaled surd denominator at -inf");
        let bounded_noise_surd_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand noise at +inf");
        let bounded_noise_surd_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_den,
            x,
            InfSign::Neg,
        )
        .expect("bounded radicand noise at -inf");
        let bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand and denominator noise at +inf");
        let bounded_noise_surd_noisy_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_noisy_den,
            x,
            InfSign::Neg,
        )
        .expect("bounded radicand and denominator noise at -inf");
        let scaled_bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            scaled_bounded_noise_surd_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("scaled bounded radicand and denominator noise at +inf");
        let bounded_noise_surd_scaled_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_scaled_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand and scaled denominator noise at +inf");

        let one = BigRational::from_integer(BigInt::from(1));
        let minus_one = -one.clone();
        let two = BigRational::from_integer(BigInt::from(2));
        let three = BigRational::from_integer(BigInt::from(3));
        let three_halves = BigRational::new(BigInt::from(3), BigInt::from(2));
        let minus_three_halves = -three_halves.clone();
        let fifteen_halves = BigRational::new(BigInt::from(15), BigInt::from(2));
        let three_quarters = BigRational::new(BigInt::from(3), BigInt::from(4));
        assert!(matches!(ctx.get(pos_out), Expr::Number(n) if n == &one));
        assert!(matches!(ctx.get(neg_out), Expr::Number(n) if n == &minus_one));
        assert!(matches!(ctx.get(scaled_out), Expr::Number(n) if n == &one));
        assert!(matches!(ctx.get(even_den_out), Expr::Number(n) if n == &one));
        assert!(matches!(
            ctx.get(irrational_coeff_out),
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                    && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))
        ));
        assert!(matches!(
            ctx.get(scaled_surd_den_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(neg_scaled_surd_den_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(noisy_scaled_surd_den_pos_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                    if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                        && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(noisy_scaled_surd_den_neg_out),
            Expr::Neg(inner)
                if matches!(ctx.get(*inner), Expr::Div(num, den)
                    if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                            && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                        && matches!(ctx.get(*den), Expr::Number(n) if n == &three))
        ));
        assert!(
            matches!(ctx.get(bounded_noise_surd_den_pos_out), Expr::Number(n) if n == &three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_noisy_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
        );
        assert!(
            matches!(ctx.get(scaled_bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &fifteen_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_scaled_noisy_den_pos_out), Expr::Number(n) if n == &three_quarters)
        );
    }

    #[test]
    fn sqrt_polynomial_ratio_limit_at_infinity_rejects_unsafe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let negative_leading_coeff = parse_expr(&mut ctx, "sqrt(1 - 2*x^2)/x");
        let odd_radicand_degree = parse_expr(&mut ctx, "sqrt(x^3 + 1)/x");
        let mismatched_growth = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x^2");
        let unbounded_noise = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + x*sin(x))/(2*x + 1)");
        let unbounded_den_noise =
            parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + x*cos(x))");

        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            negative_leading_coeff,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            odd_radicand_degree,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            mismatched_growth,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_noise,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_den_noise,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn polynomial_sqrt_ratio_limit_at_infinity_handles_matching_growth() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let pos = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
        let neg = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
        let even_degree = parse_expr(&mut ctx, "x^2/sqrt(2*x^4 + 1)");
        let rational_coeff = parse_expr(&mut ctx, "x/sqrt(4*x^2 + 1)");
        let noisy = parse_expr(&mut ctx, "(3*x + 1)/sqrt(2*x^2 + x + 1)");
        let bounded_noise_num =
            parse_expr(&mut ctx, "(2*x + 1 + cos(x))/sqrt((3*x + 1)^2 + sin(x))");

        let pos_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
        let neg_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
        let even_degree_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, even_degree, x, InfSign::Neg)
                .expect("even degree");
        let rational_coeff_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, rational_coeff, x, InfSign::Pos)
                .expect("rational sqrt coefficient");
        let noisy_out = polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, noisy, x, InfSign::Pos)
            .expect("lower-order polynomial noise");
        let bounded_noise_num_pos_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Pos)
                .expect("bounded numerator and radicand noise at +inf");
        let bounded_noise_num_neg_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Neg)
                .expect("bounded numerator and radicand noise at -inf");

        let two = BigRational::from_integer(BigInt::from(2));
        let three = BigRational::from_integer(BigInt::from(3));
        let two_thirds = BigRational::new(BigInt::from(2), BigInt::from(3));
        let minus_two_thirds = -two_thirds.clone();
        assert!(matches!(
            ctx.get(pos_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                    if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                        && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(ctx.get(neg_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(even_degree_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(
            ctx.get(rational_coeff_out),
            Expr::Number(n) if n == &BigRational::new(BigInt::from(1), BigInt::from(2))
        ));
        assert!(matches!(
            ctx.get(noisy_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Mul(coeff, sqrt)
                    if matches!(ctx.get(*coeff), Expr::Number(n) if n == &three)
                        && matches!(ctx.get(*sqrt), Expr::Function(fn_id, args)
                            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                                && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(ctx.get(bounded_noise_num_pos_out), Expr::Number(n) if n == &two_thirds));
        assert!(
            matches!(ctx.get(bounded_noise_num_neg_out), Expr::Number(n) if n == &minus_two_thirds)
        );
    }

    #[test]
    fn polynomial_sqrt_ratio_limit_at_infinity_rejects_unsafe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let negative_leading_coeff = parse_expr(&mut ctx, "x/sqrt(1 - 2*x^2)");
        let odd_radicand_degree = parse_expr(&mut ctx, "x/sqrt(x^3 + 1)");
        let mismatched_growth = parse_expr(&mut ctx, "x/sqrt(x^4 + 1)");
        let unbounded_num_noise =
            parse_expr(&mut ctx, "(2*x + 1 + x*cos(x))/sqrt((3*x + 1)^2 + sin(x))");

        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            negative_leading_coeff,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            odd_radicand_degree,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            mismatched_growth,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_num_noise,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn polynomial_limit_at_infinity_handles_numeric_leading_coeff_and_parity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let positive_even = parse_expr(&mut ctx, "x^2 + 1");
        let negative_odd = parse_expr(&mut ctx, "x - 2*x^3");

        let pos_even_out = polynomial_limit_at_infinity(&mut ctx, positive_even, x, InfSign::Neg)
            .expect("positive even polynomial");
        let neg_odd_pos_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Pos)
            .expect("negative odd at +inf");
        let neg_odd_neg_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Neg)
            .expect("negative odd at -inf");

        assert!(matches!(
            ctx.get(pos_even_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(neg_odd_pos_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(neg_odd_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
    }

    #[test]
    fn polynomial_limit_at_infinity_rejects_symbolic_leading_coeff() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let symbolic_lc = parse_expr(&mut ctx, "y*x^2 + 1");

        let out = polynomial_limit_at_infinity(&mut ctx, symbolic_lc, x, InfSign::Pos);

        assert!(out.is_none());
    }

    #[test]
    fn elementary_function_limit_at_infinity_handles_exact_growth_cases() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let sqrt_x = parse_expr(&mut ctx, "sqrt(x)");
        let cbrt_x = parse_expr(&mut ctx, "cbrt(x)");
        let cbrt_neg_linear = parse_expr(&mut ctx, "cbrt(1 - x)");
        let asinh_x = parse_expr(&mut ctx, "asinh(x)");
        let asinh_neg_linear = parse_expr(&mut ctx, "asinh(1 - x)");
        let acosh_x = parse_expr(&mut ctx, "acosh(x)");
        let acosh_neg_linear = parse_expr(&mut ctx, "acosh(1 - x)");
        let atan_x = parse_expr(&mut ctx, "atan(x)");
        let arctan_neg_linear = parse_expr(&mut ctx, "arctan(1 - x)");
        let tanh_x = parse_expr(&mut ctx, "tanh(x)");
        let tanh_neg_linear = parse_expr(&mut ctx, "tanh(1 - x)");
        let sinh_x = parse_expr(&mut ctx, "sinh(x)");
        let sinh_neg_linear = parse_expr(&mut ctx, "sinh(1 - x)");
        let cosh_x = parse_expr(&mut ctx, "cosh(x)");
        let cosh_neg_linear = parse_expr(&mut ctx, "cosh(1 - x)");
        let ln_x = parse_expr(&mut ctx, "ln(x)");
        let ln_quadratic = parse_expr(&mut ctx, "ln(x^2)");
        let shifted_ln_quadratic = parse_expr(&mut ctx, "ln(x^2 - 3)");
        let base_log_quadratic = parse_expr(&mut ctx, "log(2, x^2)");
        let reciprocal_base_log_quadratic = parse_expr(&mut ctx, "log(1/2, x^2)");
        let exp_x = parse_expr(&mut ctx, "exp(x)");
        let exp_neg_x = parse_expr(&mut ctx, "exp(-x)");
        let exp_two_x = parse_expr(&mut ctx, "exp(2*x)");
        let ln_neg_linear = parse_expr(&mut ctx, "ln(-x + 1)");
        let ln_negative_tail_quadratic = parse_expr(&mut ctx, "ln(3 - x^2)");
        let base_log_negative_tail = parse_expr(&mut ctx, "log(2, 3 - x^2)");
        let invalid_base_log_quadratic = parse_expr(&mut ctx, "log(1, x^2)");
        let sqrt_neg_linear = parse_expr(&mut ctx, "sqrt(1 - x)");
        let cbrt_exp_x = parse_expr(&mut ctx, "cbrt(exp(x))");
        let cbrt_exp_neg_x = parse_expr(&mut ctx, "cbrt(exp(-x))");
        let asinh_exp_x = parse_expr(&mut ctx, "asinh(exp(x))");
        let asinh_exp_neg_x = parse_expr(&mut ctx, "asinh(exp(-x))");
        let acosh_exp_x = parse_expr(&mut ctx, "acosh(exp(x))");
        let acosh_exp_neg_x = parse_expr(&mut ctx, "acosh(exp(-x))");
        let atan_exp_x = parse_expr(&mut ctx, "atan(exp(x))");
        let arctan_exp_neg_x = parse_expr(&mut ctx, "arctan(exp(-x))");
        let tanh_exp_x = parse_expr(&mut ctx, "tanh(exp(x))");
        let tanh_exp_neg_x = parse_expr(&mut ctx, "tanh(exp(-x))");
        let sinh_exp_x = parse_expr(&mut ctx, "sinh(exp(x))");
        let sinh_exp_neg_x = parse_expr(&mut ctx, "sinh(exp(-x))");
        let cosh_exp_x = parse_expr(&mut ctx, "cosh(exp(x))");
        let cosh_exp_neg_x = parse_expr(&mut ctx, "cosh(exp(-x))");
        let exp_quadratic = parse_expr(&mut ctx, "exp(x^2)");
        let negative_tail_exp_quartic = parse_expr(&mut ctx, "exp(2 - x^4)");
        let exp_cubic = parse_expr(&mut ctx, "exp(x^3 - 2*x)");
        let parametric_tail_exp_quadratic = parse_expr(&mut ctx, "exp(a*x^2 + 1)");
        let nested_exp_quadratic = parse_expr(&mut ctx, "exp(exp(x^2))");
        let cbrt_quadratic = parse_expr(&mut ctx, "cbrt(x^2)");
        let negative_tail_cbrt_quartic = parse_expr(&mut ctx, "cbrt(2 - x^4)");
        let parametric_tail_cbrt_quadratic = parse_expr(&mut ctx, "cbrt(a*x^2 + 1)");
        let cbrt_exp_quadratic = parse_expr(&mut ctx, "cbrt(exp(x^2))");
        let asinh_quadratic = parse_expr(&mut ctx, "asinh(x^2)");
        let negative_tail_asinh_quartic = parse_expr(&mut ctx, "asinh(2 - x^4)");
        let parametric_tail_asinh_quadratic = parse_expr(&mut ctx, "asinh(a*x^2 + 1)");
        let asinh_exp_quadratic = parse_expr(&mut ctx, "asinh(exp(x^2))");
        let acosh_quadratic = parse_expr(&mut ctx, "acosh(x^2)");
        let shifted_acosh_quadratic = parse_expr(&mut ctx, "acosh(x^2 - 3)");
        let negative_tail_acosh_quadratic = parse_expr(&mut ctx, "acosh(3 - x^2)");
        let parametric_tail_acosh_quadratic = parse_expr(&mut ctx, "acosh(a*x^2 + 1)");
        let acosh_exp_quadratic = parse_expr(&mut ctx, "acosh(exp(x^2))");
        let atan_quadratic = parse_expr(&mut ctx, "atan(x^2)");
        let negative_tail_atan_quartic = parse_expr(&mut ctx, "atan(2 - x^4)");
        let arctan_cubic = parse_expr(&mut ctx, "arctan(x^3 - 2*x)");
        let parametric_tail_atan_quadratic = parse_expr(&mut ctx, "atan(a*x^2 + 1)");
        let arctan_exp_quadratic = parse_expr(&mut ctx, "arctan(exp(x^2))");
        let tanh_quadratic = parse_expr(&mut ctx, "tanh(x^2)");
        let negative_tail_tanh_quartic = parse_expr(&mut ctx, "tanh(2 - x^4)");
        let parametric_tail_tanh_quadratic = parse_expr(&mut ctx, "tanh(a*x^2 + 1)");
        let tanh_exp_quadratic = parse_expr(&mut ctx, "tanh(exp(x^2))");
        let sinh_quadratic = parse_expr(&mut ctx, "sinh(x^2)");
        let negative_tail_sinh_quartic = parse_expr(&mut ctx, "sinh(2 - x^4)");
        let sinh_cubic = parse_expr(&mut ctx, "sinh(x^3 - 2*x)");
        let parametric_tail_sinh_quadratic = parse_expr(&mut ctx, "sinh(a*x^2 + 1)");
        let sinh_exp_quadratic = parse_expr(&mut ctx, "sinh(exp(x^2))");
        let cosh_quadratic = parse_expr(&mut ctx, "cosh(x^2)");
        let negative_tail_cosh_quartic = parse_expr(&mut ctx, "cosh(2 - x^4)");
        let parametric_tail_cosh_quadratic = parse_expr(&mut ctx, "cosh(a*x^2 + 1)");
        let cosh_exp_quadratic = parse_expr(&mut ctx, "cosh(exp(x^2))");

        let sqrt_pos = elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Pos)
            .expect("sqrt at +inf");
        let cbrt_pos = elementary_function_limit_at_infinity(&mut ctx, cbrt_x, x, InfSign::Pos)
            .expect("cbrt at +inf");
        let cbrt_neg = elementary_function_limit_at_infinity(&mut ctx, cbrt_x, x, InfSign::Neg)
            .expect("cbrt at -inf");
        let cbrt_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, cbrt_neg_linear, x, InfSign::Pos)
                .expect("cbrt(1 - x) at +inf");
        let asinh_pos = elementary_function_limit_at_infinity(&mut ctx, asinh_x, x, InfSign::Pos)
            .expect("asinh at +inf");
        let asinh_neg = elementary_function_limit_at_infinity(&mut ctx, asinh_x, x, InfSign::Neg)
            .expect("asinh at -inf");
        let asinh_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, asinh_neg_linear, x, InfSign::Pos)
                .expect("asinh(1 - x) at +inf");
        let acosh_pos = elementary_function_limit_at_infinity(&mut ctx, acosh_x, x, InfSign::Pos)
            .expect("acosh at +inf");
        let acosh_neg_linear_neg =
            elementary_function_limit_at_infinity(&mut ctx, acosh_neg_linear, x, InfSign::Neg)
                .expect("acosh(1 - x) at -inf");
        let acosh_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, acosh_quadratic, x, InfSign::Pos)
                .expect("acosh(x^2) at +inf");
        let acosh_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, acosh_quadratic, x, InfSign::Neg)
                .expect("acosh(x^2) at -inf");
        let shifted_acosh_quadratic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            shifted_acosh_quadratic,
            x,
            InfSign::Pos,
        )
        .expect("acosh(x^2 - 3) at +inf");
        let atan_pos = elementary_function_limit_at_infinity(&mut ctx, atan_x, x, InfSign::Pos)
            .expect("atan at +inf");
        let atan_neg = elementary_function_limit_at_infinity(&mut ctx, atan_x, x, InfSign::Neg)
            .expect("atan at -inf");
        let arctan_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, arctan_neg_linear, x, InfSign::Pos)
                .expect("arctan(1 - x) at +inf");
        let tanh_pos = elementary_function_limit_at_infinity(&mut ctx, tanh_x, x, InfSign::Pos)
            .expect("tanh at +inf");
        let tanh_neg = elementary_function_limit_at_infinity(&mut ctx, tanh_x, x, InfSign::Neg)
            .expect("tanh at -inf");
        let tanh_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, tanh_neg_linear, x, InfSign::Pos)
                .expect("tanh(1 - x) at +inf");
        let sinh_pos = elementary_function_limit_at_infinity(&mut ctx, sinh_x, x, InfSign::Pos)
            .expect("sinh at +inf");
        let sinh_neg = elementary_function_limit_at_infinity(&mut ctx, sinh_x, x, InfSign::Neg)
            .expect("sinh at -inf");
        let sinh_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, sinh_neg_linear, x, InfSign::Pos)
                .expect("sinh(1 - x) at +inf");
        let cosh_pos = elementary_function_limit_at_infinity(&mut ctx, cosh_x, x, InfSign::Pos)
            .expect("cosh at +inf");
        let cosh_neg = elementary_function_limit_at_infinity(&mut ctx, cosh_x, x, InfSign::Neg)
            .expect("cosh at -inf");
        let cosh_neg_linear_pos =
            elementary_function_limit_at_infinity(&mut ctx, cosh_neg_linear, x, InfSign::Pos)
                .expect("cosh(1 - x) at +inf");
        let ln_pos = elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Pos)
            .expect("ln at +inf");
        let ln_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, ln_quadratic, x, InfSign::Pos)
                .expect("ln(x^2) at +inf");
        let ln_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, ln_quadratic, x, InfSign::Neg)
                .expect("ln(x^2) at -inf");
        let shifted_ln_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, shifted_ln_quadratic, x, InfSign::Pos)
                .expect("ln(x^2 - 3) at +inf");
        let base_log_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, base_log_quadratic, x, InfSign::Pos)
                .expect("log(2, x^2) at +inf");
        let reciprocal_base_log_quadratic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            reciprocal_base_log_quadratic,
            x,
            InfSign::Pos,
        )
        .expect("log(1/2, x^2) at +inf");
        let exp_pos = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Pos)
            .expect("exp at +inf");
        let exp_neg = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Neg)
            .expect("exp at -inf");
        let exp_neg_x_pos =
            elementary_function_limit_at_infinity(&mut ctx, exp_neg_x, x, InfSign::Pos)
                .expect("exp(-x) at +inf");
        let exp_two_x_neg =
            elementary_function_limit_at_infinity(&mut ctx, exp_two_x, x, InfSign::Neg)
                .expect("exp(2*x) at -inf");
        let exp_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, exp_quadratic, x, InfSign::Pos)
                .expect("exp(x^2) at +inf");
        let exp_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, exp_quadratic, x, InfSign::Neg)
                .expect("exp(x^2) at -inf");
        let negative_tail_exp_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_exp_quartic,
            x,
            InfSign::Pos,
        )
        .expect("exp(2 - x^4) at +inf");
        let exp_cubic_neg =
            elementary_function_limit_at_infinity(&mut ctx, exp_cubic, x, InfSign::Neg)
                .expect("exp(x^3 - 2*x) at -inf");
        let ln_neg_linear_neg =
            elementary_function_limit_at_infinity(&mut ctx, ln_neg_linear, x, InfSign::Neg)
                .expect("ln(-x + 1) at -inf");
        let sqrt_neg_linear_neg =
            elementary_function_limit_at_infinity(&mut ctx, sqrt_neg_linear, x, InfSign::Neg)
                .expect("sqrt(1 - x) at -inf");
        let cbrt_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, cbrt_exp_x, x, InfSign::Pos)
                .expect("cbrt(exp(x)) at +inf");
        let cbrt_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, cbrt_exp_neg_x, x, InfSign::Pos)
                .expect("cbrt(exp(-x)) at +inf");
        let cbrt_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, cbrt_quadratic, x, InfSign::Pos)
                .expect("cbrt(x^2) at +inf");
        let negative_tail_cbrt_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_cbrt_quartic,
            x,
            InfSign::Pos,
        )
        .expect("cbrt(2 - x^4) at +inf");
        let asinh_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, asinh_exp_x, x, InfSign::Pos)
                .expect("asinh(exp(x)) at +inf");
        let asinh_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, asinh_exp_neg_x, x, InfSign::Pos)
                .expect("asinh(exp(-x)) at +inf");
        let asinh_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, asinh_quadratic, x, InfSign::Pos)
                .expect("asinh(x^2) at +inf");
        let negative_tail_asinh_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_asinh_quartic,
            x,
            InfSign::Pos,
        )
        .expect("asinh(2 - x^4) at +inf");
        let acosh_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, acosh_exp_x, x, InfSign::Pos)
                .expect("acosh(exp(x)) at +inf");
        let acosh_exp_neg_neg =
            elementary_function_limit_at_infinity(&mut ctx, acosh_exp_neg_x, x, InfSign::Neg)
                .expect("acosh(exp(-x)) at -inf");
        let atan_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, atan_exp_x, x, InfSign::Pos)
                .expect("atan(exp(x)) at +inf");
        let arctan_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, arctan_exp_neg_x, x, InfSign::Pos)
                .expect("arctan(exp(-x)) at +inf");
        let atan_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, atan_quadratic, x, InfSign::Pos)
                .expect("atan(x^2) at +inf");
        let atan_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, atan_quadratic, x, InfSign::Neg)
                .expect("atan(x^2) at -inf");
        let negative_tail_atan_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_atan_quartic,
            x,
            InfSign::Pos,
        )
        .expect("atan(2 - x^4) at +inf");
        let arctan_cubic_neg =
            elementary_function_limit_at_infinity(&mut ctx, arctan_cubic, x, InfSign::Neg)
                .expect("arctan(x^3 - 2*x) at -inf");
        let tanh_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, tanh_exp_x, x, InfSign::Pos)
                .expect("tanh(exp(x)) at +inf");
        let tanh_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, tanh_exp_neg_x, x, InfSign::Pos)
                .expect("tanh(exp(-x)) at +inf");
        let tanh_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, tanh_quadratic, x, InfSign::Pos)
                .expect("tanh(x^2) at +inf");
        let tanh_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, tanh_quadratic, x, InfSign::Neg)
                .expect("tanh(x^2) at -inf");
        let negative_tail_tanh_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_tanh_quartic,
            x,
            InfSign::Pos,
        )
        .expect("tanh(2 - x^4) at +inf");
        let sinh_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, sinh_exp_x, x, InfSign::Pos)
                .expect("sinh(exp(x)) at +inf");
        let sinh_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, sinh_exp_neg_x, x, InfSign::Pos)
                .expect("sinh(exp(-x)) at +inf");
        let sinh_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, sinh_quadratic, x, InfSign::Pos)
                .expect("sinh(x^2) at +inf");
        let sinh_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, sinh_quadratic, x, InfSign::Neg)
                .expect("sinh(x^2) at -inf");
        let negative_tail_sinh_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_sinh_quartic,
            x,
            InfSign::Pos,
        )
        .expect("sinh(2 - x^4) at +inf");
        let sinh_cubic_neg =
            elementary_function_limit_at_infinity(&mut ctx, sinh_cubic, x, InfSign::Neg)
                .expect("sinh(x^3 - 2*x) at -inf");
        let cosh_exp_pos =
            elementary_function_limit_at_infinity(&mut ctx, cosh_exp_x, x, InfSign::Pos)
                .expect("cosh(exp(x)) at +inf");
        let cosh_exp_neg_pos =
            elementary_function_limit_at_infinity(&mut ctx, cosh_exp_neg_x, x, InfSign::Pos)
                .expect("cosh(exp(-x)) at +inf");
        let cosh_quadratic_pos =
            elementary_function_limit_at_infinity(&mut ctx, cosh_quadratic, x, InfSign::Pos)
                .expect("cosh(x^2) at +inf");
        let cosh_quadratic_neg =
            elementary_function_limit_at_infinity(&mut ctx, cosh_quadratic, x, InfSign::Neg)
                .expect("cosh(x^2) at -inf");
        let negative_tail_cosh_quartic_pos = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_cosh_quartic,
            x,
            InfSign::Pos,
        )
        .expect("cosh(2 - x^4) at +inf");

        assert!(matches!(
            ctx.get(sqrt_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(cbrt_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(cbrt_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(cbrt_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(asinh_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(asinh_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(asinh_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(acosh_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(acosh_neg_linear_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(acosh_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(acosh_quadratic_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(shifted_acosh_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert_eq!(display_expr(&ctx, atan_pos), "pi / 2");
        assert_eq!(display_expr(&ctx, atan_neg), "-pi / 2");
        assert_eq!(display_expr(&ctx, arctan_neg_linear_pos), "-pi / 2");
        assert_eq!(display_expr(&ctx, tanh_pos), "1");
        assert_eq!(display_expr(&ctx, tanh_neg), "-1");
        assert_eq!(display_expr(&ctx, tanh_neg_linear_pos), "-1");
        assert!(matches!(
            ctx.get(sinh_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(sinh_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(sinh_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(cosh_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(cosh_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(cosh_neg_linear_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(ln_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(ln_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(ln_quadratic_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(shifted_ln_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(base_log_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(reciprocal_base_log_quadratic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(exp_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(exp_neg_x_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(exp_two_x_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(exp_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_quadratic_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(negative_tail_exp_quartic_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(exp_cubic_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(ln_neg_linear_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(sqrt_neg_linear_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(cbrt_exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(cbrt_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(cbrt_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(negative_tail_cbrt_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(asinh_exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(asinh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(asinh_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(negative_tail_asinh_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(acosh_exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(acosh_exp_neg_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert_eq!(display_expr(&ctx, atan_exp_pos), "pi / 2");
        assert!(
            matches!(ctx.get(arctan_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert_eq!(display_expr(&ctx, atan_quadratic_pos), "pi / 2");
        assert_eq!(display_expr(&ctx, atan_quadratic_neg), "pi / 2");
        assert_eq!(
            display_expr(&ctx, negative_tail_atan_quartic_pos),
            "-pi / 2"
        );
        assert_eq!(display_expr(&ctx, arctan_cubic_neg), "-pi / 2");
        assert_eq!(display_expr(&ctx, tanh_exp_pos), "1");
        assert!(
            matches!(ctx.get(tanh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert_eq!(display_expr(&ctx, tanh_quadratic_pos), "1");
        assert_eq!(display_expr(&ctx, tanh_quadratic_neg), "1");
        assert_eq!(display_expr(&ctx, negative_tail_tanh_quartic_pos), "-1");
        assert!(matches!(
            ctx.get(sinh_exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(sinh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(sinh_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(sinh_quadratic_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(negative_tail_sinh_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(sinh_cubic_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(cosh_exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert_eq!(display_expr(&ctx, cosh_exp_neg_pos), "1");
        assert!(matches!(
            ctx.get(cosh_quadratic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(cosh_quadratic_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(negative_tail_cosh_quartic_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Neg).is_none());
        assert!(elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Neg).is_none());
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, acosh_x, x, InfSign::Neg).is_none()
        );
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            ln_negative_tail_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            base_log_negative_tail,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            invalid_base_log_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, acosh_neg_linear, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, acosh_exp_x, x, InfSign::Neg).is_none()
        );
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, acosh_exp_neg_x, x, InfSign::Pos)
                .is_none()
        );
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            nested_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_cbrt_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            cbrt_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_asinh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            asinh_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_acosh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_acosh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            acosh_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_atan_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            arctan_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_tanh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            tanh_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_sinh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            sinh_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_tail_cosh_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            cosh_exp_quadratic,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn elementary_function_limit_at_infinity_handles_rational_positive_unbounded_arguments() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let ln_rational = parse_expr(&mut ctx, "ln((x^2 + 1)/(x + 1))");
        let base_log_rational = parse_expr(&mut ctx, "log(2, (x^2 + 1)/(x + 1))");
        let reciprocal_base_log_rational = parse_expr(&mut ctx, "log(1/2, (x^2 + 1)/(x + 1))");
        let acosh_rational = parse_expr(&mut ctx, "acosh((x^2 + 1)/(x + 1))");
        let negative_tail_log = parse_expr(&mut ctx, "ln((x^2 + 1)/(1 - x))");
        let proper_ln = parse_expr(&mut ctx, "ln((x + 1)/(x^2 + 1))");
        let proper_log2 = parse_expr(&mut ctx, "log2((x + 1)/(x^2 + 1))");
        let proper_base_log = parse_expr(&mut ctx, "log(2, (x + 1)/(x^2 + 1))");
        let proper_reciprocal_base_log = parse_expr(&mut ctx, "log(1/2, (x + 1)/(x^2 + 1))");
        let constant_num_log = parse_expr(&mut ctx, "log2(1/(x^2 + 1))");
        let proper_acosh = parse_expr(&mut ctx, "acosh((x + 1)/(x^2 + 1))");
        let parametric_log = parse_expr(&mut ctx, "ln((a*x^2 + 1)/(x + 1))");
        let negative_zero_tail_log = parse_expr(&mut ctx, "ln((1 - x)/(x^2 + 1))");
        let parametric_zero_tail_log = parse_expr(&mut ctx, "ln((a*x + 1)/(x^2 + 1))");
        let finite_ratio_ln = parse_expr(&mut ctx, "ln((2*x^2 + 1)/(x^2 + 1))");
        let finite_ratio_log2 = parse_expr(&mut ctx, "log2((2*x^2 + 1)/(x^2 + 1))");
        let finite_ratio_log10 = parse_expr(&mut ctx, "log10((100*x^2 + 1)/(x^2 + 1))");
        let finite_ratio_base_log = parse_expr(&mut ctx, "log(2, (2*x^2 + 1)/(x^2 + 1))");
        let finite_ratio_reciprocal_base_log =
            parse_expr(&mut ctx, "log(1/2, (2*x^2 + 1)/(x^2 + 1))");
        let unit_ratio_ln = parse_expr(&mut ctx, "ln((x^2 + 1)/(x^2 + 1))");
        let finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((2*x^2 + 1)/(x^2 + 1))");
        let finite_square_ratio_sqrt = parse_expr(&mut ctx, "sqrt((4*x^2 + 1)/(x^2 + 1))");
        let unit_ratio_sqrt = parse_expr(&mut ctx, "sqrt((x^2 + 1)/(x^2 + 1))");
        let proper_sqrt = parse_expr(&mut ctx, "sqrt((x + 1)/(x^2 + 1))");
        let negative_direction_proper_sqrt = parse_expr(&mut ctx, "sqrt((1 - x)/(x^2 + 1))");
        let negative_finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((1 - 2*x^2)/(x^2 + 1))");
        let parametric_zero_tail_sqrt = parse_expr(&mut ctx, "sqrt((a*x + 1)/(x^2 + 1))");
        let parametric_finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((a*x^2 + 1)/(x^2 + 1))");
        let unbounded_ratio_cbrt = parse_expr(&mut ctx, "cbrt((x^2 + 1)/(x + 1))");
        let negative_unbounded_ratio_cbrt = parse_expr(&mut ctx, "cbrt((x^2 + 1)/(1 - x))");
        let finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((8*x^2 + 1)/(x^2 + 1))");
        let negative_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((1 - 8*x^2)/(x^2 + 1))");
        let nonexact_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((2*x^2 + 1)/(x^2 + 1))");
        let proper_cbrt = parse_expr(&mut ctx, "cbrt((x + 1)/(x^2 + 1))");
        let negative_zero_tail_cbrt = parse_expr(&mut ctx, "cbrt((1 - x)/(x^2 + 1))");
        let parametric_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((a*x^2 + 1)/(x^2 + 1))");
        let finite_ratio_acosh = parse_expr(&mut ctx, "acosh((2*x^2 + 1)/(x^2 + 1))");
        let unit_ratio_acosh = parse_expr(&mut ctx, "acosh((x^2 + 1)/(x^2 + 1))");
        let small_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((x^2 + 1)/(2*x^2 + 1))");
        let negative_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((1 - 2*x^2)/(x^2 + 1))");
        let parametric_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((a*x^2 + 1)/(x^2 + 1))");
        let negative_finite_ratio_log = parse_expr(&mut ctx, "ln((1 - 2*x^2)/(x^2 + 1))");
        let parametric_finite_ratio_log = parse_expr(&mut ctx, "ln((a*x^2 + 1)/(x^2 + 1))");

        let ln_out = elementary_function_limit_at_infinity(&mut ctx, ln_rational, x, InfSign::Pos)
            .expect("ln rational positive unbounded");
        let base_log_out =
            elementary_function_limit_at_infinity(&mut ctx, base_log_rational, x, InfSign::Pos)
                .expect("base log rational positive unbounded");
        let reciprocal_base_log_out = elementary_function_limit_at_infinity(
            &mut ctx,
            reciprocal_base_log_rational,
            x,
            InfSign::Pos,
        )
        .expect("base < 1 log rational positive unbounded");
        let acosh_out =
            elementary_function_limit_at_infinity(&mut ctx, acosh_rational, x, InfSign::Pos)
                .expect("acosh rational positive unbounded");
        let proper_ln_out =
            elementary_function_limit_at_infinity(&mut ctx, proper_ln, x, InfSign::Pos)
                .expect("ln rational positive zero tail");
        let proper_log2_out =
            elementary_function_limit_at_infinity(&mut ctx, proper_log2, x, InfSign::Pos)
                .expect("log2 rational positive zero tail");
        let proper_base_log_out =
            elementary_function_limit_at_infinity(&mut ctx, proper_base_log, x, InfSign::Pos)
                .expect("base log rational positive zero tail");
        let proper_reciprocal_base_log_out = elementary_function_limit_at_infinity(
            &mut ctx,
            proper_reciprocal_base_log,
            x,
            InfSign::Pos,
        )
        .expect("base < 1 log rational positive zero tail");
        let constant_num_log_out =
            elementary_function_limit_at_infinity(&mut ctx, constant_num_log, x, InfSign::Pos)
                .expect("log2 rational positive zero tail with constant numerator");
        let finite_ratio_ln_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_ln, x, InfSign::Pos)
                .expect("ln rational positive finite tail");
        let finite_ratio_log2_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_log2, x, InfSign::Pos)
                .expect("log2 rational positive finite tail");
        let finite_ratio_log10_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_log10, x, InfSign::Pos)
                .expect("log10 rational positive finite tail");
        let finite_ratio_base_log_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_base_log, x, InfSign::Pos)
                .expect("base log rational positive finite tail");
        let finite_ratio_reciprocal_base_log_out = elementary_function_limit_at_infinity(
            &mut ctx,
            finite_ratio_reciprocal_base_log,
            x,
            InfSign::Pos,
        )
        .expect("base < 1 log rational positive finite tail");
        let unit_ratio_ln_out =
            elementary_function_limit_at_infinity(&mut ctx, unit_ratio_ln, x, InfSign::Pos)
                .expect("ln rational unit finite tail");
        let finite_ratio_sqrt_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_sqrt, x, InfSign::Pos)
                .expect("sqrt rational positive finite tail");
        let finite_square_ratio_sqrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            finite_square_ratio_sqrt,
            x,
            InfSign::Pos,
        )
        .expect("sqrt rational positive finite square tail");
        let unit_ratio_sqrt_out =
            elementary_function_limit_at_infinity(&mut ctx, unit_ratio_sqrt, x, InfSign::Pos)
                .expect("sqrt rational unit finite tail");
        let proper_sqrt_out =
            elementary_function_limit_at_infinity(&mut ctx, proper_sqrt, x, InfSign::Pos)
                .expect("sqrt rational positive zero tail");
        let negative_direction_proper_sqrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_direction_proper_sqrt,
            x,
            InfSign::Neg,
        )
        .expect("sqrt rational positive zero tail at negative infinity");
        let unbounded_ratio_cbrt_out =
            elementary_function_limit_at_infinity(&mut ctx, unbounded_ratio_cbrt, x, InfSign::Pos)
                .expect("cbrt rational positive unbounded tail");
        let negative_unbounded_ratio_cbrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_unbounded_ratio_cbrt,
            x,
            InfSign::Pos,
        )
        .expect("cbrt rational negative unbounded tail");
        let finite_ratio_cbrt_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_cbrt, x, InfSign::Pos)
                .expect("cbrt rational exact finite tail");
        let negative_finite_ratio_cbrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_finite_ratio_cbrt,
            x,
            InfSign::Pos,
        )
        .expect("cbrt rational exact negative finite tail");
        let nonexact_finite_ratio_cbrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            nonexact_finite_ratio_cbrt,
            x,
            InfSign::Pos,
        )
        .expect("cbrt rational non-exact finite tail");
        let proper_cbrt_out =
            elementary_function_limit_at_infinity(&mut ctx, proper_cbrt, x, InfSign::Pos)
                .expect("cbrt rational positive zero tail");
        let negative_zero_tail_cbrt_out = elementary_function_limit_at_infinity(
            &mut ctx,
            negative_zero_tail_cbrt,
            x,
            InfSign::Pos,
        )
        .expect("cbrt rational negative zero tail");
        let finite_ratio_acosh_out =
            elementary_function_limit_at_infinity(&mut ctx, finite_ratio_acosh, x, InfSign::Pos)
                .expect("acosh rational finite tail strictly inside domain");

        assert!(matches!(
            ctx.get(ln_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(reciprocal_base_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(acosh_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(proper_ln_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(proper_log2_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(
            matches!(ctx.get(proper_base_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(proper_reciprocal_base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(constant_num_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert_eq!(display_expr(&ctx, finite_ratio_ln_out), "ln(2)");
        assert!(matches!(
            ctx.get(finite_ratio_log2_out),
            Expr::Number(value) if value == &rational_one()
        ));
        assert!(matches!(
            ctx.get(finite_ratio_log10_out),
            Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
        ));
        assert!(matches!(
            ctx.get(finite_ratio_base_log_out),
            Expr::Number(value) if value == &rational_one()
        ));
        assert!(matches!(
            ctx.get(finite_ratio_reciprocal_base_log_out),
            Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(-1))
        ));
        assert!(matches!(
            ctx.get(unit_ratio_ln_out),
            Expr::Number(value) if value.is_zero()
        ));
        assert_eq!(display_expr(&ctx, finite_ratio_sqrt_out), "sqrt(2)");
        assert!(matches!(
            ctx.get(finite_square_ratio_sqrt_out),
            Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
        ));
        assert!(matches!(
            ctx.get(unit_ratio_sqrt_out),
            Expr::Number(value) if value == &rational_one()
        ));
        assert!(matches!(ctx.get(proper_sqrt_out), Expr::Number(value) if value.is_zero()));
        assert!(
            matches!(ctx.get(negative_direction_proper_sqrt_out), Expr::Number(value) if value.is_zero())
        );
        assert!(matches!(
            ctx.get(unbounded_ratio_cbrt_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(negative_unbounded_ratio_cbrt_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        );
        assert!(matches!(
            ctx.get(finite_ratio_cbrt_out),
            Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
        ));
        assert!(matches!(
            ctx.get(negative_finite_ratio_cbrt_out),
            Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(-2))
        ));
        assert_eq!(
            display_expr(&ctx, nonexact_finite_ratio_cbrt_out),
            "cbrt(2)"
        );
        assert!(matches!(ctx.get(proper_cbrt_out), Expr::Number(value) if value.is_zero()));
        assert!(
            matches!(ctx.get(negative_zero_tail_cbrt_out), Expr::Number(value) if value.is_zero())
        );
        assert_eq!(display_expr(&ctx, finite_ratio_acosh_out), "acosh(2)");
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_tail_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, proper_acosh, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, parametric_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_zero_tail_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_zero_tail_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_direction_proper_sqrt,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_zero_tail_sqrt,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_finite_ratio_sqrt,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_finite_ratio_sqrt,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_finite_ratio_cbrt,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, unit_ratio_acosh, x, InfSign::Pos)
                .is_none()
        );
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            small_finite_ratio_acosh,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_finite_ratio_acosh,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_finite_ratio_acosh,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            negative_finite_ratio_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(elementary_function_limit_at_infinity(
            &mut ctx,
            parametric_finite_ratio_log,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn additive_limit_at_infinity_combines_finite_and_infinite_terms_conservatively() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let sqrt_plus_one = parse_expr(&mut ctx, "sqrt(x) + 1");
        let decaying_exp_plus_one = parse_expr(&mut ctx, "exp(-x) + 1");
        let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
        let poly_cancel = parse_expr(&mut ctx, "x^2 - x^2");

        let sqrt_plus_one_out =
            try_limit_rules_at_infinity(&mut ctx, sqrt_plus_one, x, InfSign::Pos)
                .expect("sqrt plus one");
        let decaying_exp_plus_one_out =
            try_limit_rules_at_infinity(&mut ctx, decaying_exp_plus_one, x, InfSign::Pos)
                .expect("decaying exp plus one");
        let poly_cancel_out = try_limit_rules_at_infinity(&mut ctx, poly_cancel, x, InfSign::Pos)
            .expect("polynomial cancellation");

        assert!(matches!(
            ctx.get(sqrt_plus_one_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(decaying_exp_plus_one_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
        );
        let exp_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
                .expect("exp dominates polynomial");
        assert!(matches!(
            ctx.get(exp_minus_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_cancel_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn at_infinity_composition_handles_symbolic_finite_factors_and_radical_tails() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        // Symbolic finite factors compose (pi/2 from arctan).
        for source in ["2*arctan(x)", "arctan(x)/2", "arctan(x)*arctan(x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_some(),
                "must resolve: {source}"
            );
        }
        // Radical unbounded tails reach the saturating composition table.
        let cases = [
            ("arctan(sqrt(x))", "pi / 2"),
            ("arctan(-sqrt(x))", "-pi / 2"),
            ("tanh(sqrt(x))", "1"),
            ("e^(-sqrt(x))", "0"),
            ("arctan(x^(3/2))", "pi / 2"),
        ];
        for (source, expected) in cases {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_eq!(
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: out
                    }
                ),
                expected,
                "{source}"
            );
        }
        // x * arctan(x) stays refused here: infinite times symbolic
        // finite needs a numeric scale to fix the sign.
        let mixed = parse_expr(&mut ctx, "x*arctan(x)");
        assert!(try_limit_rules_at_infinity(&mut ctx, mixed, x, InfSign::Pos).is_none());
    }

    #[test]
    fn multiplicative_limit_at_infinity_combines_only_determined_products_and_quotients() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let scaled_sqrt = parse_expr(&mut ctx, "2*sqrt(x)");
        let neg_sqrt = parse_expr(&mut ctx, "-sqrt(x)");
        let reciprocal_exp = parse_expr(&mut ctx, "1/exp(x)");
        let indeterminate_exp_difference = parse_expr(&mut ctx, "exp(x)-exp(x)");

        let scaled_sqrt_out = try_limit_rules_at_infinity(&mut ctx, scaled_sqrt, x, InfSign::Pos)
            .expect("scaled sqrt");
        let neg_sqrt_out =
            try_limit_rules_at_infinity(&mut ctx, neg_sqrt, x, InfSign::Pos).expect("neg sqrt");
        let reciprocal_exp_out =
            try_limit_rules_at_infinity(&mut ctx, reciprocal_exp, x, InfSign::Pos)
                .expect("reciprocal exp");

        assert!(matches!(
            ctx.get(scaled_sqrt_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(neg_sqrt_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(reciprocal_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            indeterminate_exp_difference,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn exponential_polynomial_dominance_handles_only_exact_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
        let poly_minus_exp = parse_expr(&mut ctx, "x^2 - exp(x)");
        let poly_over_exp = parse_expr(&mut ctx, "x^2/exp(x)");
        let exp_over_poly = parse_expr(&mut ctx, "exp(x)/x^2");
        let poly_times_decaying_exp = parse_expr(&mut ctx, "x*exp(x)");
        let poly_over_linear_exp = parse_expr(&mut ctx, "x^2/exp(2*x)");
        let linear_exp_over_poly = parse_expr(&mut ctx, "exp(2*x)/x^2");
        let poly_times_decaying_linear_exp = parse_expr(&mut ctx, "x^2*exp(2*x)");
        let poly_over_decaying_linear_exp = parse_expr(&mut ctx, "x^2/exp(-2*x)");
        let constant_over_decaying_linear_exp = parse_expr(&mut ctx, "1/exp(-2*x)");
        let poly_over_polynomial_exp = parse_expr(&mut ctx, "x^2/exp(x^2)");
        let polynomial_exp_over_poly = parse_expr(&mut ctx, "exp(x^2)/x^2");
        let polynomial_times_decaying_polynomial_exp = parse_expr(&mut ctx, "x^2*exp(2 - x^4)");
        let polynomial_over_decaying_polynomial_exp = parse_expr(&mut ctx, "x/exp(-x^2)");
        let neg_scaled_polynomial_exp_over_poly = parse_expr(&mut ctx, "-2*exp(x^2)/x^2");
        let parametric_exp_over_poly = parse_expr(&mut ctx, "exp(a*x^2)/x");
        let nested_exp_over_poly = parse_expr(&mut ctx, "exp(exp(x^2))/x");
        let even_poly_over_exp_neg = parse_expr(&mut ctx, "x^2/exp(x)");
        let odd_poly_over_exp_neg = parse_expr(&mut ctx, "x/exp(x)");
        let neg_scaled_exp_den_neg = parse_expr(&mut ctx, "x^2/(-2*exp(x))");
        let zero_scaled_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(x))");
        let zero_scaled_linear_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(2*x))");

        let exp_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
                .expect("exp dominates polynomial difference");
        let poly_minus_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_minus_exp, x, InfSign::Pos)
                .expect("negative exponential dominance");
        let poly_over_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_exp, x, InfSign::Pos)
                .expect("polynomial over exp");
        let exp_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_poly, x, InfSign::Pos)
                .expect("exp over polynomial");
        let poly_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_exp, x, InfSign::Neg)
                .expect("polynomial times decaying exp");
        let poly_over_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_linear_exp, x, InfSign::Pos)
                .expect("polynomial over linear exp");
        let linear_exp_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, linear_exp_over_poly, x, InfSign::Pos)
                .expect("linear exp over polynomial");
        let poly_times_decaying_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_linear_exp, x, InfSign::Neg)
                .expect("polynomial times decaying linear exp");
        let poly_over_decaying_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_decaying_linear_exp, x, InfSign::Pos)
                .expect("polynomial over decaying linear exp");
        let constant_over_decaying_linear_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            constant_over_decaying_linear_exp,
            x,
            InfSign::Pos,
        )
        .expect("constant over decaying linear exp");
        let even_poly_over_exp_neg_out =
            try_limit_rules_at_infinity(&mut ctx, even_poly_over_exp_neg, x, InfSign::Neg)
                .expect("even polynomial over decaying exp");
        let odd_poly_over_exp_neg_out =
            try_limit_rules_at_infinity(&mut ctx, odd_poly_over_exp_neg, x, InfSign::Neg)
                .expect("odd polynomial over decaying exp");
        let neg_scaled_exp_den_neg_out =
            try_limit_rules_at_infinity(&mut ctx, neg_scaled_exp_den_neg, x, InfSign::Neg)
                .expect("negative scaled exp denominator");
        let poly_over_polynomial_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_polynomial_exp, x, InfSign::Pos)
                .expect("polynomial over polynomial exp");
        let polynomial_exp_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_poly, x, InfSign::Pos)
                .expect("polynomial exp over polynomial");
        let polynomial_times_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            polynomial_times_decaying_polynomial_exp,
            x,
            InfSign::Pos,
        )
        .expect("polynomial times decaying polynomial exp");
        let polynomial_over_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            polynomial_over_decaying_polynomial_exp,
            x,
            InfSign::Neg,
        )
        .expect("polynomial over decaying polynomial exp");
        let neg_scaled_polynomial_exp_over_poly_out = try_limit_rules_at_infinity(
            &mut ctx,
            neg_scaled_polynomial_exp_over_poly,
            x,
            InfSign::Pos,
        )
        .expect("negative scaled polynomial exp over polynomial");

        assert!(matches!(
            ctx.get(exp_minus_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(poly_minus_exp_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(poly_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(exp_over_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(poly_over_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(linear_exp_over_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_times_decaying_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_decaying_linear_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(constant_over_decaying_linear_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(even_poly_over_exp_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(odd_poly_over_exp_neg_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(neg_scaled_exp_den_neg_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(poly_over_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(polynomial_exp_over_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(polynomial_times_decaying_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(polynomial_over_decaying_polynomial_exp_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(neg_scaled_polynomial_exp_over_poly_out),
            Expr::Neg(_)
        ));
        assert!(
            try_limit_rules_at_infinity(&mut ctx, parametric_exp_over_poly, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, nested_exp_over_poly, x, InfSign::Pos).is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_exp_den, x, InfSign::Pos).is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_linear_exp_den, x, InfSign::Pos)
                .is_none()
        );
    }

    #[test]
    fn subpolynomial_polynomial_dominance_handles_only_domain_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let log_over_poly = parse_expr(&mut ctx, "ln(x)/x");
        let poly_over_log = parse_expr(&mut ctx, "x/ln(x)");
        let root_over_poly = parse_expr(&mut ctx, "sqrt(x)/x");
        let poly_over_root = parse_expr(&mut ctx, "x/sqrt(x)");
        let cbrt_over_poly = parse_expr(&mut ctx, "cbrt(x)/x");
        let poly_over_cbrt = parse_expr(&mut ctx, "x/cbrt(x)");
        let poly_over_neg_tail_cbrt = parse_expr(&mut ctx, "x/cbrt(1 - x)");
        let asinh_over_poly = parse_expr(&mut ctx, "asinh(x)/x");
        let poly_over_asinh = parse_expr(&mut ctx, "x/asinh(x)");
        let poly_over_neg_tail_asinh = parse_expr(&mut ctx, "x/asinh(1 - x)");
        let acosh_over_poly = parse_expr(&mut ctx, "acosh(x)/x");
        let poly_over_acosh = parse_expr(&mut ctx, "x/acosh(x)");
        let poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(x^2)/x");
        let shifted_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(x^2 - 3)/x");
        let poly_over_poly_arg_acosh = parse_expr(&mut ctx, "x/acosh(x^2)");
        let neg_tail_acosh_over_even_poly = parse_expr(&mut ctx, "acosh(1 - x)/x^2");
        let neg_tail_poly_over_acosh = parse_expr(&mut ctx, "x/acosh(1 - x)");
        let base_log_over_poly = parse_expr(&mut ctx, "log(2, x)/x");
        let unary_log10_over_poly = parse_expr(&mut ctx, "log10(x)/x");
        let poly_arg_log_over_poly = parse_expr(&mut ctx, "ln(x^2)/x");
        let shifted_poly_arg_log_over_poly = parse_expr(&mut ctx, "ln(x^2 - 3)/x");
        let poly_over_poly_arg_log = parse_expr(&mut ctx, "x/ln(x^2)");
        let base_log_poly_arg_over_poly = parse_expr(&mut ctx, "log(2, x^2)/x");
        let poly_over_half_base_log = parse_expr(&mut ctx, "x/log(1/2, x)");
        let e_base_log_over_poly = parse_expr(&mut ctx, "log(e, x)/x");
        let pi_base_log_over_poly = parse_expr(&mut ctx, "log(pi, x)/x");
        let phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi, x)/x");
        let poly_over_reciprocal_e_base_log = parse_expr(&mut ctx, "x/log(1/e, x)");
        let powered_e_base_log_over_poly = parse_expr(&mut ctx, "log(e^2, x)/x");
        let powered_phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi^3, x)/x");
        let poly_over_negative_power_e_base_log = parse_expr(&mut ctx, "x/log(e^-2, x)");
        let poly_over_reciprocal_power_pi_base_log = parse_expr(&mut ctx, "x/log((1/pi)^2, x)");
        let neg_tail_log_over_poly = parse_expr(&mut ctx, "ln(1 - x)/x^2");
        let neg_tail_poly_over_log = parse_expr(&mut ctx, "x/ln(1 - x)");
        let log_minus_poly = parse_expr(&mut ctx, "ln(x) - x");
        let poly_minus_root = parse_expr(&mut ctx, "x - sqrt(x)");
        let bad_domain_log = parse_expr(&mut ctx, "ln(x)/x");
        let bad_domain_base_log = parse_expr(&mut ctx, "log(2, x)/x");
        let invalid_base_log = parse_expr(&mut ctx, "log(1, x)/x");
        let negative_named_base_log = parse_expr(&mut ctx, "log(-e, x)/x");
        let zero_power_named_base_log = parse_expr(&mut ctx, "log(e^0, x)/x");
        let negative_tail_poly_log = parse_expr(&mut ctx, "ln(3 - x^2)/x");
        let parametric_leading_tail_poly_log = parse_expr(&mut ctx, "ln(a*x^2 + 1)/x");
        let nonlinear_cbrt_over_poly = parse_expr(&mut ctx, "cbrt(x^2)/x");
        let poly_over_nonlinear_cbrt = parse_expr(&mut ctx, "x/cbrt(x^2)");
        let nonlinear_asinh_over_poly = parse_expr(&mut ctx, "asinh(x^2)/x");
        let poly_over_nonlinear_asinh = parse_expr(&mut ctx, "x/asinh(x^2)");
        let bad_domain_acosh = parse_expr(&mut ctx, "acosh(x)/x");
        let bad_domain_neg_tail_acosh = parse_expr(&mut ctx, "x/acosh(1 - x)");
        let negative_tail_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(3 - x^2)/x");
        let poly_over_negative_tail_poly_arg_acosh = parse_expr(&mut ctx, "x/acosh(3 - x^2)");
        let parametric_tail_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(a*x^2 + 1)/x");
        let subpoly_over_subpoly = parse_expr(&mut ctx, "ln(x)/sqrt(x)");
        let zero_scaled_log_den = parse_expr(&mut ctx, "x/(0*ln(x))");

        let log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, log_over_poly, x, InfSign::Pos)
                .expect("log over polynomial");
        let poly_over_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_log, x, InfSign::Pos)
                .expect("polynomial over log");
        let root_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, root_over_poly, x, InfSign::Pos)
                .expect("root over polynomial");
        let poly_over_root_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_root, x, InfSign::Pos)
                .expect("polynomial over root");
        let cbrt_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, cbrt_over_poly, x, InfSign::Pos)
                .expect("cube root over polynomial");
        let poly_over_cbrt_pos_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_cbrt, x, InfSign::Pos)
                .expect("polynomial over positive-tail cube root");
        let poly_over_cbrt_neg_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_cbrt, x, InfSign::Neg)
                .expect("polynomial over negative-tail cube root");
        let poly_over_neg_tail_cbrt_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_neg_tail_cbrt, x, InfSign::Pos)
                .expect("polynomial over negative linear-tail cube root");
        let asinh_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, asinh_over_poly, x, InfSign::Pos)
                .expect("asinh over polynomial");
        let poly_over_asinh_pos_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_asinh, x, InfSign::Pos)
                .expect("polynomial over positive-tail asinh");
        let poly_over_asinh_neg_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_asinh, x, InfSign::Neg)
                .expect("polynomial over negative-tail asinh");
        let poly_over_neg_tail_asinh_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_neg_tail_asinh, x, InfSign::Pos)
                .expect("polynomial over negative linear-tail asinh");
        let acosh_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, acosh_over_poly, x, InfSign::Pos)
                .expect("acosh over polynomial");
        let poly_over_acosh_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_acosh, x, InfSign::Pos)
                .expect("polynomial over positive-tail acosh");
        let poly_arg_acosh_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, poly_arg_acosh_over_poly, x, InfSign::Pos)
                .expect("polynomial-argument acosh over polynomial");
        let shifted_poly_arg_acosh_over_poly_out = try_limit_rules_at_infinity(
            &mut ctx,
            shifted_poly_arg_acosh_over_poly,
            x,
            InfSign::Pos,
        )
        .expect("shifted polynomial-argument acosh over polynomial");
        let poly_over_poly_arg_acosh_pos_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_acosh, x, InfSign::Pos)
                .expect("polynomial over polynomial-argument acosh");
        let poly_over_poly_arg_acosh_neg_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_acosh, x, InfSign::Neg)
                .expect("negative-approach polynomial over polynomial-argument acosh");
        let neg_tail_acosh_over_even_poly_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_acosh_over_even_poly, x, InfSign::Neg)
                .expect("negative-approach acosh over even polynomial");
        let neg_tail_poly_over_acosh_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_poly_over_acosh, x, InfSign::Neg)
                .expect("negative-approach polynomial over acosh");
        let base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, base_log_over_poly, x, InfSign::Pos)
                .expect("general-base log over polynomial");
        let unary_log10_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, unary_log10_over_poly, x, InfSign::Pos)
                .expect("log10 over polynomial");
        let poly_arg_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, poly_arg_log_over_poly, x, InfSign::Pos)
                .expect("polynomial-argument log over polynomial");
        let shifted_poly_arg_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, shifted_poly_arg_log_over_poly, x, InfSign::Pos)
                .expect("shifted polynomial-argument log over polynomial");
        let poly_over_poly_arg_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_log, x, InfSign::Pos)
                .expect("polynomial over polynomial-argument log");
        let base_log_poly_arg_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, base_log_poly_arg_over_poly, x, InfSign::Pos)
                .expect("base log with polynomial argument over polynomial");
        let poly_over_half_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_half_base_log, x, InfSign::Pos)
                .expect("polynomial over base < 1 log");
        let e_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, e_base_log_over_poly, x, InfSign::Pos)
                .expect("e-base log over polynomial");
        let pi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, pi_base_log_over_poly, x, InfSign::Pos)
                .expect("pi-base log over polynomial");
        let phi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, phi_base_log_over_poly, x, InfSign::Pos)
                .expect("phi-base log over polynomial");
        let poly_over_reciprocal_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_reciprocal_e_base_log, x, InfSign::Pos)
                .expect("polynomial over reciprocal e-base log");
        let powered_e_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, powered_e_base_log_over_poly, x, InfSign::Pos)
                .expect("powered e-base log over polynomial");
        let powered_phi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, powered_phi_base_log_over_poly, x, InfSign::Pos)
                .expect("powered phi-base log over polynomial");
        let poly_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            poly_over_negative_power_e_base_log,
            x,
            InfSign::Pos,
        )
        .expect("polynomial over negative powered e-base log");
        let poly_over_reciprocal_power_pi_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            poly_over_reciprocal_power_pi_base_log,
            x,
            InfSign::Pos,
        )
        .expect("polynomial over reciprocal power pi-base log");
        let neg_tail_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_log_over_poly, x, InfSign::Neg)
                .expect("negative-tail log over polynomial");
        let neg_tail_poly_over_log_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_poly_over_log, x, InfSign::Neg)
                .expect("negative-tail polynomial over log");
        let log_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, log_minus_poly, x, InfSign::Pos)
                .expect("log minus polynomial");
        let poly_minus_root_out =
            try_limit_rules_at_infinity(&mut ctx, poly_minus_root, x, InfSign::Pos)
                .expect("polynomial minus root");

        assert!(
            matches!(ctx.get(log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(root_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_root_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(cbrt_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_cbrt_pos_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(poly_over_cbrt_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(poly_over_neg_tail_cbrt_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(asinh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_asinh_pos_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(poly_over_asinh_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(poly_over_neg_tail_asinh_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_acosh_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_arg_acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(shifted_poly_arg_acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_poly_arg_acosh_pos_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(poly_over_poly_arg_acosh_neg_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(neg_tail_acosh_over_even_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(neg_tail_poly_over_acosh_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(unary_log10_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(poly_arg_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(shifted_poly_arg_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_poly_arg_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(base_log_poly_arg_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(poly_over_half_base_log_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(pi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_reciprocal_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(powered_e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(powered_phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_negative_power_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(poly_over_reciprocal_power_pi_base_log_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(neg_tail_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(neg_tail_poly_over_log_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(log_minus_poly_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(poly_minus_root_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_log, x, InfSign::Neg).is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_base_log, x, InfSign::Neg).is_none()
        );
        let invalid_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, invalid_base_log, x, InfSign::Pos)
                .expect("invalid-base log has empty real domain");
        assert_eq!(display_expr(&ctx, invalid_base_log_out), "undefined");
        let negative_named_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, negative_named_base_log, x, InfSign::Pos)
                .expect("negative-base log has empty real domain");
        assert_eq!(display_expr(&ctx, negative_named_base_log_out), "undefined");
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_power_named_base_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, negative_tail_poly_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            parametric_leading_tail_poly_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, nonlinear_cbrt_over_poly, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, poly_over_nonlinear_cbrt, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, nonlinear_asinh_over_poly, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, poly_over_nonlinear_asinh, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_acosh, x, InfSign::Neg).is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_neg_tail_acosh, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            negative_tail_poly_arg_acosh_over_poly,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            poly_over_negative_tail_poly_arg_acosh,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            parametric_tail_poly_arg_acosh_over_poly,
            x,
            InfSign::Pos
        )
        .is_none());
        // ln(x)/sqrt(x): the subpolynomial/polynomial rule declines (sqrt is
        // not an integer-degree polynomial), but the polylog/power dominance
        // rule now resolves it to 0 (a positive power dominates the logarithm).
        let subpoly_over_subpoly_out =
            try_limit_rules_at_infinity(&mut ctx, subpoly_over_subpoly, x, InfSign::Pos)
                .expect("ln(x)/sqrt(x) resolves via polylog/power dominance");
        assert_eq!(display_expr(&ctx, subpoly_over_subpoly_out), "0");
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_log_den, x, InfSign::Pos).is_none()
        );
    }

    #[test]
    fn polylog_power_dominance_at_infinity_resolves_fractional_and_higher_log() {
        // A positive power of x dominates any power of the logarithm:
        // ln(x)^a / x^b -> 0 and x^b / ln(x)^a -> +inf, for fractional b and
        // higher log powers a that the subpolynomial/polynomial rule misses.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("ln(x)/sqrt(x)", "0"),
            ("ln(x)/x^(1/3)", "0"),
            ("ln(x)^2/x", "0"),
            ("ln(x)^3/x", "0"),
            ("ln(x)^2/x^2", "0"),
            ("ln(x)^2/sqrt(x)", "0"),
            ("sqrt(x)/ln(x)", "infinity"),
            ("x/ln(x)^2", "infinity"),
            // Negated power numerator (top-level Neg) flips the sign.
            ("-x/ln(x)", "-infinity"),
            ("-sqrt(x)/ln(x)^2", "-infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("polylog/power dominance must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn bounded_noise_rational_quotient_at_infinity_uses_leading_ratio() {
        // A polynomial plus bounded additive noise has the polynomial's
        // growth: (x + sin x)/x -> 1, the noise is dominated.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("(x + sin(x))/x", "1"),
            ("(2*x + cos(x))/x", "2"),
            ("(x^2 + sin(x))/(x^2 - 1)", "1"),
            ("(x + sin(x))/(2*x + 1)", "1/2"),
            ("x/(x + sin(x))", "1"),
            ("(x + cos(x))/x^2", "0"),
            ("(x^2 + sin(x))/x", "infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("bounded-noise quotient must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn general_base_exponential_and_inf_to_zero_power_at_infinity() {
        // b^x growth and the inf^0 form, which together close (2^x+3^x)^(1/x)=3.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("2^x", "infinity"),
            ("(1/2)^x", "0"),
            ("2^(-x)", "0"),
            ("1^x", "1"),
            ("(2^x+3^x)^(1/x)", "3"),
            ("(2^x+3^x+5^x)^(1/x)", "5"),
            ("x^(1/x)", "1"),
            ("(x^2+1)^(1/x)", "1"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve at +inf: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
        // At -infinity, 2^x -> 0.
        let two_x = parse_expr(&mut ctx, "2^x");
        let out =
            try_limit_rules_at_infinity(&mut ctx, two_x, x, InfSign::Neg).expect("2^x at -inf");
        assert_eq!(display_expr(&ctx, out), "0");
    }

    #[test]
    fn inf_to_zero_power_declines_non_divergent_base() {
        // The base must diverge to +inf (positive, ln real). x^x (base -> inf
        // but exp does not -> 0) and a 1^inf base are NOT inf^0.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["x^x", "(1+1/x)^x"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                inf_to_zero_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "inf^0 must decline: {source}"
            );
        }
    }

    #[test]
    fn log_exp_sum_dominance_resolves_to_log_of_dominant_base() {
        // ln(sum of exponentials)/x -> ln(dominant effective base).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("ln(2^x+3^x)/x", "ln(3)"),
            ("ln(2^x+3^x+5^x)/x", "ln(5)"),
            ("ln(2^x+1)/x", "ln(2)"),
            ("ln(5^x-3^x)/x", "ln(5)"),
            ("ln(2^(2*x)+3^x)/x", "ln(4)"),
            ("ln(2*5^x-5^x)/x", "ln(5)"),
            ("ln(3*2^x+3^x)/(2*x)", "1/2 * ln(3)"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = log_exp_sum_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("log-exp-sum must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn log_exp_sum_dominance_declines_unsound_and_out_of_class() {
        // Negative dominant coefficient (sum -> -inf, ln undefined), an exact
        // dominant cancellation, a higher-order denominator, an e-base sum, and
        // a non-exponential argument all stay residual.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in [
            "ln(3^x-5^x)/x",
            "ln(5^x-5^x+3^x)/x",
            "ln(2^x+3^x)/x^2",
            "ln(exp(x)+exp(2*x))/x",
            "ln(x^2+1)/x",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                log_exp_sum_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "log-exp-sum must decline: {source}"
            );
        }
    }

    #[test]
    fn log_difference_at_infinity_collapses_to_log_of_ratio() {
        // ln(P) - ln(Q) -> ln(lim P/Q): a finite ln(lc_P/lc_Q) when degrees
        // match, +inf when P outgrows Q, -inf when Q outgrows P.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("ln(x+1) - ln(x)", "0"),
            ("ln(2*x) - ln(x)", "ln(2)"),
            ("ln(x^2+x) - ln(x^2-x)", "0"),
            ("ln(3*x+1) - ln(x)", "ln(3)"),
            ("ln(x) - ln(2*x)", "ln(1/2)"),
            ("ln(x^2) - ln(x)", "infinity"),
            ("ln(x) - ln(x^2)", "-infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("log difference must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn log_difference_at_infinity_declines_non_polynomial_or_negative() {
        // A non-polynomial log argument, a negative-leading argument (ln
        // undefined as x -> +inf), and the wrong approach must decline.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["ln(sin(x)) - ln(x)", "ln(-x) - ln(x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                log_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "log difference must decline: {source}"
            );
        }
        let neg_approach = parse_expr(&mut ctx, "ln(x+1) - ln(x)");
        assert!(
            log_difference_limit_at_infinity(&mut ctx, neg_approach, x, InfSign::Neg).is_none(),
            "x -> -inf makes ln undefined, so the rule must decline"
        );
    }

    #[test]
    fn bounded_noise_rational_quotient_declines_unbounded_noise() {
        // x*sin(x) is unbounded, so (x + x sin x)/x has no limit and must
        // stay residual; a pure poly/poly ratio is left to the exact rule.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let unbounded = parse_expr(&mut ctx, "(x + x*sin(x))/x");
        assert!(
            bounded_noise_rational_limit_at_infinity(&mut ctx, unbounded, x, InfSign::Pos)
                .is_none(),
            "unbounded noise must decline"
        );
        let pure_poly = parse_expr(&mut ctx, "(x^2 + 1)/(2*x^2 - 3)");
        assert!(
            bounded_noise_rational_limit_at_infinity(&mut ctx, pure_poly, x, InfSign::Pos)
                .is_none(),
            "pure poly/poly is left to the exact rational rule"
        );
    }

    #[test]
    fn polylog_power_dominance_declines_non_dominating_shapes() {
        // Not a polylog-over-power (or vice versa): a zero scale, a log/log
        // ratio, an oscillating factor, and the left (x -> -inf) approach
        // where the logarithm is undefined.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["x/(0*ln(x))", "ln(x)/ln(x)", "sin(x)/x^(1/2)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                polylog_power_dominance_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "polylog/power dominance must decline: {source}"
            );
        }
        let log_over_sqrt = parse_expr(&mut ctx, "ln(x)/sqrt(x)");
        assert!(
            polylog_power_dominance_limit_at_infinity(&mut ctx, log_over_sqrt, x, InfSign::Neg)
                .is_none(),
            "ln(x) is undefined as x -> -inf, so the rule must decline"
        );
    }

    #[test]
    fn exponential_subpolynomial_dominance_handles_only_domain_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(x)");
        let exp_over_log = parse_expr(&mut ctx, "exp(x)/ln(x)");
        let root_times_decaying_exp = parse_expr(&mut ctx, "sqrt(x)*exp(-x)");
        let cbrt_times_decaying_exp = parse_expr(&mut ctx, "cbrt(x)*exp(-x)");
        let asinh_times_decaying_exp = parse_expr(&mut ctx, "asinh(x)*exp(-x)");
        let acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(x)*exp(-x)");
        let poly_arg_acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(x^2)*exp(-x)");
        let log_over_decaying_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
        let cbrt_over_decaying_exp = parse_expr(&mut ctx, "cbrt(1 - x)/exp(-x)");
        let asinh_over_decaying_exp = parse_expr(&mut ctx, "asinh(1 - x)/exp(-x)");
        let neg_tail_acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(1 - x)*exp(x)");
        let negative_log_over_decaying_exp = parse_expr(&mut ctx, "-ln(x)/exp(-x)");
        let exp_over_negative_root = parse_expr(&mut ctx, "exp(x)/(-sqrt(x))");
        let exp_over_neg_tail_cbrt = parse_expr(&mut ctx, "exp(x)/cbrt(1 - x)");
        let exp_over_neg_tail_asinh = parse_expr(&mut ctx, "exp(x)/asinh(1 - x)");
        let exp_over_acosh = parse_expr(&mut ctx, "exp(x)/acosh(x)");
        let base_log_over_exp = parse_expr(&mut ctx, "log(2, x)/exp(x)");
        let exp_over_half_base_log = parse_expr(&mut ctx, "exp(x)/log(1/2, x)");
        let exp_over_unary_log2 = parse_expr(&mut ctx, "exp(x)/log2(x)");
        let exp_over_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e, x)");
        let exp_over_reciprocal_e_base_log = parse_expr(&mut ctx, "exp(x)/log(1/e, x)");
        let exp_over_powered_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^2, x)");
        let exp_over_negative_power_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^-2, x)");
        let exp_minus_log = parse_expr(&mut ctx, "exp(x) - ln(x)");
        let log_minus_exp = parse_expr(&mut ctx, "ln(x) - exp(x)");
        let neg_tail_log_times_decaying_exp = parse_expr(&mut ctx, "ln(1 - x)*exp(x)");
        let bad_domain_log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
        let invalid_base_log_over_exp = parse_expr(&mut ctx, "log(1, x)/exp(x)");
        let negative_named_base_log_over_exp = parse_expr(&mut ctx, "log(-e, x)/exp(x)");
        let polynomial_exp_over_log = parse_expr(&mut ctx, "exp(x^2)/ln(x)");
        let polynomial_exp_over_cbrt = parse_expr(&mut ctx, "exp(x^2)/cbrt(x)");
        let log_over_polynomial_exp = parse_expr(&mut ctx, "ln(x)/exp(x^2)");
        let root_times_decaying_polynomial_exp = parse_expr(&mut ctx, "sqrt(x)*exp(2 - x^4)");
        let polynomial_exp_over_negative_root = parse_expr(&mut ctx, "exp(x^2)/(-sqrt(x))");
        let parametric_polynomial_exp_over_log = parse_expr(&mut ctx, "exp(a*x^2)/ln(x)");
        let nested_polynomial_exp_over_log = parse_expr(&mut ctx, "exp(exp(x^2))/ln(x)");
        let exp_over_nonlinear_cbrt = parse_expr(&mut ctx, "exp(x)/cbrt(x^2)");
        let exp_over_nonlinear_asinh = parse_expr(&mut ctx, "exp(x)/asinh(x^2)");
        let bad_domain_exp_over_acosh = parse_expr(&mut ctx, "exp(x)/acosh(1 - x)");
        let exp_over_poly_arg_acosh = parse_expr(&mut ctx, "exp(x)/acosh(x^2)");
        let exp_over_negative_tail_poly_arg_acosh = parse_expr(&mut ctx, "exp(x)/acosh(3 - x^2)");
        let zero_exp_denominator = parse_expr(&mut ctx, "ln(x)/(0*exp(x))");

        let log_over_exp_out = try_limit_rules_at_infinity(&mut ctx, log_over_exp, x, InfSign::Pos)
            .expect("log over growing exp");
        let exp_over_log_out = try_limit_rules_at_infinity(&mut ctx, exp_over_log, x, InfSign::Pos)
            .expect("growing exp over log");
        let root_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, root_times_decaying_exp, x, InfSign::Pos)
                .expect("root times decaying exp");
        let cbrt_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, cbrt_times_decaying_exp, x, InfSign::Pos)
                .expect("cube root times decaying exp");
        let asinh_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, asinh_times_decaying_exp, x, InfSign::Pos)
                .expect("asinh times decaying exp");
        let acosh_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, acosh_times_decaying_exp, x, InfSign::Pos)
                .expect("acosh times decaying exp");
        let poly_arg_acosh_times_decaying_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            poly_arg_acosh_times_decaying_exp,
            x,
            InfSign::Pos,
        )
        .expect("polynomial-argument acosh times decaying exp");
        let log_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, log_over_decaying_exp, x, InfSign::Pos)
                .expect("log over decaying exp");
        let cbrt_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, cbrt_over_decaying_exp, x, InfSign::Pos)
                .expect("negative-tail cube root over decaying exp");
        let asinh_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, asinh_over_decaying_exp, x, InfSign::Pos)
                .expect("negative-tail asinh over decaying exp");
        let neg_tail_acosh_times_decaying_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            neg_tail_acosh_times_decaying_exp,
            x,
            InfSign::Neg,
        )
        .expect("negative-approach acosh times decaying exp");
        let negative_log_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, negative_log_over_decaying_exp, x, InfSign::Pos)
                .expect("negative log over decaying exp");
        let exp_over_negative_root_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_negative_root, x, InfSign::Pos)
                .expect("exp over negative root");
        let exp_over_neg_tail_cbrt_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_neg_tail_cbrt, x, InfSign::Pos)
                .expect("exp over negative-tail cube root");
        let exp_over_neg_tail_asinh_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_neg_tail_asinh, x, InfSign::Pos)
                .expect("exp over negative-tail asinh");
        let exp_over_acosh_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_acosh, x, InfSign::Pos)
                .expect("exp over positive-tail acosh");
        let exp_over_poly_arg_acosh_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_poly_arg_acosh, x, InfSign::Pos)
                .expect("exp over polynomial-argument acosh");
        let base_log_over_exp_out =
            try_limit_rules_at_infinity(&mut ctx, base_log_over_exp, x, InfSign::Pos)
                .expect("general-base log over growing exp");
        let exp_over_half_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_half_base_log, x, InfSign::Pos)
                .expect("growing exp over base < 1 log");
        let exp_over_unary_log2_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_unary_log2, x, InfSign::Pos)
                .expect("growing exp over log2");
        let exp_over_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_e_base_log, x, InfSign::Pos)
                .expect("growing exp over e-base log");
        let exp_over_reciprocal_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_reciprocal_e_base_log, x, InfSign::Pos)
                .expect("growing exp over reciprocal e-base log");
        let exp_over_powered_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_powered_e_base_log, x, InfSign::Pos)
                .expect("growing exp over powered e-base log");
        let exp_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            exp_over_negative_power_e_base_log,
            x,
            InfSign::Pos,
        )
        .expect("growing exp over negative powered e-base log");
        let exp_minus_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_log, x, InfSign::Pos)
                .expect("exp minus log");
        let log_minus_exp_out =
            try_limit_rules_at_infinity(&mut ctx, log_minus_exp, x, InfSign::Pos)
                .expect("log minus exp");
        let neg_tail_log_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_log_times_decaying_exp, x, InfSign::Neg)
                .expect("negative-tail log times decaying exp");
        let polynomial_exp_over_log_out =
            try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_log, x, InfSign::Pos)
                .expect("polynomial exp over log");
        let polynomial_exp_over_cbrt_out =
            try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_cbrt, x, InfSign::Pos)
                .expect("polynomial exp over cube root");
        let log_over_polynomial_exp_out =
            try_limit_rules_at_infinity(&mut ctx, log_over_polynomial_exp, x, InfSign::Pos)
                .expect("log over polynomial exp");
        let root_times_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            root_times_decaying_polynomial_exp,
            x,
            InfSign::Pos,
        )
        .expect("root times decaying polynomial exp");
        let polynomial_exp_over_negative_root_out = try_limit_rules_at_infinity(
            &mut ctx,
            polynomial_exp_over_negative_root,
            x,
            InfSign::Pos,
        )
        .expect("polynomial exp over negative root");

        assert!(
            matches!(ctx.get(log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(exp_over_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(root_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(cbrt_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(asinh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(poly_arg_acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(log_over_decaying_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(cbrt_over_decaying_exp_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(asinh_over_decaying_exp_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(neg_tail_acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(negative_log_over_decaying_exp_out),
            Expr::Neg(_)
        ));
        assert!(matches!(ctx.get(exp_over_negative_root_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(exp_over_neg_tail_cbrt_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(exp_over_neg_tail_asinh_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(exp_over_acosh_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_poly_arg_acosh_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(base_log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(exp_over_half_base_log_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(exp_over_unary_log2_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_e_base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_reciprocal_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(exp_over_powered_e_base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_negative_power_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(exp_minus_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(log_minus_exp_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(neg_tail_log_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(polynomial_exp_over_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(polynomial_exp_over_cbrt_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(log_over_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(root_times_decaying_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(polynomial_exp_over_negative_root_out),
            Expr::Neg(_)
        ));
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_log_over_exp, x, InfSign::Neg)
                .is_none()
        );
        let invalid_base_log_over_exp_out =
            try_limit_rules_at_infinity(&mut ctx, invalid_base_log_over_exp, x, InfSign::Pos)
                .expect("invalid-base log over exp has empty real domain");
        assert_eq!(
            display_expr(&ctx, invalid_base_log_over_exp_out),
            "undefined"
        );
        let negative_named_base_log_over_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            negative_named_base_log_over_exp,
            x,
            InfSign::Pos,
        )
        .expect("negative-base log over exp has empty real domain");
        assert_eq!(
            display_expr(&ctx, negative_named_base_log_over_exp_out),
            "undefined"
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            parametric_polynomial_exp_over_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            nested_polynomial_exp_over_log,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, exp_over_nonlinear_cbrt, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, exp_over_nonlinear_asinh, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_exp_over_acosh, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            exp_over_negative_tail_poly_arg_acosh,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_exp_denominator, x, InfSign::Pos).is_none()
        );
    }

    #[test]
    fn apply_power_rule_handles_zero_and_negative_exponents() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let x0 = parse_expr(&mut ctx, "x^0");
        let xneg = parse_expr(&mut ctx, "x^-3");

        let out0 = apply_power_rule(&mut ctx, x0, x, InfSign::Pos).expect("x^0");
        let out_neg = apply_power_rule(&mut ctx, xneg, x, InfSign::Neg).expect("x^-3");

        assert!(
            matches!(ctx.get(out0), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
        );
        assert!(
            matches!(ctx.get(out_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn apply_rational_power_rule_resolves_fractional_exponents_at_positive_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let neg_half = parse_expr(&mut ctx, "x^(-1/2)");
        let pos_half = parse_expr(&mut ctx, "x^(1/2)");
        let three_halves = parse_expr(&mut ctx, "x^(3/2)");

        let out_neg =
            apply_rational_power_rule(&mut ctx, neg_half, x, InfSign::Pos).expect("x^(-1/2) -> 0");
        let out_pos =
            apply_rational_power_rule(&mut ctx, pos_half, x, InfSign::Pos).expect("x^(1/2) -> ∞");
        let out_three = apply_rational_power_rule(&mut ctx, three_halves, x, InfSign::Pos)
            .expect("x^(3/2) -> ∞");

        assert!(
            matches!(ctx.get(out_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(out_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(out_three),
            Expr::Constant(Constant::Infinity)
        ));
    }

    #[test]
    fn apply_rational_power_rule_declines_integers_and_negative_infinity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        // Integer exponents stay with `apply_power_rule`.
        let int_exp = parse_expr(&mut ctx, "x^2");
        assert!(apply_rational_power_rule(&mut ctx, int_exp, x, InfSign::Pos).is_none());

        // A non-integer power of a negative magnitude is not real-valued.
        let frac = parse_expr(&mut ctx, "x^(1/2)");
        assert!(apply_rational_power_rule(&mut ctx, frac, x, InfSign::Neg).is_none());

        // Base must be exactly the limit variable.
        let other = parse_expr(&mut ctx, "y^(1/2)");
        assert!(apply_rational_power_rule(&mut ctx, other, x, InfSign::Pos).is_none());
    }

    #[test]
    fn apply_reciprocal_power_rule_handles_one_over_xn() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr1 = parse_expr(&mut ctx, "1/x");
        let expr2 = parse_expr(&mut ctx, "5/x^3");

        let out1 = apply_reciprocal_power_rule(&mut ctx, expr1, x).expect("1/x");
        let out2 = apply_reciprocal_power_rule(&mut ctx, expr2, x).expect("5/x^3");

        assert!(
            matches!(ctx.get(out1), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(out2), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn try_limit_rules_at_infinity_resolves_constant_and_variable() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let c = parse_expr(&mut ctx, "7");

        let c_out = try_limit_rules_at_infinity(&mut ctx, c, x, InfSign::Pos).expect("constant");
        let x_out = try_limit_rules_at_infinity(&mut ctx, x, x, InfSign::Neg).expect("variable");

        assert_eq!(c_out, c);
        assert!(matches!(ctx.get(x_out), Expr::Neg(_)));
    }

    #[test]
    fn try_limit_rules_at_infinity_uses_rational_poly_fallback() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "x^2/x^3");

        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("rational");
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn exp_sum_quotient_dominance_resolves_by_dominant_base() {
        // A quotient of exponential sums is decided by the dominant base.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("3^x/2^x", "infinity"),
            ("2^x/3^x", "0"),
            ("(2^x+3^x)/3^x", "1"),
            ("(3^x-2^x)/(3^x+2^x)", "1"),
            ("(2*3^x+2^x)/3^x", "2"),
            ("(5^x+2^x)/(3^x+4^x)", "infinity"),
            ("(-3^x)/2^x", "-infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = exp_sum_quotient_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("exp quotient must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn general_exp_vs_polynomial_dominance_resolves() {
        // An exponential (base > 1) beats any polynomial in a quotient.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("2^x/x^2", "infinity"),
            ("x^10/2^x", "0"),
            ("2^x/x", "infinity"),
            ("(2^x+3^x)/x^5", "infinity"),
            ("x^3/(2^x+3^x)", "0"),
            ("(-2^x)/x^2", "-infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out =
                general_exp_vs_polynomial_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .unwrap_or_else(|| panic!("exp-vs-poly must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn polynomial_times_decaying_exponential_collapses_to_zero() {
        // A polynomial times a decaying exponential -> 0 (decay beats growth).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, approach) in [
            ("x*2^(-x)", InfSign::Pos),
            ("x^3*2^(-x)", InfSign::Pos),
            ("x^2*(1/2)^x", InfSign::Pos),
            ("x*2^x", InfSign::Neg),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out =
                polynomial_times_decaying_exponential_at_infinity(&mut ctx, expr, x, approach)
                    .unwrap_or_else(|| panic!("poly*decaying-exp must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), "0", "{source}");
        }
        // A GROWING exponential factor is not this rule (x*2^x at +inf -> +inf).
        let grow = parse_expr(&mut ctx, "x*2^x");
        assert!(
            polynomial_times_decaying_exponential_at_infinity(&mut ctx, grow, x, InfSign::Pos)
                .is_none(),
            "a growing exponential is not the decaying-product rule"
        );
    }

    #[test]
    fn general_exp_vs_polynomial_dominance_declines_pure_rational_and_decaying() {
        // Both polynomial (no exponential), and a decaying base (<= 1) which is
        // not a growing exponential, stay out of this rule.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["x^2/x^3", "(1/2)^x/x^2"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                general_exp_vs_polynomial_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "exp-vs-poly must decline: {source}"
            );
        }
    }

    #[test]
    fn exp_sum_quotient_dominance_declines_non_exponential_and_cancelled() {
        // No growing exponential on a side (pure rational, exp-vs-poly) and a
        // cancelled dominant denominator coefficient stay out of this rule.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["(x^2+1)/(x+1)", "2^x/x^2", "3^x/(2^x+3^x-3^x)"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                exp_sum_quotient_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "exp quotient must decline: {source}"
            );
        }
    }

    #[test]
    fn rational_difference_at_infinity_combines_into_one_fraction() {
        // inf - inf of rational functions: the per-term additive rule leaves it
        // residual; combining over a common denominator resolves it.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("(x^2+1)/(x+1) - x", "-1"),
            ("x^2/(x-1) - x", "1"),
            ("(x^3+1)/(x^2+1) - x", "0"),
            ("x - x^2/(x+1)", "1"),
            ("x^2/(x+1) - x^2/(x+2)", "1"),
            ("(2*x^2)/(x+1) - 2*x", "-2"),
            ("x^3/(x+1) - x", "infinity"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = rational_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("rational difference must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn rational_difference_at_infinity_declines_non_rational_operands() {
        // Non-rational operands (sqrt/sin/exp) make the multipoly conversion
        // fail, so the conjugate/elementary/dominance paths keep their forms.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["sqrt(x^2+1) - x", "sin(x) - x", "exp(x) - x"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                rational_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "rational difference must decline non-rational operands: {source}"
            );
        }
    }

    #[test]
    fn one_to_infinity_power_resolves_the_e_family() {
        // 1^inf: base -> 1, exponent -> inf, limit = exp(lim exp*(base-1)).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("(1+1/x)^x", "e"),
            ("(1+2/x)^x", "e^2"),
            ("(1+1/x)^(2*x)", "e^2"),
            ("(1+3/x)^(2*x)", "e^6"),
            ("(1-1/x)^x", "e^(-1)"),
            ("(1+1/x^2)^x", "1"),
            ("(1+1/x)^(x^2)", "infinity"),
            ("((2*x+1)/(2*x-1))^x", "e"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = one_to_infinity_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("1^inf must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn one_to_infinity_power_declines_non_indeterminate_and_oscillating() {
        // Honest declines: a constant base (not -> 1), a base -> inf (x^x), a
        // FINITE exponent (1^5 = 1 is continuous, not indeterminate), and an
        // oscillating base-1 whose product has no limit.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in ["2^x", "x^x", "(1+1/x)^5", "(1+sin(x)/x)^x"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                one_to_infinity_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
                "1^inf must decline: {source}"
            );
        }
    }

    #[test]
    fn product_log_unit_argument_resolves_inf_times_zero() {
        // inf * 0 with ln(f), f -> 1: lim g*ln(f) = lim g*(f-1) (ln(1+h) ~ h).
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for (source, expected) in [
            ("x*ln(1+1/x)", "1"),
            ("x*ln(1+2/x)", "2"),
            ("x*ln((x+1)/x)", "1"),
            ("x*ln(1-1/x)", "-1"),
            ("x^2*ln(1+1/x^2)", "1"),
            ("ln(1+1/x)*x", "1"),
            ("x*ln(1+1/x^2)", "0"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = product_log_unit_argument_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("inf*0 ln must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn product_log_unit_argument_declines_non_unit_arg_and_finite_cofactor() {
        // Honest declines: ln argument -> inf (x ln x), a constant argument
        // (ln(2) x, arg = 2 != 1), a non-divergent cofactor (the continuous
        // finite*0 case), and an oscillating reduction with no limit.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        for source in [
            "x*ln(x)",
            "ln(2)*x",
            "(1/x)*ln(1+1/x)",
            "x*ln(1+sin(1/x)/x)",
        ] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                product_log_unit_argument_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "inf*0 ln must decline: {source}"
            );
        }
    }

    #[test]
    fn finite_one_to_infinity_power_resolves_the_e_definition() {
        // 1^inf at a finite point: (1+x)^(1/x) = e, the textbook definition.
        // The product limit is computed by the full finite machinery, so
        // SECOND-ORDER cases (cos(x)^(1/x^2) = e^(-1/2)) resolve correctly.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [
            ("(1+x)^(1/x)", "e"),
            ("(1+2*x)^(1/x)", "e^2"),
            ("(1+x)^(3/x)", "e^3"),
            ("(1-x)^(1/x)", "e^(-1)"),
            ("(1+sin(x))^(1/x)", "e"),
            ("cos(x)^(1/x^2)", "e^(-1/2)"),
        ] {
            let expr = parse_expr(&mut ctx, source);
            let out = apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, zero)
                .unwrap_or_else(|| panic!("finite 1^inf must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_zero_base_power_resolves_x_to_the_x() {
        // The 0^0 form x^g -> exp(lim g ln x) at 0+: x^x = exp(lim x ln x) = 1.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        for (source, expected) in [("x^x", "1"), ("x^(2*x)", "1"), ("x^(x^2)", "1")] {
            let expr = parse_expr(&mut ctx, source);
            let out =
                apply_finite_zero_base_power_rule(&mut ctx, expr, x, zero, FiniteLimitSide::Right)
                    .unwrap_or_else(|| panic!("0^0 must resolve: {source}"));
            assert_eq!(display_expr(&ctx, out), expected, "{source}");
        }
    }

    #[test]
    fn finite_zero_base_power_declines_left_side_and_non_variable_base() {
        // Honest declines: the LEFT side (x^x is complex for x < 0), a
        // non-variable base (sign unknown), and a nonzero point.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let one = parse_expr(&mut ctx, "1");
        let xx = parse_expr(&mut ctx, "x^x");
        assert!(
            apply_finite_zero_base_power_rule(&mut ctx, xx, x, zero, FiniteLimitSide::Left)
                .is_none(),
            "0^0 is real only from the right of 0"
        );
        let sinx = parse_expr(&mut ctx, "sin(x)^x");
        assert!(
            apply_finite_zero_base_power_rule(&mut ctx, sinx, x, zero, FiniteLimitSide::Right)
                .is_none(),
            "a non-variable base has unknown sign"
        );
        assert!(
            apply_finite_zero_base_power_rule(&mut ctx, xx, x, one, FiniteLimitSide::Right)
                .is_none(),
            "0^0 is the form at 0, not at 1"
        );
    }

    #[test]
    fn finite_one_to_infinity_power_declines_continuous_and_non_unit_base() {
        // Honest declines: a base not -> 1 (x^x, (2+x)^(1/x)), a zero product
        // limit (the continuous 1^finite case (1+x)^x and (1+x^2)^(1/x)), and
        // a constant exponent.
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let zero = parse_expr(&mut ctx, "0");
        let one = parse_expr(&mut ctx, "1");
        for source in ["x^x", "(2+x)^(1/x)", "(1+x)^x", "(1+x^2)^(1/x)", "(1+x)^2"] {
            let expr = parse_expr(&mut ctx, source);
            assert!(
                apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, zero).is_none(),
                "finite 1^inf must decline: {source}"
            );
        }
        // At x = 1 the base (1+x) -> 2, not 1: not the indeterminate form.
        let expr = parse_expr(&mut ctx, "(1+x)^(1/x)");
        assert!(
            apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, one).is_none(),
            "finite 1^inf needs a unit base limit"
        );
    }

    #[test]
    fn try_limit_rules_at_infinity_uses_polynomial_growth_before_residual() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "2*x^3 + x");

        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("polynomial");

        assert!(matches!(ctx.get(out), Expr::Constant(Constant::Infinity)));
    }

    #[test]
    fn bounded_elementary_over_divergent_limit_at_infinity_resolves_to_zero() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let cases = [
            "sin(x)/x",
            "cos(2*x + 1)/(x^2 + 1)",
            "(2*sin(x) - cos(x))/(-x)",
            "sin(x)*cos(x)/exp(x)",
            "arctan(x)/x",
            "atan(x^2 + 1)/(x^2 + 1)",
            "(arctan(x) + sin(x))/(0 - x)",
            "tanh(x)/x",
            "tanh(x^2 + 1)/(x^2 + 1)",
            "(tanh(x) - cos(x))/exp(x)",
            "sin(sqrt(x))/x",
            "sin(ln(x))/x",
        ];

        for expr in cases {
            let parsed = parse_expr(&mut ctx, expr);
            let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("expected bounded-over-divergent zero for {expr}"));
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "expected zero for {expr}, got {:?}",
                ctx.get(out)
            );
        }

        for expr in ["sin(sqrt(-x))/x", "sin(ln(-x))/x"] {
            let parsed = parse_expr(&mut ctx, expr);
            let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Neg)
                .unwrap_or_else(|| {
                    panic!("expected negative-infinity bounded-over-divergent zero for {expr}")
                });
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "expected zero for {expr}, got {:?}",
                ctx.get(out)
            );
        }
    }

    #[test]
    fn bounded_elementary_over_divergent_limit_at_infinity_rejects_unbounded_or_nondominant_shapes()
    {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let unbounded_num = parse_expr(&mut ctx, "x*sin(x)/x");
        let nondominant_den = parse_expr(&mut ctx, "sin(x)/cos(x)");
        let arctan_nondominant_den = parse_expr(&mut ctx, "arctan(x)/cos(x)");
        let tanh_nondominant_den = parse_expr(&mut ctx, "tanh(x)/cos(x)");
        let sqrt_domain_conflict = parse_expr(&mut ctx, "sin(sqrt(1 - x))/x");
        let log_domain_conflict = parse_expr(&mut ctx, "sin(ln(1 - x))/x");
        let neg_sqrt_domain_conflict = parse_expr(&mut ctx, "sin(sqrt(x))/x");
        let neg_log_domain_conflict = parse_expr(&mut ctx, "sin(ln(x))/x");

        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            unbounded_num,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            arctan_nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            tanh_nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            sqrt_domain_conflict,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            log_domain_conflict,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            neg_sqrt_domain_conflict,
            x,
            InfSign::Neg,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            neg_log_domain_conflict,
            x,
            InfSign::Neg,
        )
        .is_none());
    }

    #[test]
    fn bounded_elementary_times_decaying_exp_limit_at_infinity_resolves_to_zero() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let cases = [
            "sin(x)*exp(-x)",
            "exp(-2*x)*cos(x)",
            "sin(x)*exp(2 - x^4)",
            "exp(-x^2)*cos(x)",
            "(sin(x) + cos(x))*exp(-x)",
            "(sin(x) + cos(x))*exp(1 - x^2)",
            "arctan(x)*exp(-x)",
            "arctan(x)*exp(2 - x^4)",
            "-tanh(x)*exp(-x)",
            "-tanh(x)*exp(-x^2)",
            "sin(sqrt(x))*exp(-x)",
            "sin(ln(x))*exp(-x)",
        ];

        for expr in cases {
            let parsed = parse_expr(&mut ctx, expr);
            let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("expected bounded-times-decaying-exp zero for {expr}"));
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "expected zero for {expr}, got {:?}",
                ctx.get(out)
            );
        }

        let tan_product = parse_expr(&mut ctx, "tan(x)*exp(-x)");
        let bad_sqrt_domain = parse_expr(&mut ctx, "sin(sqrt(1 - x))*exp(-x)");
        let bad_log_domain = parse_expr(&mut ctx, "sin(ln(1 - x))*exp(-x)");
        let bad_poly_exp_sqrt_domain = parse_expr(&mut ctx, "sin(sqrt(1 - x))*exp(2 - x^4)");
        let parametric_exp_tail = parse_expr(&mut ctx, "sin(x)*exp(a*x^2)");
        let nested_exp_tail = parse_expr(&mut ctx, "sin(x)*exp(exp(0 - x))");
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            tan_product,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            bad_sqrt_domain,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            bad_log_domain,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            bad_poly_exp_sqrt_domain,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            parametric_exp_tail,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            nested_exp_tail,
            x,
            InfSign::Pos,
        )
        .is_none());
    }

    #[test]
    fn presimplify_safe_for_limit_applies_allowlisted_rewrites() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "x + 0");
        let out = presimplify_safe_for_limit(&mut ctx, expr);
        assert_eq!(out, x);
    }

    #[test]
    fn presimplify_safe_for_limit_does_not_apply_domain_sensitive_rewrites() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x/x");
        let out = presimplify_safe_for_limit(&mut ctx, expr);
        assert!(matches!(ctx.get(out), Expr::Div(_, _)));
    }

    #[test]
    fn reciprocal_substitution_resolves_notable_infinity_limits() {
        // `lim_{x→∞} g(x) = lim_{u→0⁺} g(1/u)`: the notable products `x·f(c/x)` the direct ∞ rules
        // miss. The artifact reducer turns the substituted `f(1/(1/x))·(1/x)` into `f(x)/x`.
        for (src, expected) in [
            ("x*sin(1/x)", "1"),
            ("x*sin(3/x)", "3"),
            ("x*tan(1/x)", "1"),
            ("x*arctan(1/x)", "1"),
            ("x*(exp(1/x)-1)", "1"),
            ("2*x*sin(1/x)", "2"),
        ] {
            let mut ctx = Context::new();
            let expr = parse_expr(&mut ctx, src);
            let x = ctx.var("x");
            let result =
                try_limit_at_infinity_by_reciprocal_substitution(&mut ctx, expr, x, InfSign::Pos)
                    .unwrap_or_else(|| panic!("must resolve: {src}"));
            assert_eq!(display_expr(&ctx, result), expected, "{src}");
        }
        // Genuinely limitless oscillators must decline — the substitution must not fabricate a value.
        for src in ["x*sin(x)", "sin(x)"] {
            let mut ctx = Context::new();
            let expr = parse_expr(&mut ctx, src);
            let x = ctx.var("x");
            assert!(
                try_limit_at_infinity_by_reciprocal_substitution(&mut ctx, expr, x, InfSign::Pos)
                    .is_none(),
                "{src} has no limit and must decline"
            );
        }
    }

    #[test]
    fn abs_wrapped_unbounded_arguments_resolve_at_both_infinities() {
        let mut ctx = Context::new();
        let cases = [
            ("ln(|x|)", InfSign::Pos, "infinity"),
            ("ln(|x|)", InfSign::Neg, "infinity"),
            ("|x^2+1|", InfSign::Pos, "infinity"),
        ];
        for (source, approach, expected) in cases {
            let expr = cas_parser::parse(source, &mut ctx).expect(source);
            let var = ctx.var("x");
            let result = try_limit_rules_at_infinity(&mut ctx, expr, var, approach)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_eq!(
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: result
                    }
                ),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn one_sided_composition_resolves_endpoint_combinations() {
        let mut ctx = Context::new();
        let cases = [
            ("x*ln(x)", "0"),
            ("x*ln(x) - x", "0"),
            ("2*sqrt(x)", "0"),
            ("sqrt(x)*ln(x)", "0"),
            ("3*ln(x)", "-infinity"),
            // Power atom: (x - 0)^q -> 0 from the right for rational q > 0.
            ("x^(3/2)", "0"),
            ("x^(1/3)", "0"),
            // Product of two variable factors (no constant cofactor).
            ("sqrt(x)*x", "0"),
            // Division by a foldable rational constant.
            ("x^(1/3 + 1) / (1/3 + 1)", "0"),
            ("ln(x)/2", "-infinity"),
        ];
        for (source, expected) in cases {
            let expr = cas_parser::parse(source, &mut ctx).expect(source);
            let var = ctx.var("x");
            let zero = ctx.num(0);
            let result = try_limit_rules_at_finite_one_sided(
                &mut ctx,
                expr,
                var,
                zero,
                FiniteLimitSide::Right,
            )
            .unwrap_or_else(|| panic!("must resolve: {source}"));
            assert_eq!(
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: result
                    }
                ),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn one_sided_composition_refuses_indeterminate_forms() {
        let mut ctx = Context::new();
        // -infinity + infinity and the left side of the log domain.
        for source in ["ln(x) + 1/x", "ln(x) - ln(x)"] {
            let expr = cas_parser::parse(source, &mut ctx).expect(source);
            let var = ctx.var("x");
            let zero = ctx.num(0);
            assert!(
                try_limit_rules_at_finite_one_sided(
                    &mut ctx,
                    expr,
                    var,
                    zero,
                    FiniteLimitSide::Right,
                )
                .is_none(),
                "must refuse: {source}"
            );
        }
    }

    #[test]
    fn one_sided_product_combination_guards_indeterminate_signs() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let two = ctx.num(2);
        let infinity = ctx.add(Expr::Constant(Constant::Infinity));
        let neg_infinity = ctx.add(Expr::Neg(infinity));
        let pi = ctx.add(Expr::Constant(Constant::Pi));

        // 0 * infinity is indeterminate.
        assert!(combine_limit_product(&mut ctx, zero, infinity).is_none());
        // Symbolic finite cofactor: sign unknown, refuse.
        assert!(combine_limit_product(&mut ctx, pi, infinity).is_none());
        // Numeric nonzero cofactor decides the sign.
        let scaled = combine_limit_product(&mut ctx, two, neg_infinity).expect("signed");
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: scaled
                }
            ),
            "-infinity"
        );
        // infinity * -infinity carries the product sign.
        let crossed = combine_limit_product(&mut ctx, infinity, neg_infinity).expect("signed");
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: crossed
                }
            ),
            "-infinity"
        );
        // Finite * finite folds numerically.
        let folded = combine_limit_product(&mut ctx, two, two).expect("folded");
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: folded
                }
            ),
            "4"
        );
    }
}
