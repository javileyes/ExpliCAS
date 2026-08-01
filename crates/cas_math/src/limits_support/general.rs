//! `limits_support`: familia `general`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

/// Create a residual limit expression: `limit(expr, var, approach_symbol)`.
pub(crate) fn mk_limit(ctx: &mut Context, expr: ExprId, var: ExprId, approach: InfSign) -> ExprId {
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
pub(crate) fn mk_limit_for_approach(
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

/// Rule 1: Constant - lim c = c (if `expr` doesn't depend on `var`).
pub(crate) fn apply_constant_rule(ctx: &Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !depends_on(ctx, expr, var) {
        Some(expr)
    } else {
        None
    }
}

pub(super) fn apply_static_empty_real_domain_rule(
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

/// Finite-point 0/0 limit of `(scale*sqrt(a x + b) + k) / den(x)` where
/// the radical numerator vanishes at the point (so the form is 0/0):
/// rationalize by the conjugate. (scale sqrt + k)(scale sqrt - k) =
/// scale^2 (a x + b) - k^2 is a polynomial, so the quotient becomes
/// [scale^2(ax+b) - k^2] / [den(x) (scale sqrt + k - wait, conjugate)].
/// The polynomial part is a removable rational hole and the conjugate
/// is continuous (nonzero) at the point. Covers (sqrt(x)-2)/(x-4)=1/4
/// and (sqrt(x+1)-2)/(x-3)=1/4.
pub(super) fn apply_finite_radical_conjugate_rule(
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
pub(super) fn apply_finite_radical_difference_conjugate_rule(
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

pub(super) fn apply_finite_bilateral_sqrt_endpoint_rule(
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

/// Pairwise-coprime refinement (factor refinement) of a set of integers > 1:
/// repeatedly split any non-coprime pair `a, b` into `gcd, a/gcd, b/gcd`
/// until all elements are pairwise coprime. Needs NO primality testing, and
/// terminates because each split strictly divides the product of elements.
/// Every input factors completely over the result, and pairwise-coprime
/// integers > 1 have Q-linearly independent logarithms (each element owns a
/// prime the others lack) — the backbone of the exact zero decision below.
pub(super) fn coprime_refinement(values: &[BigInt]) -> Vec<BigInt> {
    use num_integer::Integer;
    let one = BigInt::one();
    let mut set: Vec<BigInt> = values.iter().filter(|v| **v > one).cloned().collect();
    loop {
        set.sort();
        set.dedup();
        let mut split: Option<(usize, usize, BigInt)> = None;
        'scan: for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                let g = set[i].gcd(&set[j]);
                if g > one {
                    split = Some((i, j, g));
                    break 'scan;
                }
            }
        }
        let Some((i, j, g)) = split else {
            return set;
        };
        let a = &set[i] / &g;
        let b = &set[j] / &g;
        set.remove(j);
        set.remove(i);
        for piece in [g, a, b] {
            if piece > one {
                set.push(piece);
            }
        }
    }
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
pub(super) fn first_order_equivalent_poly(
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

pub(super) fn is_finite_total_real_unary_builtin(builtin: BuiltinFn) -> bool {
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

pub(super) fn is_finite_partial_domain_unary_builtin(builtin: BuiltinFn) -> bool {
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

pub(super) fn finite_total_real_unary_result(
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

fn finite_abs_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    if finite_expr_proven_positive(ctx, argument_limit) {
        return Some(argument_limit);
    }

    let Expr::Neg(inner) = ctx.get(argument_limit).clone() else {
        return None;
    };
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

pub(super) fn finite_partial_domain_unary_result(
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

pub(super) fn pow_one_third_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

pub(super) fn apply_finite_total_real_unary_composition_rule(
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

pub(super) fn apply_finite_partial_domain_unary_composition_rule(
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
pub(super) fn apply_finite_squeeze_bounded_product_rule(
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
pub(super) fn apply_finite_bilateral_even_saturating_pole_rule(
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

pub(super) fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
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

/// Recursively replace any constant subexpression with its literal value. The
/// symbolic differentiator emits unfolded arithmetic — notably exponents like
/// `(x-1)^(2-1)` — that `Polynomial::from_expr` and the continuous-limit rules
/// reject; folding `2-1` to `1` (via `as_rational_const`, which evaluates
/// arithmetic, unlike literal-only matchers) makes the differentiated form
/// consumable. Structure with the variable is preserved.
/// Fold every fully-numeric subtree to its exact rational literal (`x^(2-1)` →
/// `x^1` → handled by the Pow arm, `6-3·2` → `0`). Public: raw-derivative
/// consumers (Taylor, the potential verifier) need it before polynomial
/// conversion — `multipoly_from_expr` rejects non-literal exponents.
pub fn fold_constant_subexprs(ctx: &mut Context, expr: ExprId) -> ExprId {
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
        // f^(k)(point) / k!  — the substituted tree is fully constant; fold it to a
        // compact rational NOW, or the raw k-th-derivative tree (exponential in k)
        // overflows the simplifier's depth budget at order ≥ 6 and the whole series
        // leaks as an unfolded raw tree.
        let value = cas_ast::substitute_expr_by_id(ctx, derivative, var_id, point);
        let value = fold_constant_subexprs(ctx, value);
        let value = tidy_taylor_units(ctx, value);
        // F1 (Fase 3): a coefficient that is not a real finite value means the
        // function is NOT smooth at the point — the definition does not apply.
        // Decline to the honest residual instead of emitting a series that the
        // simplifier collapses to `undefined` (`taylor(ln(x),x,0,2)` used to
        // ANSWER `undefined`; the answer is "no Maclaurin expansion", an echo).
        if taylor_coefficient_is_singular(ctx, value) {
            return None;
        }
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
            // Fold the chain-rule constant litter each iteration: the raw k-th derivative
            // tree grows exponentially in k otherwise, and its unfolded `u^(2-1)·0`-style
            // subtrees are also what collapsed arc-function expansions to `undefined`.
            derivative = fold_constant_subexprs(ctx, derivative);
            derivative = tidy_taylor_units(ctx, derivative);
            factorial *= BigInt::from(k + 1);
        }
    }
    sum
}

/// Bottom-up tidy pass for Taylor derivative/coefficient trees, applied AFTER
/// [`fold_constant_subexprs`] on the Taylor routes only (the L'Hôpital route
/// keeps the plain fold — its 209-case lane pins exact shapes): the generic
/// `a^u` derivative rule litters `ln(e)` factors that the plain fold cannot
/// fold (`ln(e)` is not a rational literal), and at high orders the chains
/// (`ln(e)·ln(e)·…`) push the emitted series past the simplifier's depth
/// budget, leaking unsimplified output. Rewrites `ln(e) → 1`, drops unit
/// factors, and unwraps `base^1`.
fn tidy_taylor_units(ctx: &mut Context, expr: ExprId) -> ExprId {
    let one_test = |ctx: &Context, e: ExprId| {
        crate::numeric_eval::as_rational_const(ctx, e).is_some_and(|v| v.is_one())
    };
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Ln)
                && args.len() == 1
                && matches!(ctx.get(args[0]), Expr::Constant(Constant::E)) =>
        {
            ctx.num(1)
        }
        Expr::Function(fn_id, args) => {
            let args2: Vec<ExprId> = args.iter().map(|a| tidy_taylor_units(ctx, *a)).collect();
            ctx.add(Expr::Function(fn_id, args2))
        }
        Expr::Mul(l, r) => {
            let l2 = tidy_taylor_units(ctx, l);
            let r2 = tidy_taylor_units(ctx, r);
            if one_test(ctx, l2) {
                return r2;
            }
            if one_test(ctx, r2) {
                return l2;
            }
            ctx.add(Expr::Mul(l2, r2))
        }
        Expr::Pow(base, exp) => {
            use num_traits::Zero;
            let base2 = tidy_taylor_units(ctx, base);
            let exp2 = tidy_taylor_units(ctx, exp);
            if one_test(ctx, exp2) {
                return base2;
            }
            // `e^0 → 1` (a substituted `e^(x+y)` at the origin): safe for a
            // base that is provably nonzero — E and nonzero rationals cover
            // every shape the Taylor substitution produces.
            let exp_is_zero =
                crate::numeric_eval::as_rational_const(ctx, exp2).is_some_and(|v| v.is_zero());
            if exp_is_zero {
                let base_nonzero = matches!(ctx.get(base2), Expr::Constant(Constant::E))
                    || crate::numeric_eval::as_rational_const(ctx, base2)
                        .is_some_and(|v| !v.is_zero());
                if base_nonzero {
                    return ctx.num(1);
                }
            }
            ctx.add(Expr::Pow(base2, exp2))
        }
        Expr::Add(l, r) => {
            let l2 = tidy_taylor_units(ctx, l);
            let r2 = tidy_taylor_units(ctx, r);
            ctx.add(Expr::Add(l2, r2))
        }
        Expr::Sub(l, r) => {
            let l2 = tidy_taylor_units(ctx, l);
            let r2 = tidy_taylor_units(ctx, r);
            ctx.add(Expr::Sub(l2, r2))
        }
        Expr::Div(l, r) => {
            let l2 = tidy_taylor_units(ctx, l);
            let r2 = tidy_taylor_units(ctx, r);
            if one_test(ctx, r2) {
                return l2;
            }
            ctx.add(Expr::Div(l2, r2))
        }
        Expr::Neg(inner) => {
            let i2 = tidy_taylor_units(ctx, inner);
            ctx.add(Expr::Neg(i2))
        }
        _ => expr,
    }
}

/// Number of multi-indices of dimension `d` with total degree ≤ `order`:
/// `C(order+d, d)`. `None` on overflow (which certainly exceeds the cap).
fn multivar_term_count(order: usize, d: usize) -> Option<u128> {
    let mut acc: u128 = 1;
    for i in 1..=d {
        acc = acc.checked_mul((order + i) as u128)? / (i as u128);
    }
    Some(acc)
}

/// Multivariate Taylor expansion by TOTAL degree (F2, Fase 3):
/// `Σ_{|α| ≤ order} ∂^α f(a) / α! · Π (xᵢ − aᵢ)^αᵢ`, built incrementally —
/// each ∂^α derives from its parent `α − e_i` (first nonzero slot), folding the
/// chain-rule constant litter per step (the same discipline the univariate
/// definitional route needs to stay bounded). All-or-nothing: any failing
/// derivative or singular coefficient (see [`taylor_coefficient_is_singular`])
/// declines the whole expansion. Terms with a folded rational-zero coefficient
/// are skipped for display quality; symbolic coefficients are kept.
pub fn taylor_multivar_series_expr(
    ctx: &mut Context,
    target: ExprId,
    var_names: &[String],
    points: &[ExprId],
    order: usize,
) -> Option<ExprId> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};
    use std::collections::HashMap;

    let d = var_names.len();
    if d == 0 || d > TAYLOR_MULTIVAR_MAX_VARS || points.len() != d {
        return None;
    }
    if multivar_term_count(order, d)? > TAYLOR_MULTIVAR_MAX_TERMS {
        return None;
    }
    let var_ids: Vec<ExprId> = var_names.iter().map(|v| ctx.var(v)).collect();

    // Level-by-level derivative table: level k maps α (|α| = k) to ∂^α f.
    let mut level: HashMap<Vec<u32>, ExprId> = HashMap::new();
    level.insert(vec![0u32; d], target);
    let mut sum: Option<ExprId> = None;
    for k in 0..=(order as u32) {
        for alpha in multi_indices_of_degree(d, k) {
            let derivative = *level.get(&alpha)?;
            // Coefficient: substitute the expansion point into ∂^α f, fold, and
            // divide by α!.
            let mut value = derivative;
            for (var_id, point) in var_ids.iter().zip(points) {
                value = cas_ast::substitute_expr_by_id(ctx, value, *var_id, *point);
            }
            let value = fold_constant_subexprs(ctx, value);
            let value = tidy_taylor_units(ctx, value);
            if taylor_coefficient_is_singular(ctx, value) {
                return None;
            }
            let is_zero_coeff =
                crate::numeric_eval::as_rational_const(ctx, value).is_some_and(|v| v.is_zero());
            if !is_zero_coeff {
                let mut factorial = BigInt::one();
                for &a in &alpha {
                    for i in 2..=a {
                        factorial *= BigInt::from(i);
                    }
                }
                let factorial_expr = ctx.add(Expr::Number(BigRational::from_integer(factorial)));
                let mut term = ctx.add(Expr::Div(value, factorial_expr));
                // · Π (xᵢ − aᵢ)^αᵢ
                for ((var_id, point), &a) in var_ids.iter().zip(points).zip(&alpha) {
                    if a == 0 {
                        continue;
                    }
                    let base = if crate::numeric_eval::as_rational_const(ctx, *point)
                        .is_some_and(|p| p.is_zero())
                    {
                        *var_id
                    } else {
                        ctx.add(Expr::Sub(*var_id, *point))
                    };
                    let factor = if a == 1 {
                        base
                    } else {
                        let exponent = ctx.num(a as i64);
                        ctx.add(Expr::Pow(base, exponent))
                    };
                    term = ctx.add(Expr::Mul(term, factor));
                }
                sum = Some(match sum {
                    None => term,
                    Some(acc) => ctx.add(Expr::Add(acc, term)),
                });
            }
            // Seed the next level from this node: ∂^(α + e_i) for every i.
            if k < order as u32 {
                for i in 0..d {
                    let mut child = alpha.clone();
                    child[i] += 1;
                    if level.contains_key(&child) {
                        continue;
                    }
                    let dchild =
                        crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                            ctx,
                            derivative,
                            &var_names[i],
                        )?;
                    let dchild = fold_constant_subexprs(ctx, dchild);
                    let dchild = tidy_taylor_units(ctx, dchild);
                    level.insert(child, dchild);
                }
            }
        }
    }
    Some(sum.unwrap_or_else(|| ctx.num(0)))
}

/// True when a FOLDED Taylor coefficient (a constant tree — the expansion
/// variable is already substituted out) is provably singular or non-real:
/// division by a zero constant (`sin(0)/0`), a logarithm of a constant `<= 0`
/// (`ln(0)` at the boundary, `ln(-1)` in the real domain), a zero base raised
/// to a negative constant power, a non-finite sentinel, or an imaginary value
/// (even root of a provably-negative constant). Symbolic coefficients (`e^y`
/// in a parametric expansion) are NOT flagged — only decidable constants are.
fn taylor_coefficient_is_singular(ctx: &Context, value: ExprId) -> bool {
    use num_traits::Signed;
    if crate::numeric_eval::expr_contains_imaginary(ctx, value) {
        return true;
    }
    // `as_rational_const` does not fold `Pow` (so `0^2` — the shape a
    // substituted `x^2` denominator takes — would slip through); recognise a
    // zero base under a positive constant exponent explicitly.
    fn is_zero_const(ctx: &Context, e: ExprId) -> bool {
        use num_traits::{Signed, Zero};
        if crate::numeric_eval::as_rational_const(ctx, e).is_some_and(|v| v.is_zero()) {
            return true;
        }
        match ctx.get(e) {
            Expr::Pow(base, exp) => {
                is_zero_const(ctx, *base)
                    && crate::numeric_eval::as_rational_const(ctx, *exp)
                        .is_some_and(|v| v.is_positive())
            }
            _ => false,
        }
    }
    let mut stack = vec![value];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(Constant::Undefined | Constant::Infinity) => return true,
            Expr::Div(l, r) => {
                if is_zero_const(ctx, *r) {
                    return true;
                }
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Pow(base, exp) => {
                if is_zero_const(ctx, *base)
                    && crate::numeric_eval::as_rational_const(ctx, *exp)
                        .is_some_and(|v| v.is_negative())
                {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && (ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Ln)
                        || ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Log)) =>
            {
                if crate::numeric_eval::as_rational_const(ctx, args[0])
                    .is_some_and(|v| !v.is_positive())
                {
                    return true;
                }
                stack.push(args[0]);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for a in args {
                    stack.push(*a);
                }
            }
            _ => {}
        }
    }
    false
}

/// Compose a standard analytic function with the already-expanded inner
/// series. The inner series must satisfy the function's expansion point
/// (0 for everything except ln, which needs 1).
pub(super) fn compose_standard_series(
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

pub(super) fn combine_limit_product(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
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

/// var - point structurally: the bare var at point 0, or Sub(var, point)
/// up to numeric equality of the point.
pub(super) fn is_var_shift(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> bool {
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

pub(super) fn try_limit_rules_at_finite_one_sided(
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
pub(crate) fn apply_variable_rule(
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

pub(super) fn negate_limit_result(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(sign) = infinity_sign_of_expr(ctx, expr) {
        return mk_infinity(ctx, neg_inf_sign(sign));
    }

    match ctx.get(expr).clone() {
        Expr::Number(value) => ctx.add(Expr::Number(-value)),
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn finite_numeric_expr(ctx: &mut Context, value: BigRational) -> ExprId {
    ctx.add(Expr::Number(value))
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

/// Like [`finite_add_result`] but declines the indeterminate OPPOSITE-sign
/// `∞ + (−∞)` (the additive twin of `finite_sub_result`'s same-sign guard),
/// so the caller can combine over a common denominator instead of building a
/// garbage `Add(∞, −∞)`. Same-sign `∞ + ∞` is a determinate ±∞ and falls
/// through.
pub(super) fn finite_add_result_checked(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
    if let (Some(lhs_sign), Some(rhs_sign)) = (
        limit_value_infinite_sign(ctx, lhs),
        limit_value_infinite_sign(ctx, rhs),
    ) {
        if lhs_sign != rhs_sign {
            return None;
        }
        // Same-sign `∞ + ∞ = ±∞`: return the (already ±∞) operand rather than
        // letting `finite_add_result` build a literal `Add(∞, ∞)` node.
        return Some(lhs);
    }
    Some(finite_add_result(ctx, lhs, rhs))
}

pub(super) fn finite_sub_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
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

pub(super) fn finite_mul_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
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

pub(super) fn finite_div_result(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
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

pub(super) fn finite_neg_result(ctx: &mut Context, inner: ExprId) -> ExprId {
    if let Some(value) = numeric_limit_value(ctx, inner) {
        return finite_numeric_expr(ctx, -value);
    }
    negate_limit_result(ctx, inner)
}

pub(super) fn scaled_abs_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
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

/// One orientation: `factor * diff`. Combines the factor's leading term with the
/// decaying difference's leading term; finite only when the exponents sum to 0.
pub(super) fn radical_conjugate_product_oriented(
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
pub(super) fn scaled_cube_root_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
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

pub(super) fn cbrt_conjugate_product_oriented(
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

/// `scale * (P)^(1/n)` -> (scale, radicand P, n) for an integer `n >= 2`.
pub(super) fn scaled_nth_root_pow_base(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId, u32)> {
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

pub(super) fn nth_root_conjugate_product_oriented(
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

pub(super) fn combine_add_limit_results(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
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

pub(super) fn combine_sub_limit_results(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
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

pub(super) fn combine_mul_limit_results(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
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

pub(super) fn combine_div_limit_results(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
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

/// Unwrap a single `abs(P)` wrapper, returning `P`; otherwise the expression unchanged. `ln|P|` and
/// `ln(P)` share the same `+inf` tail once `P -> +inf`, so the log-sum rule strips the `abs` first.
pub(super) fn strip_single_abs(ctx: &Context, expr: ExprId) -> ExprId {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs)) {
            return args[0];
        }
    }
    expr
}

/// Exact GAUSSIAN-rational evaluation of a fully-numeric tree: `(re, im)` as
/// `BigRational` pairs, with `i` as a first-class value (F11, Fase 3). `None`
/// for anything transcendental (E, Pi, function calls) — the complex selective
/// path only decides what it can prove exactly.
pub(super) fn eval_gaussian_const_deep(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational)> {
    use num_traits::{ToPrimitive, Zero};
    let zero = || BigRational::from_integer(0.into());
    match ctx.get(expr) {
        Expr::Number(n) => Some((n.clone(), zero())),
        Expr::Constant(Constant::I) => Some((zero(), BigRational::from_integer(1.into()))),
        Expr::Add(l, r) => {
            let (a, b) = eval_gaussian_const_deep(ctx, *l)?;
            let (c, d) = eval_gaussian_const_deep(ctx, *r)?;
            Some((a + c, b + d))
        }
        Expr::Sub(l, r) => {
            let (a, b) = eval_gaussian_const_deep(ctx, *l)?;
            let (c, d) = eval_gaussian_const_deep(ctx, *r)?;
            Some((a - c, b - d))
        }
        Expr::Mul(l, r) => {
            let (a, b) = eval_gaussian_const_deep(ctx, *l)?;
            let (c, d) = eval_gaussian_const_deep(ctx, *r)?;
            Some((a.clone() * c.clone() - b.clone() * d.clone(), a * d + b * c))
        }
        Expr::Div(l, r) => {
            let (a, b) = eval_gaussian_const_deep(ctx, *l)?;
            let (c, d) = eval_gaussian_const_deep(ctx, *r)?;
            let norm = c.clone() * c.clone() + d.clone() * d.clone();
            if norm.is_zero() {
                return None;
            }
            Some((
                (a.clone() * c.clone() + b.clone() * d.clone()) / norm.clone(),
                (b * c - a * d) / norm,
            ))
        }
        Expr::Neg(inner) => {
            let (a, b) = eval_gaussian_const_deep(ctx, *inner)?;
            Some((-a, -b))
        }
        Expr::Hold(inner) => eval_gaussian_const_deep(ctx, *inner),
        Expr::Pow(base, exp) => {
            let e = crate::numeric_eval::as_rational_const(ctx, *exp)?;
            if !e.is_integer() {
                return None;
            }
            let n = e.to_integer().to_i64().filter(|n| (0..=32).contains(n))?;
            let (mut ra, mut rb) = (BigRational::from_integer(1.into()), zero());
            let (a, b) = eval_gaussian_const_deep(ctx, *base)?;
            for _ in 0..n {
                let na = ra.clone() * a.clone() - rb.clone() * b.clone();
                let nb = ra * b.clone() + rb * a.clone();
                ra = na;
                rb = nb;
            }
            Some((ra, rb))
        }
        _ => None,
    }
}

/// ANALYTIC-shape check for the complex selective path (F11): the tree may use
/// polynomials over the variable, Gaussian constants, `exp/sin/cos/sinh/cosh`
/// with POLYNOMIAL (entire) arguments, and `Div` of such — a meromorphic
/// function with no essential singularities. Anything else (abs, conjugate,
/// tanh/atan, transcendental compositions like `e^(1/z²)` whose exp-argument
/// carries a pole) fails the shape and stays residual under complex.
fn expr_is_analytic_shape(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Signed;
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            expr_is_analytic_shape(ctx, *l) && expr_is_analytic_shape(ctx, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_is_analytic_shape(ctx, *inner),
        Expr::Pow(base, exp) => {
            // `e^g` (the parser normalizes `exp(g)` to `Pow(E, g)`): entire
            // iff g is entire — this is exactly what EXCLUDES `e^(-1/z²)`
            // (its exponent carries a pole → essential singularity).
            if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
                return expr_is_entire_polynomial_shape(ctx, *exp);
            }
            crate::numeric_eval::as_rational_const(ctx, *exp)
                .is_some_and(|e| e.is_integer() && !e.is_negative())
                && expr_is_analytic_shape(ctx, *base)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            use cas_ast::BuiltinFn;
            let entire = ctx.is_builtin(*fn_id, BuiltinFn::Exp)
                || ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                || ctx.is_builtin(*fn_id, BuiltinFn::Cosh);
            // The transcendental's ARGUMENT must itself be entire (no Div):
            // `e^(-1/z²)` has an essential singularity precisely because its
            // exp-argument carries a pole.
            entire && expr_is_entire_polynomial_shape(ctx, args[0])
        }
        _ => false,
    }
}

/// F11 (Fase 3): SELECTIVE complex re-grant inside the F0 kill-switch — only
/// for ANALYTIC shapes, decided exactly:
/// - Direct substitution at a GAUSSIAN point when every denominator proves
///   nonzero (meromorphic and analytic at the point): `sin(z)` at `i`,
///   `1/(z²+1)` at `2i`.
/// - Delegation to the REAL engine at a REAL rational point: the shape is
///   meromorphic with no essential singularities, so a FINITE real limit IS
///   the complex limit (`sin(z)/z → 1`, `(z²−1)/(z−1) → 2`); a real ±∞ marks
///   a pole (no complex ∞ under decision D7 — residual), and `undefined`
///   verdicts are NOT re-emitted (real-lateral reasoning does not transfer).
///
/// Everything else keeps the honest kill-switch residual — the 7 F0 WRONGs
/// remain never-fabricate pins by construction (their shapes fail here).
pub(super) fn try_complex_limit_selective(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> Option<ExprId> {
    use num_traits::Zero;
    let Approach::Finite(point) = approach else {
        return None;
    };
    if !expr_is_analytic_shape(ctx, expr) {
        return None;
    }
    // ENTIRE shape (no Div anywhere): continuous on all of ℂ — direct
    // substitution is valid at ANY finite point, including transcendental
    // ones like `i·π` (`exp(z)` at `i·π` → Euler folds to −1 downstream).
    if !expr_contains_div(ctx, expr) {
        if expr_contains_nonfinite_sentinel(ctx, point) {
            return None;
        }
        let mut value = cas_ast::substitute_expr_by_id(ctx, expr, var, point);
        value = fold_constant_subexprs(ctx, value);
        return Some(tidy_taylor_units(ctx, value));
    }
    let (_re, im) = eval_gaussian_const_deep(ctx, point)?;
    if !all_denominators_provably_nonzero_at(ctx, expr, var, point) {
        // Some denominator vanishes (or is undecidable) at the point. At a
        // REAL rational point, delegate to the real engine: meromorphy makes
        // a finite real limit the complex limit (removable singularity).
        if !im.is_zero() {
            return None;
        }
        let opts = LimitOptions::default();
        let outcome = eval_limit_at_infinity(ctx, expr, var, Approach::Finite(point), &opts);
        if outcome.warning.is_some() {
            return None;
        }
        if matches!(
            ctx.get(outcome.expr),
            Expr::Constant(Constant::Infinity | Constant::Undefined)
        ) {
            return None;
        }
        if matches!(ctx.get(outcome.expr), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
        {
            return None;
        }
        return Some(outcome.expr);
    }
    // Analytic at the point: the limit IS the substituted value.
    let mut value = cas_ast::substitute_expr_by_id(ctx, expr, var, point);
    value = fold_constant_subexprs(ctx, value);
    Some(tidy_taylor_units(ctx, value))
}

/// True when the tree contains any `Div` node.
fn expr_contains_div(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Div(_, _) => return true,
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            _ => {}
        }
    }
    false
}

/// True when the tree contains an `Infinity`/`Undefined` sentinel.
fn expr_contains_nonfinite_sentinel(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Constant(Constant::Infinity | Constant::Undefined) => return true,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            _ => {}
        }
    }
    false
}

pub(super) fn finite_residual_point(
    ctx: &Context,
    var: ExprId,
    point: ExprId,
) -> Option<FiniteResidualPoint> {
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

pub(super) fn finite_single_function_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    Some((ctx.builtin_of(fn_id)?, args[0]))
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

/// Classify a one-sided limit RESULT expression; `None` for anything the
/// combiner must not reason about (residual `limit(...)` calls, `undefined`,
/// `i`, or expressions still containing ∞ in a compound position).
fn classify_lateral_limit_result(ctx: &Context, expr: ExprId) -> Option<LateralLimitClass> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => return Some(LateralLimitClass::PosInfinity),
        Expr::Constant(Constant::Undefined) | Expr::Constant(Constant::I) => return None,
        Expr::Neg(inner) => {
            if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) {
                return Some(LateralLimitClass::NegInfinity);
            }
        }
        _ => {}
    }
    // Finite iff nothing exotic survives inside (a residual limit call, an
    // embedded infinity/undefined) — a plain finite value/constant tree.
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(Constant::Infinity | Constant::Undefined | Constant::I) => {
                return None;
            }
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "limit" {
                    return None;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { .. } | Expr::SessionRef(_) => return None,
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
        }
    }
    Some(LateralLimitClass::Finite(expr))
}

/// DNE by OSCILLATION: the WHOLE limit target is a bare `sin/cos/tan(g)` and a
/// lateral limit of the inner `g` at the point classifies as ±∞ — the trig of an
/// argument racing to ±∞ oscillates forever on that side (bounded for sin/cos,
/// pole-riddled for tan), so no lateral limit exists there and the bilateral
/// limit DOES NOT EXIST, unconditionally. Only the bare composition: a
/// bounded-oscillating FACTOR (`x·sin(1/x)`) belongs to the squeeze machinery
/// and resolves before this. Conservative: if neither lateral of `g` proves
/// infinite, decline — never fabricate a DNE from an unproven divergence.
pub(super) fn try_dne_by_oscillation(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<LimitEvalOutcome> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    let fname = match builtin {
        BuiltinFn::Sin => "sin",
        BuiltinFn::Cos => "cos",
        BuiltinFn::Tan => "tan",
        _ => return None,
    };
    let inner = args[0];
    let classify_side = |ctx: &mut Context, side: FiniteLimitSide| {
        try_limit_rules_at_finite_one_sided(ctx, inner, var, point, side)
            .and_then(|l| classify_lateral_limit_result(ctx, l))
    };
    let left = classify_side(ctx, FiniteLimitSide::Left);
    let right = classify_side(ctx, FiniteLimitSide::Right);
    use LateralLimitClass::{NegInfinity, PosInfinity};
    let side = match (left, right) {
        (Some(PosInfinity | NegInfinity), Some(PosInfinity | NegInfinity)) => "both sides",
        (Some(PosInfinity | NegInfinity), _) => "the left side",
        (_, Some(PosInfinity | NegInfinity)) => "the right side",
        _ => return None,
    };
    Some(LimitEvalOutcome {
        expr: ctx.add(Expr::Constant(Constant::Undefined)),
        warning: Some(format!(
            "the limit does not exist: {fname}(u) OSCILLATES — its inner argument diverges to ±∞ on {side} of the point, and {fname} has no limit at ±∞"
        )),
    })
}

/// Try to settle a BILATERAL finite-point limit from its two one-sided
/// limits. Returns `Some` only when both laterals compute to classifiable
/// results:
/// - both `+∞` / both `−∞` → the signed divergence;
/// - both finite and STRUCTURALLY equal (or equal exact rationals) → that
///   value (belt-and-braces: the direct rules usually catch these);
/// - both finite exact rationals that DIFFER, or any ±∞ mismatch → the
///   bilateral limit does not exist → `undefined`, with both laterals quoted
///   in the warning (the educational payload);
/// - anything else (symbolic finite pair not provably equal, a lateral that
///   is residual/undefined) → `None`: keep the honest bilateral residual —
///   never fabricate a DNE from an unproven inequality.
pub(super) fn try_bilateral_limit_from_lateral_agreement(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<LimitEvalOutcome> {
    let left = try_limit_rules_at_finite_one_sided(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right = try_limit_rules_at_finite_one_sided(ctx, expr, var, point, FiniteLimitSide::Right)?;
    let left_class = classify_lateral_limit_result(ctx, left)?;
    let right_class = classify_lateral_limit_result(ctx, right)?;

    use LateralLimitClass::{Finite, NegInfinity, PosInfinity};
    let describe = |ctx: &Context, class: LateralLimitClass| -> String {
        match class {
            PosInfinity => "+∞".to_string(),
            NegInfinity => "−∞".to_string(),
            Finite(e) => match crate::numeric_eval::as_rational_const(ctx, e) {
                Some(q) => q.to_string(),
                None => "a finite value".to_string(),
            },
        }
    };
    let dne = |ctx: &mut Context,
               lc: LateralLimitClass,
               rc: LateralLimitClass|
     -> LimitEvalOutcome {
        let l = describe(ctx, lc);
        let r = describe(ctx, rc);
        LimitEvalOutcome {
            expr: ctx.add(Expr::Constant(Constant::Undefined)),
            warning: Some(format!(
                "the bilateral limit does not exist: the one-sided limits disagree (left: {l}, right: {r})"
            )),
        }
    };
    match (left_class, right_class) {
        (PosInfinity, PosInfinity) => Some(LimitEvalOutcome {
            expr: ctx.add(Expr::Constant(Constant::Infinity)),
            warning: None,
        }),
        (NegInfinity, NegInfinity) => {
            let inf = ctx.add(Expr::Constant(Constant::Infinity));
            Some(LimitEvalOutcome {
                expr: ctx.add(Expr::Neg(inf)),
                warning: None,
            })
        }
        (PosInfinity, NegInfinity)
        | (NegInfinity, PosInfinity)
        | (PosInfinity | NegInfinity, Finite(_))
        | (Finite(_), PosInfinity | NegInfinity) => Some(dne(ctx, left_class, right_class)),
        (Finite(l), Finite(r)) => {
            if cas_ast::ordering::compare_expr(ctx, l, r) == std::cmp::Ordering::Equal {
                return Some(LimitEvalOutcome {
                    expr: l,
                    warning: None,
                });
            }
            let lv = crate::numeric_eval::as_rational_const(ctx, l);
            let rv = crate::numeric_eval::as_rational_const(ctx, r);
            match (lv, rv) {
                (Some(a), Some(b)) if a == b => Some(LimitEvalOutcome {
                    expr: l,
                    warning: None,
                }),
                // Exact rationals that differ: a PROVEN disagreement.
                (Some(_), Some(_)) => Some(dne(ctx, left_class, right_class)),
                // Symbolic finite pair not provably equal: stay residual.
                _ => None,
            }
        }
    }
}

pub(super) fn finite_residual_warning(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> String {
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

/// Multivariate limit by PROVEN continuity (F7, Fase 3): substitute the point
/// only when the expression is visibly continuous there — a tree of
/// `Add/Sub/Mul/Neg`, non-negative integer powers, continuous total-real unary
/// calls of such, and `Div` whose SUBSTITUTED denominator folds to a NONZERO
/// exact rational. Anything else returns `None` (honest residual): the
/// existence question for general multivariate limits is path-dependent and is
/// NOT decided here (F8 owns the negative side).
pub fn try_multivar_limit_by_continuity(
    ctx: &mut Context,
    expr: ExprId,
    var_ids: &[ExprId],
    points: &[ExprId],
) -> Option<ExprId> {
    if !expr_is_visibly_continuous_at(ctx, expr, var_ids, points) {
        return None;
    }
    let mut value = expr;
    for (var_id, point) in var_ids.iter().zip(points) {
        value = cas_ast::substitute_expr_by_id(ctx, value, *var_id, *point);
    }
    let value = fold_constant_subexprs(ctx, value);
    let value = tidy_taylor_units(ctx, value);
    Some(value)
}

/// Shape check behind [`try_multivar_limit_by_continuity`]: every node must be
/// continuous at the point, decided EXACTLY. `Div` (and negative integer
/// powers) require the substituted denominator to fold to a nonzero rational —
/// the `finite_denominator_proven_nonzero` doctrine, never f64.
fn expr_is_visibly_continuous_at(
    ctx: &mut Context,
    expr: ExprId,
    var_ids: &[ExprId],
    points: &[ExprId],
) -> bool {
    use num_traits::Signed;
    match ctx.get(expr).clone() {
        Expr::Number(_) | Expr::Variable(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            expr_is_visibly_continuous_at(ctx, l, var_ids, points)
                && expr_is_visibly_continuous_at(ctx, r, var_ids, points)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            expr_is_visibly_continuous_at(ctx, inner, var_ids, points)
        }
        Expr::Pow(base, exp) => {
            let Some(e) = crate::numeric_eval::as_rational_const(ctx, exp) else {
                return false;
            };
            if !e.is_integer() {
                return false;
            }
            if e.is_negative() {
                // base^(-m) = 1/base^m: needs the substituted base nonzero.
                return expr_is_visibly_continuous_at(ctx, base, var_ids, points)
                    && substituted_folds_to_nonzero_rational(ctx, base, var_ids, points);
            }
            expr_is_visibly_continuous_at(ctx, base, var_ids, points)
        }
        Expr::Div(num, den) => {
            expr_is_visibly_continuous_at(ctx, num, var_ids, points)
                && expr_is_visibly_continuous_at(ctx, den, var_ids, points)
                && substituted_folds_to_nonzero_rational(ctx, den, var_ids, points)
        }
        // Continuous total-real unary functions of a continuous argument.
        Expr::Function(fn_id, args) if args.len() == 1 => {
            use cas_ast::BuiltinFn;
            let continuous = ctx.is_builtin(fn_id, BuiltinFn::Sin)
                || ctx.is_builtin(fn_id, BuiltinFn::Cos)
                || ctx.is_builtin(fn_id, BuiltinFn::Exp)
                || ctx.is_builtin(fn_id, BuiltinFn::Sinh)
                || ctx.is_builtin(fn_id, BuiltinFn::Cosh)
                || ctx.is_builtin(fn_id, BuiltinFn::Atan)
                || ctx.is_builtin(fn_id, BuiltinFn::Arctan);
            continuous && expr_is_visibly_continuous_at(ctx, args[0], var_ids, points)
        }
        _ => false,
    }
}

/// DNE-by-paths for a TWO-variable limit at a rational point (F8, Fase 3):
/// run the concrete battery `{y=b, x=a, y=b±(x−a), y=b±(x−a)², x=a±(y−b)²}`
/// through the ENTIRE univariate limit engine as the per-path oracle, and
/// decide ONLY from proven facts — two paths with DIFFERENT exact rational
/// limits, or one path whose univariate limit provably does not exist
/// (disagreeing laterals / oscillation, which arrive as `undefined`). A path
/// whose univariate limit stays residual is NEVER a witness, and agreeing
/// paths NEVER prove existence (the central soundness pin: the caller keeps
/// the honest residual).
pub fn try_multivar_dne_by_paths(
    ctx: &mut Context,
    expr: ExprId,
    var_ids: &[ExprId],
    points: &[ExprId],
) -> Option<MultivarDneByPaths> {
    if var_ids.len() != 2 || points.len() != 2 {
        return None;
    }
    // The battery shifts by the point: both coordinates must be exact rationals.
    let a_val = eval_rational_const_deep(ctx, points[0])?;
    let b_val = eval_rational_const_deep(ctx, points[1])?;
    let (x_id, y_id) = (var_ids[0], var_ids[1]);
    let (a_pt, b_pt) = (points[0], points[1]);
    let name_of = |ctx: &Context, id: ExprId| -> String {
        match ctx.get(id) {
            Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
            _ => "?".to_string(),
        }
    };
    let x_name = name_of(ctx, x_id);
    let y_name = name_of(ctx, y_id);
    // Rendered coordinates for the path displays (rationals by construction).
    let a_disp = a_val.to_string();
    let b_disp = b_val.to_string();
    // `x` at the origin, `(x − a)` otherwise — display fragment of the core.
    let core_disp = |var: &str, at: &BigRational, at_disp: &str| -> String {
        use num_traits::Zero;
        if at.is_zero() {
            var.to_string()
        } else {
            format!("({var} - {at_disp})")
        }
    };
    let with_offset = |base: &BigRational, base_disp: &str, core: String| -> String {
        use num_traits::Zero;
        if base.is_zero() {
            core
        } else {
            format!("{base_disp} + {core}")
        }
    };

    // Structurally-minimal shifted core: `x` at the origin, `x − a` otherwise.
    let shifted = |ctx: &mut Context, var: ExprId, at: ExprId, at_val: &BigRational| -> ExprId {
        use num_traits::Zero;
        if at_val.is_zero() {
            var
        } else {
            ctx.add(Expr::Sub(var, at))
        }
    };
    let offset =
        |ctx: &mut Context, base: ExprId, base_val: &BigRational, core: ExprId| -> ExprId {
            use num_traits::Zero;
            if base_val.is_zero() {
                core
            } else {
                ctx.add(Expr::Add(base, core))
            }
        };

    // Build the battery as (substituted var, path expr, remaining var, its point).
    let mut paths: Vec<(ExprId, ExprId, ExprId, ExprId, String)> = Vec::new();
    // y fixed / x fixed.
    paths.push((y_id, b_pt, x_id, a_pt, format!("{y_name} = {b_disp}")));
    paths.push((x_id, a_pt, y_id, b_pt, format!("{x_name} = {a_disp}")));
    // y = b ± (x−a) and y = b ± (x−a)².
    for sign in [1i32, -1] {
        let sign_str = if sign < 0 { "-" } else { "" };
        let core = shifted(ctx, x_id, a_pt, &a_val);
        let core = if sign < 0 {
            ctx.add(Expr::Neg(core))
        } else {
            core
        };
        let path = offset(ctx, b_pt, &b_val, core);
        let disp = with_offset(
            &b_val,
            &b_disp,
            format!("{sign_str}{}", core_disp(&x_name, &a_val, &a_disp)),
        );
        paths.push((y_id, path, x_id, a_pt, format!("{y_name} = {disp}")));

        let base = shifted(ctx, x_id, a_pt, &a_val);
        let two = ctx.num(2);
        let sq = ctx.add(Expr::Pow(base, two));
        let sq = if sign < 0 { ctx.add(Expr::Neg(sq)) } else { sq };
        let path = offset(ctx, b_pt, &b_val, sq);
        let disp = with_offset(
            &b_val,
            &b_disp,
            format!("{sign_str}{}^2", core_disp(&x_name, &a_val, &a_disp)),
        );
        paths.push((y_id, path, x_id, a_pt, format!("{y_name} = {disp}")));
    }
    // x = a ± (y−b)².
    for sign in [1i32, -1] {
        let sign_str = if sign < 0 { "-" } else { "" };
        let base = shifted(ctx, y_id, b_pt, &b_val);
        let two = ctx.num(2);
        let sq = ctx.add(Expr::Pow(base, two));
        let sq = if sign < 0 { ctx.add(Expr::Neg(sq)) } else { sq };
        let path = offset(ctx, a_pt, &a_val, sq);
        let disp = with_offset(
            &a_val,
            &a_disp,
            format!("{sign_str}{}^2", core_disp(&y_name, &b_val, &b_disp)),
        );
        paths.push((x_id, path, y_id, b_pt, format!("{x_name} = {disp}")));
    }

    let opts = LimitOptions::default();
    let mut witnesses: Vec<(String, BigRational)> = Vec::new();
    for (sub_var, path_expr, remaining_var, remaining_point, path_display) in paths {
        let restricted = cas_ast::substitute_expr_by_id(ctx, expr, sub_var, path_expr);
        let outcome = eval_limit_at_infinity(
            ctx,
            restricted,
            remaining_var,
            Approach::Finite(remaining_point),
            &opts,
        );
        let is_undefined = matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined));
        if is_undefined {
            // Proven DNE along ONE path (disagreeing laterals / oscillation —
            // those verdicts always arrive as `undefined`) decides the whole.
            return Some(MultivarDneByPaths {
                witness_a: MultivarPathWitness {
                    path_display,
                    value_display: "el límite univariado por este camino no existe".to_string(),
                },
                witness_b: None,
            });
        }
        if outcome.warning.is_some() {
            // Residual path: NEVER a witness (pinned soundness clause).
            continue;
        }
        let Some(value) = limit_result_rational(ctx, outcome.expr) else {
            continue;
        };
        if let Some((first_path, first_val)) = witnesses.first() {
            if *first_val != value {
                return Some(MultivarDneByPaths {
                    witness_a: MultivarPathWitness {
                        path_display: first_path.clone(),
                        value_display: format!("{first_val}"),
                    },
                    witness_b: Some(MultivarPathWitness {
                        path_display,
                        value_display: format!("{value}"),
                    }),
                });
            }
        } else {
            witnesses.push((path_display, value));
        }
    }
    None
}

/// Componentwise matrix limit, all-or-nothing:
/// - every entry limit resolves → the matrix of entry limits;
/// - any entry is `undefined` (DNE) → the whole limit is `undefined` (a matrix
///   limit exists iff every entry limit does), quoting that entry's warning;
/// - any entry declines → the WHOLE matrix stays an honest residual (never a
///   partially-evaluated matrix), quoting the declining entry's warning.
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_limit_matrix_componentwise(
    ctx: &mut Context,
    matrix_expr: ExprId,
    rows: usize,
    cols: usize,
    data: Vec<ExprId>,
    var: ExprId,
    approach: Approach,
    opts: &LimitOptions,
) -> LimitEvalOutcome {
    if data.len() > MATRIX_LIMIT_MAX_CELLS {
        let residual = mk_limit_for_approach(ctx, matrix_expr, var, approach);
        return LimitEvalOutcome {
            expr: residual,
            warning: Some(format!(
                "Matrix limits are bounded to {MATRIX_LIMIT_MAX_CELLS} cells"
            )),
        };
    }
    let mut entry_limits = Vec::with_capacity(data.len());
    for entry in data {
        let outcome = eval_limit_at_infinity(ctx, entry, var, approach, opts);
        // Proven DNE for an entry (undefined comes only from the proven paths:
        // disagreeing laterals, oscillation) decides the WHOLE matrix — check
        // before the decline branch, because those proofs carry a warning too.
        if matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined)) {
            let detail = outcome
                .warning
                .unwrap_or_else(|| "an entry limit does not exist".to_string());
            return LimitEvalOutcome {
                expr: outcome.expr,
                warning: Some(format!("the matrix limit does not exist: {detail}")),
            };
        }
        if let Some(entry_warning) = outcome.warning {
            let residual = mk_limit_for_approach(ctx, matrix_expr, var, approach);
            return LimitEvalOutcome {
                expr: residual,
                warning: Some(format!("matrix entry declines: {entry_warning}")),
            };
        }
        entry_limits.push(outcome.expr);
    }
    LimitEvalOutcome {
        expr: ctx.add(Expr::Matrix {
            rows,
            cols,
            data: entry_limits,
        }),
        warning: None,
    }
}

// ── LINT SLICE presimplify_safe: BEGIN ──────────────────────────────────────
// Everything from here to the END marker is the allowlist-only presimplify
// pipeline audited by scripts/lint_limit_presimplify.sh (no rationalization,
// no expansion, no general simplifier, no domain assumptions). The markers
// are the lint's anchors — they moved here WITH the code when the old anchor
// (a const/cfg(test) bracket in limits_support.rs) was left behind by the
// submodule split, which silently emptied the audited slice. Keep them glued
// to the pipeline if it moves again; the lint fails closed without them.

/// Multiply two expressions, dropping a unit factor (`1·e → e`).
pub(super) fn mul_drop_unit(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_one(ctx, a) {
        return b;
    }
    if expr_is_one(ctx, b) {
        return a;
    }
    ctx.add(Expr::Mul(a, b))
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
pub(crate) fn presimplify_safe_for_limit(ctx: &mut Context, expr: ExprId) -> ExprId {
    presimplify_recursive(ctx, expr, 0)
}
// ── LINT SLICE presimplify_safe: END ────────────────────────────────────────
