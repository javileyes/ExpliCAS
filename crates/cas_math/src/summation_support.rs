use crate::build::mul2_raw;
use crate::expr_nary;
use crate::expr_relations::is_structurally_zero;
use crate::polynomial::Polynomial;
use cas_ast::ordering::compare_expr;
use cas_ast::views::as_rational_const;
use cas_ast::{substitute_expr_by_id, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteAggregateCall {
    pub term: ExprId,
    pub var_expr: ExprId,
    pub var_name: String,
    pub start_expr: ExprId,
    pub end_expr: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumEvaluationKind {
    Telescoping,
    SumOfFirstIntegers,
    SumOfSquares,
    SumOfCubes,
    SumOfConstant,
    GeometricPower,
    /// Linear combination of integer powers, summed term by term via the Faulhaber
    /// closed forms (`sum(2k)`, `sum(k^2+k)`, `sum(3k^2 - k + 1)`). Reached only when
    /// the dedicated single-power builders decline (a scaled or multi-term polynomial).
    PolynomialLinearity,
    FiniteDirect {
        start: i64,
        end: i64,
    },
    /// Series with an infinite upper bound whose divergence is classified
    /// (`±infinity`) instead of substituting `infinity` into a finite formula.
    DivergentInfinite,
    /// Convergent infinite series with a closed form (`sum(c·r^k, k, a, inf) =
    /// c·r^a/(1-r)` for a rational ratio `|r| < 1`).
    ConvergentInfinite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumEvaluationPlan {
    pub call: FiniteAggregateCall,
    pub candidate: ExprId,
    pub kind: SumEvaluationKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductEvaluationKind {
    Telescoping,
    FactorizedTelescoping,
    ProductOfFirstIntegers,
    ProductOfPowers,
    ProductOfConstant,
    FiniteDirect {
        start: i64,
        end: i64,
    },
    /// Product with an infinite upper bound whose divergence is classified
    /// (`±infinity`/`0`) instead of substituting `infinity` into a finite formula.
    DivergentInfinite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductEvaluationPlan {
    pub call: FiniteAggregateCall,
    pub candidate: ExprId,
    pub kind: ProductEvaluationKind,
}

/// Parse finite aggregate call shape:
/// - `sum(term, var, start, end)`
/// - `product(term, var, start, end)`
pub fn try_extract_finite_aggregate_call(
    ctx: &Context,
    expr: ExprId,
    callee_name: &str,
) -> Option<FiniteAggregateCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != callee_name || args.len() != 4 {
        return None;
    }

    let term = args[0];
    let var_expr = args[1];
    let start_expr = args[2];
    let end_expr = args[3];

    let var_name = if let Expr::Variable(sym_id) = ctx.get(var_expr) {
        ctx.sym_name(*sym_id).to_string()
    } else {
        return None;
    };

    Some(FiniteAggregateCall {
        term,
        var_expr,
        var_name,
        start_expr,
        end_expr,
    })
}

/// Extract numeric range bounds for finite direct evaluation.
///
/// Returns `(start, end)` only when:
/// - both bounds are exact integers
/// - `start <= end`
/// - range length is <= `max_span`
pub fn try_extract_bounded_integer_range(
    ctx: &Context,
    start_expr: ExprId,
    end_expr: ExprId,
    max_span: i64,
) -> Option<(i64, i64)> {
    let start = crate::expr_extract::extract_i64_integer(ctx, start_expr)?;
    let end = crate::expr_extract::extract_i64_integer(ctx, end_expr)?;
    if start > end {
        return None;
    }
    if end - start > max_span {
        return None;
    }
    Some((start, end))
}

/// Build finite sum by substituting integer values into `term`.
pub fn build_finite_sum_substitution(
    ctx: &mut Context,
    term: ExprId,
    var_expr: ExprId,
    start: i64,
    end: i64,
) -> ExprId {
    let mut result = ctx.num(0);
    for k in start..=end {
        let k_expr = ctx.num(k);
        let substituted = substitute_expr_by_id(ctx, term, var_expr, k_expr);
        result = ctx.add(Expr::Add(result, substituted));
    }
    result
}

/// Build finite product by substituting integer values into `term`.
pub fn build_finite_product_substitution(
    ctx: &mut Context,
    term: ExprId,
    var_expr: ExprId,
    start: i64,
    end: i64,
) -> ExprId {
    let mut result = ctx.num(1);
    for k in start..=end {
        let k_expr = ctx.num(k);
        let substituted = substitute_expr_by_id(ctx, term, var_expr, k_expr);
        result = mul2_raw(ctx, result, substituted);
    }
    result
}

/// True when `expr` is the literal `+infinity` constant.
fn is_positive_infinity(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(Constant::Infinity))
}

fn make_infinity(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Infinity))
}

fn make_neg_infinity(ctx: &mut Context) -> ExprId {
    let inf = make_infinity(ctx);
    ctx.add(Expr::Neg(inf))
}

/// Classify a series `sum(term, k, a, infinity)` whose upper bound is infinite.
///
/// Returns the divergence value (`±infinity`, or `0` for a zero summand) when it
/// can be proven, and `None` to leave the series UNEVALUATED (convergent or
/// unclassifiable — e.g. `sum((1/2)^k)`, `sum(1/k^2)`). It never substitutes
/// `infinity` into a finite closed form, so malformed tokens like
/// `1/2*infinity^2` or `2^infinity` are no longer produced. Returning a genuine
/// `Constant::Infinity` (rather than a folded `infinity!`/`pow(c,infinity)` atom)
/// lets `0 * sum(...)` and `sum(...) - sum(...)` resolve to `undefined` via the
/// extended-real arithmetic rules.
fn classify_infinite_sum(ctx: &mut Context, call: &FiniteAggregateCall) -> Option<ExprId> {
    use std::cmp::Ordering;
    let zero = BigRational::zero();
    // Constant summand c: sum diverges to sign(c)*infinity (or is 0 if c == 0).
    if let Some(c) = as_rational_const(ctx, call.term, 8) {
        return Some(match c.cmp(&zero) {
            Ordering::Greater => make_infinity(ctx),
            Ordering::Less => make_neg_infinity(ctx),
            Ordering::Equal => ctx.num(0),
        });
    }
    // Polynomial summand of degree >= 1: diverges with the sign of the leading
    // coefficient (the high-degree term eventually dominates).
    if let Ok(poly) = Polynomial::from_expr(ctx, call.term, &call.var_name) {
        let deg = poly.degree();
        if deg >= 1 {
            let leading = poly
                .coeffs
                .get(deg)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if !leading.is_zero() {
                return Some(if leading.is_negative() {
                    make_neg_infinity(ctx)
                } else {
                    make_infinity(ctx)
                });
            }
        }
    }
    // Geometric `r^k` with a rational base `r > 1`: diverges to +infinity.
    // (|r| < 1 converges and r <= -1 oscillates -> left unevaluated.)
    let pow_parts = match ctx.get(call.term) {
        Expr::Pow(base, exp) => Some((*base, *exp)),
        _ => None,
    };
    if let Some((base, exp)) = pow_parts {
        if is_named_var(ctx, exp, &call.var_name) {
            if let Some(r) = as_rational_const(ctx, base, 8) {
                if r > BigRational::one() {
                    return Some(make_infinity(ctx));
                }
            }
        }
    }
    None
}

/// `b^n` for an integer exponent `n` (negative → reciprocal). `None` for `0^(n<0)`.
fn rational_pow_int(base: &BigRational, n: i64) -> Option<BigRational> {
    if n >= 0 {
        Some(num_traits::pow(base.clone(), n as usize))
    } else if base.is_zero() {
        None
    } else {
        Some(num_traits::pow(base.clone(), (-n) as usize).recip())
    }
}

/// Extract `(coefficient, ratio)` from a term that is structurally `c · r^k` with
/// rational `c`, `r`. Recognises the `b^(s·k+t)` power form (`s`, `t` integers, so
/// `r=b^s`, `c=b^t` stay rational, including a reciprocal `2^(-k)` and a negative base
/// `(-1/2)^k`), a `Div(constant, geometric)`, a `Mul(constant, geometric)`, and a `Neg`.
/// The structural match is the proof the term is genuinely geometric — a non-geometric
/// summand like `k·2^k` or `2^k+3^k` returns `None`.
fn extract_geometric_term(
    ctx: &Context,
    term: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    match ctx.get(term) {
        Expr::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;
            let b = as_rational_const(ctx, base, 8)?;
            // Exponent must be linear in `var`: `s·k + t`.
            let ep = Polynomial::from_expr(ctx, exp, var).ok()?;
            if ep.degree() != 1 {
                return None;
            }
            let s = ep.coeffs.get(1).cloned().unwrap_or_else(BigRational::zero);
            let t = ep.coeffs.first().cloned().unwrap_or_else(BigRational::zero);
            if !s.is_integer() || !t.is_integer() {
                return None;
            }
            let si = s.to_integer().try_into().ok()?;
            let ti = t.to_integer().try_into().ok()?;
            let ratio = rational_pow_int(&b, si)?;
            let coeff = rational_pow_int(&b, ti)?;
            Some((coeff, ratio))
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            let c = as_rational_const(ctx, num, 8)?; // numerator independent of k
            let (dc, dr) = extract_geometric_term(ctx, den, var)?;
            if dr.is_zero() || dc.is_zero() {
                return None;
            }
            // c / (dc · dr^k) = (c/dc) · (1/dr)^k
            Some((c / dc, dr.recip()))
        }
        Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            if let Some(c) = as_rational_const(ctx, a, 8) {
                let (gc, gr) = extract_geometric_term(ctx, b, var)?;
                Some((c * gc, gr))
            } else if let Some(c) = as_rational_const(ctx, b, 8) {
                let (gc, gr) = extract_geometric_term(ctx, a, var)?;
                Some((c * gc, gr))
            } else {
                None
            }
        }
        Expr::Neg(inner) => extract_geometric_term(ctx, *inner, var).map(|(c, r)| (-c, r)),
        _ => None,
    }
}

/// Closed form of a convergent geometric series `sum(c·r^k, k, a, inf) = c·r^a/(1-r)`
/// for a rational ratio `|r| < 1` and an integer lower bound `a`. Returns `None` when the
/// term is not geometric, the ratio does not converge (`|r| ≥ 1`, or `r = 0`), or the
/// lower bound is not an integer literal (a symbolic lower bound is a sound peldaño).
fn try_convergent_infinite_geometric_sum(
    ctx: &mut Context,
    call: &FiniteAggregateCall,
) -> Option<ExprId> {
    let (coeff, ratio) = extract_geometric_term(ctx, call.term, &call.var_name)?;
    // Convergence: |r| < 1 and r ≠ 0 (r = 0 leaves only the k = a term — out of scope).
    if ratio.is_zero() || ratio.abs() >= BigRational::one() {
        return None;
    }
    let a = crate::expr_extract::extract_i64_integer(ctx, call.start_expr)?;
    let ratio_a = rational_pow_int(&ratio, a)?;
    // c · r^a / (1 - r), an exact rational.
    let value = coeff * ratio_a / (BigRational::one() - &ratio);
    Some(ctx.add(Expr::Number(value)))
}

/// Closed form of a convergent arithmetic-geometric series `sum(p(k)·c·r^k, k, a, inf)` for a
/// polynomial cofactor `p` of degree 1 or 2 and `|r| < 1` (integer lower bound `a ≥ 0`). The infinite
/// tails from `k = 1` are exact rationals — `Σ r^k = r/(1−r)`, `Σ k·r^k = r/(1−r)²`,
/// `Σ k²·r^k = r(1+r)/(1−r)³` — and the lower bound is corrected by the finite head (`Σ_{1}^{a-1}`)
/// or, for `a = 0`, the extra `k = 0` term `p(0) = γ`. Returns `None` on divergence or a non-integer
/// / negative lower bound (sound peldaños). Pure geometric (degree 0) is declined to its own builder.
fn try_convergent_infinite_arithmetic_geometric_sum(
    ctx: &mut Context,
    call: &FiniteAggregateCall,
) -> Option<ExprId> {
    let (geometric_coefficient, ratio, [gamma, beta, alpha]) =
        decompose_arithmetic_geometric(ctx, call.term, &call.var_name)?;
    if ratio.is_zero() || ratio.abs() >= BigRational::one() {
        return None;
    }
    let a = crate::expr_extract::extract_i64_integer(ctx, call.start_expr)?;
    if a < 0 {
        return None;
    }
    let one = BigRational::one();
    let one_minus_r = one.clone() - ratio.clone();
    let s0 = ratio.clone() / one_minus_r.clone();
    let s1 = ratio.clone() / (one_minus_r.clone() * one_minus_r.clone());
    let s2 = ratio.clone() * (one.clone() + ratio.clone())
        / (one_minus_r.clone() * one_minus_r.clone() * one_minus_r.clone());
    let mut value = alpha.clone() * s2 + beta.clone() * s1 + gamma.clone() * s0;
    // Lower-bound correction relative to the `k ≥ 1` tails above.
    if a == 0 {
        value += gamma.clone(); // the extra k = 0 term: p(0)·r^0 = γ
    } else {
        for k in 1..a {
            let rk = rational_pow_int(&ratio, k)?;
            let kk = BigRational::from_integer(k.into());
            let pk = alpha.clone() * kk.clone() * kk.clone() + beta.clone() * kk + gamma.clone();
            value -= pk * rk;
        }
    }
    value *= geometric_coefficient;
    Some(ctx.add(Expr::Number(value)))
}

/// Classify a product `product(term, k, a, infinity)` whose upper bound is
/// infinite. Returns the divergence value when provable, else `None`.
fn classify_infinite_product(ctx: &mut Context, call: &FiniteAggregateCall) -> Option<ExprId> {
    let one = BigRational::one();
    let zero = BigRational::zero();
    // product of the index variable itself (factorial-like): diverges to +infinity
    // when the range starts at >= 1.
    if is_named_var(ctx, call.term, &call.var_name) {
        return match crate::expr_extract::extract_i64_integer(ctx, call.start_expr) {
            Some(start) if start >= 1 => Some(make_infinity(ctx)),
            _ => None,
        };
    }
    // Constant factor c (independent of k): c^infinity.
    if let Some(c) = as_rational_const(ctx, call.term, 8) {
        if c > one {
            return Some(make_infinity(ctx)); // |c| > 1 -> diverges
        }
        if c == one {
            return Some(ctx.num(1)); // 1^infinity = 1
        }
        if c > zero && c < one {
            return Some(ctx.num(0)); // 0 < c < 1 -> c^infinity = 0
        }
        // c <= 0: sign oscillates / includes zero -> leave unevaluated.
    }
    None
}

/// Build the best available finite-sum evaluation plan for `sum(...)`.
///
/// Preference order:
/// 1. Telescoping rational pattern.
/// 2. Closed form for `sum(k, k, m, n)`.
/// 3. Direct finite substitution when bounds are small integers.
/// 4. Closed form for `sum(k^2, k, m, n)`.
/// 5. Closed form for `sum(k^3, k, 1, n)`.
/// 6. Closed form for `sum(c, k, 1, n)` when `c` is independent of `k`.
/// 7. Closed form for `sum(a^k, k, m, n)` when `a` is an integer > 1.
pub fn try_plan_finite_sum_evaluation(
    ctx: &mut Context,
    expr: ExprId,
    max_span: i64,
) -> Option<SumEvaluationPlan> {
    let call = try_extract_finite_aggregate_call(ctx, expr, "sum")?;

    // Empty range (lower > upper): the empty sum is 0 by convention. This MUST
    // be checked before any closed form, which would otherwise evaluate its
    // formula at reversed bounds (e.g. `sum(k, k, 6, 3)` → 3·4/2 − 5·6/2 = −9
    // instead of 0). `start..=end` with `start > end` is empty, so the direct
    // builder returns the additive identity.
    if let (Some(start), Some(end)) = (
        crate::expr_extract::extract_i64_integer(ctx, call.start_expr),
        crate::expr_extract::extract_i64_integer(ctx, call.end_expr),
    ) {
        if start > end {
            let candidate =
                build_finite_sum_substitution(ctx, call.term, call.var_expr, start, end);
            return Some(SumEvaluationPlan {
                call,
                candidate,
                kind: SumEvaluationKind::FiniteDirect { start, end },
            });
        }
    }

    if let Some(candidate) = try_build_telescoping_rational_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::Telescoping,
        });
    }

    if let Some(candidate) = try_build_telescoping_consecutive_factor_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::Telescoping,
        });
    }

    // Infinite upper bound: NOT a finite sum. The closed-form builders below would
    // substitute `infinity` into finite formulas (e.g. n(n+1)/2 -> 1/2*infinity^2,
    // (r^(n+1)-r^a)/(r-1) -> 2^infinity-1). Classify the divergence here and return
    // (or leave unevaluated) so those builders never run on an infinite bound.
    if is_positive_infinity(ctx, call.end_expr) {
        // Convergent geometric series `sum(c·r^k, k, a, inf) = c·r^a/(1-r)` (|r| < 1)
        // is tried first; otherwise classify the divergence.
        if let Some(candidate) = try_convergent_infinite_geometric_sum(ctx, &call) {
            return Some(SumEvaluationPlan {
                call,
                candidate,
                kind: SumEvaluationKind::ConvergentInfinite,
            });
        }
        if let Some(candidate) = try_convergent_infinite_arithmetic_geometric_sum(ctx, &call) {
            return Some(SumEvaluationPlan {
                call,
                candidate,
                kind: SumEvaluationKind::ConvergentInfinite,
            });
        }
        return classify_infinite_sum(ctx, &call).map(|candidate| SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::DivergentInfinite,
        });
    }

    if let Some(candidate) = try_build_sum_of_first_integers(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::SumOfFirstIntegers,
        });
    }

    if let Some((start, end)) =
        try_extract_bounded_integer_range(ctx, call.start_expr, call.end_expr, max_span)
    {
        let candidate = build_finite_sum_substitution(ctx, call.term, call.var_expr, start, end);
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::FiniteDirect { start, end },
        });
    }

    if let Some(candidate) = try_build_sum_of_squares(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::SumOfSquares,
        });
    }

    if let Some(candidate) = try_build_sum_of_cubes(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::SumOfCubes,
        });
    }

    if let Some(candidate) = try_build_sum_of_constant(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::SumOfConstant,
        });
    }

    if let Some(candidate) = try_build_geometric_power_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::GeometricPower,
        });
    }

    // General rational-ratio finite geometric (`sum(1/2^k, k, a, n)`,
    // `sum((2/3)^k, k, 1, n)`): the integer-base builder above declines a fractional
    // or negative ratio, so this catches them via the shared geometric matcher.
    if let Some(candidate) = try_build_geometric_rational_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::GeometricPower,
        });
    }

    // Arithmetic-geometric `sum(c·k·r^k)`: a linear cofactor times a geometric power.
    if let Some(candidate) = try_build_arithmetic_geometric_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::GeometricPower,
        });
    }

    // Linearity over a sum of geometric / arithmetic-geometric terms (the distributed affine
    // cofactor `(αk+β)·r^k` the engine expands to `r^k + α·k·r^(k+1)`).
    if let Some(candidate) = try_build_geometric_additive_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::GeometricPower,
        });
    }

    // Linearity fallback for any remaining polynomial summand (`sum(2k)`,
    // `sum(k^2+k)`): tried LAST so the dedicated single-power and constant builders
    // above keep their exact output; this only fires on the scaled/multi-term cases
    // they declined.
    if let Some(candidate) = try_build_polynomial_sum(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(SumEvaluationPlan {
            call,
            candidate,
            kind: SumEvaluationKind::PolynomialLinearity,
        });
    }

    None
}

pub fn try_build_sum_of_first_integers(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if !is_named_var(ctx, summand, var)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return Some(build_triangular_number(ctx, end));
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let upper_numerator = mul2_raw(ctx, end, end_plus_one);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower_numerator = mul2_raw(ctx, start, start_minus_one);
    let numerator = ctx.add(Expr::Sub(upper_numerator, lower_numerator));
    Some(ctx.add(Expr::Div(numerator, two)))
}

fn build_triangular_number(ctx: &mut Context, end: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let numerator = mul2_raw(ctx, end, end_plus_one);
    ctx.add(Expr::Div(numerator, two))
}

pub fn try_build_sum_of_squares(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if !expr_is_square_of_named_var(ctx, summand, var)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return Some(build_square_pyramid_number(ctx, end));
    }

    let one = ctx.num(1);
    let six = ctx.num(6);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let two_end_plus_one = build_double_plus_one(ctx, end);
    let upper_first_product = mul2_raw(ctx, end, end_plus_one);
    let upper_numerator = mul2_raw(ctx, upper_first_product, two_end_plus_one);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let two_start_minus_one = build_double_minus_one(ctx, start);
    let lower_first_product = mul2_raw(ctx, start, start_minus_one);
    let lower_numerator = mul2_raw(ctx, lower_first_product, two_start_minus_one);
    let numerator = ctx.add(Expr::Sub(upper_numerator, lower_numerator));
    Some(ctx.add(Expr::Div(numerator, six)))
}

fn build_square_pyramid_number(ctx: &mut Context, end: ExprId) -> ExprId {
    let one = ctx.num(1);
    let six = ctx.num(6);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let two_end_plus_one = build_double_plus_one(ctx, end);
    let first_product = mul2_raw(ctx, end, end_plus_one);
    let numerator = mul2_raw(ctx, first_product, two_end_plus_one);
    ctx.add(Expr::Div(numerator, six))
}

fn build_double_plus_one(ctx: &mut Context, value: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let doubled = mul2_raw(ctx, two, value);
    ctx.add(Expr::Add(doubled, one))
}

fn build_double_minus_one(ctx: &mut Context, value: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let doubled = mul2_raw(ctx, two, value);
    ctx.add(Expr::Sub(doubled, one))
}

pub fn try_build_sum_of_cubes(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if !expr_is_cube_of_named_var(ctx, summand, var)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    let one = ctx.num(1);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let upper = build_triangular_square(ctx, end, end_plus_one);
    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return Some(upper);
    }

    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = build_triangular_square(ctx, start, start_minus_one);
    Some(ctx.add(Expr::Sub(upper, lower)))
}

fn build_triangular_square(ctx: &mut Context, first: ExprId, second: ExprId) -> ExprId {
    let two = ctx.num(2);
    let square = ctx.num(2);
    let numerator = mul2_raw(ctx, first, second);
    let triangular_number = ctx.add(Expr::Div(numerator, two));
    ctx.add(Expr::Pow(triangular_number, square))
}

/// Binomial coefficient `C(n, k)` as an exact `BigInt` (incremental, no factorial overflow).
fn binomial(n: usize, k: usize) -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};
    if k > n {
        return BigInt::zero();
    }
    let k = k.min(n - k);
    let mut result = BigInt::one();
    for i in 0..k {
        result = result * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    result
}

/// Coefficients (index `i` = coefficient of `n^i`) of the Faulhaber polynomial
/// `S_p(n) = Σ_{k=1}^{n} k^p`, via the recurrence
/// `(p+1)·S_p = (n+1)^(p+1) − 1 − Σ_{j=0}^{p-1} C(p+1, j)·S_j` with `S_0(n) = n`.
fn faulhaber_power_sum_coeffs(p: usize) -> Vec<BigRational> {
    use num_traits::{One, Zero};
    let mut sums: Vec<Vec<BigRational>> = Vec::with_capacity(p + 1);
    sums.push(vec![BigRational::zero(), BigRational::one()]); // S_0(n) = n
    for q in 1..=p {
        let m = q + 1;
        // (n+1)^m = Σ_{i=0}^{m} C(m, i) n^i
        let mut acc: Vec<BigRational> = (0..=m)
            .map(|i| BigRational::from_integer(binomial(m, i)))
            .collect();
        acc[0] -= BigRational::one(); // − 1
        for (j, sj) in sums.iter().enumerate().take(q) {
            // − C(m, j) · S_j
            let c = BigRational::from_integer(binomial(m, j));
            for (i, coeff) in sj.iter().enumerate() {
                acc[i] -= &c * coeff;
            }
        }
        let denom = BigRational::from_integer(num_bigint::BigInt::from(q + 1));
        for a in acc.iter_mut() {
            *a /= &denom;
        }
        sums.push(acc);
    }
    sums.pop().unwrap_or_default()
}

/// Build `Σ_i coeffs[i]·n^i` (a polynomial in the bound expression `n`) by Horner.
fn horner_power_sum_expr(ctx: &mut Context, coeffs: &[BigRational], n: ExprId) -> ExprId {
    let mut acc: Option<ExprId> = None;
    for coeff in coeffs.iter().rev() {
        let coeff_node = ctx.add(Expr::Number(coeff.clone()));
        acc = Some(match acc {
            None => coeff_node,
            Some(prev) => {
                let scaled = mul2_raw(ctx, prev, n);
                ctx.add(Expr::Add(scaled, coeff_node))
            }
        });
    }
    acc.unwrap_or_else(|| ctx.num(0))
}

/// Closed form for `Σ_{k=1}^{n} k^p`. `p ∈ {1, 2, 3}` reuse the same factored expressions
/// the dedicated single-power builders emit (keeping their output byte-identical);
/// `p ≥ 4` build the Faulhaber polynomial directly. Returns `None` for `p = 0`.
fn power_sum_one_to(ctx: &mut Context, p: usize, n: ExprId) -> Option<ExprId> {
    match p {
        0 => None,
        1 => Some(build_triangular_number(ctx, n)),
        2 => Some(build_square_pyramid_number(ctx, n)),
        3 => {
            let one = ctx.num(1);
            let n_plus_one = ctx.add(Expr::Add(n, one));
            Some(build_triangular_square(ctx, n, n_plus_one))
        }
        _ => {
            let coeffs = faulhaber_power_sum_coeffs(p);
            Some(horner_power_sum_expr(ctx, &coeffs, n))
        }
    }
}

/// Closed form for `Σ_{k=start}^{end} k^p` via `Σ_{1}^{end} − Σ_{1}^{start−1}` (the
/// `k = 0` term vanishes for `p ≥ 1`, so a `start` of 0 or 1 needs no lower correction).
fn power_sum_start_to_end(
    ctx: &mut Context,
    p: usize,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let upper = power_sum_one_to(ctx, p, end)?;
    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return Some(upper);
    }
    let one = ctx.num(1);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = power_sum_one_to(ctx, p, start_minus_one)?;
    Some(ctx.add(Expr::Sub(upper, lower)))
}

/// Upper degree bound for the polynomial-summation linearity path. The Faulhaber recurrence
/// is exact at any degree; this cap just keeps the closed form from ballooning on pathological
/// inputs while covering every realistic polynomial summand.
const MAX_POLYNOMIAL_SUM_DEGREE: usize = 12;

/// Sum of any polynomial summand by LINEARITY: `Σ Σ_p a_p·k^p = Σ_p a_p·(Σ k^p)`,
/// each power summed by its Faulhaber closed form. Covers `sum(2k)`, `sum(k^2+k)`,
/// `sum(3k^2 - k + 1)`, `sum(k^4)`, `sum(k^5 - k)`, `sum(k(k+1))`, etc. — the scaled/
/// multi-term cases the dedicated single-power builders decline. Constants are owned by
/// [`try_build_sum_of_constant`] and bare single powers by their dedicated builders earlier
/// in the dispatch, so this only ever runs on what they left.
pub fn try_build_polynomial_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if contains_named_var(ctx, start, var) || contains_named_var(ctx, end, var) {
        return None;
    }

    // The summand must be a polynomial in the index variable; `2^k`, `1/k`, `sin(k)`
    // (and any rational/transcendental form) decline here and stay residual.
    let poly = Polynomial::from_expr(ctx, summand, var).ok()?;
    let degree = poly.degree();
    if degree == 0 || degree > MAX_POLYNOMIAL_SUM_DEGREE || poly.is_zero() {
        return None;
    }

    let coeffs = poly.coeffs.clone();
    let mut terms: Vec<ExprId> = Vec::new();
    for (p, coeff) in coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let coeff_node = ctx.add(Expr::Number(coeff.clone()));
        let term = if p == 0 {
            let count = build_inclusive_range_count(ctx, start, end);
            mul2_raw(ctx, coeff_node, count)
        } else {
            let power_sum = power_sum_start_to_end(ctx, p, start, end)?;
            mul2_raw(ctx, coeff_node, power_sum)
        };
        terms.push(term);
    }

    let mut combined = *terms.first()?;
    for &term in &terms[1..] {
        combined = ctx.add(Expr::Add(combined, term));
    }
    Some(combined)
}

pub fn try_build_sum_of_constant(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // A structurally-zero summand sums to 0 over ANY range — finite, symbolic, or
    // infinite. Returning 0 here BEFORE computing the term count avoids building
    // `0 * (inf - 1 + 1)` = `0 * inf`, which the simplifier folds to `undefined`:
    // `sum(0, k, 1, inf)` and `sum(k - k, k, 1, inf)` are 0, not undefined.
    if is_structurally_zero(ctx, summand) {
        return Some(ctx.num(0));
    }

    if contains_named_var(ctx, summand, var)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    let term_count = build_inclusive_range_count(ctx, start, end);
    Some(mul2_raw(ctx, summand, term_count))
}

fn build_inclusive_range_count(ctx: &mut Context, start: ExprId, end: ExprId) -> ExprId {
    if expr_is_one(ctx, start) {
        return end;
    }

    let one = ctx.num(1);
    if expr_is_zero(ctx, start) {
        return ctx.add(Expr::Add(end, one));
    }

    let end_minus_start = ctx.add(Expr::Sub(end, start));
    ctx.add(Expr::Add(end_minus_start, one))
}

pub fn try_build_geometric_power_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if contains_named_var(ctx, start, var) || contains_named_var(ctx, end, var) {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(summand) else {
        return None;
    };
    let base = *base;
    let exp = *exp;
    if !is_named_var(ctx, exp, var) {
        return None;
    }
    let base_value = crate::expr_extract::extract_i64_integer(ctx, base)?;
    if base_value <= 1 {
        return None;
    }

    let one = ctx.num(1);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let upper_power = ctx.add(Expr::Pow(base, end_plus_one));
    let lower_power = if expr_is_zero(ctx, start) {
        one
    } else {
        ctx.add(Expr::Pow(base, start))
    };
    let numerator = ctx.add(Expr::Sub(upper_power, lower_power));
    let denominator_value = base_value - 1;
    if denominator_value == 1 {
        return Some(numerator);
    }
    let denominator = ctx.num(denominator_value);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

/// Closed form of a finite geometric sum with a general rational ratio:
/// `sum(c·r^k, k, a, n) = c·(r^a - r^(n+1))/(1-r)` for `r ≠ 1`. Reuses the shared
/// [`extract_geometric_term`] matcher, so it covers fractional and negative ratios that
/// the integer-base [`try_build_geometric_power_sum`] declines, with a numeric OR symbolic
/// upper bound. Returns `None` for a non-geometric summand, `r = 1` (a constant sum, owned
/// elsewhere), or a bound that contains the index variable.
pub fn try_build_geometric_rational_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if contains_named_var(ctx, start, var) || contains_named_var(ctx, end, var) {
        return None;
    }
    let (coeff, ratio) = extract_geometric_term(ctx, summand, var)?;
    if ratio == BigRational::one() {
        return None; // constant sum (r = 1) is handled by the sum-of-constant builder
    }

    // c · (r^a - r^(n+1)) / (1 - r).
    let ratio_num = ctx.add(Expr::Number(ratio.clone()));
    let r_pow_a = ctx.add(Expr::Pow(ratio_num, start));
    let one = ctx.num(1);
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let r_pow_np1 = ctx.add(Expr::Pow(ratio_num, end_plus_one));
    let diff = ctx.add(Expr::Sub(r_pow_a, r_pow_np1));
    let coeff_num = ctx.add(Expr::Number(coeff));
    let numerator = mul2_raw(ctx, coeff_num, diff);
    let one_minus_r = ctx.add(Expr::Number(BigRational::one() - &ratio));
    Some(ctx.add(Expr::Div(numerator, one_minus_r)))
}

/// Build the best available finite-product evaluation plan for `product(...)`.
///
/// Preference order:
/// 1. Shift-1 telescoping ratio pattern.
/// 2. Factorizable `1 - 1/k^2` telescoping pattern.
/// 3. Direct finite substitution when bounds are small integers.
/// 4. Closed form for `product(k, k, m, n)`.
/// 5. Closed form for `product(k^p, k, m, n)` when `p` is an integer > 1.
/// 6. Closed form for `product(c, k, m, n)` when `c`, `m`, and `n` are independent of `k`.
pub fn try_plan_finite_product_evaluation(
    ctx: &mut Context,
    expr: ExprId,
    max_span: i64,
) -> Option<ProductEvaluationPlan> {
    let call = try_extract_finite_aggregate_call(ctx, expr, "product")?;

    // Empty range (lower > upper): the empty product is 1 by convention. This
    // MUST be checked before any closed form, which would otherwise evaluate its
    // formula at reversed bounds (e.g. `product(k, k, 6, 3)` → 3!/5! = 1/20
    // instead of 1). `start..=end` with `start > end` is empty, so the direct
    // builder returns the multiplicative identity.
    if let (Some(start), Some(end)) = (
        crate::expr_extract::extract_i64_integer(ctx, call.start_expr),
        crate::expr_extract::extract_i64_integer(ctx, call.end_expr),
    ) {
        if start > end {
            let candidate =
                build_finite_product_substitution(ctx, call.term, call.var_expr, start, end);
            return Some(ProductEvaluationPlan {
                call,
                candidate,
                kind: ProductEvaluationKind::FiniteDirect { start, end },
            });
        }
    }

    if let Some(candidate) = try_build_telescoping_product_shift1(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::Telescoping,
        });
    }

    if let Some(candidate) = try_build_factorizable_product_for_one_minus_reciprocal_square(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::FactorizedTelescoping,
        });
    }

    // Infinite upper bound: NOT a finite product. Classify divergence here so the
    // closed-form builders below never substitute `infinity` (e.g. -> infinity! or
    // 2^infinity, which fold as finite atoms and make 0*product(...) collapse to 0).
    if is_positive_infinity(ctx, call.end_expr) {
        return classify_infinite_product(ctx, &call).map(|candidate| ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::DivergentInfinite,
        });
    }

    if let Some((start, end)) =
        try_extract_bounded_integer_range(ctx, call.start_expr, call.end_expr, max_span)
    {
        let candidate =
            build_finite_product_substitution(ctx, call.term, call.var_expr, start, end);
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::FiniteDirect { start, end },
        });
    }

    if let Some(candidate) = try_build_product_of_first_integers(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::ProductOfFirstIntegers,
        });
    }

    if let Some(candidate) = try_build_product_of_powers(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::ProductOfPowers,
        });
    }

    if let Some(candidate) = try_build_product_of_constant(
        ctx,
        call.term,
        &call.var_name,
        call.start_expr,
        call.end_expr,
    ) {
        return Some(ProductEvaluationPlan {
            call,
            candidate,
            kind: ProductEvaluationKind::ProductOfConstant,
        });
    }

    None
}

pub fn try_build_product_of_first_integers(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if !is_named_var(ctx, factor, var)
        || expr_is_non_positive_integer(ctx, start)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    let upper = ctx.call("fact", vec![end]);
    if expr_is_one(ctx, start) {
        return Some(upper);
    }

    let one = ctx.num(1);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = ctx.call("fact", vec![start_minus_one]);
    Some(ctx.add(Expr::Div(upper, lower)))
}

pub fn try_build_product_of_powers(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let exponent = expr_named_var_positive_integer_power(ctx, factor, var)?;
    if expr_is_non_positive_integer(ctx, start)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    let factorial = ctx.call("fact", vec![end]);
    let exponent = ctx.num(exponent);
    if expr_is_one(ctx, start) {
        return Some(ctx.add(Expr::Pow(factorial, exponent)));
    }

    let one = ctx.num(1);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = ctx.call("fact", vec![start_minus_one]);
    let quotient = ctx.add(Expr::Div(factorial, lower));
    Some(ctx.add(Expr::Pow(quotient, exponent)))
}

pub fn try_build_product_of_constant(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if contains_named_var(ctx, factor, var)
        || contains_named_var(ctx, start, var)
        || contains_named_var(ctx, end, var)
    {
        return None;
    }

    let term_count = build_inclusive_range_count(ctx, start, end);
    Some(ctx.add(Expr::Pow(factor, term_count)))
}

/// Extract the integer offset from a linear form `var + k`, `k + var`, `var - k`, or `var`.
///
/// Returns:
/// - `Some(0)` for plain `var`
/// - `Some(k)` for `var + k` / `k + var`
/// - `Some(-k)` for `var - k`
pub fn extract_linear_offset(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some(0),
        Expr::Add(l, r) => {
            if let Expr::Variable(sym_id) = ctx.get(*l) {
                if ctx.sym_name(*sym_id) == var {
                    return crate::expr_extract::extract_i64_integer(ctx, *r);
                }
            }
            if let Expr::Variable(sym_id) = ctx.get(*r) {
                if ctx.sym_name(*sym_id) == var {
                    return crate::expr_extract::extract_i64_integer(ctx, *l);
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            if let Expr::Variable(sym_id) = ctx.get(*l) {
                if ctx.sym_name(*sym_id) == var {
                    return crate::expr_extract::extract_i64_integer(ctx, *r).map(|c| -c);
                }
            }
            None
        }
        _ => None,
    }
}

/// Detect reciprocal power forms for a target variable.
///
/// Matches:
/// - `1/var^n`
/// - `1/var`
/// - `var^(-n)`
///
/// Returns the positive power `n`.
pub fn detect_reciprocal_power(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*num) {
            if n.is_one() {
                if let Expr::Pow(base, exp) = ctx.get(*den) {
                    if let Expr::Variable(sym_id) = ctx.get(*base) {
                        if ctx.sym_name(*sym_id) == var {
                            if let Some(power) = crate::expr_extract::extract_i64_integer(ctx, *exp)
                            {
                                return Some(power);
                            }
                        }
                    }
                }
                if let Expr::Variable(sym_id) = ctx.get(*den) {
                    if ctx.sym_name(*sym_id) == var {
                        return Some(1);
                    }
                }
            }
        }
    }

    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Variable(sym_id) = ctx.get(*base) {
            if ctx.sym_name(*sym_id) == var {
                if let Expr::Neg(inner_exp) = ctx.get(*exp) {
                    if let Some(power) = crate::expr_extract::extract_i64_integer(ctx, *inner_exp) {
                        return Some(power);
                    }
                }
                if let Some(power) = crate::expr_extract::extract_i64_integer(ctx, *exp) {
                    if power < 0 {
                        return Some(-power);
                    }
                }
            }
        }
    }

    None
}

/// Detect `1 - reciprocal_power(var)` pattern.
///
/// Matches:
/// - `1 - 1/var^n`
/// - `1 - var^(-n)`
/// - `1 + (-(1/var^n))`
pub fn detect_one_minus_reciprocal_power(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                return detect_reciprocal_power(ctx, *right, var);
            }
        }
    }

    if let Expr::Add(left, right) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*right) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*left) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*right) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
    }

    None
}

/// Extract `(coefficient, offset)` from an integer linear expression `coefficient·var + offset`,
/// or `None` if it is not linear with integer coefficients.
fn linear_int_coeff_offset(ctx: &Context, expr: ExprId, var: &str) -> Option<(i64, i64)> {
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let to_i64 = |coeff: Option<&BigRational>| -> Option<i64> {
        let value = coeff.cloned().unwrap_or_else(BigRational::zero);
        if !value.is_integer() {
            return None;
        }
        value.to_integer().try_into().ok()
    };
    Some((to_i64(poly.coeffs.get(1))?, to_i64(poly.coeffs.first())?))
}

/// Build `coefficient·anchor + offset` (the affine factor `c·k + off` evaluated at `anchor`).
fn affine_value_at(ctx: &mut Context, anchor: ExprId, coefficient: i64, offset: i64) -> ExprId {
    let scaled = if coefficient == 1 {
        anchor
    } else {
        let coeff = ctx.num(coefficient);
        ctx.add(Expr::Mul(coeff, anchor))
    };
    shift_expr(ctx, scaled, offset)
}

/// Build telescoping product closed form for shift-1 pattern `(k+a)/(k+b)` where `a-b=1`.
///
/// Returns `(end + a) / (start + b)` as an expression when applicable.
pub fn try_build_telescoping_product_shift1(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(factor) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    // Orientation-aware INTEGER telescoping: factors `α·k + c` with equal nonzero coefficient α and
    // offsets differing by exactly α (one factor is the other shifted one step). The DIRECTION
    // matters — `∏ (k+1)/k = n+1` but `∏ k/(k+1) = 1/(n+1)` (the reciprocal). The old code returned
    // the same `n+1` for both orientations (a wrong answer for every denominator-higher product).
    if let (Some((num_coeff, num_off)), Some((den_coeff, den_off))) = (
        linear_int_coeff_offset(ctx, num, var),
        linear_int_coeff_offset(ctx, den, var),
    ) {
        if num_coeff == den_coeff && num_coeff != 0 {
            let alpha = num_coeff;
            let end_plus_one = shift_expr(ctx, end, 1);
            if den_off == num_off + alpha {
                // den(k) = num(k+1): ∏ num(k)/num(k+1) = num(start)/num(end+1).
                let num_start = affine_value_at(ctx, start, alpha, num_off);
                let num_end1 = affine_value_at(ctx, end_plus_one, alpha, num_off);
                return Some(ctx.add(Expr::Div(num_start, num_end1)));
            }
            if num_off == den_off + alpha {
                // num(k) = den(k+1): ∏ den(k+1)/den(k) = den(end+1)/den(start).
                let den_start = affine_value_at(ctx, start, alpha, den_off);
                let den_end1 = affine_value_at(ctx, end_plus_one, alpha, den_off);
                return Some(ctx.add(Expr::Div(den_end1, den_start)));
            }
        }
    }

    if let Some((base, _coeff)) =
        detect_affine_consecutive_telescoping_sum_base_and_gap(ctx, den, num, var)
    {
        let var_expr = ctx.var(var);
        let start_base = substitute_expr_by_id(ctx, base, var_expr, start);
        let one = ctx.num(1);
        let end_plus_one = ctx.add(Expr::Add(end, one));
        let end_next_base = substitute_expr_by_id(ctx, base, var_expr, end_plus_one);
        // `base` is the LOWER of the two factors. When it is the denominator the numerator is the
        // higher factor (the product grows): `base(end+1)/base(start)`. When `base` is the numerator
        // the denominator is higher (the product shrinks): the reciprocal `base(start)/base(end+1)`.
        // (The old code always used the first orientation — a reciprocal wrong answer for the
        // denominator-higher case.) `compare_expr` here is on the original factors, not a
        // substituted form, so it is reliable for symbolic affine bases too.
        let numerator_is_higher = compare_expr(ctx, base, den) == std::cmp::Ordering::Equal;
        return Some(if numerator_is_higher {
            ctx.add(Expr::Div(end_next_base, start_base))
        } else {
            ctx.add(Expr::Div(start_base, end_next_base))
        });
    }

    let base = extract_unit_shifted_base(ctx, den, var)?;
    let numerator_expected = shift_expr(ctx, base, 1);
    if compare_expr(ctx, num, numerator_expected) != std::cmp::Ordering::Equal {
        return None;
    }

    let var_expr = ctx.var(var);
    let start_base = substitute_expr_by_id(ctx, base, var_expr, start);
    let end_base = substitute_expr_by_id(ctx, base, var_expr, end);
    let end_plus_one = shift_expr(ctx, end_base, 1);

    Some(ctx.add(Expr::Div(end_plus_one, start_base)))
}

/// Build product closed form for `1 - 1/var^2`:
/// `∏_{start}^{end}(1 - 1/k^2) = ((start-1)*(end+1))/(start*end)`, and the convergent infinite
/// product `∏_{start}^∞ = (start-1)/start`.
pub fn try_build_factorizable_product_for_one_minus_reciprocal_square(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let base = detect_factorized_telescoping_square_base(ctx, factor, var)?;
    let var_expr = ctx.var(var);
    let start_base = substitute_expr_by_id(ctx, base, var_expr, start);

    // Convergent infinite product: the boundary factor `(end_base+1)/end_base → 1`, leaving
    // `(start_base − 1)/start_base`. Emit the limit directly (the ∞-substitution of the finite form
    // `((start-1)(end+1))/(start·end)` reduces to 1 instead of the true limit). Requires a literal
    // integer `start ≥ 1` (the factors `1 - 1/k² ∈ (0,1]` over `[start, ∞)`, so the product
    // converges); a symbolic or non-positive lower bound declines.
    if is_positive_infinity(ctx, end) {
        let start_value = crate::expr_extract::extract_i64_integer(ctx, start)?;
        if start_value < 1 {
            return None;
        }
        let start_minus_1 = shift_expr(ctx, start_base, -1);
        return Some(ctx.add(Expr::Div(start_minus_1, start_base)));
    }

    let end_base = substitute_expr_by_id(ctx, base, var_expr, end);
    let start_minus_1 = shift_expr(ctx, start_base, -1);
    let end_plus_1 = shift_expr(ctx, end_base, 1);

    let combined_num = mul2_raw(ctx, start_minus_1, end_plus_1);
    let combined_den = mul2_raw(ctx, start_base, end_base);
    Some(ctx.add(Expr::Div(combined_num, combined_den)))
}

pub fn detect_factorized_telescoping_square_base(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    detect_one_minus_reciprocal_square_base(ctx, expr, var)
        .or_else(|| detect_factorized_one_minus_reciprocal_square(ctx, expr, var))
}

fn detect_one_minus_reciprocal_square_base(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                return detect_reciprocal_square_base(ctx, *right, var);
            }
        }
    }

    if let Expr::Add(left, right) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*right) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*left) {
                    return detect_reciprocal_square_base(ctx, *inner, var);
                }
            }
        }
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*right) {
                    return detect_reciprocal_square_base(ctx, *inner, var);
                }
            }
        }
    }

    None
}

fn detect_reciprocal_square_base(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*num) {
            if n.is_one() {
                return extract_square_of_unit_shifted_base(ctx, *den, var);
            }
        }
    }

    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Some(power) = negative_integer_exponent(ctx, *exp) {
            if power == 2 {
                return extract_unit_shifted_base(ctx, *base, var);
            }
        }
    }

    None
}

fn detect_factorized_one_minus_reciprocal_square(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let base = extract_square_of_unit_shifted_base(ctx, *den, var)?;
    if !is_square_minus_one_of_base(ctx, *num, base) {
        return None;
    }
    Some(base)
}

pub fn extract_unit_shifted_base(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some(expr),
        Expr::Add(left, right) => {
            if is_named_var(ctx, *left, var) && !contains_named_var(ctx, *right, var) {
                return Some(expr);
            }
            if is_named_var(ctx, *right, var) && !contains_named_var(ctx, *left, var) {
                return Some(expr);
            }
            None
        }
        Expr::Sub(left, right) => {
            if is_named_var(ctx, *left, var) && !contains_named_var(ctx, *right, var) {
                return Some(expr);
            }
            None
        }
        _ => None,
    }
}

fn contains_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => return true,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn is_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
}

/// Decompose `summand` as `c · p(k) · r^k`, with `r` a rational ratio ≠ 0, 1 and `p` a polynomial
/// cofactor of degree 1 or 2 in the index. Returns `(c, r, [γ, β, α])` for `p(k) = α·k² + β·k + γ`.
/// Flattens the product into one geometric factor `r^k` and a polynomial cofactor (the bare `k`,
/// `k²`, constants); a single `Div(p(k), r^k)` leaf (e.g. `k/2^k`) is the geometric written with a
/// negative exponent, which `mul_leaves` does not split, so `len == 1` is admissible. Degree 0 (a
/// pure geometric) and non-polynomial cofactors (`1/k`, `√k`) are declined.
fn decompose_arithmetic_geometric(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational, [BigRational; 3])> {
    let leaves = expr_nary::mul_leaves(ctx, summand);
    if leaves.is_empty() {
        return None;
    }
    let mut geometric: Option<(BigRational, BigRational)> = None;
    let mut cofactor_leaves: Vec<ExprId> = Vec::new();
    for &leaf in &leaves {
        if let Some((coefficient, ratio)) = extract_geometric_term(ctx, leaf, var) {
            if ratio.is_zero() || ratio.is_one() || geometric.is_some() {
                return None;
            }
            geometric = Some((coefficient, ratio));
        } else if let Expr::Div(num, den) = *ctx.get(leaf) {
            // `num / (dc·dr^k) = num · (1/dc)·(1/dr)^k`: the denominator carries the geometric, the
            // numerator joins the polynomial cofactor. Declines (numerator → cofactor) when the
            // denominator is not geometric, so `1/(k+1)` etc. fall through to `from_expr` failure.
            match extract_geometric_term(ctx, den, var) {
                Some((dc, dr)) if !dc.is_zero() && !dr.is_zero() && !dr.is_one() => {
                    if geometric.is_some() {
                        return None;
                    }
                    geometric = Some((dc.recip(), dr.recip()));
                    cofactor_leaves.push(num);
                }
                _ => cofactor_leaves.push(leaf),
            }
        } else {
            cofactor_leaves.push(leaf);
        }
    }
    let (geometric_coefficient, ratio) = geometric?;
    if geometric_coefficient.is_zero() {
        return None;
    }
    let cofactor = product_of_leaves(ctx, &cofactor_leaves)?;
    let poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let degree = poly.degree();
    if degree == 0 || degree > 2 {
        return None;
    }
    let coeff = |i: usize| {
        poly.coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    Some((geometric_coefficient, ratio, [coeff(0), coeff(1), coeff(2)]))
}

/// Closed form for the polynomial-times-geometric sum `Σ_{k=start}^{end} p(k)·c·r^k`, where
/// `p` is a polynomial in the index of degree 1 or 2 and `r` is a rational ratio ≠ 0, 1.
/// Sums by linearity `Σ(α·k² + β·k + γ)·r^k = α·S₂ + β·S₁ + γ·S₀`, with `S₀ = Σr^k`, `S₁ = Σk·r^k`,
/// `S₂ = Σk²·r^k`. Degree 0 (a pure geometric) is declined so the geometric builder owns it.
pub fn try_build_arithmetic_geometric_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if contains_named_var(ctx, start, var) || contains_named_var(ctx, end, var) {
        return None;
    }
    let (geometric_coefficient, ratio, [gamma, beta, alpha]) =
        decompose_arithmetic_geometric(ctx, summand, var)?;

    // α·S₂ + β·S₁ + γ·S₀, then the geometric coefficient out front.
    let mut terms: Vec<ExprId> = Vec::new();
    if !alpha.is_zero() {
        let s2 = arithmetic_geometric_square_closed_form(ctx, &ratio, start, end);
        let coeff = ctx.add(Expr::Number(alpha));
        terms.push(mul2_raw(ctx, coeff, s2));
    }
    if !beta.is_zero() {
        let s1 = arithmetic_geometric_closed_form(ctx, &ratio, start, end);
        let coeff = ctx.add(Expr::Number(beta));
        terms.push(mul2_raw(ctx, coeff, s1));
    }
    if !gamma.is_zero() {
        let s0 = geometric_sum_closed_form(ctx, &ratio, start, end);
        let coeff = ctx.add(Expr::Number(gamma));
        terms.push(mul2_raw(ctx, coeff, s0));
    }
    let mut iter = terms.into_iter();
    let first = iter.next()?;
    let combined = iter.fold(first, |acc, term| ctx.add(Expr::Add(acc, term)));
    let coefficient_expr = ctx.add(Expr::Number(geometric_coefficient));
    Some(mul2_raw(ctx, coefficient_expr, combined))
}

/// Multiply a non-empty slice of factors left to right; `None` if the slice is empty.
fn product_of_leaves(ctx: &mut Context, leaves: &[ExprId]) -> Option<ExprId> {
    let mut iter = leaves.iter().copied();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, leaf| mul2_raw(ctx, acc, leaf)))
}

/// Sum of an additive combination of geometric / arithmetic-geometric terms by LINEARITY:
/// `Σ(g₁ ± g₂ ± …) = Σg₁ ± Σg₂ ± …`. Handles the distributed affine cofactor the engine
/// produces (`(2k+1)·2^k` → `2^k + k·2^(k+1)`). Declines unless EVERY term is summable as a
/// geometric or arithmetic-geometric (so polynomial sums fall through to their own builder).
pub fn try_build_geometric_additive_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(summand), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    let terms = expr_nary::AddView::from_expr(ctx, summand).terms;
    if terms.len() < 2 {
        return None;
    }
    let mut total: Option<ExprId> = None;
    for (term, sign) in terms {
        let term_sum = try_build_arithmetic_geometric_sum(ctx, term, var, start, end)
            .or_else(|| try_build_geometric_rational_sum(ctx, term, var, start, end))?;
        let negative = matches!(sign, expr_nary::Sign::Neg);
        total = Some(match total {
            None if negative => ctx.add(Expr::Neg(term_sum)),
            None => term_sum,
            Some(acc) if negative => ctx.add(Expr::Sub(acc, term_sum)),
            Some(acc) => ctx.add(Expr::Add(acc, term_sum)),
        });
    }
    total
}

/// `Σ_{k=start}^{end} k·r^k = T(end) − T(start−1)` with `T(m) = r·(1 − (m+1)r^m + m·r^(m+1))/(1−r)^2`
/// (the `k = 0` term vanishes, so a `start` of 0 or 1 needs no lower correction).
fn arithmetic_geometric_closed_form(
    ctx: &mut Context,
    ratio: &BigRational,
    start: ExprId,
    end: ExprId,
) -> ExprId {
    let upper = arithmetic_geometric_one_to(ctx, ratio, end);
    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return upper;
    }
    let one = ctx.num(1);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = arithmetic_geometric_one_to(ctx, ratio, start_minus_one);
    ctx.add(Expr::Sub(upper, lower))
}

fn arithmetic_geometric_one_to(ctx: &mut Context, ratio: &BigRational, m: ExprId) -> ExprId {
    let r = ctx.add(Expr::Number(ratio.clone()));
    let one = ctx.num(1);
    let two = ctx.num(2);
    // (m + 1)·r^m
    let m_plus_one = ctx.add(Expr::Add(m, one));
    let r_pow_m = ctx.add(Expr::Pow(r, m));
    let term2 = mul2_raw(ctx, m_plus_one, r_pow_m);
    // m·r^(m+1)
    let m_plus_one_exp = ctx.add(Expr::Add(m, one));
    let r_pow_m1 = ctx.add(Expr::Pow(r, m_plus_one_exp));
    let term3 = mul2_raw(ctx, m, r_pow_m1);
    // 1 − (m+1)r^m + m·r^(m+1)
    let inner = {
        let diff = ctx.add(Expr::Sub(one, term2));
        ctx.add(Expr::Add(diff, term3))
    };
    let numerator = mul2_raw(ctx, r, inner);
    // (1 − r)^2
    let one_minus_r = ctx.add(Expr::Sub(one, r));
    let denominator = ctx.add(Expr::Pow(one_minus_r, two));
    ctx.add(Expr::Div(numerator, denominator))
}

/// `S₀ = Σ_{k=start}^{end} r^k = (r^start − r^(end+1))/(1−r)` (coefficient 1), the geometric
/// partial sum reused by the polynomial-cofactor arithmetic-geometric builder for the `γ` term.
fn geometric_sum_closed_form(
    ctx: &mut Context,
    ratio: &BigRational,
    start: ExprId,
    end: ExprId,
) -> ExprId {
    let r = ctx.add(Expr::Number(ratio.clone()));
    let one = ctx.num(1);
    let r_pow_a = ctx.add(Expr::Pow(r, start));
    let end_plus_one = ctx.add(Expr::Add(end, one));
    let r_pow_np1 = ctx.add(Expr::Pow(r, end_plus_one));
    let diff = ctx.add(Expr::Sub(r_pow_a, r_pow_np1));
    let one_minus_r = ctx.add(Expr::Number(BigRational::one() - ratio));
    ctx.add(Expr::Div(diff, one_minus_r))
}

/// `S₂ = Σ_{k=start}^{end} k²·r^k = U₂(end) − U₂(start−1)` with
/// `U₂(m) = r·(1 + r − (m+1)²·r^m + (2m²+2m−1)·r^(m+1) − m²·r^(m+2))/(1−r)^3`. The `k = 0` term
/// vanishes (`0²·r^0 = 0`), so a `start` of 0 or 1 needs no lower correction (mirrors S₁).
fn arithmetic_geometric_square_closed_form(
    ctx: &mut Context,
    ratio: &BigRational,
    start: ExprId,
    end: ExprId,
) -> ExprId {
    let upper = arithmetic_geometric_square_one_to(ctx, ratio, end);
    if expr_is_zero(ctx, start) || expr_is_one(ctx, start) {
        return upper;
    }
    let one = ctx.num(1);
    let start_minus_one = ctx.add(Expr::Sub(start, one));
    let lower = arithmetic_geometric_square_one_to(ctx, ratio, start_minus_one);
    ctx.add(Expr::Sub(upper, lower))
}

/// `U₂(m) = Σ_{k=1}^{m} k²·r^k = r·(1 + r − (m+1)²·r^m + (2m²+2m−1)·r^(m+1) − m²·r^(m+2))/(1−r)^3`.
fn arithmetic_geometric_square_one_to(ctx: &mut Context, ratio: &BigRational, m: ExprId) -> ExprId {
    let r = ctx.add(Expr::Number(ratio.clone()));
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    // (m+1)²·r^m
    let m_plus_one = ctx.add(Expr::Add(m, one));
    let m_plus_one_sq = ctx.add(Expr::Pow(m_plus_one, two));
    let r_pow_m = ctx.add(Expr::Pow(r, m));
    let term_a = mul2_raw(ctx, m_plus_one_sq, r_pow_m);
    // (2m² + 2m − 1)·r^(m+1)
    let m_sq = ctx.add(Expr::Pow(m, two));
    let two_m_sq = mul2_raw(ctx, two, m_sq);
    let two_m = mul2_raw(ctx, two, m);
    let cofactor = {
        let sum = ctx.add(Expr::Add(two_m_sq, two_m));
        ctx.add(Expr::Sub(sum, one))
    };
    let m_plus_one_exp = ctx.add(Expr::Add(m, one));
    let r_pow_m1 = ctx.add(Expr::Pow(r, m_plus_one_exp));
    let term_b = mul2_raw(ctx, cofactor, r_pow_m1);
    // m²·r^(m+2)
    let m_sq2 = ctx.add(Expr::Pow(m, two));
    let m_plus_two = ctx.add(Expr::Add(m, two));
    let r_pow_m2 = ctx.add(Expr::Pow(r, m_plus_two));
    let term_c = mul2_raw(ctx, m_sq2, r_pow_m2);
    // 1 + r − term_a + term_b − term_c
    let inner = {
        let s1 = ctx.add(Expr::Add(one, r));
        let s2 = ctx.add(Expr::Sub(s1, term_a));
        let s3 = ctx.add(Expr::Add(s2, term_b));
        ctx.add(Expr::Sub(s3, term_c))
    };
    let numerator = mul2_raw(ctx, r, inner);
    // (1 − r)^3
    let one_minus_r = ctx.add(Expr::Sub(one, r));
    let denominator = ctx.add(Expr::Pow(one_minus_r, three));
    ctx.add(Expr::Div(numerator, denominator))
}

fn expr_is_square_of_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    is_named_var(ctx, *base, var) && crate::expr_extract::extract_i64_integer(ctx, *exp) == Some(2)
}

fn expr_is_cube_of_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    is_named_var(ctx, *base, var) && crate::expr_extract::extract_i64_integer(ctx, *exp) == Some(3)
}

fn expr_named_var_positive_integer_power(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_named_var(ctx, *base, var) {
        return None;
    }
    let exponent = crate::expr_extract::extract_i64_integer(ctx, *exp)?;
    (exponent > 1).then_some(exponent)
}

fn extract_square_of_unit_shifted_base(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if crate::expr_extract::extract_i64_integer(ctx, *exp) != Some(2) {
        return None;
    }
    extract_unit_shifted_base(ctx, *base, var)
}

fn negative_integer_exponent(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Neg(inner) = ctx.get(expr) {
        return crate::expr_extract::extract_i64_integer(ctx, *inner);
    }
    crate::expr_extract::extract_i64_integer(ctx, expr)
        .filter(|power| *power < 0)
        .map(|power| -power)
}

fn is_square_minus_one_of_base(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            expr_is_square_of_base(ctx, *left, base) && expr_is_one(ctx, *right)
        }
        Expr::Add(left, right) => {
            (expr_is_square_of_base(ctx, *left, base) && expr_is_negative_one(ctx, *right))
                || (expr_is_square_of_base(ctx, *right, base) && expr_is_negative_one(ctx, *left))
        }
        _ => false,
    }
}

fn expr_is_square_of_base(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    let Expr::Pow(inner, exp) = ctx.get(expr) else {
        return false;
    };
    crate::expr_extract::extract_i64_integer(ctx, *exp) == Some(2)
        && compare_expr(ctx, *inner, base) == std::cmp::Ordering::Equal
}

fn expr_is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(0.into()))
}

fn expr_is_non_positive_integer(ctx: &Context, expr: ExprId) -> bool {
    crate::expr_extract::extract_i64_integer(ctx, expr).is_some_and(|value| value <= 0)
}

fn expr_is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer((-1).into()))
}

fn shift_expr(ctx: &mut Context, base: ExprId, delta: i64) -> ExprId {
    if delta == 0 {
        return base;
    }
    let shift = ctx.num(delta.abs());
    if delta > 0 {
        ctx.add(Expr::Add(base, shift))
    } else {
        ctx.add(Expr::Sub(base, shift))
    }
}

/// Greatest common divisor of two non-negative `i64`, at least 1.
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs().max(1)
}

/// Build the affine factor `d·k − n` (`d ≥ 1`).
fn build_affine_factor(ctx: &mut Context, var: &str, d: i64, n: i64) -> ExprId {
    let var_expr = ctx.var(var);
    let base = if d == 1 {
        var_expr
    } else {
        let coeff = ctx.num(d);
        ctx.add(Expr::Mul(coeff, var_expr))
    };
    shift_expr(ctx, base, -n)
}

/// Factor a quadratic denominator `A·k² + B·k + C` (integer coefficients, `A > 0`) with two distinct
/// rational roots into its linear/affine factors `(d1·k − n1)(d2·k − n2)`. The engine expands a
/// factored denominator like `(k-1)(k+1)` to `k²-1` (and `(2k-1)(2k+1)` to `4k²-1`), so the
/// telescoping builder needs this to recover the factors. Requires `d1·d2 == A` (a primitive
/// factorisation, numerator stays 1 with no leftover scale). Returns `None` for non-integer
/// coefficients, `A ≤ 0`, irreducible (negative discriminant), repeated-root, irrational-root, or
/// non-primitive quadratics — all left as honest residuals.
fn factor_telescoping_quadratic_denominator(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let poly = Polynomial::from_expr(ctx, den, var).ok()?;
    if poly.degree() != 2 {
        return None;
    }
    let as_i64 = |coeff: Option<&BigRational>| -> Option<i64> {
        let value = coeff.cloned().unwrap_or_else(BigRational::zero);
        if !value.is_integer() {
            return None;
        }
        value.to_integer().try_into().ok()
    };
    let a = as_i64(poly.coeffs.get(2))?;
    let b = as_i64(poly.coeffs.get(1))?;
    let c = as_i64(poly.coeffs.first())?;
    if a <= 0 {
        return None; // negative leading coefficient left residual
    }
    let discriminant = b
        .checked_mul(b)?
        .checked_sub(4i64.checked_mul(a)?.checked_mul(c)?)?;
    if discriminant <= 0 {
        return None; // ≤ 0 → repeated root or irreducible over ℝ
    }
    // Exact integer square root (seed with f64, then correct — the keep/drop decision is the exact
    // `root*root == discriminant` test, never the float).
    let mut root = (discriminant as f64).sqrt() as i64;
    while root > 0 && root * root > discriminant {
        root -= 1;
    }
    while (root + 1) * (root + 1) <= discriminant {
        root += 1;
    }
    if root * root != discriminant {
        return None; // irrational roots
    }
    // Roots `(-B ± √disc)/(2A)` as reduced fractions `n/d` with `d > 0`.
    let denom = 2i64.checked_mul(a)?;
    let reduce = |numer: i64| -> (i64, i64) {
        let g = gcd_i64(numer, denom);
        (numer / g, denom / g)
    };
    let (n1, d1) = reduce(-b + root);
    let (n2, d2) = reduce(-b - root);
    if n1 * d2 == n2 * d1 {
        return None; // repeated root
    }
    // The affine factors `(d1·k − n1)(d2·k − n2)` have leading coefficient `d1·d2`, which must equal
    // `A` for them to multiply back to `den` (so the numerator stays 1, no leftover scale factor).
    if d1.checked_mul(d2)? != a {
        return None;
    }
    let factor1 = build_affine_factor(ctx, var, d1, n1);
    let factor2 = build_affine_factor(ctx, var, d2, n2);
    Some((factor1, factor2))
}

/// Build the telescoping closed form for `1/((k+a)(k+a+1)···(k+a+m-1))` — a product of `m ≥ 3`
/// CONSECUTIVE linear factors (`Σ 1/(k(k+1)(k+2)) = 1/4`, `Σ 1/(k(k+1)(k+2)(k+3)) = 1/18`). Uses the
/// higher-order telescope `1/∏_{j=0}^{m-1}(k+a+j) = 1/(m-1)·[g(k) − g(k+1)]` with `g(k) =
/// 1/∏_{j=0}^{m-2}(k+a+j)` (the first `m-1` factors), so `Σ_{k=start}^{end} = 1/(m-1)·[g(start) −
/// g(end+1)]`, and the convergent infinite sum (`start+a ≥ 1`) is `1/(m-1)·g(start)`. Two factors are
/// handled by the dedicated builder; non-consecutive or non-unit-numerator products are declined.
pub fn try_build_telescoping_consecutive_factor_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(summand) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if !matches!(ctx.get(num), Expr::Number(n) if n.is_one()) {
        return None;
    }
    let factors = expr_nary::mul_leaves(ctx, den);
    let m = factors.len();
    if m < 3 {
        return None; // two factors have their own builder
    }
    let mut offsets: Vec<i64> = factors
        .iter()
        .map(|&f| extract_linear_offset(ctx, f, var))
        .collect::<Option<_>>()?;
    offsets.sort_unstable();
    let a = offsets[0];
    if (0..m as i64).any(|j| offsets[j as usize] != a + j) {
        return None; // only CONSECUTIVE factors telescope to this closed form
    }
    // g(anchor) = 1/∏_{j=0}^{m-2}(anchor + lo + j): the reciprocal of the first m-1 factors.
    let reciprocal_first_factors = |ctx: &mut Context, anchor: ExprId, lo: i64| -> ExprId {
        let mut product: Option<ExprId> = None;
        for j in 0..(m as i64 - 1) {
            let term = shift_expr(ctx, anchor, lo + j);
            product = Some(match product {
                None => term,
                Some(acc) => ctx.add(Expr::Mul(acc, term)),
            });
        }
        let product = product.expect("m ≥ 3 yields at least two factors");
        let one = ctx.num(1);
        ctx.add(Expr::Div(one, product))
    };
    let order = ctx.num(m as i64 - 1);

    // Convergent infinite sum: the boundary `g(end+1) → 0`, leaving `1/(m-1)·g(start)`. Emit the
    // limit directly (the ∞-substitution path cannot reduce the degree-(m-1) boundary term).
    // Requires a literal integer `start` with `start+a ≥ 1`: all factors are positive over `[start,
    // ∞)` (no pole), so the positive `~1/k^m` tail converges; otherwise decline (residual).
    if is_positive_infinity(ctx, end) {
        let start_value = crate::expr_extract::extract_i64_integer(ctx, start)?;
        if start_value.checked_add(a)? < 1 {
            return None;
        }
        let g_start = reciprocal_first_factors(ctx, start, a);
        return Some(ctx.add(Expr::Div(g_start, order)));
    }

    // Finite: 1/(m-1)·[g(start) − g(end+1)].
    let g_start = reciprocal_first_factors(ctx, start, a);
    let g_end_next = reciprocal_first_factors(ctx, end, a + 1);
    let diff = ctx.add(Expr::Sub(g_start, g_end_next));
    Some(ctx.add(Expr::Div(diff, order)))
}

/// Build telescoping rational sum closed form for `1/((k+b)*(k+c))`.
///
/// Uses identity:
/// `1/((k+b)(k+c)) = (1/(c-b)) * (1/(k+b) - 1/(k+c))` for `b != c`.
pub fn try_build_telescoping_rational_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(summand) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let Expr::Number(n) = ctx.get(num) else {
        return None;
    };
    if !n.is_one() {
        return None;
    }

    let (factor1, factor2) = match ctx.get(den) {
        Expr::Mul(l, r) => (*l, *r),
        // The engine expands a factored denominator like `(k-1)(k+1)` to `k²-1`; recover the two
        // linear factors so the telescoping logic below can run.
        _ => factor_telescoping_quadratic_denominator(ctx, den, var)?,
    };

    if let Some(base) = detect_consecutive_telescoping_sum_base(ctx, factor1, factor2, var) {
        let var_expr = ctx.var(var);
        let start_base = substitute_expr_by_id(ctx, base, var_expr, start);
        let end_base = substitute_expr_by_id(ctx, base, var_expr, end);
        let end_shifted = shift_expr(ctx, end_base, 1);

        let one1 = ctx.num(1);
        let one2 = ctx.num(1);
        let first_term = ctx.add(Expr::Div(one1, start_base));
        let second_term = ctx.add(Expr::Div(one2, end_shifted));
        return Some(ctx.add(Expr::Sub(first_term, second_term)));
    }

    if let Some((base, coeff)) =
        detect_affine_consecutive_telescoping_sum_base_and_gap(ctx, factor1, factor2, var)
    {
        let var_expr = ctx.var(var);
        let one = ctx.num(1);
        let start_base = substitute_expr_by_id(ctx, base, var_expr, start);
        let end_plus_one = ctx.add(Expr::Add(end, one));
        let end_next_base = substitute_expr_by_id(ctx, base, var_expr, end_plus_one);

        let one1 = ctx.num(1);
        let one2 = ctx.num(1);
        let first_term = ctx.add(Expr::Div(one1, start_base));
        let second_term = ctx.add(Expr::Div(one2, end_next_base));
        let diff = ctx.add(Expr::Sub(first_term, second_term));
        return Some(ctx.add(Expr::Div(diff, coeff)));
    }

    let (offset1, offset2) = match (
        extract_linear_offset(ctx, factor1, var),
        extract_linear_offset(ctx, factor2, var),
    ) {
        (Some(offset1), Some(offset2)) => (offset1, offset2),
        _ => return None,
    };
    if offset1 == offset2 {
        return None; // a repeated factor `1/(k+b)^2` is not telescoping
    }

    // `1/((k+b)(k+c)) = 1/(c-b)·(1/(k+b) − 1/(k+c))`. With `b` the SMALLER offset and gap `d = c−b`,
    // `Σ_{k=start}^{end} (1/(k+b) − 1/(k+b+d))` telescopes leaving `d` boundary terms at EACH end:
    //   (1/d)·[ Σ_{j=0}^{d-1} 1/(start+b+j) − Σ_{j=0}^{d-1} 1/(end+b+1+j) ].
    // The previous single-term form (one boundary term per end) was correct ONLY for `d = 1` and
    // produced WRONG values for wider gaps (e.g. `Σ 1/(k(k+2))` gave 1/2 instead of 3/4).
    let b = offset1.min(offset2);
    let gap = (offset1 - offset2).unsigned_abs() as i64;
    let mut boundary = |anchor: ExprId, base_shift: i64| -> ExprId {
        let mut sum: Option<ExprId> = None;
        for j in 0..gap {
            let denom = shift_expr(ctx, anchor, base_shift + j);
            let one = ctx.num(1);
            let term = ctx.add(Expr::Div(one, denom));
            sum = Some(match sum {
                None => term,
                Some(acc) => ctx.add(Expr::Add(acc, term)),
            });
        }
        sum.expect("gap >= 1 yields at least one boundary term")
    };
    let start_sum = boundary(start, b);
    let end_sum = boundary(end, b + 1);
    let diff = ctx.add(Expr::Sub(start_sum, end_sum));

    let result = if gap == 1 {
        diff
    } else {
        let gap_expr = ctx.num(gap);
        ctx.add(Expr::Div(diff, gap_expr))
    };
    Some(result)
}

fn detect_consecutive_telescoping_sum_base(
    ctx: &mut Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<ExprId> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let Some(base) = extract_unit_shifted_base(ctx, base_candidate, var) else {
            continue;
        };
        let shifted = shift_expr(ctx, base, 1);
        if compare_expr(ctx, other_factor, shifted) == std::cmp::Ordering::Equal {
            return Some(base);
        }
    }
    None
}

fn detect_affine_consecutive_telescoping_sum_base_and_gap(
    ctx: &mut Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let (coeff, offset) = extract_affine_var_coeff_and_offset(ctx, base_candidate, var)?;
        let (other_coeff, other_offset) =
            extract_affine_var_coeff_and_offset(ctx, other_factor, var)?;
        if compare_expr(ctx, coeff, other_coeff) != std::cmp::Ordering::Equal {
            continue;
        }
        if additive_gap_relation_holds(ctx, offset, coeff, other_offset) {
            return Some((base_candidate, coeff));
        }
    }
    None
}

fn additive_gap_relation_holds(ctx: &Context, base: ExprId, gap: ExprId, target: ExprId) -> bool {
    let (base_terms, base_constant) = additive_signature(ctx, base);
    let (gap_terms, gap_constant) = additive_signature(ctx, gap);
    let (target_terms, target_constant) = additive_signature(ctx, target);

    let mut combined_terms = base_terms;
    for (basis, coeff) in gap_terms {
        if let Some((_, existing_coeff)) = combined_terms
            .iter_mut()
            .find(|(existing_basis, _)| same_basis(ctx, existing_basis, &basis))
        {
            *existing_coeff += coeff.clone();
        } else {
            combined_terms.push((basis, coeff));
        }
    }
    combined_terms.retain(|(_, coeff)| *coeff != BigRational::from_integer(0.into()));
    sort_signature_terms(ctx, &mut combined_terms);

    combined_terms == target_terms && base_constant + gap_constant == target_constant
}

fn additive_signature(
    ctx: &Context,
    expr: ExprId,
) -> (Vec<(Vec<ExprId>, BigRational)>, BigRational) {
    let mut terms: Vec<(Vec<ExprId>, BigRational)> = Vec::new();
    let mut constant = BigRational::from_integer(0.into());

    for (term, sign) in expr_nary::AddView::from_expr(ctx, expr).terms {
        if let Some(value) = as_rational_const(ctx, term, 8) {
            match sign {
                expr_nary::Sign::Pos => constant += value,
                expr_nary::Sign::Neg => constant -= value,
            }
            continue;
        }

        let (basis, coeff) = scaled_term_signature(ctx, term);
        let signed_coeff = match sign {
            expr_nary::Sign::Pos => coeff,
            expr_nary::Sign::Neg => -coeff,
        };
        if let Some((_, existing_coeff)) = terms
            .iter_mut()
            .find(|(existing_basis, _)| same_basis(ctx, existing_basis, &basis))
        {
            *existing_coeff += signed_coeff;
        } else {
            terms.push((basis, signed_coeff));
        }
    }

    terms.retain(|(_, coeff)| *coeff != BigRational::from_integer(0.into()));
    sort_signature_terms(ctx, &mut terms);

    (terms, constant)
}

fn scaled_term_signature(ctx: &Context, expr: ExprId) -> (Vec<ExprId>, BigRational) {
    let factors = expr_nary::mul_leaves(ctx, expr);
    let mut numeric_coeff = BigRational::from_integer(1.into());
    let mut basis = Vec::new();

    for factor in factors {
        if let Some(value) = as_rational_const(ctx, factor, 8) {
            numeric_coeff *= value;
        } else {
            basis.push(factor);
        }
    }

    basis.sort_by(|left, right| compare_expr(ctx, *left, *right));
    if basis.is_empty() {
        basis.push(expr);
    }
    (basis, numeric_coeff)
}

fn same_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| compare_expr(ctx, *l, *r) == std::cmp::Ordering::Equal)
}

fn sort_signature_terms(ctx: &Context, terms: &mut [(Vec<ExprId>, BigRational)]) {
    terms.sort_by(|(left_basis, _), (right_basis, _)| compare_basis(ctx, left_basis, right_basis));
}

fn compare_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let ord = compare_expr(ctx, *l, *r);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    left.len().cmp(&right.len())
}

fn extract_affine_var_coeff_and_offset(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let mut coeff = None;
    let mut offset_terms = Vec::new();

    for (term, sign) in expr_nary::AddView::from_expr(ctx, expr).terms {
        if contains_named_var(ctx, term, var) {
            let (term_coeff, term_offset) =
                extract_affine_linear_term_coeff_and_offset(ctx, term, var)?;
            if as_rational_const(ctx, term_offset, 8)
                .is_none_or(|value| value != BigRational::from_integer(0.into()))
            {
                return None;
            }

            let signed_coeff = match sign {
                expr_nary::Sign::Pos => term_coeff,
                expr_nary::Sign::Neg => ctx.add(Expr::Neg(term_coeff)),
            };
            if coeff.is_some() {
                return None;
            }
            coeff = Some(signed_coeff);
        } else {
            offset_terms.push((term, sign));
        }
    }

    Some((coeff?, build_signed_additive_expr(ctx, &offset_terms)))
}

fn extract_affine_linear_term_coeff_and_offset(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    if is_named_var(ctx, expr, var) {
        return Some((ctx.num(1), ctx.num(0)));
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    let mut saw_var = false;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if is_named_var(ctx, factor, var) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if contains_named_var(ctx, factor, var) {
            return None;
        } else {
            coeff_factors.push(factor);
        }
    }

    if !saw_var {
        return None;
    }

    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else if coeff_factors.len() == 1 {
        coeff_factors[0]
    } else {
        expr_nary::build_balanced_mul(ctx, &coeff_factors)
    };
    Some((coeff, ctx.num(0)))
}

fn build_signed_additive_expr(ctx: &mut Context, terms: &[(ExprId, expr_nary::Sign)]) -> ExprId {
    let mut iter = terms.iter();
    let Some(&(first_term, first_sign)) = iter.next() else {
        return ctx.num(0);
    };

    let mut result = match first_sign {
        expr_nary::Sign::Pos => first_term,
        expr_nary::Sign::Neg => ctx.add(Expr::Neg(first_term)),
    };

    for &(term, sign) in iter {
        result = match sign {
            expr_nary::Sign::Pos => ctx.add(Expr::Add(result, term)),
            expr_nary::Sign::Neg => ctx.add(Expr::Sub(result, term)),
        };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::{
        build_finite_product_substitution, build_finite_sum_substitution,
        detect_one_minus_reciprocal_power, detect_reciprocal_power, extract_linear_offset,
        try_build_factorizable_product_for_one_minus_reciprocal_square,
        try_build_geometric_power_sum, try_build_product_of_constant,
        try_build_product_of_first_integers, try_build_product_of_powers,
        try_build_sum_of_constant, try_build_sum_of_cubes, try_build_sum_of_first_integers,
        try_build_sum_of_squares, try_build_telescoping_product_shift1,
        try_build_telescoping_rational_sum, try_extract_bounded_integer_range,
        try_extract_finite_aggregate_call, try_plan_finite_product_evaluation,
        try_plan_finite_sum_evaluation, ProductEvaluationKind, SumEvaluationKind,
    };
    use cas_ast::ordering::compare_expr;
    use cas_ast::{substitute_expr_by_id, Context, Expr};
    use num_rational::BigRational;

    fn eval_small_int(ctx: &Context, id: cas_ast::ExprId) -> Option<i64> {
        match ctx.get(id) {
            Expr::Number(_) => crate::expr_extract::extract_i64_integer(ctx, id),
            Expr::Add(l, r) => Some(eval_small_int(ctx, *l)? + eval_small_int(ctx, *r)?),
            Expr::Sub(l, r) => Some(eval_small_int(ctx, *l)? - eval_small_int(ctx, *r)?),
            Expr::Mul(l, r) => Some(eval_small_int(ctx, *l)? * eval_small_int(ctx, *r)?),
            Expr::Neg(inner) => Some(-eval_small_int(ctx, *inner)?),
            _ => None,
        }
    }

    fn eval_small_rat(ctx: &Context, id: cas_ast::ExprId) -> Option<BigRational> {
        match ctx.get(id) {
            Expr::Number(n) => Some(n.clone()),
            Expr::Add(l, r) => Some(eval_small_rat(ctx, *l)? + eval_small_rat(ctx, *r)?),
            Expr::Sub(l, r) => Some(eval_small_rat(ctx, *l)? - eval_small_rat(ctx, *r)?),
            Expr::Mul(l, r) => Some(eval_small_rat(ctx, *l)? * eval_small_rat(ctx, *r)?),
            Expr::Div(l, r) => {
                let den = eval_small_rat(ctx, *r)?;
                if den == BigRational::from_integer(0.into()) {
                    None
                } else {
                    Some(eval_small_rat(ctx, *l)? / den)
                }
            }
            Expr::Pow(base, exponent) => {
                let exponent = crate::expr_extract::extract_i64_integer(ctx, *exponent)?;
                if exponent < 0 {
                    return None;
                }
                let base = eval_small_rat(ctx, *base)?;
                let mut result = BigRational::from_integer(1.into());
                for _ in 0..exponent {
                    result *= base.clone();
                }
                Some(result)
            }
            Expr::Neg(inner) => Some(-eval_small_rat(ctx, *inner)?),
            _ => None,
        }
    }

    #[test]
    fn extracts_offsets_from_linear_forms() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let c3 = ctx.num(3);
        let c5 = ctx.num(5);

        let k_plus_3 = ctx.add(Expr::Add(k, c3));
        let five_plus_k = ctx.add(Expr::Add(c5, k));
        let k_minus_3 = ctx.add(Expr::Sub(k, c3));

        assert_eq!(extract_linear_offset(&ctx, k, "k"), Some(0));
        assert_eq!(extract_linear_offset(&ctx, k_plus_3, "k"), Some(3));
        assert_eq!(extract_linear_offset(&ctx, five_plus_k, "k"), Some(5));
        assert_eq!(extract_linear_offset(&ctx, k_minus_3, "k"), Some(-3));
    }

    #[test]
    fn detects_reciprocal_power_forms() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let two = ctx.num(2);
        let k_sq = ctx.add(Expr::Pow(k, two));
        let one = ctx.num(1);
        let inv_k_sq = ctx.add(Expr::Div(one, k_sq));
        let neg_two = ctx.add(Expr::Neg(two));
        let k_neg_two = ctx.add(Expr::Pow(k, neg_two));

        assert_eq!(detect_reciprocal_power(&ctx, inv_k_sq, "k"), Some(2));
        assert_eq!(detect_reciprocal_power(&ctx, k_neg_two, "k"), Some(2));
    }

    #[test]
    fn detects_one_minus_reciprocal_power_forms() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let two = ctx.num(2);
        let k_sq = ctx.add(Expr::Pow(k, two));
        let one = ctx.num(1);
        let inv_k_sq = ctx.add(Expr::Div(one, k_sq));
        let sub_form = ctx.add(Expr::Sub(one, inv_k_sq));
        let neg_inv = ctx.add(Expr::Neg(inv_k_sq));
        let add_neg_form = ctx.add(Expr::Add(one, neg_inv));

        assert_eq!(
            detect_one_minus_reciprocal_power(&ctx, sub_form, "k"),
            Some(2)
        );
        assert_eq!(
            detect_one_minus_reciprocal_power(&ctx, add_neg_form, "k"),
            Some(2)
        );
    }

    #[test]
    fn extracts_finite_aggregate_call_shape() {
        let mut ctx = Context::new();
        let term = ctx.var("k");
        let var = ctx.var("k");
        let one = ctx.num(1);
        let ten = ctx.num(10);
        let expr = ctx.call("sum", vec![term, var, one, ten]);

        let parsed = try_extract_finite_aggregate_call(&ctx, expr, "sum").expect("parse");
        assert_eq!(parsed.term, term);
        assert_eq!(parsed.var_expr, var);
        assert_eq!(parsed.var_name, "k");
        assert_eq!(parsed.start_expr, one);
        assert_eq!(parsed.end_expr, ten);
    }

    #[test]
    fn extracts_bounded_integer_range() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let five = ctx.num(5);
        let six = ctx.num(6);
        assert_eq!(
            try_extract_bounded_integer_range(&ctx, one, five, 10),
            Some((1, 5))
        );
        assert_eq!(
            try_extract_bounded_integer_range(&ctx, one, six, 3),
            None,
            "range exceeds bound"
        );
    }

    #[test]
    fn builds_finite_sum_and_product_by_substitution() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let three = ctx.num(3);

        let sum_expr = build_finite_sum_substitution(&mut ctx, k, k, 1, 3);
        let sum_value = eval_small_int(&ctx, sum_expr).expect("sum int");
        assert_eq!(sum_value, 6);

        let product_expr = build_finite_product_substitution(&mut ctx, k, k, 1, 3);
        let product_value = eval_small_int(&ctx, product_expr).expect("product int");
        assert_eq!(product_value, 6);

        // Keep values alive and ensure no accidental dependence on unused locals.
        assert_eq!(
            try_extract_bounded_integer_range(&ctx, one, three, 10),
            Some((1, 3))
        );
    }

    #[test]
    fn builds_telescoping_product_shift1_result() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let four = ctx.num(4);
        let k_plus_1 = ctx.add(Expr::Add(k, one));
        let factor = ctx.add(Expr::Div(k_plus_1, k));

        let result =
            try_build_telescoping_product_shift1(&mut ctx, factor, "k", two, four).expect("build");
        let Expr::Div(num, den) = ctx.get(result) else {
            panic!("expected division");
        };
        assert_eq!(eval_small_int(&ctx, *num), Some(5));
        assert_eq!(eval_small_int(&ctx, *den), Some(2));
    }

    #[test]
    fn telescoping_product_respects_numerator_denominator_orientation() {
        // P0 regression: `∏ k/(k+1) = 1/(n+1)`, NOT `n+1` — the old builder returned the same
        // `n+1` for both `k/(k+1)` and `(k+1)/k` (a reciprocal wrong answer for every
        // denominator-higher product). Covers integer and affine factors, both orientations.
        type Factor = fn(i64) -> (i64, i64);
        let cases: [(&str, Factor); 4] = [
            ("k/(k+1)", |k| (k, k + 1)),
            ("(k+1)/k", |k| (k + 1, k)),
            ("(2*k+1)/(2*k+3)", |k| (2 * k + 1, 2 * k + 3)),
            ("(2*k+3)/(2*k+1)", |k| (2 * k + 3, 2 * k + 1)),
        ];
        for (expr, factor) in cases {
            for (a, n) in [(1i64, 5i64), (2, 7)] {
                let mut ctx = Context::new();
                let parsed = cas_parser::parse(expr, &mut ctx).expect("parse");
                let start = ctx.num(a);
                let end = ctx.num(n);
                let result =
                    try_build_telescoping_product_shift1(&mut ctx, parsed, "k", start, end)
                        .expect("build");
                let mut brute = BigRational::from_integer(1.into());
                for k in a..=n {
                    let (numer, denom) = factor(k);
                    brute *= BigRational::new(numer.into(), denom.into());
                }
                assert_eq!(
                    eval_small_rat(&ctx, result),
                    Some(brute),
                    "{expr} [{a},{n}]"
                );
            }
        }
    }

    #[test]
    fn builds_telescoping_product_shift1_result_with_affine_symbolic_base() {
        let mut ctx = Context::new();
        let factor =
            cas_parser::parse("(a*k+b+a)/(a*k+b)", &mut ctx).expect("parse affine symbolic factor");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let a = ctx.var("a");
        let b = ctx.var("b");

        let result =
            try_build_telescoping_product_shift1(&mut ctx, factor, "k", one, n).expect("build");
        let two = ctx.num(2);
        let one_num = ctx.num(1);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, result, a, two);
        let result = substitute_expr_by_id(&mut ctx, result, b, one_num);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(11.into(), 3.into()))
        );
    }

    #[test]
    fn builds_telescoping_product_shift1_result_with_affine_symbolic_arbitrary_shift() {
        let mut ctx = Context::new();
        let factor = cas_parser::parse("(a*k+b+c+a)/(a*k+b+c)", &mut ctx)
            .expect("parse shifted affine symbolic factor");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let result =
            try_build_telescoping_product_shift1(&mut ctx, factor, "k", one, n).expect("build");
        let two = ctx.num(2);
        let one_num = ctx.num(1);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, result, a, two);
        let result = substitute_expr_by_id(&mut ctx, result, b, one_num);
        let result = substitute_expr_by_id(&mut ctx, result, c, three);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(7.into(), 3.into()))
        );
    }

    #[test]
    fn builds_factorizable_product_square_result() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let four = ctx.num(4);
        let k_sq = ctx.add(Expr::Pow(k, two));
        let inv_k_sq = ctx.add(Expr::Div(one, k_sq));
        let factor = ctx.add(Expr::Sub(one, inv_k_sq));

        let result = try_build_factorizable_product_for_one_minus_reciprocal_square(
            &mut ctx, factor, "k", two, four,
        )
        .expect("build");
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(5.into(), 8.into()))
        );

        // Convergent infinite product `∏_{k=2}^∞ (1 - 1/k²) = (2-1)/2 = 1/2` (the ∞-substitution of
        // the finite form wrongly gave 1).
        let two = ctx.num(2);
        let inf = super::make_infinity(&mut ctx);
        let infinite = try_build_factorizable_product_for_one_minus_reciprocal_square(
            &mut ctx, factor, "k", two, inf,
        )
        .expect("convergent infinite product");
        assert_eq!(
            eval_small_rat(&ctx, infinite),
            Some(BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn builds_factorizable_product_square_result_after_fraction_simplification() {
        let mut ctx = Context::new();
        let factor = cas_parser::parse("(k^2 - 1)/k^2", &mut ctx).expect("factor");
        let two = ctx.num(2);
        let n = ctx.var("n");

        let result = try_build_factorizable_product_for_one_minus_reciprocal_square(
            &mut ctx, factor, "k", two, n,
        )
        .expect("build");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(
            rendered.contains("n + 1") && rendered.contains("2") && rendered.contains("n"),
            "expected telescoping closed form, got {rendered}"
        );
    }

    #[test]
    fn builds_factorizable_product_square_result_with_shifted_base() {
        let mut ctx = Context::new();
        let factor = cas_parser::parse("1 - 1/(k+2)^2", &mut ctx).expect("factor");
        let one = ctx.num(1);
        let five = ctx.num(5);

        let result = try_build_factorizable_product_for_one_minus_reciprocal_square(
            &mut ctx, factor, "k", one, five,
        )
        .expect("build");
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(16.into(), 21.into()))
        );
    }

    #[test]
    fn builds_telescoping_rational_sum_result() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let four = ctx.num(4);
        let k_plus_1 = ctx.add(Expr::Add(k, one));
        let den = ctx.add(Expr::Mul(k, k_plus_1));
        let summand = ctx.add(Expr::Div(one, den));

        let result =
            try_build_telescoping_rational_sum(&mut ctx, summand, "k", one, four).expect("build");
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(4.into(), 5.into()))
        );
    }

    #[test]
    fn telescoping_rational_sum_wide_gap_matches_brute_force() {
        // Regression for the gap ≥ 2 wrong-answer: the closed form must keep `gap` boundary terms at
        // EACH end, so `Σ_{1}^{n} 1/(k(k+2))` is `(1/2)(3/2 − 1/(n+1) − 1/(n+2))`, not the
        // single-term `(n+1)/(2(n+2))` (which wrongly limits to 1/2 instead of 3/4).
        for (p, q) in [(0i64, 2i64), (0, 3), (0, 4), (1, 3), (2, 5)] {
            for (a, n) in [(1i64, 6i64), (2, 9), (3, 7)] {
                let mut ctx = Context::new();
                let summand =
                    cas_parser::parse(&format!("1/((k+{p})*(k+{q}))"), &mut ctx).expect("parse");
                let start = ctx.num(a);
                let end = ctx.num(n);
                let result = try_build_telescoping_rational_sum(&mut ctx, summand, "k", start, end)
                    .expect("build");
                let mut brute = BigRational::from_integer(0.into());
                for k in a..=n {
                    brute += BigRational::new(1.into(), ((k + p) * (k + q)).into());
                }
                assert_eq!(
                    eval_small_rat(&ctx, result),
                    Some(brute),
                    "Σ 1/((k+{p})(k+{q})) over [{a},{n}]"
                );
            }
        }
    }

    #[test]
    fn telescoping_rational_sum_factors_expanded_quadratic_denominator() {
        // The engine expands a factored denominator like `(k-1)(k+1)` to `k²-1`; the builder must
        // factor the monic quadratic back into linear factors and telescope (the gap-≥2 cases also
        // exercise the fixed multi-boundary-term formula).
        let den = |expr: &str, k: i64| -> i64 {
            match expr {
                "k^2-1" => k * k - 1,
                "k^2-4" => k * k - 4,
                "k^2+3*k+2" => k * k + 3 * k + 2,
                "k^2-5*k+6" => k * k - 5 * k + 6,
                _ => unreachable!(),
            }
        };
        for expr in ["k^2-1", "k^2-4", "k^2+3*k+2", "k^2-5*k+6"] {
            for (a, n) in [(4i64, 9i64), (5, 11)] {
                let mut ctx = Context::new();
                let summand = cas_parser::parse(&format!("1/({expr})"), &mut ctx).expect("parse");
                let start = ctx.num(a);
                let end = ctx.num(n);
                let result = try_build_telescoping_rational_sum(&mut ctx, summand, "k", start, end)
                    .expect("build");
                let mut brute = BigRational::from_integer(0.into());
                for k in a..=n {
                    brute += BigRational::new(1.into(), den(expr, k).into());
                }
                assert_eq!(
                    eval_small_rat(&ctx, result),
                    Some(brute),
                    "1/({expr}) [{a},{n}]"
                );
            }
        }
        // Irreducible (`k²+1`, `k²+k+1`), non-primitive (`2k²-2`, factors `(k-1)(k+1)` don't carry
        // the leading 2), and non-telescoping affine (`9k²-1` → `(3k-1)(3k+1)`, an index gap of 2 in
        // the affine base, declined by the affine path) quadratics stay residual.
        for expr in ["k^2+1", "k^2+k+1", "2*k^2-2", "9*k^2-1"] {
            let mut ctx = Context::new();
            let summand = cas_parser::parse(&format!("1/({expr})"), &mut ctx).expect("parse");
            let one = ctx.num(1);
            let n = ctx.num(5);
            assert!(
                try_build_telescoping_rational_sum(&mut ctx, summand, "k", one, n).is_none(),
                "1/({expr}) must stay residual"
            );
        }
    }

    #[test]
    fn telescoping_three_factor_sum_matches_brute_force() {
        // `1/((k+a)(k+a+1)(k+a+2))` — three consecutive linear factors (the classic
        // `Σ 1/(k(k+1)(k+2))`) — telescopes via the second-order identity. Finite bounds only.
        let den = |offsets: (i64, i64, i64), k: i64| -> i64 {
            (k + offsets.0) * (k + offsets.1) * (k + offsets.2)
        };
        for offsets in [(0i64, 1i64, 2i64), (1, 2, 3), (2, 3, 4)] {
            for (a, n) in [(1i64, 6i64), (3, 9)] {
                let mut ctx = Context::new();
                let summand = cas_parser::parse(
                    &format!("1/((k+{})*(k+{})*(k+{}))", offsets.0, offsets.1, offsets.2),
                    &mut ctx,
                )
                .expect("parse");
                let start = ctx.num(a);
                let end = ctx.num(n);
                let result = super::try_build_telescoping_consecutive_factor_sum(
                    &mut ctx, summand, "k", start, end,
                )
                .expect("build");
                let mut brute = BigRational::from_integer(0.into());
                for k in a..=n {
                    brute += BigRational::new(1.into(), den(offsets, k).into());
                }
                assert_eq!(
                    eval_small_rat(&ctx, result),
                    Some(brute),
                    "3-factor {offsets:?} [{a},{n}]"
                );
            }
        }
        // Non-consecutive factors and an infinite bound are declined (honest residuals).
        let mut ctx = Context::new();
        let nonconsec = cas_parser::parse("1/(k*(k+1)*(k+3))", &mut ctx).expect("parse");
        let one = ctx.num(1);
        let n = ctx.num(5);
        assert!(
            super::try_build_telescoping_consecutive_factor_sum(&mut ctx, nonconsec, "k", one, n)
                .is_none(),
            "non-consecutive 3-factor must stay residual"
        );
        // Convergent infinite sum (start+a ≥ 1): `Σ_{k=1}^∞ 1/(k(k+1)(k+2)) = (1/2)·1/(1·2) = 1/4`.
        let consec = cas_parser::parse("1/(k*(k+1)*(k+2))", &mut ctx).expect("parse");
        let one = ctx.num(1);
        let inf = super::make_infinity(&mut ctx);
        let infinite =
            super::try_build_telescoping_consecutive_factor_sum(&mut ctx, consec, "k", one, inf)
                .expect("convergent infinite sum");
        assert_eq!(
            eval_small_rat(&ctx, infinite),
            Some(BigRational::new(1.into(), 4.into()))
        );
        // A lower bound that crosses a pole (start+a < 1, here `k=0`) declines — the term is
        // undefined, not a convergent tail.
        let consec = cas_parser::parse("1/(k*(k+1)*(k+2))", &mut ctx).expect("parse");
        let zero = ctx.num(0);
        let inf = super::make_infinity(&mut ctx);
        assert!(
            super::try_build_telescoping_consecutive_factor_sum(&mut ctx, consec, "k", zero, inf)
                .is_none(),
            "a lower bound at a pole must decline"
        );
    }

    #[test]
    fn telescoping_consecutive_factor_sum_generalizes_to_m_factors() {
        // `1/∏_{j=0}^{m-1}(k+a+j)` for m = 4, 5 consecutive factors: `Σ 1/(k(k+1)(k+2)(k+3)) = 1/18`,
        // `Σ 1/(k…(k+4)) = 1/96`. Finite folds match brute force; the infinite value is `1/(m-1)·g(1)`.
        let products: [(&str, &[i64]); 3] = [
            ("(k)*(k+1)*(k+2)*(k+3)", &[0, 1, 2, 3]),
            ("(k+1)*(k+2)*(k+3)*(k+4)", &[1, 2, 3, 4]),
            ("(k)*(k+1)*(k+2)*(k+3)*(k+4)", &[0, 1, 2, 3, 4]),
        ];
        for (expr, offsets) in products {
            let den = |k: i64| -> i64 { offsets.iter().map(|&o| k + o).product() };
            for (a, n) in [(1i64, 6i64), (2, 8)] {
                let mut ctx = Context::new();
                let summand = cas_parser::parse(&format!("1/({expr})"), &mut ctx).expect("parse");
                let start = ctx.num(a);
                let end = ctx.num(n);
                let result = super::try_build_telescoping_consecutive_factor_sum(
                    &mut ctx, summand, "k", start, end,
                )
                .expect("build");
                let mut brute = BigRational::from_integer(0.into());
                for k in a..=n {
                    brute += BigRational::new(1.into(), den(k).into());
                }
                assert_eq!(
                    eval_small_rat(&ctx, result),
                    Some(brute),
                    "{expr} [{a},{n}]"
                );
            }
        }
        // `Σ_{k=1}^∞ 1/(k(k+1)(k+2)(k+3)) = (1/3)·1/(1·2·3) = 1/18`.
        let mut ctx = Context::new();
        let summand = cas_parser::parse("1/(k*(k+1)*(k+2)*(k+3))", &mut ctx).expect("parse");
        let one = ctx.num(1);
        let inf = super::make_infinity(&mut ctx);
        let infinite =
            super::try_build_telescoping_consecutive_factor_sum(&mut ctx, summand, "k", one, inf)
                .expect("convergent infinite 4-factor sum");
        assert_eq!(
            eval_small_rat(&ctx, infinite),
            Some(BigRational::new(1.into(), 18.into()))
        );
    }

    #[test]
    fn telescoping_rational_sum_factors_non_monic_affine_denominator() {
        // `4k²-1 = (2k-1)(2k+1)`: the difference of squares (which the engine keeps expanded)
        // telescopes via the affine path — the classic `Σ 1/(4k²-1) = 1/2`.
        for (a, n) in [(1i64, 7i64), (2, 9)] {
            let mut ctx = Context::new();
            let summand = cas_parser::parse("1/(4*k^2-1)", &mut ctx).expect("parse");
            let start = ctx.num(a);
            let end = ctx.num(n);
            let result = try_build_telescoping_rational_sum(&mut ctx, summand, "k", start, end)
                .expect("build");
            let mut brute = BigRational::from_integer(0.into());
            for k in a..=n {
                brute += BigRational::new(1.into(), (4 * k * k - 1).into());
            }
            assert_eq!(
                eval_small_rat(&ctx, result),
                Some(brute),
                "Σ 1/(4k²-1) [{a},{n}]"
            );
        }
    }

    #[test]
    fn builds_telescoping_rational_sum_result_with_symbolic_shift() {
        let mut ctx = Context::new();
        let summand = cas_parser::parse("1/((k+a)*(k+a+1))", &mut ctx).expect("summand");
        let one = ctx.num(1);
        let n = ctx.var("n");

        let result =
            try_build_telescoping_rational_sum(&mut ctx, summand, "k", one, n).expect("build");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(
            rendered.contains("a + 1")
                && (rendered.contains("a + n + 1") || rendered.contains("n + a + 1")),
            "expected symbolic telescoping closed form, got {rendered}"
        );
    }

    #[test]
    fn builds_telescoping_rational_sum_result_with_affine_symbolic_shift() {
        let mut ctx = Context::new();
        let summand = cas_parser::parse("1/((a*k+b)*(a*k+b+a))", &mut ctx).expect("summand");
        let one = ctx.num(1);
        let n = ctx.var("n");

        let result =
            try_build_telescoping_rational_sum(&mut ctx, summand, "k", one, n).expect("build");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: result
            }
        );
        assert!(
            (rendered.contains("1 / a") || rendered.contains(")/a") || rendered.contains(") / a"))
                && (rendered.contains("a + b")
                    || rendered.contains("b + a")
                    || rendered.contains("b + 1 * a"))
                && (rendered.contains("a·n + a + b")
                    || rendered.contains("a·n + b + a")
                    || rendered.contains("a + a·n + b")
                    || rendered.contains("a * (n + 1) + b")),
            "expected affine symbolic telescoping closed form, got {rendered}"
        );
    }

    #[test]
    fn builds_telescoping_rational_sum_result_with_affine_symbolic_arbitrary_shift() {
        let mut ctx = Context::new();
        let summand = cas_parser::parse("1/((a*k+b+c)*(a*k+b+c+a))", &mut ctx).expect("summand");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let result =
            try_build_telescoping_rational_sum(&mut ctx, summand, "k", one, n).expect("build");
        let two = ctx.num(2);
        let one_num = ctx.num(1);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, result, a, two);
        let result = substitute_expr_by_id(&mut ctx, result, b, one_num);
        let result = substitute_expr_by_id(&mut ctx, result, c, three);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::new(1.into(), 21.into()))
        );
    }

    #[test]
    fn sum_evaluation_plan_prefers_telescoping_then_finite_direct() {
        let mut ctx = Context::new();

        let telescoping = cas_parser::parse("sum(1/(k*(k+1)), k, 1, 4)", &mut ctx).expect("sum");
        let plan1 =
            try_plan_finite_sum_evaluation(&mut ctx, telescoping, 1000).expect("telescoping");
        assert!(matches!(plan1.kind, SumEvaluationKind::Telescoping));

        let finite = cas_parser::parse("sum(k^2, k, 1, 5)", &mut ctx).expect("sum");
        let plan2 = try_plan_finite_sum_evaluation(&mut ctx, finite, 1000).expect("finite");
        assert!(matches!(
            plan2.kind,
            SumEvaluationKind::FiniteDirect { start: 1, end: 5 }
        ));
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_first_integers() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k, k, 1, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("n*(n+1)/2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfFirstIntegers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn infinite_bound_classifies_divergence_instead_of_finite_formula() {
        // Round-4 Cluster L: an infinite upper bound must NOT substitute into a
        // finite closed form (which leaked 1/2*infinity^2, 2^infinity, infinity!).
        let mut ctx = Context::new();
        for s in [
            "sum(k, k, 1, inf)",
            "sum(k^3, k, 1, inf)",
            "sum(2^k, k, 0, inf)",
        ] {
            let e = cas_parser::parse(s, &mut ctx).unwrap_or_else(|_| panic!("parse {s}"));
            let plan = try_plan_finite_sum_evaluation(&mut ctx, e, 1000)
                .unwrap_or_else(|| panic!("plan {s}"));
            assert!(
                matches!(plan.kind, SumEvaluationKind::DivergentInfinite),
                "{s} kind"
            );
            assert!(
                matches!(
                    ctx.get(plan.candidate),
                    Expr::Constant(cas_ast::Constant::Infinity)
                ),
                "{s} -> infinity"
            );
        }
        // Divergent product -> infinity (not infinity!).
        let p = cas_parser::parse("product(k, k, 1, inf)", &mut ctx).expect("product");
        let plan = try_plan_finite_product_evaluation(&mut ctx, p, 1000).expect("product plan");
        assert!(matches!(
            plan.kind,
            ProductEvaluationKind::DivergentInfinite
        ));
        assert!(matches!(
            ctx.get(plan.candidate),
            Expr::Constant(cas_ast::Constant::Infinity)
        ));
        // Convergent geometric now evaluates to its closed form `1/(1-1/2) = 2`.
        let conv = cas_parser::parse("sum((1/2)^k, k, 0, inf)", &mut ctx).expect("conv");
        let conv_plan = try_plan_finite_sum_evaluation(&mut ctx, conv, 1000).expect("conv plan");
        assert!(matches!(
            conv_plan.kind,
            SumEvaluationKind::ConvergentInfinite
        ));
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: conv_plan.candidate
                }
            ),
            "2"
        );
        // Finite bound is unaffected (still a finite closed form, never divergent).
        let fin = cas_parser::parse("sum(k, k, 1, 10)", &mut ctx).expect("fin");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, fin, 1000).expect("fin plan");
        assert!(matches!(plan.kind, SumEvaluationKind::SumOfFirstIntegers));
        assert!(!matches!(plan.kind, SumEvaluationKind::DivergentInfinite));
    }

    #[test]
    fn convergent_geometric_series_closed_forms_and_rejections() {
        // (source, expected closed form) for convergent geometric series |r| < 1.
        for (src, expect) in [
            ("sum(1/2^k, k, 0, inf)", "2"),
            ("sum(1/3^k, k, 0, inf)", "3/2"),
            ("sum((1/2)^k, k, 1, inf)", "1"),
            ("sum(2*(1/3)^k, k, 0, inf)", "3"),
            ("sum((2/3)^k, k, 0, inf)", "3"),
            ("sum((-1/2)^k, k, 0, inf)", "2/3"), // alternating
            ("sum(3^(-k), k, 0, inf)", "3/2"),   // reciprocal-exponent form
            ("sum(5/4^k, k, 2, inf)", "5/12"),   // coefficient + offset start
        ] {
            let mut ctx = Context::new();
            let e = cas_parser::parse(src, &mut ctx).expect(src);
            let plan = try_plan_finite_sum_evaluation(&mut ctx, e, 1000)
                .unwrap_or_else(|| panic!("{src} should have a plan"));
            assert!(
                matches!(plan.kind, SumEvaluationKind::ConvergentInfinite),
                "{src} should be ConvergentInfinite"
            );
            assert_eq!(
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &ctx,
                        id: plan.candidate
                    }
                ),
                expect,
                "{src}"
            );
        }

        // Rejections: divergent (|r| ≥ 1) and non-geometric summands.
        // `2^k`/`(3/2)^k` diverge to +infinity (classified, not convergent);
        // `(-2)^k` oscillates, `k·2^k` and `2^k+3^k` are not pure geometric.
        for (src, must_be_convergent) in [
            ("sum(2^k, k, 0, inf)", false),     // -> infinity (DivergentInfinite)
            ("sum((3/2)^k, k, 0, inf)", false), // -> infinity
            ("sum((-2)^k, k, 0, inf)", false),  // oscillates -> no plan
            ("sum(k*2^k, k, 0, inf)", false),   // arithmetico-geometric -> no plan
            ("sum(2^k+3^k, k, 0, inf)", false), // sum of geometrics -> no plan
        ] {
            let mut ctx = Context::new();
            let e = cas_parser::parse(src, &mut ctx).expect(src);
            let plan = try_plan_finite_sum_evaluation(&mut ctx, e, 1000);
            let is_convergent = plan
                .as_ref()
                .map(|p| matches!(p.kind, SumEvaluationKind::ConvergentInfinite))
                .unwrap_or(false);
            assert_eq!(is_convergent, must_be_convergent, "{src}");
        }
    }

    // Fold a constant expression (including integer powers) to an exact rational.
    fn fold_const(ctx: &Context, e: cas_ast::ExprId) -> Option<BigRational> {
        use num_traits::{ToPrimitive, Zero};
        match ctx.get(e) {
            Expr::Number(n) => Some(n.clone()),
            Expr::Neg(i) => fold_const(ctx, *i).map(|v| -v),
            Expr::Add(l, r) => Some(fold_const(ctx, *l)? + fold_const(ctx, *r)?),
            Expr::Sub(l, r) => Some(fold_const(ctx, *l)? - fold_const(ctx, *r)?),
            Expr::Mul(l, r) => Some(fold_const(ctx, *l)? * fold_const(ctx, *r)?),
            Expr::Div(l, r) => {
                let d = fold_const(ctx, *r)?;
                if d.is_zero() {
                    None
                } else {
                    Some(fold_const(ctx, *l)? / d)
                }
            }
            Expr::Pow(b, ex) => {
                let bv = fold_const(ctx, *b)?;
                let ev = fold_const(ctx, *ex)?;
                if !ev.is_integer() {
                    return None;
                }
                super::rational_pow_int(&bv, ev.to_integer().to_i64()?)
            }
            _ => None,
        }
    }

    #[test]
    fn finite_geometric_rational_ratio_closed_forms() {
        // (summand, start a, upper n, expected value of `sum(summand, k, a, n)`).
        // Verified by substituting `m -> n` in the symbolic closed form and folding to an
        // exact rational — display-independent (the formatter shows `(2/3)^m` as `2/3^m`).
        for (src, a, n, num, den) in [
            ("1/2^k", 0i64, 3i64, 15i64, 8i64),
            ("(2/3)^k", 0, 5, 665, 243),
            ("(2/3)^k", 1, 4, 130, 81),
            ("1/3^k", 1, 3, 13, 27),
            ("(-1/2)^k", 0, 4, 11, 16),
            ("5/4^k", 2, 5, 425, 1024),
        ] {
            let mut ctx = Context::new();
            let full =
                cas_parser::parse(&format!("sum({src}, k, {a}, m)"), &mut ctx).expect("parse");
            let plan =
                try_plan_finite_sum_evaluation(&mut ctx, full, 1000).expect("geometric plan");
            assert!(
                matches!(plan.kind, SumEvaluationKind::GeometricPower),
                "{src} should be a geometric closed form"
            );
            let m = ctx.var("m");
            let nval = ctx.num(n);
            let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
            let value = fold_const(&ctx, substituted)
                .unwrap_or_else(|| panic!("{src} closed form should fold at m={n}"));
            assert_eq!(
                value,
                BigRational::new(num.into(), den.into()),
                "sum({src}, k, {a}, {n})"
            );
        }
    }

    #[test]
    fn polynomial_linearity_closed_forms_match_known_values() {
        // (summand, start a, upper n, expected exact value of `sum(summand, k, a, n)`).
        // Verified by substituting `m -> n` in the symbolic closed form and folding to an
        // exact rational — display-independent.
        for (src, a, n, num, den) in [
            ("2*k", 1i64, 5i64, 30i64, 1i64), // 2·(1+..+5) = 30
            ("k^2+k", 1, 4, 40, 1),           // (2+6+12+20) = 40
            ("3*k^2-k+1", 1, 3, 39, 1),       // (3+11+25) = 39
            ("k+1", 1, 4, 14, 1),             // (2+3+4+5) = 14
            ("k*(k+1)", 1, 3, 20, 1),         // (2+6+12) = 20, product expanded
            ("2*k", 2, 4, 18, 1),             // 2·(2+3+4) = 18, symbolic-style start
        ] {
            let mut ctx = Context::new();
            let full =
                cas_parser::parse(&format!("sum({src}, k, {a}, m)"), &mut ctx).expect("parse");
            let plan =
                try_plan_finite_sum_evaluation(&mut ctx, full, 1000).expect("linearity plan");
            assert!(
                matches!(plan.kind, SumEvaluationKind::PolynomialLinearity),
                "{src} should sum by polynomial linearity"
            );
            let m = ctx.var("m");
            let nval = ctx.num(n);
            let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
            let value = fold_const(&ctx, substituted)
                .unwrap_or_else(|| panic!("{src} closed form should fold at m={n}"));
            assert_eq!(
                value,
                BigRational::new(num.into(), den.into()),
                "sum({src}, k, {a}, {n})"
            );
        }
    }

    #[test]
    fn polynomial_linearity_matches_brute_force_over_a_sweep() {
        // Independent brute-force ground truth: Σ_{k=start}^{n} (a3 k^3 + a2 k^2 + a1 k + a0)
        // computed term by term, compared to the symbolic closed form folded at the same n.
        // Sweeps coefficient signs/zeros, integer and symbolic-style starts, and bounds.
        let coeff_sets: [(i64, i64, i64, i64); 6] = [
            (0, 0, 2, 0),  // 2k
            (0, 1, 1, 0),  // k^2 + k
            (3, 0, -1, 1), // 3k^3 - k + 1
            (1, -2, 0, 5), // k^3 - 2k^2 + 5
            (0, 0, 1, 1),  // k + 1
            (2, 3, 0, 0),  // 2k^3 + 3k^2
        ];
        for (a3, a2, a1, a0) in coeff_sets {
            // Build the summand source `a3*k^3 + a2*k^2 + a1*k + a0`, dropping zero terms.
            let mut parts: Vec<String> = Vec::new();
            if a3 != 0 {
                parts.push(format!("({a3})*k^3"));
            }
            if a2 != 0 {
                parts.push(format!("({a2})*k^2"));
            }
            if a1 != 0 {
                parts.push(format!("({a1})*k"));
            }
            if a0 != 0 {
                parts.push(format!("({a0})"));
            }
            if parts.is_empty() {
                continue;
            }
            let summand = parts.join(" + ");
            for (start, n) in [(1i64, 6i64), (1, 9), (2, 7), (3, 8), (0, 5)] {
                let brute: i64 = (start..=n)
                    .map(|k| a3 * k * k * k + a2 * k * k + a1 * k + a0)
                    .sum();
                let mut ctx = Context::new();
                let full = cas_parser::parse(&format!("sum({summand}, k, {start}, m)"), &mut ctx)
                    .expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for {summand} [{start}..]"));
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value = fold_const(&ctx, substituted)
                    .unwrap_or_else(|| panic!("fold {summand} at m={n}"));
                assert_eq!(
                    value,
                    BigRational::from_integer(brute.into()),
                    "sum({summand}, k, {start}, {n}) closed form != brute force {brute}"
                );
            }
        }
    }

    #[test]
    fn arithmetic_geometric_sums_match_brute_force() {
        // Σ_{k=start}^{n} k·r^k closed form folded at concrete n vs the brute-force sum.
        for (ratio_num, ratio_den) in [(2i64, 1i64), (3, 1), (1, 2)] {
            for (start, n) in [(1i64, 5i64), (1, 7), (2, 6), (3, 5)] {
                let ratio = BigRational::new(ratio_num.into(), ratio_den.into());
                let mut brute = BigRational::from_integer(0.into());
                for k in start..=n {
                    let pow = super::rational_pow_int(&ratio, k).expect("pow");
                    brute += BigRational::from_integer(k.into()) * pow;
                }
                let mut ctx = Context::new();
                let src = if ratio_den == 1 {
                    format!("sum(k*{ratio_num}^k, k, {start}, m)")
                } else {
                    format!("sum(k*(1/{ratio_den})^k, k, {start}, m)")
                };
                let full = cas_parser::parse(&src, &mut ctx).expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for {src}"));
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value =
                    fold_const(&ctx, substituted).unwrap_or_else(|| panic!("fold {src} at m={n}"));
                assert_eq!(
                    value, brute,
                    "sum(k·{ratio_num}/{ratio_den}^k, k, {start}, {n})"
                );
            }
        }
    }

    #[test]
    fn affine_arithmetic_geometric_sums_match_brute_force() {
        // The DISTRIBUTED / constant-coefficient forms the engine produces for an affine
        // cofactor `(αk+β)·r^k` (a constant-times-`k·r^k`, or a sum of a geometric and an
        // arithmetic-geometric term). Each `term(k)` mirrors the summand string exactly.
        type BruteTerm = fn(i64) -> i128;
        let cases: [(&str, BruteTerm); 3] = [
            ("3*k*2^k", |k| 3 * k as i128 * 2i128.pow(k as u32)),
            ("2^k + k*2^k", |k| {
                2i128.pow(k as u32) + k as i128 * 2i128.pow(k as u32)
            }),
            ("2^k + 2*k*2^(k+1)", |k| {
                2i128.pow(k as u32) + 2 * k as i128 * 2i128.pow((k + 1) as u32)
            }),
        ];
        for (summand, term) in cases {
            for (start, n) in [(1i64, 5i64), (2, 6), (1, 7)] {
                let brute: i128 = (start..=n).map(term).sum();
                let mut ctx = Context::new();
                let src = format!("sum({summand}, k, {start}, m)");
                let full = cas_parser::parse(&src, &mut ctx).expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for {src}"));
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value =
                    fold_const(&ctx, substituted).unwrap_or_else(|| panic!("fold {src} at m={n}"));
                assert_eq!(
                    value,
                    BigRational::from_integer(brute.into()),
                    "sum({summand}, k, {start}, {n})"
                );
            }
        }
    }

    #[test]
    fn fractional_quotient_arithmetic_geometric_sums_match_brute_force() {
        // The Div spelling `p(k)/r^k` of a fractional-ratio arithmetic-geometric sum (e.g. the
        // textbook Σ k/2^k → 2). `mul_leaves` keeps the quotient as one leaf, so the builder reads
        // the geometric off the DENOMINATOR (ratio 1/r) and the numerator as the cofactor.
        type BruteTerm = fn(i64) -> BigRational;
        let cases: [(&str, BruteTerm); 3] = [
            ("k/2^k", |k| {
                BigRational::new(k.into(), 2i64.pow(k as u32).into())
            }),
            ("k/3^k", |k| {
                BigRational::new(k.into(), 3i64.pow(k as u32).into())
            }),
            ("(2*k+1)/2^k", |k| {
                BigRational::new((2 * k + 1).into(), 2i64.pow(k as u32).into())
            }),
        ];
        for (summand, term) in cases {
            for (start, n) in [(1i64, 5i64), (2, 6), (1, 7)] {
                let mut brute = BigRational::from_integer(0.into());
                for k in start..=n {
                    brute += term(k);
                }
                let mut ctx = Context::new();
                let src = format!("sum({summand}, k, {start}, m)");
                let full = cas_parser::parse(&src, &mut ctx).expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for {src}"));
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value =
                    fold_const(&ctx, substituted).unwrap_or_else(|| panic!("fold {src} at m={n}"));
                assert_eq!(value, brute, "sum({summand}, k, {start}, {n})");
            }
        }
    }

    #[test]
    fn convergent_infinite_arithmetic_geometric_sums_are_exact() {
        // `Σ_{k=a}^∞ p(k)·r^k` for |r| < 1 is an exact rational (tails r/(1−r), r/(1−r)², r(1+r)/
        // (1−r)³ corrected for the lower bound). Each closed form is checked against its known value.
        for (src, num, den) in [
            ("sum(k/2^k, k, 1, inf)", 2i64, 1i64),
            ("sum(k^2/2^k, k, 1, inf)", 6, 1),
            ("sum(k*(1/2)^k, k, 1, inf)", 2, 1),
            ("sum(k/3^k, k, 0, inf)", 3, 4),
            ("sum((2*k+1)/2^k, k, 1, inf)", 5, 1),
            ("sum(k/3^k, k, 2, inf)", 5, 12),
        ] {
            let mut ctx = Context::new();
            let full = cas_parser::parse(src, &mut ctx).expect("parse");
            let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                .unwrap_or_else(|| panic!("plan for {src}"));
            let value = fold_const(&ctx, plan.candidate).unwrap_or_else(|| panic!("fold {src}"));
            assert_eq!(value, BigRational::new(num.into(), den.into()), "{src}");
        }
        // Divergent ratio (|r| ≥ 1) is not a convergent arithmetic-geometric series.
        let mut ctx = Context::new();
        let div = cas_parser::parse("sum(k*2^k, k, 1, inf)", &mut ctx).expect("parse");
        let call = super::try_extract_finite_aggregate_call(&ctx, div, "sum").expect("call");
        assert!(super::try_convergent_infinite_arithmetic_geometric_sum(&mut ctx, &call).is_none());
    }

    #[test]
    fn quadratic_arithmetic_geometric_sums_match_brute_force() {
        // Σ p(k)·r^k for a degree-2 cofactor p(k) = α·k² + β·k + γ: the closed form
        // α·S₂ + β·S₁ + γ·S₀ folded at concrete n vs the brute-force sum.
        type BruteTerm = fn(i64) -> i128;
        let cases: [(&str, BruteTerm); 4] = [
            ("k^2*2^k", |k| (k as i128).pow(2) * 2i128.pow(k as u32)),
            ("k^2*3^k", |k| (k as i128).pow(2) * 3i128.pow(k as u32)),
            ("(2*k^2-3*k+1)*2^k", |k| {
                (2 * (k as i128).pow(2) - 3 * k as i128 + 1) * 2i128.pow(k as u32)
            }),
            ("(k^2+1)*2^k", |k| {
                ((k as i128).pow(2) + 1) * 2i128.pow(k as u32)
            }),
        ];
        for (summand, term) in cases {
            for (start, n) in [(1i64, 5i64), (2, 6), (1, 7)] {
                let brute: i128 = (start..=n).map(term).sum();
                let mut ctx = Context::new();
                let src = format!("sum({summand}, k, {start}, m)");
                let full = cas_parser::parse(&src, &mut ctx).expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for {src}"));
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value =
                    fold_const(&ctx, substituted).unwrap_or_else(|| panic!("fold {src} at m={n}"));
                assert_eq!(
                    value,
                    BigRational::from_integer(brute.into()),
                    "sum({summand}, k, {start}, {n})"
                );
            }
        }
    }

    #[test]
    fn pronic_cofactor_arithmetic_geometric_builder_is_exact() {
        // The pronic cofactors k(k±1) = k²±k carry a common index factor, so the engine
        // oscillates between the factored `2^k·k·(k+1)` and the distributed
        // `k²·2^k + k·2^k` summand and `sum((k²+k)·2^k)` stays an honest residual END TO END
        // (a separate orchestration concern, tracked as a peldaño). The arithmetic-geometric
        // BUILDER is exact regardless: its closed form must fold to Σ k(k+1)·r^k. This pins the
        // math so a future orchestration fix lands a verified-correct result, not a new bug.
        let mut ctx = Context::new();
        let summand = cas_parser::parse("2^k*(k^2+k)", &mut ctx).expect("parse");
        let start = ctx.num(1);
        let end = ctx.var("m");
        let candidate =
            super::try_build_arithmetic_geometric_sum(&mut ctx, summand, "k", start, end)
                .expect("builder handles the pronic cofactor");
        for n in [3i64, 5, 7] {
            let brute: i128 = (1..=n)
                .map(|k| (k as i128) * (k as i128 + 1) * 2i128.pow(k as u32))
                .sum();
            let m = ctx.var("m");
            let nval = ctx.num(n);
            let substituted = cas_ast::substitute_expr_by_id(&mut ctx, candidate, m, nval);
            let value = fold_const(&ctx, substituted).expect("fold pronic closed form");
            assert_eq!(
                value,
                BigRational::from_integer(brute.into()),
                "Σ k(k+1)·2^k at n={n}"
            );
        }
    }

    #[test]
    fn faulhaber_high_degree_sums_match_brute_force() {
        // For p = 4..8 the closed form S_p(n) has no dedicated builder; verify the symbolic
        // Faulhaber polynomial folds to the brute-force Σ_{k=start}^{n} k^p at concrete n.
        for p in 4u32..=8 {
            for (start, n) in [(1i64, 5i64), (1, 7), (2, 6), (3, 8)] {
                let brute: i128 = (start..=n).map(|k| (k as i128).pow(p)).sum();
                let mut ctx = Context::new();
                let full = cas_parser::parse(&format!("sum(k^{p}, k, {start}, m)"), &mut ctx)
                    .expect("parse");
                let plan = try_plan_finite_sum_evaluation(&mut ctx, full, 1000)
                    .unwrap_or_else(|| panic!("plan for k^{p} [{start}..]"));
                assert!(
                    matches!(plan.kind, SumEvaluationKind::PolynomialLinearity),
                    "k^{p} should sum by Faulhaber linearity"
                );
                let m = ctx.var("m");
                let nval = ctx.num(n);
                let substituted = cas_ast::substitute_expr_by_id(&mut ctx, plan.candidate, m, nval);
                let value =
                    fold_const(&ctx, substituted).unwrap_or_else(|| panic!("fold k^{p} at m={n}"));
                assert_eq!(
                    value,
                    BigRational::from_integer(brute.into()),
                    "sum(k^{p}, k, {start}, {n}) closed form != brute force {brute}"
                );
            }
        }
    }

    #[test]
    fn polynomial_linearity_leaves_owned_and_out_of_scope_shapes_alone() {
        // Bare single powers up to cube and constants stay owned by their dedicated builders
        // (the linearity fallback runs after them); degree 4..12 are summed by Faulhaber
        // linearity; degree > 12 and non-polynomial summands are declined (honest residual).
        for (src, expected_owned_kind) in [
            (
                "sum(k, k, 1, n)",
                Some(SumEvaluationKind::SumOfFirstIntegers),
            ),
            ("sum(k^2, k, 1, n)", Some(SumEvaluationKind::SumOfSquares)),
            ("sum(k^3, k, 1, n)", Some(SumEvaluationKind::SumOfCubes)),
            ("sum(5, k, 1, n)", Some(SumEvaluationKind::SumOfConstant)),
            ("sum(2^k, k, 1, n)", Some(SumEvaluationKind::GeometricPower)),
            (
                "sum(k^4, k, 1, n)",
                Some(SumEvaluationKind::PolynomialLinearity),
            ),
            (
                "sum(k^12, k, 1, n)",
                Some(SumEvaluationKind::PolynomialLinearity),
            ),
            ("sum(k^13, k, 1, n)", None), // degree > 12: declined to bound the closed form
            ("sum(1/k, k, 1, n)", None),  // not a polynomial in k
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000);
            match expected_owned_kind {
                Some(kind) => {
                    let plan = plan.unwrap_or_else(|| panic!("{src} should have a plan"));
                    assert_eq!(
                        std::mem::discriminant(&plan.kind),
                        std::mem::discriminant(&kind),
                        "{src} should keep its dedicated builder, not polynomial linearity"
                    );
                }
                None => assert!(plan.is_none(), "{src} should stay residual"),
            }
        }
    }

    #[test]
    fn reversed_bounds_collapse_to_empty_sum_and_product() {
        let zero = BigRational::from_integer(0.into());
        let one = BigRational::from_integer(1.into());
        let mut ctx = Context::new();

        // sum(k, k, 6, 3): empty range -> 0 (NOT the reversed closed form -9).
        let s = cas_parser::parse("sum(k, k, 6, 3)", &mut ctx).expect("sum");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, s, 1000).expect("empty sum plan");
        assert!(matches!(
            plan.kind,
            SumEvaluationKind::FiniteDirect { start: 6, end: 3 }
        ));
        assert_eq!(eval_small_rat(&ctx, plan.candidate), Some(zero.clone()));

        // product(k, k, 6, 3): empty range -> 1 (NOT 3!/5! = 1/20).
        let p = cas_parser::parse("product(k, k, 6, 3)", &mut ctx).expect("product");
        let pplan =
            try_plan_finite_product_evaluation(&mut ctx, p, 1000).expect("empty product plan");
        assert!(matches!(
            pplan.kind,
            ProductEvaluationKind::FiniteDirect { start: 6, end: 3 }
        ));
        assert_eq!(eval_small_rat(&ctx, pplan.candidate), Some(one));

        // Negative-bound empty range too: sum(k, k, 0, -3) -> 0.
        let neg = cas_parser::parse("sum(k, k, 0, -3)", &mut ctx).expect("sum neg");
        let neg_plan = try_plan_finite_sum_evaluation(&mut ctx, neg, 1000).expect("empty sum plan");
        assert_eq!(eval_small_rat(&ctx, neg_plan.candidate), Some(zero));

        // A non-reversed range is unaffected (the guard must not fire): 1..5 -> 15.
        let normal = cas_parser::parse("sum(k, k, 1, 5)", &mut ctx).expect("normal");
        let normal_plan = try_plan_finite_sum_evaluation(&mut ctx, normal, 1000).expect("plan");
        assert_eq!(
            eval_small_rat(&ctx, normal_plan.candidate),
            Some(BigRational::from_integer(15.into()))
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_first_integers_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k, k, m, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("(n*(n+1)-m*(m-1))/2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfFirstIntegers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
        let m = ctx.var("m");
        let n = ctx.var("n");
        let two = ctx.num(2);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, plan.candidate, m, two);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::from_integer(9.into()))
        );
    }

    #[test]
    fn sum_of_first_integers_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("k", &mut ctx).expect("term");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_sum_of_first_integers(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_squares() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k^2, k, 1, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("n*(n+1)*(2*n+1)/6", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfSquares));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_squares_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k^2, k, m, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("(n*(n+1)*(2*n+1)-m*(m-1)*(2*m-1))/6", &mut ctx)
            .expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfSquares));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
        let m = ctx.var("m");
        let n = ctx.var("n");
        let two = ctx.num(2);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, plan.candidate, m, two);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::from_integer(29.into()))
        );
    }

    #[test]
    fn sum_of_squares_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("k^2", &mut ctx).expect("term");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_sum_of_squares(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_cubes() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k^3, k, 1, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("(n*(n+1)/2)^2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfCubes));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_cubes_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(k^3, k, m, n)", &mut ctx).expect("sum");
        let expected =
            cas_parser::parse("(n*(n+1)/2)^2 - (m*(m-1)/2)^2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfCubes));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
        let m = ctx.var("m");
        let n = ctx.var("n");
        let two = ctx.num(2);
        let four = ctx.num(4);
        let result = substitute_expr_by_id(&mut ctx, plan.candidate, m, two);
        let result = substitute_expr_by_id(&mut ctx, result, n, four);
        assert_eq!(
            eval_small_rat(&ctx, result),
            Some(BigRational::from_integer(99.into()))
        );
    }

    #[test]
    fn sum_of_cubes_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("k^3", &mut ctx).expect("term");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_sum_of_cubes(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_constant() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(c, k, 1, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("c*n", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfConstant));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_of_constant_rejects_bound_variable_summand() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("k + c", &mut ctx).expect("term");
        let start = cas_parser::parse("1", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_sum_of_constant(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn sum_of_zero_summand_is_zero_for_any_bounds_including_infinite() {
        // A zero summand sums to 0 over ANY range — the early structurally-zero
        // check must fire BEFORE the term count, so `sum(0, k, 1, inf)` never
        // builds `0 * inf` (which would fold to undefined). Covers literal 0 and a
        // structurally-zero summand, with finite/symbolic/infinite bounds.
        for (term_src, start_src, end_src) in [
            ("0", "1", "inf"),
            ("0", "1", "10"),
            ("0", "1", "n"),
            ("k - k", "1", "inf"),
        ] {
            let mut ctx = Context::new();
            let term = cas_parser::parse(term_src, &mut ctx).expect("term");
            let start = cas_parser::parse(start_src, &mut ctx).expect("start");
            let end = cas_parser::parse(end_src, &mut ctx).expect("end");
            let result = try_build_sum_of_constant(&mut ctx, term, "k", start, end)
                .expect("zero summand sums to 0");
            let zero = ctx.num(0);
            assert_eq!(
                compare_expr(&ctx, result, zero),
                std::cmp::Ordering::Equal,
                "sum({term_src}, k, {start_src}, {end_src}) must be 0"
            );
        }
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_sum_of_constant_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(c, k, m, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("c*(n-m+1)", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::SumOfConstant));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_of_constant_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("c", &mut ctx).expect("term");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_sum_of_constant(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_geometric_power_base_two() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(2^k, k, 0, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("2^(n+1)-1", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::GeometricPower));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_geometric_power_base_two_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(2^k, k, m, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("2^(n+1)-2^m", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::GeometricPower));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_geometric_power_base_three() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(3^k, k, 0, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("(3^(n+1)-1)/2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::GeometricPower));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sum_evaluation_plan_detects_symbolic_geometric_power_base_three_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("sum(3^k, k, m, n)", &mut ctx).expect("sum");
        let expected = cas_parser::parse("(3^(n+1)-3^m)/2", &mut ctx).expect("closed form");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, SumEvaluationKind::GeometricPower));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn geometric_power_sum_rejects_symbolic_base() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("r^k", &mut ctx).expect("term");
        let start = cas_parser::parse("0", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_geometric_power_sum(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn geometric_power_sum_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let term = cas_parser::parse("2^k", &mut ctx).expect("term");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_geometric_power_sum(&mut ctx, term, "k", start, end).is_none());
    }

    #[test]
    fn product_evaluation_plan_detects_telescoping_variants() {
        let mut ctx = Context::new();

        let telescoping = cas_parser::parse("product((k+1)/k, k, 1, n)", &mut ctx).expect("prod");
        let plan1 = try_plan_finite_product_evaluation(&mut ctx, telescoping, 1000)
            .expect("product telescoping");
        assert!(matches!(plan1.kind, ProductEvaluationKind::Telescoping));

        let factorized = cas_parser::parse("product(1-1/k^2, k, 2, n)", &mut ctx).expect("prod");
        let plan2 =
            try_plan_finite_product_evaluation(&mut ctx, factorized, 1000).expect("factorized");
        assert!(matches!(
            plan2.kind,
            ProductEvaluationKind::FactorizedTelescoping
        ));

        let simplified_factorized =
            cas_parser::parse("product((k^2 - 1)/k^2, k, 2, n)", &mut ctx).expect("prod");
        let plan3 = try_plan_finite_product_evaluation(&mut ctx, simplified_factorized, 1000)
            .expect("simplified factorized");
        assert!(matches!(
            plan3.kind,
            ProductEvaluationKind::FactorizedTelescoping
        ));

        let shifted_factorized =
            cas_parser::parse("product(1 - 1/(k+a)^2, k, 1, n)", &mut ctx).expect("prod");
        let plan4 = try_plan_finite_product_evaluation(&mut ctx, shifted_factorized, 1000)
            .expect("shifted factorized");
        assert!(matches!(
            plan4.kind,
            ProductEvaluationKind::FactorizedTelescoping
        ));
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_first_integers() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k, k, 1, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("n!", &mut ctx).expect("factorial");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(
            plan.kind,
            ProductEvaluationKind::ProductOfFirstIntegers
        ));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_of_first_integers_rejects_non_unit_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k", &mut ctx).expect("factor");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_first_integers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_first_integers_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k, k, m, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("n!/(m-1)!", &mut ctx).expect("factorial quotient");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(
            plan.kind,
            ProductEvaluationKind::ProductOfFirstIntegers
        ));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_of_first_integers_rejects_zero_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k", &mut ctx).expect("factor");
        let start = cas_parser::parse("0", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_first_integers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_of_first_integers_rejects_negative_integer_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k", &mut ctx).expect("factor");
        let start = cas_parser::parse("-1", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_first_integers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_squares() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k^2, k, 1, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("(n!)^2", &mut ctx).expect("factorial square");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, ProductEvaluationKind::ProductOfPowers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_squares_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k^2, k, m, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("(n!/(m-1)!)^2", &mut ctx).expect("factorial quotient");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, ProductEvaluationKind::ProductOfPowers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_cubes_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k^3, k, m, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("(n!/(m-1)!)^3", &mut ctx).expect("factorial quotient");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, ProductEvaluationKind::ProductOfPowers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_fourth_powers_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(k^4, k, m, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("(n!/(m-1)!)^4", &mut ctx).expect("factorial quotient");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(plan.kind, ProductEvaluationKind::ProductOfPowers));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_of_powers_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k^2", &mut ctx).expect("factor");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_powers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_of_powers_rejects_zero_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k^2", &mut ctx).expect("factor");
        let start = cas_parser::parse("0", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_powers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_of_powers_rejects_negative_integer_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k^2", &mut ctx).expect("factor");
        let start = cas_parser::parse("-1", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_powers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_of_powers_rejects_non_positive_exponent() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k^0", &mut ctx).expect("factor");
        let start = cas_parser::parse("m", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_powers(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_constant() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(c, k, 1, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("c^n", &mut ctx).expect("power");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(
            plan.kind,
            ProductEvaluationKind::ProductOfConstant
        ));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_of_constant_rejects_bound_variable_factor() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("k + c", &mut ctx).expect("factor");
        let start = cas_parser::parse("1", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_constant(&mut ctx, factor, "k", start, end).is_none());
    }

    #[test]
    fn product_evaluation_plan_detects_symbolic_product_of_constant_symbolic_lower_bound() {
        let mut ctx = Context::new();

        let expr = cas_parser::parse("product(c, k, m, n)", &mut ctx).expect("product");
        let expected = cas_parser::parse("c^(n-m+1)", &mut ctx).expect("power");
        let plan =
            try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("closed form plan");

        assert!(matches!(
            plan.kind,
            ProductEvaluationKind::ProductOfConstant
        ));
        assert_eq!(
            compare_expr(&ctx, plan.candidate, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn product_of_constant_rejects_bound_variable_lower_bound() {
        let mut ctx = Context::new();

        let factor = cas_parser::parse("c", &mut ctx).expect("factor");
        let start = cas_parser::parse("k", &mut ctx).expect("start");
        let end = cas_parser::parse("n", &mut ctx).expect("end");

        assert!(try_build_product_of_constant(&mut ctx, factor, "k", start, end).is_none());
    }
}
