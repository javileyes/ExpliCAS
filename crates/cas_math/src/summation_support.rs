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
    FiniteDirect {
        start: i64,
        end: i64,
    },
    /// Series with an infinite upper bound whose divergence is classified
    /// (`±infinity`) instead of substituting `infinity` into a finite formula.
    DivergentInfinite,
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

    // Infinite upper bound: NOT a finite sum. The closed-form builders below would
    // substitute `infinity` into finite formulas (e.g. n(n+1)/2 -> 1/2*infinity^2,
    // (r^(n+1)-r^a)/(r-1) -> 2^infinity-1). Classify the divergence here and return
    // (or leave unevaluated) so those builders never run on an infinite bound.
    if is_positive_infinity(ctx, call.end_expr) {
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

    if let (Some(num_offset), Some(den_offset)) = (
        extract_linear_offset(ctx, num, var),
        extract_linear_offset(ctx, den, var),
    ) {
        if num_offset - den_offset == 1 {
            let end_plus_offset = shift_expr(ctx, end, num_offset);
            let start_plus_offset = shift_expr(ctx, start, den_offset);
            return Some(ctx.add(Expr::Div(end_plus_offset, start_plus_offset)));
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
        return Some(ctx.add(Expr::Div(end_next_base, start_base)));
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
/// `∏(1 - 1/k^2) = ((start-1)*(end+1))/(start*end)`.
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
        _ => return None,
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
    let a = offset2 - offset1;
    if a == 0 {
        return None;
    }

    let start_shifted = if offset1 == 0 {
        start
    } else {
        let offset = ctx.num(offset1);
        ctx.add(Expr::Add(start, offset))
    };

    let end_shifted = if offset2 == 0 {
        end
    } else {
        let offset = ctx.num(offset2);
        ctx.add(Expr::Add(end, offset))
    };

    let one1 = ctx.num(1);
    let one2 = ctx.num(1);
    let first_term = ctx.add(Expr::Div(one1, start_shifted));
    let second_term = ctx.add(Expr::Div(one2, end_shifted));
    let diff = ctx.add(Expr::Sub(first_term, second_term));

    let result = if a.abs() == 1 {
        if a > 0 {
            diff
        } else {
            ctx.add(Expr::Neg(diff))
        }
    } else {
        let a_expr = ctx.num(a.abs());
        let unsigned_result = ctx.add(Expr::Div(diff, a_expr));
        if a > 0 {
            unsigned_result
        } else {
            ctx.add(Expr::Neg(unsigned_result))
        }
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
        // Convergent geometric stays UNEVALUATED (no plan), not a wrong value.
        let conv = cas_parser::parse("sum((1/2)^k, k, 0, inf)", &mut ctx).expect("conv");
        assert!(try_plan_finite_sum_evaluation(&mut ctx, conv, 1000).is_none());
        // Finite bound is unaffected (still a finite closed form, never divergent).
        let fin = cas_parser::parse("sum(k, k, 1, 10)", &mut ctx).expect("fin");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, fin, 1000).expect("fin plan");
        assert!(matches!(plan.kind, SumEvaluationKind::SumOfFirstIntegers));
        assert!(!matches!(plan.kind, SumEvaluationKind::DivergentInfinite));
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
