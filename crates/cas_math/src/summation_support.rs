use crate::build::mul2_raw;
use cas_ast::{substitute_expr_by_id, Context, Expr, ExprId};
use num_traits::One;

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
    FiniteDirect { start: i64, end: i64 },
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
    FiniteDirect { start: i64, end: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductEvaluationPlan {
    pub call: FiniteAggregateCall,
    pub candidate: ExprId,
    pub kind: ProductEvaluationKind,
}

/// Render a user-facing description for a finite `sum(...)` rewrite.
pub fn render_sum_evaluation_desc_with<F>(
    kind: &SumEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        SumEvaluationKind::Telescoping => format!(
            "Telescoping sum: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::FiniteDirect { start, end } => format!(
            "sum({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
    }
}

/// Render a user-facing description for a finite `product(...)` rewrite.
pub fn render_product_evaluation_desc_with<F>(
    kind: &ProductEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        ProductEvaluationKind::Telescoping => format!(
            "Telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FactorizedTelescoping => format!(
            "Factorized telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FiniteDirect { start, end } => format!(
            "product({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
    }
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

/// Build the best available finite-sum evaluation plan for `sum(...)`.
///
/// Preference order:
/// 1. Telescoping rational pattern.
/// 2. Direct finite substitution when bounds are small integers.
pub fn try_plan_finite_sum_evaluation(
    ctx: &mut Context,
    expr: ExprId,
    max_span: i64,
) -> Option<SumEvaluationPlan> {
    let call = try_extract_finite_aggregate_call(ctx, expr, "sum")?;

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

    None
}

/// Build the best available finite-product evaluation plan for `product(...)`.
///
/// Preference order:
/// 1. Shift-1 telescoping ratio pattern.
/// 2. Factorizable `1 - 1/k^2` telescoping pattern.
/// 3. Direct finite substitution when bounds are small integers.
pub fn try_plan_finite_product_evaluation(
    ctx: &mut Context,
    expr: ExprId,
    max_span: i64,
) -> Option<ProductEvaluationPlan> {
    let call = try_extract_finite_aggregate_call(ctx, expr, "product")?;

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

    None
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

    let num_offset = extract_linear_offset(ctx, num, var)?;
    let den_offset = extract_linear_offset(ctx, den, var)?;
    if num_offset - den_offset != 1 {
        return None;
    }

    let end_plus_offset = if num_offset == 0 {
        end
    } else {
        let offset = ctx.num(num_offset);
        ctx.add(Expr::Add(end, offset))
    };

    let start_plus_offset = if den_offset == 0 {
        start
    } else {
        let offset = ctx.num(den_offset);
        ctx.add(Expr::Add(start, offset))
    };

    Some(ctx.add(Expr::Div(end_plus_offset, start_plus_offset)))
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
    let power = detect_one_minus_reciprocal_power(ctx, factor, var)?;
    if power != 2 {
        return None;
    }

    let start_minus_1 = if let Some(n) = crate::expr_extract::extract_i64_integer(ctx, start) {
        ctx.num(n - 1)
    } else {
        let one = ctx.num(1);
        ctx.add(Expr::Sub(start, one))
    };

    let end_plus_1 = if let Some(n) = crate::expr_extract::extract_i64_integer(ctx, end) {
        ctx.num(n + 1)
    } else {
        let one = ctx.num(1);
        ctx.add(Expr::Add(end, one))
    };

    let combined_num = mul2_raw(ctx, start_minus_1, end_plus_1);
    let combined_den = mul2_raw(ctx, start, end);
    Some(ctx.add(Expr::Div(combined_num, combined_den)))
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

    let offset1 = extract_linear_offset(ctx, factor1, var)?;
    let offset2 = extract_linear_offset(ctx, factor2, var)?;
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

#[cfg(test)]
mod tests {
    use super::{
        build_finite_product_substitution, build_finite_sum_substitution,
        detect_one_minus_reciprocal_power, detect_reciprocal_power, extract_linear_offset,
        render_product_evaluation_desc_with, render_sum_evaluation_desc_with,
        try_build_factorizable_product_for_one_minus_reciprocal_square,
        try_build_telescoping_product_shift1, try_build_telescoping_rational_sum,
        try_extract_bounded_integer_range, try_extract_finite_aggregate_call,
        try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, ProductEvaluationKind,
        SumEvaluationKind,
    };
    use cas_ast::{Context, Expr};
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
        let Expr::Div(num, den) = ctx.get(result) else {
            panic!("expected division");
        };
        let Expr::Mul(nl, nr) = ctx.get(*num) else {
            panic!("expected mul numerator");
        };
        let Expr::Mul(dl, dr) = ctx.get(*den) else {
            panic!("expected mul denominator");
        };
        assert_eq!(crate::expr_extract::extract_i64_integer(&ctx, *nl), Some(1));
        assert_eq!(crate::expr_extract::extract_i64_integer(&ctx, *nr), Some(5));
        assert_eq!(crate::expr_extract::extract_i64_integer(&ctx, *dl), Some(2));
        assert_eq!(crate::expr_extract::extract_i64_integer(&ctx, *dr), Some(4));
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
    }

    #[test]
    fn renders_sum_evaluation_descriptions() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("sum(k^2, k, 1, 5)", &mut ctx).expect("sum");
        let plan = try_plan_finite_sum_evaluation(&mut ctx, expr, 1000).expect("plan");
        let desc = render_sum_evaluation_desc_with(&plan.kind, &plan.call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
        });
        assert!(desc.contains("sum(") || desc.contains("Telescoping sum"));
    }

    #[test]
    fn renders_product_evaluation_descriptions() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("product(k, k, 1, 3)", &mut ctx).expect("product");
        let plan = try_plan_finite_product_evaluation(&mut ctx, expr, 1000).expect("plan");
        let desc = render_product_evaluation_desc_with(&plan.kind, &plan.call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: &ctx, id })
        });
        assert!(desc.contains("product(") || desc.contains("product: Π"));
    }
}
