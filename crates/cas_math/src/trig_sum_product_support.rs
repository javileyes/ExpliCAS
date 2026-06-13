use crate::expr_nary::add_leaves;
use crate::expr_rewrite::smart_mul;
use crate::numeric::as_i64;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::trig_linear_support::{
    build_coef_times_base, extract_linear_coefficients, extract_sin_cos_fn_arg,
};
use crate::trig_roots_flatten::flatten_mul_chain;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use std::cmp::Ordering;

/// Extract the argument from a trig function: `sin(arg)`/`cos(arg)` -> `arg`.
pub fn extract_trig_arg(ctx: &Context, id: ExprId, fn_name: &str) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(id) {
        if ctx.builtin_of(*fn_id).is_some_and(|b| b.name() == fn_name) && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

/// Extract two trig args from a 2-term sum: `sin(A)+sin(B)` / `cos(A)+cos(B)`.
pub fn extract_trig_two_term_sum(
    ctx: &Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    let terms = add_leaves(ctx, expr);
    if terms.len() != 2 {
        return None;
    }
    let arg1 = extract_trig_arg(ctx, terms[0], fn_name)?;
    let arg2 = extract_trig_arg(ctx, terms[1], fn_name)?;
    Some((arg1, arg2))
}

/// Extract two trig args from a 2-term difference:
/// - `Sub(sin(A), sin(B))`
/// - `Add(sin(A), Neg(sin(B)))`
pub fn extract_trig_two_term_diff(
    ctx: &Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    if let Expr::Sub(l, r) = ctx.get(expr) {
        let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
        let arg2 = extract_trig_arg(ctx, *r, fn_name)?;
        return Some((arg1, arg2));
    }

    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Expr::Neg(inner) = ctx.get(*r) {
            let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
        if let Expr::Neg(inner) = ctx.get(*l) {
            let arg1 = extract_trig_arg(ctx, *r, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
    }

    None
}

/// Check if two pairs match as multisets: `{a1, a2} == {b1, b2}`.
pub fn args_match_as_multiset(
    ctx: &Context,
    a1: ExprId,
    a2: ExprId,
    b1: ExprId,
    b2: ExprId,
) -> bool {
    let direct = cas_ast::ordering::compare_expr(ctx, a1, b1) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, a2, b2) == Ordering::Equal;
    let crossed = cas_ast::ordering::compare_expr(ctx, a1, b2) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, a2, b1) == Ordering::Equal;
    direct || crossed
}

/// Normalize an expression for even functions: `f(-x) == f(x)`.
pub fn normalize_for_even_fn(ctx: &Context, expr: ExprId) -> ExprId {
    let minus_one = BigRational::from_integer(BigInt::from(-1));

    if let Expr::Neg(inner) = ctx.get(expr) {
        return *inner;
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &minus_one {
                return *r;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &minus_one {
                return *l;
            }
        }
    }
    expr
}

/// Simplify a numeric division in coefficient-linear forms.
/// Examples: `4*x/2 -> 2*x`, `-2*x/2 -> -x`, `4/2 -> 2`.
pub fn simplify_numeric_div(ctx: &mut Context, expr: ExprId) -> ExprId {
    let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return expr;
    };

    let Some(den_val) = as_i64(ctx, den) else {
        return expr;
    };
    if den_val == 0 {
        return expr;
    }

    if let Expr::Mul(l, r) = ctx.get(num) {
        let (l, r) = (*l, *r);
        if let Some(coeff) = as_i64(ctx, l) {
            if coeff % den_val == 0 {
                let new_coeff = coeff / den_val;
                if new_coeff == 1 {
                    return r;
                }
                if new_coeff == -1 {
                    return ctx.add(Expr::Neg(r));
                }
                let new_coeff_expr = ctx.num(new_coeff);
                return ctx.add(Expr::Mul(new_coeff_expr, r));
            }
        }
        if let Some(coeff) = as_i64(ctx, r) {
            if coeff % den_val == 0 {
                let new_coeff = coeff / den_val;
                if new_coeff == 1 {
                    return l;
                }
                if new_coeff == -1 {
                    return ctx.add(Expr::Neg(l));
                }
                let new_coeff_expr = ctx.num(new_coeff);
                return ctx.add(Expr::Mul(l, new_coeff_expr));
            }
        }
    }

    if let Some(num_val) = as_i64(ctx, num) {
        if num_val % den_val == 0 {
            return ctx.num(num_val / den_val);
        }
    }

    expr
}

/// Build `(A-B)/2`, optionally canonicalizing order before subtraction.
///
/// The caller provides `simplify_expr` to pre-simplify the numerator difference
/// before dividing by 2 (engine currently uses `collect` for this step).
pub fn build_half_diff_with_simplifier(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    canonical_order: bool,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> ExprId {
    let (first, second) =
        if canonical_order && cas_ast::ordering::compare_expr(ctx, a, b) == Ordering::Greater {
            (b, a)
        } else {
            (a, b)
        };

    let diff = ctx.add(Expr::Sub(first, second));
    let diff_simplified = simplify_expr(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    simplify_numeric_div(ctx, result)
}

/// Build `(A+B)/2`, pre-simplifying the sum via the caller-provided callback.
pub fn build_avg_with_simplifier(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> ExprId {
    let sum = ctx.add(Expr::Add(a, b));
    let sum_simplified = simplify_expr(ctx, sum);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(sum_simplified, two));
    simplify_numeric_div(ctx, result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigSumProductRewrite {
    pub rewritten: ExprId,
    pub kind: TrigSumProductRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigProductToSumRewrite {
    pub rewritten: ExprId,
    pub kind: TrigProductToSumRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigSumToProductContractionRewrite {
    pub rewritten: ExprId,
    pub kind: TrigSumToProductContractionRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigSumProductRewriteKind {
    Werner,
    SinSum,
    SinDiff,
    CosSum,
    CosDiff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigProductToSumRewriteKind {
    SinCos,
    CosSin,
    CosCos,
    SinSin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigSumToProductContractionRewriteKind {
    SinSum,
    SinDiff,
    CosSum,
    CosDiff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TanDifferenceRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigSumQuotientRewritePlan {
    pub num_id: ExprId,
    pub den_id: ExprId,
    pub intermediate_num: ExprId,
    pub intermediate_den: ExprId,
    pub state_after_step1: ExprId,
    pub state_after_step2: ExprId,
    pub rewritten: ExprId,
    pub desc_step1: &'static str,
    pub desc_step2: &'static str,
    pub desc_step3: &'static str,
}

pub struct CosDiffSinDiffQuotientRewritePlan {
    pub num_id: ExprId,
    pub den_id: ExprId,
    pub intermediate_num: ExprId,
    pub intermediate_den: ExprId,
    pub state_after_step1: ExprId,
    pub state_after_step2: ExprId,
    pub rewritten: ExprId,
    pub introduced_nonzero: ExprId,
    pub result_nonzero: ExprId,
    pub desc_step1: &'static str,
    pub desc_step2: &'static str,
    pub desc_step3: &'static str,
}

/// Detect and rewrite `tan(a-b)` to `(tan(a)-tan(b))/(1+tan(a)*tan(b))`.
pub fn try_rewrite_tan_difference_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TanDifferenceRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Tan) || args.len() != 1 {
        return None;
    }

    let Expr::Sub(a, b) = ctx.get(args[0]) else {
        return None;
    };
    let (a, b) = (*a, *b);

    let tan_a = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![a]);
    let tan_b = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![b]);
    let numerator = ctx.add(Expr::Sub(tan_a, tan_b));

    let tan_a2 = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![a]);
    let tan_b2 = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![b]);
    let product = ctx.add(Expr::Mul(tan_a2, tan_b2));
    let one = ctx.num(1);
    let denominator = ctx.add(Expr::Add(one, product));

    let rewritten = ctx.add(Expr::Div(numerator, denominator));
    Some(TanDifferenceRewrite { rewritten })
}

/// Plan the sum-to-product quotient rewrite:
/// - `(sin(A)+sin(B))/(cos(A)+cos(B)) -> sin((A+B)/2)/cos((A+B)/2)`
/// - `(sin(A)-sin(B))/(cos(A)+cos(B)) -> sin((A-B)/2)/cos((A-B)/2)`
///
/// Returns intermediate states to allow engine-side chained step narration.
pub fn try_plan_sin_cos_sum_quotient_div_expr(
    ctx: &mut Context,
    expr: ExprId,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> Option<TrigSumQuotientRewritePlan> {
    let (num_id, den_id) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return None;
    };

    let (cos_c, cos_d) = extract_trig_two_term_sum(ctx, den_id, "cos")?;

    enum NumeratorPattern {
        Sum { sin_a: ExprId, sin_b: ExprId },
        Diff { sin_a: ExprId, sin_b: ExprId },
    }

    let pattern = if let Some((sin_a, sin_b)) = extract_trig_two_term_sum(ctx, num_id, "sin") {
        NumeratorPattern::Sum { sin_a, sin_b }
    } else if let Some((sin_a, sin_b)) = extract_trig_two_term_diff(ctx, num_id, "sin") {
        NumeratorPattern::Diff { sin_a, sin_b }
    } else {
        return None;
    };

    let (sin_a, sin_b, is_diff) = match pattern {
        NumeratorPattern::Sum { sin_a, sin_b } => (sin_a, sin_b, false),
        NumeratorPattern::Diff { sin_a, sin_b } => (sin_a, sin_b, true),
    };

    if !args_match_as_multiset(ctx, sin_a, sin_b, cos_c, cos_d) {
        return None;
    }

    let avg = build_avg_with_simplifier(ctx, sin_a, sin_b, simplify_expr);
    let avg_normalized = normalize_for_even_fn(ctx, avg);

    if is_diff {
        let half_diff = build_half_diff_with_simplifier(ctx, sin_a, sin_b, false, simplify_expr);
        let half_diff_for_cos = normalize_for_even_fn(ctx, half_diff);

        let sin_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
        let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_for_cos]);
        let rewritten = ctx.add(Expr::Div(sin_half_diff, cos_half_diff));

        let two = ctx.num(2);
        let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg_normalized]);
        let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
        let num_product = smart_mul(ctx, cos_avg, sin_half);
        let intermediate_num = smart_mul(ctx, two, num_product);

        let two = ctx.num(2);
        let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg_normalized]);
        let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_for_cos]);
        let den_product = smart_mul(ctx, cos_avg, cos_half);
        let intermediate_den = smart_mul(ctx, two, den_product);

        let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
        let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

        return Some(TrigSumQuotientRewritePlan {
            num_id,
            den_id,
            intermediate_num,
            intermediate_den,
            state_after_step1,
            state_after_step2,
            rewritten,
            desc_step1: "sin(A)−sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)",
            desc_step2: "cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)",
            desc_step3: "Cancel common factors 2 and cos(avg)",
        });
    }

    let half_diff = build_half_diff_with_simplifier(ctx, sin_a, sin_b, true, simplify_expr);
    let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);

    let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
    let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
    let rewritten = ctx.add(Expr::Div(sin_avg, cos_avg));

    let two = ctx.num(2);
    let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);
    let sin_avg_for_num = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
    let num_product = smart_mul(ctx, sin_avg_for_num, cos_half_diff);
    let intermediate_num = smart_mul(ctx, two, num_product);

    let two = ctx.num(2);
    let cos_avg_for_den = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
    let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);
    let den_product = smart_mul(ctx, cos_avg_for_den, cos_half_diff);
    let intermediate_den = smart_mul(ctx, two, den_product);

    let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
    let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

    Some(TrigSumQuotientRewritePlan {
        num_id,
        den_id,
        intermediate_num,
        intermediate_den,
        state_after_step1,
        state_after_step2,
        rewritten,
        desc_step1: "sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)",
        desc_step2: "cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)",
        desc_step3: "Cancel common factors 2 and cos(half_diff)",
    })
}

/// Plan the quotient contraction:
/// - `(cos(A)-cos(B))/(sin(B)-sin(A)) -> tan((A+B)/2)`
/// - `(cos(A)-cos(B))/(sin(A)-sin(B)) -> -tan((A+B)/2)`
///
/// This keeps the rewrite narrow and lets the engine attach the nonzero
/// requirements introduced by canceling the shared sine factor.
pub fn try_plan_cos_diff_sin_diff_quotient_div_expr(
    ctx: &mut Context,
    expr: ExprId,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> Option<CosDiffSinDiffQuotientRewritePlan> {
    let (num_id, den_id) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return None;
    };

    let (cos_a, cos_b) = extract_trig_two_term_diff(ctx, num_id, "cos")?;
    let (sin_l, sin_r) = extract_trig_two_term_diff(ctx, den_id, "sin")?;

    let direct = cas_ast::ordering::compare_expr(ctx, cos_a, sin_l) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, cos_b, sin_r) == Ordering::Equal;
    let reversed = cas_ast::ordering::compare_expr(ctx, cos_a, sin_r) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, cos_b, sin_l) == Ordering::Equal;

    if !direct && !reversed {
        return None;
    }

    let avg = build_avg_with_simplifier(ctx, cos_a, cos_b, simplify_expr);
    let half_gap = build_half_diff_with_simplifier(ctx, sin_l, sin_r, false, simplify_expr);
    let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
    let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
    let common_factor = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_gap]);
    let two = ctx.num(2);

    let num_product = smart_mul(ctx, sin_avg, common_factor);
    let base_num = smart_mul(ctx, two, num_product);
    let intermediate_num = if reversed {
        base_num
    } else {
        ctx.add(Expr::Neg(base_num))
    };
    let den_product = smart_mul(ctx, cos_avg, common_factor);
    let intermediate_den = smart_mul(ctx, two, den_product);
    let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
    let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));
    let tan_avg = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![avg]);
    let rewritten = if reversed {
        tan_avg
    } else {
        ctx.add(Expr::Neg(tan_avg))
    };

    let (desc_step1, desc_step2) = if reversed {
        (
            "cos(A)−cos(B) = 2·sin((A+B)/2)·sin((B−A)/2)",
            "sin(B)−sin(A) = 2·cos((A+B)/2)·sin((B−A)/2)",
        )
    } else {
        (
            "cos(A)−cos(B) = -2·sin((A+B)/2)·sin((A−B)/2)",
            "sin(A)−sin(B) = 2·cos((A+B)/2)·sin((A−B)/2)",
        )
    };

    Some(CosDiffSinDiffQuotientRewritePlan {
        num_id,
        den_id,
        intermediate_num,
        intermediate_den,
        state_after_step1,
        state_after_step2,
        rewritten,
        introduced_nonzero: common_factor,
        result_nonzero: cos_avg,
        desc_step1,
        desc_step2,
        desc_step3: "Cancel common factors 2 and sin(half_gap)",
    })
}

/// Try the Werner product-to-sum identity:
/// `2 * sin(A) * cos(B) -> sin(A+B) + sin(A-B)`.
///
/// This intentionally matches the current engine behavior only (sin-cos form),
/// while preserving any factors outside the matched `2*sin(A)*cos(B)` kernel.
pub fn try_rewrite_product_to_sum_werner_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigSumProductRewrite> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut two_idx: Option<usize> = None;
    let mut sin_arg: Option<ExprId> = None;
    let mut sin_idx: Option<usize> = None;
    let mut cos_arg: Option<ExprId> = None;
    let mut cos_idx: Option<usize> = None;

    for (idx, &factor) in factors.iter().enumerate() {
        match ctx.get(factor) {
            Expr::Number(n) => {
                if *n == BigRational::from_integer(2.into()) && two_idx.is_none() {
                    two_idx = Some(idx);
                }
            }
            Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(*fn_id) {
                Some(cas_ast::BuiltinFn::Sin) if sin_arg.is_none() => {
                    sin_arg = Some(args[0]);
                    sin_idx = Some(idx);
                }
                Some(cas_ast::BuiltinFn::Cos) if cos_arg.is_none() => {
                    cos_arg = Some(args[0]);
                    cos_idx = Some(idx);
                }
                _ => {}
            },
            _ => {}
        }
    }

    let (two_idx, sin_idx, cos_idx, a, b) = match (two_idx, sin_idx, cos_idx, sin_arg, cos_arg) {
        (Some(two_idx), Some(sin_idx), Some(cos_idx), Some(a), Some(b)) => {
            (two_idx, sin_idx, cos_idx, a, b)
        }
        _ => return None,
    };

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let sin_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![a_plus_b]);
    let sin_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![a_minus_b]);
    let rewritten = ctx.add(Expr::Add(sin_sum, sin_diff));
    let rewritten = factors
        .into_iter()
        .enumerate()
        .filter_map(|(idx, factor)| {
            (idx != two_idx && idx != sin_idx && idx != cos_idx).then_some(factor)
        })
        .fold(rewritten, |acc, factor| smart_mul(ctx, acc, factor));

    Some(TrigSumProductRewrite {
        rewritten,
        kind: TrigSumProductRewriteKind::Werner,
    })
}

/// Product-to-sum (Werner) WITHOUT a literal factor of 2, emitting the
/// explicit `/2`. Covers all four combinations of a single sin/cos
/// product with DIFFERENT arguments:
///   sin(A) cos(B) -> (sin(A+B) + sin(A-B)) / 2
///   cos(A) cos(B) -> (cos(A+B) + cos(A-B)) / 2
///   sin(A) sin(B) -> (cos(A-B) - cos(A+B)) / 2
/// Gated on A != B so the equal-argument owners (sin(x)cos(x) -> sin^2/2,
/// sin^2, cos^2 power reduction) keep their forms. Used by the
/// integration-prep product-to-sum rule so integrate(sin(3x)cos(5x))
/// reaches a sum of single-frequency terms the integrator owns.
pub fn try_rewrite_product_to_sum_no_coefficient_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigSumProductRewrite> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    // Collect exactly two single-argument sin/cos factors; everything
    // else is a passthrough multiplier.
    let mut trig: Vec<(usize, bool, ExprId)> = Vec::new();
    for (idx, &factor) in factors.iter().enumerate() {
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if args.len() == 1 {
                match ctx.builtin_of(*fn_id) {
                    Some(cas_ast::BuiltinFn::Sin) => trig.push((idx, true, args[0])),
                    Some(cas_ast::BuiltinFn::Cos) => trig.push((idx, false, args[0])),
                    _ => {}
                }
            }
        }
    }
    if trig.len() != 2 {
        return None;
    }
    let (idx1, is_sin1, arg1) = trig[0];
    let (idx2, is_sin2, arg2) = trig[1];
    // A != B: equal arguments belong to the power-reduction / f*f' owners.
    if cas_ast::ordering::compare_expr(ctx, arg1, arg2) == Ordering::Equal {
        return None;
    }

    let inner = match (is_sin1, is_sin2) {
        (true, false) | (false, true) => {
            // sin(A) cos(B): A is the sine argument, B the cosine argument.
            let (sin_arg, cos_arg) = if is_sin1 { (arg1, arg2) } else { (arg2, arg1) };
            let sum = ctx.add(Expr::Add(sin_arg, cos_arg));
            let diff = ctx.add(Expr::Sub(sin_arg, cos_arg));
            let sin_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![sum]);
            let sin_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![diff]);
            ctx.add(Expr::Add(sin_sum, sin_diff))
        }
        (false, false) => {
            // cos(A) cos(B) -> cos(A+B) + cos(A-B).
            let sum = ctx.add(Expr::Add(arg1, arg2));
            let diff = ctx.add(Expr::Sub(arg1, arg2));
            let cos_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![sum]);
            let cos_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![diff]);
            ctx.add(Expr::Add(cos_sum, cos_diff))
        }
        (true, true) => {
            // sin(A) sin(B) -> cos(A-B) - cos(A+B).
            let sum = ctx.add(Expr::Add(arg1, arg2));
            let diff = ctx.add(Expr::Sub(arg1, arg2));
            let cos_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![sum]);
            let cos_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![diff]);
            ctx.add(Expr::Sub(cos_diff, cos_sum))
        }
    };

    // Multiply back the non-trig passthrough factors.
    let rewritten = factors
        .into_iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != idx1 && idx != idx2).then_some(factor))
        .fold(inner, |acc, factor| smart_mul(ctx, acc, factor));
    let two = ctx.num(2);
    let rewritten = ctx.add(Expr::Div(rewritten, two));

    Some(TrigSumProductRewrite {
        rewritten,
        kind: TrigSumProductRewriteKind::Werner,
    })
}

/// Standalone sum-to-product identities for two-term trig sums/differences.
///
/// Gating policy matches current engine behavior: apply only when both
/// arguments are rational multiples of π.
pub fn try_rewrite_trig_sum_to_product_expr(
    ctx: &mut Context,
    expr: ExprId,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> Option<TrigSumProductRewrite> {
    enum Pattern {
        SinSum { arg1: ExprId, arg2: ExprId },
        SinDiff { a: ExprId, b: ExprId },
        CosSum { arg1: ExprId, arg2: ExprId },
        CosDiff { a: ExprId, b: ExprId },
    }

    let pattern = if let Some((arg1, arg2)) = extract_trig_two_term_sum(ctx, expr, "sin") {
        Pattern::SinSum { arg1, arg2 }
    } else if let Some((a, b)) = extract_trig_two_term_diff(ctx, expr, "sin") {
        Pattern::SinDiff { a, b }
    } else if let Some((arg1, arg2)) = extract_trig_two_term_sum(ctx, expr, "cos") {
        Pattern::CosSum { arg1, arg2 }
    } else if let Some((a, b)) = extract_trig_two_term_diff(ctx, expr, "cos") {
        Pattern::CosDiff { a, b }
    } else {
        return None;
    };

    let (arg_a, arg_b, is_diff, fn_name) = match pattern {
        Pattern::SinSum { arg1, arg2 } => (arg1, arg2, false, "sin"),
        Pattern::SinDiff { a, b } => (a, b, true, "sin"),
        Pattern::CosSum { arg1, arg2 } => (arg1, arg2, false, "cos"),
        Pattern::CosDiff { a, b } => (a, b, true, "cos"),
    };

    if extract_rational_pi_multiple(ctx, arg_a).is_none()
        || extract_rational_pi_multiple(ctx, arg_b).is_none()
    {
        return None;
    }

    let avg = build_avg_with_simplifier(ctx, arg_a, arg_b, simplify_expr);
    let half_diff = build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, simplify_expr);
    let two = ctx.num(2);

    let (rewritten, kind) = match (fn_name, is_diff) {
        ("sin", false) => {
            let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
            let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff]);
            let product = smart_mul(ctx, sin_avg, cos_half);
            (
                smart_mul(ctx, two, product),
                TrigSumProductRewriteKind::SinSum,
            )
        }
        ("sin", true) => {
            let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
            let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
            let product = smart_mul(ctx, cos_avg, sin_half);
            (
                smart_mul(ctx, two, product),
                TrigSumProductRewriteKind::SinDiff,
            )
        }
        ("cos", false) => {
            let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);
            let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
            let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);
            let product = smart_mul(ctx, cos_avg, cos_half);
            (
                smart_mul(ctx, two, product),
                TrigSumProductRewriteKind::CosSum,
            )
        }
        ("cos", true) => {
            let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
            let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
            let product = smart_mul(ctx, sin_avg, sin_half);
            let two_product = smart_mul(ctx, two, product);
            (
                ctx.add(Expr::Neg(two_product)),
                TrigSumProductRewriteKind::CosDiff,
            )
        }
        _ => return None,
    };

    Some(TrigSumProductRewrite { rewritten, kind })
}

/// Full product-to-sum identities:
/// - `2*sin(a)*cos(b) -> sin(a+b) + sin(a-b)`
/// - `2*cos(a)*sin(b) -> sin(a+b) - sin(a-b)`
/// - `2*cos(a)*cos(b) -> cos(a+b) + cos(a-b)`
/// - `2*sin(a)*sin(b) -> cos(a-b) - cos(a+b)`
///
/// Preserves any extra multiplicative factors by multiplying them back.
pub fn try_rewrite_product_to_sum_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigProductToSumRewrite> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    let mut has_two = false;
    let mut two_idx: Option<usize> = None;
    let mut trig_funcs: Vec<(usize, &'static str, ExprId)> = Vec::new();

    for (i, &factor) in factors.iter().enumerate() {
        match ctx.get(factor) {
            Expr::Number(n) => {
                if *n == BigRational::from_integer(2.into()) {
                    has_two = true;
                    two_idx = Some(i);
                }
            }
            Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(*fn_id) {
                Some(cas_ast::BuiltinFn::Sin) => trig_funcs.push((i, "sin", args[0])),
                Some(cas_ast::BuiltinFn::Cos) => trig_funcs.push((i, "cos", args[0])),
                _ => {}
            },
            _ => {}
        }
    }

    if !has_two || trig_funcs.len() != 2 {
        return None;
    }

    let (idx1, name1, arg1) = trig_funcs[0];
    let (idx2, name2, arg2) = trig_funcs[1];
    let two_idx = two_idx?;

    let mut remaining: Vec<ExprId> = Vec::new();
    for (i, &factor) in factors.iter().enumerate() {
        if i != two_idx && i != idx1 && i != idx2 {
            remaining.push(factor);
        }
    }

    let (rewritten, kind) = match (name1, name2) {
        ("sin", "cos") => {
            let sum_arg = ctx.add(Expr::Add(arg1, arg2));
            let diff_arg = ctx.add(Expr::Sub(arg1, arg2));
            let sin_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![sum_arg]);
            let sin_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![diff_arg]);
            (
                ctx.add(Expr::Add(sin_sum, sin_diff)),
                TrigProductToSumRewriteKind::SinCos,
            )
        }
        ("cos", "sin") => {
            let sum_arg = ctx.add(Expr::Add(arg1, arg2));
            let diff_arg = ctx.add(Expr::Sub(arg1, arg2));
            let sin_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![sum_arg]);
            let sin_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![diff_arg]);
            (
                ctx.add(Expr::Sub(sin_sum, sin_diff)),
                TrigProductToSumRewriteKind::CosSin,
            )
        }
        ("cos", "cos") => {
            let sum_arg = ctx.add(Expr::Add(arg1, arg2));
            let diff_arg = ctx.add(Expr::Sub(arg1, arg2));
            let cos_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![sum_arg]);
            let cos_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![diff_arg]);
            (
                ctx.add(Expr::Add(cos_sum, cos_diff)),
                TrigProductToSumRewriteKind::CosCos,
            )
        }
        ("sin", "sin") => {
            let sum_arg = ctx.add(Expr::Add(arg1, arg2));
            let diff_arg = ctx.add(Expr::Sub(arg1, arg2));
            let cos_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![sum_arg]);
            let cos_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![diff_arg]);
            (
                ctx.add(Expr::Sub(cos_diff, cos_sum)),
                TrigProductToSumRewriteKind::SinSin,
            )
        }
        _ => return None,
    };

    let rewritten = remaining
        .into_iter()
        .fold(rewritten, |acc, factor| smart_mul(ctx, acc, factor));
    Some(TrigProductToSumRewrite { rewritten, kind })
}

/// Contraction identities:
/// - `sin(a)+sin(b) -> 2*sin((a+b)/2)*cos((a-b)/2)`
/// - `sin(a)-sin(b) -> 2*cos((a+b)/2)*sin((a-b)/2)`
/// - `cos(a)+cos(b) -> 2*cos((a+b)/2)*cos((a-b)/2)`
/// - `cos(a)-cos(b) -> -2*sin((a+b)/2)*sin((a-b)/2)`
///
/// Applies only when both arguments are linear multiples of the same base.
pub fn try_rewrite_sum_to_product_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigSumToProductContractionRewrite> {
    let (left, right, is_add) = match ctx.get(expr) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

    let (l_name, l_arg) = extract_sin_cos_fn_arg(ctx, left)?;
    let (r_name, r_arg) = extract_sin_cos_fn_arg(ctx, right)?;
    if l_name != r_name {
        return None;
    }

    let (base, coef_a, coef_b) = extract_linear_coefficients(ctx, l_arg, r_arg)?;
    let sum_coef = &coef_a + &coef_b;
    let diff_coef = &coef_a - &coef_b;
    let two = BigRational::from_integer(2.into());
    let half_sum = sum_coef / &two;
    let half_diff = diff_coef / &two;

    let half_sum_arg = build_coef_times_base(ctx, &half_sum, base);
    let half_diff_arg = build_coef_times_base(ctx, &half_diff, base);
    let two_id = ctx.num(2);

    let (rewritten, kind) = match (l_name, is_add) {
        ("sin", true) => {
            let sin_half_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_sum_arg]);
            let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(sin_half_sum, cos_half_diff));
            (
                ctx.add(Expr::Mul(two_id, product)),
                TrigSumToProductContractionRewriteKind::SinSum,
            )
        }
        ("sin", false) => {
            let cos_half_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_sum_arg]);
            let sin_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(cos_half_sum, sin_half_diff));
            (
                ctx.add(Expr::Mul(two_id, product)),
                TrigSumToProductContractionRewriteKind::SinDiff,
            )
        }
        ("cos", true) => {
            let cos_half_sum = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_sum_arg]);
            let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(cos_half_sum, cos_half_diff));
            (
                ctx.add(Expr::Mul(two_id, product)),
                TrigSumToProductContractionRewriteKind::CosSum,
            )
        }
        ("cos", false) => {
            let sin_half_sum = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_sum_arg]);
            let sin_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(sin_half_sum, sin_half_diff));
            let two_times = ctx.add(Expr::Mul(two_id, product));
            (
                ctx.add(Expr::Neg(two_times)),
                TrigSumToProductContractionRewriteKind::CosDiff,
            )
        }
        _ => return None,
    };

    Some(TrigSumToProductContractionRewrite { rewritten, kind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_equality::SemanticEqualityChecker;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn extract_sum_and_diff_patterns() {
        let mut ctx = Context::new();
        let sum = parse("sin(a)+sin(b)", &mut ctx).expect("sum");
        let diff = parse("sin(a)-sin(b)", &mut ctx).expect("diff");

        let sum_args = extract_trig_two_term_sum(&ctx, sum, "sin");
        let diff_args = extract_trig_two_term_diff(&ctx, diff, "sin");

        assert!(sum_args.is_some());
        assert!(diff_args.is_some());
    }

    #[test]
    fn multiset_match_is_order_invariant() {
        let mut ctx = Context::new();
        let a = parse("x", &mut ctx).expect("a");
        let b = parse("y", &mut ctx).expect("b");
        assert!(args_match_as_multiset(&ctx, a, b, b, a));
    }

    #[test]
    fn normalize_for_even_strips_negation_shapes() {
        let mut ctx = Context::new();
        let neg = parse("-x", &mut ctx).expect("neg");
        let mul = parse("-1*x", &mut ctx).expect("mul");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, normalize_for_even_fn(&ctx, neg), x),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, normalize_for_even_fn(&ctx, mul), x),
            Ordering::Equal
        );
    }

    #[test]
    fn simplify_numeric_div_reduces_linear_forms() {
        let mut ctx = Context::new();
        let expr = parse("(4*x)/2", &mut ctx).expect("expr");
        let simplified = simplify_numeric_div(&mut ctx, expr);
        let expected = parse("2*x", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, simplified, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn build_half_diff_with_simplifier_respects_canonical_order() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let y = parse("y", &mut ctx).expect("y");
        let x = parse("x", &mut ctx).expect("x");

        let canonical = build_half_diff_with_simplifier(&mut ctx, y, x, true, passthrough);
        let expected_canonical = parse("(x-y)/2", &mut ctx).expect("expected_canonical");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, canonical, expected_canonical),
            Ordering::Equal
        );

        let non_canonical = build_half_diff_with_simplifier(&mut ctx, y, x, false, passthrough);
        let expected_non_canonical = parse("(y-x)/2", &mut ctx).expect("expected_non_canonical");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, non_canonical, expected_non_canonical),
            Ordering::Equal
        );
    }

    #[test]
    fn build_avg_with_simplifier_applies_callback_before_division() {
        fn collapse_duplicate_add(ctx: &mut Context, expr: ExprId) -> ExprId {
            let (l, r) = if let Expr::Add(l, r) = ctx.get(expr) {
                (*l, *r)
            } else {
                return expr;
            };
            if cas_ast::ordering::compare_expr(ctx, l, r) == Ordering::Equal {
                let two = ctx.num(2);
                return ctx.add(Expr::Mul(two, l));
            }
            expr
        }

        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let avg = build_avg_with_simplifier(&mut ctx, x, x, collapse_duplicate_add);
        let expected = parse("x", &mut ctx).expect("expected");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, avg, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_werner_sin_cos_product() {
        let mut ctx = Context::new();
        let expr = parse("2*sin(x)*cos(y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_product_to_sum_werner_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("sin(x+y)+sin(x-y)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_werner_sin_cos_product_preserving_extra_factors() {
        let mut ctx = Context::new();
        let expr = parse("2*2*sin(x)*cos(y)*z", &mut ctx).expect("parse");
        let rewrite = try_rewrite_product_to_sum_werner_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rendered(&ctx, rewrite.rewritten),
            "(sin(x + y) + sin(x - y)) * 2 * z"
        );
    }

    #[test]
    fn does_not_rewrite_without_required_shape() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)*cos(y)", &mut ctx).expect("parse");
        assert!(try_rewrite_product_to_sum_werner_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn no_coefficient_product_to_sum_covers_all_combinations() {
        let mut ctx = Context::new();
        for (source, expected) in [
            (
                "sin(3*x)*cos(5*x)",
                "(sin(3 * x + 5 * x) + sin(3 * x - 5 * x)) / 2",
            ),
            (
                "cos(3*x)*cos(5*x)",
                "(cos(3 * x + 5 * x) + cos(3 * x - 5 * x)) / 2",
            ),
            (
                "sin(3*x)*sin(5*x)",
                "(cos(3 * x - 5 * x) - cos(3 * x + 5 * x)) / 2",
            ),
        ] {
            let expr = parse(source, &mut ctx).expect(source);
            let rewrite = try_rewrite_product_to_sum_no_coefficient_expr(&mut ctx, expr)
                .unwrap_or_else(|| panic!("must rewrite: {source}"));
            assert_eq!(rendered(&ctx, rewrite.rewritten), expected, "{source}");
        }
    }

    #[test]
    fn no_coefficient_product_to_sum_declines_equal_arguments() {
        // Equal arguments keep their power-reduction / f*f' owners; a
        // single non-product trig factor has nothing to pair with.
        let mut ctx = Context::new();
        for source in ["sin(x)*cos(x)", "cos(2*x)*cos(2*x)", "sin(3*x)", "sin(x)*x"] {
            let expr = parse(source, &mut ctx).expect(source);
            assert!(
                try_rewrite_product_to_sum_no_coefficient_expr(&mut ctx, expr).is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn rewrites_tan_difference_identity_expansion() {
        let mut ctx = Context::new();
        let expr = parse("tan(a-b)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_tan_difference_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("(tan(a)-tan(b))/(1+tan(a)*tan(b))", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn plans_sum_quotient_rewrite_for_sum_case() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("(sin(a)+sin(b))/(cos(a)+cos(b))", &mut ctx).expect("expr");
        let plan =
            try_plan_sin_cos_sum_quotient_div_expr(&mut ctx, expr, passthrough).expect("plan");
        let expected = parse("sin((a+b)/2)/cos((a+b)/2)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
        assert_eq!(
            plan.desc_step1,
            "sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)"
        );
    }

    #[test]
    fn plans_sum_quotient_rewrite_for_diff_case() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("(sin(a)-sin(b))/(cos(a)+cos(b))", &mut ctx).expect("expr");
        let plan =
            try_plan_sin_cos_sum_quotient_div_expr(&mut ctx, expr, passthrough).expect("plan");
        let expected = parse("sin((a-b)/2)/cos((a-b)/2)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
        assert_eq!(
            plan.desc_step1,
            "sin(A)−sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)"
        );
    }

    #[test]
    fn does_not_plan_sum_quotient_for_mismatched_args() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("(sin(a)+sin(b))/(cos(a)+cos(c))", &mut ctx).expect("expr");
        assert!(try_plan_sin_cos_sum_quotient_div_expr(&mut ctx, expr, passthrough).is_none());
    }

    #[test]
    fn plans_cos_diff_sin_diff_quotient_to_tan_avg() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("(cos(x)-cos(3*x))/(sin(3*x)-sin(x))", &mut ctx).expect("expr");
        let plan = try_plan_cos_diff_sin_diff_quotient_div_expr(&mut ctx, expr, passthrough)
            .expect("plan");
        let expected = parse("tan((x + 3*x)/2)", &mut ctx).expect("expected");
        let introduced = parse("sin((3*x - x)/2)", &mut ctx).expect("introduced");
        let result_nonzero = parse("cos((x + 3*x)/2)", &mut ctx).expect("result_nonzero");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.introduced_nonzero, introduced),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.result_nonzero, result_nonzero),
            Ordering::Equal
        );
        assert_eq!(
            plan.desc_step1,
            "cos(A)−cos(B) = 2·sin((A+B)/2)·sin((B−A)/2)"
        );
    }

    #[test]
    fn does_not_plan_cos_diff_sin_diff_quotient_for_mismatched_args() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("(cos(a)-cos(b))/(sin(c)-sin(a))", &mut ctx).expect("expr");
        assert!(
            try_plan_cos_diff_sin_diff_quotient_div_expr(&mut ctx, expr, passthrough).is_none()
        );
    }

    #[test]
    fn rewrites_product_to_sum_sin_sin() {
        let mut ctx = Context::new();
        let expr = parse("2*sin(x)*sin(y)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_product_to_sum_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("cos(x-y)-cos(x+y)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_product_to_sum_with_remaining_factor() {
        let mut ctx = Context::new();
        let expr = parse("2*sin(x)*cos(y)*z", &mut ctx).expect("expr");
        let rewrite = try_rewrite_product_to_sum_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("(sin(x+y)+sin(x-y))*z", &mut ctx).expect("expected");
        let sem = SemanticEqualityChecker::new(&ctx);
        assert!(sem.are_equal(rewrite.rewritten, expected));
    }

    #[test]
    fn rewrites_product_to_sum_with_additive_rational_remaining_factor() {
        let mut ctx = Context::new();
        let expr = parse("(1/x + 1/(x+1))*(2*sin(x)*cos(2*x))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_product_to_sum_expr(&mut ctx, expr).expect("rewrite");
        let expected =
            parse("(sin(x + 2*x) + sin(x - 2*x))*(1/x + 1/(x+1))", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_sum_to_product_contraction_for_sine_sum() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)+sin(3*x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_sum_to_product_contraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("2*sin(2*x)*cos(-x)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_sum_to_product_contraction_for_sine_difference_special() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*x)-sin(x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_sum_to_product_contraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("2*cos(2*x)*sin(x)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn does_not_rewrite_sum_to_product_for_mixed_functions() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)+cos(x)", &mut ctx).expect("expr");
        assert!(try_rewrite_sum_to_product_contraction_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rewrites_trig_sum_to_product_when_both_args_are_pi_multiples() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("sin(pi/3)+sin(pi/6)", &mut ctx).expect("expr");
        let rewrite =
            try_rewrite_trig_sum_to_product_expr(&mut ctx, expr, passthrough).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(
            rewritten_str,
            "2 * sin((pi / 3 + pi / 6) / 2) * cos((pi / 3 - pi / 6) / 2)"
        );
    }

    #[test]
    fn does_not_rewrite_trig_sum_to_product_for_symbolic_args() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let expr = parse("sin(a)+sin(b)", &mut ctx).expect("expr");
        assert!(try_rewrite_trig_sum_to_product_expr(&mut ctx, expr, passthrough).is_none());
    }
}
