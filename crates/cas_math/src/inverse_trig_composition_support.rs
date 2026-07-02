use crate::expr_nary::build_balanced_add;
use crate::expr_relations::{extract_negated_inner, is_negation};
use crate::expr_terms::collect_additive_terms_flat_add;
use crate::fraction_forms::are_denominators_opposite;
use crate::numeric_eval::as_rational_const;
use crate::root_forms::extract_square_root_base;
use crate::trig_reciprocal_support::{are_reciprocals, has_reciprocal_atan_pair};
use crate::trig_roots_flatten::flatten_add_sub_chain;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_traits::One;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseTrigCompositionKind {
    SinArcsin,
    CosArccos,
    CosArcsin,
    SinArccos,
    TanArcsin,
    SinArctan,
    CosArctan,
    TanArctan,
    ArctanTanArctan,
    ArcsinSinArctan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseTrigCompositionPlan {
    pub kind: InverseTrigCompositionKind,
    pub assume_defined: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseTrigCompositionRewritePlan {
    pub rewritten: ExprId,
    pub kind: InverseTrigCompositionKind,
    pub assume_defined_expr: Option<ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InverseTrigCompositionMode {
    Assume,
    Strict,
    Generic,
}

fn inverse_trig_composition_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> InverseTrigCompositionMode {
    if assume_mode {
        InverseTrigCompositionMode::Assume
    } else if strict_mode {
        InverseTrigCompositionMode::Strict
    } else {
        InverseTrigCompositionMode::Generic
    }
}

/// Domain-policy planner for `sin(arcsin(x))` and `cos(arccos(x))`.
///
/// `arg_in_unit_interval_proven` should be true only when strict mode can prove
/// the inverse-function input lies in `[-1, 1]`.
pub(crate) fn plan_inverse_trig_composition_with_mode_flags(
    kind: InverseTrigCompositionKind,
    arg_in_unit_interval_proven: bool,
    assume_mode: bool,
    strict_mode: bool,
) -> Option<InverseTrigCompositionPlan> {
    let mode = inverse_trig_composition_mode_from_flags(assume_mode, strict_mode);
    match mode {
        InverseTrigCompositionMode::Strict => {
            if arg_in_unit_interval_proven {
                Some(InverseTrigCompositionPlan {
                    kind,
                    assume_defined: false,
                })
            } else {
                None
            }
        }
        InverseTrigCompositionMode::Generic => Some(InverseTrigCompositionPlan {
            kind,
            assume_defined: false,
        }),
        InverseTrigCompositionMode::Assume => Some(InverseTrigCompositionPlan {
            kind,
            assume_defined: true,
        }),
    }
}

/// Strict-mode helper: checks whether `expr` is a numeric literal in `[-1, 1]`.
pub(crate) fn is_number_in_unit_interval(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        let one = num_rational::BigRational::one();
        let neg_one = -one.clone();
        *n >= neg_one && *n <= one
    } else {
        false
    }
}

/// True when `expr` is a numeric literal strictly outside `[-1, 1]` — i.e.
/// provably OUTSIDE the domain of `arcsin`/`arccos`. Then any forward composition
/// `sin(arcsin(c))` / `cos(arccos(c))` is undefined over the reals (the inner
/// inverse has no real value), so it must NOT simplify to `c`.
fn is_number_outside_unit_interval(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        let one = num_rational::BigRational::one();
        *n > one || *n < -one
    } else {
        false
    }
}

fn is_exact_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == num_rational::BigRational::one())
}

fn is_exact_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n)
            if *n == num_rational::BigRational::from_integer(2.into())
    )
}

fn match_square_base_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if is_exact_two(ctx, *exp) {
        Some(*base)
    } else {
        None
    }
}

fn match_one_plus_square_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let terms = collect_additive_terms_flat_add(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut saw_one = false;
    let mut square_base = None;
    for term in terms {
        if is_exact_one(ctx, term) {
            if saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }
        let base = match_square_base_expr(ctx, term)?;
        if square_base.is_some() {
            return None;
        }
        square_base = Some(base);
    }

    if saw_one {
        square_base
    } else {
        None
    }
}

fn match_sqrt_one_plus_square_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, expr)?;
    match_one_plus_square_expr(ctx, radicand)
}

fn match_reciprocal_sqrt_one_plus_square_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exp_value = as_rational_const(ctx, *exp)?;
    if *exp_value.numer() != (-1).into() || *exp_value.denom() != 2.into() {
        return None;
    }
    match_one_plus_square_expr(ctx, *base)
}

fn collect_mul_factors(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors(ctx, *left, factors);
            collect_mul_factors(ctx, *right, factors);
        }
        _ => factors.push(expr),
    }
}

fn match_var_times_sqrt_one_plus_square_expr(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let mut factors = Vec::new();
    collect_mul_factors(ctx, expr, &mut factors);
    if factors.len() != 2 {
        return false;
    }

    let mut saw_var = false;
    let mut saw_sqrt = false;
    for factor in factors {
        if !saw_var && compare_expr(ctx, factor, var) == std::cmp::Ordering::Equal {
            saw_var = true;
            continue;
        }
        if !saw_sqrt
            && match_sqrt_one_plus_square_expr(ctx, factor).is_some_and(|sqrt_var| {
                compare_expr(ctx, sqrt_var, var) == std::cmp::Ordering::Equal
            })
        {
            saw_sqrt = true;
            continue;
        }
        return false;
    }

    saw_var && saw_sqrt
}

fn match_var_times_reciprocal_sqrt_one_plus_square_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let mut factors = Vec::new();
    collect_mul_factors(ctx, expr, &mut factors);
    if factors.len() != 2 {
        return None;
    }

    for (candidate_var, candidate_root) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        if let Some(root_var) = match_reciprocal_sqrt_one_plus_square_expr(ctx, candidate_root) {
            if compare_expr(ctx, candidate_var, root_var) == std::cmp::Ordering::Equal {
                return Some(candidate_var);
            }
        }
    }

    None
}

fn build_sqrt_one_plus_square(ctx: &mut Context, x: ExprId) -> ExprId {
    let two = ctx.num(2);
    let x_squared = ctx.add(Expr::Pow(x, two));
    let one = ctx.num(1);
    let radicand = ctx.add(Expr::Add(x_squared, one));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand])
}

fn build_sqrt_one_minus_square(ctx: &mut Context, x: ExprId) -> ExprId {
    let two = ctx.num(2);
    let x_squared = ctx.add(Expr::Pow(x, two));
    let one = ctx.num(1);
    let radicand = ctx.add(Expr::Sub(one, x_squared));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand])
}

fn build_sin_arctan_projection(ctx: &mut Context, x: ExprId) -> ExprId {
    let denominator = build_sqrt_one_plus_square(ctx, x);
    ctx.add(Expr::Div(x, denominator))
}

fn build_cos_arctan_projection(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let denominator = build_sqrt_one_plus_square(ctx, x);
    ctx.add(Expr::Div(one, denominator))
}

fn build_tan_arcsin_projection(ctx: &mut Context, x: ExprId) -> ExprId {
    let denominator = build_sqrt_one_minus_square(ctx, x);
    ctx.add(Expr::Div(x, denominator))
}

fn try_match_safe_arcsin_sin_arctan_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(var) = match_var_times_reciprocal_sqrt_one_plus_square_expr(ctx, expr) {
        return Some(var);
    }

    if let Expr::Function(sin_name, sin_args) = ctx.get(expr) {
        let is_sin = ctx.is_builtin(*sin_name, BuiltinFn::Sin);
        if is_sin && sin_args.len() == 1 {
            let atan_expr = sin_args[0];
            if let Expr::Function(atan_name, atan_args) = ctx.get(atan_expr) {
                let is_atan = ctx.is_builtin(*atan_name, BuiltinFn::Arctan)
                    || ctx.is_builtin(*atan_name, BuiltinFn::Atan);
                if is_atan && atan_args.len() == 1 {
                    return Some(atan_args[0]);
                }
            }
        }
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };

    if let Some(var) = match_sqrt_one_plus_square_expr(ctx, *den) {
        if compare_expr(ctx, *num, var) == std::cmp::Ordering::Equal {
            return Some(var);
        }
    }

    if let Some(var) = match_one_plus_square_expr(ctx, *den) {
        if match_var_times_sqrt_one_plus_square_expr(ctx, *num, var) {
            return Some(var);
        }
    }

    None
}

/// Plan direct inverse-trig compositions (`sin(arcsin(x))`, etc.) under domain policy.
pub fn try_plan_inverse_trig_composition_expr(
    ctx: &mut Context,
    expr: ExprId,
    assume_mode: bool,
    strict_mode: bool,
) -> Option<InverseTrigCompositionRewritePlan> {
    let Expr::Function(outer_name, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }

    let inner_expr = outer_args[0];
    let inner_fn_info = match ctx.get(inner_expr) {
        Expr::Function(inner_name, inner_args) if inner_args.len() == 1 => {
            Some((*inner_name, inner_args[0]))
        }
        _ => None,
    };

    if let Some((inner_name, x)) = inner_fn_info {
        // An inner arcsin/arccos of a literal outside [-1, 1] is undefined over the
        // reals, so NO forward composition simplifies: `sin(arcsin(2))` is not `2`,
        // `cos(arccos(5))` is not `5`. Keep it symbolic.
        if (ctx.is_builtin(inner_name, BuiltinFn::Arcsin)
            || ctx.is_builtin(inner_name, BuiltinFn::Asin)
            || ctx.is_builtin(inner_name, BuiltinFn::Arccos)
            || ctx.is_builtin(inner_name, BuiltinFn::Acos))
            && is_number_outside_unit_interval(ctx, x)
        {
            return None;
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Sin)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arcsin)
                || ctx.is_builtin(inner_name, BuiltinFn::Asin))
        {
            let plan = plan_inverse_trig_composition_with_mode_flags(
                InverseTrigCompositionKind::SinArcsin,
                is_number_in_unit_interval(ctx, x),
                assume_mode,
                strict_mode,
            )?;
            return Some(InverseTrigCompositionRewritePlan {
                rewritten: x,
                kind: plan.kind,
                assume_defined_expr: if plan.assume_defined { Some(x) } else { None },
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Cos)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arccos)
                || ctx.is_builtin(inner_name, BuiltinFn::Acos))
        {
            let plan = plan_inverse_trig_composition_with_mode_flags(
                InverseTrigCompositionKind::CosArccos,
                is_number_in_unit_interval(ctx, x),
                assume_mode,
                strict_mode,
            )?;
            return Some(InverseTrigCompositionRewritePlan {
                rewritten: x,
                kind: plan.kind,
                assume_defined_expr: if plan.assume_defined { Some(x) } else { None },
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Cos)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arcsin)
                || ctx.is_builtin(inner_name, BuiltinFn::Asin))
        {
            let rewritten = build_sqrt_one_minus_square(ctx, x);
            return Some(InverseTrigCompositionRewritePlan {
                rewritten,
                kind: InverseTrigCompositionKind::CosArcsin,
                assume_defined_expr: None,
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Sin)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arccos)
                || ctx.is_builtin(inner_name, BuiltinFn::Acos))
        {
            let rewritten = build_sqrt_one_minus_square(ctx, x);
            return Some(InverseTrigCompositionRewritePlan {
                rewritten,
                kind: InverseTrigCompositionKind::SinArccos,
                assume_defined_expr: None,
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Tan)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arcsin)
                || ctx.is_builtin(inner_name, BuiltinFn::Asin))
        {
            let rewritten = build_tan_arcsin_projection(ctx, x);
            return Some(InverseTrigCompositionRewritePlan {
                rewritten,
                kind: InverseTrigCompositionKind::TanArcsin,
                assume_defined_expr: None,
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Tan)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arctan)
                || ctx.is_builtin(inner_name, BuiltinFn::Atan))
        {
            return Some(InverseTrigCompositionRewritePlan {
                rewritten: x,
                kind: InverseTrigCompositionKind::TanArctan,
                assume_defined_expr: None,
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Sin)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arctan)
                || ctx.is_builtin(inner_name, BuiltinFn::Atan))
        {
            let rewritten = build_sin_arctan_projection(ctx, x);
            return Some(InverseTrigCompositionRewritePlan {
                rewritten,
                kind: InverseTrigCompositionKind::SinArctan,
                assume_defined_expr: None,
            });
        }

        if ctx.is_builtin(*outer_name, BuiltinFn::Cos)
            && (ctx.is_builtin(inner_name, BuiltinFn::Arctan)
                || ctx.is_builtin(inner_name, BuiltinFn::Atan))
        {
            let rewritten = build_cos_arctan_projection(ctx, x);
            return Some(InverseTrigCompositionRewritePlan {
                rewritten,
                kind: InverseTrigCompositionKind::CosArctan,
                assume_defined_expr: None,
            });
        }

        let is_outer_arctan = ctx.is_builtin(*outer_name, BuiltinFn::Arctan)
            || ctx.is_builtin(*outer_name, BuiltinFn::Atan);
        let is_inner_tan = ctx.is_builtin(inner_name, BuiltinFn::Tan);
        if is_outer_arctan && is_inner_tan {
            if let Expr::Function(innermost_name, innermost_args) = ctx.get(x) {
                let is_innermost_arctan = ctx.is_builtin(*innermost_name, BuiltinFn::Arctan)
                    || ctx.is_builtin(*innermost_name, BuiltinFn::Atan);
                if is_innermost_arctan && innermost_args.len() == 1 {
                    return Some(InverseTrigCompositionRewritePlan {
                        rewritten: x,
                        kind: InverseTrigCompositionKind::ArctanTanArctan,
                        assume_defined_expr: None,
                    });
                }
            }
        }
    }

    if (ctx.is_builtin(*outer_name, BuiltinFn::Arcsin)
        || ctx.is_builtin(*outer_name, BuiltinFn::Asin))
        && try_match_safe_arcsin_sin_arctan_argument(ctx, inner_expr).is_some()
    {
        let x = try_match_safe_arcsin_sin_arctan_argument(ctx, inner_expr)?;
        let rewritten = ctx.call_builtin(BuiltinFn::Arctan, vec![x]);
        return Some(InverseTrigCompositionRewritePlan {
            rewritten,
            kind: InverseTrigCompositionKind::ArcsinSinArctan,
            assume_defined_expr: None,
        });
    }

    None
}

/// Build sum of all terms except indices `skip_i` and `skip_j`.
///
/// Returns `None` when no terms remain.
pub(crate) fn build_sum_without(
    ctx: &mut Context,
    terms: &[ExprId],
    skip_i: usize,
    skip_j: usize,
) -> Option<ExprId> {
    let remaining: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != skip_i && *idx != skip_j)
        .map(|(_, &term)| term)
        .collect();

    match remaining.len() {
        0 => None,
        _ => Some(build_balanced_add(ctx, &remaining)),
    }
}

/// Combine optional additive base with a new term.
pub(crate) fn combine_with_term(
    ctx: &mut Context,
    base: Option<ExprId>,
    new_term: ExprId,
) -> ExprId {
    match base {
        None => new_term,
        Some(b) => ctx.add(Expr::Add(b, new_term)),
    }
}

#[derive(Debug, Clone)]
pub struct PairWithNegationPlan {
    pub final_result: ExprId,
    pub local_before: ExprId,
    pub local_after: ExprId,
    pub desc: String,
}

/// Generic planner for additive pair identities with negation/coefficient support.
///
/// If `check_fn(f(x), g(x)) = (V, desc)`, this planner also handles:
/// - `-f(x) - g(x) -> -V`
/// - `k*f(x) + k*g(x) -> k*V`
/// - `k*(-f(x)) + k*(-g(x)) -> -k*V`
#[allow(clippy::too_many_arguments)] // all arguments are semantically distinct inputs
pub(crate) fn try_plan_pair_with_negation<F>(
    ctx: &mut Context,
    term_i: ExprId,
    term_j: ExprId,
    terms: &[ExprId],
    i: usize,
    j: usize,
    check_fn: F,
) -> Option<PairWithNegationPlan>
where
    F: Fn(&mut Context, &Expr, &Expr) -> Option<(ExprId, String)>,
{
    let term_i_data = ctx.get(term_i).clone();
    let term_j_data = ctx.get(term_j).clone();

    if let Some((result, desc)) = check_fn(ctx, &term_i_data, &term_j_data) {
        let remaining = build_sum_without(ctx, terms, i, j);
        let final_result = combine_with_term(ctx, remaining, result);
        let local_before = ctx.add(Expr::Add(term_i, term_j));
        return Some(PairWithNegationPlan {
            final_result,
            local_before,
            local_after: result,
            desc,
        });
    }

    if let (Expr::Neg(inner_i), Expr::Neg(inner_j)) = (&term_i_data, &term_j_data) {
        let inner_i_data = ctx.get(*inner_i).clone();
        let inner_j_data = ctx.get(*inner_j).clone();
        if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
            let neg_result = ctx.add(Expr::Neg(result));
            let remaining = build_sum_without(ctx, terms, i, j);
            let final_result = combine_with_term(ctx, remaining, neg_result);
            let local_before = ctx.add(Expr::Add(term_i, term_j));
            return Some(PairWithNegationPlan {
                final_result,
                local_before,
                local_after: neg_result,
                desc: format!("-[{}]", desc),
            });
        }
    }

    if let (Expr::Mul(coef_i, inner_i), Expr::Mul(coef_j, inner_j)) = (&term_i_data, &term_j_data) {
        if compare_expr(ctx, *coef_i, *coef_j) == std::cmp::Ordering::Equal {
            let inner_i_data = ctx.get(*inner_i).clone();
            let inner_j_data = ctx.get(*inner_j).clone();
            if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
                let scaled_result = ctx.add(Expr::Mul(*coef_i, result));
                let remaining = build_sum_without(ctx, terms, i, j);
                let final_result = combine_with_term(ctx, remaining, scaled_result);
                let local_before = ctx.add(Expr::Add(term_i, term_j));
                return Some(PairWithNegationPlan {
                    final_result,
                    local_before,
                    local_after: scaled_result,
                    desc: format!("k·[{}]", desc),
                });
            }
        }
    }

    if let (Expr::Mul(coef_i, inner_i), Expr::Mul(coef_j, inner_j)) = (&term_i_data, &term_j_data) {
        if compare_expr(ctx, *coef_i, *coef_j) == std::cmp::Ordering::Equal {
            if let (Expr::Neg(neg_i), Expr::Neg(neg_j)) = (ctx.get(*inner_i), ctx.get(*inner_j)) {
                let inner_i_data = ctx.get(*neg_i).clone();
                let inner_j_data = ctx.get(*neg_j).clone();
                if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
                    let scaled_result = ctx.add(Expr::Mul(*coef_i, result));
                    let neg_scaled = ctx.add(Expr::Neg(scaled_result));
                    let remaining = build_sum_without(ctx, terms, i, j);
                    let final_result = combine_with_term(ctx, remaining, neg_scaled);
                    let local_before = ctx.add(Expr::Add(term_i, term_j));
                    return Some(PairWithNegationPlan {
                        final_result,
                        local_before,
                        local_after: neg_scaled,
                        desc: format!("-k·[{}]", desc),
                    });
                }
            }
        }
    }

    None
}

#[derive(Debug, Clone)]
pub struct AtanRationalAdditionPlan {
    pub final_result: ExprId,
    pub local_before: ExprId,
    pub local_after: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseTrigUnaryRewriteKind {
    ArcsinNegative,
    ArctanNegative,
    ArccosNegative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseTrigUnaryRewrite {
    pub rewritten: ExprId,
    pub kind: InverseTrigUnaryRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseTrigReciprocalRewriteKind {
    ArcsecToArccos,
    ArccscToArcsin,
    ArccotToArctan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseTrigReciprocalRewrite {
    pub rewritten: ExprId,
    pub kind: InverseTrigReciprocalRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrincipalBranchInverseTrigKind {
    ArcsinSin,
    ArccosCos,
    ArctanTan,
    ArctanSinOverCos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrincipalBranchInverseTrigPlan {
    pub rewritten: ExprId,
    pub local_before: ExprId,
    pub local_after: ExprId,
    pub kind: PrincipalBranchInverseTrigKind,
    pub assumption_fn: &'static str,
}

/// Plan `atan(a) + atan(b) -> atan((a+b)/(1-ab))` for rational arguments,
/// valid only when `1-ab > 0` and `ab != 1`.
pub(crate) fn try_plan_atan_rational_add_pair_expr(
    ctx: &mut Context,
    terms: &[ExprId],
    i: usize,
    j: usize,
) -> Option<AtanRationalAdditionPlan> {
    let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
        (ctx.get(terms[i]), ctx.get(terms[j]))
    else {
        return None;
    };

    let (name_i, args_i) = (*name_i, args_i.clone());
    let (name_j, args_j) = (*name_j, args_j.clone());
    let is_i_atan = matches!(
        ctx.builtin_of(name_i),
        Some(BuiltinFn::Atan | BuiltinFn::Arctan)
    );
    let is_j_atan = matches!(
        ctx.builtin_of(name_j),
        Some(BuiltinFn::Atan | BuiltinFn::Arctan)
    );
    if !is_i_atan || !is_j_atan || args_i.len() != 1 || args_j.len() != 1 {
        return None;
    }

    let arg_i = args_i[0];
    let arg_j = args_j[0];
    let (Some(a), Some(b)) = (as_rational_const(ctx, arg_i), as_rational_const(ctx, arg_j)) else {
        return None;
    };

    let ab = &a * &b;
    let one = num_rational::BigRational::from_integer(1.into());
    if ab == one {
        return None;
    }

    let one_minus_ab = &one - &ab;
    if one_minus_ab <= num_rational::BigRational::from_integer(0.into()) {
        return None;
    }

    let a_plus_b = &a + &b;
    let result_val = &a_plus_b / &one_minus_ab;
    let result_num = ctx.add(Expr::Number(result_val));
    let result_atan = ctx.call_builtin(BuiltinFn::Arctan, vec![result_num]);

    let remaining = build_sum_without(ctx, terms, i, j);
    let final_result = combine_with_term(ctx, remaining, result_atan);
    let local_before = ctx.add(Expr::Add(terms[i], terms[j]));
    let desc = format!("arctan({}) + arctan({}) = arctan((a+b)/(1-ab))", a, b);

    Some(AtanRationalAdditionPlan {
        final_result,
        local_before,
        local_after: result_atan,
        desc,
    })
}

/// Plan `arcsin(x) + arccos(x) = π/2` across an additive pair with sign handling.
pub(crate) fn try_plan_inverse_trig_sum_pair_expr(
    ctx: &mut Context,
    terms: &[ExprId],
    i: usize,
    j: usize,
) -> Option<PairWithNegationPlan> {
    try_plan_pair_with_negation(
        ctx,
        terms[i],
        terms[j],
        terms,
        i,
        j,
        |ctx, expr_i, expr_j| {
            if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                (expr_i, expr_j)
            {
                if args_i.len() == 1 && args_j.len() == 1 {
                    let arg_i = args_i[0];
                    let arg_j = args_j[0];
                    let args_equal = arg_i == arg_j
                        || compare_expr(ctx, arg_i, arg_j) == std::cmp::Ordering::Equal;

                    if args_equal {
                        // `arcsin`/`arccos` are real only on `[-1, 1]`. A concrete
                        // argument provably OUTSIDE the unit interval makes BOTH
                        // terms undefined, so `arcsin(x) + arccos(x) = π/2` must NOT
                        // collapse `arccos(2) + arcsin(2)` (nor the
                        // `arcsec(1/2) + arccsc(1/2)` form it reduces to) to π/2.
                        if is_number_outside_unit_interval(ctx, arg_i) {
                            return None;
                        }
                        let is_i_arcsin = matches!(
                            ctx.builtin_of(*name_i),
                            Some(BuiltinFn::Arcsin | BuiltinFn::Asin)
                        );
                        let is_j_arcsin = matches!(
                            ctx.builtin_of(*name_j),
                            Some(BuiltinFn::Arcsin | BuiltinFn::Asin)
                        );
                        let is_i_arccos = matches!(
                            ctx.builtin_of(*name_i),
                            Some(BuiltinFn::Arccos | BuiltinFn::Acos)
                        );
                        let is_j_arccos = matches!(
                            ctx.builtin_of(*name_j),
                            Some(BuiltinFn::Arccos | BuiltinFn::Acos)
                        );

                        if (is_i_arcsin && is_j_arccos) || (is_i_arccos && is_j_arcsin) {
                            let pi = ctx.add(Expr::Constant(Constant::Pi));
                            let two = ctx.num(2);
                            let pi_half = ctx.add(Expr::Div(pi, two));
                            return Some((pi_half, "arcsin(x) + arccos(x) = π/2".to_string()));
                        }
                    }
                }
            }
            None
        },
    )
}

/// Plan `arctan(x) + arctan(1/x) = π/2` across an additive pair with sign handling.
pub(crate) fn try_plan_inverse_atan_reciprocal_pair_expr(
    ctx: &mut Context,
    terms: &[ExprId],
    i: usize,
    j: usize,
) -> Option<PairWithNegationPlan> {
    try_plan_pair_with_negation(
        ctx,
        terms[i],
        terms[j],
        terms,
        i,
        j,
        |ctx, expr_i, expr_j| {
            if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                (expr_i, expr_j)
            {
                let is_i_atan = matches!(
                    ctx.builtin_of(*name_i),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                );
                let is_j_atan = matches!(
                    ctx.builtin_of(*name_j),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                );
                if is_i_atan
                    && is_j_atan
                    && args_i.len() == 1
                    && args_j.len() == 1
                    && are_reciprocals(ctx, args_i[0], args_j[0])
                {
                    let pi = ctx.add(Expr::Constant(Constant::Pi));
                    let two = ctx.num(2);
                    let pi_half = ctx.add(Expr::Div(pi, two));
                    // arctan(x) + arctan(1/x) = (π/2)·sign(x), NOT π/2: the sum is
                    // -π/2 for x < 0 (both arctans negative). `x` and `1/x` share the
                    // same sign, so use whichever argument is not a fraction for a
                    // cleaner `sign(x)`. For a literal argument `sign(c)` folds back to
                    // ±1, recovering the unconditional ±π/2.
                    let sign_arg = if matches!(ctx.get(args_i[0]), Expr::Div(_, _))
                        && !matches!(ctx.get(args_j[0]), Expr::Div(_, _))
                    {
                        args_j[0]
                    } else {
                        args_i[0]
                    };
                    let sign_fn = ctx.call_builtin(BuiltinFn::Sign, vec![sign_arg]);
                    let result = crate::build::mul2_raw(ctx, pi_half, sign_fn);
                    return Some((
                        result,
                        "arctan(x) + arctan(1/x) = (π/2)·sign(x)".to_string(),
                    ));
                }
            }
            None
        },
    )
}

fn are_arctan_args_opposite(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    if is_negation(ctx, left, right) || are_denominators_opposite(ctx, left, right) {
        return true;
    }

    let left_terms = flatten_add_sub_chain(ctx, left);
    let right_terms = flatten_add_sub_chain(ctx, right);
    if left_terms.len() < 2 || left_terms.len() != right_terms.len() {
        return false;
    }

    let mut used_right = vec![false; right_terms.len()];
    'left: for left_term in left_terms {
        for (idx, right_term) in right_terms.iter().enumerate() {
            if !used_right[idx] && is_negation(ctx, left_term, *right_term) {
                used_right[idx] = true;
                continue 'left;
            }
        }
        return false;
    }
    true
}

/// Plan `arctan(u) + arctan(v) = 0` when `u` and `v` are structural opposites.
pub(crate) fn try_plan_inverse_atan_opposite_pair_expr(
    ctx: &mut Context,
    terms: &[ExprId],
    i: usize,
    j: usize,
) -> Option<PairWithNegationPlan> {
    try_plan_pair_with_negation(
        ctx,
        terms[i],
        terms[j],
        terms,
        i,
        j,
        |ctx, expr_i, expr_j| {
            if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                (expr_i, expr_j)
            {
                let is_i_atan = matches!(
                    ctx.builtin_of(*name_i),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                );
                let is_j_atan = matches!(
                    ctx.builtin_of(*name_j),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                );
                if is_i_atan
                    && is_j_atan
                    && args_i.len() == 1
                    && args_j.len() == 1
                    && are_arctan_args_opposite(ctx, args_i[0], args_j[0])
                {
                    let zero = ctx.num(0);
                    return Some((zero, "arctan(u) + arctan(-u) = 0".to_string()));
                }
            }
            None
        },
    )
}

/// Rewrite odd/even inverse-trig negative-argument identities.
pub fn try_rewrite_inverse_trig_negative_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InverseTrigUnaryRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let inner = extract_negated_inner(ctx, args[0])?;
    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin) => {
            let arcsin_inner = ctx.call_builtin(BuiltinFn::Arcsin, vec![inner]);
            Some(InverseTrigUnaryRewrite {
                rewritten: ctx.add(Expr::Neg(arcsin_inner)),
                kind: InverseTrigUnaryRewriteKind::ArcsinNegative,
            })
        }
        Some(BuiltinFn::Arctan) => {
            let arctan_inner = ctx.call_builtin(BuiltinFn::Arctan, vec![inner]);
            Some(InverseTrigUnaryRewrite {
                rewritten: ctx.add(Expr::Neg(arctan_inner)),
                kind: InverseTrigUnaryRewriteKind::ArctanNegative,
            })
        }
        Some(BuiltinFn::Arccos) => {
            let arccos_inner = ctx.call_builtin(BuiltinFn::Arccos, vec![inner]);
            let pi = ctx.add(Expr::Constant(Constant::Pi));
            Some(InverseTrigUnaryRewrite {
                rewritten: ctx.add(Expr::Sub(pi, arccos_inner)),
                kind: InverseTrigUnaryRewriteKind::ArccosNegative,
            })
        }
        _ => None,
    }
}

/// Rewrite `arcsec(x)` to `arccos(1/x)`.
pub fn try_rewrite_arcsec_to_arccos_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InverseTrigReciprocalRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Asec | BuiltinFn::Arcsec)
        )
    {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, arg));
    let rewritten = ctx.call_builtin(BuiltinFn::Arccos, vec![reciprocal]);
    Some(InverseTrigReciprocalRewrite {
        rewritten,
        kind: InverseTrigReciprocalRewriteKind::ArcsecToArccos,
    })
}

/// Scan an additive expression for `arcsin(x) + arccos(x)` identity pairs.
pub fn try_plan_inverse_trig_sum_add_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PairWithNegationPlan> {
    if !matches!(ctx.get(expr), Expr::Add(_, _)) {
        return None;
    }
    let terms = flatten_add_sub_chain(ctx, expr);
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            if let Some(plan) = try_plan_inverse_trig_sum_pair_expr(ctx, &terms, i, j) {
                return Some(plan);
            }
        }
    }
    None
}

/// Scan an additive expression for `arctan(x) + arctan(1/x)` pairs.
///
/// `parent_is_add` should be true when the current expression is a sub-sum of
/// another `Add`, in which case this planner is intentionally disabled.
pub fn try_plan_inverse_atan_reciprocal_add_expr(
    ctx: &mut Context,
    expr: ExprId,
    parent_is_add: bool,
) -> Option<PairWithNegationPlan> {
    if !matches!(ctx.get(expr), Expr::Add(_, _)) || parent_is_add {
        return None;
    }
    let terms = flatten_add_sub_chain(ctx, expr);
    if terms.len() < 2 {
        return None;
    }
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            if let Some(plan) = try_plan_inverse_atan_reciprocal_pair_expr(ctx, &terms, i, j) {
                return Some(plan);
            }
        }
    }
    None
}

/// Scan an additive expression for `arctan(u) + arctan(-u)` pairs.
///
/// `parent_is_add` should be true when the current expression is a sub-sum of
/// another `Add`, in which case this planner is intentionally disabled.
pub fn try_plan_inverse_atan_opposite_add_expr(
    ctx: &mut Context,
    expr: ExprId,
    parent_is_add: bool,
) -> Option<PairWithNegationPlan> {
    if !matches!(ctx.get(expr), Expr::Add(_, _)) || parent_is_add {
        return None;
    }
    let terms = flatten_add_sub_chain(ctx, expr);
    if terms.len() < 2 {
        return None;
    }
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            if let Some(plan) = try_plan_inverse_atan_opposite_pair_expr(ctx, &terms, i, j) {
                return Some(plan);
            }
        }
    }
    None
}

/// Scan an additive expression for rational `atan(a) + atan(b)` (Machin) pairs.
///
/// `parent_is_add` should be true when the current expression is a sub-sum of
/// another `Add`, in which case this planner is intentionally disabled.
pub fn try_plan_atan_rational_add_expr(
    ctx: &mut Context,
    expr: ExprId,
    parent_is_add: bool,
) -> Option<AtanRationalAdditionPlan> {
    if !matches!(ctx.get(expr), Expr::Add(_, _)) || parent_is_add {
        return None;
    }

    let terms = flatten_add_sub_chain(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    // Reciprocal pairs should be handled by the dedicated rule first.
    if has_reciprocal_atan_pair(ctx, &terms) {
        return None;
    }

    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            if let Some(plan) = try_plan_atan_rational_add_pair_expr(ctx, &terms, i, j) {
                return Some(plan);
            }
        }
    }
    None
}

/// Rewrite `arccsc(x)` to `arcsin(1/x)`.
pub fn try_rewrite_arccsc_to_arcsin_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InverseTrigReciprocalRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Acsc | BuiltinFn::Arccsc)
        )
    {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, arg));
    let rewritten = ctx.call_builtin(BuiltinFn::Arcsin, vec![reciprocal]);
    Some(InverseTrigReciprocalRewrite {
        rewritten,
        kind: InverseTrigReciprocalRewriteKind::ArccscToArcsin,
    })
}

/// Rewrite `arccot(x)` to `arctan(1/x)`.
pub fn try_rewrite_arccot_to_arctan_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InverseTrigReciprocalRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, arg));
    let rewritten = ctx.call_builtin(BuiltinFn::Arctan, vec![reciprocal]);
    Some(InverseTrigReciprocalRewrite {
        rewritten,
        kind: InverseTrigReciprocalRewriteKind::ArccotToArctan,
    })
}

/// Rational multiple `k` of π from `arg = k·π`, unwrapping `Neg` and numeric
/// `Div`/`Mul` at any position (`-π/2`, `3·π/4`, `π/4`). `None` for any non-numeric
/// shape. Local so the principal-branch gate can decide a `k·π` argument exactly.
fn signed_pi_multiple(ctx: &Context, arg: ExprId) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;
    match ctx.get(arg) {
        Expr::Constant(Constant::Pi) => Some(BigRational::one()),
        Expr::Neg(inner) => signed_pi_multiple(ctx, *inner).map(|k| -k),
        Expr::Div(num, den) => {
            let kn = signed_pi_multiple(ctx, *num)?;
            let d = as_rational_const(ctx, *den)?;
            if d == BigRational::from_integer(0.into()) {
                None
            } else {
                Some(kn / d)
            }
        }
        Expr::Mul(l, r) => {
            if let Some(c) = as_rational_const(ctx, *l) {
                signed_pi_multiple(ctx, *r).map(|k| k * c)
            } else if let Some(c) = as_rational_const(ctx, *r) {
                signed_pi_multiple(ctx, *l).map(|k| k * c)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// True when folding `arcfn(fn(u)) → u` is PROVABLY sound — `u` lies in the inverse
/// function's principal range, so `u` IS the principal value. Round-4 Cluster M:
/// a numeric / `k·π` argument OUTSIDE the range must NOT fold to the bare argument
/// (it would assert a provably-FALSE range membership and return a wrong value).
///
///   - `k·π` (exact via the coefficient): `arcsin·sin` `|k| ≤ 1/2`; `arccos·cos`
///     `0 ≤ k ≤ 1`; `arctan·tan` `|k| < 1/2`.
///   - a bare rational `v` (decided against tight rational bounds on `π/2` / `π`,
///     PROVABLY-IN only; an out-of-range or boundary literal yields `false`).
///   - any other (free symbol / symbolic expression): `true` — the membership is
///     genuinely unknown and the fold is an honest conditional assumption.
fn principal_fold_is_sound(ctx: &Context, u: ExprId, kind: PrincipalBranchInverseTrigKind) -> bool {
    use num_rational::BigRational;
    use PrincipalBranchInverseTrigKind::{ArccosCos, ArcsinSin, ArctanSinOverCos, ArctanTan};

    let half = BigRational::new(1.into(), 2.into());
    let one = BigRational::one();
    let zero = BigRational::from_integer(0.into());

    if let Some(k) = signed_pi_multiple(ctx, u) {
        return match kind {
            ArcsinSin => -&half <= k && k <= half,
            ArccosCos => zero <= k && k <= one,
            ArctanTan | ArctanSinOverCos => -&half < k && k < half,
        };
    }
    if let Some(v) = as_rational_const(ctx, u) {
        let l_half = BigRational::new(15707.into(), 10000.into()); // < π/2
        let l_pi = BigRational::new(31415.into(), 10000.into()); // < π
        return match kind {
            ArcsinSin | ArctanTan | ArctanSinOverCos => -&l_half <= v && v <= l_half,
            ArccosCos => zero <= v && v <= l_pi,
        };
    }
    // Symbolic argument: membership unknown — keep the assumption-gated fold.
    true
}

/// Plan principal-branch inverse-trig rewrites that require a range assumption.
pub fn try_plan_principal_branch_inverse_trig_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PrincipalBranchInverseTrigPlan> {
    let Expr::Function(outer_name, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }

    let inner = outer_args[0];
    let inner_fn_info = if let Expr::Function(fn_id, args) = ctx.get(inner) {
        Some((*fn_id, args.clone()))
    } else {
        None
    };

    if ctx.is_builtin(*outer_name, BuiltinFn::Arcsin) {
        if let Some((inner_name, inner_args)) = &inner_fn_info {
            if ctx.is_builtin(*inner_name, BuiltinFn::Sin)
                && inner_args.len() == 1
                && principal_fold_is_sound(
                    ctx,
                    inner_args[0],
                    PrincipalBranchInverseTrigKind::ArcsinSin,
                )
            {
                let u = inner_args[0];
                return Some(PrincipalBranchInverseTrigPlan {
                    rewritten: u,
                    local_before: expr,
                    local_after: u,
                    kind: PrincipalBranchInverseTrigKind::ArcsinSin,
                    assumption_fn: "arcsin",
                });
            }
        }
    }

    if ctx.is_builtin(*outer_name, BuiltinFn::Arccos) {
        if let Some((inner_name, inner_args)) = &inner_fn_info {
            if ctx.is_builtin(*inner_name, BuiltinFn::Cos)
                && inner_args.len() == 1
                && principal_fold_is_sound(
                    ctx,
                    inner_args[0],
                    PrincipalBranchInverseTrigKind::ArccosCos,
                )
            {
                let u = inner_args[0];
                return Some(PrincipalBranchInverseTrigPlan {
                    rewritten: u,
                    local_before: expr,
                    local_after: u,
                    kind: PrincipalBranchInverseTrigKind::ArccosCos,
                    assumption_fn: "arccos",
                });
            }
        }
    }

    if ctx.is_builtin(*outer_name, BuiltinFn::Arctan) {
        if let Some((inner_name, inner_args)) = &inner_fn_info {
            if ctx.is_builtin(*inner_name, BuiltinFn::Tan)
                && inner_args.len() == 1
                && principal_fold_is_sound(
                    ctx,
                    inner_args[0],
                    PrincipalBranchInverseTrigKind::ArctanTan,
                )
            {
                let u = inner_args[0];
                return Some(PrincipalBranchInverseTrigPlan {
                    rewritten: u,
                    local_before: expr,
                    local_after: u,
                    kind: PrincipalBranchInverseTrigKind::ArctanTan,
                    assumption_fn: "arctan",
                });
            }
        }

        if let Expr::Div(num, den) = ctx.get(inner) {
            let num_fn_info = if let Expr::Function(n, a) = ctx.get(*num) {
                Some((*n, a.clone()))
            } else {
                None
            };
            let den_fn_info = if let Expr::Function(n, a) = ctx.get(*den) {
                Some((*n, a.clone()))
            } else {
                None
            };

            if let (Some((n_name, n_args)), Some((d_name, d_args))) = (num_fn_info, den_fn_info) {
                if ctx.is_builtin(n_name, BuiltinFn::Sin)
                    && ctx.is_builtin(d_name, BuiltinFn::Cos)
                    && n_args.len() == 1
                    && d_args.len() == 1
                    && (n_args[0] == d_args[0]
                        || compare_expr(ctx, n_args[0], d_args[0]) == std::cmp::Ordering::Equal)
                    && principal_fold_is_sound(
                        ctx,
                        n_args[0],
                        PrincipalBranchInverseTrigKind::ArctanSinOverCos,
                    )
                {
                    let u = n_args[0];
                    return Some(PrincipalBranchInverseTrigPlan {
                        rewritten: u,
                        local_before: expr,
                        local_after: u,
                        kind: PrincipalBranchInverseTrigKind::ArctanSinOverCos,
                        assumption_fn: "arctan",
                    });
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        build_sum_without, combine_with_term, is_number_in_unit_interval,
        plan_inverse_trig_composition_with_mode_flags, try_plan_atan_rational_add_expr,
        try_plan_atan_rational_add_pair_expr, try_plan_inverse_atan_reciprocal_add_expr,
        try_plan_inverse_atan_reciprocal_pair_expr, try_plan_inverse_trig_composition_expr,
        try_plan_inverse_trig_sum_add_expr, try_plan_inverse_trig_sum_pair_expr,
        try_plan_pair_with_negation, try_plan_principal_branch_inverse_trig_expr,
        try_rewrite_arccot_to_arctan_expr, try_rewrite_arccsc_to_arcsin_expr,
        try_rewrite_arcsec_to_arccos_expr, try_rewrite_inverse_trig_negative_expr,
        InverseTrigCompositionKind, InverseTrigCompositionPlan,
    };
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn principal_branch_fold_gated_by_provable_range_membership() {
        // Round-4 Cluster M: arcfn(fn(u)) -> u only when `u` is PROVABLY in the
        // principal range. An out-of-range numeric / k·π argument must NOT fold to
        // the bare argument (which would assert a provably-false membership).
        let mut ctx = Context::new();
        let plans = |ctx: &mut Context, src: &str| -> bool {
            let e = parse(src, ctx).expect("parse");
            super::try_plan_principal_branch_inverse_trig_expr(ctx, e).is_some()
        };
        // In range -> folds. (Canonical builtin names: the eval pipeline
        // normalizes `asin`->`arcsin` before this planner; `parse` does not.)
        assert!(plans(&mut ctx, "arcsin(sin(1))"));
        assert!(plans(&mut ctx, "arccos(cos(2))"));
        assert!(plans(&mut ctx, "arccos(cos(3))")); // 3 < π
        assert!(plans(&mut ctx, "arcsin(sin(pi/4))"));
        assert!(plans(&mut ctx, "arctan(tan(pi/6))"));
        assert!(plans(&mut ctx, "arcsin(sin(-1))"));
        // Free symbol -> assumption-gated fold is kept.
        assert!(plans(&mut ctx, "arcsin(sin(x))"));
        // Out of range -> NOT folded (provably-false membership).
        assert!(!plans(&mut ctx, "arcsin(sin(3))")); // 3 > π/2
        assert!(!plans(&mut ctx, "arccos(cos(10))")); // 10 > π
        assert!(!plans(&mut ctx, "arctan(tan(3*pi/4))")); // 3/4 > 1/2
        assert!(!plans(&mut ctx, "arcsin(sin(2))")); // 2 > π/2
        assert!(!plans(&mut ctx, "arccos(cos(-1))")); // < 0, outside [0, π]
    }

    #[test]
    fn strict_requires_interval_proof() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            false,
            true,
        );
        assert_eq!(out, None);
    }

    #[test]
    fn inverse_trig_composition_plan_respects_mode_flags() {
        let mut ctx = Context::new();
        let expr = parse("sin(arcsin(x))", &mut ctx).expect("parse");
        let strict_blocked = try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true);
        assert!(strict_blocked.is_none());

        let assume =
            try_plan_inverse_trig_composition_expr(&mut ctx, expr, true, false).expect("plan");
        assert_eq!(assume.rewritten, parse("x", &mut ctx).expect("x"));
        assert!(assume.assume_defined_expr.is_some());
    }

    #[test]
    fn strict_accepts_proven_interval() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::CosArccos,
            true,
            false,
            true,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                kind: InverseTrigCompositionKind::CosArccos,
                assume_defined: false,
            })
        );
    }

    #[test]
    fn assume_always_applies_with_assumption() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            true,
            false,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                kind: InverseTrigCompositionKind::SinArcsin,
                assume_defined: true,
            })
        );
    }

    #[test]
    fn generic_applies_without_assumption() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            false,
            false,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                kind: InverseTrigCompositionKind::SinArcsin,
                assume_defined: false,
            })
        );
    }

    #[test]
    fn inverse_trig_composition_plan_detects_arctan_right_triangle_projections() {
        for (source_text, target_text, expected_kind) in [
            (
                "sin(arctan(x))",
                "x/sqrt(x^2 + 1)",
                InverseTrigCompositionKind::SinArctan,
            ),
            (
                "cos(arctan(x))",
                "1/sqrt(x^2 + 1)",
                InverseTrigCompositionKind::CosArctan,
            ),
        ] {
            let mut ctx = Context::new();
            let expr = parse(source_text, &mut ctx).expect("parse source");
            let expected = parse(target_text, &mut ctx).expect("parse target");
            let plan =
                try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");

            assert_eq!(plan.kind, expected_kind);
            assert_eq!(
                cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
                std::cmp::Ordering::Equal
            );
        }
    }

    #[test]
    fn inverse_trig_composition_plan_detects_arcsin_arccos_complement_projections() {
        for (source_text, target_text, expected_kind) in [
            (
                "cos(arcsin(x))",
                "sqrt(1 - x^2)",
                InverseTrigCompositionKind::CosArcsin,
            ),
            (
                "sin(arccos(x))",
                "sqrt(1 - x^2)",
                InverseTrigCompositionKind::SinArccos,
            ),
        ] {
            let mut ctx = Context::new();
            let expr = parse(source_text, &mut ctx).expect("parse source");
            let expected = parse(target_text, &mut ctx).expect("parse target");
            let plan =
                try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");

            assert_eq!(plan.kind, expected_kind);
            assert_eq!(
                cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
                std::cmp::Ordering::Equal
            );
        }
    }

    #[test]
    fn inverse_trig_composition_plan_detects_tan_arcsin_projection() {
        let mut ctx = Context::new();
        let expr = parse("tan(arcsin(x))", &mut ctx).expect("parse source");
        let expected = parse("x/sqrt(1 - x^2)", &mut ctx).expect("parse target");
        let plan =
            try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");

        assert_eq!(plan.kind, InverseTrigCompositionKind::TanArcsin);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn assume_priority_over_strict() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::CosArccos,
            false,
            true,
            true,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                kind: InverseTrigCompositionKind::CosArccos,
                assume_defined: true,
            })
        );
    }

    #[test]
    fn unit_interval_detection_accepts_and_rejects_expected_values() {
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let minus_two = ctx.num(-2);
        let x = ctx.var("x");

        assert!(is_number_in_unit_interval(&ctx, half));
        assert!(!is_number_in_unit_interval(&ctx, minus_two));
        assert!(!is_number_in_unit_interval(&ctx, x));

        // Explicit bound checks
        let minus_one = ctx.num(-1);
        let plus_one = ctx.num(1);
        let three = ctx.num(3);
        let two = ctx.num(2);
        let over_one = ctx.add(Expr::Div(three, two));
        assert!(is_number_in_unit_interval(&ctx, minus_one));
        assert!(is_number_in_unit_interval(&ctx, plus_one));
        assert!(!is_number_in_unit_interval(&ctx, over_one));
    }

    #[test]
    fn sum_helpers_build_remaining_sum_and_append_term() {
        let mut ctx = Context::new();
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");
        let c = parse("c", &mut ctx).expect("c");
        let d = parse("d", &mut ctx).expect("d");
        let terms = vec![a, b, c, d];

        let remaining = build_sum_without(&mut ctx, &terms, 1, 3).expect("remaining");
        let appended = combine_with_term(&mut ctx, Some(remaining), b);

        let expected_remaining = parse("a+c", &mut ctx).expect("a+c");
        let expected_appended = parse("(a+c)+b", &mut ctx).expect("(a+c)+b");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, remaining, expected_remaining),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, appended, expected_appended),
            std::cmp::Ordering::Equal
        );
        assert_eq!(combine_with_term(&mut ctx, None, c), c);
    }

    #[test]
    fn pair_with_negation_plans_positive_match() {
        let mut ctx = Context::new();
        let a = parse("f(x)", &mut ctx).expect("f");
        let b = parse("g(x)", &mut ctx).expect("g");
        let terms = vec![a, b];

        let plan = try_plan_pair_with_negation(&mut ctx, a, b, &terms, 0, 1, |ctx, _i, _j| {
            let h = parse("h(x)", ctx).ok()?;
            Some((h, "pair".to_string()))
        });

        let plan = plan.expect("plan");
        assert_eq!(plan.desc, "pair");
        let expected = parse("h(x)", &mut ctx).expect("h");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.final_result, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn atan_rational_addition_plan_applies() {
        let mut ctx = Context::new();
        let t1 = parse("atan(1/2)", &mut ctx).expect("atan1");
        let t2 = parse("atan(1/3)", &mut ctx).expect("atan2");
        let terms = vec![t1, t2];
        let plan = try_plan_atan_rational_add_pair_expr(&mut ctx, &terms, 0, 1).expect("plan");
        assert!(plan.desc.contains("arctan(1/2) + arctan(1/3)"));
        let shown = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: plan.local_after
            }
        );
        assert!(shown.contains("arctan(1)"));
    }

    #[test]
    fn inverse_trig_sum_pair_plan_applies() {
        let mut ctx = Context::new();
        let t1 = parse("asin(x)", &mut ctx).expect("asin");
        let t2 = parse("acos(x)", &mut ctx).expect("acos");
        let terms = vec![t1, t2];
        let plan = try_plan_inverse_trig_sum_pair_expr(&mut ctx, &terms, 0, 1).expect("plan");
        let expected = parse("pi/2", &mut ctx).expect("pi/2");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.local_after, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn inverse_atan_reciprocal_pair_plan_applies() {
        let mut ctx = Context::new();
        let t1 = parse("atan(2)", &mut ctx).expect("atan");
        let t2 = parse("atan(1/2)", &mut ctx).expect("atan");
        let terms = vec![t1, t2];
        let plan =
            try_plan_inverse_atan_reciprocal_pair_expr(&mut ctx, &terms, 0, 1).expect("plan");
        // arctan(x) + arctan(1/x) = (π/2)·sign(x), not π/2 (it is -π/2 for x < 0).
        // The raw plan carries sign(2); the engine later folds sign(2) -> 1 -> π/2.
        let rendered = cas_formatter::DisplayExpr {
            context: &ctx,
            id: plan.local_after,
        }
        .to_string();
        assert!(
            rendered.contains("sign(2)") && rendered.contains("pi"),
            "expected (pi/2)*sign(2), got {rendered}"
        );
    }

    #[test]
    fn inverse_trig_unary_rewrites_apply() {
        let mut ctx = Context::new();
        let neg_asin = parse("arcsin(-x)", &mut ctx).expect("parse");
        let arcsec = parse("arcsec(x)", &mut ctx).expect("parse");
        let arccsc = parse("arccsc(x)", &mut ctx).expect("parse");
        let arccot = parse("arccot(x)", &mut ctx).expect("parse");
        assert!(try_rewrite_inverse_trig_negative_expr(&mut ctx, neg_asin).is_some());
        assert!(try_rewrite_arcsec_to_arccos_expr(&mut ctx, arcsec).is_some());
        assert!(try_rewrite_arccsc_to_arcsin_expr(&mut ctx, arccsc).is_some());
        assert!(try_rewrite_arccot_to_arctan_expr(&mut ctx, arccot).is_some());
    }

    #[test]
    fn principal_branch_plan_detects_tan_quotient() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sin(u)/cos(u))", &mut ctx).expect("parse");
        let plan = try_plan_principal_branch_inverse_trig_expr(&mut ctx, expr).expect("plan");
        let u = parse("u", &mut ctx).expect("u");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, u),
            std::cmp::Ordering::Equal
        );
        assert_eq!(plan.assumption_fn, "arctan");
    }

    #[test]
    fn inverse_trig_sum_add_plan_applies() {
        let mut ctx = Context::new();
        let expr = parse("asin(x) + acos(x)", &mut ctx).expect("parse");
        let plan = try_plan_inverse_trig_sum_add_expr(&mut ctx, expr).expect("plan");
        let expected = parse("pi/2", &mut ctx).expect("pi/2");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.local_after, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn inverse_atan_reciprocal_add_honors_parent_guard() {
        let mut ctx = Context::new();
        let expr = parse("atan(2) + atan(1/2)", &mut ctx).expect("parse");
        assert!(try_plan_inverse_atan_reciprocal_add_expr(&mut ctx, expr, false).is_some());
        assert!(try_plan_inverse_atan_reciprocal_add_expr(&mut ctx, expr, true).is_none());
    }

    #[test]
    fn atan_rational_add_honors_reciprocal_guard() {
        let mut ctx = Context::new();
        let reciprocal_expr = parse("atan(2) + atan(1/2)", &mut ctx).expect("parse");
        assert!(try_plan_atan_rational_add_expr(&mut ctx, reciprocal_expr, false).is_none());

        let machin_expr = parse("atan(1/2) + atan(1/3)", &mut ctx).expect("parse");
        assert!(try_plan_atan_rational_add_expr(&mut ctx, machin_expr, false).is_some());
    }

    #[test]
    fn inverse_trig_composition_plan_detects_safe_arcsin_sin_arctan_expansion() {
        let mut ctx = Context::new();
        let expr = parse("asin(x/sqrt(x^2 + 1))", &mut ctx).expect("parse");
        let plan =
            try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");
        let expected = parse("arctan(x)", &mut ctx).expect("expected");
        assert_eq!(plan.kind, InverseTrigCompositionKind::ArcsinSinArctan);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn inverse_trig_composition_plan_detects_normalized_safe_arcsin_sin_arctan_expansion() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x*sqrt(x^2 + 1)/(x^2 + 1))", &mut ctx).expect("parse");
        let plan =
            try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");
        let expected = parse("arctan(x)", &mut ctx).expect("expected");
        assert_eq!(plan.kind, InverseTrigCompositionKind::ArcsinSinArctan);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn inverse_trig_composition_plan_detects_canonical_reciprocal_root_projection() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x*(x^2 + 1)^(-1/2))", &mut ctx).expect("parse");
        let plan =
            try_plan_inverse_trig_composition_expr(&mut ctx, expr, false, true).expect("plan");
        let expected = parse("arctan(x)", &mut ctx).expect("expected");
        assert_eq!(plan.kind, InverseTrigCompositionKind::ArcsinSinArctan);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, plan.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }
}
