//! Shared real-domain predicates for calculus-facing policies.

use crate::{polynomial::Polynomial, tri_proof::TriProof};
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundedInverseRealDomainRejection {
    SourceDomainEmpty,
    DerivativeDomainEmpty,
}

/// Returns true for constants whose real absolute value is known to exceed 1.
pub fn known_constant_abs_exceeds_one(ctx: &Context, expr: ExprId) -> bool {
    if crate::numeric_eval::as_rational_const(ctx, expr)
        .is_some_and(|value| value.abs() > BigRational::one())
    {
        return true;
    }

    match ctx.get(expr) {
        Expr::Constant(Constant::Pi | Constant::E | Constant::Phi) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => known_constant_abs_exceeds_one(ctx, *inner),
        _ => false,
    }
}

/// Returns true for constants whose real value is known to be positive and > 1.
pub(crate) fn known_positive_constant_exceeds_one(ctx: &Context, expr: ExprId) -> bool {
    if crate::numeric_eval::as_rational_const(ctx, expr)
        .is_some_and(|value| value > BigRational::one())
    {
        return true;
    }

    match ctx.get(expr) {
        Expr::Constant(Constant::Pi | Constant::E | Constant::Phi) => true,
        Expr::Hold(inner) => known_positive_constant_exceeds_one(ctx, *inner),
        _ => false,
    }
}

/// Returns true when `expr > 0` has no real-domain solution provable within
/// `proof_depth`, by proving sign-preserved `-expr >= 0`.
pub fn positive_condition_is_impossible_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    if nonpositive_scaled_nonnegative_factor_is_proven(ctx, expr, proof_depth) {
        return true;
    }

    let negated = ctx.add(Expr::Neg(expr));
    nonnegative_condition_is_proven_over_reals(ctx, negated, proof_depth)
}

/// Returns true when `expr > 0` is provable over the real domain within
/// `proof_depth`. This is intentionally conservative: it delegates to the
/// sign prover and adds only the named-constant offset pattern already used by
/// bounded inverse domain policy.
pub fn positive_condition_is_proven_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    if proof_depth == 0 {
        return false;
    }
    if crate::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value > Zero::zero()) {
        return true;
    }
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }

    let normalized = crate::expr_normalization::normalize_condition_expr_preserve_sign(ctx, expr);
    if crate::prove_sign::prove_positive_depth_with(
        ctx,
        normalized,
        proof_depth,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
    {
        return true;
    }

    if positive_condition_is_proven_by_offset_structure(ctx, expr, proof_depth - 1) {
        return true;
    }
    if positive_condition_is_proven_by_positive_term_plus_nonnegative_remainder(
        ctx,
        expr,
        proof_depth - 1,
    ) {
        return true;
    }
    if positive_condition_is_proven_by_gt_one_offset_sum(ctx, expr, proof_depth - 1) {
        return true;
    }

    let one = ctx.num(1);
    let shifted = ctx.add(Expr::Add(expr, one));
    expr_is_known_gt_one_by_positive_constant_offset(ctx, shifted, proof_depth)
}

fn positive_condition_is_proven_by_offset_structure(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Hold(inner) => positive_condition_is_proven_over_reals(ctx, inner, proof_depth),
        Expr::Add(left, right) => {
            (positive_condition_is_proven_over_reals(ctx, left, proof_depth)
                && nonnegative_condition_is_proven_over_reals(ctx, right, proof_depth))
                || (positive_condition_is_proven_over_reals(ctx, right, proof_depth)
                    && nonnegative_condition_is_proven_over_reals(ctx, left, proof_depth))
        }
        Expr::Sub(left, right) => {
            crate::numeric_eval::as_rational_const(ctx, right).is_some_and(|value| value.is_one())
                && expr_is_known_gt_one_by_positive_constant_offset(ctx, left, proof_depth)
        }
        _ => false,
    }
}

fn positive_condition_is_proven_by_positive_term_plus_nonnegative_remainder(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let mut positive_terms = Vec::new();
    let mut negative_terms = Vec::new();
    collect_signed_additive_terms(ctx, expr, true, &mut positive_terms, &mut negative_terms);
    if !negative_terms.is_empty() || positive_terms.len() < 2 {
        return false;
    }

    for positive_index in 0..positive_terms.len() {
        let positive_term = positive_terms[positive_index];
        if !positive_condition_is_proven_over_reals(ctx, positive_term, proof_depth) {
            continue;
        }

        let remainder_terms: Vec<_> = positive_terms
            .iter()
            .enumerate()
            .filter_map(|(idx, term)| (idx != positive_index).then_some(*term))
            .collect();
        let remainder = crate::expr_nary::build_balanced_add(ctx, &remainder_terms);
        if nonnegative_condition_is_proven_over_reals(ctx, remainder, proof_depth) {
            return true;
        }
    }

    false
}

fn positive_condition_is_proven_by_gt_one_offset_sum(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let mut positive_terms = Vec::new();
    let mut negative_terms = Vec::new();
    collect_signed_additive_terms(ctx, expr, true, &mut positive_terms, &mut negative_terms);

    let mut negative_one_count = 0usize;
    for term in negative_terms {
        if crate::numeric_eval::as_rational_const(ctx, term).is_some_and(|value| value.is_one()) {
            negative_one_count += 1;
        } else {
            return false;
        }
    }

    let mut nonnegative_or_gt_one_terms = Vec::new();
    for term in positive_terms {
        if crate::numeric_eval::as_rational_const(ctx, term)
            .is_some_and(|value| value == -BigRational::one())
        {
            negative_one_count += 1;
        } else {
            nonnegative_or_gt_one_terms.push(term);
        }
    }

    if negative_one_count != 1 {
        return false;
    }

    let mut has_gt_one_term = false;
    for term in nonnegative_or_gt_one_terms {
        if known_positive_constant_exceeds_one(ctx, term) {
            has_gt_one_term = true;
        } else if !nonnegative_condition_is_proven_over_reals(ctx, term, proof_depth) {
            return false;
        }
    }

    has_gt_one_term
}

fn collect_signed_additive_terms(
    ctx: &Context,
    expr: ExprId,
    positive_sign: bool,
    positive_terms: &mut Vec<ExprId>,
    negative_terms: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_additive_terms(
                ctx,
                *left,
                positive_sign,
                positive_terms,
                negative_terms,
            );
            collect_signed_additive_terms(
                ctx,
                *right,
                positive_sign,
                positive_terms,
                negative_terms,
            );
        }
        Expr::Sub(left, right) => {
            collect_signed_additive_terms(
                ctx,
                *left,
                positive_sign,
                positive_terms,
                negative_terms,
            );
            collect_signed_additive_terms(
                ctx,
                *right,
                !positive_sign,
                positive_terms,
                negative_terms,
            );
        }
        Expr::Neg(inner) => collect_signed_additive_terms(
            ctx,
            *inner,
            !positive_sign,
            positive_terms,
            negative_terms,
        ),
        _ if positive_sign => positive_terms.push(expr),
        _ => negative_terms.push(expr),
    }
}

fn nonpositive_scaled_nonnegative_factor_is_proven(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    if crate::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value <= Zero::zero())
    {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            factor_is_known_nonnegative_for_positive_impossibility(ctx, inner, proof_depth)
        }
        Expr::Mul(left, right) => {
            let left_const = crate::numeric_eval::as_rational_const(ctx, left);
            if let Some(scale) = left_const {
                return scale.is_zero()
                    || (scale.is_negative()
                        && factor_is_known_nonnegative_for_positive_impossibility(
                            ctx,
                            right,
                            proof_depth,
                        ));
            }

            let right_const = crate::numeric_eval::as_rational_const(ctx, right);
            if let Some(scale) = right_const {
                return scale.is_zero()
                    || (scale.is_negative()
                        && factor_is_known_nonnegative_for_positive_impossibility(
                            ctx,
                            left,
                            proof_depth,
                        ));
            }

            false
        }
        _ => false,
    }
}

fn factor_is_known_nonnegative_for_positive_impossibility(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs | BuiltinFn::Sqrt))
    ) || nonnegative_condition_is_proven_over_reals(ctx, expr, proof_depth)
}

/// Returns true when `base` cannot be a valid real logarithm base because
/// `base > 0` is impossible or because the base is the constant `1`.
pub fn log_base_is_invalid_over_reals(ctx: &mut Context, base: ExprId, proof_depth: usize) -> bool {
    positive_condition_is_impossible_over_reals(ctx, base, proof_depth)
        || crate::numeric_eval::as_rational_const(ctx, base).is_some_and(|value| value.is_one())
}

/// Returns true when a logarithm call has no real-domain value for any input
/// assignment provable within `proof_depth`.
pub(crate) fn logarithm_real_domain_is_empty_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    match builtin {
        Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            positive_condition_is_impossible_over_reals(ctx, args[0], proof_depth)
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            log_base_is_invalid_over_reals(ctx, args[0], proof_depth)
                || positive_condition_is_impossible_over_reals(ctx, args[1], proof_depth)
        }
        _ => false,
    }
}

/// Returns true when a bounded inverse function call has no real-domain value
/// for any input assignment provable within `proof_depth`.
pub(crate) fn bounded_inverse_real_domain_is_empty_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    bounded_inverse_source_domain_is_empty_over_reals(ctx, builtin, args, proof_depth)
}

/// Classifies why a bounded inverse function call has no usable real-domain
/// derivative points.
pub fn bounded_inverse_real_domain_rejection_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> Option<BoundedInverseRealDomainRejection> {
    if bounded_inverse_source_domain_is_empty_over_reals(ctx, builtin, args, proof_depth) {
        return Some(BoundedInverseRealDomainRejection::SourceDomainEmpty);
    }
    if bounded_inverse_derivative_real_domain_is_empty_over_reals(ctx, builtin, args, proof_depth) {
        return Some(BoundedInverseRealDomainRejection::DerivativeDomainEmpty);
    }
    None
}

fn bounded_inverse_source_domain_is_empty_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    let [arg] = args else {
        return false;
    };

    match builtin {
        Some(BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos) => {
            if closed_inverse_trig_constant_source_arg_is_defined(ctx, *arg) {
                return false;
            }
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(*arg, two));
            let one = ctx.num(1);
            let interval_condition = ctx.add(Expr::Sub(one, arg_squared));
            nonnegative_condition_is_impossible_over_reals(ctx, interval_condition, proof_depth)
                || known_constant_abs_exceeds_one(ctx, *arg)
                || bounded_inverse_arg_is_proven_outside_closed_unit_interval(
                    ctx,
                    *arg,
                    proof_depth,
                )
        }
        Some(BuiltinFn::Atanh) => {
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(*arg, two));
            let one = ctx.num(1);
            let open_interval_condition = ctx.add(Expr::Sub(one, arg_squared));
            positive_condition_is_impossible_over_reals(ctx, open_interval_condition, proof_depth)
                || known_constant_abs_exceeds_one(ctx, *arg)
                || bounded_inverse_arg_is_proven_outside_closed_unit_interval(
                    ctx,
                    *arg,
                    proof_depth,
                )
        }
        Some(BuiltinFn::Acosh) => {
            let one = ctx.num(1);
            let lower_bound_condition = ctx.add(Expr::Sub(*arg, one));
            nonnegative_condition_is_impossible_over_reals(ctx, lower_bound_condition, proof_depth)
        }
        _ => false,
    }
}

fn bounded_inverse_arg_is_proven_outside_closed_unit_interval(
    ctx: &mut Context,
    arg: ExprId,
    proof_depth: usize,
) -> bool {
    expr_is_known_gt_one_by_positive_constant_offset(ctx, arg, proof_depth)
        || expr_is_known_lt_minus_one_by_negative_constant_offset(ctx, arg, proof_depth)
}

fn expr_is_known_gt_one_by_positive_constant_offset(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Hold(inner) => {
            expr_is_known_gt_one_by_positive_constant_offset(ctx, inner, proof_depth)
        }
        Expr::Add(left, right) => {
            (known_positive_constant_exceeds_one(ctx, left)
                && nonnegative_condition_is_proven_over_reals(ctx, right, proof_depth))
                || (known_positive_constant_exceeds_one(ctx, right)
                    && nonnegative_condition_is_proven_over_reals(ctx, left, proof_depth))
        }
        _ => false,
    }
}

fn expr_is_known_lt_minus_one_by_negative_constant_offset(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Hold(inner) => {
            expr_is_known_lt_minus_one_by_negative_constant_offset(ctx, inner, proof_depth)
        }
        Expr::Neg(inner) => {
            expr_is_known_gt_one_by_positive_constant_offset(ctx, inner, proof_depth)
        }
        Expr::Add(left, right) => {
            let neg_right = ctx.add(Expr::Neg(right));
            let left_lt_minus_one =
                expr_is_known_lt_minus_one_by_negative_constant_offset(ctx, left, proof_depth)
                    && nonnegative_condition_is_proven_over_reals(ctx, neg_right, proof_depth);
            let neg_left = ctx.add(Expr::Neg(left));
            let right_lt_minus_one =
                expr_is_known_lt_minus_one_by_negative_constant_offset(ctx, right, proof_depth)
                    && nonnegative_condition_is_proven_over_reals(ctx, neg_left, proof_depth);

            left_lt_minus_one || right_lt_minus_one
        }
        _ => false,
    }
}

/// Returns true when a bounded inverse function call has no real-domain points
/// where the derivative rule has a finite real value, provable within
/// `proof_depth`.
pub(crate) fn bounded_inverse_derivative_real_domain_is_empty_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    let [arg] = args else {
        return false;
    };

    match builtin {
        Some(BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos) => {
            if closed_inverse_trig_constant_source_arg_is_defined(ctx, *arg) {
                return false;
            }
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(*arg, two));
            let one = ctx.num(1);
            let derivative_condition = ctx.add(Expr::Sub(one, arg_squared));
            positive_condition_is_impossible_over_reals(ctx, derivative_condition, proof_depth)
                || closed_inverse_trig_derivative_domain_is_empty_over_reals(
                    ctx,
                    builtin,
                    args,
                    proof_depth,
                )
        }
        Some(BuiltinFn::Atanh) => {
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(*arg, two));
            let one = ctx.num(1);
            let derivative_condition = ctx.add(Expr::Sub(one, arg_squared));
            positive_condition_is_impossible_over_reals(ctx, derivative_condition, proof_depth)
        }
        Some(BuiltinFn::Acosh) => {
            let one = ctx.num(1);
            let derivative_condition = ctx.add(Expr::Sub(*arg, one));
            positive_condition_is_impossible_over_reals(ctx, derivative_condition, proof_depth)
        }
        _ => false,
    }
}

/// Returns true when a closed inverse-trig call has no real-domain points where
/// the derivative rule has a finite real value. Source-empty checks should run
/// before using this as a rejection reason.
pub(crate) fn closed_inverse_trig_derivative_domain_is_empty_over_reals(
    ctx: &Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    if !matches!(
        builtin,
        Some(BuiltinFn::Asin | BuiltinFn::Acos | BuiltinFn::Arcsin | BuiltinFn::Arccos)
    ) {
        return false;
    }

    let [arg] = args else {
        return false;
    };
    if closed_inverse_trig_constant_source_arg_is_defined(ctx, *arg) {
        return false;
    }

    let mut scratch = ctx.clone();
    let squared_or_radicand =
        if let Some(radicand) = crate::root_forms::extract_square_root_base(&scratch, *arg) {
            radicand
        } else {
            let two = scratch.num(2);
            scratch.add(Expr::Pow(*arg, two))
        };
    let one = scratch.num(1);
    let open_interval_gap = scratch.add(Expr::Sub(one, squared_or_radicand));

    if positive_condition_is_impossible_over_reals(&mut scratch, open_interval_gap, proof_depth) {
        return true;
    }

    closed_inverse_trig_boundary_only_polynomial_arg(&scratch, *arg)
}

fn closed_inverse_trig_constant_source_arg_is_defined(ctx: &Context, arg: ExprId) -> bool {
    if crate::numeric_eval::as_rational_const(ctx, arg)
        .is_some_and(|value| value.abs() <= BigRational::one())
    {
        return true;
    }

    match ctx.get(arg) {
        Expr::Neg(inner) | Expr::Hold(inner) => {
            closed_inverse_trig_constant_source_arg_is_defined(ctx, *inner)
        }
        _ => false,
    }
}

fn closed_inverse_trig_boundary_only_polynomial_arg(ctx: &Context, arg: ExprId) -> bool {
    let vars = cas_ast::collect_variables(ctx, arg);
    if vars.len() != 1 {
        return false;
    }

    let Some(var) = vars.iter().next() else {
        return false;
    };
    let Ok(poly) = Polynomial::from_expr(ctx, arg, var) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a.is_zero() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let vertex_value = c - (b.clone() * b) / (four * a.clone());

    (a.is_positive() && vertex_value.is_one())
        || (a.is_negative() && vertex_value == -BigRational::one())
}

/// Returns true when a variable-independent calculus expression has a statically
/// empty real domain. Unknown symbolic domains are deliberately preserved.
pub(crate) fn real_domain_is_empty_for_static_expr(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
) -> bool {
    real_domain_issue_over_reals(ctx, expr, proof_depth, scan_depth, false)
}

/// Returns true when `expr` contains a statically empty real-domain subexpression
/// or a nonfinite/undefined constant. This is the shared calculus guard for
/// commands that must reject nonfinite operands instead of treating them as
/// valid constants.
pub(crate) fn real_domain_is_empty_or_nonfinite_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
) -> bool {
    real_domain_issue_over_reals(ctx, expr, proof_depth, scan_depth, true)
}

/// Returns true for top-level nonfinite constants, allowing transparent wrappers
/// such as unary negation and hold.
pub fn nonfinite_or_undefined_constant(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => nonfinite_or_undefined_constant(ctx, *inner),
        _ => false,
    }
}

fn real_domain_issue_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
    include_nonfinite: bool,
) -> bool {
    if include_nonfinite && nonfinite_or_undefined_constant(ctx, expr) {
        return true;
    }
    if scan_depth == 0 {
        return false;
    }

    match ctx.get(expr).clone() {
        Expr::Constant(Constant::Undefined) => true,
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            real_domain_issue_over_reals(ctx, left, proof_depth, scan_depth - 1, include_nonfinite)
                || real_domain_issue_over_reals(
                    ctx,
                    right,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Div(num, den) => {
            crate::numeric_eval::as_rational_const(ctx, den).is_some_and(|value| value.is_zero())
                || real_domain_issue_over_reals(
                    ctx,
                    num,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
                || real_domain_issue_over_reals(
                    ctx,
                    den,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Pow(base, exp) => {
            if rational_exponent_requires_nonnegative_base(ctx, exp)
                && nonnegative_condition_is_impossible_over_reals(ctx, base, proof_depth)
            {
                return true;
            }
            real_domain_issue_over_reals(ctx, base, proof_depth, scan_depth - 1, include_nonfinite)
                || real_domain_issue_over_reals(
                    ctx,
                    exp,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            real_domain_issue_over_reals(ctx, inner, proof_depth, scan_depth - 1, include_nonfinite)
        }
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id);
            if logarithm_real_domain_is_empty_over_reals(ctx, builtin, &args, proof_depth) {
                return true;
            }
            if bounded_inverse_real_domain_is_empty_over_reals(ctx, builtin, &args, proof_depth) {
                return true;
            }
            if matches!(builtin, Some(BuiltinFn::Sqrt))
                && args.len() == 1
                && nonnegative_condition_is_impossible_over_reals(ctx, args[0], proof_depth)
            {
                return true;
            }

            args.into_iter().any(|arg| {
                real_domain_issue_over_reals(
                    ctx,
                    arg,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
            })
        }
        _ => false,
    }
}

fn rational_exponent_requires_nonnegative_base(ctx: &Context, exp: ExprId) -> bool {
    crate::numeric_eval::as_rational_const(ctx, exp)
        .is_some_and(|value| !value.denom().is_one() && value.denom().is_even())
}

/// Returns true when `expr >= 0` has no real-domain solution provable within
/// `proof_depth`, by proving sign-preserved `-expr > 0`.
pub fn nonnegative_condition_is_impossible_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let negated = ctx.add(Expr::Neg(expr));
    let normalized =
        crate::expr_normalization::normalize_condition_expr_preserve_sign(ctx, negated);
    crate::prove_sign::prove_positive_depth_with(
        ctx,
        normalized,
        proof_depth,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

/// Returns true when `expr >= 0` is provable over the real domain within
/// `proof_depth` after sign-preserving condition normalization.
pub fn nonnegative_condition_is_proven_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let normalized = crate::expr_normalization::normalize_condition_expr_preserve_sign(ctx, expr);
    crate::prove_sign::prove_nonnegative_depth_with(
        ctx,
        normalized,
        proof_depth,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_impossible_positive_condition_for_nonpositive_quadratics() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let neg_x_squared = ctx.add(Expr::Neg(x_squared));
        let one = ctx.num(1);
        let nonpositive = ctx.add(Expr::Sub(neg_x_squared, one));

        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx,
            nonpositive,
            12
        ));

        let neg_square = ctx.add(Expr::Neg(x_squared));
        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx, neg_square, 12
        ));
    }

    #[test]
    fn detects_impossible_positive_condition_for_nonpositive_scaled_nonnegative_factor() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Sub(x, one));
        let abs_shifted = ctx.call_builtin(BuiltinFn::Abs, vec![shifted]);
        let neg_three = ctx.num(-3);
        let negative_scaled_abs = ctx.add(Expr::Mul(neg_three, abs_shifted));

        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx,
            negative_scaled_abs,
            12
        ));

        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);
        let neg_sqrt_x = ctx.add(Expr::Neg(sqrt_x));
        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx, neg_sqrt_x, 12
        ));

        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Sub(x, one));
        let abs_shifted = ctx.call_builtin(BuiltinFn::Abs, vec![shifted]);
        let zero = ctx.num(0);
        let zero_scaled_abs = ctx.add(Expr::Mul(zero, abs_shifted));
        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx,
            zero_scaled_abs,
            12
        ));
    }

    #[test]
    fn detects_invalid_real_log_base_policy() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let neg_two = ctx.add(Expr::Neg(two));

        assert!(log_base_is_invalid_over_reals(&mut ctx, zero, 12));
        assert!(log_base_is_invalid_over_reals(&mut ctx, one, 12));
        assert!(log_base_is_invalid_over_reals(&mut ctx, neg_two, 12));

        let two = ctx.num(2);
        assert!(!log_base_is_invalid_over_reals(&mut ctx, two, 12));
    }

    #[test]
    fn classifies_symbolic_constants_outside_bounded_inverse_domains() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let e = ctx.add(Expr::Constant(Constant::E));
        let neg_e = ctx.add(Expr::Neg(e));
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let zero = ctx.num(0);

        for builtin in [BuiltinFn::Arcsin, BuiltinFn::Atanh] {
            for arg in [pi, neg_e, phi] {
                assert!(known_constant_abs_exceeds_one(&ctx, arg));
                assert_eq!(
                    bounded_inverse_real_domain_rejection_over_reals(
                        &mut ctx,
                        Some(builtin),
                        &[arg],
                        12,
                    ),
                    Some(BoundedInverseRealDomainRejection::SourceDomainEmpty)
                );
            }

            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(builtin),
                    &[zero],
                    12,
                ),
                None
            );
        }

        assert!(!known_constant_abs_exceeds_one(&ctx, zero));
    }

    #[test]
    fn classifies_positive_constants_exceeding_one_without_abs_leakage() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let e = ctx.add(Expr::Constant(Constant::E));
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let neg_pi = ctx.add(Expr::Neg(pi));
        let two = ctx.num(2);
        let one = ctx.num(1);
        let half = ctx.add(Expr::Div(one, two));
        let zero = ctx.num(0);

        for expr in [pi, e, phi, two] {
            assert!(known_positive_constant_exceeds_one(&ctx, expr));
        }
        for expr in [neg_pi, one, half, zero] {
            assert!(!known_positive_constant_exceeds_one(&ctx, expr));
        }
    }

    #[test]
    fn detects_named_constant_offsets_outside_bounded_inverse_domains() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let phi_plus_square = ctx.add(Expr::Add(phi, x_squared));
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let pi_plus_square = ctx.add(Expr::Add(pi, x_squared));

        for arg in [phi_plus_square, pi_plus_square] {
            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(BuiltinFn::Arcsin),
                    &[arg],
                    12,
                ),
                Some(BoundedInverseRealDomainRejection::SourceDomainEmpty)
            );
            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(BuiltinFn::Atanh),
                    &[arg],
                    12,
                ),
                Some(BoundedInverseRealDomainRejection::SourceDomainEmpty)
            );
        }

        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let phi_plus_square = ctx.add(Expr::Add(phi, x_squared));
        let neg_phi_plus_square = ctx.add(Expr::Neg(phi_plus_square));
        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Arccos),
                &[neg_phi_plus_square],
                12,
            ),
            Some(BoundedInverseRealDomainRejection::SourceDomainEmpty)
        );
    }

    #[test]
    fn keeps_partial_named_and_small_offsets_inside_bounded_inverse_domains() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let one = ctx.num(1);
        let half = ctx.add(Expr::Div(one, two));
        let half_plus_square = ctx.add(Expr::Add(half, x_squared));
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let pi_minus_square = ctx.add(Expr::Sub(pi, x_squared));

        for arg in [half_plus_square, pi_minus_square] {
            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(BuiltinFn::Arcsin),
                    &[arg],
                    12,
                ),
                None
            );
        }
    }

    #[test]
    fn keeps_closed_inverse_trig_finite_boundary_constants_defined() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let neg_one = ctx.add(Expr::Neg(one));

        for arg in [one, neg_one] {
            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(BuiltinFn::Arcsin),
                    &[arg],
                    12,
                ),
                None
            );
            assert_eq!(
                bounded_inverse_real_domain_rejection_over_reals(
                    &mut ctx,
                    Some(BuiltinFn::Arccos),
                    &[arg],
                    12,
                ),
                None
            );
        }
    }

    #[test]
    fn detects_impossible_nonnegative_condition_for_strictly_negative_quadratics() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let neg_x_squared = ctx.add(Expr::Neg(x_squared));
        let one = ctx.num(1);
        let strictly_negative = ctx.add(Expr::Sub(neg_x_squared, one));

        assert!(nonnegative_condition_is_impossible_over_reals(
            &mut ctx,
            strictly_negative,
            12
        ));

        let x_squared = ctx.add(Expr::Pow(x, two));
        let nonpositive = ctx.add(Expr::Neg(x_squared));
        assert!(!nonnegative_condition_is_impossible_over_reals(
            &mut ctx,
            nonpositive,
            12
        ));
    }

    #[test]
    fn proves_shifted_square_nonnegative_condition() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let shifted_square = ctx.add(Expr::Pow(shifted, two));

        assert!(nonnegative_condition_is_proven_over_reals(
            &mut ctx,
            shifted_square,
            12
        ));
    }

    #[test]
    fn preserves_unknown_or_nonempty_positive_conditions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let one = ctx.num(1);
        let sign_changing = ctx.add(Expr::Sub(x_squared, one));

        assert!(!positive_condition_is_impossible_over_reals(
            &mut ctx,
            sign_changing,
            12
        ));

        let one = ctx.num(1);
        let strictly_positive = ctx.add(Expr::Add(x_squared, one));
        assert!(!positive_condition_is_impossible_over_reals(
            &mut ctx,
            strictly_positive,
            12
        ));
    }

    #[test]
    fn proves_strict_positive_named_constant_offsets_over_reals() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let phi_plus_square = ctx.add(Expr::Add(phi, x_squared));
        let one = ctx.num(1);
        let lower_branch = ctx.add(Expr::Sub(phi_plus_square, one));

        assert!(positive_condition_is_proven_over_reals(
            &mut ctx,
            lower_branch,
            12
        ));

        let half = ctx.add(Expr::Div(one, two));
        let half_plus_square = ctx.add(Expr::Add(half, x_squared));
        let uncertain_lower_branch = ctx.add(Expr::Sub(half_plus_square, one));
        assert!(!positive_condition_is_proven_over_reals(
            &mut ctx,
            uncertain_lower_branch,
            12
        ));

        for source in ["x^2 + 1 + phi", "x^2 + phi - 1", "x^2 - 1 + phi"] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(source, &mut ctx).unwrap();
            assert!(
                positive_condition_is_proven_over_reals(&mut ctx, expr, 12),
                "source: {source}"
            );
        }

        for source in [
            "4*x^2 + 12*x + 9 + phi",
            "4*x^2 + 12*x + 9 + pi",
            "(2*x+3)^2 + phi",
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(source, &mut ctx).unwrap();
            assert!(
                positive_condition_is_proven_over_reals(&mut ctx, expr, 12),
                "source: {source}"
            );
        }

        let mut ctx = Context::new();
        let uncertain = cas_parser::parse("4*x^2 + 12*x + 9 - phi", &mut ctx).unwrap();
        assert!(!positive_condition_is_proven_over_reals(
            &mut ctx, uncertain, 12
        ));
    }

    #[test]
    fn classifies_bounded_inverse_domain_rejection_reason() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let neg_x_squared = ctx.add(Expr::Neg(x_squared));

        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Acosh),
                &[neg_x_squared],
                12,
            ),
            Some(BoundedInverseRealDomainRejection::SourceDomainEmpty)
        );

        let one = ctx.num(1);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let one_minus_x_squared = ctx.add(Expr::Sub(one, x_squared));

        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Acosh),
                &[one_minus_x_squared],
                12,
            ),
            Some(BoundedInverseRealDomainRejection::DerivativeDomainEmpty)
        );

        let one = ctx.num(1);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let one_plus_x_squared = ctx.add(Expr::Add(one, x_squared));

        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Acosh),
                &[one_plus_x_squared],
                12,
            ),
            None
        );

        let x = ctx.var("x");
        let two = ctx.num(2);
        let one = ctx.num(1);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let x_squared_plus_one = ctx.add(Expr::Add(x_squared, one));
        let sqrt_boundary = ctx.call_builtin(BuiltinFn::Sqrt, vec![x_squared_plus_one]);

        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Arcsin),
                &[sqrt_boundary],
                12,
            ),
            Some(BoundedInverseRealDomainRejection::DerivativeDomainEmpty)
        );

        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let shifted_square = ctx.add(Expr::Pow(shifted, two));
        let one = ctx.num(1);
        let shifted_boundary = ctx.add(Expr::Add(shifted_square, one));

        assert_eq!(
            bounded_inverse_real_domain_rejection_over_reals(
                &mut ctx,
                Some(BuiltinFn::Arcsin),
                &[shifted_boundary],
                12,
            ),
            Some(BoundedInverseRealDomainRejection::DerivativeDomainEmpty)
        );
    }

    #[test]
    fn detects_closed_inverse_trig_empty_derivative_boundary_domains() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let one = ctx.num(1);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let x_squared_plus_one = ctx.add(Expr::Add(x_squared, one));
        let sqrt_boundary = ctx.call_builtin(BuiltinFn::Sqrt, vec![x_squared_plus_one]);

        assert!(closed_inverse_trig_derivative_domain_is_empty_over_reals(
            &ctx,
            Some(BuiltinFn::Arcsin),
            &[sqrt_boundary],
            12,
        ));

        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let shifted_square = ctx.add(Expr::Pow(shifted, two));
        let one = ctx.num(1);
        let shifted_boundary = ctx.add(Expr::Add(shifted_square, one));

        assert!(closed_inverse_trig_derivative_domain_is_empty_over_reals(
            &ctx,
            Some(BuiltinFn::Arcsin),
            &[shifted_boundary],
            12,
        ));

        let x = ctx.var("x");
        assert!(!closed_inverse_trig_derivative_domain_is_empty_over_reals(
            &ctx,
            Some(BuiltinFn::Arcsin),
            &[x],
            12,
        ));
    }

    #[test]
    fn detects_static_empty_real_domain_expressions() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let neg_one = ctx.add(Expr::Neg(one));
        let sqrt_neg_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![neg_one]);
        let ln_zero = ctx.call_builtin(BuiltinFn::Ln, vec![zero]);
        let base_one_log = ctx.call_builtin(BuiltinFn::Log, vec![one, two]);
        let x = ctx.var("x");
        let x_squared = ctx.add(Expr::Pow(x, two));
        let x_squared_plus_two = ctx.add(Expr::Add(x_squared, two));
        let acos_empty = ctx.call_builtin(BuiltinFn::Acos, vec![x_squared_plus_two]);
        let atanh_empty = ctx.call_builtin(BuiltinFn::Atanh, vec![x_squared_plus_two]);

        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx,
            sqrt_neg_one,
            12,
            16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx, ln_zero, 12, 16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx,
            base_one_log,
            12,
            16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx, acos_empty, 12, 16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx,
            atanh_empty,
            12,
            16
        ));
    }

    #[test]
    fn preserves_symbolic_or_nonempty_static_domains() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let y_squared = ctx.add(Expr::Pow(y, two));
        let nonempty = ctx.add(Expr::Neg(y_squared));
        let sqrt_nonempty = ctx.call_builtin(BuiltinFn::Sqrt, vec![nonempty]);
        let ln_symbolic = ctx.call_builtin(BuiltinFn::Ln, vec![y]);
        let sqrt_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![one]);
        let y_squared_plus_one = ctx.add(Expr::Add(y_squared, one));
        let acos_point_domain = ctx.call_builtin(BuiltinFn::Acos, vec![y_squared_plus_one]);
        let atanh_symbolic = ctx.call_builtin(BuiltinFn::Atanh, vec![y]);

        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            sqrt_nonempty,
            12,
            16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            ln_symbolic,
            12,
            16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx, sqrt_one, 12, 16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            acos_point_domain,
            12,
            16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            atanh_symbolic,
            12,
            16
        ));
    }

    #[test]
    fn nonfinite_scan_is_explicit_policy_not_static_empty_domain() {
        let mut ctx = Context::new();
        let infinity = ctx.add(Expr::Constant(Constant::Infinity));
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(infinity, one));

        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx, expr, 12, 16
        ));
        assert!(real_domain_is_empty_or_nonfinite_over_reals(
            &mut ctx, expr, 12, 16
        ));
    }
}
