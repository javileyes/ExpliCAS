//! Symbolic differentiation helpers shared by differentiation-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_nary::{
    add_leaves, build_balanced_add, build_balanced_mul, mul_leaves, AddView, Sign,
};
use crate::expr_predicates::contains_named_var;
use crate::polynomial::Polynomial;
use crate::prove_sign::prove_positive_depth_with;
use crate::root_forms::{
    extract_square_root_base, try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind,
};
use crate::tri_proof::TriProof;
use cas_ast::{ordering::compare_expr, BuiltinFn, Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

const SYMBOLIC_DIFF_SIGN_PROOF_DEPTH: usize = 8;

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn add_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

fn sub_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        ctx.add(Expr::Neg(right))
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

fn mul_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) || is_zero(ctx, right) {
        ctx.num(0)
    } else if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        mul2_raw(ctx, left, right)
    }
}

fn div_pruned(ctx: &mut Context, num: ExprId, den: ExprId) -> ExprId {
    if is_zero(ctx, num) {
        ctx.num(0)
    } else if is_one(ctx, den) {
        num
    } else {
        ctx.add(Expr::Div(num, den))
    }
}

fn neg_pruned(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr) {
        Expr::Number(value) => ctx.add(Expr::Number(-value.clone())),
        Expr::Neg(inner) => *inner,
        Expr::Div(num, den) => {
            if let Expr::Neg(inner_num) = ctx.get(*num) {
                return ctx.add(Expr::Div(*inner_num, *den));
            }
            ctx.add(Expr::Neg(expr))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn rational_constant_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Add(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left + right)
        }
        Expr::Sub(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left - right)
        }
        Expr::Mul(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left * right)
        }
        Expr::Div(num, den) => {
            let numerator = rational_constant_value(ctx, *num)?;
            let denominator = rational_constant_value(ctx, *den)?;
            if denominator.is_zero() {
                None
            } else {
                Some(numerator / denominator)
            }
        }
        Expr::Neg(inner) => rational_constant_value(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn expr_is_additive(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn expr_contains_plain_log_abs(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) =>
        {
            matches!(
                ctx.get(args[0]),
                Expr::Function(abs_fn_id, abs_args)
                    if abs_args.len() == 1
                        && ctx.builtin_of(*abs_fn_id) == Some(BuiltinFn::Abs)
            )
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            expr_contains_plain_log_abs(ctx, *left) || expr_contains_plain_log_abs(ctx, *right)
        }
        Expr::Div(num, den) => {
            expr_contains_plain_log_abs(ctx, *num) || expr_contains_plain_log_abs(ctx, *den)
        }
        Expr::Neg(inner) => expr_contains_plain_log_abs(ctx, *inner),
        Expr::Pow(_, exp) => expr_contains_plain_log_abs(ctx, *exp),
        _ => false,
    }
}

fn scale_expr_pruned(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale.is_zero() || is_zero(ctx, expr) {
        return ctx.num(0);
    }
    if scale.is_one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return neg_pruned(ctx, expr);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    mul_pruned(ctx, scale_expr, expr)
}

fn differentiate_scaled_additive_log_abs_by_linearity(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2
        || !terms
            .iter()
            .any(|(_scale, term)| expr_contains_plain_log_abs(ctx, *term))
    {
        return None;
    }

    let mut out = ctx.num(0);
    for (scale, term) in terms {
        let derivative = differentiate_symbolic_expr(ctx, term, var)?;
        let scaled_derivative = scale_expr_pruned(ctx, scale, derivative);
        out = add_pruned(ctx, out, scaled_derivative);
    }

    Some(out)
}

fn differentiate_additive_by_linearity(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(add_pruned(ctx, dl, dr))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(sub_pruned(ctx, dl, dr))
        }
        _ => None,
    }
}

fn collect_scaled_add_terms(
    ctx: &Context,
    expr: ExprId,
    scale: BigRational,
    terms: &mut Vec<(BigRational, ExprId)>,
) {
    if scale.is_zero() {
        return;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_scaled_add_terms(ctx, *left, scale.clone(), terms);
            collect_scaled_add_terms(ctx, *right, scale, terms);
        }
        Expr::Sub(left, right) => {
            collect_scaled_add_terms(ctx, *left, scale.clone(), terms);
            collect_scaled_add_terms(ctx, *right, -scale, terms);
        }
        Expr::Neg(inner) => collect_scaled_add_terms(ctx, *inner, -scale, terms),
        Expr::Div(num, den) => {
            if let Some(den_scale) = rational_constant_value(ctx, *den) {
                if !den_scale.is_zero() {
                    collect_scaled_add_terms(ctx, *num, scale / den_scale, terms);
                    return;
                }
            }
            terms.push((scale, expr));
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let original_scale = scale.clone();
            let mut scaled = scale;
            let mut additive = None;
            let mut can_distribute = true;

            for factor in factors {
                if let Some(value) = rational_constant_value(ctx, factor) {
                    scaled *= value;
                    continue;
                }

                if expr_is_additive(ctx, factor) {
                    if additive.replace(factor).is_some() {
                        terms.push((original_scale, expr));
                        return;
                    }
                    continue;
                }

                can_distribute = false;
                break;
            }

            if can_distribute {
                if let Some(additive) = additive {
                    collect_scaled_add_terms(ctx, additive, scaled, terms);
                } else {
                    terms.push((original_scale, expr));
                }
            } else {
                terms.push((original_scale, expr));
            }
        }
        _ => terms.push((scale, expr)),
    }
}

fn scaled_polynomial(poly: &Polynomial, scale: BigRational) -> Polynomial {
    poly.mul(&Polynomial::new(vec![scale], poly.var.clone()))
}

fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

fn exact_rational_sqrt(value: &BigRational) -> Option<BigRational> {
    if value.is_negative() {
        return None;
    }

    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(BigRational::new(sqrt_num, sqrt_den))
    } else {
        None
    }
}

fn split_constant_scaled_single_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (scale, core) = split_constant_scaled_single_factor(ctx, *inner)?;
            return Some((-scale, core));
        }
        Expr::Div(num, den) => {
            let den_scale = rational_constant_value(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (scale, core) = split_constant_scaled_single_factor(ctx, *num)?;
            return Some((scale / den_scale, core));
        }
        _ => {}
    }

    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut core = None;

    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }
        if core.replace(factor).is_some() {
            return None;
        }
    }

    core.map(|core| (scale, core))
}

fn arctan_reciprocal_affine_factor(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial)> {
    let arg = unary_builtin_arg(ctx, expr, BuiltinFn::Arctan)
        .or_else(|| unary_builtin_arg(ctx, expr, BuiltinFn::Atan))?;
    let reciprocal_base = unit_reciprocal_base(ctx, arg)?;
    let base_poly = Polynomial::from_expr(ctx, reciprocal_base, var).ok()?;
    if base_poly.degree() != 1 {
        return None;
    }
    Some((expr, base_poly))
}

fn arctan_reciprocal_affine_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut arctan_expr = None;
    let mut base_poly = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if let Some((atan, base)) = arctan_reciprocal_affine_factor(ctx, factor, var) {
            if arctan_expr.replace(atan).is_some() {
                return None;
            }
            base_poly = Some(base);
            continue;
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((arctan_expr?, base_poly?, cofactor_poly))
}

fn atanh_affine_factor(ctx: &Context, expr: ExprId, var: &str) -> Option<(ExprId, Polynomial)> {
    let arg = unary_builtin_arg(ctx, expr, BuiltinFn::Atanh)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    Some((expr, arg_poly))
}

fn atanh_affine_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut atanh_expr = None;
    let mut arg_poly = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if let Some((atanh, arg)) = atanh_affine_factor(ctx, factor, var) {
            if atanh_expr.replace(atanh).is_some() {
                return None;
            }
            arg_poly = Some(arg);
            continue;
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((atanh_expr?, arg_poly?, cofactor_poly))
}

fn scaled_arctan_affine_term(
    ctx: &Context,
    scale: BigRational,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, Polynomial)> {
    let (term_scale, core) = split_constant_scaled_single_factor(ctx, expr)
        .unwrap_or_else(|| (BigRational::one(), expr));
    let arg = unary_builtin_arg(ctx, core, BuiltinFn::Arctan)
        .or_else(|| unary_builtin_arg(ctx, core, BuiltinFn::Atan))?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    (arg_poly.degree() == 1).then_some((scale * term_scale, arg_poly))
}

fn positive_quadratic_rational_term(
    ctx: &Context,
    scale: BigRational,
    expr: ExprId,
    var: &str,
) -> Option<(Polynomial, Polynomial)> {
    let (term_scale, core) = split_constant_scaled_single_factor(ctx, expr)
        .unwrap_or_else(|| (BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core) else {
        return None;
    };

    let (den_scale, den_core) = split_constant_scaled_single_factor(ctx, *den)?;
    if den_scale.is_zero() {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, *num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den_core, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }

    Some((
        denominator,
        scaled_polynomial(&numerator, scale * term_scale / den_scale),
    ))
}

fn arctan_affine_polynomial_cofactor_term(
    ctx: &Context,
    scale: BigRational,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = scale;
    let mut arg_poly = None;
    let mut cofactor = Polynomial::one(var.to_string());

    for factor in factors {
        if let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Arctan)
            .or_else(|| unary_builtin_arg(ctx, factor, BuiltinFn::Atan))
        {
            if arg_poly.is_some() {
                return None;
            }
            let arg = Polynomial::from_expr(ctx, arg, var).ok()?;
            if arg.degree() != 1 {
                return None;
            }
            arg_poly = Some(arg);
            continue;
        }

        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor = cofactor.mul(&factor_poly);
    }

    Some((scale, arg_poly?, cofactor))
}

fn positive_quadratic_square_primitive_scale(
    denominator: &Polynomial,
    rational_numerator: &Polynomial,
    arctan_arg: &Polynomial,
    arctan_scale: BigRational,
) -> Option<BigRational> {
    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let discriminant = four.clone() * a.clone() * c - b.clone() * b.clone();
    if discriminant <= BigRational::zero() {
        return None;
    }
    let discriminant_root = exact_rational_sqrt(&discriminant)?;

    let expected_arg = Polynomial::new(
        vec![
            b / discriminant_root.clone(),
            two * a.clone() / discriminant_root.clone(),
        ],
        denominator.var.clone(),
    );
    if arctan_arg != &expected_arg {
        return None;
    }

    let ratio = constant_polynomial_ratio(rational_numerator, &expected_arg)?;
    let primitive_scale = ratio * discriminant_root.clone();
    if primitive_scale.is_zero() {
        return None;
    }

    let expected_arctan_scale =
        four * a * primitive_scale.clone() / (discriminant * discriminant_root);
    if arctan_scale != expected_arctan_scale {
        return None;
    }

    Some(primitive_scale)
}

fn positive_quadratic_square_result(
    ctx: &mut Context,
    denominator: Polynomial,
    primitive_scale: BigRational,
) -> ExprId {
    let denominator_expr = denominator.to_expr(ctx);
    let two = ctx.num(2);
    let squared = ctx.add(Expr::Pow(denominator_expr, two));
    let numerator = ctx.add(Expr::Number(primitive_scale));
    div_pruned(ctx, numerator, squared)
}

fn positive_quadratic_square_sum_primitive_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut denominator = None;
    let mut rational_numerator = Polynomial::zero(var.to_string());
    let mut arctan_arg = None;
    let mut arctan_scale = BigRational::zero();

    for (scale, term) in terms {
        if let Some((candidate_denominator, numerator)) =
            positive_quadratic_rational_term(ctx, scale.clone(), term, var)
        {
            if let Some(existing) = &denominator {
                if existing != &candidate_denominator {
                    return None;
                }
            } else {
                denominator = Some(candidate_denominator);
            }
            rational_numerator = rational_numerator.add(&numerator);
            continue;
        }

        if let Some((term_scale, candidate_arg)) = scaled_arctan_affine_term(ctx, scale, term, var)
        {
            if let Some(existing) = &arctan_arg {
                if existing != &candidate_arg {
                    return None;
                }
            } else {
                arctan_arg = Some(candidate_arg);
            }
            arctan_scale += term_scale;
            continue;
        }

        return None;
    }

    let denominator = denominator?;
    let arctan_arg = arctan_arg?;
    let primitive_scale = positive_quadratic_square_primitive_scale(
        &denominator,
        &rational_numerator,
        &arctan_arg,
        arctan_scale,
    )?;
    Some(positive_quadratic_square_result(
        ctx,
        denominator,
        primitive_scale,
    ))
}

fn positive_quadratic_square_combined_primitive_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (den_scale, den_core) = split_constant_scaled_single_factor(ctx, *den)?;
    if den_scale.is_zero() {
        return None;
    }
    let denominator = Polynomial::from_expr(ctx, den_core, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }

    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, *num, BigRational::one() / den_scale, &mut terms);
    let mut rational_numerator = Polynomial::zero(var.to_string());
    let mut arctan_arg = None;
    let mut arctan_cofactor = Polynomial::zero(var.to_string());

    for (scale, term) in terms {
        if let Some((term_scale, candidate_arg, cofactor)) =
            arctan_affine_polynomial_cofactor_term(ctx, scale.clone(), term, var)
        {
            if let Some(existing) = &arctan_arg {
                if existing != &candidate_arg {
                    return None;
                }
            } else {
                arctan_arg = Some(candidate_arg);
            }
            arctan_cofactor = arctan_cofactor.add(&scaled_polynomial(&cofactor, term_scale));
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var).ok()?;
        rational_numerator = rational_numerator.add(&scaled_polynomial(&term_poly, scale));
    }

    let arctan_scale = constant_polynomial_ratio(&arctan_cofactor, &denominator)?;
    let primitive_scale = positive_quadratic_square_primitive_scale(
        &denominator,
        &rational_numerator,
        &arctan_arg?,
        arctan_scale,
    )?;
    Some(positive_quadratic_square_result(
        ctx,
        denominator,
        primitive_scale,
    ))
}

fn positive_quadratic_square_primitive_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    positive_quadratic_square_sum_primitive_derivative(ctx, expr, var)
        .or_else(|| positive_quadratic_square_combined_primitive_derivative(ctx, expr, var))
}

fn asinh_affine_factor(ctx: &Context, expr: ExprId, var: &str) -> Option<(ExprId, Polynomial)> {
    let arg = unary_builtin_arg(ctx, expr, BuiltinFn::Asinh)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    Some((expr, arg_poly))
}

fn asinh_affine_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut asinh_expr = None;
    let mut arg_poly = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if let Some((asinh, arg)) = asinh_affine_factor(ctx, factor, var) {
            if asinh_expr.replace(asinh).is_some() {
                return None;
            }
            arg_poly = Some(arg);
            continue;
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((asinh_expr?, arg_poly?, cofactor_poly))
}

fn acosh_affine_factor(ctx: &Context, expr: ExprId, var: &str) -> Option<(ExprId, Polynomial)> {
    let arg = unary_builtin_arg(ctx, expr, BuiltinFn::Acosh)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    Some((expr, arg_poly))
}

fn acosh_affine_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut acosh_expr = None;
    let mut arg_poly = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if let Some((acosh, arg)) = acosh_affine_factor(ctx, factor, var) {
            if acosh_expr.replace(acosh).is_some() {
                return None;
            }
            arg_poly = Some(arg);
            continue;
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((acosh_expr?, arg_poly?, cofactor_poly))
}

fn sqrt_product_pair_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut radicands = Vec::new();

    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }

        let radicand = unary_builtin_arg(ctx, factor, BuiltinFn::Sqrt)?;
        radicands.push(Polynomial::from_expr(ctx, radicand, var).ok()?);
    }

    if radicands.len() != 2 {
        return None;
    }
    Some((scale, radicands.remove(0), radicands.remove(0)))
}

fn sqrt_gap_term(ctx: &Context, expr: ExprId, var: &str) -> Option<(BigRational, Polynomial)> {
    let (scale, core) = split_constant_scaled_single_factor(ctx, expr)?;
    let radicand = if let Some(radicand) = unary_builtin_arg(ctx, core, BuiltinFn::Sqrt) {
        radicand
    } else {
        let Expr::Pow(base, exp) = ctx.get(core) else {
            return None;
        };
        if !matches!(
            ctx.get(*exp),
            Expr::Number(n) if *n == BigRational::new(1.into(), 2.into())
        ) {
            return None;
        }
        *base
    };

    Some((scale, Polynomial::from_expr(ctx, radicand, var).ok()?))
}

fn ln_quadratic_term(ctx: &Context, expr: ExprId, var: &str) -> Option<(BigRational, Polynomial)> {
    let (scale, core) = split_constant_scaled_single_factor(ctx, expr)?;
    let ln_arg = unary_builtin_arg(ctx, core, BuiltinFn::Ln)?;
    let poly = Polynomial::from_expr(ctx, ln_arg, var).ok()?;
    if poly.is_zero() || poly.degree() > 2 {
        return None;
    }
    Some((scale, poly))
}

fn log_minus_one_core(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(log_expr, one) if is_one(ctx, *one) => Some(*log_expr),
        Expr::Add(left, right) => {
            if rational_constant_value(ctx, *left) == Some(-BigRational::one()) {
                return Some(*right);
            }
            if rational_constant_value(ctx, *right) == Some(-BigRational::one()) {
                return Some(*left);
            }
            None
        }
        _ => None,
    }
}

fn log_minus_one_square_base(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::from_integer(2.into())) {
        return None;
    }

    let log_expr = log_minus_one_core(ctx, *base)?;
    let log_base = unary_builtin_arg(ctx, log_expr, BuiltinFn::Ln)?;
    Some((log_expr, log_base))
}

fn log_shift_square_polynomial_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut log_expr = None;
    let mut log_base_poly = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if log_expr.is_none() {
            if let Some((candidate_log_expr, log_base)) = log_minus_one_square_base(ctx, factor) {
                log_base_poly = Some(Polynomial::from_expr(ctx, log_base, var).ok()?);
                log_expr = Some(candidate_log_expr);
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((log_expr?, log_base_poly?, cofactor_poly))
}

fn natural_log_power_factor(ctx: &Context, factor: ExprId) -> Option<(ExprId, ExprId, u32)> {
    match ctx.get(factor) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) =>
        {
            Some((factor, args[0], 1))
        }
        Expr::Pow(base, exp) => {
            let power = match ctx.get(*exp) {
                Expr::Number(n) if *n == BigRational::from_integer(2.into()) => 2,
                Expr::Number(n) if *n == BigRational::from_integer(3.into()) => 3,
                _ => return None,
            };
            let Expr::Function(fn_id, args) = ctx.get(*base) else {
                return None;
            };
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) {
                Some((*base, args[0], power))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn log_power_polynomial_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, u32, Polynomial)> {
    let factors = mul_leaves(ctx, expr);
    let mut log_expr = None;
    let mut log_base_poly = None;
    let mut power = None;
    let mut cofactor_poly = Polynomial::one(var.to_string());

    for factor in factors {
        if log_expr.is_none() {
            if let Some((candidate_log_expr, log_base, candidate_power)) =
                natural_log_power_factor(ctx, factor)
            {
                log_base_poly = Some(Polynomial::from_expr(ctx, log_base, var).ok()?);
                log_expr = Some(candidate_log_expr);
                power = Some(candidate_power);
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    Some((log_expr?, log_base_poly?, power?, cofactor_poly))
}

fn expand_polynomial_times_additive_term(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<Vec<(BigRational, ExprId)>> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut additive = None;
    let mut cofactor_factors = Vec::new();

    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }

        if expr_is_additive(ctx, factor) {
            if Polynomial::from_expr(ctx, factor, var).is_ok() {
                cofactor_factors.push(factor);
                continue;
            }
            if additive.replace(factor).is_some() {
                return None;
            }
            continue;
        }

        Polynomial::from_expr(ctx, factor, var).ok()?;
        cofactor_factors.push(factor);
    }

    let additive = additive?;
    if cofactor_factors.is_empty() {
        return None;
    }

    let cofactor = build_balanced_mul(ctx, &cofactor_factors);
    let mut out = Vec::new();
    for (term, sign) in AddView::from_expr(ctx, additive).terms {
        let term_scale = match sign {
            Sign::Pos => scale.clone(),
            Sign::Neg => -scale.clone(),
        };
        let expanded = mul_pruned(ctx, cofactor, term);
        out.push((term_scale, expanded));
    }
    Some(out)
}

fn collect_log_square_by_parts_terms(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<(BigRational, ExprId)> {
    let mut raw_terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut raw_terms);

    let mut out = Vec::new();
    for (scale, term) in raw_terms {
        if let Some(expanded_terms) = expand_polynomial_times_additive_term(ctx, term, var) {
            for (inner_scale, inner_term) in expanded_terms {
                out.push((scale.clone() * inner_scale, inner_term));
            }
        } else {
            out.push((scale, term));
        }
    }
    out
}

fn log_square_by_parts_derivative(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let terms = collect_log_square_by_parts_terms(ctx, expr, var);
    if terms.len() < 2 {
        return None;
    }

    let mut log_expr = None;
    let mut base_poly = None;
    let mut shifted_square_poly = Polynomial::zero(var.to_string());
    let mut log_cube_poly = Polynomial::zero(var.to_string());
    let mut log_square_poly = Polynomial::zero(var.to_string());
    let mut log_linear_poly = Polynomial::zero(var.to_string());
    let mut plain_poly = Polynomial::zero(var.to_string());

    for (scale, term) in terms {
        if let Some((candidate_log_expr, candidate_base_poly, cofactor_poly)) =
            log_shift_square_polynomial_term(ctx, term, var)
        {
            if let Some(existing) = log_expr {
                if compare_expr(ctx, existing, candidate_log_expr) != Ordering::Equal {
                    return None;
                }
            } else {
                log_expr = Some(candidate_log_expr);
            }

            if let Some(existing) = &base_poly {
                if existing != &candidate_base_poly {
                    return None;
                }
            } else {
                base_poly = Some(candidate_base_poly);
            }

            shifted_square_poly =
                shifted_square_poly.add(&scaled_polynomial(&cofactor_poly, scale));
            continue;
        }

        if let Some((candidate_log_expr, candidate_base_poly, power, cofactor_poly)) =
            log_power_polynomial_term(ctx, term, var)
        {
            if let Some(existing) = log_expr {
                if compare_expr(ctx, existing, candidate_log_expr) != Ordering::Equal {
                    return None;
                }
            } else {
                log_expr = Some(candidate_log_expr);
            }

            if let Some(existing) = &base_poly {
                if existing != &candidate_base_poly {
                    return None;
                }
            } else {
                base_poly = Some(candidate_base_poly);
            }

            let scaled = scaled_polynomial(&cofactor_poly, scale);
            match power {
                1 => log_linear_poly = log_linear_poly.add(&scaled),
                2 => log_square_poly = log_square_poly.add(&scaled),
                3 => log_cube_poly = log_cube_poly.add(&scaled),
                _ => return None,
            }
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var).ok()?;
        plain_poly = plain_poly.add(&scaled_polynomial(&term_poly, scale));
    }

    let log_expr = log_expr?;
    let base_poly = base_poly?;
    let (by_parts_poly, output_power) = if !shifted_square_poly.is_zero() {
        if !log_cube_poly.is_zero() || !log_square_poly.is_zero() || !log_linear_poly.is_zero() {
            return None;
        }
        if plain_poly.is_zero() || shifted_square_poly != plain_poly {
            return None;
        }
        (shifted_square_poly, 2)
    } else if !log_cube_poly.is_zero() {
        let minus_three = -BigRational::from_integer(3.into());
        let six = BigRational::from_integer(6.into());
        let minus_six = -six.clone();
        if log_square_poly != scaled_polynomial(&log_cube_poly, minus_three) {
            return None;
        }
        if log_linear_poly != scaled_polynomial(&log_cube_poly, six) {
            return None;
        }
        if plain_poly != scaled_polynomial(&log_cube_poly, minus_six) {
            return None;
        }
        (log_cube_poly, 3)
    } else {
        if log_square_poly.is_zero() || plain_poly.is_zero() {
            return None;
        }
        let minus_two = -BigRational::from_integer(2.into());
        let two = BigRational::from_integer(2.into());
        if log_linear_poly != scaled_polynomial(&log_square_poly, minus_two) {
            return None;
        }
        if plain_poly != scaled_polynomial(&log_square_poly, two) {
            return None;
        }
        (log_square_poly, 2)
    };

    let scale = constant_polynomial_ratio(&by_parts_poly, &base_poly)?;
    if scale.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative_poly = scaled_polynomial(&base_poly.derivative(), scale);
    let derivative = derivative_poly.to_expr(ctx);
    let power = ctx.num(output_power);
    let log_power = ctx.add(Expr::Pow(log_expr, power));
    Some(mul_pruned(ctx, derivative, log_power))
}

fn trig_power_term(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId, u32)> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Sin) {
        return Some((BuiltinFn::Sin, arg, 1));
    }
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Cos) {
        return Some((BuiltinFn::Cos, arg, 1));
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(power) = ctx.get(*exp) else {
        return None;
    };
    if power.denom() != &1.into() {
        return None;
    }
    let power = power.numer().to_u32()?;
    if !matches!(power, 3 | 5) {
        return None;
    }

    if let Some(arg) = unary_builtin_arg(ctx, *base, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, arg, power))
    } else {
        unary_builtin_arg(ctx, *base, BuiltinFn::Cos).map(|arg| (BuiltinFn::Cos, arg, power))
    }
}

fn trig_power_coeff(terms: &[(u32, BigRational)], power: u32) -> BigRational {
    terms
        .iter()
        .filter_map(|(term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

fn trig_square_power_reduction_primitive_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let raw_terms = collect_log_square_by_parts_terms(ctx, expr, var);
    if raw_terms.len() < 2 {
        return None;
    }

    let mut plain_poly = Polynomial::zero(var.to_string());
    let mut sin_arg = None;
    let mut sin_coeff = BigRational::zero();

    for (scale, term) in raw_terms {
        let (scale, term) =
            if let Some((term_scale, core)) = split_constant_scaled_single_factor(ctx, term) {
                (scale * term_scale, core)
            } else {
                (scale, term)
            };

        if let Some(arg) = unary_builtin_arg(ctx, term, BuiltinFn::Sin) {
            if let Some(existing) = sin_arg {
                if compare_expr(ctx, existing, arg) != Ordering::Equal {
                    return None;
                }
            } else {
                sin_arg = Some(arg);
            }
            sin_coeff += scale;
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var).ok()?;
        plain_poly = plain_poly.add(&scaled_polynomial(&term_poly, scale));
    }

    if plain_poly.degree() > 1 {
        return None;
    }
    let linear_coeff = plain_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if linear_coeff != BigRational::new(1.into(), 2.into()) {
        return None;
    }

    let sin_arg = sin_arg?;
    let sin_arg_poly = Polynomial::from_expr(ctx, sin_arg, var).ok()?;
    if sin_arg_poly.degree() != 1 {
        return None;
    }
    let sin_arg_slope = sin_arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if sin_arg_slope.is_zero() {
        return None;
    }

    let expected_sin_coeff = BigRational::new((-1).into(), 2.into()) / sin_arg_slope.clone();
    let expected_cos_coeff = BigRational::new(1.into(), 2.into()) / sin_arg_slope.clone();
    let target_builtin = if sin_coeff == expected_sin_coeff {
        BuiltinFn::Sin
    } else if sin_coeff == expected_cos_coeff {
        BuiltinFn::Cos
    } else {
        return None;
    };

    let half = BigRational::new(1.into(), 2.into());
    let target_arg = scaled_polynomial(&sin_arg_poly, half).to_expr(ctx);
    let target = ctx.call_builtin(target_builtin, vec![target_arg]);
    let two = ctx.num(2);
    Some(ctx.add(Expr::Pow(target, two)))
}

fn trig_fifth_power_reduction_primitive_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let raw_terms = collect_log_square_by_parts_terms(ctx, expr, var);
    if raw_terms.len() < 3 {
        return None;
    }

    let mut companion_builtin = None;
    let mut arg = None;
    let mut terms = Vec::new();

    for (scale, term) in raw_terms {
        let (term_scale, term) = split_constant_scaled_single_factor(ctx, term)?;
        let scale = scale * term_scale;
        let (candidate_builtin, candidate_arg, power) = trig_power_term(ctx, term)?;
        if let Some(existing) = companion_builtin {
            if existing != candidate_builtin {
                return None;
            }
        } else {
            companion_builtin = Some(candidate_builtin);
        }

        if let Some(existing) = arg {
            if compare_expr(ctx, existing, candidate_arg) != Ordering::Equal {
                return None;
            }
        } else {
            arg = Some(candidate_arg);
        }

        terms.push((power, scale));
    }

    if terms
        .iter()
        .any(|(power, coeff)| !matches!(*power, 1 | 3 | 5) && !coeff.is_zero())
    {
        return None;
    }

    let companion_builtin = companion_builtin?;
    let arg = arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let expected = |numerator: i64, denominator: i64| {
        BigRational::new(numerator.into(), denominator.into()) / slope.clone()
    };
    let (target_builtin, expected_one, expected_three, expected_five) = match companion_builtin {
        BuiltinFn::Cos => (
            BuiltinFn::Sin,
            expected(-1, 1),
            expected(2, 3),
            expected(-1, 5),
        ),
        BuiltinFn::Sin => (
            BuiltinFn::Cos,
            expected(1, 1),
            expected(-2, 3),
            expected(1, 5),
        ),
        _ => return None,
    };

    if trig_power_coeff(&terms, 1) != expected_one
        || trig_power_coeff(&terms, 3) != expected_three
        || trig_power_coeff(&terms, 5) != expected_five
    {
        return None;
    }

    let target = ctx.call_builtin(target_builtin, vec![arg]);
    let five = ctx.num(5);
    Some(ctx.add(Expr::Pow(target, five)))
}

fn arctan_reciprocal_affine_by_parts_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut arctan_expr = None;
    let mut base_poly = None;
    let mut arctan_cofactor_poly = Polynomial::zero(var.to_string());
    let mut ln_scale = BigRational::zero();
    let mut ln_poly = None;

    for (scale, term) in terms {
        if let Some((atan, base, cofactor)) = arctan_reciprocal_affine_term(ctx, term, var) {
            if let Some(existing) = arctan_expr {
                if compare_expr(ctx, existing, atan) != Ordering::Equal {
                    return None;
                }
            } else {
                arctan_expr = Some(atan);
            }

            if let Some(existing) = &base_poly {
                if existing != &base {
                    return None;
                }
            } else {
                base_poly = Some(base);
            }

            arctan_cofactor_poly = arctan_cofactor_poly.add(&scaled_polynomial(&cofactor, scale));
            continue;
        }

        if let Some((term_scale, poly)) = ln_quadratic_term(ctx, term, var) {
            ln_scale += scale * term_scale;
            if let Some(existing) = &ln_poly {
                if existing != &poly {
                    return None;
                }
            } else {
                ln_poly = Some(poly);
            }
            continue;
        }

        return None;
    }

    let arctan_expr = arctan_expr?;
    let base_poly = base_poly?;
    let ln_poly = ln_poly?;
    let slope = base_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let expected_arctan_cofactor = base_poly.div_scalar(&slope);
    if arctan_cofactor_poly != expected_arctan_cofactor {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    let expected_ln_scale = BigRational::one() / (two * slope);
    if ln_scale != expected_ln_scale {
        return None;
    }

    let one_poly = Polynomial::one(var.to_string());
    let expected_ln_poly = base_poly.mul(&base_poly).add(&one_poly);
    if ln_poly != expected_ln_poly {
        return None;
    }

    Some(arctan_expr)
}

fn atanh_affine_by_parts_derivative(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut atanh_expr = None;
    let mut arg_poly = None;
    let mut atanh_cofactor_poly = Polynomial::zero(var.to_string());
    let mut ln_scale = BigRational::zero();
    let mut ln_poly = None;

    for (scale, term) in terms {
        if let Some((atanh, arg, cofactor)) = atanh_affine_term(ctx, term, var) {
            if let Some(existing) = atanh_expr {
                if compare_expr(ctx, existing, atanh) != Ordering::Equal {
                    return None;
                }
            } else {
                atanh_expr = Some(atanh);
            }

            if let Some(existing) = &arg_poly {
                if existing != &arg {
                    return None;
                }
            } else {
                arg_poly = Some(arg);
            }

            atanh_cofactor_poly = atanh_cofactor_poly.add(&scaled_polynomial(&cofactor, scale));
            continue;
        }

        if let Some((term_scale, poly)) = ln_quadratic_term(ctx, term, var) {
            ln_scale += scale * term_scale;
            if let Some(existing) = &ln_poly {
                if existing != &poly {
                    return None;
                }
            } else {
                ln_poly = Some(poly);
            }
            continue;
        }

        return None;
    }

    let atanh_expr = atanh_expr?;
    let arg_poly = arg_poly?;
    let ln_poly = ln_poly?;
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let expected_atanh_cofactor = arg_poly.div_scalar(&slope);
    if atanh_cofactor_poly != expected_atanh_cofactor {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    let expected_ln_scale = BigRational::one() / (two * slope);
    if ln_scale != expected_ln_scale {
        return None;
    }

    let one_poly = Polynomial::one(var.to_string());
    let expected_ln_poly = one_poly.sub(&arg_poly.mul(&arg_poly));
    if ln_poly != expected_ln_poly {
        return None;
    }

    Some(atanh_expr)
}

fn asinh_affine_by_parts_derivative(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut asinh_expr = None;
    let mut arg_poly = None;
    let mut asinh_cofactor_poly = Polynomial::zero(var.to_string());
    let mut sqrt_scale = BigRational::zero();
    let mut sqrt_gap = None;

    for (scale, term) in terms {
        if let Some((asinh, arg, cofactor)) = asinh_affine_term(ctx, term, var) {
            if let Some(existing) = asinh_expr {
                if compare_expr(ctx, existing, asinh) != Ordering::Equal {
                    return None;
                }
            } else {
                asinh_expr = Some(asinh);
            }

            if let Some(existing) = &arg_poly {
                if existing != &arg {
                    return None;
                }
            } else {
                arg_poly = Some(arg);
            }

            asinh_cofactor_poly = asinh_cofactor_poly.add(&scaled_polynomial(&cofactor, scale));
            continue;
        }

        if let Some((term_scale, gap)) = sqrt_gap_term(ctx, term, var) {
            sqrt_scale += scale * term_scale;
            if sqrt_gap.replace(gap).is_some() {
                return None;
            }
            continue;
        }

        return None;
    }

    let asinh_expr = asinh_expr?;
    let arg_poly = arg_poly?;
    let sqrt_gap = sqrt_gap?;
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let expected_asinh_cofactor = arg_poly.div_scalar(&slope);
    if asinh_cofactor_poly != expected_asinh_cofactor {
        return None;
    }

    let expected_sqrt_scale = -BigRational::one() / slope;
    if sqrt_scale != expected_sqrt_scale {
        return None;
    }

    let one_poly = Polynomial::one(var.to_string());
    let expected_gap = arg_poly.mul(&arg_poly).add(&one_poly);
    if sqrt_gap != expected_gap {
        return None;
    }

    Some(asinh_expr)
}

fn acosh_affine_by_parts_derivative(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_add_terms(ctx, expr, BigRational::one(), &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut acosh_expr = None;
    let mut arg_poly = None;
    let mut acosh_cofactor_poly = Polynomial::zero(var.to_string());
    let mut sqrt_scale = BigRational::zero();
    let mut sqrt_pair = None;
    let mut sqrt_gap = None;

    for (scale, term) in terms {
        if let Some((acosh, arg, cofactor)) = acosh_affine_term(ctx, term, var) {
            if let Some(existing) = acosh_expr {
                if compare_expr(ctx, existing, acosh) != Ordering::Equal {
                    return None;
                }
            } else {
                acosh_expr = Some(acosh);
            }

            if let Some(existing) = &arg_poly {
                if existing != &arg {
                    return None;
                }
            } else {
                arg_poly = Some(arg);
            }

            acosh_cofactor_poly = acosh_cofactor_poly.add(&scaled_polynomial(&cofactor, scale));
            continue;
        }

        if let Some((term_scale, left, right)) = sqrt_product_pair_term(ctx, term, var) {
            sqrt_scale += scale * term_scale;
            if sqrt_pair.replace((left, right)).is_some() {
                return None;
            }
            continue;
        }

        if let Some((term_scale, gap)) = sqrt_gap_term(ctx, term, var) {
            sqrt_scale += scale * term_scale;
            if sqrt_gap.replace(gap).is_some() {
                return None;
            }
            continue;
        }

        return None;
    }

    let acosh_expr = acosh_expr?;
    let arg_poly = arg_poly?;
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let expected_acosh_cofactor = arg_poly.div_scalar(&slope);
    if acosh_cofactor_poly != expected_acosh_cofactor {
        return None;
    }

    let expected_sqrt_scale = -BigRational::one() / slope.clone();
    if sqrt_scale != expected_sqrt_scale {
        return None;
    }

    let one_poly = Polynomial::one(var.to_string());
    if let Some((sqrt_left, sqrt_right)) = sqrt_pair {
        let expected_left = arg_poly.sub(&one_poly);
        let expected_right = arg_poly.add(&one_poly);
        let same_order = sqrt_left == expected_left && sqrt_right == expected_right;
        let swapped_order = sqrt_left == expected_right && sqrt_right == expected_left;
        if !same_order && !swapped_order {
            return None;
        }
    } else if let Some(sqrt_gap) = sqrt_gap {
        let expected_gap = arg_poly.mul(&arg_poly).sub(&one_poly);
        if sqrt_gap != expected_gap {
            return None;
        }
    } else {
        return None;
    }

    Some(acosh_expr)
}

fn hyperbolic_linear_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    unary_builtin_arg(ctx, expr, BuiltinFn::Sinh)
        .map(|arg| (BuiltinFn::Sinh, arg))
        .or_else(|| unary_builtin_arg(ctx, expr, BuiltinFn::Cosh).map(|arg| (BuiltinFn::Cosh, arg)))
}

fn polynomial_times_builtin(ctx: &mut Context, poly: &Polynomial, builtin_expr: ExprId) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    let poly_expr = poly.to_expr(ctx);
    mul_pruned(ctx, poly_expr, builtin_expr)
}

fn try_linear_times_hyperbolic_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (hyperbolic_index, factor) in factors.iter().copied().enumerate() {
        let Some((builtin, arg)) = hyperbolic_linear_factor(ctx, factor) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != hyperbolic_index).then_some(factor))
            .collect();
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        let companion_builtin = match builtin {
            BuiltinFn::Sinh => BuiltinFn::Cosh,
            BuiltinFn::Cosh => BuiltinFn::Sinh,
            _ => return None,
        };
        let companion = ctx.call_builtin(companion_builtin, vec![arg]);
        let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, factor);
        let term_from_chain = polynomial_times_builtin(ctx, &scaled_cofactor, companion);

        return Some(add_pruned(ctx, term_from_cofactor, term_from_chain));
    }

    None
}

fn try_linear_times_cot_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (cot_index, factor) in factors.iter().copied().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Cot) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != cot_index).then_some(factor))
            .collect();
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, factor);
        let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
        let numerator = scaled_cofactor.to_expr(ctx);
        let term_from_chain = div_pruned(ctx, numerator, sin_sq);

        return Some(sub_pruned(ctx, term_from_cofactor, term_from_chain));
    }

    None
}

fn try_linear_times_arctan_reciprocal_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (arctan_index, factor) in factors.iter().copied().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Arctan)
            .or_else(|| unary_builtin_arg(ctx, factor, BuiltinFn::Atan))
        else {
            continue;
        };
        let Some(reciprocal_base) = unit_reciprocal_base(ctx, arg) else {
            continue;
        };

        let base_poly = Polynomial::from_expr(ctx, reciprocal_base, var).ok()?;
        if base_poly.degree() != 1 {
            continue;
        }
        let base_slope = base_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if base_slope.is_zero() {
            continue;
        }

        let mut cofactor_poly = Polynomial::new(vec![BigRational::one()], var.to_string());
        for factor in factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != arctan_index).then_some(factor))
        {
            let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
            cofactor_poly = cofactor_poly.mul(&factor_poly);
        }
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![base_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, factor);
        let two = ctx.num(2);
        let base_sq = ctx.add(Expr::Pow(reciprocal_base, two));
        let one = ctx.num(1);
        let denominator = add_pruned(ctx, base_sq, one);
        let numerator = scaled_cofactor.to_expr(ctx);
        let term_from_chain = div_pruned(ctx, numerator, denominator);

        return Some(sub_pruned(ctx, term_from_cofactor, term_from_chain));
    }

    None
}

fn try_linear_times_tanh_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (tanh_index, factor) in factors.iter().copied().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Tanh) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != tanh_index).then_some(factor))
            .collect();
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, factor);
        let cosh_sq = squared_builtin_call(ctx, BuiltinFn::Cosh, arg);
        let numerator = scaled_cofactor.to_expr(ctx);
        let term_from_chain = div_pruned(ctx, numerator, cosh_sq);

        return Some(add_pruned(ctx, term_from_cofactor, term_from_chain));
    }

    None
}

fn sec_csc_linear_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    unary_builtin_arg(ctx, expr, BuiltinFn::Sec)
        .map(|arg| (BuiltinFn::Sec, arg))
        .or_else(|| unary_builtin_arg(ctx, expr, BuiltinFn::Csc).map(|arg| (BuiltinFn::Csc, arg)))
}

fn try_linear_times_sec_csc_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (trig_index, factor) in factors.iter().copied().enumerate() {
        let Some((builtin, arg)) = sec_csc_linear_factor(ctx, factor) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != trig_index).then_some(factor))
            .collect();
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        return match builtin {
            BuiltinFn::Sec => {
                let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                let cos_sq = squared_builtin_call(ctx, BuiltinFn::Cos, arg);
                let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, cos_u);
                let term_from_chain = polynomial_times_builtin(ctx, &scaled_cofactor, sin_u);
                let numerator = add_pruned(ctx, term_from_cofactor, term_from_chain);
                Some(div_pruned(ctx, numerator, cos_sq))
            }
            BuiltinFn::Csc => {
                let term_from_cofactor =
                    polynomial_times_builtin(ctx, &cofactor_derivative, factor);
                let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
                let scaled_expr = scaled_cofactor.to_expr(ctx);
                let numerator = mul_pruned(ctx, cos_u, scaled_expr);
                let term_from_chain = div_pruned(ctx, numerator, sin_sq);
                Some(sub_pruned(ctx, term_from_cofactor, term_from_chain))
            }
            _ => None,
        };
    }

    None
}

fn one_minus_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![inner])
}

fn should_preserve_asinh_shifted_linear_radicand(ctx: &Context, arg: ExprId, var: &str) -> bool {
    if !matches!(ctx.get(arg), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let Ok(poly) = Polynomial::from_expr(ctx, arg, var) else {
        return false;
    };
    if poly.degree() != 1 {
        return false;
    }

    let offset = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    !offset.is_zero() && !slope.is_zero()
}

fn positive_scaled_arg(ctx: &Context, arg: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(arg) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(scale) = positive_rational_const(ctx, l) {
                Some((scale, r))
            } else {
                positive_rational_const(ctx, r).map(|scale| (scale, l))
            }
        }
        Expr::Div(num, den) => {
            let den_value = positive_rational_const(ctx, *den)?;
            Some((reciprocal_positive_rational(&den_value), *num))
        }
        _ => None,
    }
}

fn positive_rational_const(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = cas_ast::views::as_rational_const(ctx, expr, 4)?;
    value.is_positive().then_some(value)
}

fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

#[derive(Clone)]
struct SqrtRationalScale {
    rational: BigRational,
    sqrt_radicand: BigRational,
}

impl SqrtRationalScale {
    fn one() -> Self {
        Self {
            rational: BigRational::one(),
            sqrt_radicand: BigRational::one(),
        }
    }

    fn mul(&mut self, other: &Self) {
        self.rational *= other.rational.clone();
        self.sqrt_radicand *= other.sqrt_radicand.clone();
    }

    fn mul_rational(&mut self, value: BigRational) {
        self.rational *= value;
    }

    fn reciprocal(&self) -> Option<Self> {
        if self.rational.is_zero() || self.sqrt_radicand.is_zero() {
            return None;
        }

        Some(Self {
            rational: BigRational::one() / (&self.rational * &self.sqrt_radicand),
            sqrt_radicand: self.sqrt_radicand.clone(),
        })
    }

    fn square_rational(&self) -> BigRational {
        &self.rational * &self.rational * &self.sqrt_radicand
    }
}

fn exact_positive_rational_sqrt(value: &BigRational) -> Option<BigRational> {
    if !value.is_positive() {
        return None;
    }

    let numer_sqrt = value.numer().sqrt();
    let denom_sqrt = value.denom().sqrt();
    if &numer_sqrt * &numer_sqrt == *value.numer() && &denom_sqrt * &denom_sqrt == *value.denom() {
        Some(BigRational::new(numer_sqrt, denom_sqrt))
    } else {
        None
    }
}

fn sqrt_rational_scale_factor(ctx: &Context, expr: ExprId) -> Option<SqrtRationalScale> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if value.is_zero() {
            return None;
        }
        return Some(SqrtRationalScale {
            rational: value,
            sqrt_radicand: BigRational::one(),
        });
    }

    let radicand = extract_square_root_base(ctx, expr)?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    radicand_value.is_positive().then_some(SqrtRationalScale {
        rational: BigRational::one(),
        sqrt_radicand: radicand_value,
    })
}

fn scale_expr_by_sqrt_rational(
    ctx: &mut Context,
    mut scale: SqrtRationalScale,
    expr: ExprId,
) -> ExprId {
    if scale.rational.is_zero() {
        return ctx.num(0);
    }

    if let Some(sqrt_value) = exact_positive_rational_sqrt(&scale.sqrt_radicand) {
        scale.rational *= sqrt_value;
        scale.sqrt_radicand = BigRational::one();
    }

    if scale.sqrt_radicand.is_one() {
        return scale_expr_by_rational(ctx, scale.rational, expr);
    }

    let mut factors = Vec::new();
    if !scale.rational.is_one() {
        factors.push(ctx.add(Expr::Number(scale.rational)));
    }
    let radicand = ctx.add(Expr::Number(scale.sqrt_radicand));
    factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    factors.push(expr);
    build_balanced_mul(ctx, &factors)
}

fn scaled_sqrt_polynomial_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(SqrtRationalScale, ExprId, Polynomial)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        if contains_named_var(ctx, den, var) {
            return None;
        }
        let den_scale = sqrt_rational_scale_factor(ctx, den)?;
        let (mut scale, poly_expr, poly) = scaled_sqrt_polynomial_arg(ctx, num, var)?;
        scale.mul(&den_scale.reciprocal()?);
        return Some((scale, poly_expr, poly));
    }

    let factors = mul_leaves(ctx, expr);
    let mut scale = SqrtRationalScale::one();
    let mut poly_expr = None;
    let mut poly = None;

    for factor in factors {
        if !contains_named_var(ctx, factor, var) {
            let factor_scale = sqrt_rational_scale_factor(ctx, factor)?;
            scale.mul(&factor_scale);
            continue;
        }

        if poly_expr.is_some() {
            return None;
        }
        let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        if factor_poly.is_zero() {
            return None;
        }
        poly_expr = Some(factor);
        poly = Some(factor_poly);
    }

    Some((scale, poly_expr?, poly?))
}

fn compact_polynomial_square_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let factored = crate::factor::factor(ctx, expr);
    let two = ctx.num(2);
    if let Expr::Pow(base, exp) = ctx.get(factored).clone() {
        if let Expr::Number(n) = ctx.get(exp) {
            if n.is_integer() && n.is_positive() {
                let doubled = BigRational::from_integer(n.to_integer() * 2);
                let doubled = ctx.add(Expr::Number(doubled));
                return ctx.add(Expr::Pow(base, doubled));
            }
        }
    }
    ctx.add(Expr::Pow(factored, two))
}

fn constant_scaled_atanh_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(SqrtRationalScale, ExprId)> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Atanh) {
        return Some((SqrtRationalScale::one(), arg));
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (mut scale, arg) = constant_scaled_atanh_parts(ctx, inner, var)?;
            scale.mul_rational(-BigRational::one());
            Some((scale, arg))
        }
        Expr::Div(num, den) => {
            if contains_named_var(ctx, den, var) {
                return None;
            }
            let (mut scale, arg) = constant_scaled_atanh_parts(ctx, num, var)?;
            let den_scale = sqrt_rational_scale_factor(ctx, den)?;
            scale.mul(&den_scale.reciprocal()?);
            Some((scale, arg))
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let mut outer_scale = SqrtRationalScale::one();
            let mut atanh_arg = None;

            for factor in factors {
                if let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Atanh) {
                    if atanh_arg.replace(arg).is_some() {
                        return None;
                    }
                    continue;
                }

                if contains_named_var(ctx, factor, var) {
                    return None;
                }
                let factor_scale = sqrt_rational_scale_factor(ctx, factor)?;
                outer_scale.mul(&factor_scale);
            }

            Some((outer_scale, atanh_arg?))
        }
        _ => None,
    }
}

fn try_constant_scaled_atanh_polynomial_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_scale, atanh_arg) = constant_scaled_atanh_parts(ctx, expr, var)?;
    let (arg_scale, poly_expr, poly) = scaled_sqrt_polynomial_arg(ctx, atanh_arg, var)?;
    if arg_scale.rational.is_one() && arg_scale.sqrt_radicand.is_one() {
        return None;
    }
    let derivative_poly = poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let arg_scale_squared = arg_scale.square_rational();
    if arg_scale_squared.is_zero() {
        return None;
    }

    let denominator_scale = BigRational::from_integer(arg_scale_squared.denom().clone());
    let scaled_square_coeff = BigRational::from_integer(arg_scale_squared.numer().clone());

    let mut numerator_scale = outer_scale;
    numerator_scale.mul(&arg_scale);
    numerator_scale.mul_rational(denominator_scale.clone());

    let derivative_expr = derivative_poly.to_expr(ctx);
    let numerator = scale_expr_by_sqrt_rational(ctx, numerator_scale, derivative_expr);

    let denominator_constant = ctx.add(Expr::Number(denominator_scale));
    let poly_square = compact_polynomial_square_expr(ctx, poly_expr);
    let scaled_poly_square = scale_expr_by_rational(ctx, scaled_square_coeff, poly_square);
    let denominator = sub_pruned(ctx, denominator_constant, scaled_poly_square);

    Some(div_pruned(ctx, numerator, denominator))
}

fn constant_scaled_bounded_inverse_trig_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(SqrtRationalScale, i32, ExprId)> {
    if let Some((sign, arg)) = unary_bounded_inverse_trig_arg(ctx, expr) {
        return Some((SqrtRationalScale::one(), sign, arg));
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (mut scale, sign, arg) =
                constant_scaled_bounded_inverse_trig_parts(ctx, inner, var)?;
            scale.mul_rational(-BigRational::one());
            Some((scale, sign, arg))
        }
        Expr::Div(num, den) => {
            if contains_named_var(ctx, den, var) {
                return None;
            }
            let (mut scale, sign, arg) = constant_scaled_bounded_inverse_trig_parts(ctx, num, var)?;
            let den_scale = sqrt_rational_scale_factor(ctx, den)?;
            scale.mul(&den_scale.reciprocal()?);
            Some((scale, sign, arg))
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let mut outer_scale = SqrtRationalScale::one();
            let mut inverse_trig = None;

            for factor in factors {
                if let Some((sign, arg)) = unary_bounded_inverse_trig_arg(ctx, factor) {
                    if inverse_trig.replace((sign, arg)).is_some() {
                        return None;
                    }
                    continue;
                }

                if contains_named_var(ctx, factor, var) {
                    return None;
                }
                let factor_scale = sqrt_rational_scale_factor(ctx, factor)?;
                outer_scale.mul(&factor_scale);
            }

            let (sign, arg) = inverse_trig?;
            Some((outer_scale, sign, arg))
        }
        _ => None,
    }
}

fn unary_bounded_inverse_trig_arg(ctx: &Context, expr: ExprId) -> Option<(i32, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => Some((1, args[0])),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => Some((-1, args[0])),
        _ => None,
    }
}

fn try_constant_scaled_bounded_inverse_trig_polynomial_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_scale, sign, arg) = constant_scaled_bounded_inverse_trig_parts(ctx, expr, var)?;
    let (arg_scale, poly_expr, poly) = scaled_sqrt_polynomial_arg(ctx, arg, var)?;
    if arg_scale.sqrt_radicand.is_one() && outer_scale.sqrt_radicand.is_one() {
        return None;
    }

    let derivative_poly = poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let arg_scale_squared = arg_scale.square_rational();
    if arg_scale_squared.is_zero() {
        return None;
    }

    let denominator_scale = BigRational::from_integer(arg_scale_squared.denom().clone());
    let scaled_square_coeff = BigRational::from_integer(arg_scale_squared.numer().clone());

    let mut numerator_scale = outer_scale;
    numerator_scale.mul(&arg_scale);
    numerator_scale.mul(&SqrtRationalScale {
        rational: BigRational::one(),
        sqrt_radicand: denominator_scale.clone(),
    });
    if sign < 0 {
        numerator_scale.mul_rational(-BigRational::one());
    }

    let derivative_expr = derivative_poly.to_expr(ctx);
    let numerator = scale_expr_by_sqrt_rational(ctx, numerator_scale, derivative_expr);

    let denominator_constant = ctx.add(Expr::Number(denominator_scale));
    let poly_square = compact_polynomial_square_expr(ctx, poly_expr);
    let scaled_poly_square = scale_expr_by_rational(ctx, scaled_square_coeff, poly_square);
    let denominator_radicand = sub_pruned(ctx, denominator_constant, scaled_poly_square);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![denominator_radicand]);

    Some(div_pruned(ctx, numerator, denominator))
}

fn scaled_one_minus_arg_square_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
) -> Option<ExprId> {
    let (scale, inner) = positive_scaled_arg(ctx, arg)?;
    if scale.is_one() {
        return None;
    }

    let scale_squared = &scale * &scale;
    let offset = BigRational::one() / scale_squared;
    let two = ctx.num(2);
    let inner_sq = ctx.add(Expr::Pow(inner, two));
    let offset_expr = ctx.add(Expr::Number(offset));
    let radicand = ctx.add(Expr::Sub(offset_expr, inner_sq));
    let den = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_scale = ctx.add(Expr::Number(reciprocal_positive_rational(&scale)));
    let numerator = mul_pruned(ctx, d_arg, numerator_scale);
    Some(ctx.add(Expr::Div(numerator, den)))
}

fn unit_reciprocal_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if is_one(ctx, *num) => Some(*den),
        Expr::Pow(base, exp)
            if matches!(
                ctx.get(*exp),
                Expr::Number(n) if n.is_integer() && n.to_integer() == (-1).into()
            ) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn surd_numerator_reciprocal_base(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let radicand = extract_square_root_base(ctx, num)?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    Some(ctx.add(Expr::Div(den, num)))
}

fn one_minus_reciprocal_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> (ExprId, ExprId) {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let reciprocal_arg_sq = ctx.add(Expr::Div(one, arg_sq));
    let inner = ctx.add(Expr::Sub(one, reciprocal_arg_sq));
    (ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]), arg_sq)
}

fn quotient_over_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = extract_square_root_base(ctx, den) {
                return Some((num, radicand));
            }

            let den_value = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if !den_value.is_positive() {
                return None;
            }

            let factors = mul_leaves(ctx, num);
            for (index, factor) in factors.iter().enumerate() {
                let Some(radicand) = extract_square_root_base(ctx, *factor) else {
                    continue;
                };
                let Some(radicand_value) = cas_ast::views::as_rational_const(ctx, radicand, 8)
                else {
                    continue;
                };
                if radicand_value != den_value {
                    continue;
                }

                let remaining: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(factor_index, factor)| (factor_index != index).then_some(*factor))
                    .collect();
                return Some((build_balanced_mul(ctx, &remaining), radicand));
            }

            None
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, arg);
            for (index, factor) in factors.iter().enumerate() {
                let Some(sqrt_radicand) = extract_square_root_base(ctx, *factor) else {
                    continue;
                };
                let sqrt_radicand_value = cas_ast::views::as_rational_const(ctx, sqrt_radicand, 8)?;
                if !sqrt_radicand_value.is_positive() {
                    return None;
                }

                let remaining: Vec<_> = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(factor_index, factor)| (factor_index != index).then_some(*factor))
                    .collect();
                if remaining.is_empty() {
                    return None;
                }

                let reciprocal_radicand = ctx.add(Expr::Number(reciprocal_positive_rational(
                    &sqrt_radicand_value,
                )));
                return Some((build_balanced_mul(ctx, &remaining), reciprocal_radicand));
            }

            None
        }
        _ => None,
    }
}

fn negative_half_power(ctx: &mut Context, base: ExprId) -> ExprId {
    let minus_one = ctx.num(-1);
    let two = ctx.num(2);
    let neg_half = ctx.add(Expr::Div(minus_one, two));
    ctx.add(Expr::Pow(base, neg_half))
}

fn is_positive_half_exponent(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8)
        .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
}

fn sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    extract_square_root_base(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Pow(base, exp) if is_positive_half_exponent(ctx, *exp) => Some(*base),
        _ => None,
    })
}

fn sqrt_linear_chain_coeff(ctx: &Context, radicand: ExprId, var: &str) -> Option<BigRational> {
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let derivative = radicand_poly.derivative();
    if derivative.is_zero() || derivative.degree() != 0 {
        return None;
    }
    let derivative_coeff = derivative
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    Some(derivative_coeff / BigRational::new(2.into(), 1.into()))
}

fn rational_numerator_denominator(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }
    Some((
        BigRational::from_integer(value.numer().clone()),
        BigRational::from_integer(value.denom().clone()),
    ))
}

fn compact_sqrt_log_abs_trig_derivative(
    ctx: &mut Context,
    inner_expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (derivative_builtin, sign, trig_arg) =
        if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sin) {
            (BuiltinFn::Cot, BigRational::one(), arg)
        } else if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cos) {
            (BuiltinFn::Tan, -BigRational::one(), arg)
        } else {
            return None;
        };

    let radicand = sqrt_like_radicand(ctx, trig_arg)?;
    let chain_coeff = sign * sqrt_linear_chain_coeff(ctx, radicand, var)?;
    let negative = chain_coeff.is_negative();
    let (numerator_coeff, denominator_coeff) = rational_numerator_denominator(&chain_coeff.abs())?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig = ctx.call_builtin(derivative_builtin, vec![sqrt_radicand]);
    let numerator = scale_expr_by_rational(ctx, numerator_coeff, trig);
    let denominator = if denominator_coeff.is_one() {
        sqrt_radicand
    } else {
        let denominator_coeff = ctx.add(Expr::Number(denominator_coeff));
        mul_pruned(ctx, denominator_coeff, sqrt_radicand)
    };

    let quotient = div_pruned(ctx, numerator, denominator);
    Some(if negative {
        ctx.add(Expr::Neg(quotient))
    } else {
        quotient
    })
}

fn compact_sqrt_log_hyperbolic_derivative(
    ctx: &mut Context,
    inner_expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    enum HyperbolicLogShape {
        TanhNumerator,
        TanhDenominator,
    }

    let (shape, hyperbolic_arg) =
        if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cosh) {
            (HyperbolicLogShape::TanhNumerator, arg)
        } else if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sinh) {
            (HyperbolicLogShape::TanhDenominator, arg)
        } else {
            return None;
        };

    let radicand = sqrt_like_radicand(ctx, hyperbolic_arg)?;
    let chain_coeff = sqrt_linear_chain_coeff(ctx, radicand, var)?;
    let negative = chain_coeff.is_negative();
    let (numerator_coeff, denominator_coeff) = rational_numerator_denominator(&chain_coeff.abs())?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![sqrt_radicand]);
    let one = ctx.num(1);

    let quotient = match shape {
        HyperbolicLogShape::TanhNumerator => {
            let numerator = scale_expr_by_rational(ctx, numerator_coeff, tanh);
            let denominator = if denominator_coeff.is_one() {
                sqrt_radicand
            } else {
                let denominator_coeff = ctx.add(Expr::Number(denominator_coeff));
                mul_pruned(ctx, denominator_coeff, sqrt_radicand)
            };
            div_pruned(ctx, numerator, denominator)
        }
        HyperbolicLogShape::TanhDenominator => {
            let numerator = scale_expr_by_rational(ctx, numerator_coeff, one);
            let denominator_core = build_balanced_mul(ctx, &[tanh, sqrt_radicand]);
            let denominator = if denominator_coeff.is_one() {
                denominator_core
            } else {
                let denominator_coeff = ctx.add(Expr::Number(denominator_coeff));
                build_balanced_mul(ctx, &[denominator_coeff, denominator_core])
            };
            div_pruned(ctx, numerator, denominator)
        }
    };

    Some(if negative {
        ctx.add(Expr::Neg(quotient))
    } else {
        quotient
    })
}

fn value_preserving_primitive_gap(
    ctx: &mut Context,
    gap: ExprId,
    sqrt_radicand_value: &BigRational,
) -> (ExprId, BigRational) {
    use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let Ok(poly) = multipoly_from_expr(ctx, gap, &budget) else {
        return (gap, sqrt_radicand_value.clone());
    };
    let (content, primitive) = poly.primitive_part();
    if !content.is_positive() {
        return (gap, sqrt_radicand_value.clone());
    }
    let primitive_expr = multipoly_to_expr(&primitive, ctx);
    if content.is_one() {
        return (primitive_expr, sqrt_radicand_value.clone());
    }

    (primitive_expr, sqrt_radicand_value / content)
}

fn is_intrinsically_positive_real(ctx: &Context, expr: ExprId) -> bool {
    prove_positive_depth_with(
        ctx,
        expr,
        SYMBOLIC_DIFF_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn is_positive_integer_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => n.is_integer() && n.is_positive(),
            _ => false,
        },
        _ => false,
    }
}

fn is_positive_even_integer_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => n.is_integer() && n.is_positive() && n.to_integer().is_even(),
            _ => false,
        },
        _ => false,
    }
}

fn acosh_left_radicand(ctx: &mut Context, arg: ExprId) -> ExprId {
    match ctx.get(arg) {
        Expr::Add(left, right) if is_one(ctx, *right) => *left,
        Expr::Add(left, right) if is_one(ctx, *left) => *right,
        _ => {
            let one = ctx.num(1);
            ctx.add(Expr::Sub(arg, one))
        }
    }
}

fn has_perfect_square_sqrt_rewrite(ctx: &mut Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return false,
    }

    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]);
    try_rewrite_simplify_square_root_expr(ctx, sqrt_expr)
        .is_some_and(|rewrite| rewrite.kind == SimplifySquareRootRewriteKind::PerfectSquare)
}

fn positive_arg_arcsec_like_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    sign: i64,
) -> Option<ExprId> {
    if is_positive_integer_power(ctx, arg) || !is_intrinsically_positive_real(ctx, arg) {
        return None;
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let gap = ctx.add(Expr::Sub(arg_sq, one));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = mul_pruned(ctx, arg, sqrt_gap);
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(d_arg))
    } else {
        d_arg
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn positive_surd_quotient_arcsec_like_derivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    sign: i64,
) -> Option<ExprId> {
    let (num, radicand) = quotient_over_sqrt_parts(ctx, arg)?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive()
        || is_positive_integer_power(ctx, num)
        || !is_intrinsically_positive_real(ctx, num)
    {
        return None;
    }

    let d_num = differentiate_symbolic_expr(ctx, num, var)?;
    let two = ctx.num(2);
    let num_sq = ctx.add(Expr::Pow(num, two));
    let gap = ctx.add(Expr::Sub(num_sq, radicand));
    let (gap, sqrt_radicand_value) = value_preserving_primitive_gap(ctx, gap, &radicand_value);
    let sqrt_radicand = ctx.add(Expr::Number(sqrt_radicand_value));
    let sqrt_radicand = if is_one(ctx, sqrt_radicand) {
        None
    } else {
        Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_radicand]))
    };
    let reciprocal_gap_root = negative_half_power(ctx, gap);

    let numerator_factors = if let Some(sqrt_radicand) = sqrt_radicand {
        vec![sqrt_radicand, d_num, reciprocal_gap_root]
    } else {
        vec![d_num, reciprocal_gap_root]
    };
    let mut numerator = build_balanced_mul(ctx, &numerator_factors);
    if sign < 0 {
        numerator = ctx.add(Expr::Neg(numerator));
    }

    Some(div_pruned(ctx, numerator, num))
}

fn arcsec_like_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    var: &str,
    sign: i64,
) -> ExprId {
    if let Some(derivative) = positive_surd_quotient_arcsec_like_derivative(ctx, arg, var, sign) {
        return derivative;
    }
    if let Some(derivative) = positive_arg_arcsec_like_derivative(ctx, arg, d_arg, sign) {
        return derivative;
    }

    let one = ctx.num(1);
    let (sqrt_gap, arg_sq) = one_minus_reciprocal_arg_square_sqrt(ctx, arg);
    let numerator = mul_pruned(ctx, d_arg, sqrt_gap);
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(numerator))
    } else {
        numerator
    };
    let denominator = ctx.add(Expr::Sub(arg_sq, one));
    ctx.add(Expr::Div(numerator, denominator))
}

fn arccot_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let denominator = ctx.add(Expr::Add(arg_sq, one));
    let numerator = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(numerator, denominator))
}

fn acosh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let left = acosh_left_radicand(ctx, arg);
    let one = ctx.num(1);
    let minus_one = ctx.num(-1);
    let two = ctx.num(2);
    let neg_half = ctx.add(Expr::Div(minus_one, two));
    let right = ctx.add(Expr::Add(arg, one));

    if is_positive_even_integer_power(ctx, left) || has_perfect_square_sqrt_rewrite(ctx, left) {
        let sqrt_left = ctx.call_builtin(BuiltinFn::Sqrt, vec![left]);
        let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
        let denominator = mul_pruned(ctx, sqrt_left, sqrt_right);
        return div_pruned(ctx, d_arg, denominator);
    }

    let left_inv_sqrt = ctx.add(Expr::Pow(left, neg_half));
    let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
    let numerator = mul_pruned(ctx, d_arg, left_inv_sqrt);
    ctx.add(Expr::Div(numerator, sqrt_right))
}

fn atanh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let minus_two = ctx.num(-2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    let sqrt_inner = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
    let inv_square = ctx.add(Expr::Pow(sqrt_inner, minus_two));
    mul_pruned(ctx, d_arg, inv_square)
}

fn constant_base_log_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    base: i64,
) -> ExprId {
    let base = ctx.num(base);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let den = mul_pruned(ctx, arg, ln_base);
    ctx.add(Expr::Div(d_arg, den))
}

#[derive(Clone, Copy)]
enum ReciprocalTrigDerivativeKind {
    Sec,
    Csc,
    Cot,
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn canonical_reciprocal_trig_div_kind(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ReciprocalTrigDerivativeKind, ExprId)> {
    if is_one(ctx, num) {
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Cos) {
            return Some((ReciprocalTrigDerivativeKind::Sec, arg));
        }
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Sin) {
            return Some((ReciprocalTrigDerivativeKind::Csc, arg));
        }
    }

    let cos_arg = unary_builtin_arg(ctx, num, BuiltinFn::Cos)?;
    let sin_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sin)?;
    if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
        Some((ReciprocalTrigDerivativeKind::Cot, cos_arg))
    } else {
        None
    }
}

fn canonical_hyperbolic_coth_div_arg(ctx: &Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    let cosh_arg = unary_builtin_arg(ctx, num, BuiltinFn::Cosh)?;
    let sinh_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sinh)?;
    (compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal).then_some(cosh_arg)
}

fn constant_scaled_unary_builtin_denominator(
    ctx: &mut Context,
    den: ExprId,
    builtin: BuiltinFn,
) -> Option<(BigRational, ExprId)> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut builtin_arg = None;

    for factor in factors {
        if builtin_arg.is_none() {
            if let Some(arg) = unary_builtin_arg(ctx, factor, builtin) {
                builtin_arg = Some(arg);
                continue;
            }
        }

        let factor_scale = cas_ast::views::as_rational_const(ctx, factor, 4)?;
        scale *= factor_scale;
    }

    if scale.is_zero() {
        return None;
    }

    builtin_arg.map(|arg| (scale, arg))
}

fn constant_scaled_polynomial_power_denominator(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(BigRational, ExprId, i64)> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut power_base_and_exp = None;

    for factor in factors {
        if power_base_and_exp.is_none() {
            if let Expr::Pow(base, exp) = *ctx.get(factor) {
                if let Some(power) = crate::numeric::as_i64(ctx, exp) {
                    if (2..=8).contains(&power) {
                        let poly = Polynomial::from_expr(ctx, base, var).ok()?;
                        if (1..=2).contains(&poly.degree()) {
                            power_base_and_exp = Some((base, power));
                            continue;
                        }
                    }
                }
            }
        }

        let factor_scale = cas_ast::views::as_rational_const(ctx, factor, 4)?;
        scale *= factor_scale;
    }

    if scale.is_zero() {
        return None;
    }

    power_base_and_exp.map(|(base, power)| (scale, base, power))
}

fn scale_expr_by_rational(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale.is_zero() {
        return ctx.num(0);
    }
    if scale.is_one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(expr));
    }

    if let Some(expr_scale) = cas_ast::views::as_rational_const(ctx, expr, 4) {
        return ctx.add(Expr::Number(scale * expr_scale));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    mul_pruned(ctx, scale_expr, expr)
}

fn non_integer_rational_constant_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let scale = rational_constant_value(ctx, expr)?;
    (!scale.denom().is_one()).then_some(scale)
}

fn scaled_square_denominator_ratio(
    ctx: &mut Context,
    scale: BigRational,
    numerator_core: ExprId,
    denominator_square: ExprId,
) -> ExprId {
    if scale.is_zero() {
        return ctx.num(0);
    }

    let numerator_scale = BigRational::from_integer(scale.numer().clone());
    let denominator_scale = BigRational::from_integer(scale.denom().clone());
    let numerator = scale_expr_by_rational(ctx, numerator_scale, numerator_core);
    let denominator = scale_expr_by_rational(ctx, denominator_scale, denominator_square);
    div_pruned(ctx, numerator, denominator)
}

fn cosh_arg_cofactor(ctx: &mut Context, expr: ExprId, arg: ExprId) -> Option<ExprId> {
    if let Some(cosh_arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Cosh) {
        if compare_expr(ctx, cosh_arg, arg) == Ordering::Equal {
            return Some(ctx.num(1));
        }
    }

    if let Expr::Neg(inner) = *ctx.get(expr) {
        let cofactor = cosh_arg_cofactor(ctx, inner, arg)?;
        return Some(ctx.add(Expr::Neg(cofactor)));
    }

    let factors = mul_leaves(ctx, expr);
    let cosh_index = factors.iter().position(|factor| {
        unary_builtin_arg(ctx, *factor, BuiltinFn::Cosh)
            .is_some_and(|cosh_arg| compare_expr(ctx, cosh_arg, arg) == Ordering::Equal)
    })?;
    let cofactor_factors: Vec<_> = factors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != cosh_index).then_some(factor))
        .collect();

    Some(build_balanced_mul(ctx, &cofactor_factors))
}

fn cosh_arg_linear_cofactor(ctx: &mut Context, num: ExprId, arg: ExprId) -> Option<ExprId> {
    if let Some(cofactor) = cosh_arg_cofactor(ctx, num, arg) {
        return Some(cofactor);
    }

    let terms = add_leaves(ctx, num);
    if terms.len() < 2 {
        return None;
    }

    let mut cofactors = Vec::with_capacity(terms.len());
    for term in terms {
        cofactors.push(cosh_arg_cofactor(ctx, term, arg)?);
    }
    Some(build_balanced_add(ctx, &cofactors))
}

fn try_linear_times_hyperbolic_coth_linear_div_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let sinh_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sinh)?;
    let arg_poly = Polynomial::from_expr(ctx, sinh_arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let arg_slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if arg_slope.is_zero() {
        return None;
    }

    let cofactor = cosh_arg_linear_cofactor(ctx, num, sinh_arg)?;
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    if cofactor_poly.degree() > 1 {
        return None;
    }

    let cofactor_derivative = cofactor_poly.derivative();
    let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
    let scaled_cofactor = cofactor_poly.mul(&slope_poly);

    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![sinh_arg]);
    let quotient = div_pruned(ctx, cosh, den);
    let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, quotient);
    let sinh_sq = squared_builtin_call(ctx, BuiltinFn::Sinh, sinh_arg);
    let numerator = scaled_cofactor.to_expr(ctx);
    let term_from_chain = div_pruned(ctx, numerator, sinh_sq);

    Some(sub_pruned(ctx, term_from_cofactor, term_from_chain))
}

fn try_constant_scaled_hyperbolic_reciprocal_tanh_div_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 4)?;
    let (denominator_scale, arg) =
        constant_scaled_unary_builtin_denominator(ctx, den, BuiltinFn::Tanh)?;
    let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
    let scale = -(numerator_scale / denominator_scale);
    let numerator = scale_expr_by_rational(ctx, scale, d_arg);
    let sinh_sq = squared_builtin_call(ctx, BuiltinFn::Sinh, arg);
    Some(div_pruned(ctx, numerator, sinh_sq))
}

fn try_constant_scaled_hyperbolic_reciprocal_div_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 4)?;

    for (builtin, companion) in [
        (BuiltinFn::Sinh, BuiltinFn::Cosh),
        (BuiltinFn::Cosh, BuiltinFn::Sinh),
    ] {
        let Some((denominator_scale, arg)) =
            constant_scaled_unary_builtin_denominator(ctx, den, builtin)
        else {
            continue;
        };

        let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
        let scale = -(numerator_scale.clone() / denominator_scale);
        let scaled_d_arg = scale_expr_by_rational(ctx, scale, d_arg);
        let companion_arg = ctx.call_builtin(companion, vec![arg]);
        let numerator = mul_pruned(ctx, scaled_d_arg, companion_arg);
        let denominator = squared_builtin_call(ctx, builtin, arg);
        return Some(div_pruned(ctx, numerator, denominator));
    }

    None
}

fn try_constant_scaled_polynomial_reciprocal_power_div_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 4)?;
    let (denominator_scale, base, power) =
        constant_scaled_polynomial_power_denominator(ctx, den, var)?;
    let d_base = Polynomial::from_expr(ctx, base, var)
        .ok()?
        .derivative()
        .to_expr(ctx);
    let power_scale = BigRational::from_integer(power.into());
    let scale = -(numerator_scale * power_scale / denominator_scale);
    let numerator = scale_expr_by_rational(ctx, scale, d_base);
    let next_power = ctx.add(Expr::Number(BigRational::from_integer((power + 1).into())));
    let denominator = ctx.add(Expr::Pow(base, next_power));
    Some(div_pruned(ctx, numerator, denominator))
}

fn squared_builtin_call(ctx: &mut Context, builtin: BuiltinFn, arg: ExprId) -> ExprId {
    let call = ctx.call_builtin(builtin, vec![arg]);
    let two = ctx.num(2);
    ctx.add(Expr::Pow(call, two))
}

fn bounded_positive_integer_value(ctx: &Context, expr: ExprId, min: i64, max: i64) -> Option<i64> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    if !value.denom().is_one() {
        return None;
    }
    let value = value.to_integer().to_i64()?;
    (min..=max).contains(&value).then_some(value)
}

fn scaled_reciprocal_trig_power_primitive_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let denominator_scale = rational_constant_value(ctx, den)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let (outer_sign, core) = match ctx.get(num).clone() {
        Expr::Neg(inner) => (-BigRational::one(), inner),
        _ => (BigRational::one(), num),
    };
    let Expr::Pow(base, exp) = ctx.get(core).clone() else {
        return None;
    };
    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    let (reciprocal_square_builtin, derivative_sign) = match builtin {
        BuiltinFn::Tan => (BuiltinFn::Sec, BigRational::one()),
        BuiltinFn::Cot => (BuiltinFn::Csc, -BigRational::one()),
        _ => return None,
    };
    let power = bounded_positive_integer_value(ctx, exp, 2, 6)?;
    let arg = args[0];
    let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;

    let base_power = if power == 2 {
        ctx.call_builtin(builtin, vec![arg])
    } else {
        let base_call = ctx.call_builtin(builtin, vec![arg]);
        let next_power = ctx.num(power - 1);
        ctx.add(Expr::Pow(base_call, next_power))
    };
    let reciprocal_square = squared_builtin_call(ctx, reciprocal_square_builtin, arg);
    let product = mul_pruned(ctx, base_power, reciprocal_square);
    let product = mul_pruned(ctx, product, d_arg);
    let scale =
        outer_sign * derivative_sign * BigRational::from_integer(power.into()) / denominator_scale;

    Some(scale_expr_by_rational(ctx, scale, product))
}

fn scaled_trig_power_factor(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
    power: i64,
) -> Option<(BigRational, ExprId)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut arg = None;

    for factor in factors {
        let factor = match ctx.get(factor).clone() {
            Expr::Neg(inner) => {
                scale = -scale;
                inner
            }
            _ => factor,
        };
        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            scale *= rational_constant_value(ctx, factor)?;
            continue;
        };
        let (fn_id, args) = match ctx.get(base).clone() {
            Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
            _ => {
                scale *= rational_constant_value(ctx, factor)?;
                continue;
            }
        };
        if ctx.builtin_of(fn_id) != Some(builtin) {
            scale *= rational_constant_value(ctx, factor)?;
            continue;
        }
        if bounded_positive_integer_value(ctx, exp, power, power)? != power {
            return None;
        }
        if arg.replace(args[0]).is_some() {
            return None;
        }
    }

    Some((scale, arg?))
}

fn scaled_trig_quotient_power_derivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    for power in 2..=6 {
        for (num_builtin, den_builtin, primitive_builtin, reciprocal_square_builtin, sign) in [
            (
                BuiltinFn::Sin,
                BuiltinFn::Cos,
                BuiltinFn::Tan,
                BuiltinFn::Sec,
                BigRational::one(),
            ),
            (
                BuiltinFn::Cos,
                BuiltinFn::Sin,
                BuiltinFn::Cot,
                BuiltinFn::Csc,
                -BigRational::one(),
            ),
        ] {
            let Some((num_scale, num_arg)) = scaled_trig_power_factor(ctx, num, num_builtin, power)
            else {
                continue;
            };
            let Some((den_scale, den_arg)) = scaled_trig_power_factor(ctx, den, den_builtin, power)
            else {
                continue;
            };
            if den_scale.is_zero() || compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
                continue;
            }

            let d_arg = differentiate_symbolic_expr(ctx, num_arg, var)?;
            let base_power = if power == 2 {
                ctx.call_builtin(primitive_builtin, vec![num_arg])
            } else {
                let base = ctx.call_builtin(primitive_builtin, vec![num_arg]);
                let exp = ctx.num(power - 1);
                ctx.add(Expr::Pow(base, exp))
            };
            let reciprocal_square = squared_builtin_call(ctx, reciprocal_square_builtin, num_arg);
            let product = mul_pruned(ctx, base_power, reciprocal_square);
            let product = mul_pruned(ctx, product, d_arg);
            let scale = sign * BigRational::from_integer(power.into()) * num_scale / den_scale;
            return Some(scale_expr_by_rational(ctx, scale, product));
        }
    }

    None
}

fn secant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sec_u = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
    let tan_u = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let table_rule = mul_pruned(ctx, sec_u, tan_u);
    mul_pruned(ctx, d_arg, table_rule)
}

fn cosecant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let csc_u = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
    let cot_u = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let table_rule = mul_pruned(ctx, csc_u, cot_u);
    let negative_table_rule = ctx.add(Expr::Neg(table_rule));
    mul_pruned(ctx, d_arg, negative_table_rule)
}

fn cotangent_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
    if let Some(scale) = non_integer_rational_constant_value(ctx, d_arg) {
        let one = ctx.num(1);
        return scaled_square_denominator_ratio(ctx, -scale, one, sin_sq);
    }
    let neg_d_arg = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(neg_d_arg, sin_sq))
}

fn hyperbolic_coth_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sinh_sq = squared_builtin_call(ctx, BuiltinFn::Sinh, arg);
    let neg_d_arg = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(neg_d_arg, sinh_sq))
}

fn reciprocal_trig_derivative(
    ctx: &mut Context,
    kind: ReciprocalTrigDerivativeKind,
    arg: ExprId,
    d_arg: ExprId,
) -> ExprId {
    match kind {
        ReciprocalTrigDerivativeKind::Sec => secant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Csc => cosecant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Cot => cotangent_derivative(ctx, arg, d_arg),
    }
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = unary_builtin_arg(ctx, left, left_builtin)?;
    let right_arg = unary_builtin_arg(ctx, right, right_builtin)?;
    (compare_expr(ctx, left_arg, right_arg) == Ordering::Equal).then_some(left_arg)
}

fn unordered_same_arg_unary_sum(
    ctx: &Context,
    expr: ExprId,
    first_builtin: BuiltinFn,
    second_builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    same_arg_unary_pair(ctx, *left, first_builtin, *right, second_builtin)
        .or_else(|| same_arg_unary_pair(ctx, *right, first_builtin, *left, second_builtin))
}

fn csc_minus_cot_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, *right, BuiltinFn::Cot)
        }
        Expr::Add(left, right) => {
            let negated_right = match ctx.get(*right) {
                Expr::Neg(inner) => *inner,
                _ => return None,
            };
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, negated_right, BuiltinFn::Cot)
        }
        _ => None,
    }
}

fn add_one_sin_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_one(ctx, *left) {
        unary_builtin_arg(ctx, *right, BuiltinFn::Sin)
    } else if is_one(ctx, *right) {
        unary_builtin_arg(ctx, *left, BuiltinFn::Sin)
    } else {
        None
    }
}

fn plus_or_minus_one_cos_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_one(ctx, *left) => {
            unary_builtin_arg(ctx, *right, BuiltinFn::Cos)
        }
        Expr::Sub(left, right) if is_one(ctx, *right) => {
            unary_builtin_arg(ctx, *left, BuiltinFn::Cos)
        }
        _ => None,
    }
}

fn same_arg_quotient_over_builtin_denominator(
    ctx: &Context,
    expr: ExprId,
    numerator_arg: impl Fn(&Context, ExprId) -> Option<ExprId>,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num_arg = numerator_arg(ctx, *num)?;
    let den_arg = unary_builtin_arg(ctx, *den, denominator_builtin)?;
    (compare_expr(ctx, num_arg, den_arg) == Ordering::Equal).then_some(num_arg)
}

fn log_abs_reciprocal_trig_primitive_derivative(
    ctx: &mut Context,
    inner_expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(arg) = unordered_same_arg_unary_sum(ctx, inner_expr, BuiltinFn::Sec, BuiltinFn::Tan)
    {
        let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
        let sec_arg = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
        return Some(mul_pruned(ctx, d_arg, sec_arg));
    }

    if let Some(arg) = csc_minus_cot_arg(ctx, inner_expr) {
        let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
        let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
        return Some(mul_pruned(ctx, d_arg, csc_arg));
    }

    if let Some(arg) =
        same_arg_quotient_over_builtin_denominator(ctx, inner_expr, add_one_sin_arg, BuiltinFn::Cos)
    {
        let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
        let sec_arg = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
        return Some(mul_pruned(ctx, d_arg, sec_arg));
    }

    if let Some(arg) = same_arg_quotient_over_builtin_denominator(
        ctx,
        inner_expr,
        plus_or_minus_one_cos_arg,
        BuiltinFn::Sin,
    ) {
        let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
        let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
        return Some(mul_pruned(ctx, d_arg, csc_arg));
    }

    None
}

fn log_abs_builtin_derivative(ctx: &mut Context, arg: ExprId, var: &str) -> Option<ExprId> {
    let inner_expr = unary_builtin_arg(ctx, arg, BuiltinFn::Abs)?;

    if let Some(derivative) = log_abs_reciprocal_trig_primitive_derivative(ctx, inner_expr, var) {
        return Some(derivative);
    }
    if let Some(derivative) = compact_sqrt_log_abs_trig_derivative(ctx, inner_expr, var) {
        return Some(derivative);
    }
    if let Some(derivative) = compact_sqrt_log_hyperbolic_derivative(ctx, inner_expr, var) {
        return Some(derivative);
    }

    if let Expr::Div(num, den) = ctx.get(inner_expr) {
        let (num, den) = (*num, *den);
        let d_num = differentiate_symbolic_expr(ctx, num, var)?;
        let d_den = differentiate_symbolic_expr(ctx, den, var)?;
        if is_zero(ctx, d_den) {
            return Some(div_pruned(ctx, d_num, num));
        }
        if is_zero(ctx, d_num) {
            let den_part = div_pruned(ctx, d_den, den);
            return Some(ctx.add(Expr::Neg(den_part)));
        }

        let term1 = mul_pruned(ctx, d_num, den);
        let term2 = mul_pruned(ctx, num, d_den);
        let numerator = sub_pruned(ctx, term1, term2);
        let denominator = mul_pruned(ctx, num, den);
        return Some(div_pruned(ctx, numerator, denominator));
    }

    if let Expr::Mul(left, right) = ctx.get(inner_expr) {
        let (left, right) = (*left, *right);
        let d_left = differentiate_symbolic_expr(ctx, left, var)?;
        let d_right = differentiate_symbolic_expr(ctx, right, var)?;
        if is_zero(ctx, d_right) {
            return Some(div_pruned(ctx, d_left, left));
        }
        if is_zero(ctx, d_left) {
            return Some(div_pruned(ctx, d_right, right));
        }

        let term1 = mul_pruned(ctx, d_left, right);
        let term2 = mul_pruned(ctx, left, d_right);
        let numerator = add_pruned(ctx, term1, term2);
        let denominator = mul_pruned(ctx, left, right);
        return Some(div_pruned(ctx, numerator, denominator));
    }

    if let Some(trig_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sin) {
        let d_trig_arg = differentiate_symbolic_expr(ctx, trig_arg, var)?;
        let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![trig_arg]);
        let numerator = mul_pruned(ctx, d_trig_arg, cos_u);
        return Some(ctx.add(Expr::Div(numerator, inner_expr)));
    }

    if let Some(trig_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cos) {
        let d_trig_arg = differentiate_symbolic_expr(ctx, trig_arg, var)?;
        let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![trig_arg]);
        let neg_sin_u = ctx.add(Expr::Neg(sin_u));
        let numerator = mul_pruned(ctx, d_trig_arg, neg_sin_u);
        return Some(ctx.add(Expr::Div(numerator, inner_expr)));
    }

    if let Some(hyperbolic_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sinh) {
        let d_hyperbolic_arg = differentiate_symbolic_expr(ctx, hyperbolic_arg, var)?;
        let tanh_u = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
        return Some(ctx.add(Expr::Div(d_hyperbolic_arg, tanh_u)));
    }

    if let Some(hyperbolic_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cosh) {
        let d_hyperbolic_arg = differentiate_symbolic_expr(ctx, hyperbolic_arg, var)?;
        let tanh_u = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
        return Some(mul_pruned(ctx, d_hyperbolic_arg, tanh_u));
    }

    let d_inner = differentiate_symbolic_expr(ctx, inner_expr, var)?;
    Some(div_pruned(ctx, d_inner, inner_expr))
}

fn fixed_base_log_abs_derivative(
    ctx: &mut Context,
    arg: ExprId,
    base: ExprId,
    var: &str,
) -> Option<ExprId> {
    let derivative = log_abs_builtin_derivative(ctx, arg, var)?;
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let one = ctx.num(1);
    let reciprocal_ln_base = ctx.add(Expr::Div(one, ln_base));
    Some(mul_pruned(ctx, reciprocal_ln_base, derivative))
}

fn variable_base_log_abs_derivative(
    ctx: &mut Context,
    base: ExprId,
    d_base: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg_ratio = log_abs_builtin_derivative(ctx, arg, var)?;
    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let two = ctx.num(2);
    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));
    let base_ratio =
        log_abs_builtin_derivative(ctx, base, var).unwrap_or_else(|| div_pruned(ctx, d_base, base));
    let term_arg = mul_pruned(ctx, arg_ratio, ln_base);
    let term_base = mul_pruned(ctx, ln_arg, base_ratio);
    let numerator = sub_pruned(ctx, term_arg, term_base);
    Some(ctx.add(Expr::Div(numerator, ln_base_sq)))
}

fn numeric_fixed_base_log_abs_derivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    base: i64,
) -> Option<ExprId> {
    let base = ctx.num(base);
    fixed_base_log_abs_derivative(ctx, arg, base, var)
}

fn abs_quotient_derivative(
    ctx: &mut Context,
    arg: ExprId,
    quotient_num: ExprId,
    quotient_den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let d_num = differentiate_symbolic_expr(ctx, quotient_num, var)?;
    let d_den = differentiate_symbolic_expr(ctx, quotient_den, var)?;
    let term1 = mul_pruned(ctx, d_num, quotient_den);
    let term2 = mul_pruned(ctx, quotient_num, d_den);
    let quotient_derivative_num = sub_pruned(ctx, term1, term2);
    let numerator = mul_pruned(ctx, quotient_num, quotient_derivative_num);

    let three = ctx.num(3);
    let den_cubed = ctx.add(Expr::Pow(quotient_den, three));
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let denominator = mul_pruned(ctx, abs_arg, den_cubed);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

/// Differentiate `expr` with respect to variable `var`.
pub fn differentiate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if !contains_named_var(ctx, expr, var) {
        return Some(ctx.num(0));
    }

    if let Some(derivative) = differentiate_scaled_additive_log_abs_by_linearity(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = arctan_reciprocal_affine_by_parts_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = atanh_affine_by_parts_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = positive_quadratic_square_primitive_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = asinh_affine_by_parts_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = acosh_affine_by_parts_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = log_square_by_parts_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = trig_square_power_reduction_primitive_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = trig_fifth_power_reduction_primitive_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) = try_constant_scaled_atanh_polynomial_derivative(ctx, expr, var) {
        return Some(derivative);
    }

    if let Some(derivative) =
        try_constant_scaled_bounded_inverse_trig_polynomial_derivative(ctx, expr, var)
    {
        return Some(derivative);
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) => {
            if ctx.sym_name(*sym_id) == var {
                Some(ctx.num(1))
            } else {
                Some(ctx.num(0))
            }
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => differentiate_additive_by_linearity(ctx, expr, var),
        Expr::Neg(inner) => {
            let inner = *inner;
            let d_inner = differentiate_symbolic_expr(ctx, inner, var)?;
            Some(neg_pruned(ctx, d_inner))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(derivative) = try_linear_times_sec_csc_linear_derivative(ctx, expr, var) {
                return Some(derivative);
            }
            if let Some(derivative) = try_linear_times_cot_linear_derivative(ctx, expr, var) {
                return Some(derivative);
            }
            if let Some(derivative) =
                try_linear_times_arctan_reciprocal_linear_derivative(ctx, expr, var)
            {
                return Some(derivative);
            }
            if let Some(derivative) = try_linear_times_tanh_linear_derivative(ctx, expr, var) {
                return Some(derivative);
            }
            if let Some(derivative) = try_linear_times_hyperbolic_linear_derivative(ctx, expr, var)
            {
                return Some(derivative);
            }
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            Some(add_pruned(ctx, term1, term2))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(derivative) =
                scaled_reciprocal_trig_power_primitive_derivative(ctx, l, r, var)
            {
                return Some(derivative);
            }
            if let Some(derivative) = scaled_trig_quotient_power_derivative(ctx, l, r, var) {
                return Some(derivative);
            }
            if !contains_named_var(ctx, r, var) {
                let dl = differentiate_symbolic_expr(ctx, l, var)?;
                return Some(div_pruned(ctx, dl, r));
            }
            if let Some((kind, arg)) = canonical_reciprocal_trig_div_kind(ctx, l, r) {
                let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
                return Some(reciprocal_trig_derivative(ctx, kind, arg, d_arg));
            }
            if let Some(arg) = canonical_hyperbolic_coth_div_arg(ctx, l, r) {
                let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
                return Some(hyperbolic_coth_derivative(ctx, arg, d_arg));
            }
            if let Some(derivative) =
                try_constant_scaled_hyperbolic_reciprocal_tanh_div_derivative(ctx, l, r, var)
            {
                return Some(derivative);
            }
            if let Some(derivative) =
                try_constant_scaled_hyperbolic_reciprocal_div_derivative(ctx, l, r, var)
            {
                return Some(derivative);
            }
            if let Some(derivative) =
                try_constant_scaled_polynomial_reciprocal_power_div_derivative(ctx, l, r, var)
            {
                return Some(derivative);
            }
            if let Some(derivative) =
                try_linear_times_hyperbolic_coth_linear_div_derivative(ctx, l, r, var)
            {
                return Some(derivative);
            }

            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            let num = sub_pruned(ctx, term1, term2);
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let db = differentiate_symbolic_expr(ctx, base, var)?;
            let de = differentiate_symbolic_expr(ctx, exp, var)?;

            if !contains_named_var(ctx, exp, var) {
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = mul_pruned(ctx, exp, pow_term);
                Some(mul_pruned(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                let ln_a = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term = mul_pruned(ctx, expr, ln_a);
                Some(mul_pruned(ctx, term, de))
            } else {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term1 = mul_pruned(ctx, de, ln_base);
                let term2_num = mul_pruned(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = add_pruned(ctx, term1, term2);
                Some(mul_pruned(ctx, expr, inner))
            }
        }
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Log)) && args.len() == 2 {
                let base = args[0];
                let arg = args[1];
                if contains_named_var(ctx, base, var) {
                    let db = differentiate_symbolic_expr(ctx, base, var)?;
                    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
                    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                    let two = ctx.num(2);
                    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));

                    if contains_named_var(ctx, arg, var) {
                        if let Some(derivative) =
                            variable_base_log_abs_derivative(ctx, base, db, arg, var)
                        {
                            return Some(derivative);
                        }

                        let da = differentiate_symbolic_expr(ctx, arg, var)?;
                        let arg_ratio = ctx.add(Expr::Div(da, arg));
                        let base_ratio = ctx.add(Expr::Div(db, base));
                        let term_arg = mul_pruned(ctx, arg_ratio, ln_base);
                        let term_base = mul_pruned(ctx, ln_arg, base_ratio);
                        let numerator = sub_pruned(ctx, term_arg, term_base);
                        return Some(ctx.add(Expr::Div(numerator, ln_base_sq)));
                    }

                    let denominator = mul_pruned(ctx, base, ln_base_sq);
                    let numerator = mul_pruned(ctx, ln_arg, db);
                    let neg_numerator = ctx.add(Expr::Neg(numerator));
                    return Some(ctx.add(Expr::Div(neg_numerator, denominator)));
                }

                if let Some(derivative) = fixed_base_log_abs_derivative(ctx, arg, base, var) {
                    return Some(derivative);
                }

                let da = differentiate_symbolic_expr(ctx, arg, var)?;
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let den = mul_pruned(ctx, arg, ln_base);
                return Some(ctx.add(Expr::Div(da, den)));
            }

            if args.len() != 1 {
                return None;
            }
            let arg = args[0];

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) {
                if let Some(derivative) = log_abs_builtin_derivative(ctx, arg, var) {
                    return Some(derivative);
                }
                if let Some(derivative) = compact_sqrt_log_hyperbolic_derivative(ctx, arg, var) {
                    return Some(derivative);
                }
            }
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Log2) {
                if let Some(derivative) = numeric_fixed_base_log_abs_derivative(ctx, arg, var, 2) {
                    return Some(derivative);
                }
            }
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Log10) {
                if let Some(derivative) = numeric_fixed_base_log_abs_derivative(ctx, arg, var, 10) {
                    return Some(derivative);
                }
            }

            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                    if let Some(recip_base) = unit_reciprocal_base(ctx, arg)
                        .or_else(|| surd_numerator_reciprocal_base(ctx, arg))
                    {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, var, 1));
                    }
                }
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                    if let Some(recip_base) = unit_reciprocal_base(ctx, arg)
                        .or_else(|| surd_numerator_reciprocal_base(ctx, arg))
                    {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, var, -1));
                    }
                }
                Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                    if let Some(recip_base) = unit_reciprocal_base(ctx, arg) {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arccot_derivative(ctx, recip_base, d_base));
                    }
                }
                _ => {}
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Abs) {
                if let Expr::Div(num, den) = ctx.get(arg) {
                    let (num, den) = (*num, *den);
                    return abs_quotient_derivative(ctx, arg, num, den, var);
                }
            }

            let da = differentiate_symbolic_expr(ctx, arg, var)?;

            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    Some(mul_pruned(ctx, cos_u, da))
                }
                Some(BuiltinFn::Cos) => {
                    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let neg_sin = ctx.add(Expr::Neg(sin_u));
                    Some(mul_pruned(ctx, neg_sin, da))
                }
                Some(BuiltinFn::Tan) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    let two = ctx.num(2);
                    let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                    if let Some(scale) = non_integer_rational_constant_value(ctx, da) {
                        let one = ctx.num(1);
                        return Some(scaled_square_denominator_ratio(ctx, scale, one, cos_sq));
                    }
                    Some(ctx.add(Expr::Div(da, cos_sq)))
                }
                Some(BuiltinFn::Sec) => Some(secant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Csc) => Some(cosecant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Cot) => Some(cotangent_derivative(ctx, arg, da)),
                Some(BuiltinFn::Sinh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    Some(mul_pruned(ctx, cosh_u, da))
                }
                Some(BuiltinFn::Cosh) => {
                    let sinh_u = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    Some(mul_pruned(ctx, sinh_u, da))
                }
                Some(BuiltinFn::Tanh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    let two = ctx.num(2);
                    let cosh_sq = ctx.add(Expr::Pow(cosh_u, two));
                    Some(ctx.add(Expr::Div(da, cosh_sq)))
                }
                Some(BuiltinFn::Sqrt) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    let minus_one = ctx.num(-1);
                    let neg_half = ctx.add(Expr::Div(minus_one, two));
                    let pow_term = ctx.add(Expr::Pow(arg, neg_half));
                    let term = mul_pruned(ctx, half, pow_term);
                    Some(mul_pruned(ctx, term, da))
                }
                Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let den = ctx.add(Expr::Add(one, arg_sq));
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                    if let Some(scaled) = scaled_one_minus_arg_square_derivative(ctx, arg, da) {
                        return Some(scaled);
                    }
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                    let neg_da = ctx.add(Expr::Neg(da));
                    if let Some(scaled) = scaled_one_minus_arg_square_derivative(ctx, arg, neg_da) {
                        return Some(scaled);
                    }
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    Some(ctx.add(Expr::Div(neg_da, den)))
                }
                Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => {
                    Some(arcsec_like_derivative(ctx, arg, da, var, 1))
                }
                Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
                    Some(arcsec_like_derivative(ctx, arg, da, var, -1))
                }
                Some(BuiltinFn::Arccot | BuiltinFn::Acot) => Some(arccot_derivative(ctx, arg, da)),
                Some(BuiltinFn::Asinh) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let inner = ctx.add(Expr::Add(arg_sq, one));
                    let inner = if should_preserve_asinh_shifted_linear_radicand(ctx, arg, var) {
                        cas_ast::hold::wrap_hold(ctx, inner)
                    } else {
                        inner
                    };
                    let den = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Acosh) => Some(acosh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Atanh) => Some(atanh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Exp) => Some(mul_pruned(ctx, expr, da)),
                Some(BuiltinFn::Ln) => Some(ctx.add(Expr::Div(da, arg))),
                Some(BuiltinFn::Log2) => Some(constant_base_log_derivative(ctx, arg, da, 2)),
                Some(BuiltinFn::Log10) => Some(constant_base_log_derivative(ctx, arg, da, 10)),
                Some(BuiltinFn::Abs) => {
                    let term = ctx.add(Expr::Div(arg, expr));
                    Some(mul_pruned(ctx, term, da))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::differentiate_symbolic_expr;
    use crate::eval_f64;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use std::collections::HashMap;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn eval_at(ctx: &Context, id: cas_ast::ExprId, x: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x);
        eval_f64(ctx, id, &vars).expect("numeric eval")
    }

    #[test]
    fn differentiates_product_rule() {
        let mut ctx = Context::new();
        let expr = parse("x*sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("sin(x)"));
        assert!(text.contains("cos(x)"));
    }

    #[test]
    fn differentiates_chain_rule_exp() {
        let mut ctx = Context::new();
        let expr = parse("exp(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("exp(x^2)") || text.contains("e^(x^2)"));
    }

    #[test]
    fn prunes_zero_product_terms_from_polynomial_derivative() {
        let mut ctx = Context::new();
        let expr = parse("x^3 + 2*x^2 - 5*x + 1", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(!text.contains("0 ·"), "unexpected zero product in {text}");
        assert!(!text.contains("· 1"), "unexpected unit factor in {text}");
        assert!(text.contains("3"));
        assert!(text.contains("- 5"));
    }

    #[test]
    fn prunes_unit_chain_factor_for_sine() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cos(x)");
    }

    #[test]
    fn differentiates_sqrt_log_abs_trig_chain_compactly() {
        let mut ctx = Context::new();
        let expr = parse("-ln(abs(cos(sqrt(3*x+1))))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

        assert_eq!(
            rendered(&ctx, out),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn differentiates_sqrt_log_hyperbolic_chain_compactly() {
        let mut ctx = Context::new();
        let expr = parse("ln(cosh((3*x+1)^(1/2)))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

        assert_eq!(
            rendered(&ctx, out),
            "3 * tanh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );

        let expr = parse("ln(abs(sinh((3*x+1)^(1/2))))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

        assert_eq!(
            rendered(&ctx, out),
            "3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn differentiates_constant_denominator_without_quotient_rule_noise() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)/2", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cos(x) / 2");
    }

    #[test]
    fn differentiates_reciprocal_trig_power_primitives_compactly() {
        let mut ctx = Context::new();
        let expr = parse("tan(x)^2/2", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "tan(x) * sec(x)^2");

        let expr = parse("tan(2*x + 1)^2/2", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "tan(2 * x + 1) * sec(2 * x + 1)^2 * 2");

        let expr = parse("-cot(x)^2/2", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cot(x) * csc(x)^2");

        let expr = parse("sin(x)^2/(2*cos(x)^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "tan(x) * sec(x)^2");

        let expr = parse("-cos(x)^2/(2*sin(x)^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cot(x) * csc(x)^2");
    }

    #[test]
    fn differentiates_linear_times_arctan_reciprocal_linear_compactly() {
        let mut ctx = Context::new();
        let expr = parse("x*arctan(1/(2*x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(
            rendered(&ctx, out),
            "arctan(1 / (2 * x + 1)) - 2 * x / ((2 * x + 1)^2 + 1)"
        );

        let expr = parse("(x+1/2)*arctan(1/(2*x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(
            rendered(&ctx, out),
            "arctan(1 / (2 * x + 1)) - (2 * x + 1) / ((2 * x + 1)^2 + 1)"
        );

        let expr = parse("((2*x+1)*arctan(1/(2*x+1)))/2", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(
            rendered(&ctx, out),
            "(2 * arctan(1 / (2 * x + 1)) - (4 * x + 2) / ((2 * x + 1)^2 + 1)) / 2"
        );

        let expr = parse(
            "ln(4*x^2+4*x+2)/4 + arctan(1/(2*x+1))/2 + x*arctan(1/(2*x+1))",
            &mut ctx,
        )
        .expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "arctan(1 / (2 * x + 1))");
    }

    #[test]
    fn differentiates_atanh_affine_by_parts_antiderivative_compactly() {
        let mut ctx = Context::new();
        let expr = parse("1/2*ln(1-x^2) + x*atanh(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "atanh(x)");

        let expr =
            parse("1/2*(1/2*ln(1-(2*x+1)^2) + (2*x+1)*atanh(2*x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "atanh(2 * x + 1)");
    }

    #[test]
    fn differentiates_positive_quadratic_square_primitives_compactly() {
        let cases = [
            ("1/2*arctan(x) + x/(2*(x^2+1))", "1 / (x^2 + 1)^2"),
            (
                "1/2*arctan(x+1) + (x+1)/(2*(x^2+2*x+2))",
                "1 / (x^2 + 2 * x + 2)^2",
            ),
            ("1/4*arctan(2*x) + x/(2*(4*x^2+1))", "1 / (4 * x^2 + 1)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }

        let mut ctx = Context::new();
        let expr = parse("1/4*arctan(2*x) + x/(2*(4*x^2+1))", &mut ctx).expect("parse");
        let held = cas_ast::hold::wrap_hold(&mut ctx, expr);
        let out = differentiate_symbolic_expr(&mut ctx, held, "x").expect("diff held");
        assert_eq!(rendered(&ctx, out), "1 / (4 * x^2 + 1)^2");

        let mut ctx = Context::new();
        let expr = parse("(2*arctan(2*x)+8*arctan(2*x)*x^2+4*x)/(32*x^2+8)", &mut ctx)
            .expect("parse combined");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff combined");
        assert_eq!(rendered(&ctx, out), "64 / (32 * x^2 + 8)^2");
    }

    #[test]
    fn differentiates_asinh_affine_by_parts_antiderivative_compactly() {
        let mut ctx = Context::new();
        let expr = parse("x*asinh(x) - sqrt(x^2+1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "asinh(x)");

        let expr =
            parse("1/2*((2*x+1)*asinh(2*x+1) - sqrt((2*x+1)^2+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "asinh(2 * x + 1)");

        let expr =
            parse("1/2*(sqrt((1-2*x)^2+1) - asinh(1-2*x)*(1-2*x))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "asinh(1 - 2 * x)");
    }

    #[test]
    fn differentiates_acosh_affine_by_parts_antiderivative_compactly() {
        let mut ctx = Context::new();
        let expr = parse("x*acosh(x) - sqrt(x-1)*sqrt(x+1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "acosh(x)");

        let expr = parse("x*acosh(x) - sqrt(x^2-1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "acosh(x)");

        let expr = parse(
            "1/2*((2*x+1)*acosh(2*x+1) - sqrt(2*x)*sqrt(2*x+2))",
            &mut ctx,
        )
        .expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "acosh(2 * x + 1)");
    }

    #[test]
    fn differentiates_log_square_by_parts_antiderivative_compactly() {
        let mut ctx = Context::new();
        let expr = parse("(x^2+x-1)*(ln(x^2+x-1)-1)^2 + (x^2+x-1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "(2 * x + 1) * ln(x^2 + x - 1)^2");

        let expr = parse("2*((x^2+x+1)*(ln(x^2+x+1)-1)^2 + (x^2+x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "(4 * x + 2) * ln(x^2 + x + 1)^2");

        let expr = parse("(x^2+x+1)*(ln(x^2+x+1)^2 + 2 - 2*ln(x^2+x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "(2 * x + 1) * ln(x^2 + x + 1)^2");
    }

    #[test]
    fn differentiates_log_cube_by_parts_antiderivative_compactly() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2+1)*(ln(x^2+1)^3 - 3*ln(x^2+1)^2 + 6*ln(x^2+1) - 6)",
            &mut ctx,
        )
        .expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "2 * x * ln(x^2 + 1)^3");

        let expr = parse(
            "(x^2+x+1)*(ln(x^2+x+1)^3 - 3*ln(x^2+x+1)^2 + 6*ln(x^2+x+1) - 6)",
            &mut ctx,
        )
        .expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "(2 * x + 1) * ln(x^2 + x + 1)^3");
    }

    #[test]
    fn differentiates_trig_fifth_power_reduction_primitives_compactly() {
        let cases = [
            ("1/5*(10/3*cos(x)^3 - cos(x)^5 - 5*cos(x))", "sin(x)^5"),
            ("1/5*(sin(x)^5 + 5*sin(x) - 10/3*sin(x)^3)", "cos(x)^5"),
            (
                "1/10*(10/3*cos(2*x+1)^3 - cos(2*x+1)^5 - 5*cos(2*x+1))",
                "sin(2 * x + 1)^5",
            ),
            (
                "1/10*(sin(2*x+1)^5 + 5*sin(2*x+1) - 10/3*sin(2*x+1)^3)",
                "cos(2 * x + 1)^5",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_trig_square_power_reduction_primitives_compactly() {
        let cases = [
            ("1/2*x - 1/4*sin(2*x)", "sin(x)^2"),
            ("1/4*sin(2*x) + 1/2*x", "cos(x)^2"),
            ("1/2*x - 1/8*sin(4*x+2)", "sin(2 * x + 1)^2"),
            ("1/8*sin(4*x+2) + 1/2*x", "cos(2 * x + 1)^2"),
            ("1/2*x + 1/6*sin(2-3*x)", "sin(1 - 3/2 * x)^2"),
            ("1/2*x - 1/6*sin(2-3*x)", "cos(1 - 3/2 * x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_reciprocal_trig_functions_directly() {
        let cases = [
            ("sec(x)", "sec(x) * tan(x)"),
            ("csc(x)", "-(csc(x) * cot(x))"),
            ("cot(x)", "-1 / sin(x)^2"),
            ("1/cos(x)", "sec(x) * tan(x)"),
            ("1/sin(x)", "-(csc(x) * cot(x))"),
            ("cos(x)/sin(x)", "-1 / sin(x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }

        let mut ctx = Context::new();
        let expr = parse("acosh(x^2 + 2*x + 2)", &mut ctx).expect("parse expanded square");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff expanded square");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("sqrt(x^2 + 2 * x + 2 - 1)"),
            "expanded perfect-square acosh derivative should keep sqrt(left): {text}"
        );
        assert!(
            !text.contains("(x^2 + 2 * x + 2 - 1)^(-1 / 2)"),
            "expanded perfect-square acosh derivative should avoid inverse-power left radicand: {text}"
        );
    }

    #[test]
    fn differentiates_reciprocal_trig_chain_rule_directly() {
        let cases = [
            ("sec(2*x + 1)", "2 * sec(2 * x + 1) * tan(2 * x + 1)"),
            ("csc(2*x + 1)", "-2 * csc(2 * x + 1) * cot(2 * x + 1)"),
            ("cot(2*x + 1)", "-2 / sin(2 * x + 1)^2"),
            ("tan((3*x + 2)/2)", "3 / (2 * cos((3 * x + 2) / 2)^2)"),
            ("cot((2 - 3*x)/2)", "3 / (2 * sin((2 - 3 * x) / 2)^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_builtins_through_domain_carrying_quotients() {
        let cases = [
            ("ln(abs(sin(x)))", "cos(x) / sin(x)"),
            ("ln(abs(cos(x)))", "-sin(x) / cos(x)"),
            ("ln(abs(sinh(x)))", "1 / tanh(x)"),
            ("ln(abs(cosh(x)))", "tanh(x)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_additive_log_abs_partial_fraction_primitive_by_linearity() {
        let mut ctx = Context::new();
        let expr = parse(
            "(-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1))",
            &mut ctx,
        )
        .expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(!text.contains("ln("), "{text}");
        assert!(text.contains("x + 1"), "{text}");
        assert!(text.contains("x - 1"), "{text}");
        assert!(text.contains("(x - 1)^2"), "{text}");
    }

    #[test]
    fn differentiates_reciprocal_trig_log_abs_primitives_directly() {
        let cases = [
            ("ln(abs(sec(x)+tan(x)))", "sec(x)"),
            ("ln(abs(sec(2*x+1)+tan(2*x+1)))", "2 * sec(2 * x + 1)"),
            ("ln(abs((1+sin(2*x+1))/cos(2*x+1)))", "2 * sec(2 * x + 1)"),
            ("ln(abs(csc(x)-cot(x)))", "csc(x)"),
            ("ln(abs(csc(2*x+1)-cot(2*x+1)))", "2 * csc(2 * x + 1)"),
            ("ln(abs((cos(2*x+1)-1)/sin(2*x+1)))", "2 * csc(2 * x + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_quotient_without_abs_squared_noise() {
        let cases = [
            (
                "ln(abs((x-1)/(x+1)))",
                "(x + 1 - (x - 1)) / ((x - 1) * (x + 1))",
            ),
            ("ln(abs(x/y))", "1 / x"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_product_without_abs_squared_noise() {
        let cases = [
            ("ln(abs(x*y))", "1 / x"),
            (
                "ln(abs((x-1)*(x+1)))",
                "(x + x + 1 - 1) / ((x + 1) * (x - 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_generic_log_abs_without_abs_squared_noise() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("/ (x^2 - 1)") || text.contains("/(x^2 - 1)"),
            "generic ln(abs(u)) derivative should divide by u directly: {text}"
        );
        assert!(
            !text.contains('|'),
            "generic ln(abs(u)) derivative should not route through abs noise: {text}"
        );
    }

    #[test]
    fn differentiates_fixed_base_log_abs_without_abs_squared_noise() {
        let cases = [
            ("log(2, abs(x^2-1))", "ln(2)"),
            ("log(y, abs(x^2-1))", "ln(y)"),
            ("log2(abs(x^2-1))", "ln(2)"),
            ("log10(abs(x^2-1))", "ln(10)"),
        ];

        for (input, expected_log_base) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                text.contains(expected_log_base),
                "fixed-base log(abs(u)) derivative should divide by {expected_log_base}: {text}"
            );
            assert!(
                text.contains("x^2 - 1"),
                "fixed-base log(abs(u)) derivative should divide by u directly: {text}"
            );
            assert!(
                !text.contains('|'),
                "fixed-base log(abs(u)) derivative should not route through abs noise: {text}"
            );
        }
    }

    #[test]
    fn differentiates_variable_base_log_abs_without_abs_squared_noise() {
        let mut ctx = Context::new();
        let expr = parse("log(x, abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("ln(x)"), "{text}");
        assert!(text.contains("ln(|x^2 - 1|)"), "{text}");
        assert!(
            text.contains("/ (x^2 - 1)")
                || text.contains("/(x^2 - 1)")
                || text.contains("/((x^2 - 1))"),
            "variable-base log(abs(u)) derivative should divide by u directly: {text}"
        );
        assert!(
            !text.contains("/ |x^2 - 1|") && !text.contains("/(|x^2 - 1|"),
            "variable-base log(abs(u)) derivative should not divide by abs(u): {text}"
        );
        assert!(
            !text.contains("|x^2 - 1|)^2"),
            "variable-base log(abs(u)) derivative should not route through abs-squared cleanup: {text}"
        );
    }

    #[test]
    fn differentiates_variable_abs_base_log_abs_without_abs_base_noise() {
        let mut ctx = Context::new();
        let expr = parse("log(abs(x), abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("ln(|x|)"), "{text}");
        assert!(text.contains("ln(|x^2 - 1|)"), "{text}");
        assert!(
            text.contains("/ x") || text.contains("/x"),
            "variable abs-base log(abs(u)) derivative should divide by the base inner directly: {text}"
        );
        assert!(
            !text.contains("/ |x|") && !text.contains("x / |x|"),
            "variable abs-base log(abs(u)) derivative should not route through abs-base cleanup: {text}"
        );
    }

    #[test]
    fn differentiates_abs_quotient_without_nested_fraction_noise() {
        let mut ctx = Context::new();
        let expr = parse("abs((x-1)/(x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("|(x - 1) / (x + 1)|"), "{text}");
        assert!(text.contains("(x + 1)^3"), "{text}");
        assert!(
            !text.contains("(x + 1)^2"),
            "unexpected quotient-rule denominator expansion in {text}"
        );
    }

    #[test]
    fn differentiates_hyperbolic_functions() {
        let cases = [
            ("sinh(x)", "cosh(x)"),
            ("cosh(x)", "sinh(x)"),
            ("tanh(x)", "1 / cosh(x)^2"),
            ("tanh(2*x + 1)", "2 / cosh(2 * x + 1)^2"),
            ("tanh(1 - 2*x)", "-2 / cosh(1 - 2 * x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("cosh(x^2)"), "{text}");
        assert!(text.contains("2 * x"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_linear_times_hyperbolic_linear_with_compact_chain_factor() {
        let cases = [
            (
                "(2/3*x+2/3)*cosh((3*x+2)/2)",
                "2/3 * cosh((3 * x + 2) / 2) + (x + 1) * sinh((3 * x + 2) / 2)",
            ),
            ("4/9*sinh((3*x+2)/2)", "2/3 * cosh((3 * x + 2) / 2)"),
            (
                "(x+1)*tanh(2*x+1)",
                "tanh(2 * x + 1) + (2 * x + 2) / cosh(2 * x + 1)^2",
            ),
            (
                "(3*x+2)*tanh(2*x+1)",
                "3 * tanh(2 * x + 1) + (6 * x + 4) / cosh(2 * x + 1)^2",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_coth_quotient_directly() {
        let cases = [
            ("cosh(x)/sinh(x)", "-1 / sinh(x)^2"),
            (
                "(x+1)*cosh(2*x+1)/sinh(2*x+1)",
                "cosh(2 * x + 1) / sinh(2 * x + 1) - (2 * x + 2) / sinh(2 * x + 1)^2",
            ),
            (
                "(x*cosh(2*x+1)+cosh(2*x+1))/sinh(2*x+1)",
                "cosh(2 * x + 1) / sinh(2 * x + 1) - (2 * x + 2) / sinh(2 * x + 1)^2",
            ),
            ("1/tanh(x)", "-1 / sinh(x)^2"),
            ("(-1)/(2*tanh(2*x+1))", "1 / sinh(2 * x + 1)^2"),
            ("-1/(2*cosh(2*x+1))", "sinh(2 * x + 1) / cosh(2 * x + 1)^2"),
            ("-1/(2*sinh(2*x+1))", "cosh(2 * x + 1) / sinh(2 * x + 1)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_constant_scaled_polynomial_reciprocal_powers_directly() {
        let cases = [
            ("-1/(2*(x^2+x-1)^2)", "(2 * x + 1) / (x^2 + x - 1)^3"),
            ("-1/(2*(x^2+x-1)^3)", "3/2 * (2 * x + 1) / (x^2 + x - 1)^4"),
            ("-1/(4*(x+1)^4)", "1 / (x + 1)^5"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_total_domain_inverse_functions() {
        let cases = [("arctan(x)", "x^2 + 1"), ("asinh(x)", "x^2 + 1")];

        for (input, expected_core) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(text.contains(expected_core), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }
    }

    #[test]
    fn differentiates_constant_base_unary_logs() {
        let cases = [
            ("log2(x)", "1 / (x * ln(2))"),
            ("log10(x)", "1 / (x * ln(10))"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_variable_base_constant_argument_logs_conservatively() {
        let cases = [("log(x, 2)", "ln(2)"), ("log(x, y)", "ln(y)")];

        for (input, expected_log_arg) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                text.contains(expected_log_arg),
                "input: {input}, got: {text}"
            );
            assert!(text.contains("ln(x)^2"), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }

        let mut ctx = Context::new();
        let expr = parse("log(x, x + 1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("ln(x)"), "got: {text}");
        assert!(text.contains("ln(x + 1)"), "got: {text}");
        assert!(text.contains("ln(x)^2"), "got: {text}");
        assert!(!text.contains("diff("), "got: {text}");
    }

    #[test]
    fn differentiates_inverse_reciprocal_trig_directly() {
        let cases = [
            ("arcsec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccot(x)", "-1 / (x^2 + 1)"),
            ("asec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acot(x)", "-1 / (x^2 + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn inverse_reciprocal_trig_chain_rule_keeps_compact_core() {
        let mut ctx = Context::new();
        let expr = parse("arcsec((x^2 + 1)^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("sqrt(1 - 1 / (x^2 + 1)^2^2)"), "{text}");
        assert!(
            !text.contains("+") || !text.contains("x^3"),
            "unexpected quotient-rule expansion in {text}"
        );
    }

    #[test]
    fn positive_inverse_reciprocal_trig_chain_rule_uses_direct_positive_argument_form() {
        let cases = [("arcsec(x^2 + 1)", false), ("arccsc(x^2 + 1)", true)];

        for (input, expect_negative) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("1 - 1 /"),
                "positive argument derivative should not use reciprocal-square gap: {text}"
            );
            assert!(
                text.contains("sqrt((x^2 + 1)^2 - 1)"),
                "positive argument derivative should expose the direct gap: {text}"
            );
            assert_eq!(
                text.starts_with("-"),
                expect_negative,
                "unexpected sign for {input}: {text}"
            );
        }
    }

    #[test]
    fn positive_surd_quotient_inverse_reciprocal_trig_uses_direct_gap() {
        let cases = [
            ("arcsec((x^2+x+3)/sqrt(2))", false),
            ("arccsc((x^2+x+3)/sqrt(2))", true),
        ];

        for (input, expect_negative) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("1 - 1 /") && !text.contains("1 - 2 /"),
                "surd quotient derivative should expose the direct gap: {text}"
            );
            assert!(
                text.contains("x^4 + 2 * x^3 + 7 * x^2 + 6 * x + 7"),
                "surd quotient derivative should expose q^2-k: {text}"
            );
            assert_eq!(
                text.starts_with("-"),
                expect_negative,
                "unexpected sign for {input}: {text}"
            );
        }
    }

    #[test]
    fn inverse_reciprocal_trig_surd_scale_preserves_derivative_value() {
        let cases = [
            ("arcsec((x^2+x+3)/sqrt(1/2))", false),
            ("arcsec(sqrt(2)*(x^2+x+3))", false),
            ("arccsc(sqrt(2)*(x^2+x+3))", true),
        ];

        for (input, expect_negative) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);
            let expected = parse("(2*x+1)/((x^2+x+3)*sqrt(2*(x^2+x+3)^2 - 1))", &mut ctx)
                .expect("parse expected");

            assert!(
                !text.contains("1 - 1 /") && !text.contains("x^8"),
                "surd-scaled derivative should use compact direct gap: {text}"
            );
            assert_eq!(
                text.starts_with("-"),
                expect_negative,
                "unexpected sign for {input}: {text}"
            );

            for sample in [-1.0, 0.0, 1.0] {
                let actual = eval_at(&ctx, out, sample);
                let expected_value = eval_at(&ctx, expected, sample);
                let expected_value = if expect_negative {
                    -expected_value
                } else {
                    expected_value
                };
                assert!(
                    (actual - expected_value).abs() < 1e-10,
                    "unexpected derivative value for {input} at x={sample}: actual={actual}, expected={expected_value}, text={text}"
                );
            }
        }
    }

    #[test]
    fn reciprocal_inverse_trig_rewrite_targets_keep_compact_derivative_core() {
        let cases = [
            "arccos(1/(x^2 + 1)^2)",
            "arcsin(1/(x^2 + 1)^2)",
            "arctan(1/(x^2 + 1)^2)",
        ];

        for input in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("/sqrt("),
                "unexpected quotient-rule reciprocal derivative in {text}"
            );
            assert!(
                !text.contains("x^3"),
                "unexpected expanded chain factor in {text}"
            );
        }
    }

    #[test]
    fn differentiates_constant_scaled_bounded_inverse_trig_surd_quotient_with_scaled_gap() {
        let mut ctx = Context::new();
        let expr = parse("arcsin((x^2+x+1)^2/sqrt(2/3))/sqrt(3)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 - 3 *"), "got: {text}");
        assert!(
            text.contains("4 * x^3 + 6 * x^2 + 6 * x + 2"),
            "got: {text}"
        );
        assert!(!text.contains("1 - 3/2"), "got: {text}");
        assert!(!text.contains("sqrt(3)"), "got: {text}");
    }

    #[test]
    fn differentiates_acosh_with_domain_safe_radicals() {
        let cases = [
            ("acosh(x)", "(x - 1)^(-1 / 2) / sqrt(x + 1)"),
            (
                "acosh(2*x + 1)",
                "2 * (2 * x)^(-1 / 2) / sqrt(2 * x + 1 + 1)",
            ),
            (
                "acosh(x^2 + 1)",
                "2 * x^(2 - 1) / (sqrt(x^2) * sqrt(x^2 + 1 + 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_atanh_with_open_unit_interval_witness() {
        let cases = [
            ("atanh(x)", "sqrt(1 - x^2)^(-2)"),
            ("atanh(x^2)", "(x^(2 - 1) * 2)/(sqrt(1 - x^2^2)^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_constant_scaled_atanh_surd_polynomial_directly() {
        let cases = [
            "1/3 * atanh(1/3 * sqrt(3) * (x^2 + 2*x + 1)) * sqrt(3)",
            "atanh((x^2 + 2*x + 1)/sqrt(3))/sqrt(3)",
        ];

        for input in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("sqrt(1 -") && !text.contains("atanh("),
                "direct atanh derivative should avoid the generic sqrt-gap route for {input}: {text}"
            );
            for sample in [-0.5, 0.0, 0.5] {
                let actual = eval_at(&ctx, out, sample);
                let expected = (2.0 * sample + 2.0) / (3.0 - (sample + 1.0).powi(4));
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "unexpected derivative value at x={sample} for {input}: actual={actual}, expected={expected}, text={text}"
                );
            }
        }
    }

    #[test]
    fn differentiates_sqrt_chain_rule_without_power_presimplification() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(2 - x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(-1 / 2)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_power_chain_rule_through_canonical_negation() {
        let mut ctx = Context::new();
        let expr = parse("(2 + (-x))^(1/2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(1 / 2 - 1)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_bounded_inverse_trig_functions() {
        let cases = [
            ("arcsin(x)", "1 / sqrt(1 - x^2)"),
            ("arccos(x)", "-1 / sqrt(1 - x^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_bounded_inverse_trig_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("asin(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("sqrt(1 - x^2^2)"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_inverse_function_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("arctan(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("x^2^2 + 1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_log_with_constant_base() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("x * ln(2)") || text.contains("ln(2) * x"),
            "{text}"
        );
        assert!(!text.contains("diff("), "{text}");
    }
}
