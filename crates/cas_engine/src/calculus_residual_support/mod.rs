use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_domain::exprs_equivalent;
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::try_rewrite_simplify_square_root_expr;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

mod affine_trig_power;
use affine_trig_power::*;

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn expr_exact_noise_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if left == right {
        return true;
    }

    match (ctx.get(left), ctx.get(right)) {
        (Expr::Number(left), Expr::Number(right)) => left == right,
        (Expr::Constant(left), Expr::Constant(right)) => left == right,
        (Expr::Variable(left), Expr::Variable(right)) => left == right,
        _ => expr_eq(ctx, left, right),
    }
}

fn exprs_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if expr_eq(ctx, left, right) || exprs_equivalent(ctx, left, right) {
        return true;
    }

    let Some(left_base) = reciprocal_sqrt_like_base(ctx, left) else {
        return false;
    };
    let Some(right_base) = reciprocal_sqrt_like_base(ctx, right) else {
        return false;
    };

    expr_eq(ctx, left_base, right_base) || exprs_equivalent(ctx, left_base, right_base)
}

fn rendered_exprs_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: left
        }
    ) == format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: right
        }
    )
}

fn one_expr(ctx: &mut Context) -> ExprId {
    ctx.num(1)
}

fn expr_is_one(ctx: &mut Context, expr: ExprId) -> bool {
    let one = one_expr(ctx);
    expr_eq(ctx, expr, one)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1).then_some(args[0])
}

fn is_positive_half(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr)
        .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
}

fn is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr)
        .is_some_and(|value| value == BigRational::new((-1).into(), 2.into()))
}

fn sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if is_positive_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn reciprocal_sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_negative_half(ctx, *exp) => Some(*base),
        Expr::Div(num, den) if is_one_constant(ctx, *num) => sqrt_like_base(ctx, *den),
        _ => None,
    }
}

fn denominator_factor_multiset_matches(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut used = vec![false; right.len()];
    'outer: for left_factor in left {
        for (index, right_factor) in right.iter().enumerate() {
            if !used[index] && exprs_match(ctx, *left_factor, *right_factor) {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn rational_polynomial_terms_match(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some((left_num, left_den)) = quotient_numerator_denominator(ctx, left) else {
        return false;
    };
    let Some((right_num, right_den)) = quotient_numerator_denominator(ctx, right) else {
        return false;
    };
    let Ok(left_num) = Polynomial::from_expr(ctx, left_num, var_name) else {
        return false;
    };
    let Ok(left_den) = Polynomial::from_expr(ctx, left_den, var_name) else {
        return false;
    };
    let Ok(right_num) = Polynomial::from_expr(ctx, right_num, var_name) else {
        return false;
    };
    let Ok(right_den) = Polynomial::from_expr(ctx, right_den, var_name) else {
        return false;
    };

    left_num.mul(&right_den) == right_num.mul(&left_den)
}

fn signed_rational_polynomial_terms_match(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some((left_num, left_den)) = signed_quotient_numerator_denominator(ctx, left) else {
        return false;
    };
    let Some((right_num, right_den)) = signed_quotient_numerator_denominator(ctx, right) else {
        return false;
    };
    let Ok(left_num) = Polynomial::from_expr(ctx, left_num, var_name) else {
        return false;
    };
    let Ok(left_den) = Polynomial::from_expr(ctx, left_den, var_name) else {
        return false;
    };
    let Ok(right_num) = Polynomial::from_expr(ctx, right_num, var_name) else {
        return false;
    };
    let Ok(right_den) = Polynomial::from_expr(ctx, right_den, var_name) else {
        return false;
    };

    left_num.mul(&right_den) == right_num.mul(&left_den)
}

fn antiderivative_term_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if exprs_match(ctx, left, right) {
        return true;
    }

    if rational_polynomial_terms_match(ctx, left, right, var_name) {
        return true;
    }

    if multiplicative_antiderivative_factor_multiset_matches(ctx, left, right, var_name) {
        return true;
    }

    if quotient_term_matches(ctx, left, right) {
        return true;
    }

    for builtin in [BuiltinFn::Arctan, BuiltinFn::Atan] {
        let Some((left_scale, left_arg)) = scaled_unary_builtin_rational_target(ctx, left, builtin)
        else {
            continue;
        };
        let Some((right_scale, right_arg)) =
            scaled_unary_builtin_rational_target(ctx, right, builtin)
        else {
            continue;
        };
        if left_scale == right_scale && exprs_match(ctx, left_arg, right_arg) {
            return true;
        }
    }

    false
}

fn multiplicative_antiderivative_factor_multiset_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left);
    let right_factors = cas_math::expr_nary::mul_leaves(ctx, right);
    if left_factors.len() != right_factors.len() || left_factors.len() < 2 {
        return false;
    }

    let mut used = vec![false; right_factors.len()];
    'outer: for left_factor in left_factors {
        for (index, right_factor) in right_factors.iter().copied().enumerate() {
            if !used[index]
                && antiderivative_factor_matches(ctx, left_factor, right_factor, var_name)
            {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn antiderivative_factor_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if exprs_match(ctx, left, right) || quotient_term_matches(ctx, left, right) {
        return true;
    }

    let left_terms = cas_math::expr_nary::add_terms_signed(ctx, left);
    let right_terms = cas_math::expr_nary::add_terms_signed(ctx, right);
    left_terms.len() > 1
        && right_terms.len() > 1
        && additive_term_multiset_matches(ctx, left, right, var_name)
}

fn quotient_term_matches(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some((left_num, left_den)) = quotient_numerator_denominator(ctx, left) else {
        return false;
    };
    let Some((right_num, right_den)) = quotient_numerator_denominator(ctx, right) else {
        return false;
    };

    term_shape_matches(ctx, left_num, right_num) && term_shape_matches(ctx, left_den, right_den)
}

fn term_shape_matches(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    exprs_match(ctx, left, right)
        || multiplicative_factor_multiset_matches(ctx, left, right)
        || format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: left
            }
        ) == format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: right
            }
        )
}

fn multiplicative_factor_multiset_matches(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left);
    let right_factors = cas_math::expr_nary::mul_leaves(ctx, right);
    if left_factors.len() != right_factors.len() || left_factors.len() < 2 {
        return false;
    }

    let mut used = vec![false; right_factors.len()];
    'outer: for left_factor in left_factors {
        for (index, right_factor) in right_factors.iter().enumerate() {
            if !used[index] && exprs_match(ctx, left_factor, *right_factor) {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn additive_term_multiset_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if antiderivative_term_matches(ctx, left, right, var_name) {
        return true;
    }

    let left_terms = cas_math::expr_nary::add_terms_signed(ctx, left);
    let right_terms = cas_math::expr_nary::add_terms_signed(ctx, right);
    if left_terms.len() != right_terms.len() || left_terms.len() < 2 {
        return false;
    }

    let mut used = vec![false; right_terms.len()];
    'outer: for (left_term, left_sign) in left_terms {
        for (index, (right_term, right_sign)) in right_terms.iter().copied().enumerate() {
            if !used[index]
                && signed_antiderivative_term_matches(
                    ctx, left_term, left_sign, right_term, right_sign, var_name,
                )
            {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn strip_additive_constants_for_antiderivative_match(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return expr;
    }

    let mut removed_constant = false;
    let mut retained_terms = Vec::new();
    for (term, sign) in terms {
        if cas_math::expr_predicates::contains_named_var(ctx, term, var_name) {
            let signed_term = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => ctx.add(Expr::Neg(term)),
            };
            retained_terms.push(signed_term);
        } else {
            removed_constant = true;
        }
    }

    if removed_constant {
        cas_math::expr_nary::build_balanced_add(ctx, &retained_terms)
    } else {
        expr
    }
}

fn hyperbolic_polynomial_term_parts(
    ctx: &mut Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<(BuiltinFn, ExprId, Polynomial)> {
    let (scale, core) = signed_rational_scaled_term(ctx, term, sign);
    if scale.is_zero() {
        return None;
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, core);
    let mut hyperbolic_index = None;
    let mut hyperbolic_builtin = None;
    let mut hyperbolic_arg = None;
    for (index, factor) in factors.iter().copied().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let Some(builtin @ (BuiltinFn::Sinh | BuiltinFn::Cosh)) = ctx.builtin_of(*fn_id) else {
            continue;
        };
        if hyperbolic_index.is_some() {
            return None;
        }
        hyperbolic_index = Some(index);
        hyperbolic_builtin = Some(builtin);
        hyperbolic_arg = Some(args[0]);
    }

    let cofactor_factors: Vec<_> = factors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, factor)| (Some(index) != hyperbolic_index).then_some(factor))
        .collect();
    let cofactor_expr = match cofactor_factors.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => cas_math::expr_nary::build_balanced_mul(ctx, &cofactor_factors),
    };
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor_expr, var_name).ok()?;
    let scale_poly = Polynomial::new(vec![scale], var_name.to_string());
    Some((
        hyperbolic_builtin?,
        hyperbolic_arg?,
        scale_poly.mul(&cofactor_poly),
    ))
}

fn add_hyperbolic_polynomial_part(
    ctx: &Context,
    parts: &mut Vec<(BuiltinFn, ExprId, Polynomial)>,
    builtin: BuiltinFn,
    arg: ExprId,
    polynomial: Polynomial,
) {
    for (existing_builtin, existing_arg, existing_polynomial) in parts.iter_mut() {
        if *existing_builtin == builtin && expr_eq(ctx, *existing_arg, arg) {
            *existing_polynomial = existing_polynomial.add(&polynomial);
            return;
        }
    }
    parts.push((builtin, arg, polynomial));
}

fn hyperbolic_polynomial_parts(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Vec<(BuiltinFn, ExprId, Polynomial)>> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.is_empty() || terms.len() > 6 {
        return None;
    }

    let mut parts = Vec::new();
    for (term, sign) in terms {
        let (builtin, arg, polynomial) =
            hyperbolic_polynomial_term_parts(ctx, term, sign, var_name)?;
        add_hyperbolic_polynomial_part(ctx, &mut parts, builtin, arg, polynomial);
    }
    parts.retain(|(_, _, polynomial)| !polynomial.is_zero());
    Some(parts)
}

fn hyperbolic_polynomial_additive_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some(left_parts) = hyperbolic_polynomial_parts(ctx, left, var_name) else {
        return false;
    };
    let Some(right_parts) = hyperbolic_polynomial_parts(ctx, right, var_name) else {
        return false;
    };
    if left_parts.len() != right_parts.len() {
        return false;
    }

    let mut used = vec![false; right_parts.len()];
    'outer: for (left_builtin, left_arg, left_poly) in left_parts {
        for (index, (right_builtin, right_arg, right_poly)) in right_parts.iter().enumerate() {
            if !used[index]
                && left_builtin == *right_builtin
                && expr_eq(ctx, left_arg, *right_arg)
                && left_poly == *right_poly
            {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

fn signed_antiderivative_term_matches(
    ctx: &mut Context,
    left_term: ExprId,
    left_sign: cas_math::expr_nary::Sign,
    right_term: ExprId,
    right_sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> bool {
    let (left_scale, left_core) = signed_rational_scaled_term(ctx, left_term, left_sign);
    let (right_scale, right_core) = signed_rational_scaled_term(ctx, right_term, right_sign);
    if left_scale == right_scale
        && antiderivative_term_matches(ctx, left_core, right_core, var_name)
    {
        return true;
    }

    let signed_left = scale_expr_by_rational(ctx, left_core, left_scale);
    let signed_right = scale_expr_by_rational(ctx, right_core, right_scale);
    signed_rational_polynomial_terms_match(ctx, signed_left, signed_right, var_name)
        || antiderivative_term_matches(ctx, signed_left, signed_right, var_name)
}

fn signed_rational_scaled_term(
    ctx: &mut Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
) -> (BigRational, ExprId) {
    let (scale, core) = rational_scaled_term(ctx, term);
    match sign {
        cas_math::expr_nary::Sign::Pos => (scale, core),
        cas_math::expr_nary::Sign::Neg => (-scale, core),
    }
}

fn rational_scaled_term(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (scale, core) = rational_scaled_term(ctx, inner);
            (-scale, core)
        }
        Expr::Div(num, den) => {
            let (num_scale, num_core) = rational_scaled_term(ctx, num);
            if let Some(den_scale) = cas_math::numeric_eval::as_rational_const(ctx, den) {
                if !den_scale.is_zero() {
                    return (num_scale / den_scale, num_core);
                }
            }
            (num_scale, ctx.add(Expr::Div(num_core, den)))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut cores = Vec::new();
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                let (factor_scale, factor_core) = rational_scaled_term(ctx, factor);
                scale *= factor_scale;
                if !expr_is_one(ctx, factor_core) {
                    cores.push(factor_core);
                }
            }
            let core = match cores.as_slice() {
                [] => ctx.num(1),
                [single] => *single,
                _ => cas_math::expr_nary::build_balanced_mul(ctx, &cores),
            };
            (scale, core)
        }
        _ => {
            if let Some(scale) = cas_math::numeric_eval::as_rational_const(ctx, expr) {
                (scale, ctx.num(1))
            } else {
                (BigRational::one(), expr)
            }
        }
    }
}

fn scaled_reciprocal_sqrt_polynomial_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, Polynomial, ExprId)> {
    let (scale, core) = rational_scaled_term(ctx, term);
    if scale.is_zero() {
        return None;
    }

    let base = reciprocal_sqrt_like_base(ctx, core)
        .or_else(|| rationalized_reciprocal_sqrt_base(ctx, core))?;
    let vars = cas_ast::collect_variables(ctx, base);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    if poly.is_zero() {
        return None;
    }

    Some((scale, poly, base))
}

fn reciprocal_sqrt_product_denominator_base(
    ctx: &mut Context,
    denominator: ExprId,
) -> Option<(BigRational, ExprId)> {
    let mut scale = BigRational::one();
    let mut bases = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            if value.is_zero() {
                return None;
            }
            scale *= value;
            continue;
        }
        let base = sqrt_like_base(ctx, factor)?;
        bases.push(base);
    }

    if bases.len() < 2 {
        return None;
    }

    Some((scale, cas_math::expr_nary::build_balanced_mul(ctx, &bases)))
}

fn scaled_reciprocal_sqrt_product_or_power_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, Polynomial, ExprId, bool)> {
    let term = cas_ast::hold::strip_all_holds(ctx, term);
    let (mut scale, core) = rational_scaled_term(ctx, term);
    let core = cas_ast::hold::strip_all_holds(ctx, core);

    let (base, product_denominator) = if let Some(base) = reciprocal_sqrt_like_base(ctx, core) {
        (base, false)
    } else {
        let Expr::Div(num, den) = ctx.get(core).clone() else {
            return None;
        };
        let (num_scale, num_core) = rational_scaled_term(ctx, num);
        if !expr_is_one(ctx, num_core) {
            return None;
        }
        let (den_scale, base) = reciprocal_sqrt_product_denominator_base(ctx, den)?;
        scale *= num_scale;
        scale /= den_scale;
        (base, true)
    };

    let vars = cas_ast::collect_variables(ctx, base);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    if poly.is_zero() {
        return None;
    }

    Some((scale, poly, base, product_denominator))
}

fn scaled_reciprocal_sqrt_product_terms_required_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (left_scale, left_poly, _left_base, left_product) =
        scaled_reciprocal_sqrt_product_or_power_term(ctx, left)?;
    let (right_scale, right_poly, _right_base, right_product) =
        scaled_reciprocal_sqrt_product_or_power_term(ctx, right)?;

    if !left_product && !right_product {
        return None;
    }
    if left_poly.var != right_poly.var || !same_nonzero_sign(&left_scale, &right_scale) {
        return None;
    }

    let left_scale_sq = left_scale.clone() * left_scale;
    let right_scale_sq = right_scale.clone() * right_scale;
    if scale_polynomial(&right_poly, &left_scale_sq)
        != scale_polynomial(&left_poly, &right_scale_sq)
    {
        return None;
    }

    Some(Vec::new())
}

fn rationalized_reciprocal_sqrt_base(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(term).clone() else {
        return None;
    };

    let mut sqrt_base = None;
    let mut remaining_numerator_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, num) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if sqrt_base.is_none() {
            if let Some(base) = sqrt_like_base(ctx, factor) {
                if expr_eq(ctx, base, den) || exprs_equivalent(ctx, base, den) {
                    sqrt_base = Some(base);
                    continue;
                }
            }
        }
        remaining_numerator_factors.push(factor);
    }

    let base = sqrt_base?;
    let remaining = cas_math::expr_nary::build_balanced_mul(ctx, &remaining_numerator_factors);
    expr_is_one(ctx, remaining).then_some(base)
}

fn polynomial_over_reciprocal_sqrt_term(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let term = cas_ast::hold::strip_all_holds(ctx, term);
    match ctx.get(term).clone() {
        Expr::Neg(inner) => {
            let (numerator, base) = polynomial_over_reciprocal_sqrt_term(ctx, inner, var_name)?;
            Some((numerator.neg(), base))
        }
        Expr::Div(num, den) => {
            let den = cas_ast::hold::strip_all_holds(ctx, den);
            if let Some(base) = sqrt_like_base(ctx, den) {
                let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
                let base = Polynomial::from_expr(ctx, base, var_name).ok()?;
                if !base.is_zero() {
                    return Some((numerator, base));
                }
            }
            if let Some((numerator, base)) =
                polynomial_over_rationalized_sqrt_term(ctx, num, den, var_name)
            {
                return Some((numerator, base));
            }
            None
        }
        Expr::Mul(_, _) => {
            let mut numerator = Polynomial::one(var_name.to_string());
            let mut base_poly = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
                let factor = cas_ast::hold::strip_all_holds(ctx, factor);
                if base_poly.is_none() {
                    if let Some(base) = reciprocal_sqrt_like_base(ctx, factor) {
                        let base = Polynomial::from_expr(ctx, base, var_name).ok()?;
                        if base.is_zero() {
                            return None;
                        }
                        base_poly = Some(base);
                        continue;
                    }
                }

                let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
                numerator = numerator.mul(&factor_poly);
            }

            Some((numerator, base_poly?))
        }
        _ => {
            let base = reciprocal_sqrt_like_base(ctx, term)?;
            let base = Polynomial::from_expr(ctx, base, var_name).ok()?;
            (!base.is_zero()).then(|| (Polynomial::one(var_name.to_string()), base))
        }
    }
}

fn polynomial_over_rationalized_sqrt_term(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if denominator_poly.is_zero() {
        return None;
    }

    let mut numerator = Polynomial::one(var_name.to_string());
    let mut found_matching_sqrt = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, num) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if !found_matching_sqrt {
            if let Some(base) = sqrt_like_base(ctx, factor) {
                let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
                if base_poly == denominator_poly || exprs_equivalent(ctx, base, den) {
                    found_matching_sqrt = true;
                    continue;
                }
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        numerator = numerator.mul(&factor_poly);
    }

    found_matching_sqrt.then_some((numerator, denominator_poly))
}

fn same_nonzero_sign(left: &BigRational, right: &BigRational) -> bool {
    (left > &BigRational::zero() && right > &BigRational::zero())
        || (left < &BigRational::zero() && right < &BigRational::zero())
}

fn scaled_reciprocal_sqrt_polynomial_terms_required_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (left_scale, left_poly, left_base) = scaled_reciprocal_sqrt_polynomial_term(ctx, left)?;
    let (right_scale, right_poly, right_base) = scaled_reciprocal_sqrt_polynomial_term(ctx, right)?;

    if left_poly.var != right_poly.var || !same_nonzero_sign(&left_scale, &right_scale) {
        return None;
    }

    let left_scale_sq = left_scale.clone() * left_scale;
    let right_scale_sq = right_scale.clone() * right_scale;
    if scale_polynomial(&right_poly, &left_scale_sq)
        != scale_polynomial(&left_poly, &right_scale_sq)
    {
        return None;
    }

    Some(vec![
        crate::ImplicitCondition::Positive(left_base),
        crate::ImplicitCondition::Positive(right_base),
    ])
}

fn scaled_sqrt_polynomial_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, Polynomial, ExprId)> {
    let (scale, core) = rational_scaled_term(ctx, term);
    if scale.is_zero() {
        return None;
    }

    let base = sqrt_like_base(ctx, core)?;
    let vars = cas_ast::collect_variables(ctx, base);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    if poly.is_zero() {
        return None;
    }

    Some((scale, poly, base))
}

fn scaled_sqrt_polynomial_over_sqrt_terms_required_conditions(
    ctx: &mut Context,
    sqrt_term: ExprId,
    quotient_term: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (sqrt_scale, sqrt_poly, sqrt_base) = scaled_sqrt_polynomial_term(ctx, sqrt_term)?;
    let (quotient_num, quotient_base) =
        polynomial_over_reciprocal_sqrt_term(ctx, quotient_term, &sqrt_poly.var)?;

    if sqrt_poly != quotient_base {
        return None;
    }

    let expected_num = scale_polynomial(&sqrt_poly, &sqrt_scale);
    if quotient_num != expected_num {
        return None;
    }

    if strictly_positive_quadratic_base(ctx, sqrt_base) {
        Some(Vec::new())
    } else {
        Some(vec![crate::ImplicitCondition::Positive(sqrt_base)])
    }
}

fn scaled_sqrt_over_base_product_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, ExprId, ExprId)> {
    let (outer_scale, core) = rational_scaled_term(ctx, term);
    if outer_scale.is_zero() {
        return None;
    }
    let core = cas_ast::hold::strip_all_holds(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };

    let (num_scale, num_core) = rational_scaled_term(ctx, num);
    if num_scale.is_zero() {
        return None;
    }
    let num_core = cas_ast::hold::strip_all_holds(ctx, num_core);
    let sqrt_base = sqrt_like_base(ctx, num_core)?;

    let (den_scale, den_core) = rational_scaled_term(ctx, den);
    if den_scale.is_zero() {
        return None;
    }
    let den_factors = cas_math::expr_nary::mul_leaves(ctx, den_core);
    let remaining = remove_denominator_factor_matching_base(ctx, &den_factors, sqrt_base)?;
    let [product_base] = remaining.as_slice() else {
        return None;
    };

    Some((
        outer_scale * num_scale / den_scale,
        sqrt_base,
        *product_base,
    ))
}

fn scaled_reciprocal_sqrt_base_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = rational_scaled_term(ctx, term);
    if scale.is_zero() {
        return None;
    }
    let core = cas_ast::hold::strip_all_holds(ctx, core);
    let base = reciprocal_sqrt_like_base(ctx, core)?;
    Some((scale, base))
}

fn sqrt_over_product_matches_reciprocal_product_root_conditions(
    ctx: &mut Context,
    sqrt_over_product: ExprId,
    reciprocal_root: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (left_scale, sqrt_base, product_base) =
        scaled_sqrt_over_base_product_term(ctx, sqrt_over_product)?;
    let (right_scale, reciprocal_base) = scaled_reciprocal_sqrt_base_term(ctx, reciprocal_root)?;
    if left_scale != right_scale {
        return None;
    }

    let vars = cas_ast::collect_variables(ctx, sqrt_base);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let sqrt_poly = Polynomial::from_expr(ctx, sqrt_base, var_name).ok()?;
    let product_poly = Polynomial::from_expr(ctx, product_base, var_name).ok()?;
    let reciprocal_poly = Polynomial::from_expr(ctx, reciprocal_base, var_name).ok()?;
    if sqrt_poly.is_zero() || product_poly.is_zero() {
        return None;
    }

    let expected = sqrt_poly.mul(&product_poly.mul(&product_poly));
    if reciprocal_poly != expected {
        return None;
    }

    Some(vec![
        crate::ImplicitCondition::Positive(sqrt_base),
        crate::ImplicitCondition::Positive(product_base),
    ])
}

fn remove_denominator_factor_matching_base(
    ctx: &Context,
    factors: &[ExprId],
    base: ExprId,
) -> Option<Vec<ExprId>> {
    let index = factors
        .iter()
        .position(|factor| exprs_match(ctx, *factor, base))?;
    Some(
        factors
            .iter()
            .enumerate()
            .filter_map(|(candidate_index, factor)| (candidate_index != index).then_some(*factor))
            .collect(),
    )
}

fn quotient_numerator_denominator(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => Some((*numerator, *denominator)),
        _ => None,
    }
}

fn quotient_or_term_numerator_denominator(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => (expr, ctx.num(1)),
    }
}

fn denominator_factors_without_one(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    if expr_is_one(ctx, expr) {
        Vec::new()
    } else {
        cas_math::expr_nary::mul_leaves(ctx, expr).to_vec()
    }
}

fn denominator_product_matches_with_rational_content(
    ctx: &mut Context,
    expected: ExprId,
    actual: ExprId,
) -> bool {
    if exprs_match(ctx, expected, actual) || term_shape_matches(ctx, expected, actual) {
        return true;
    }

    let mut expected_scale = BigRational::one();
    let mut expected_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expected) {
        if let Some(scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            expected_scale *= scale;
        } else {
            expected_factors.push(factor);
        }
    }

    let mut actual_factors = cas_math::expr_nary::mul_leaves(ctx, actual).to_vec();
    let mut index = 0;
    while index < expected_factors.len() {
        if let Some(pos) = actual_factors
            .iter()
            .position(|actual_factor| exprs_match(ctx, expected_factors[index], *actual_factor))
        {
            expected_factors.remove(index);
            actual_factors.remove(pos);
        } else {
            index += 1;
        }
    }

    match (expected_factors.as_slice(), actual_factors.as_slice()) {
        ([], []) => expected_scale.is_one(),
        ([expected_factor], [actual_factor]) => {
            let scaled_expected =
                scale_expr_by_rational(ctx, *expected_factor, expected_scale.clone());
            exprs_match(ctx, scaled_expected, *actual_factor)
                || term_shape_matches(ctx, scaled_expected, *actual_factor)
        }
        ([], [actual_factor]) => {
            let expected_constant = ctx.add(Expr::Number(expected_scale));
            exprs_match(ctx, expected_constant, *actual_factor)
        }
        _ => false,
    }
}

fn signed_quotient_numerator_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (numerator, denominator) = quotient_numerator_denominator(ctx, inner)?;
            let numerator = ctx.add(Expr::Neg(numerator));
            Some((numerator, denominator))
        }
        _ => quotient_numerator_denominator(ctx, expr),
    }
}

fn reciprocal_half_power_shared_denominator_terms_cancel(
    ctx: &mut Context,
    reciprocal_term: ExprId,
    expanded_term: ExprId,
) -> Option<()> {
    let (reciprocal_scale, reciprocal_core) = rational_scaled_term(ctx, reciprocal_term);
    let (expanded_scale, expanded_core) = rational_scaled_term(ctx, expanded_term);
    let (reciprocal_num, reciprocal_den) =
        quotient_or_term_numerator_denominator(ctx, reciprocal_core);
    let (expanded_num, expanded_den) = quotient_or_term_numerator_denominator(ctx, expanded_core);

    let base = reciprocal_sqrt_like_base(ctx, reciprocal_num)?;
    let expanded_base = sqrt_like_base(ctx, expanded_num)?;
    if !exprs_match(ctx, base, expanded_base) {
        return None;
    }

    if reciprocal_scale == expanded_scale {
        let reciprocal_den_factors = denominator_factors_without_one(ctx, reciprocal_den);
        let expanded_den_factors = denominator_factors_without_one(ctx, expanded_den);
        if let Some(expanded_without_base) =
            remove_denominator_factor_matching_base(ctx, &expanded_den_factors, base)
        {
            if denominator_factor_multiset_matches(
                ctx,
                &reciprocal_den_factors,
                &expanded_without_base,
            ) {
                return Some(());
            }
        }
    }

    if reciprocal_scale.is_zero() {
        return None;
    }
    let expected_den_core = if expr_is_one(ctx, reciprocal_den) {
        base
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[base, reciprocal_den])
    };
    let expected_den =
        scale_expr_by_rational(ctx, expected_den_core, expanded_scale / reciprocal_scale);
    denominator_product_matches_with_rational_content(ctx, expected_den, expanded_den).then_some(())
}

fn strictly_positive_quadratic_base(ctx: &Context, base: ExprId) -> bool {
    let vars = cas_ast::collect_variables(ctx, base);
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };
    let Ok(poly) = Polynomial::from_expr(ctx, base, var_name) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 || poly.coeffs[2] <= BigRational::zero() {
        return false;
    }

    let four = BigRational::from_integer(4.into());
    let discriminant = poly.coeffs[1].clone() * poly.coeffs[1].clone()
        - four * poly.coeffs[2].clone() * poly.coeffs[0].clone();
    discriminant < BigRational::zero()
}

fn positive_denominator_power_parts(
    ctx: &Context,
    denominator: ExprId,
) -> Option<(ExprId, BigRational)> {
    positive_power_factor_parts(ctx, denominator)
}

fn positive_power_factor_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    if let Some(base) = sqrt_like_base(ctx, expr) {
        return Some((base, BigRational::new(1.into(), 2.into())));
    }

    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_math::numeric_eval::as_rational_const(ctx, *exponent)?;
    (exponent > BigRational::zero()).then_some((*base, exponent))
}

fn split_positive_power_factor_from_product(
    ctx: &mut Context,
    product: ExprId,
) -> Option<(ExprId, BigRational, ExprId)> {
    if let Some((base, exponent)) = positive_power_factor_parts(ctx, product) {
        return Some((base, exponent, ctx.num(1)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, product).to_vec();
    for (index, factor) in factors.iter().copied().enumerate() {
        let Some((base, exponent)) = positive_power_factor_parts(ctx, factor) else {
            continue;
        };
        let remaining: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        return Some((
            base,
            exponent,
            cas_math::expr_nary::build_balanced_mul(ctx, &remaining),
        ));
    }

    None
}

fn power_numerator_over_power_denominator_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, ExprId, ExprId, BigRational, BigRational)> {
    let (scale, core) = rational_scaled_term(ctx, term);
    let (numerator, denominator) = quotient_numerator_denominator(ctx, core)?;
    let (power_base, numerator_exponent, remaining_numerator) =
        split_positive_power_factor_from_product(ctx, numerator)?;
    let (denominator_base, denominator_exponent) =
        positive_denominator_power_parts(ctx, denominator)?;
    if !exprs_match(ctx, power_base, denominator_base) {
        return None;
    }

    Some((
        scale,
        remaining_numerator,
        denominator_base,
        numerator_exponent,
        denominator_exponent,
    ))
}

fn numerator_over_power_denominator_term(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(BigRational, ExprId, ExprId, BigRational)> {
    let (scale, core) = rational_scaled_term(ctx, term);
    let (numerator, denominator) = quotient_numerator_denominator(ctx, core)?;
    let (denominator_base, denominator_exponent) =
        positive_denominator_power_parts(ctx, denominator)?;
    Some((scale, numerator, denominator_base, denominator_exponent))
}

fn power_numerator_shifted_power_denominator_terms_required_conditions(
    ctx: &mut Context,
    power_term: ExprId,
    plain_term: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (power_scale, power_numerator, power_base, numerator_exponent, power_denominator_exponent) =
        power_numerator_over_power_denominator_term(ctx, power_term)?;
    let (plain_scale, plain_numerator, plain_base, plain_denominator_exponent) =
        numerator_over_power_denominator_term(ctx, plain_term)?;
    if power_scale != plain_scale
        || !exprs_match(ctx, power_numerator, plain_numerator)
        || !exprs_match(ctx, power_base, plain_base)
    {
        return None;
    }

    if power_denominator_exponent - numerator_exponent != plain_denominator_exponent {
        return None;
    }

    let required_conditions = if strictly_positive_quadratic_base(ctx, power_base) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(power_base)]
    };
    Some(required_conditions)
}

fn half_integer_offset(value: &BigRational) -> Option<usize> {
    let offset = value - BigRational::new(1.into(), 2.into());
    if offset.is_negative() || !offset.is_integer() {
        return None;
    }
    offset.to_integer().try_into().ok()
}

fn polynomial_power(poly: &Polynomial, exponent: usize) -> Polynomial {
    let mut result = Polynomial::one(poly.var.clone());
    for _ in 0..exponent {
        result = result.mul(poly);
    }
    result
}

fn scaled_polynomial_power(
    poly: &Polynomial,
    base: &Polynomial,
    max_exponent: usize,
) -> Option<(BigRational, usize)> {
    if poly.is_zero() || base.is_zero() || poly.var != base.var {
        return None;
    }

    for exponent in 0..=max_exponent {
        let base_power = polynomial_power(base, exponent);
        if base_power.is_zero() || base_power.degree() != poly.degree() {
            continue;
        }
        let scale = poly.leading_coeff() / base_power.leading_coeff();
        if scale.is_zero() {
            continue;
        }
        if scale_polynomial(&base_power, &scale) == *poly {
            return Some((scale, exponent));
        }
    }

    None
}

fn positive_half_power_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    if let Some(base) = sqrt_like_base(ctx, expr) {
        return Some((base, BigRational::new(1.into(), 2.into())));
    }

    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_math::numeric_eval::as_rational_const(ctx, *exponent)?;
    half_integer_offset(&exponent)?;
    (exponent > BigRational::zero()).then_some((*base, exponent))
}

fn positive_half_power_product_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, BigRational)> {
    if let Some(parts) = positive_half_power_parts(ctx, expr) {
        return Some((parts.0, parts.1, BigRational::one()));
    }

    let mut base = None;
    let mut exponent = BigRational::zero();
    let mut scale = BigRational::one();
    let mut other_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            if value.is_zero() {
                return None;
            }
            scale *= value;
            continue;
        }
        if let Some((factor_base, factor_exponent)) = positive_half_power_parts(ctx, factor) {
            if let Some(existing_base) = base {
                if !exprs_match(ctx, existing_base, factor_base) {
                    return None;
                }
            } else {
                base = Some(factor_base);
            }
            exponent += factor_exponent;
        } else {
            other_factors.push(factor);
        }
    }

    let base = base?;
    for factor in other_factors {
        if exprs_match(ctx, factor, base) {
            exponent += BigRational::one();
            continue;
        }

        let Expr::Pow(factor_base, factor_exp) = ctx.get(factor) else {
            return None;
        };
        let factor_exp = cas_math::numeric_eval::as_rational_const(ctx, *factor_exp)?;
        if !factor_exp.is_positive() || !factor_exp.is_integer() {
            return None;
        }
        if !exprs_match(ctx, *factor_base, base) {
            return None;
        }
        exponent += factor_exp;
    }
    half_integer_offset(&exponent)?;
    Some((base, exponent, scale))
}

fn half_power_polynomial_term(
    ctx: &mut Context,
    core: ExprId,
    var_name: &str,
) -> Option<(Polynomial, ExprId, BigRational)> {
    let core = cas_ast::hold::strip_all_holds(ctx, core);
    match ctx.get(core).clone() {
        Expr::Div(num, den) => {
            if let Some((base, denominator_exponent, denominator_scale)) =
                positive_half_power_product_parts(ctx, den)
            {
                let numerator = Polynomial::from_expr(ctx, num, var_name)
                    .ok()?
                    .div_scalar(&denominator_scale);
                return Some((numerator, base, denominator_exponent));
            }

            let (numerator, base, numerator_denominator_exponent) =
                half_power_polynomial_term(ctx, num, var_name)?;
            let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
            let den_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
            let (denominator_scale, denominator_integer_power) =
                scaled_polynomial_power(&den_poly, &base_poly, 6)?;
            let denominator_exponent = numerator_denominator_exponent
                + BigRational::from_integer((denominator_integer_power as i64).into());
            Some((
                numerator.div_scalar(&denominator_scale),
                base,
                denominator_exponent,
            ))
        }
        Expr::Pow(base, exponent) => {
            let exponent = cas_math::numeric_eval::as_rational_const(ctx, exponent)?;
            let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
            if exponent.is_negative() {
                let denominator_exponent = -exponent;
                half_integer_offset(&denominator_exponent)?;
                Some((
                    Polynomial::one(var_name.to_string()),
                    base,
                    denominator_exponent,
                ))
            } else {
                let offset = half_integer_offset(&exponent)?;
                let numerator = polynomial_power(&base_poly, offset + 1);
                Some((numerator, base, BigRational::new(1.into(), 2.into())))
            }
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            let base = args[0];
            let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
            Some((base_poly, base, BigRational::new(1.into(), 2.into())))
        }
        Expr::Mul(_, _) => {
            let mut numerator = Polynomial::one(var_name.to_string());
            let mut base = None;
            let mut denominator_exponent = BigRational::zero();
            let mut saw_half_power = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, core) {
                let factor = cas_ast::hold::strip_all_holds(ctx, factor);
                if let Some((factor_numerator, factor_base, factor_denominator_exponent)) =
                    half_power_polynomial_term(ctx, factor, var_name)
                {
                    if let Some(existing_base) = base {
                        if !exprs_match(ctx, existing_base, factor_base) {
                            return None;
                        }
                    } else {
                        base = Some(factor_base);
                    }
                    numerator = numerator.mul(&factor_numerator);
                    denominator_exponent += factor_denominator_exponent;
                    saw_half_power = true;
                } else {
                    let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
                    numerator = numerator.mul(&factor_poly);
                }
            }
            saw_half_power.then_some((numerator, base?, denominator_exponent))
        }
        _ => None,
    }
}

/// shared = c * candidate for a single rational constant c.
fn proportional_polynomial_ratio(
    shared: &Polynomial,
    candidate: &Polynomial,
) -> Option<BigRational> {
    if shared.coeffs.len() != candidate.coeffs.len() || candidate.is_zero() {
        return None;
    }
    let mut ratio: Option<BigRational> = None;
    for (left, right) in shared.coeffs.iter().zip(candidate.coeffs.iter()) {
        match (left.is_zero(), right.is_zero()) {
            (true, true) => continue,
            (true, false) | (false, true) => return None,
            (false, false) => {
                let candidate_ratio = left / right;
                if let Some(existing) = &ratio {
                    if *existing != candidate_ratio {
                        return None;
                    }
                } else {
                    ratio = Some(candidate_ratio);
                }
            }
        }
    }
    ratio
}

fn rational_square_root(value: &BigRational) -> Option<BigRational> {
    let numer = value.numer().sqrt();
    let denom = value.denom().sqrt();
    let candidate = BigRational::new(numer, denom);
    (&candidate * &candidate == *value).then_some(candidate)
}

fn power_rational(base: &BigRational, exponent: i32) -> BigRational {
    let mut acc = BigRational::from_integer(1.into());
    for _ in 0..exponent.unsigned_abs() {
        acc *= base;
    }
    if exponent < 0 {
        acc.recip()
    } else {
        acc
    }
}

fn half_power_polynomial_sum_combined(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(Polynomial, ExprId)> {
    let raw_terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if !(2..=4).contains(&raw_terms.len()) {
        return None;
    }

    // Bounded recursive flatten: a rational-scaled additive GROUP
    // distributes exactly (1/2 * (A - B) contributes 1/2*A and -1/2*B),
    // which is how verification residuals keep factored (and nested)
    // antiderivative shapes. Then merge structurally equal cores: their
    // rational scales add exactly, which cancels non-radical pairs
    // (arcsin terms) before the half-power parse sees them.
    let mut worklist: Vec<(BigRational, ExprId, usize)> = Vec::with_capacity(raw_terms.len());
    for (term, sign) in raw_terms {
        let (scale, core) = signed_rational_scaled_term(ctx, term, sign);
        worklist.push((scale, core, 0));
    }
    let mut scaled_terms: Vec<(BigRational, ExprId)> = Vec::new();
    while let Some((scale, core, depth)) = worklist.pop() {
        if scale.is_zero() {
            continue;
        }
        if depth < 3 && matches!(ctx.get(core), Expr::Add(_, _) | Expr::Sub(_, _)) {
            for (sub_term, sub_sign) in cas_math::expr_nary::add_terms_signed(ctx, core) {
                let (sub_scale, sub_core) = signed_rational_scaled_term(ctx, sub_term, sub_sign);
                worklist.push((scale.clone() * sub_scale, sub_core, depth + 1));
            }
            continue;
        }
        if let Some((existing_scale, _)) = scaled_terms
            .iter_mut()
            .find(|(_, existing_core)| exprs_match(ctx, *existing_core, core))
        {
            *existing_scale += scale;
            continue;
        }
        scaled_terms.push((scale, core));
        if scaled_terms.len() > 12 {
            return None;
        }
    }
    scaled_terms.retain(|(scale, _)| !scale.is_zero());
    if scaled_terms.len() > 8 {
        return None;
    }

    let mut parsed_terms = Vec::with_capacity(scaled_terms.len());
    let mut shared_base: Option<ExprId> = None;
    let mut max_denominator_exponent = BigRational::zero();
    let mut var_name: Option<String> = None;
    for (scale, core) in scaled_terms {
        let inferred_var = if let Some(existing_var) = var_name.as_deref() {
            existing_var.to_string()
        } else {
            let vars = cas_ast::collect_variables(ctx, core);
            if vars.len() != 1 {
                return None;
            }
            vars.iter().next()?.to_string()
        };
        let (numerator, base, denominator_exponent) =
            half_power_polynomial_term(ctx, core, &inferred_var)?;
        let mut scale = scale;
        if let Some(existing_base) = shared_base {
            if !exprs_match(ctx, existing_base, base) {
                // Proportional bases rescale exactly: base = shared/c
                // with rational c > 0 whose square root is rational
                // (the 1 - 4x^2 vs 1/4 - x^2 pair from scaled-argument
                // derivatives) gives base^(-d) = c^d * shared^(-d).
                let shared_poly = Polynomial::from_expr(ctx, existing_base, &inferred_var).ok()?;
                let candidate_poly = Polynomial::from_expr(ctx, base, &inferred_var).ok()?;
                let ratio = proportional_polynomial_ratio(&shared_poly, &candidate_poly)?;
                if !ratio.is_positive() {
                    return None;
                }
                let sqrt_ratio = rational_square_root(&ratio)?;
                // c^d with d half-integer: (sqrt c)^(2d), 2d integer.
                let doubled = denominator_exponent.clone() * BigRational::from_integer(2.into());
                if !doubled.is_integer() {
                    return None;
                }
                let exponent = i32::try_from(i64::try_from(&doubled.to_integer()).ok()?).ok()?;
                scale *= power_rational(&sqrt_ratio, exponent);
            }
        } else {
            shared_base = Some(base);
            var_name = Some(inferred_var.clone());
        }
        if denominator_exponent > max_denominator_exponent {
            max_denominator_exponent = denominator_exponent.clone();
        }
        parsed_terms.push((scale, numerator, denominator_exponent, inferred_var));
    }

    let base = shared_base?;
    let var_name = var_name?;
    let base_poly = Polynomial::from_expr(ctx, base, &var_name).ok()?;
    if base_poly.is_zero() {
        return None;
    }

    let mut combined = Polynomial::zero(var_name);
    for (scale, numerator, denominator_exponent, term_var) in parsed_terms {
        if combined.var != term_var {
            return None;
        }
        let exponent_gap = max_denominator_exponent.clone() - denominator_exponent;
        if exponent_gap.is_negative() || !exponent_gap.is_integer() {
            return None;
        }
        let exponent_gap: usize = exponent_gap.to_integer().try_into().ok()?;
        let lifted = numerator.mul(&polynomial_power(&base_poly, exponent_gap));
        combined = combined.add(&scale_polynomial(&lifted, &scale));
    }

    Some((combined, base))
}

fn half_power_base_required_conditions(
    ctx: &Context,
    base: ExprId,
) -> Vec<crate::ImplicitCondition> {
    if strictly_positive_quadratic_base(ctx, base) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(base)]
    }
}

fn half_power_polynomial_sum_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (combined, base) = half_power_polynomial_sum_combined(ctx, expr)?;
    if !combined.is_zero() {
        return None;
    }

    Some(half_power_base_required_conditions(ctx, base))
}

fn half_power_polynomial_sum_nonzero_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (combined, base) = half_power_polynomial_sum_combined(ctx, expr)?;
    (!combined.is_zero()).then(|| half_power_base_required_conditions(ctx, base))
}

fn two_term_residual_difference_terms(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right)),
        Expr::Add(_, _) => {
            let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
            if terms.len() != 2 {
                return None;
            }
            match (terms[0], terms[1]) {
                (
                    (left, cas_math::expr_nary::Sign::Pos),
                    (right, cas_math::expr_nary::Sign::Neg),
                ) => Some((left, right)),
                (
                    (left, cas_math::expr_nary::Sign::Neg),
                    (right, cas_math::expr_nary::Sign::Pos),
                ) => Some((right, left)),
                ((left, left_sign), (right, right_sign)) => {
                    let (left_scale, left_core) = signed_rational_scaled_term(ctx, left, left_sign);
                    let (right_scale, right_core) =
                        signed_rational_scaled_term(ctx, right, right_sign);
                    if left_scale.is_zero()
                        || right_scale.is_zero()
                        || same_nonzero_sign(&left_scale, &right_scale)
                    {
                        return None;
                    }
                    let left_positive = scale_expr_by_rational(ctx, left_core, left_scale.abs());
                    let right_positive = scale_expr_by_rational(ctx, right_core, right_scale.abs());
                    if left_scale.is_positive() {
                        Some((left_positive, right_positive))
                    } else {
                        Some((right_positive, left_positive))
                    }
                }
            }
        }
        _ => None,
    }
}

pub(crate) fn try_reciprocal_half_power_shared_denominator_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Some(result) =
        try_reciprocal_half_power_residual_reciprocal_shifted_difference_root_zero(ctx, expr)
    {
        return Some(result);
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        if cas_math::numeric_eval::as_rational_const(ctx, den).is_some_and(|value| value.is_zero())
        {
            return None;
        }

        let (zero, mut required_conditions) =
            try_reciprocal_half_power_shared_denominator_residual_root_zero(ctx, num)?;
        if cas_math::numeric_eval::as_rational_const(ctx, den).is_none() {
            required_conditions.push(crate::ImplicitCondition::NonZero(den));
        }
        return Some((zero, required_conditions));
    }

    // Mirror of the Div branch: a NONZERO rational scale multiplying the
    // whole sum is irrelevant when the residual is exactly zero
    // (Mul(1/4, Add(...)) roots reach here before Combine Constants
    // normalizes them to Div form, and shortcuts run only once).
    if let Expr::Mul(left, right) = ctx.get(expr).clone() {
        let scaled_inner = [(left, right), (right, left)]
            .into_iter()
            .find_map(|(scale, inner)| {
                let value = cas_math::numeric_eval::as_rational_const(ctx, scale)?;
                (!value.is_zero()).then_some(inner)
            });
        if let Some(inner) = scaled_inner {
            let (zero, required_conditions) =
                try_reciprocal_half_power_shared_denominator_residual_root_zero(ctx, inner)?;
            return Some((zero, required_conditions));
        }
        return None;
    }

    if let Some(required_conditions) = half_power_polynomial_sum_required_conditions(ctx, expr) {
        return Some((ctx.num(0), required_conditions));
    }

    let (left, right) = two_term_residual_difference_terms(ctx, expr)?;

    reciprocal_half_power_shared_denominator_terms_cancel(ctx, left, right)
        .or_else(|| reciprocal_half_power_shared_denominator_terms_cancel(ctx, right, left))
        .map(|_| Vec::new())
        .or_else(|| scaled_reciprocal_sqrt_polynomial_terms_required_conditions(ctx, left, right))
        .or_else(|| scaled_reciprocal_sqrt_product_terms_required_conditions(ctx, left, right))
        .or_else(|| scaled_sqrt_polynomial_over_sqrt_terms_required_conditions(ctx, left, right))
        .or_else(|| scaled_sqrt_polynomial_over_sqrt_terms_required_conditions(ctx, right, left))
        .or_else(|| sqrt_over_product_matches_reciprocal_product_root_conditions(ctx, left, right))
        .or_else(|| sqrt_over_product_matches_reciprocal_product_root_conditions(ctx, right, left))
        .or_else(|| {
            power_numerator_shifted_power_denominator_terms_required_conditions(ctx, left, right)
        })
        .or_else(|| {
            power_numerator_shifted_power_denominator_terms_required_conditions(ctx, right, left)
        })
        .map(|required_conditions| (ctx.num(0), required_conditions))
}

fn reciprocal_half_power_residual_shifted_reciprocal_conditions(
    ctx: &mut Context,
    residual_denominator: ExprId,
    target_denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (compacted_denominator, required_conditions) =
        compact_reciprocal_half_power_residual_additive_passthrough_denominator(
            ctx,
            residual_denominator,
        )?;
    exprs_match(ctx, compacted_denominator, target_denominator).then_some(required_conditions)
}

fn try_reciprocal_half_power_residual_reciprocal_shifted_difference_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(right_inner) => (*left, *right_inner),
            _ => return None,
        },
        _ => return None,
    };

    let left_denominator = unit_reciprocal_denominator(ctx, left)?;
    let right_denominator = unit_reciprocal_denominator(ctx, right)?;
    let required_conditions = reciprocal_half_power_residual_shifted_reciprocal_conditions(
        ctx,
        left_denominator,
        right_denominator,
    )
    .or_else(|| {
        reciprocal_half_power_residual_shifted_reciprocal_conditions(
            ctx,
            right_denominator,
            left_denominator,
        )
    })?;
    Some((ctx.num(0), required_conditions))
}

fn reciprocal_half_power_residual_shifted_reciprocal_mismatch_conditions(
    ctx: &mut Context,
    residual_denominator: ExprId,
    target_denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let denominator = strip_exact_zero_additive_noise_bounded(ctx, residual_denominator, 3);
    let view = cas_math::expr_nary::AddView::from_expr(ctx, denominator);
    let terms: Vec<_> = view.terms.into_iter().collect();
    let term_count = terms.len();
    if !(3..=6).contains(&term_count) {
        return None;
    }

    let full_mask = (1_u32 << term_count) - 1;
    for mask in 1..full_mask {
        let passthrough = build_signed_add_subset(ctx, &terms, mask, false);
        if !exprs_match(ctx, passthrough, target_denominator) {
            continue;
        }

        let residual_candidate = build_residual_subset_candidate(ctx, &terms, mask);
        let Some(required_conditions) =
            half_power_polynomial_sum_nonzero_required_conditions(ctx, residual_candidate)
        else {
            continue;
        };
        return Some(required_conditions);
    }

    None
}

fn compact_reciprocal_half_power_residual_shifted_reciprocal_mismatch_ordered(
    ctx: &mut Context,
    left_denominator: ExprId,
    right_denominator: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let mut required_conditions =
        reciprocal_half_power_residual_shifted_reciprocal_mismatch_conditions(
            ctx,
            left_denominator,
            right_denominator,
        )?;
    required_conditions.push(crate::ImplicitCondition::NonZero(left_denominator));
    required_conditions.push(crate::ImplicitCondition::NonZero(right_denominator));

    let numerator = ctx.add(Expr::Sub(right_denominator, left_denominator));
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[left_denominator, right_denominator]);
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some((compact, required_conditions))
}

pub(crate) fn try_reciprocal_half_power_residual_reciprocal_shifted_difference_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(right_inner) => (*left, *right_inner),
            _ => return None,
        },
        _ => return None,
    };

    let left_denominator = unit_reciprocal_denominator(ctx, left)?;
    let right_denominator = unit_reciprocal_denominator(ctx, right)?;
    compact_reciprocal_half_power_residual_shifted_reciprocal_mismatch_ordered(
        ctx,
        left_denominator,
        right_denominator,
    )
    .or_else(|| {
        compact_reciprocal_half_power_residual_shifted_reciprocal_mismatch_ordered(
            ctx,
            right_denominator,
            left_denominator,
        )
    })
}

fn compact_reciprocal_half_power_residual_additive_passthrough_denominator(
    ctx: &mut Context,
    denominator: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let denominator = strip_exact_zero_additive_noise_bounded(ctx, denominator, 3);
    let view = cas_math::expr_nary::AddView::from_expr(ctx, denominator);
    let terms: Vec<_> = view.terms.into_iter().collect();
    let term_count = terms.len();
    if !(3..=6).contains(&term_count) {
        return None;
    }

    let full_mask = (1_u32 << term_count) - 1;
    for mask in 1..full_mask {
        let passthrough_mask = full_mask ^ mask;
        if passthrough_mask == 0 {
            continue;
        }

        let residual_candidate = build_residual_subset_candidate(ctx, &terms, mask);
        let Some((_zero, mut required_conditions)) =
            try_reciprocal_half_power_shared_denominator_residual_root_zero(
                ctx,
                residual_candidate,
            )
        else {
            continue;
        };

        let passthrough = build_signed_add_subset(ctx, &terms, mask, false);
        if is_zero_constant(ctx, passthrough) {
            continue;
        }

        required_conditions.push(crate::ImplicitCondition::NonZero(passthrough));
        return Some((passthrough, required_conditions));
    }

    None
}

fn resolve_integrate_calls_for_reciprocal_half_power_residual(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if depth == 0 {
        return None;
    }

    if let Some(call) = crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, expr)
    {
        let mut required_conditions: Vec<_> =
            cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
                ctx,
                call.target,
                &call.var_name,
            )
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .collect();
        required_conditions.extend(
            cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
                ctx,
                call.target,
                &call.var_name,
            )
            .into_iter()
            .filter(|condition| !strictly_positive_quadratic_base(ctx, *condition))
            .map(crate::ImplicitCondition::Positive),
        );
        let result = cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            call.target,
            &call.var_name,
        )?;
        return Some((result, required_conditions));
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, left, depth - 1);
            let right_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, right, depth - 1);
            if left_resolved.is_none() && right_resolved.is_none() {
                return None;
            }
            let (left, mut required_conditions) = left_resolved.unwrap_or((left, Vec::new()));
            let (right, right_conditions) = right_resolved.unwrap_or((right, Vec::new()));
            required_conditions.extend(right_conditions);
            Some((ctx.add(Expr::Add(left, right)), required_conditions))
        }
        Expr::Sub(left, right) => {
            let left_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, left, depth - 1);
            let right_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, right, depth - 1);
            if left_resolved.is_none() && right_resolved.is_none() {
                return None;
            }
            let (left, mut required_conditions) = left_resolved.unwrap_or((left, Vec::new()));
            let (right, right_conditions) = right_resolved.unwrap_or((right, Vec::new()));
            required_conditions.extend(right_conditions);
            Some((ctx.add(Expr::Sub(left, right)), required_conditions))
        }
        Expr::Mul(left, right) => {
            let left_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, left, depth - 1);
            let right_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, right, depth - 1);
            if left_resolved.is_none() && right_resolved.is_none() {
                return None;
            }
            let (left, mut required_conditions) = left_resolved.unwrap_or((left, Vec::new()));
            let (right, right_conditions) = right_resolved.unwrap_or((right, Vec::new()));
            required_conditions.extend(right_conditions);
            Some((ctx.add(Expr::Mul(left, right)), required_conditions))
        }
        Expr::Div(left, right) => {
            let left_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, left, depth - 1);
            let right_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, right, depth - 1);
            if left_resolved.is_none() && right_resolved.is_none() {
                return None;
            }
            let (left, mut required_conditions) = left_resolved.unwrap_or((left, Vec::new()));
            let (right, right_conditions) = right_resolved.unwrap_or((right, Vec::new()));
            required_conditions.extend(right_conditions);
            Some((ctx.add(Expr::Div(left, right)), required_conditions))
        }
        Expr::Pow(left, right) => {
            let left_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, left, depth - 1);
            let right_resolved =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, right, depth - 1);
            if left_resolved.is_none() && right_resolved.is_none() {
                return None;
            }
            let (left, mut required_conditions) = left_resolved.unwrap_or((left, Vec::new()));
            let (right, right_conditions) = right_resolved.unwrap_or((right, Vec::new()));
            required_conditions.extend(right_conditions);
            Some((ctx.add(Expr::Pow(left, right)), required_conditions))
        }
        Expr::Neg(inner) => {
            let (inner, required_conditions) =
                resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, inner, depth - 1)?;
            Some((ctx.add(Expr::Neg(inner)), required_conditions))
        }
        _ => None,
    }
}

pub(crate) fn try_integrate_resolved_reciprocal_half_power_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (resolved, mut required_conditions) =
        resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, expr, 10)?;
    let (result, residual_conditions) =
        try_reciprocal_half_power_shared_denominator_residual_root_zero(ctx, resolved)?;
    required_conditions.extend(residual_conditions);
    Some((result, required_conditions))
}

pub(crate) fn try_integrate_resolved_reciprocal_half_power_residual_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (resolved, mut required_conditions) =
        resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, expr, 10)?;
    let (result, residual_conditions) =
        try_reciprocal_half_power_residual_reciprocal_shifted_difference_compact_mismatch(
            ctx, resolved,
        )?;
    required_conditions.extend(residual_conditions);
    Some((result, required_conditions))
}

pub(crate) fn shifted_integrate_resolved_reciprocal_half_power_residual_passthrough_nonzero_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (resolved, mut required_conditions) =
        resolve_integrate_calls_for_reciprocal_half_power_residual(ctx, expr, 10)?;
    let (_passthrough, passthrough_conditions) =
        compact_reciprocal_half_power_residual_additive_passthrough_denominator(ctx, resolved)?;
    required_conditions.extend(passthrough_conditions);
    Some(required_conditions)
}

fn expr_is_variable_named(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let Expr::Variable(sym) = ctx.get(expr) else {
        return false;
    };
    ctx.sym_name(*sym) == var_name
}

fn log_plus_numeric_constant_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Ln) {
        return Some(arg);
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if cas_math::numeric_eval::as_rational_const(ctx, *left).is_some() {
                return unary_builtin_arg(ctx, *right, BuiltinFn::Ln);
            }
            if cas_math::numeric_eval::as_rational_const(ctx, *right).is_some() {
                return unary_builtin_arg(ctx, *left, BuiltinFn::Ln);
            }
            None
        }
        Expr::Sub(left, right)
            if cas_math::numeric_eval::as_rational_const(ctx, *right).is_some() =>
        {
            unary_builtin_arg(ctx, *left, BuiltinFn::Ln)
        }
        _ => None,
    }
}

fn unit_over_two_arg_sqrt_base(
    ctx: &Context,
    expr: ExprId,
    expected_arg: ExprId,
    expected_base: ExprId,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if !is_one_constant(ctx, *num) {
        return false;
    }

    let mut numeric_den = BigRational::one();
    let mut matched_arg = false;
    let mut matched_sqrt = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if let Some(coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            numeric_den *= coeff;
            continue;
        }

        if !matched_arg && exprs_match(ctx, factor, expected_arg) {
            matched_arg = true;
            continue;
        }

        if !matched_sqrt {
            if let Some(base) = sqrt_like_base(ctx, factor) {
                if exprs_match(ctx, base, expected_base) {
                    matched_sqrt = true;
                    continue;
                }
            }
        }

        return false;
    }

    matched_arg && matched_sqrt && numeric_den == BigRational::from_integer(2.into())
}

fn diff_sqrt_log_plus_constant_residual_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let base = sqrt_like_base(ctx, call.target)?;
    let log_arg = log_plus_numeric_constant_arg(ctx, base)?;
    if !expr_is_variable_named(ctx, log_arg, &call.var_name) {
        return None;
    }
    if !unit_over_two_arg_sqrt_base(ctx, right, log_arg, base) {
        return None;
    }

    Some(vec![
        crate::ImplicitCondition::Positive(base),
        crate::ImplicitCondition::Positive(log_arg),
    ])
}

pub(crate) fn try_diff_sqrt_log_plus_constant_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => {
            let Expr::Neg(negated_right) = ctx.get(*right) else {
                return None;
            };
            (*left, *negated_right)
        }
        _ => return None,
    };

    let required_conditions = diff_sqrt_log_plus_constant_residual_conditions(ctx, left, right)
        .or_else(|| diff_sqrt_log_plus_constant_residual_conditions(ctx, right, left))?;
    Some((ctx.num(0), required_conditions))
}

fn scaled_reciprocal_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    expected_base: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Some(base) = reciprocal_sqrt_like_base(ctx, expr) {
        return exprs_match(ctx, base, expected_base).then_some(BigRational::one());
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            scaled_reciprocal_sqrt_coeff(ctx, inner, expected_base).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let den = cas_ast::hold::strip_all_holds(ctx, den);
            if let Some(base) = sqrt_like_base(ctx, den) {
                if exprs_match(ctx, base, expected_base) {
                    return cas_math::numeric_eval::as_rational_const(ctx, num);
                }
            }

            let numerator_coeff = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_sqrt = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, expected_base) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            (matched_sqrt && !denominator_coeff.is_zero())
                .then_some(numerator_coeff / denominator_coeff)
        }
        Expr::Mul(_, _) => {
            let mut coeff = BigRational::one();
            let mut matched_reciprocal = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    coeff *= factor_coeff;
                    continue;
                }

                if !matched_reciprocal {
                    if let Some(base) = reciprocal_sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, expected_base) {
                            matched_reciprocal = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            matched_reciprocal.then_some(coeff)
        }
        _ => None,
    }
}

fn sqrt_chain_derivative_scale_matches(
    ctx: &mut Context,
    actual_scale: ExprId,
    sqrt_arg: ExprId,
    var_name: &str,
) -> Option<bool> {
    sqrt_chain_derivative_scale_matches_scaled(
        ctx,
        actual_scale,
        sqrt_arg,
        var_name,
        &BigRational::one(),
    )
}

fn sqrt_chain_derivative_scale_matches_scaled(
    ctx: &mut Context,
    actual_scale: ExprId,
    sqrt_arg: ExprId,
    var_name: &str,
    expected_scale: &BigRational,
) -> Option<bool> {
    let sqrt_arg = cas_ast::hold::strip_all_holds(ctx, sqrt_arg);
    let base = sqrt_like_base(ctx, sqrt_arg)?;
    let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    let derivative = base_poly.derivative();
    if derivative.is_zero() || derivative.coeffs.len() != 1 {
        return None;
    }

    let expected_coeff =
        derivative.coeffs[0].clone() * expected_scale / BigRational::from_integer(2.into());
    let actual_coeff = scaled_reciprocal_sqrt_coeff(ctx, actual_scale, base)?;
    Some(actual_coeff == expected_coeff)
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    first_builtin: BuiltinFn,
    right: ExprId,
    second_builtin: BuiltinFn,
) -> Option<ExprId> {
    let first_arg = unary_builtin_arg(ctx, left, first_builtin)?;
    let second_arg = unary_builtin_arg(ctx, right, second_builtin)?;
    (expr_eq(ctx, first_arg, second_arg) || exprs_equivalent(ctx, first_arg, second_arg))
        .then_some(first_arg)
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
        Expr::Add(left, right) => unary_builtin_arg(ctx, *left, BuiltinFn::Csc)
            .zip(negated_unary_builtin_arg(ctx, *right, BuiltinFn::Cot))
            .or_else(|| {
                unary_builtin_arg(ctx, *right, BuiltinFn::Csc).zip(negated_unary_builtin_arg(
                    ctx,
                    *left,
                    BuiltinFn::Cot,
                ))
            })
            .and_then(|(left_arg, right_arg)| {
                exprs_equivalent(ctx, left_arg, right_arg).then_some(left_arg)
            }),
        _ => None,
    }
}

fn negated_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => unary_builtin_arg(ctx, *inner, builtin),
        Expr::Mul(left, right) if is_negative_one_constant(ctx, *left) => {
            unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Mul(left, right) if is_negative_one_constant(ctx, *right) => {
            unary_builtin_arg(ctx, *left, builtin)
        }
        _ => None,
    }
}

fn add_one_sin_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_one_constant(ctx, *left) {
        unary_builtin_arg(ctx, *right, BuiltinFn::Sin)
    } else if is_one_constant(ctx, *right) {
        unary_builtin_arg(ctx, *left, BuiltinFn::Sin)
    } else {
        None
    }
}

fn plus_or_minus_one_cos_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_one_constant(ctx, *left) => {
            unary_builtin_arg(ctx, *right, BuiltinFn::Cos)
        }
        Expr::Sub(left, right) if is_one_constant(ctx, *right) => {
            unary_builtin_arg(ctx, *left, BuiltinFn::Cos)
        }
        _ => None,
    }
}

fn is_one_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value.is_one())
}

fn is_negative_one_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr)
        .is_some_and(|value| value == -BigRational::one())
}

fn is_zero_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value.is_zero())
}

fn is_nonzero_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| !value.is_zero())
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
    exprs_equivalent(ctx, num_arg, den_arg).then_some(num_arg)
}

fn reciprocal_trig_log_abs_primitive(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = unordered_same_arg_unary_sum(ctx, inner_expr, BuiltinFn::Sec, BuiltinFn::Tan)
    {
        return Some((BuiltinFn::Sec, arg));
    }

    if let Some(arg) = csc_minus_cot_arg(ctx, inner_expr) {
        return Some((BuiltinFn::Csc, arg));
    }

    if let Some(arg) =
        same_arg_quotient_over_builtin_denominator(ctx, inner_expr, add_one_sin_arg, BuiltinFn::Cos)
    {
        return Some((BuiltinFn::Sec, arg));
    }

    if let Some(arg) = same_arg_quotient_over_builtin_denominator(
        ctx,
        inner_expr,
        plus_or_minus_one_cos_arg,
        BuiltinFn::Sin,
    ) {
        return Some((BuiltinFn::Csc, arg));
    }

    None
}

fn ln_abs_inner(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let ln_arg = unary_builtin_arg(ctx, expr, BuiltinFn::Ln)?;
    let mut inner = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs)?;
    while let Some(unwrapped) = cas_ast::hold::unwrap_hold_if_wrapped(ctx, inner) {
        inner = unwrapped;
    }
    Some(inner)
}

fn scaled_ln_abs_inner(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(inner) = ln_abs_inner(ctx, expr) {
        return Some((BigRational::one(), inner));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (scale, inner) = scaled_ln_abs_inner(ctx, *inner)?;
            Some((-scale, inner))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (scale, inner) = scaled_ln_abs_inner(ctx, *num)?;
            Some((scale / den_scale, inner))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_inner = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (factor_scale, inner) = scaled_ln_abs_inner(ctx, factor)?;
                if matched_inner.replace(inner).is_some() {
                    return None;
                }
                scale *= factor_scale;
            }

            Some((scale, matched_inner?))
        }
        _ => None,
    }
}

fn reciprocal_trig_denominator_builtin(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Sec => Some(BuiltinFn::Cos),
        BuiltinFn::Csc => Some(BuiltinFn::Sin),
        _ => None,
    }
}

fn reciprocal_trig_target_coefficient(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<BigRational> {
    if let Some(target_arg) = unary_builtin_arg(ctx, expr, builtin) {
        if exprs_equivalent(ctx, target_arg, arg) {
            return Some(BigRational::one());
        }
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            reciprocal_trig_target_coefficient(ctx, *inner, builtin, arg).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let denominator_builtin = reciprocal_trig_denominator_builtin(builtin)?;
            if let Some(den_arg) = unary_builtin_arg(ctx, *den, denominator_builtin) {
                if exprs_equivalent(ctx, den_arg, arg) {
                    return cas_math::numeric_eval::as_rational_const(ctx, *num);
                }
            }

            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let num_coeff = reciprocal_trig_target_coefficient(ctx, *num, builtin, arg)?;
            Some(num_coeff / den_scale)
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_target = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                if matched_target {
                    return None;
                }
                scale *= reciprocal_trig_target_coefficient(ctx, factor, builtin, arg)?;
                matched_target = true;
            }

            matched_target.then_some(scale)
        }
        _ => None,
    }
}

fn reciprocal_trig_log_abs_quotient_scale_expr(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let denominator_builtin = reciprocal_trig_denominator_builtin(builtin)?;
    let mut matched_target = false;
    let mut remaining_denominator_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator) {
        if unary_builtin_arg(ctx, factor, denominator_builtin)
            .is_some_and(|factor_arg| exprs_equivalent(ctx, factor_arg, arg))
        {
            if matched_target {
                return None;
            }
            matched_target = true;
            continue;
        }

        remaining_denominator_factors.push(factor);
    }

    if !matched_target {
        return None;
    }

    if remaining_denominator_factors.is_empty() {
        return Some(numerator);
    }

    let remaining_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &remaining_denominator_factors);
    Some(ctx.add(Expr::Div(numerator, remaining_denominator)))
}

fn reciprocal_trig_log_abs_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(scale) = scaled_unary_builtin_scale_expr(ctx, expr, builtin, arg) {
        return Some(scale);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let scale = reciprocal_trig_log_abs_target_scale_expr(ctx, inner, builtin, arg)?;
            Some(ctx.add(Expr::Neg(scale)))
        }
        Expr::Div(num, den) => {
            if let Some(scale) =
                reciprocal_trig_log_abs_quotient_scale_expr(ctx, num, den, builtin, arg)
            {
                return Some(scale);
            }

            let scale = reciprocal_trig_log_abs_target_scale_expr(ctx, num, builtin, arg)?;
            if expr_is_one(ctx, den) {
                return Some(scale);
            }
            Some(ctx.add(Expr::Div(scale, den)))
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) =
                    reciprocal_trig_log_abs_target_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn scaled_unary_builtin_rational_target(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(BigRational, ExprId)> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, builtin) {
        return Some((BigRational::one(), arg));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (scale, arg) = scaled_unary_builtin_rational_target(ctx, *inner, builtin)?;
            Some((-scale, arg))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (scale, arg) = scaled_unary_builtin_rational_target(ctx, *num, builtin)?;
            Some((scale / den_scale, arg))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_arg = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (target_scale, arg) =
                    scaled_unary_builtin_rational_target(ctx, factor, builtin)?;
                if matched_arg.replace(arg).is_some() {
                    return None;
                }
                scale *= target_scale;
            }

            Some((scale, matched_arg?))
        }
        _ => None,
    }
}

fn scaled_reciprocal_builtin_rational_target(
    ctx: &Context,
    expr: ExprId,
    denominator_builtin: BuiltinFn,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let numerator_scale = cas_math::numeric_eval::as_rational_const(ctx, *num)?;
            let arg = unary_builtin_arg(ctx, *den, denominator_builtin)?;
            Some((numerator_scale, arg))
        }
        Expr::Neg(inner) => {
            let (scale, arg) =
                scaled_reciprocal_builtin_rational_target(ctx, *inner, denominator_builtin)?;
            Some((-scale, arg))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_arg = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (target_scale, arg) =
                    scaled_reciprocal_builtin_rational_target(ctx, factor, denominator_builtin)?;
                if matched_arg.replace(arg).is_some() {
                    return None;
                }
                scale *= target_scale;
            }

            Some((scale, matched_arg?))
        }
        _ => None,
    }
}

fn reciprocal_trig_derivative_primitive(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    if let Some((scale, arg)) = scaled_unary_builtin_rational_target(ctx, expr, BuiltinFn::Sec) {
        return Some((BuiltinFn::Sec, arg, scale));
    }

    if let Some((scale, arg)) = scaled_unary_builtin_rational_target(ctx, expr, BuiltinFn::Csc) {
        return Some((BuiltinFn::Csc, arg, scale));
    }

    if let Some((scale, arg)) = scaled_reciprocal_builtin_rational_target(ctx, expr, BuiltinFn::Cos)
    {
        return Some((BuiltinFn::Sec, arg, scale));
    }

    if let Some((scale, arg)) = scaled_reciprocal_builtin_rational_target(ctx, expr, BuiltinFn::Sin)
    {
        return Some((BuiltinFn::Csc, arg, scale));
    }

    None
}

fn reciprocal_trig_derivative_pair(builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match builtin {
        BuiltinFn::Sec => Some((BuiltinFn::Sec, BuiltinFn::Tan)),
        BuiltinFn::Csc => Some((BuiltinFn::Csc, BuiltinFn::Cot)),
        _ => None,
    }
}

fn reciprocal_trig_derivative_sign(builtin: BuiltinFn) -> Option<BigRational> {
    match builtin {
        BuiltinFn::Sec => Some(BigRational::one()),
        BuiltinFn::Csc => Some(-BigRational::one()),
        _ => None,
    }
}

fn reciprocal_trig_derivative_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let scale = reciprocal_trig_derivative_target_scale_expr(ctx, inner, builtin, arg)?;
        return Some(ctx.add(Expr::Neg(scale)));
    }

    if let Some(scale) = raw_reciprocal_trig_derivative_target_scale_expr(ctx, expr, builtin, arg) {
        return Some(scale);
    }

    let (first_builtin, second_builtin) = reciprocal_trig_derivative_pair(builtin)?;
    let mut first_seen = false;
    let mut second_seen = false;
    let mut scale_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if !first_seen
            && unary_builtin_arg(ctx, factor, first_builtin)
                .is_some_and(|factor_arg| exprs_equivalent(ctx, factor_arg, arg))
        {
            first_seen = true;
            continue;
        }

        if !second_seen
            && unary_builtin_arg(ctx, factor, second_builtin)
                .is_some_and(|factor_arg| exprs_equivalent(ctx, factor_arg, arg))
        {
            second_seen = true;
            continue;
        }

        scale_factors.push(factor);
    }

    if !first_seen || !second_seen {
        return None;
    }

    if scale_factors.is_empty() {
        Some(ctx.num(1))
    } else {
        Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
    }
}

fn raw_reciprocal_trig_derivative_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            raw_reciprocal_trig_derivative_quotient_scale_expr(ctx, num, den, builtin, arg)
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) =
                    raw_reciprocal_trig_derivative_target_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn raw_reciprocal_trig_derivative_numerator_term_scale_expr(
    ctx: &mut Context,
    term: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, term);
    let mut numerator_index = None;

    for (idx, factor) in factors.iter().enumerate() {
        let Some(factor_arg) = unary_builtin_arg(ctx, *factor, numerator_builtin) else {
            continue;
        };
        if !exprs_equivalent(ctx, factor_arg, arg) {
            continue;
        }
        if numerator_index.replace(idx).is_some() {
            return None;
        }
    }

    let numerator_index = numerator_index?;
    let scale_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();

    Some(if scale_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    })
}

fn raw_reciprocal_trig_derivative_numerator_scale_expr(
    ctx: &mut Context,
    num: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, num);
    if terms.len() <= 1 {
        return raw_reciprocal_trig_derivative_numerator_term_scale_expr(
            ctx,
            num,
            numerator_builtin,
            arg,
        );
    }

    let mut scale_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let scale = raw_reciprocal_trig_derivative_numerator_term_scale_expr(
            ctx,
            term,
            numerator_builtin,
            arg,
        )?;
        scale_terms.push(match sign {
            cas_math::expr_nary::Sign::Pos => scale,
            cas_math::expr_nary::Sign::Neg => ctx.add(Expr::Neg(scale)),
        });
    }

    Some(cas_math::expr_nary::build_balanced_add(ctx, &scale_terms))
}

fn raw_reciprocal_trig_derivative_quotient_scale_expr(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let (numerator_builtin, denominator_builtin) = match builtin {
        BuiltinFn::Sec => (BuiltinFn::Sin, BuiltinFn::Cos),
        BuiltinFn::Csc => (BuiltinFn::Cos, BuiltinFn::Sin),
        _ => return None,
    };

    let Expr::Pow(den_base, den_exp) = ctx.get(den).clone() else {
        return None;
    };
    let power = cas_math::numeric_eval::as_rational_const(ctx, den_exp)?;
    if power != BigRational::from_integer(2.into()) {
        return None;
    }
    let den_arg = unary_builtin_arg(ctx, den_base, denominator_builtin)?;
    if !exprs_equivalent(ctx, den_arg, arg) {
        return None;
    }

    raw_reciprocal_trig_derivative_numerator_scale_expr(ctx, num, numerator_builtin, arg)
}

fn plain_trig_log_abs_primitive(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sin) {
        return Some((BuiltinFn::Cot, arg, BigRational::one()));
    }

    if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cos) {
        return Some((BuiltinFn::Tan, arg, -BigRational::one()));
    }

    None
}

fn plain_trig_quotient_builtins(builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match builtin {
        BuiltinFn::Tan => Some((BuiltinFn::Sin, BuiltinFn::Cos)),
        BuiltinFn::Cot => Some((BuiltinFn::Cos, BuiltinFn::Sin)),
        _ => None,
    }
}

fn scaled_unary_builtin_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(target_arg) = unary_builtin_arg(ctx, expr, builtin) {
        if exprs_equivalent(ctx, target_arg, arg) {
            return Some(ctx.num(1));
        }
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let scale = scaled_unary_builtin_scale_expr(ctx, inner, builtin, arg)?;
            Some(ctx.add(Expr::Neg(scale)))
        }
        Expr::Div(num, den) => {
            let scale = scaled_unary_builtin_scale_expr(ctx, num, builtin, arg)?;
            if expr_is_one(ctx, den) {
                return Some(scale);
            }
            Some(ctx.add(Expr::Div(scale, den)))
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) =
                    scaled_unary_builtin_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn plain_trig_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(scale) = scaled_unary_builtin_scale_expr(ctx, expr, builtin, arg) {
        return Some(scale);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let scale = plain_trig_target_scale_expr(ctx, inner, builtin, arg)?;
            Some(ctx.add(Expr::Neg(scale)))
        }
        Expr::Div(num, den) => {
            let (num_builtin, den_builtin) = plain_trig_quotient_builtins(builtin)?;
            let den_arg = unary_builtin_arg(ctx, den, den_builtin)?;
            if !exprs_equivalent(ctx, den_arg, arg) {
                return None;
            }
            scaled_unary_builtin_scale_expr(ctx, num, num_builtin, arg)
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) = plain_trig_target_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn scale_expr_by_rational(ctx: &mut Context, expr: ExprId, scale: BigRational) -> ExprId {
    if scale.is_one() {
        return expr;
    }

    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(expr));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    cas_math::expr_nary::build_balanced_mul(ctx, &[scale_expr, expr])
}

fn square_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}

fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

fn polynomial_derivative_scale_matches(
    ctx: &Context,
    actual_scale: ExprId,
    trig_arg: ExprId,
    var_name: &str,
    expected_scale: &BigRational,
) -> Option<bool> {
    let actual_poly = Polynomial::from_expr(ctx, actual_scale, var_name).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, trig_arg, var_name).ok()?;
    let expected_poly = scale_polynomial(&arg_poly.derivative(), expected_scale);
    Some(actual_poly == expected_poly)
}

fn diff_call_with_optional_divisor(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr).is_some() {
        let one = one_expr(ctx);
        return Some((expr, one));
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;
    crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, num)?;
    Some((num, den))
}

fn scaled_inverse_reciprocal_trig_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            let builtin = match ctx.builtin_of(*fn_id) {
                Some(
                    builtin @ (BuiltinFn::Arcsec
                    | BuiltinFn::Asec
                    | BuiltinFn::Arccsc
                    | BuiltinFn::Acsc),
                ) => builtin,
                _ => return None,
            };
            (args.len() == 1).then_some((builtin, args[0], BigRational::one()))
        }
        Expr::Neg(inner) => {
            let (builtin, arg, scale) = scaled_inverse_reciprocal_trig_target(ctx, *inner)?;
            Some((builtin, arg, -scale))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (builtin, arg, scale) = scaled_inverse_reciprocal_trig_target(ctx, *num)?;
            Some((builtin, arg, scale / den_scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= value;
                    continue;
                }

                let (builtin, arg, factor_scale) =
                    scaled_inverse_reciprocal_trig_target(ctx, factor)?;
                if matched.replace((builtin, arg)).is_some() {
                    return None;
                }
                scale *= factor_scale;
            }

            let (builtin, arg) = matched?;
            Some((builtin, arg, scale))
        }
        _ => None,
    }
}

fn inverse_reciprocal_trig_derivative_sign(builtin: BuiltinFn) -> Option<BigRational> {
    match builtin {
        BuiltinFn::Arcsec | BuiltinFn::Asec => Some(BigRational::one()),
        BuiltinFn::Arccsc | BuiltinFn::Acsc => Some(-BigRational::one()),
        _ => None,
    }
}

fn abs_arg_matches(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
    unary_builtin_arg(ctx, expr, BuiltinFn::Abs)
        .is_some_and(|abs_arg| exprs_equivalent(ctx, abs_arg, arg))
}

fn inverse_reciprocal_trig_abs_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
    gap: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            inverse_reciprocal_trig_abs_sqrt_coeff(ctx, inner, arg, gap).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let numerator_coeff = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_abs = false;
            let mut matched_sqrt = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_abs && abs_arg_matches(ctx, factor, arg) {
                    matched_abs = true;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, gap) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            (matched_abs && matched_sqrt && !denominator_coeff.is_zero())
                .then_some(numerator_coeff / denominator_coeff)
        }
        _ => None,
    }
}

fn inverse_reciprocal_trig_positive_arg_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
    gap: ExprId,
    arg_derivative: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inverse_reciprocal_trig_positive_arg_sqrt_coeff(
            ctx,
            inner,
            arg,
            gap,
            arg_derivative,
            var_name,
        )
        .map(|coeff| -coeff),
        Expr::Div(num, den) => {
            let actual_num_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_arg = false;
            let mut matched_sqrt = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_arg && exprs_match(ctx, factor, arg) {
                    matched_arg = true;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, gap) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            if !matched_arg || !matched_sqrt || denominator_coeff.is_zero() {
                return None;
            }

            let coeff = polynomial_scale_factor(arg_derivative, &actual_num_poly)?;
            Some(coeff / denominator_coeff)
        }
        _ => None,
    }
}

fn polynomial_scale_factor(expected: &Polynomial, actual: &Polynomial) -> Option<BigRational> {
    if expected.is_zero() {
        return actual.is_zero().then_some(BigRational::one());
    }
    if expected.degree() != actual.degree() {
        return None;
    }

    let mut scale = None;
    for (expected_coeff, actual_coeff) in expected.coeffs.iter().zip(actual.coeffs.iter()) {
        if expected_coeff.is_zero() {
            if !actual_coeff.is_zero() {
                return None;
            }
            continue;
        }
        let candidate = actual_coeff / expected_coeff;
        if scale.as_ref().is_some_and(|scale| scale != &candidate) {
            return None;
        }
        scale = Some(candidate);
    }

    scale
}

fn quadratic_polynomial_is_strictly_greater_than_one(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
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
    let minimum = c - (b.clone() * b) / (four * a);
    minimum > BigRational::one()
}

fn inverse_reciprocal_trig_positive_quadratic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if !quadratic_polynomial_is_strictly_greater_than_one(&arg_poly) {
        return None;
    }

    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let expected_coeff = sign * target_scale / divisor_scale;
    let one = ctx.num(1);
    let arg_sq = square_expr(ctx, arg);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = cas_math::expr_normalization::normalize_condition_expr(ctx, raw_gap);
    let actual_coeff = inverse_reciprocal_trig_positive_arg_sqrt_coeff(
        ctx,
        right,
        arg,
        gap,
        &derivative,
        &call.var_name,
    )?;
    (actual_coeff == expected_coeff).then_some(Vec::new())
}

fn inverse_reciprocal_trig_sqrt_polynomial_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let base = sqrt_like_base(ctx, arg)?;
    let base_poly = Polynomial::from_expr(ctx, base, &call.var_name).ok()?;
    let derivative = base_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let two = BigRational::from_integer(2.into());
    let expected_coeff = sign * target_scale / (two * divisor_scale);
    let gap_poly = base_poly.sub(&Polynomial::one(call.var_name.clone()));
    let gap = gap_poly.to_expr(ctx);
    let actual_coeff = inverse_reciprocal_trig_positive_arg_sqrt_coeff(
        ctx,
        right,
        base,
        gap,
        &derivative,
        &call.var_name,
    )?;
    (actual_coeff == expected_coeff).then_some(vec![crate::ImplicitCondition::Positive(gap)])
}

fn inverse_reciprocal_trig_affine_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    let derivative = arg_poly.derivative();
    let derivative_scale = derivative.coeffs.first().cloned()?;
    if derivative_scale.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let expected_coeff = sign * target_scale * derivative_scale / divisor_scale;
    let one = ctx.num(1);
    let arg_sq = square_expr(ctx, arg);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = cas_math::expr_normalization::normalize_condition_expr(ctx, raw_gap);
    let actual_coeff = inverse_reciprocal_trig_abs_sqrt_coeff(ctx, right, arg, gap)?;
    (actual_coeff == expected_coeff).then_some(vec![crate::ImplicitCondition::Positive(gap)])
}

fn is_constant_scaled_hyperbolic_reciprocal_target(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(expr) {
        return is_constant_scaled_hyperbolic_reciprocal_target(ctx, *inner);
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 4).is_none() {
        return false;
    }

    let mut matched_hyperbolic = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if unary_builtin_arg(ctx, factor, BuiltinFn::Sinh).is_some()
            || unary_builtin_arg(ctx, factor, BuiltinFn::Cosh).is_some()
        {
            if matched_hyperbolic {
                return false;
            }
            matched_hyperbolic = true;
            continue;
        }

        let Some(factor_scale) = cas_ast::views::as_rational_const(ctx, factor, 4) else {
            return false;
        };
        if factor_scale.is_zero() {
            return false;
        }
    }

    matched_hyperbolic
}

fn scaled_expected_matches(
    ctx: &mut Context,
    scale: ExprId,
    builtin: BuiltinFn,
    hyperbolic_arg: ExprId,
    right: ExprId,
) -> bool {
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
    match builtin {
        BuiltinFn::Sinh if expr_is_one(ctx, scale) => {
            let one = one_expr(ctx);
            let reciprocal_tanh = ctx.add(Expr::Div(one, tanh));
            if expr_eq(ctx, reciprocal_tanh, right) {
                return true;
            }

            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let coth_quotient = ctx.add(Expr::Div(cosh, sinh));
            expr_eq(ctx, coth_quotient, right)
        }
        BuiltinFn::Sinh => {
            let expected = ctx.add(Expr::Div(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        BuiltinFn::Cosh if expr_is_one(ctx, scale) => {
            if expr_eq(ctx, tanh, right) {
                return true;
            }

            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let tanh_quotient = ctx.add(Expr::Div(sinh, cosh));
            expr_eq(ctx, tanh_quotient, right)
        }
        BuiltinFn::Cosh => {
            if let Some((target_arg, target_scale)) = tanh_scaled_target(ctx, right) {
                let target_arg = cas_ast::hold::strip_all_holds(ctx, target_arg);
                let target_scale = cas_ast::hold::strip_all_holds(ctx, target_scale);
                let hyperbolic_arg = cas_ast::hold::strip_all_holds(ctx, hyperbolic_arg);
                let scale = cas_ast::hold::strip_all_holds(ctx, scale);
                if exprs_match(ctx, target_arg, hyperbolic_arg)
                    && exprs_match(ctx, target_scale, scale)
                {
                    return true;
                }
            }

            let expected = ctx.add(Expr::Mul(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        _ => false,
    }
}

fn log_abs_hyperbolic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let ln_arg = unary_builtin_arg(ctx, call.target, BuiltinFn::Ln)?;

    let (builtin, hyperbolic_arg) =
        if let Some(abs_arg) = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs) {
            if let Some(arg) = unary_builtin_arg(ctx, abs_arg, BuiltinFn::Sinh) {
                (BuiltinFn::Sinh, arg)
            } else {
                (
                    BuiltinFn::Cosh,
                    unary_builtin_arg(ctx, abs_arg, BuiltinFn::Cosh)?,
                )
            }
        } else {
            (
                BuiltinFn::Cosh,
                unary_builtin_arg(ctx, ln_arg, BuiltinFn::Cosh)?,
            )
        };

    if builtin == BuiltinFn::Cosh {
        if let Some((target_arg, target_scale)) = tanh_scaled_target(ctx, right) {
            if exprs_match(ctx, target_arg, hyperbolic_arg)
                && sqrt_chain_derivative_scale_matches(
                    ctx,
                    target_scale,
                    hyperbolic_arg,
                    &call.var_name,
                )
                .is_some_and(|matched| matched)
            {
                return Some(true);
            }
        }
    }

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        hyperbolic_arg,
        &call.var_name,
    )?;
    let scale = if expr_is_one(ctx, divisor) {
        derivative
    } else if expr_eq(ctx, derivative, divisor) {
        one_expr(ctx)
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(scaled_expected_matches(
        ctx,
        scale,
        builtin,
        hyperbolic_arg,
        right,
    ))
}

pub(crate) fn try_diff_log_abs_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    log_abs_hyperbolic_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

fn sinh_over_cosh_scaled_target(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);

    if let Expr::Mul(_, _) = ctx.get(expr) {
        let mut matched = None;
        let mut scale_factors = Vec::new();
        for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
            let factor = cas_ast::hold::strip_all_holds(ctx, factor);
            if matched.is_none() {
                if let Some((arg, scale)) = sinh_over_cosh_scaled_target(ctx, factor) {
                    matched = Some(arg);
                    if !expr_is_one(ctx, scale) {
                        scale_factors.push(scale);
                    }
                    continue;
                }
            }

            scale_factors.push(factor);
        }

        let arg = matched?;
        let scale = if scale_factors.is_empty() {
            ctx.num(1)
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
        };
        return Some((arg, scale));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let den = cas_ast::hold::strip_all_holds(ctx, den);
    let hyperbolic_arg = unary_builtin_arg(ctx, den, BuiltinFn::Cosh)?;

    let mut matched_sinh = false;
    let mut scale_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, num) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if !matched_sinh {
            if let Some(sinh_arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Sinh) {
                if exprs_match(ctx, sinh_arg, hyperbolic_arg) {
                    matched_sinh = true;
                    continue;
                }
            }
        }

        scale_factors.push(factor);
    }

    if !matched_sinh {
        return None;
    }

    let scale = if scale_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    };
    Some((hyperbolic_arg, scale))
}

fn tanh_scaled_target(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Tanh) {
        return Some((arg, ctx.num(1)));
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (arg, num_scale) = tanh_scaled_target(ctx, num)?;
        let den = cas_ast::hold::strip_all_holds(ctx, den);
        if expr_is_one(ctx, den) {
            return Some((arg, num_scale));
        }
        let scale = if expr_is_one(ctx, num_scale) {
            let one = ctx.num(1);
            ctx.add(Expr::Div(one, den))
        } else {
            ctx.add(Expr::Div(num_scale, den))
        };
        return Some((arg, scale));
    }

    let mut matched_arg = None;
    let mut scale_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if matched_arg.is_none() {
            if let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Tanh) {
                matched_arg = Some(arg);
                continue;
            }
        }

        scale_factors.push(factor);
    }

    let arg = matched_arg?;
    let scale = if scale_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    };
    Some((arg, scale))
}

fn hyperbolic_tanh_common_factor_residual_matches(
    ctx: &mut Context,
    quotient_expr: ExprId,
    target_expr: ExprId,
) -> Option<bool> {
    let (quotient_arg, quotient_scale) = sinh_over_cosh_scaled_target(ctx, quotient_expr)?;
    let (target_arg, target_scale) = tanh_scaled_target(ctx, target_expr)?;
    let quotient_arg = cas_ast::hold::strip_all_holds(ctx, quotient_arg);
    let quotient_scale = cas_ast::hold::strip_all_holds(ctx, quotient_scale);
    let target_arg = cas_ast::hold::strip_all_holds(ctx, target_arg);
    let target_scale = cas_ast::hold::strip_all_holds(ctx, target_scale);

    Some(
        exprs_match(ctx, quotient_arg, target_arg)
            && exprs_match(ctx, quotient_scale, target_scale),
    )
}

fn try_hyperbolic_tanh_common_factor_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    hyperbolic_tanh_common_factor_residual_matches(ctx, left, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

fn log_abs_reciprocal_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    log_abs_reciprocal_trig_target_diff_matches(ctx, call.target, &call.var_name, divisor, right)
}

fn log_abs_reciprocal_trig_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (target_scale, inner_expr) = scaled_ln_abs_inner(ctx, target)?;
    let (builtin, trig_arg) = reciprocal_trig_log_abs_primitive(ctx, inner_expr)?;

    let arg_derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx, trig_arg, var_name,
    )?;
    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let expected_rational_scale = target_scale / divisor_scale;
    let arg_derivative_rational = cas_math::numeric_eval::as_rational_const(ctx, arg_derivative);
    if let Some(arg_scale) = arg_derivative_rational.clone() {
        let expected_scale = expected_rational_scale.clone() * arg_scale;
        if let Some(actual_scale) =
            reciprocal_trig_target_coefficient(ctx, right, builtin, trig_arg)
        {
            return Some(actual_scale == expected_scale);
        }
    }

    let actual_scale = reciprocal_trig_log_abs_target_scale_expr(ctx, right, builtin, trig_arg)?;
    if let Some(matched) = sqrt_chain_derivative_scale_matches_scaled(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    ) {
        return Some(matched);
    }
    if let Some(matched) = polynomial_derivative_scale_matches(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    ) {
        if matched || arg_derivative_rational.is_some() {
            return Some(matched);
        }
    }

    let expected_scale = scale_expr_by_rational(ctx, arg_derivative, expected_rational_scale);
    Some(
        exprs_equivalent(ctx, actual_scale, expected_scale)
            || antiderivative_term_matches(ctx, actual_scale, expected_scale, var_name),
    )
}

fn integrated_log_abs_reciprocal_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let matched = log_abs_reciprocal_trig_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

fn reciprocal_trig_derivative_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (builtin, trig_arg, target_scale) = reciprocal_trig_derivative_primitive(ctx, target)?;
    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let derivative_sign = reciprocal_trig_derivative_sign(builtin)?;
    let expected_rational_scale = target_scale * derivative_sign / divisor_scale;
    let actual_scale = reciprocal_trig_derivative_target_scale_expr(ctx, right, builtin, trig_arg)?;
    polynomial_derivative_scale_matches(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    )
}

fn integrated_reciprocal_trig_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let mut direct_required_conditions = None;
    let mut condition_target = integrate_call.target;
    let antiderivative = if let Some((antiderivative, required_nonzero)) =
        cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
    {
        direct_required_conditions = Some(vec![crate::ImplicitCondition::NonZero(required_nonzero)]);
        antiderivative
    } else if let Some(antiderivative) =
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
        antiderivative
    } else {
        let normalized = crate::rules::calculus::reciprocal_trig_derivative_integrand_quotient(
            ctx,
            integrate_call.target,
        )?;
        condition_target = normalized;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            normalized,
            &integrate_call.var_name,
        )?
    };
    let matched = reciprocal_trig_derivative_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    if let Some(required_conditions) = direct_required_conditions {
        return Some(required_conditions);
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            condition_target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            condition_target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

pub(crate) fn integral_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx, target, var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx, target, var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    required_nonzero.chain(required_positive).collect()
}

pub(crate) fn integral_residual_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = integral_required_conditions(ctx, target, var_name);
    for condition in integral_residual_structural_required_conditions(ctx, target) {
        push_unique_implicit_condition(ctx, &mut conditions, condition);
    }
    conditions
}

pub(crate) fn integral_residual_structural_required_conditions(
    ctx: &mut Context,
    target: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = Vec::new();
    collect_integral_residual_structural_required_conditions(ctx, target, &mut conditions, 16);
    conditions
}

fn collect_integral_residual_structural_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
    conditions: &mut Vec<crate::ImplicitCondition>,
    depth: usize,
) {
    if depth == 0 {
        return;
    }

    if let Some(base) = reciprocal_sqrt_like_base(ctx, expr) {
        for condition in half_power_base_required_conditions(ctx, base) {
            push_unique_implicit_condition(ctx, conditions, condition);
        }
    }

    for condition in unary_residual_required_conditions(ctx, expr) {
        push_unique_implicit_condition(ctx, conditions, condition);
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_integral_residual_structural_required_conditions(
                ctx,
                left,
                conditions,
                depth - 1,
            );
            collect_integral_residual_structural_required_conditions(
                ctx,
                right,
                conditions,
                depth - 1,
            );
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_integral_residual_structural_required_conditions(
                ctx,
                inner,
                conditions,
                depth - 1,
            );
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_integral_residual_structural_required_conditions(
                    ctx,
                    arg,
                    conditions,
                    depth - 1,
                );
            }
        }
        Expr::Matrix { data, .. } => {
            for child in data {
                collect_integral_residual_structural_required_conditions(
                    ctx,
                    child,
                    conditions,
                    depth - 1,
                );
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

fn unary_residual_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return Vec::new();
    };
    if args.len() != 1 {
        return Vec::new();
    }

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Tan) | Some(BuiltinFn::Sec) => {
            let denominator = ctx.call_builtin(BuiltinFn::Cos, vec![args[0]]);
            vec![crate::ImplicitCondition::NonZero(denominator)]
        }
        Some(BuiltinFn::Cot) | Some(BuiltinFn::Csc) => {
            let denominator = ctx.call_builtin(BuiltinFn::Sin, vec![args[0]]);
            vec![crate::ImplicitCondition::NonZero(denominator)]
        }
        Some(BuiltinFn::Ln)
        | Some(BuiltinFn::Log)
        | Some(BuiltinFn::Log2)
        | Some(BuiltinFn::Log10) => {
            vec![crate::ImplicitCondition::Positive(args[0])]
        }
        _ => Vec::new(),
    }
}

fn push_unique_implicit_condition(
    ctx: &Context,
    conditions: &mut Vec<crate::ImplicitCondition>,
    condition: crate::ImplicitCondition,
) {
    let already_present = conditions
        .iter()
        .any(|existing| match (existing, &condition) {
            (
                crate::ImplicitCondition::NonZero(existing_expr),
                crate::ImplicitCondition::NonZero(condition_expr),
            )
            | (
                crate::ImplicitCondition::Positive(existing_expr),
                crate::ImplicitCondition::Positive(condition_expr),
            ) => expr_eq(ctx, *existing_expr, *condition_expr),
            _ => false,
        });
    if !already_present {
        conditions.push(condition);
    }
}

fn collect_plain_ln_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    conditions: &mut Vec<crate::ImplicitCondition>,
) {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
        {
            if let Expr::Function(abs_fn_id, abs_args) = ctx.get(args[0]) {
                if abs_args.len() == 1 && ctx.builtin_of(*abs_fn_id) == Some(BuiltinFn::Abs) {
                    return;
                }
            }
            push_unique_implicit_condition(
                ctx,
                conditions,
                crate::ImplicitCondition::Positive(args[0]),
            );
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_plain_ln_positive_conditions(ctx, left, conditions);
            collect_plain_ln_positive_conditions(ctx, right, conditions);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_plain_ln_positive_conditions(ctx, inner, conditions);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_plain_ln_positive_conditions(ctx, arg, conditions);
            }
        }
        Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::SessionRef(_)
        | Expr::Matrix { .. } => {}
    }
}

fn quadratic_affine_log_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = integral_required_conditions(ctx, target, var_name);
    collect_plain_ln_positive_conditions(ctx, target, &mut conditions);
    conditions
}

fn supported_rational_polynomial_integral_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<Vec<crate::ImplicitCondition>> {
    const MAX_SUPPORTED_RATIONAL_POLYNOMIAL_DEGREE: usize = 6;

    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let Expr::Div(num, den) = ctx.get(target) else {
        return None;
    };
    if Polynomial::from_expr(ctx, *num, var_name).is_err() {
        return None;
    }
    let Ok(denominator) = Polynomial::from_expr(ctx, *den, var_name) else {
        return None;
    };
    let degree = denominator.degree();
    if !(2..=MAX_SUPPORTED_RATIONAL_POLYNOMIAL_DEGREE).contains(&degree) {
        return None;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx,
        target,
        var_name,
    ) {
        cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, target, var_name)?;
        return Some(integral_required_conditions(ctx, target, var_name));
    }

    if degree == 5 {
        let required_nonzero = cas_math::symbolic_integration_support::integrate_symbolic_rational_linear_positive_quadratic_required_nonzero_if_target(
            ctx,
            target,
            var_name,
        )?;
        return Some(
            required_nonzero
                .into_iter()
                .map(crate::ImplicitCondition::NonZero)
                .collect(),
        );
    }
    if degree == 6
        && !cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
            ctx,
            target,
            var_name,
        )
    {
        return None;
    }

    cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, target, var_name)?;

    Some(integral_required_conditions(ctx, target, var_name))
}

fn integrated_rational_quadratic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }
    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }
    supported_rational_polynomial_integral_required_conditions(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )
}

fn explicit_positive_quadratic_cube_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = strip_additive_constants_for_antiderivative_match(
        ctx,
        diff_call.target,
        &diff_call.var_name,
    );
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(Vec::new())
}

fn explicit_positive_quadratic_square_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = strip_additive_constants_for_antiderivative_match(
        ctx,
        diff_call.target,
        &diff_call.var_name,
    );
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(Vec::new())
}

fn explicit_high_log_power_product_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if let Some(integrate_call) =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)
    {
        if diff_call.var_name != integrate_call.var_name {
            return None;
        }
        if !exprs_match(ctx, integrate_call.target, right) {
            return None;
        }
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_verifiable_log_power_product_substitution_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_verifiable_log_power_product_substitution_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = cas_ast::hold::strip_all_holds(ctx, diff_call.target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

fn explicit_quadratic_affine_log_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if let Some(integrate_call) =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)
    {
        if diff_call.var_name != integrate_call.var_name {
            return None;
        }
        if !exprs_match(ctx, integrate_call.target, right) {
            return None;
        }
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        return Some(quadratic_affine_log_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = cas_ast::hold::strip_all_holds(ctx, diff_call.target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(quadratic_affine_log_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

fn explicit_quadratic_positive_quadratic_log_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if let Some(integrate_call) =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)
    {
        if diff_call.var_name != integrate_call.var_name {
            return None;
        }
        if !exprs_match(ctx, integrate_call.target, right) {
            return None;
        }
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = cas_ast::hold::strip_all_holds(ctx, diff_call.target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

fn collect_log_abs_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
    out: &mut Vec<crate::ImplicitCondition>,
) {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
        {
            if let Expr::Function(abs_fn_id, abs_args) = ctx.get(args[0]).clone() {
                if abs_args.len() == 1 && ctx.builtin_of(abs_fn_id) == Some(BuiltinFn::Abs) {
                    out.push(crate::ImplicitCondition::NonZero(abs_args[0]));
                }
            }
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            collect_log_abs_nonzero_conditions(ctx, left, out);
            collect_log_abs_nonzero_conditions(ctx, right, out);
        }
        Expr::Pow(base, exp) => {
            collect_log_abs_nonzero_conditions(ctx, base, out);
            collect_log_abs_nonzero_conditions(ctx, exp, out);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => collect_log_abs_nonzero_conditions(ctx, inner, out),
        _ => {}
    }
}

fn explicit_log_abs_antiderivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, diff_call.target);
    let mut required_conditions = Vec::new();
    collect_log_abs_nonzero_conditions(ctx, target, &mut required_conditions);
    if required_conditions.is_empty() {
        return None;
    }

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        target,
        &diff_call.var_name,
    )?;
    let residual = ctx.add(Expr::Sub(derivative, right));
    polynomial_fraction_sum_zero_in_var(ctx, residual, &diff_call.var_name).or_else(|| {
        crate::fraction_residual_support::try_polynomial_denominator_fraction_residual_zero(
            ctx, residual,
        )
        .map(|_| ())
    })?;
    Some(required_conditions)
}

fn scale_poly(poly: &Polynomial, scale: BigRational) -> Polynomial {
    poly.mul(&Polynomial::new(vec![scale], poly.var.clone()))
}

fn multiplicative_polynomial_quotient(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut numerator = Polynomial::one(var_name.to_string());
    let mut denominator = Polynomial::one(var_name.to_string());
    let mut saw_denominator = false;

    for factor in factors {
        if let Expr::Div(num, den) = ctx.get(factor) {
            let numerator_factor = Polynomial::from_expr(ctx, *num, var_name).ok()?;
            let denominator_factor = Polynomial::from_expr(ctx, *den, var_name).ok()?;
            numerator = numerator.mul(&numerator_factor);
            denominator = denominator.mul(&denominator_factor);
            saw_denominator = true;
            continue;
        }

        if let Expr::Pow(base, exp) = ctx.get(factor) {
            if cas_ast::views::as_rational_const(ctx, *exp, 8) == Some(-BigRational::one()) {
                let base_poly = Polynomial::from_expr(ctx, *base, var_name).ok()?;
                denominator = denominator.mul(&base_poly);
                saw_denominator = true;
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        numerator = numerator.mul(&factor_poly);
    }

    saw_denominator.then_some((numerator, denominator))
}

fn collect_signed_polynomial_fraction_terms(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    scale: BigRational,
    terms: &mut Vec<(Polynomial, Polynomial)>,
) -> Option<()> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_polynomial_fraction_terms(ctx, *left, var_name, scale.clone(), terms)?;
            collect_signed_polynomial_fraction_terms(ctx, *right, var_name, scale, terms)
        }
        Expr::Sub(left, right) => {
            collect_signed_polynomial_fraction_terms(ctx, *left, var_name, scale.clone(), terms)?;
            collect_signed_polynomial_fraction_terms(ctx, *right, var_name, -scale, terms)
        }
        Expr::Neg(inner) => {
            collect_signed_polynomial_fraction_terms(ctx, *inner, var_name, -scale, terms)
        }
        Expr::Div(num, den) => {
            let numerator = match Polynomial::from_expr(ctx, *num, var_name) {
                Ok(poly) => poly,
                Err(_) => {
                    return None;
                }
            };
            let denominator = match Polynomial::from_expr(ctx, *den, var_name) {
                Ok(poly) => poly,
                Err(_) => {
                    return None;
                }
            };
            if denominator.is_zero() || denominator.degree() > 4 {
                return None;
            }
            terms.push((scale_poly(&numerator, scale), denominator));
            Some(())
        }
        _ => {
            if let Some((numerator, denominator)) =
                multiplicative_polynomial_quotient(ctx, expr, var_name)
            {
                terms.push((scale_poly(&numerator, scale), denominator));
                return Some(());
            }
            let numerator = match Polynomial::from_expr(ctx, expr, var_name) {
                Ok(poly) => poly,
                Err(_) => {
                    return None;
                }
            };
            terms.push((
                scale_poly(&numerator, scale),
                Polynomial::one(var_name.to_string()),
            ));
            Some(())
        }
    }
}

fn polynomial_fraction_sum_zero_in_var(ctx: &Context, expr: ExprId, var_name: &str) -> Option<()> {
    let mut terms = Vec::new();
    collect_signed_polynomial_fraction_terms(ctx, expr, var_name, BigRational::one(), &mut terms)?;
    if terms.len() < 2 || terms.len() > 8 {
        return None;
    }

    let mut numerator_sum = Polynomial::zero(var_name.to_string());
    for (idx, (numerator, _denominator)) in terms.iter().enumerate() {
        let mut scaled = numerator.clone();
        for (other_idx, (_other_numerator, other_denominator)) in terms.iter().enumerate() {
            if idx != other_idx {
                scaled = scaled.mul(other_denominator);
            }
        }
        numerator_sum = numerator_sum.add(&scaled);
    }
    (numerator_sum.degree() <= 12 && numerator_sum.is_zero()).then_some(())
}

pub(crate) fn try_explicit_log_abs_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    let matched = try_diff_integral_residual_wrapped_root_zero(ctx, expr, 2, |ctx, expr| {
        let (left, right) = match ctx.get(expr) {
            Expr::Sub(left, right) => (*left, *right),
            _ => return None,
        };
        let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
        let required_conditions =
            explicit_log_abs_antiderivative_diff_matches(ctx, diff_expr, divisor, right)?;
        Some((ctx.num(0), required_conditions))
    });
    matched.or_else(|| {
        let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
        let required_conditions =
            explicit_log_abs_antiderivative_diff_matches(ctx, diff_expr, divisor, right)?;
        Some((ctx.num(0), required_conditions))
    })
}

fn log_abs_plain_trig_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (target_scale, inner_expr) = scaled_ln_abs_inner(ctx, target)?;
    let (builtin, trig_arg, primitive_sign) = plain_trig_log_abs_primitive(ctx, inner_expr)?;

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let expected_rational_scale = target_scale * primitive_sign / divisor_scale;
    let actual_scale = plain_trig_target_scale_expr(ctx, right, builtin, trig_arg)?;
    if let Some(matched) = polynomial_derivative_scale_matches(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    ) {
        return Some(matched);
    }

    let arg_derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx, trig_arg, var_name,
    )?;
    let expected_scale = scale_expr_by_rational(ctx, arg_derivative, expected_rational_scale);
    Some(exprs_equivalent(ctx, actual_scale, expected_scale))
}

fn integrated_log_abs_plain_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let matched = log_abs_plain_trig_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

fn integrated_quadratic_exp_linear_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_exp_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !term_shape_matches(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_exp_trig_same_linear_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if let Some(integrate_call) =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)
    {
        if diff_call.var_name != integrate_call.var_name {
            return None;
        }

        if !cas_math::symbolic_integration_support::integrate_symbolic_is_exp_trig_same_linear_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }

        if !term_shape_matches(ctx, integrate_call.target, right) {
            return None;
        }

        return Some(Vec::new());
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_exp_trig_same_linear_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }
    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    let target = strip_additive_constants_for_antiderivative_match(
        ctx,
        diff_call.target,
        &diff_call.var_name,
    );
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(ctx, target, expected_antiderivative, &diff_call.var_name) {
        return None;
    }

    Some(Vec::new())
}

fn negative_exp_trig_antiderivative_residual_ordered(
    ctx: &mut Context,
    diff_side: ExprId,
    integrand: ExprId,
    result_scale: BigRational,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, diff_side)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_exp_trig_same_linear_target(
        ctx,
        integrand,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrand,
        &diff_call.var_name,
    )?;
    let target = strip_additive_constants_for_antiderivative_match(
        ctx,
        diff_call.target,
        &diff_call.var_name,
    );
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let positive_target = ctx.add(Expr::Neg(target));
    let positive_target = cas_ast::hold::strip_all_holds(ctx, positive_target);
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    if !additive_term_multiset_matches(
        ctx,
        positive_target,
        expected_antiderivative,
        &diff_call.var_name,
    ) {
        return None;
    }

    Some(scale_expr_by_rational(ctx, integrand, result_scale))
}

pub(crate) fn try_diff_integral_exp_trig_residual_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };

    let two = BigRational::from_integer(2.into());
    negative_exp_trig_antiderivative_residual_ordered(ctx, left, right, -two.clone())
        .or_else(|| negative_exp_trig_antiderivative_residual_ordered(ctx, right, left, two))
        .map(|result| (result, Vec::new()))
}

fn integrated_polynomial_log_reciprocal_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    let mut conditions = Vec::new();
    for condition in
        integral_required_conditions(ctx, integrate_call.target, &integrate_call.var_name)
    {
        if !conditions.contains(&condition) {
            conditions.push(condition);
        }
    }
    Some(conditions)
}

fn explicit_polynomial_log_reciprocal_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
        ctx,
        right,
        &diff_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        right,
        &diff_call.var_name,
    )?;
    if !exprs_match(ctx, diff_call.target, expected_antiderivative) {
        return None;
    }

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

fn integrated_polynomial_trig_linear_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !term_shape_matches(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_polynomial_arctan_affine_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_asinh_polynomial_substitution_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_arcsin_polynomial_substitution_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(integral_required_conditions(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ))
}

fn integrated_arctan_sqrt_var_reciprocal_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    let mut conditions = Vec::new();
    for condition in
        integral_required_conditions(ctx, integrate_call.target, &integrate_call.var_name)
    {
        if !conditions.contains(&condition) {
            conditions.push(condition);
        }
    }
    Some(conditions)
}

fn integrated_polynomial_hyperbolic_linear_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_affine_hyperbolic_square_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_hyperbolic_square_product_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_square_product_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

fn integrated_hyperbolic_quotient_substitution_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_quotient_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !term_shape_matches(ctx, integrate_call.target, right) {
        return None;
    }

    Some(integral_required_conditions(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ))
}

fn affine_hyperbolic_cubic_primitive_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(BuiltinFn, ExprId, BigRational, BigRational)> {
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let (target_scale, target_core) = rational_scaled_term(ctx, target);
    if target_scale.is_zero() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(target_core).clone() else {
        return None;
    };
    let Expr::Number(power) = ctx.get(exp) else {
        return None;
    };
    if *power != BigRational::from_integer(3.into()) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
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

    Some((builtin, args[0], slope, target_scale))
}

fn derivative_companion_for_hyperbolic_cubic_primitive(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Sinh => Some(BuiltinFn::Cosh),
        BuiltinFn::Cosh => Some(BuiltinFn::Sinh),
        _ => None,
    }
}

fn hyperbolic_square_factor_matches(
    ctx: &mut Context,
    factor: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> bool {
    let factor = cas_ast::hold::strip_all_holds(ctx, factor);
    let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
        return false;
    };
    let Expr::Number(power) = ctx.get(exp) else {
        return false;
    };
    if *power != BigRational::from_integer(2.into()) {
        return false;
    }

    unary_builtin_arg(ctx, base, builtin).is_some_and(|base_arg| exprs_match(ctx, base_arg, arg))
}

fn hyperbolic_cubic_derivative_product_matches(
    ctx: &mut Context,
    expr: ExprId,
    source_builtin: BuiltinFn,
    arg: ExprId,
    expected_scale: BigRational,
) -> bool {
    let (scale, core) = rational_scaled_term(ctx, expr);
    if scale != expected_scale {
        return false;
    }

    let Some(companion_builtin) =
        derivative_companion_for_hyperbolic_cubic_primitive(source_builtin)
    else {
        return false;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, core);
    if factors.len() != 2 {
        return false;
    }

    let mut matched_companion = false;
    let mut matched_square = false;
    for factor in factors {
        if !matched_companion {
            if let Some(factor_arg) = unary_builtin_arg(ctx, factor, companion_builtin) {
                if exprs_match(ctx, factor_arg, arg) {
                    matched_companion = true;
                    continue;
                }
            }
        }

        if !matched_square && hyperbolic_square_factor_matches(ctx, factor, source_builtin, arg) {
            matched_square = true;
            continue;
        }

        return false;
    }

    matched_companion && matched_square
}

fn explicit_hyperbolic_cubic_primitive_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<()> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, slope, target_scale) =
        affine_hyperbolic_cubic_primitive_target(ctx, call.target, &call.var_name)?;
    let expected_scale = target_scale * BigRational::from_integer(3.into()) * slope;
    hyperbolic_cubic_derivative_product_matches(ctx, right, builtin, arg, expected_scale)
        .then_some(())
}

fn hyperbolic_square_power_reduction_target(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BuiltinFn, Polynomial)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let Expr::Number(power) = ctx.get(exp) else {
        return None;
    };
    if *power != BigRational::from_integer(2.into()) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (!slope.is_zero()).then_some((builtin, arg_poly))
}

fn squared_hyperbolic_builtin_arg(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let Expr::Number(power) = ctx.get(exp) else {
        return None;
    };
    if *power != BigRational::from_integer(2.into()) {
        return None;
    }

    unary_builtin_arg(ctx, base, builtin)
}

fn hyperbolic_square_product_power_reduction_target(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, Polynomial)> {
    let (scale, core) = rational_scaled_term(ctx, expr);
    let factors = cas_math::expr_nary::mul_leaves(ctx, core);
    if factors.len() != 2 {
        return None;
    }

    let mut sinh_arg = None;
    let mut cosh_arg = None;
    for factor in factors {
        if sinh_arg.is_none() {
            if let Some(arg) = squared_hyperbolic_builtin_arg(ctx, factor, BuiltinFn::Sinh) {
                sinh_arg = Some(arg);
                continue;
            }
        }

        if cosh_arg.is_none() {
            if let Some(arg) = squared_hyperbolic_builtin_arg(ctx, factor, BuiltinFn::Cosh) {
                cosh_arg = Some(arg);
                continue;
            }
        }

        return None;
    }

    let sinh_arg = sinh_arg?;
    let cosh_arg = cosh_arg?;
    if !exprs_match(ctx, sinh_arg, cosh_arg) {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, sinh_arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (!slope.is_zero()).then_some((scale, arg_poly))
}

fn hyperbolic_square_power_reduction_primitive_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<()> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let (right_builtin, right_arg_poly) =
        hyperbolic_square_power_reduction_target(ctx, right, &call.var_name)?;

    let mut sinh_term = None;
    let mut linear_coeff = BigRational::zero();
    for (term, sign) in cas_math::expr_nary::add_terms_signed(ctx, target) {
        let (scale, core) = signed_rational_scaled_term(ctx, term, sign);
        if let Some(arg) = unary_builtin_arg(ctx, core, BuiltinFn::Sinh) {
            if sinh_term.replace((scale, arg)).is_some() {
                return None;
            }
            continue;
        }

        if matches!(ctx.get(core), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == call.var_name)
        {
            linear_coeff += scale;
            continue;
        }

        return None;
    }

    let (sinh_coeff, sinh_arg) = sinh_term?;
    let sinh_arg_poly = Polynomial::from_expr(ctx, sinh_arg, &call.var_name).ok()?;
    if sinh_arg_poly != scale_polynomial(&right_arg_poly, &BigRational::from_integer(2.into())) {
        return None;
    }

    let sinh_slope = sinh_arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if sinh_coeff * sinh_slope != BigRational::new(1.into(), 2.into()) {
        return None;
    }

    let expected_linear = match right_builtin {
        BuiltinFn::Sinh => BigRational::new((-1).into(), 2.into()),
        BuiltinFn::Cosh => BigRational::new(1.into(), 2.into()),
        _ => return None,
    };
    (linear_coeff == expected_linear).then_some(())
}

fn hyperbolic_square_product_power_reduction_primitive_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<()> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let (right_scale, right_arg_poly) =
        hyperbolic_square_product_power_reduction_target(ctx, right, &call.var_name)?;

    let mut sinh_term = None;
    let mut linear_coeff = BigRational::zero();
    for (term, sign) in cas_math::expr_nary::add_terms_signed(ctx, target) {
        let (scale, core) = signed_rational_scaled_term(ctx, term, sign);
        if let Some(arg) = unary_builtin_arg(ctx, core, BuiltinFn::Sinh) {
            if sinh_term.replace((scale, arg)).is_some() {
                return None;
            }
            continue;
        }

        if matches!(ctx.get(core), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == call.var_name)
        {
            linear_coeff += scale;
            continue;
        }

        return None;
    }

    let (sinh_coeff, sinh_arg) = sinh_term?;
    let sinh_arg_poly = Polynomial::from_expr(ctx, sinh_arg, &call.var_name).ok()?;
    if sinh_arg_poly != scale_polynomial(&right_arg_poly, &BigRational::from_integer(4.into())) {
        return None;
    }

    let sinh_slope = sinh_arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let expected_scale = right_scale / BigRational::from_integer(8.into());
    if sinh_coeff * sinh_slope != expected_scale {
        return None;
    }

    (linear_coeff == -expected_scale).then_some(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConstantPassthroughOrientation {
    Add,
    LeadingSub,
}

struct ConstantPassthrough {
    constant: BigRational,
    core: ExprId,
    orientation: ConstantPassthroughOrientation,
}

fn negated_expr_core(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(left, right) => {
            let negative_one = -BigRational::one();
            if cas_math::numeric_eval::as_rational_const(ctx, *left)
                .is_some_and(|value| value == negative_one)
            {
                Some(*right)
            } else if cas_math::numeric_eval::as_rational_const(ctx, *right)
                .is_some_and(|value| value == negative_one)
            {
                Some(*left)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn strip_constant_passthrough(ctx: &Context, expr: ExprId) -> Option<ConstantPassthrough> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(constant) = cas_math::numeric_eval::as_rational_const(ctx, *left) {
                if let Some(inner) = negated_expr_core(ctx, *right) {
                    return Some(ConstantPassthrough {
                        constant,
                        core: inner,
                        orientation: ConstantPassthroughOrientation::LeadingSub,
                    });
                }
                Some(ConstantPassthrough {
                    constant,
                    core: *right,
                    orientation: ConstantPassthroughOrientation::Add,
                })
            } else if let Some(constant) = cas_math::numeric_eval::as_rational_const(ctx, *right) {
                if let Some(inner) = negated_expr_core(ctx, *left) {
                    return Some(ConstantPassthrough {
                        constant,
                        core: inner,
                        orientation: ConstantPassthroughOrientation::LeadingSub,
                    });
                }
                Some(ConstantPassthrough {
                    constant,
                    core: *left,
                    orientation: ConstantPassthroughOrientation::Add,
                })
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if let Some(constant) = cas_math::numeric_eval::as_rational_const(ctx, *left) {
                Some(ConstantPassthrough {
                    constant,
                    core: *right,
                    orientation: ConstantPassthroughOrientation::LeadingSub,
                })
            } else {
                let constant = -cas_math::numeric_eval::as_rational_const(ctx, *right)?;
                Some(ConstantPassthrough {
                    constant,
                    core: *left,
                    orientation: ConstantPassthroughOrientation::Add,
                })
            }
        }
        _ => None,
    }
}

fn exact_zero_noise_expr_with_depth(ctx: &Context, expr: ExprId, depth: u8) -> bool {
    if is_zero_constant(ctx, expr) {
        return true;
    }
    if depth == 0 {
        return false;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            exact_zero_noise_expr_with_depth(ctx, *left, depth - 1)
                && exact_zero_noise_expr_with_depth(ctx, *right, depth - 1)
        }
        Expr::Sub(left, right) => {
            expr_exact_noise_eq(ctx, *left, *right)
                || (exact_zero_noise_expr_with_depth(ctx, *left, depth - 1)
                    && exact_zero_noise_expr_with_depth(ctx, *right, depth - 1))
        }
        _ => false,
    }
}

fn exact_zero_noise_expr(ctx: &Context, expr: ExprId) -> bool {
    exact_zero_noise_expr_with_depth(ctx, expr, 4)
}

fn strip_exact_zero_additive_noise(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) if exact_zero_noise_expr(ctx, *left) => Some(*right),
        Expr::Add(left, right) if exact_zero_noise_expr(ctx, *right) => Some(*left),
        Expr::Add(left, right) => match (ctx.get(*left), ctx.get(*right)) {
            (Expr::Sub(core, noise), _) if expr_exact_noise_eq(ctx, *noise, *right) => Some(*core),
            (_, Expr::Sub(core, noise)) if expr_exact_noise_eq(ctx, *noise, *left) => Some(*core),
            _ => None,
        },
        Expr::Sub(left, right) if exact_zero_noise_expr(ctx, *right) => Some(*left),
        Expr::Sub(left, right) => match ctx.get(*left) {
            Expr::Add(core, noise) if expr_exact_noise_eq(ctx, *noise, *right) => Some(*core),
            Expr::Add(noise, core) if expr_exact_noise_eq(ctx, *noise, *right) => Some(*core),
            _ => None,
        },
        _ => None,
    }
}

fn strip_exact_zero_additive_noise_bounded(
    ctx: &Context,
    mut expr: ExprId,
    max_depth: u8,
) -> ExprId {
    for _ in 0..max_depth {
        let Some(stripped) = strip_exact_zero_additive_noise(ctx, expr) else {
            break;
        };
        expr = stripped;
    }
    expr
}

fn rational_expr(ctx: &mut Context, value: &BigRational) -> ExprId {
    ctx.add(Expr::Number(value.clone()))
}

fn build_constant_passthrough_expr(
    ctx: &mut Context,
    constant: &BigRational,
    core: ExprId,
    orientation: ConstantPassthroughOrientation,
) -> ExprId {
    let constant = rational_expr(ctx, constant);
    match orientation {
        ConstantPassthroughOrientation::Add => ctx.add(Expr::Add(constant, core)),
        ConstantPassthroughOrientation::LeadingSub => ctx.add(Expr::Sub(constant, core)),
    }
}

fn diff_inverse_reciprocal_trig_core_difference_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, left))
        .map(|(_zero, required_conditions)| required_conditions)
}

fn diff_inverse_reciprocal_trig_core_sum_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let neg_right = ctx.add(Expr::Neg(right));
    diff_inverse_reciprocal_trig_core_difference_conditions(ctx, left, neg_right).or_else(|| {
        let neg_left = ctx.add(Expr::Neg(left));
        diff_inverse_reciprocal_trig_core_difference_conditions(ctx, right, neg_left)
    })
}

fn bounded_inverse_trig_polynomial_reciprocal_sqrt_residual_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], &call.var_name).ok()?;
    if arg_poly.degree() == 0 || arg_poly.degree() > 3 {
        return None;
    }
    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let expected_base_poly = Polynomial::one(call.var_name.clone()).sub(&arg_poly.mul(&arg_poly));
    if expected_base_poly.is_zero() {
        return None;
    }

    let (actual_numerator, actual_base_poly) =
        polynomial_over_reciprocal_sqrt_term(ctx, right, &call.var_name)?;
    let expected_numerator = scale_polynomial(&derivative, &sign);
    if actual_base_poly.var != expected_base_poly.var
        || actual_base_poly != expected_base_poly
        || actual_numerator != expected_numerator
    {
        return None;
    }

    Some(vec![crate::ImplicitCondition::Positive(
        expected_base_poly.to_expr(ctx),
    )])
}

fn bounded_inverse_trig_sqrt_polynomial_residual_conditions(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
        )
    {
        return None;
    }

    let radicand = sqrt_like_base(ctx, args[0])?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, &call.var_name).ok()?;
    if radicand_poly.degree() == 0 || radicand_poly.degree() > 3 {
        return None;
    }

    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !(exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right)
        || quotient_matches_rationalized_sqrt_denominator_variant(ctx, compact, right))
    {
        return None;
    }

    let gap = Polynomial::one(call.var_name.clone())
        .sub(&radicand_poly)
        .to_expr(ctx);
    Some(vec![
        crate::ImplicitCondition::Positive(radicand),
        crate::ImplicitCondition::Positive(gap),
    ])
}

fn compact_inverse_trig_derivative_terms_match(
    ctx: &mut Context,
    compact: ExprId,
    right: ExprId,
) -> bool {
    if exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right)
        || quotient_matches_with_normalized_signs(ctx, compact, right)
        || quotient_matches_rationalized_sqrt_denominator_variant(ctx, compact, right)
    {
        return true;
    }

    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let (Expr::Neg(compact_inner), Expr::Neg(right_inner)) =
        (ctx.get(compact).clone(), ctx.get(right).clone())
    else {
        return false;
    };

    exprs_match(ctx, compact_inner, right_inner)
        || quotient_matches_with_unordered_products(ctx, compact_inner, right_inner)
        || quotient_matches_with_normalized_signs(ctx, compact_inner, right_inner)
        || quotient_matches_rationalized_sqrt_denominator_variant(ctx, compact_inner, right_inner)
}

fn signed_quotient_parts(ctx: &mut Context, expr: ExprId) -> Option<(bool, ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (negative, num, den) = signed_quotient_parts(ctx, inner)?;
            Some((!negative, num, den))
        }
        Expr::Div(num, den) => {
            let num = cas_ast::hold::strip_all_holds(ctx, num);
            match ctx.get(num).clone() {
                Expr::Neg(inner_num) => Some((true, inner_num, den)),
                _ => Some((false, num, den)),
            }
        }
        _ => None,
    }
}

fn quotient_matches_with_normalized_signs(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let Some((left_negative, left_num, left_den)) = signed_quotient_parts(ctx, left) else {
        return false;
    };
    let Some((right_negative, right_num, right_den)) = signed_quotient_parts(ctx, right) else {
        return false;
    };
    left_negative == right_negative
        && (exprs_match(ctx, left_num, right_num)
            || unordered_product_factors_match(ctx, left_num, right_num))
        && (unordered_product_factors_match(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, right_den, left_den))
}

fn bounded_inverse_trig_sqrt_affine_quotient_residual_conditions(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
        )
    {
        return None;
    }

    let radicand = sqrt_like_base(ctx, args[0])?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };
    let numerator_poly = Polynomial::from_expr(ctx, num, &call.var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, &call.var_name).ok()?;
    if numerator_poly.degree() != 1 || denominator_poly.degree() != 1 {
        return None;
    }

    let numerator_slope = numerator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if numerator_slope.is_zero() || numerator_slope != denominator_slope {
        return None;
    }

    let numerator_constant = numerator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !(denominator_constant - numerator_constant).is_positive() {
        return None;
    }

    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !compact_inverse_trig_derivative_terms_match(ctx, compact, right) {
        return None;
    }

    Some(vec![crate::ImplicitCondition::Positive(num)])
}

fn arctan_sqrt_positive_polynomial_quotient_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<()> {
    let compact =
        crate::rules::calculus::arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call(
            ctx, diff_expr,
        )?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    (exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right))
    .then_some(())
}

fn arctan_sqrt_derivative_presentation_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !(exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right)
        || quotient_matches_with_polynomial_content_denominators(
            ctx,
            compact,
            right,
            &call.var_name,
        ))
    {
        return None;
    }

    Some(vec![crate::ImplicitCondition::Positive(radicand)])
}

fn unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let required_conditions =
        unit_interval_bounded_inverse_trig_derivative_presentation_conditions(ctx, diff_expr)?;
    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !(exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right)
        || quotient_matches_rationalized_sqrt_denominator_variant(ctx, compact, right))
    {
        return None;
    }

    Some(required_conditions)
}

fn unit_interval_bounded_inverse_trig_derivative_presentation_conditions(
    ctx: &mut Context,
    diff_expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let mut target = cas_ast::hold::strip_all_holds(ctx, call.target);
    if let Expr::Div(numerator, denominator) = ctx.get(target).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
        if denominator.is_zero() {
            return None;
        }
        target = numerator;
    }

    let mut inverse_trig_arg = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            return None;
        };
        if args.len() != 1
            || !matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
            )
            || inverse_trig_arg.replace(args[0]).is_some()
        {
            return None;
        }
    }

    let arg = inverse_trig_arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let two = BigRational::from_integer(2.into());
    let unit_interval_arg = (offset == -BigRational::one() && slope == two)
        || (offset == BigRational::one() && slope == -two);
    if !unit_interval_arg {
        return None;
    }

    let var = ctx.var(&call.var_name);
    let one = ctx.num(1);
    let one_minus_var = ctx.add(Expr::Sub(one, var));
    Some(vec![
        crate::ImplicitCondition::Positive(var),
        crate::ImplicitCondition::Positive(one_minus_var),
    ])
}

fn scaled_arcsin_target(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arcsin | BuiltinFn::Asin)
                ) =>
        {
            Some((args[0], BigRational::one()))
        }
        Expr::Neg(inner) => {
            let (arg, scale) = scaled_arcsin_target(ctx, *inner)?;
            Some((arg, -scale))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (arg, scale) = scaled_arcsin_target(ctx, *num)?;
            Some((arg, scale / den_scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_arg = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= value;
                    continue;
                }

                let (arg, factor_scale) = scaled_arcsin_target(ctx, factor)?;
                if matched_arg.replace(arg).is_some() {
                    return None;
                }
                scale *= factor_scale;
            }

            Some((matched_arg?, scale))
        }
        _ => None,
    }
}

fn scaled_asinh_target(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    scaled_unary_builtin_rational_target(ctx, expr, BuiltinFn::Asinh)
        .map(|(scale, arg)| (arg, scale))
}

fn asinh_scaled_surd_polynomial_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let (arg, target_scale) = scaled_asinh_target(ctx, target)?;
    if target_scale.is_zero() {
        return None;
    }

    let Expr::Div(arg_num, arg_den) = ctx.get(arg).clone() else {
        return None;
    };
    let arg_den_base = sqrt_like_base(ctx, arg_den)?;
    let offset_square = cas_math::numeric_eval::as_rational_const(ctx, arg_den_base)?;
    if offset_square <= BigRational::zero() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, arg_num, &call.var_name).ok()?;
    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let (right_num, right_den) = signed_quotient_numerator_denominator(ctx, right)?;
    let right_num_poly = Polynomial::from_expr(ctx, right_num, &call.var_name).ok()?;
    let derivative_to_right_scale = polynomial_scale_factor(&right_num_poly, &derivative)?;
    let denominator_scale = derivative_to_right_scale * target_scale;
    if denominator_scale <= BigRational::zero() {
        return None;
    }

    let right_base = sqrt_like_base(ctx, right_den)?;
    let right_base_poly = Polynomial::from_expr(ctx, right_base, &call.var_name).ok()?;
    let expected_base =
        Polynomial::new(vec![offset_square], call.var_name.clone()).add(&arg_poly.mul(&arg_poly));
    let scaled_right_base = scale_polynomial(
        &right_base_poly,
        &(denominator_scale.clone() * denominator_scale),
    );
    if expected_base != scaled_right_base {
        return None;
    }

    Some(Vec::new())
}

fn arcsin_scaled_surd_polynomial_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let (arg, target_scale) = scaled_arcsin_target(ctx, target)?;
    if target_scale.is_zero() {
        return None;
    }

    let Expr::Div(arg_num, arg_den) = ctx.get(arg).clone() else {
        return None;
    };
    let arg_den_base = sqrt_like_base(ctx, arg_den)?;
    let offset_square = cas_math::numeric_eval::as_rational_const(ctx, arg_den_base)?;
    if offset_square <= BigRational::zero() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, arg_num, &call.var_name).ok()?;
    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let (right_num, right_den) = signed_quotient_numerator_denominator(ctx, right)?;
    let right_num_poly = Polynomial::from_expr(ctx, right_num, &call.var_name).ok()?;
    let derivative_to_right_scale = polynomial_scale_factor(&right_num_poly, &derivative)?;
    let denominator_scale = derivative_to_right_scale * target_scale;
    if denominator_scale <= BigRational::zero() {
        return None;
    }

    let right_base = sqrt_like_base(ctx, right_den)?;
    let right_base_poly = Polynomial::from_expr(ctx, right_base, &call.var_name).ok()?;
    let expected_base =
        Polynomial::new(vec![offset_square], call.var_name.clone()).sub(&arg_poly.mul(&arg_poly));
    let scaled_right_base = scale_polynomial(
        &right_base_poly,
        &(denominator_scale.clone() * denominator_scale),
    );
    if expected_base != scaled_right_base {
        return None;
    }

    Some(vec![crate::ImplicitCondition::Positive(right_base)])
}

fn arctan_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan | BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }
    sqrt_like_base(ctx, args[0])
}

fn quotient_matches_with_unordered_products(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let (left_num, left_den, right_num, right_den) = match (ctx.get(left), ctx.get(right)) {
        (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) => {
            (*left_num, *left_den, *right_num, *right_den)
        }
        _ => return false,
    };

    (exprs_match(ctx, left_num, right_num)
        || unordered_product_factors_match(ctx, left_num, right_num))
        && (unordered_product_factors_match(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, right_den, left_den))
}

fn quotient_rationalized_sqrt_denominator_variant(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let mut sqrt_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if let Some(radicand) = sqrt_like_base(ctx, factor) {
            sqrt_factors.push(factor);
            denominator_factors.push(radicand);
        } else {
            denominator_factors.push(factor);
        }
    }

    if sqrt_factors.is_empty() {
        return None;
    }

    let sqrt_product = cas_math::expr_nary::build_balanced_mul(ctx, &sqrt_factors);
    let numerator = if expr_is_one(ctx, num) {
        sqrt_product
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[num, sqrt_product])
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn quotient_matches_rationalized_sqrt_denominator_variant(
    ctx: &mut Context,
    compact: ExprId,
    target: ExprId,
) -> bool {
    quotient_rationalized_sqrt_denominator_variant(ctx, compact).is_some_and(|variant| {
        exprs_match(ctx, variant, target)
            || quotient_matches_with_unordered_products(ctx, variant, target)
            || rendered_exprs_match(ctx, variant, target)
    })
}

fn unordered_product_factors_match(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left);
    let mut right_factors = cas_math::expr_nary::mul_leaves(ctx, right);
    if left_factors.len() != right_factors.len() {
        return false;
    }

    for left_factor in left_factors {
        let Some(pos) = right_factors
            .iter()
            .position(|right_factor| exprs_match(ctx, left_factor, *right_factor))
        else {
            return false;
        };
        right_factors.remove(pos);
    }
    true
}

fn quotient_matches_with_polynomial_content_denominators(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let (left_num, left_den, right_num, right_den) = match (ctx.get(left), ctx.get(right)) {
        (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) => {
            (*left_num, *left_den, *right_num, *right_den)
        }
        _ => return false,
    };

    if exprs_match(ctx, left_num, right_num)
        && unordered_product_factors_match_after_polynomial_content(
            ctx, left_den, right_den, var_name,
        )
    {
        return true;
    }

    let Some((left_coeff, left_factors)) =
        rational_quotient_parts_after_polynomial_content(ctx, left_num, left_den, var_name)
    else {
        return false;
    };
    let Some((right_coeff, right_factors)) =
        rational_quotient_parts_after_polynomial_content(ctx, right_num, right_den, var_name)
    else {
        return false;
    };
    left_coeff == right_coeff && unordered_factor_lists_match(ctx, left_factors, right_factors)
}

fn rational_quotient_parts_after_polynomial_content(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    var_name: &str,
) -> Option<(BigRational, Vec<ExprId>)> {
    let numerator = rational_const_for_matching(ctx, numerator)?;
    let (denominator_scale, denominator_factors) =
        product_factors_after_polynomial_content(ctx, denominator, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    Some((numerator / denominator_scale, denominator_factors))
}

fn rational_const_for_matching(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, expr) {
        return Some(value);
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => rational_const_for_matching(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn unordered_product_factors_match_after_polynomial_content(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some((left_scale, left_factors)) =
        product_factors_after_polynomial_content(ctx, left, var_name)
    else {
        return false;
    };
    let Some((right_scale, right_factors)) =
        product_factors_after_polynomial_content(ctx, right, var_name)
    else {
        return false;
    };
    if left_scale != right_scale || left_factors.len() != right_factors.len() {
        return false;
    }

    unordered_factor_lists_match(ctx, left_factors, right_factors)
}

fn unordered_factor_lists_match(
    ctx: &mut Context,
    left_factors: Vec<ExprId>,
    right_factors: Vec<ExprId>,
) -> bool {
    if left_factors.len() != right_factors.len() {
        return false;
    }
    let mut unmatched_right = right_factors;
    for left_factor in left_factors {
        let Some(pos) = unmatched_right
            .iter()
            .position(|right_factor| exprs_match(ctx, left_factor, *right_factor))
        else {
            return false;
        };
        unmatched_right.remove(pos);
    }
    true
}

fn product_factors_after_polynomial_content(
    ctx: &mut Context,
    expr: ExprId,
    _var_name: &str,
) -> Option<(BigRational, Vec<ExprId>)> {
    let mut scale = BigRational::one();
    let mut factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = rational_const_for_matching(ctx, factor) {
            scale *= value;
            continue;
        }

        let budget = PolyBudget {
            max_terms: 8,
            max_total_degree: 4,
            max_pow_exp: 4,
        };
        if let Ok(poly) = multipoly_from_expr(ctx, factor, &budget) {
            let (content, primitive) = poly.primitive_part();
            if content.is_zero() {
                return None;
            }
            if content.is_one() {
                factors.push(factor);
            } else {
                scale *= content;
                factors.push(multipoly_to_expr(&primitive, ctx));
            }
            continue;
        }

        factors.push(factor);
    }

    Some((scale, factors))
}

fn compact_sqrt_one_plus_denominator_matches_expanded_sum(
    ctx: &mut Context,
    compact: ExprId,
    expanded: ExprId,
) -> bool {
    let Some((compact_scale, compact_base)) = compact_sqrt_one_plus_denominator(ctx, compact)
    else {
        return false;
    };
    let Some((expanded_scale, expanded_base)) = expanded_sqrt_one_plus_denominator(ctx, expanded)
    else {
        return false;
    };

    compact_scale == expanded_scale && exprs_match(ctx, compact_base, expanded_base)
}

fn compact_sqrt_one_plus_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let mut scale = BigRational::one();
    let mut sqrt_base = None;
    let mut add_one_base = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }
        if let Some(base) = sqrt_like_base(ctx, factor) {
            if sqrt_base.replace(base).is_some() {
                return None;
            }
            continue;
        }
        if let Some(base) = plus_one_base(ctx, factor) {
            if add_one_base.replace(base).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    let sqrt_base = sqrt_base?;
    let add_one_base = add_one_base?;
    exprs_match(ctx, sqrt_base, add_one_base).then_some((scale, sqrt_base))
}

fn plus_one_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_one_constant(ctx, *left) {
        Some(*right)
    } else if is_one_constant(ctx, *right) {
        Some(*left)
    } else {
        None
    }
}

fn expanded_sqrt_one_plus_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };
    let left_term = scaled_sqrt_power_term(ctx, left)?;
    let right_term = scaled_sqrt_power_term(ctx, right)?;

    let (half_term, three_half_term) = if left_term.2 == BigRational::new(1.into(), 2.into())
        && right_term.2 == BigRational::new(3.into(), 2.into())
    {
        (left_term, right_term)
    } else if left_term.2 == BigRational::new(3.into(), 2.into())
        && right_term.2 == BigRational::new(1.into(), 2.into())
    {
        (right_term, left_term)
    } else {
        return None;
    };

    (half_term.0 == three_half_term.0 && exprs_match(ctx, half_term.1, three_half_term.1))
        .then_some((half_term.0, half_term.1))
}

fn scaled_sqrt_power_term(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId, BigRational)> {
    let mut scale = BigRational::one();
    let mut power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }
        let (base, exponent) = sqrt_power_factor(ctx, factor)?;
        if power.replace((base, exponent)).is_some() {
            return None;
        }
    }

    let (base, exponent) = power?;
    Some((scale, base, exponent))
}

fn sqrt_power_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    if let Some(base) = sqrt_like_base(ctx, expr) {
        return Some((base, BigRational::new(1.into(), 2.into())));
    }
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exponent) = ctx.get(*exp) else {
        return None;
    };
    if *exponent == BigRational::new(1.into(), 2.into())
        || *exponent == BigRational::new(3.into(), 2.into())
    {
        Some((*base, exponent.clone()))
    } else {
        None
    }
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if arctan_sqrt_positive_polynomial_quotient_diff_matches(ctx, left, right).is_some()
        || arctan_sqrt_positive_polynomial_quotient_diff_matches(ctx, right, left).is_some()
        || quotient_matches_with_unordered_products(ctx, left, right)
    {
        return Some(Vec::new());
    }
    arctan_sqrt_derivative_presentation_diff_matches(ctx, left, right)
        .or_else(|| arctan_sqrt_derivative_presentation_diff_matches(ctx, right, left))
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
                ctx, left, right,
            )
        })
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
                ctx, right, left,
            )
        })
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_core_sum_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let neg_right = negated_quotient_for_matching(ctx, right);
    arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(ctx, left, neg_right).or_else(
        || {
            let neg_left = negated_quotient_for_matching(ctx, left);
            arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(ctx, neg_left, right)
        },
    )
}

fn negated_quotient_for_matching(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let neg_num = ctx.add(Expr::Neg(num));
            ctx.add(Expr::Div(neg_num, den))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
    ctx: &mut Context,
    numerator_passthrough: &ConstantPassthrough,
    denominator_passthrough: &ConstantPassthrough,
) -> Option<Vec<crate::ImplicitCondition>> {
    if numerator_passthrough.orientation == denominator_passthrough.orientation {
        arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(
            ctx,
            numerator_passthrough.core,
            denominator_passthrough.core,
        )
    } else {
        arctan_sqrt_positive_polynomial_quotient_shifted_core_sum_conditions(
            ctx,
            numerator_passthrough.core,
            denominator_passthrough.core,
        )
    }
}

pub(crate) fn try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_one_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant != denominator_passthrough.constant {
        return None;
    }
    let mut required_conditions =
        arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
            ctx,
            &numerator_passthrough,
            &denominator_passthrough,
        )?;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((ctx.num(1), required_conditions))
}

pub(crate) fn try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant == denominator_passthrough.constant {
        return None;
    }
    let mut required_conditions =
        arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
            ctx,
            &numerator_passthrough,
            &denominator_passthrough,
        )?;

    let compact_numerator = build_constant_passthrough_expr(
        ctx,
        &numerator_passthrough.constant,
        denominator_passthrough.core,
        denominator_passthrough.orientation,
    );
    let compact_quotient = ctx.add(Expr::Div(compact_numerator, denominator));

    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));
    Some((compact_quotient, required_conditions))
}

pub(crate) fn try_diff_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    log_abs_reciprocal_trig_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

pub(crate) fn try_diff_inverse_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        bounded_inverse_trig_polynomial_reciprocal_sqrt_residual_conditions(ctx, left, right)
            .or_else(|| {
                bounded_inverse_trig_sqrt_polynomial_residual_conditions(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                bounded_inverse_trig_sqrt_affine_quotient_residual_conditions(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| inverse_reciprocal_trig_affine_diff_matches(ctx, diff_expr, divisor, right))
            .or_else(|| {
                inverse_reciprocal_trig_positive_quadratic_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                inverse_reciprocal_trig_sqrt_polynomial_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                arcsin_scaled_surd_polynomial_derivative_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_reciprocal_trig_derivative_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| {
                integrated_log_abs_reciprocal_trig_diff_matches(ctx, diff_expr, divisor, right)
            })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_plain_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_exp_trig_same_linear_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| integrated_log_abs_plain_trig_diff_matches(ctx, diff_expr, divisor, right))
            .or_else(|| {
                integrated_affine_trig_fourth_power_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_affine_trig_sixth_power_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_affine_trig_eighth_power_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_tan_fourth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_cot_fourth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_tan_sixth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_cot_sixth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_tan_eighth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_cot_eighth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_sec_fourth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_csc_fourth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_sec_sixth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_csc_sixth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_sec_eighth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_or_explicit_csc_eighth_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_affine_trig_fifth_power_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_affine_trig_seventh_power_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_polynomial_trig_linear_diff_matches(ctx, diff_expr, divisor, right)
            })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_exp_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_plain_trig_residual_direct_root_zero,
    )
}

pub(crate) fn try_diff_integral_quadratic_exp_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_quadratic_exp_residual_direct_root_zero,
    )
}

fn try_diff_integral_quadratic_exp_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_quadratic_exp_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_quadratic_exp_residual_zero_preorder(ctx, right, left))
}

fn try_diff_integral_quadratic_exp_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_quadratic_exp_linear_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| {
                integrated_polynomial_log_reciprocal_derivative_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                explicit_polynomial_log_reciprocal_derivative_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_hyperbolic_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_hyperbolic_residual_direct_root_zero,
    )
}

fn try_diff_integral_hyperbolic_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_hyperbolic_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_integral_hyperbolic_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    // Hyperbolic by-parts residuals are already guarded by the direct matcher;
    // allow exactly two quotient shells for the retained double-nested smoke cases.
    try_diff_integral_residual_constant_passthrough_quotient_with_nested_depth(
        ctx,
        expr,
        2,
        2,
        try_diff_integral_hyperbolic_residual_direct_root_zero,
    )
}

fn try_diff_integral_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_polynomial_hyperbolic_linear_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| {
                integrated_affine_hyperbolic_square_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_hyperbolic_square_product_diff_matches(ctx, diff_expr, divisor, right)
            })
            .or_else(|| {
                integrated_hyperbolic_quotient_substitution_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                integrated_asinh_polynomial_substitution_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_rational_quadratic_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_rational_quadratic_residual_direct_root_zero,
    )
}

fn try_diff_integral_rational_quadratic_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_rational_quadratic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_rational_quadratic_residual_zero_preorder(ctx, right, left))
}

fn unit_reciprocal_denominator(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    if !expr_is_one(ctx, numerator) {
        return None;
    }
    Some(denominator)
}

fn extract_shared_additive_shift_cores(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    let left_terms = cas_math::expr_nary::AddView::from_expr(ctx, left).terms;
    let right_terms = cas_math::expr_nary::AddView::from_expr(ctx, right).terms;
    if left_terms.len() != 2 || right_terms.len() != 2 {
        return None;
    }

    for (left_shift_index, (left_shift, left_shift_sign)) in left_terms.iter().copied().enumerate()
    {
        for (right_shift_index, (right_shift, right_shift_sign)) in
            right_terms.iter().copied().enumerate()
        {
            if left_shift_sign != right_shift_sign || !expr_eq(ctx, left_shift, right_shift) {
                continue;
            }

            let left_core = left_terms[1 - left_shift_index];
            let right_core = right_terms[1 - right_shift_index];
            if left_core.1 != right_core.1 {
                return None;
            }

            let left_core = cas_math::expr_nary::AddView {
                root: left,
                terms: smallvec::smallvec![left_core],
            }
            .rebuild(ctx);
            let right_core = cas_math::expr_nary::AddView {
                root: right,
                terms: smallvec::smallvec![right_core],
            }
            .rebuild(ctx);
            return Some((left_core, right_core));
        }
    }

    None
}

fn extract_one_shared_additive_shift_remaining_cores(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    max_terms: usize,
) -> Option<(ExprId, ExprId)> {
    let left_terms = cas_math::expr_nary::AddView::from_expr(ctx, left).terms;
    let right_terms = cas_math::expr_nary::AddView::from_expr(ctx, right).terms;
    if !(2..=max_terms).contains(&left_terms.len()) || !(2..=max_terms).contains(&right_terms.len())
    {
        return None;
    }

    for (left_shift_index, (left_shift, left_shift_sign)) in left_terms.iter().copied().enumerate()
    {
        for (right_shift_index, (right_shift, right_shift_sign)) in
            right_terms.iter().copied().enumerate()
        {
            if left_shift_sign != right_shift_sign || !expr_eq(ctx, left_shift, right_shift) {
                continue;
            }

            let left_core_terms = left_terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(idx, term)| (idx != left_shift_index).then_some(term))
                .collect();
            let right_core_terms = right_terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(idx, term)| (idx != right_shift_index).then_some(term))
                .collect();
            let left_core = cas_math::expr_nary::AddView {
                root: left,
                terms: left_core_terms,
            }
            .rebuild(ctx);
            let right_core = cas_math::expr_nary::AddView {
                root: right,
                terms: right_core_terms,
            }
            .rebuild(ctx);
            return Some((left_core, right_core));
        }
    }

    None
}

fn rational_diff_integral_shared_shift_reciprocal_conditions(
    ctx: &mut Context,
    diff_denominator: ExprId,
    target_denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (diff_core, target_core) =
        extract_shared_additive_shift_cores(ctx, diff_denominator, target_denominator)?;
    let (_zero, mut required_conditions) =
        try_diff_integral_rational_quadratic_residual_zero_preorder(ctx, diff_core, target_core)?;
    required_conditions.push(crate::ImplicitCondition::NonZero(target_denominator));
    Some(required_conditions)
}

fn hyperbolic_integral_antiderivative_matches(
    ctx: &mut Context,
    integrate_expr: ExprId,
    target_antiderivative: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, integrate_expr)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    let expected_antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let expected_antiderivative = cas_ast::hold::strip_all_holds(ctx, expected_antiderivative);
    let target_antiderivative = cas_ast::hold::strip_all_holds(ctx, target_antiderivative);
    if !additive_term_multiset_matches(
        ctx,
        target_antiderivative,
        expected_antiderivative,
        &integrate_call.var_name,
    ) && !hyperbolic_polynomial_additive_matches(
        ctx,
        target_antiderivative,
        expected_antiderivative,
        &integrate_call.var_name,
    ) {
        return None;
    }

    Some(Vec::new())
}

fn hyperbolic_integral_shared_shift_reciprocal_conditions(
    ctx: &mut Context,
    integrate_denominator: ExprId,
    target_denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (integrate_core, target_core) = extract_one_shared_additive_shift_remaining_cores(
        ctx,
        integrate_denominator,
        target_denominator,
        5,
    )?;
    let mut required_conditions =
        hyperbolic_integral_antiderivative_matches(ctx, integrate_core, target_core)?;
    required_conditions.push(crate::ImplicitCondition::NonZero(target_denominator));
    Some(required_conditions)
}

pub(crate) fn try_integral_hyperbolic_reciprocal_shifted_difference_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(right_inner) => (*left, *right_inner),
            _ => return None,
        },
        _ => return None,
    };

    let left_denominator = unit_reciprocal_denominator(ctx, left)?;
    let right_denominator = unit_reciprocal_denominator(ctx, right)?;
    let required_conditions = hyperbolic_integral_shared_shift_reciprocal_conditions(
        ctx,
        left_denominator,
        right_denominator,
    )
    .or_else(|| {
        hyperbolic_integral_shared_shift_reciprocal_conditions(
            ctx,
            right_denominator,
            left_denominator,
        )
    })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_rational_quadratic_residual_reciprocal_shifted_difference_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(right_inner) => (*left, *right_inner),
            _ => return None,
        },
        _ => return None,
    };

    let left_denominator = unit_reciprocal_denominator(ctx, left)?;
    let right_denominator = unit_reciprocal_denominator(ctx, right)?;
    let required_conditions = rational_diff_integral_shared_shift_reciprocal_conditions(
        ctx,
        left_denominator,
        right_denominator,
    )
    .or_else(|| {
        rational_diff_integral_shared_shift_reciprocal_conditions(
            ctx,
            right_denominator,
            left_denominator,
        )
    })?;
    Some((ctx.num(0), required_conditions))
}

fn diff_integral_residual_shared_shift_reciprocal_conditions(
    ctx: &mut Context,
    residual_denominator: ExprId,
    target_denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (compacted_denominator, required_conditions) =
        compact_nonzero_residual_passthrough_denominator(
            ctx,
            residual_denominator,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )?;
    if !exprs_match(ctx, compacted_denominator, target_denominator) {
        return None;
    }
    Some(required_conditions)
}

fn try_diff_integral_residual_reciprocal_shifted_difference_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(right_inner) => (*left, *right_inner),
            _ => return None,
        },
        _ => return None,
    };

    let left_denominator = unit_reciprocal_denominator(ctx, left)?;
    let right_denominator = unit_reciprocal_denominator(ctx, right)?;
    let required_conditions = diff_integral_residual_shared_shift_reciprocal_conditions(
        ctx,
        left_denominator,
        right_denominator,
    )
    .or_else(|| {
        diff_integral_residual_shared_shift_reciprocal_conditions(
            ctx,
            right_denominator,
            left_denominator,
        )
    })?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_residual_reciprocal_shifted_difference_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_additive_passthrough_root_zero(
        ctx,
        expr,
        try_diff_integral_residual_reciprocal_shifted_difference_direct_root_zero,
    )
}

fn try_diff_integral_residual_additive_passthrough_root_zero(
    ctx: &mut Context,
    expr: ExprId,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let expr = strip_exact_zero_additive_noise_bounded(ctx, expr, 3);
    if let Some(result) = direct_root_zero(ctx, expr) {
        return Some(result);
    }
    if !contains_diff_call_bounded(ctx, expr, 8) {
        return None;
    }

    let view = cas_math::expr_nary::AddView::from_expr(ctx, expr);
    let terms: Vec<_> = view.terms.into_iter().collect();
    let term_count = terms.len();
    if !(3..=6).contains(&term_count) {
        return None;
    }

    let full_mask = (1_u32 << term_count) - 1;
    for mask in 1..full_mask {
        let passthrough_mask = full_mask ^ mask;
        if passthrough_mask == 0 {
            continue;
        }

        let residual_candidate = build_residual_subset_candidate(ctx, &terms, mask);
        let Some((_zero, required_conditions)) = direct_root_zero(ctx, residual_candidate) else {
            continue;
        };

        let passthrough = build_residual_subset_candidate(ctx, &terms, passthrough_mask);
        if exact_zero_noise_expr(ctx, passthrough) {
            return Some((ctx.num(0), required_conditions));
        }
    }

    None
}

fn diff_integral_residual_shifted_quotient_exact_one_conditions(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if let Some((compacted_numerator, required_conditions)) =
        compact_nonzero_residual_passthrough_denominator(
            ctx,
            numerator,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )
    {
        if exprs_match(ctx, compacted_numerator, denominator) {
            return Some(required_conditions);
        }
    }

    if let Some((compacted_denominator, required_conditions)) =
        compact_nonzero_residual_passthrough_denominator(
            ctx,
            denominator,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )
    {
        if exprs_match(ctx, compacted_denominator, numerator) {
            return Some(required_conditions);
        }
    }

    None
}

pub(crate) fn try_diff_integral_residual_passthrough_quotient_root_one(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    if cas_math::numeric_eval::as_rational_const(ctx, denominator)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }

    let required_conditions =
        diff_integral_residual_shifted_quotient_exact_one_conditions(ctx, numerator, denominator)?;
    Some((ctx.num(1), required_conditions))
}

fn square_of_unit_quotient_candidate(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    cas_math::numeric_eval::as_rational_const(ctx, *exponent)
        .is_some_and(|value| value == BigRational::from_integer(2.into()))
        .then_some(*base)
}

fn try_diff_integral_residual_shifted_quotient_difference_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let quotient = match ctx.get(expr).clone() {
        Expr::Sub(left, right) if expr_is_one(ctx, right) => left,
        Expr::Sub(left, right) if expr_is_one(ctx, left) => right,
        Expr::Add(left, right) if expr_is_one(ctx, left) => match ctx.get(right) {
            Expr::Neg(inner) => *inner,
            _ => return None,
        },
        _ => return None,
    };
    let quotient = unit_reciprocal_denominator(ctx, quotient).unwrap_or(quotient);
    let quotient = square_of_unit_quotient_candidate(ctx, quotient).unwrap_or(quotient);
    let Expr::Div(numerator, denominator) = ctx.get(quotient) else {
        return None;
    };
    if cas_math::numeric_eval::as_rational_const(ctx, *denominator)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }

    let required_conditions = diff_integral_residual_shifted_quotient_exact_one_conditions(
        ctx,
        *numerator,
        *denominator,
    )?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_residual_shifted_quotient_difference_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_additive_passthrough_root_zero(
        ctx,
        expr,
        try_diff_integral_residual_shifted_quotient_difference_direct_root_zero,
    )
}

fn try_diff_integral_rational_quadratic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_rational_quadratic_diff_matches(ctx, diff_expr, divisor, right)?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_rational_quadratic_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    // Keep this widening local to supported rational-polynomial integration
    // residuals; the direct matcher is what prevents a broad quotient search.
    try_diff_integral_residual_constant_passthrough_quotient_with_nested_depth(
        ctx,
        expr,
        2,
        2,
        try_diff_integral_rational_quadratic_residual_direct_root_zero,
    )
}

pub(crate) fn try_explicit_positive_quadratic_cube_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_explicit_positive_quadratic_cube_antiderivative_residual_direct_root_zero,
    )
}

fn try_explicit_positive_quadratic_cube_antiderivative_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_explicit_positive_quadratic_cube_antiderivative_residual_zero_preorder(ctx, left, right)
        .or_else(|| {
            try_explicit_positive_quadratic_cube_antiderivative_residual_zero_preorder(
                ctx, right, left,
            )
        })
}

fn try_explicit_positive_quadratic_cube_antiderivative_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions = explicit_positive_quadratic_cube_antiderivative_diff_matches(
        ctx, diff_expr, divisor, right,
    )?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_explicit_positive_quadratic_cube_antiderivative_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_explicit_positive_quadratic_cube_antiderivative_residual_direct_root_zero,
    )
}

pub(crate) fn try_explicit_positive_quadratic_square_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_explicit_positive_quadratic_square_antiderivative_residual_direct_root_zero,
    )
}

fn try_explicit_positive_quadratic_square_antiderivative_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_explicit_positive_quadratic_square_antiderivative_residual_zero_preorder(ctx, left, right)
        .or_else(|| {
            try_explicit_positive_quadratic_square_antiderivative_residual_zero_preorder(
                ctx, right, left,
            )
        })
}

fn try_explicit_positive_quadratic_square_antiderivative_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions = explicit_positive_quadratic_square_antiderivative_diff_matches(
        ctx, diff_expr, divisor, right,
    )?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_explicit_positive_quadratic_square_antiderivative_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_explicit_positive_quadratic_square_antiderivative_residual_direct_root_zero,
    )
}

pub(crate) fn try_explicit_high_log_power_product_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_explicit_high_log_power_product_antiderivative_residual_direct_root_zero,
    )
}

fn try_explicit_high_log_power_product_antiderivative_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_explicit_high_log_power_product_antiderivative_residual_zero_preorder(ctx, left, right)
        .or_else(|| {
            try_explicit_high_log_power_product_antiderivative_residual_zero_preorder(
                ctx, right, left,
            )
        })
}

fn try_explicit_high_log_power_product_antiderivative_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions = explicit_high_log_power_product_antiderivative_diff_matches(
        ctx, diff_expr, divisor, right,
    )?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_explicit_high_log_power_product_antiderivative_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_explicit_high_log_power_product_antiderivative_residual_direct_root_zero,
    )
}

pub(crate) fn try_explicit_quadratic_affine_log_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_explicit_quadratic_affine_log_antiderivative_residual_direct_root_zero,
    )
}

fn try_explicit_quadratic_affine_log_antiderivative_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_explicit_quadratic_affine_log_antiderivative_residual_zero_preorder(ctx, left, right)
        .or_else(|| {
            try_explicit_quadratic_affine_log_antiderivative_residual_zero_preorder(
                ctx, right, left,
            )
        })
}

fn try_explicit_quadratic_affine_log_antiderivative_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        explicit_quadratic_affine_log_antiderivative_diff_matches(ctx, diff_expr, divisor, right)?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_direct_root_zero,
    )
}

fn try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_zero_preorder(
        ctx, left, right,
    )
    .or_else(|| {
        try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_zero_preorder(
            ctx, right, left,
        )
    })
}

fn try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        explicit_quadratic_positive_quadratic_log_antiderivative_diff_matches(
            ctx, diff_expr, divisor, right,
        )?;
    Some((ctx.num(0), required_conditions))
}

fn constant_scaled_hyperbolic_reciprocal_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !is_constant_scaled_hyperbolic_reciprocal_target(ctx, call.target) {
        return None;
    }

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        call.target,
        &call.var_name,
    )?;
    let expected = if expr_is_one(ctx, divisor) {
        derivative
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(expr_eq(ctx, expected, right))
}

fn sqrt_or_exact_factor_matches(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if exprs_match(ctx, left, right) {
        return true;
    }

    let (Some(left_base), Some(right_base)) =
        (sqrt_like_base(ctx, left), sqrt_like_base(ctx, right))
    else {
        return false;
    };
    if exprs_match(ctx, left_base, right_base) {
        return true;
    }

    let Ok(left_poly) = Polynomial::from_expr(ctx, left_base, var_name) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right_base, var_name) else {
        return false;
    };
    left_poly == right_poly
}

fn quotient_sqrt_denominator_factors_match(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some((left_num, left_den)) = quotient_numerator_denominator(ctx, left) else {
        return false;
    };
    let Some((right_num, right_den)) = quotient_numerator_denominator(ctx, right) else {
        return false;
    };
    if !exprs_match(ctx, left_num, right_num) {
        return false;
    }

    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left_den);
    let mut right_factors = cas_math::expr_nary::mul_leaves(ctx, right_den);
    if left_factors.len() != right_factors.len() {
        return false;
    }

    for left_factor in left_factors {
        let Some(pos) = right_factors.iter().position(|right_factor| {
            sqrt_or_exact_factor_matches(ctx, left_factor, *right_factor, var_name)
        }) else {
            return false;
        };
        right_factors.remove(pos);
    }
    true
}

fn diff_sqrt_acosh_split_radical_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let acosh_expr = sqrt_like_base(ctx, target)?;
    let acosh_arg = unary_builtin_arg(ctx, acosh_expr, BuiltinFn::Acosh)?;
    let Ok(arg_poly) = Polynomial::from_expr(ctx, acosh_arg, &call.var_name) else {
        return None;
    };
    let derivative = arg_poly.derivative();
    if derivative.coeffs.len() != 1 {
        return None;
    }
    let coefficient = derivative.coeffs[0].clone() / BigRational::from_integer(2.into());
    if coefficient.is_zero() {
        return None;
    }

    let one_poly = Polynomial::one(call.var_name.clone());
    let arg_minus_one = arg_poly.sub(&one_poly).to_expr(ctx);
    let arg_plus_one = arg_poly.add(&one_poly).to_expr(ctx);
    let sqrt_arg_minus_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_minus_one]);
    let sqrt_arg_plus_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_plus_one]);
    let sqrt_acosh = ctx.call_builtin(BuiltinFn::Sqrt, vec![acosh_expr]);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_arg_minus_one, sqrt_arg_plus_one, sqrt_acosh],
    );
    let numerator = rational_expr(ctx, &coefficient);
    let expected = ctx.add(Expr::Div(numerator, denominator));

    if !(exprs_match(ctx, expected, right)
        || antiderivative_term_matches(ctx, expected, right, &call.var_name)
        || quotient_matches_with_unordered_products(ctx, expected, right)
        || quotient_sqrt_denominator_factors_match(ctx, expected, right, &call.var_name)
        || quotient_matches_with_polynomial_content_denominators(
            ctx,
            expected,
            right,
            &call.var_name,
        ))
    {
        return None;
    }

    Some(vec![
        crate::ImplicitCondition::Positive(acosh_expr),
        crate::ImplicitCondition::Positive(arg_minus_one),
    ])
}

fn diff_acosh_sqrt_abs_safe_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let acosh_arg = unary_builtin_arg(ctx, target, BuiltinFn::Acosh)?;
    let radicand = sqrt_like_base(ctx, acosh_arg)?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, &call.var_name).ok()?;
    if radicand_poly.degree() > 2 {
        return None;
    }

    let numerator_poly = scale_polynomial(
        &radicand_poly.derivative(),
        &BigRational::new(1.into(), 2.into()),
    );
    if numerator_poly.is_zero() {
        return None;
    }
    let numerator = numerator_poly.to_expr(ctx);

    let one_poly = Polynomial::one(call.var_name.clone());
    let radicand_minus_one = radicand_poly.sub(&one_poly).to_expr(ctx);
    let raw_sqrt_radicand_minus_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_minus_one]);
    let sqrt_radicand_minus_one =
        try_rewrite_simplify_square_root_expr(ctx, raw_sqrt_radicand_minus_one)
            .map(|rewrite| rewrite.rewritten)
            .unwrap_or(raw_sqrt_radicand_minus_one);
    let required_nonzero = unary_builtin_arg(ctx, sqrt_radicand_minus_one, BuiltinFn::Abs)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_radicand_minus_one]);
    let expected = ctx.add(Expr::Div(numerator, denominator));

    if !(exprs_match(ctx, expected, right)
        || antiderivative_term_matches(ctx, expected, right, &call.var_name)
        || quotient_matches_with_unordered_products(ctx, expected, right)
        || quotient_sqrt_denominator_factors_match(ctx, expected, right, &call.var_name)
        || quotient_matches_with_polynomial_content_denominators(
            ctx,
            expected,
            right,
            &call.var_name,
        ))
    {
        return None;
    }

    Some(vec![crate::ImplicitCondition::NonZero(required_nonzero)])
}

pub(crate) fn try_diff_hyperbolic_reciprocal_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    constant_scaled_hyperbolic_reciprocal_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

fn try_diff_sqrt_acosh_split_radical_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        diff_sqrt_acosh_split_radical_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| diff_acosh_sqrt_abs_safe_diff_matches(ctx, diff_expr, divisor, right))?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_sqrt_acosh_split_radical_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            try_diff_sqrt_acosh_split_radical_residual_zero_preorder(ctx, left, right).or_else(
                || try_diff_sqrt_acosh_split_radical_residual_zero_preorder(ctx, right, left),
            )
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            try_diff_sqrt_acosh_split_radical_residual_zero_preorder(ctx, left, neg_right).or_else(
                || try_diff_sqrt_acosh_split_radical_residual_zero_preorder(ctx, right, neg_left),
            )
        }
        _ => None,
    }
}

fn diff_acosh_affine_reciprocal_sqrt_residual_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let acosh_arg = unary_builtin_arg(ctx, target, BuiltinFn::Acosh)?;
    let arg_poly = Polynomial::from_expr(ctx, acosh_arg, &call.var_name).ok()?;
    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let expected_base_poly = arg_poly
        .mul(&arg_poly)
        .sub(&Polynomial::one(call.var_name.clone()));
    if expected_base_poly.is_zero() {
        return None;
    }

    if derivative.coeffs.len() == 1 {
        let expected_scale = derivative.coeffs[0].clone();
        if expected_scale.is_zero() {
            return None;
        }

        let (actual_scale, actual_base_poly, _) =
            scaled_reciprocal_sqrt_polynomial_term(ctx, right)?;
        if actual_base_poly.var != expected_base_poly.var
            || !same_nonzero_sign(&expected_scale, &actual_scale)
        {
            return None;
        }

        let expected_scale_sq = expected_scale.clone() * expected_scale;
        let actual_scale_sq = actual_scale.clone() * actual_scale;
        if scale_polynomial(&expected_base_poly, &actual_scale_sq)
            != scale_polynomial(&actual_base_poly, &expected_scale_sq)
        {
            return None;
        }

        let arg_minus_one = arg_poly
            .sub(&Polynomial::one(call.var_name.clone()))
            .to_expr(ctx);
        return Some(vec![crate::ImplicitCondition::Positive(arg_minus_one)]);
    }

    let arg_minus_one = arg_poly
        .sub(&Polynomial::one(call.var_name.clone()))
        .to_expr(ctx);
    if !strictly_positive_quadratic_base(ctx, arg_minus_one) {
        return None;
    }

    let (actual_numerator, actual_base_poly) =
        polynomial_over_reciprocal_sqrt_term(ctx, right, &call.var_name)?;
    if actual_base_poly.var != expected_base_poly.var
        || actual_base_poly != expected_base_poly
        || actual_numerator != derivative
    {
        return None;
    }

    Some(Vec::new())
}

pub(crate) fn try_diff_acosh_affine_reciprocal_sqrt_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let required_conditions = match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            diff_acosh_affine_reciprocal_sqrt_residual_conditions(ctx, left, right)
                .or_else(|| diff_acosh_affine_reciprocal_sqrt_residual_conditions(ctx, right, left))
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            diff_acosh_affine_reciprocal_sqrt_residual_conditions(ctx, left, neg_right).or_else(
                || diff_acosh_affine_reciprocal_sqrt_residual_conditions(ctx, right, neg_left),
            )
        }
        _ => None,
    }?;
    Some((ctx.num(0), required_conditions))
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

fn ln_sqrt_polynomial_gap_target_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(Polynomial, Polynomial, BigRational)> {
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let arg = unary_builtin_arg(ctx, target, BuiltinFn::Ln)?;
    let mut radicand = None;
    let mut polynomial_term = Polynomial::zero(var_name.to_string());

    for (term, sign) in cas_math::expr_nary::AddView::from_expr(ctx, arg).terms {
        if let Some(term_radicand) = sqrt_like_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg || radicand.replace(term_radicand).is_some() {
                return None;
            }
            continue;
        }

        let mut term_poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
        if sign == cas_math::expr_nary::Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_term = polynomial_term.add(&term_poly);
    }

    if polynomial_term.is_zero() {
        return None;
    }
    let radicand_poly = Polynomial::from_expr(ctx, radicand?, var_name).ok()?;
    let square_gap = polynomial_term.mul(&polynomial_term).sub(&radicand_poly);
    if square_gap.degree() != 0 {
        return None;
    }
    let gap_value = square_gap.coeffs.first()?.clone();
    if !gap_value.is_positive() {
        return None;
    }

    Some((polynomial_term, radicand_poly, gap_value))
}

fn diff_ln_sqrt_polynomial_gap_residual_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (polynomial_term, expected_base_poly, gap_value) =
        ln_sqrt_polynomial_gap_target_parts(ctx, call.target, &call.var_name)?;
    let derivative = polynomial_term.derivative();
    if derivative.is_zero() {
        return None;
    }

    let (actual_numerator, actual_base_poly) =
        polynomial_over_reciprocal_sqrt_term(ctx, right, &call.var_name)?;
    if actual_base_poly.var != expected_base_poly.var {
        return None;
    }
    if actual_base_poly != expected_base_poly || actual_numerator != derivative {
        return None;
    }

    let branch_boundary = if let Some(sqrt_value) = exact_positive_rational_sqrt(&gap_value) {
        ctx.add(Expr::Number(sqrt_value))
    } else {
        let gap_expr = ctx.add(Expr::Number(gap_value));
        ctx.call_builtin(BuiltinFn::Sqrt, vec![gap_expr])
    };
    let polynomial_expr = polynomial_term.to_expr(ctx);
    let branch_gap = ctx.add(Expr::Sub(polynomial_expr, branch_boundary));
    Some(vec![crate::ImplicitCondition::Positive(branch_gap)])
}

pub(crate) fn try_diff_ln_sqrt_polynomial_gap_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let required_conditions = match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            diff_ln_sqrt_polynomial_gap_residual_conditions(ctx, left, right)
                .or_else(|| diff_ln_sqrt_polynomial_gap_residual_conditions(ctx, right, left))
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            diff_ln_sqrt_polynomial_gap_residual_conditions(ctx, left, neg_right)
                .or_else(|| diff_ln_sqrt_polynomial_gap_residual_conditions(ctx, right, neg_left))
        }
        _ => None,
    }?;
    Some((ctx.num(0), required_conditions))
}

fn try_diff_asinh_scaled_surd_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    asinh_scaled_surd_polynomial_derivative_diff_matches(ctx, diff_expr, divisor, right)?;
    Some(ctx.num(0))
}

pub(crate) fn try_diff_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    try_diff_log_abs_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_hyperbolic_reciprocal_residual_zero_preorder(ctx, left, right))
        .or_else(|| {
            let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
            explicit_hyperbolic_cubic_primitive_diff_matches(ctx, diff_expr, divisor, right)
                .map(|_| ctx.num(0))
        })
        .or_else(|| {
            let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
            hyperbolic_square_power_reduction_primitive_diff_matches(ctx, diff_expr, divisor, right)
                .map(|_| ctx.num(0))
        })
        .or_else(|| {
            let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
            hyperbolic_square_product_power_reduction_primitive_diff_matches(
                ctx, diff_expr, divisor, right,
            )
            .map(|_| ctx.num(0))
        })
        .or_else(|| try_diff_asinh_scaled_surd_residual_zero_preorder(ctx, left, right))
        .or_else(|| try_hyperbolic_tanh_common_factor_residual_zero_preorder(ctx, left, right))
}

pub(crate) fn try_diff_hyperbolic_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            try_diff_hyperbolic_residual_zero_preorder(ctx, left, right)
                .or_else(|| try_diff_hyperbolic_residual_zero_preorder(ctx, right, left))
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            try_diff_hyperbolic_residual_zero_preorder(ctx, left, neg_right)
                .or_else(|| try_diff_hyperbolic_residual_zero_preorder(ctx, right, neg_left))
        }
        _ => None,
    }
}

pub(crate) fn try_diff_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_reciprocal_trig_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_inverse_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, right).or_else(
                || try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, left),
            )
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, neg_right).or_else(
                || try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, neg_left),
            )
        }
        _ => None,
    }
}

pub(crate) fn try_diff_inverse_reciprocal_trig_shifted_quotient_root_one(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };

    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant != denominator_passthrough.constant {
        return None;
    }

    let mut required_conditions =
        if numerator_passthrough.orientation == denominator_passthrough.orientation {
            diff_inverse_reciprocal_trig_core_difference_conditions(
                ctx,
                numerator_passthrough.core,
                denominator_passthrough.core,
            )
        } else {
            diff_inverse_reciprocal_trig_core_sum_conditions(
                ctx,
                numerator_passthrough.core,
                denominator_passthrough.core,
            )
        }?;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((ctx.num(1), required_conditions))
}

pub(crate) fn try_diff_inverse_reciprocal_trig_shifted_quotient_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };

    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant == denominator_passthrough.constant {
        return None;
    }

    let (target_orientation, required_conditions) =
        if numerator_passthrough.orientation == denominator_passthrough.orientation {
            (
                denominator_passthrough.orientation,
                diff_inverse_reciprocal_trig_core_difference_conditions(
                    ctx,
                    numerator_passthrough.core,
                    denominator_passthrough.core,
                )?,
            )
        } else {
            (
                denominator_passthrough.orientation,
                diff_inverse_reciprocal_trig_core_sum_conditions(
                    ctx,
                    numerator_passthrough.core,
                    denominator_passthrough.core,
                )?,
            )
        };

    let compact_numerator = build_constant_passthrough_expr(
        ctx,
        &numerator_passthrough.constant,
        denominator_passthrough.core,
        target_orientation,
    );
    let compact_quotient = ctx.add(Expr::Div(compact_numerator, denominator));
    let mut required_conditions = required_conditions;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((compact_quotient, required_conditions))
}

pub(crate) fn try_diff_integral_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_reciprocal_trig_residual_direct_root_zero,
    )
}

pub(crate) fn try_diff_integral_reciprocal_trig_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_diff_integral_reciprocal_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_reciprocal_trig_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(left, right) => (left, right),
        Expr::Add(left, right) => match (ctx.get(left).clone(), ctx.get(right).clone()) {
            (_, Expr::Neg(right_inner)) => (left, right_inner),
            (Expr::Neg(left_inner), _) => (right, left_inner),
            _ => return None,
        },
        _ => return None,
    };
    try_diff_integral_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_reciprocal_trig_residual_zero_preorder(ctx, right, left))
}

fn try_diff_integral_reciprocal_half_power_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(left, right) => (left, right),
        Expr::Add(left, right) => match (ctx.get(left).clone(), ctx.get(right).clone()) {
            (_, Expr::Neg(right_inner)) => (left, right_inner),
            (Expr::Neg(left_inner), _) => (right, left_inner),
            _ => return None,
        },
        _ => return None,
    };
    try_diff_integral_reciprocal_half_power_residual_zero_preorder(ctx, left, right).or_else(|| {
        try_diff_integral_reciprocal_half_power_residual_zero_preorder(ctx, right, left)
    })
}

fn try_diff_integral_reciprocal_half_power_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let is_supported_root_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        );
    if !is_supported_root_product || !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some((
        ctx.num(0),
        integral_required_conditions(ctx, integrate_call.target, &integrate_call.var_name),
    ))
}

type IntegralResidualRootResult = Option<(ExprId, Vec<crate::ImplicitCondition>)>;
type IntegralResidualRootMatcher = fn(&mut Context, ExprId) -> IntegralResidualRootResult;

fn try_any_diff_integral_residual_wrapped_root_zero(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    primary_root_zero: IntegralResidualRootMatcher,
) -> IntegralResidualRootResult {
    for matcher in [
        primary_root_zero,
        try_diff_integral_quadratic_exp_residual_direct_root_zero,
        try_diff_integral_hyperbolic_residual_direct_root_zero,
        try_diff_integral_rational_quadratic_residual_direct_root_zero,
        try_diff_integral_reciprocal_trig_residual_direct_root_zero,
        try_diff_integral_plain_trig_residual_direct_root_zero,
        try_diff_integral_inverse_trig_residual_direct_root_zero,
        try_diff_integral_reciprocal_half_power_residual_direct_root_zero,
    ] {
        if let Some(result) =
            try_diff_integral_residual_wrapped_root_zero(ctx, expr, depth, matcher)
        {
            return Some(result);
        }
    }

    None
}

fn try_diff_integral_residual_wrapped_root_zero(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> IntegralResidualRootResult {
    if let Some(result) = direct_root_zero(ctx, expr) {
        return Some(result);
    }

    if depth == 0 {
        return None;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if is_zero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else if is_zero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if is_zero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else if is_zero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            try_diff_integral_residual_wrapped_root_zero(ctx, inner, depth - 1, direct_root_zero)
        }
        Expr::Mul(left, right) => {
            if is_nonzero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else if is_nonzero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            if cas_math::numeric_eval::as_rational_const(ctx, den)
                .is_some_and(|value| value.is_zero())
            {
                return None;
            }
            let (zero, mut required_conditions) = try_diff_integral_residual_wrapped_root_zero(
                ctx,
                num,
                depth - 1,
                direct_root_zero,
            )?;
            if cas_math::numeric_eval::as_rational_const(ctx, den).is_none() {
                required_conditions.push(crate::ImplicitCondition::NonZero(den));
            }
            Some((zero, required_conditions))
        }
        _ => None,
    }
}

fn compact_nonzero_residual_passthrough_denominator(
    ctx: &mut Context,
    denominator: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let denominator = strip_exact_zero_additive_noise_bounded(ctx, denominator, 3);
    let (scale, denominator) = strip_single_nonzero_constant_factor(ctx, denominator)
        .unwrap_or((BigRational::one(), denominator));
    if let Some((denominator, required_conditions)) =
        compact_nonzero_residual_passthrough_denominator_factor(
            ctx,
            denominator,
            depth,
            direct_root_zero,
        )
    {
        let denominator = apply_compacted_denominator_scale(ctx, &scale, denominator)?;
        return Some((denominator, required_conditions));
    }

    if let Some((denominator, required_conditions)) =
        compact_nonzero_residual_additive_passthrough_denominator_factor(
            ctx,
            denominator,
            depth,
            direct_root_zero,
        )
    {
        let denominator = apply_compacted_denominator_scale(ctx, &scale, denominator)?;
        return Some((denominator, required_conditions));
    }

    if let Some((denominator, required_conditions)) =
        compact_nonzero_residual_passthrough_div_denominator(
            ctx,
            &scale,
            denominator,
            depth,
            direct_root_zero,
        )
    {
        return Some((denominator, required_conditions));
    }

    compact_nonzero_residual_passthrough_product_denominator(
        ctx,
        scale,
        denominator,
        depth,
        direct_root_zero,
    )
}

fn compact_nonzero_residual_passthrough_denominator_factor(
    ctx: &mut Context,
    denominator: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let passthrough = strip_constant_passthrough(ctx, denominator)?;
    if passthrough.constant.is_zero() {
        return None;
    }

    let (_zero, required_conditions) = try_any_diff_integral_residual_wrapped_root_zero(
        ctx,
        passthrough.core,
        depth,
        direct_root_zero,
    )?;
    let denominator = rational_expr(ctx, &passthrough.constant);
    Some((denominator, required_conditions))
}

fn signed_add_term(ctx: &mut Context, term: ExprId, sign: cas_math::expr_nary::Sign) -> ExprId {
    match sign {
        cas_math::expr_nary::Sign::Pos => term,
        cas_math::expr_nary::Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn build_signed_add_subset(
    ctx: &mut Context,
    terms: &[(ExprId, cas_math::expr_nary::Sign)],
    mask: u32,
    selected: bool,
) -> ExprId {
    let rebuilt_terms: Vec<_> = terms
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| ((mask & (1 << index)) != 0) == selected)
        .map(|(_, (term, sign))| signed_add_term(ctx, term, sign))
        .collect();
    cas_math::expr_nary::build_balanced_add(ctx, &rebuilt_terms)
}

fn build_residual_subset_candidate(
    ctx: &mut Context,
    terms: &[(ExprId, cas_math::expr_nary::Sign)],
    mask: u32,
) -> ExprId {
    let selected: Vec<_> = terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| ((mask & (1 << index)) != 0).then_some(term))
        .collect();
    match selected.as_slice() {
        [(left, cas_math::expr_nary::Sign::Pos), (right, cas_math::expr_nary::Sign::Neg)] => {
            ctx.add(Expr::Sub(*left, *right))
        }
        [(left, cas_math::expr_nary::Sign::Neg), (right, cas_math::expr_nary::Sign::Pos)] => {
            ctx.add(Expr::Sub(*right, *left))
        }
        _ => build_signed_add_subset(ctx, terms, mask, true),
    }
}

fn contains_diff_call_bounded(ctx: &Context, expr: ExprId, depth: u8) -> bool {
    if crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr).is_some() {
        return true;
    }
    if depth == 0 {
        return false;
    }

    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            contains_diff_call_bounded(ctx, *left, depth - 1)
                || contains_diff_call_bounded(ctx, *right, depth - 1)
        }
        Expr::Pow(base, exponent) => {
            contains_diff_call_bounded(ctx, *base, depth - 1)
                || contains_diff_call_bounded(ctx, *exponent, depth - 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_diff_call_bounded(ctx, *inner, depth - 1),
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| contains_diff_call_bounded(ctx, *arg, depth - 1)),
        _ => false,
    }
}

fn is_total_real_polynomial_passthrough(ctx: &Context, expr: ExprId, depth: u8) -> bool {
    if depth == 0 {
        return false;
    }

    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => true,
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            is_total_real_polynomial_passthrough(ctx, *left, depth - 1)
                && is_total_real_polynomial_passthrough(ctx, *right, depth - 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            is_total_real_polynomial_passthrough(ctx, *inner, depth - 1)
        }
        Expr::Pow(base, exponent) => {
            is_total_real_polynomial_passthrough(ctx, *base, depth - 1)
                && cas_math::numeric_eval::as_rational_const(ctx, *exponent)
                    .is_some_and(|value| value.denom().is_one() && value >= BigRational::zero())
        }
        Expr::Div(_, _) | Expr::Function(_, _) | Expr::Matrix { .. } | Expr::SessionRef(_) => false,
    }
}

fn residual_with_total_real_additive_passthrough_conditions(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<Vec<crate::ImplicitCondition>> {
    let expr = strip_exact_zero_additive_noise_bounded(ctx, expr, 3);
    if let Some((_zero, required_conditions)) =
        try_any_diff_integral_residual_wrapped_root_zero(ctx, expr, depth, direct_root_zero)
    {
        return Some(required_conditions);
    }

    if !contains_diff_call_bounded(ctx, expr, 8) {
        return None;
    }

    let view = cas_math::expr_nary::AddView::from_expr(ctx, expr);
    let terms: Vec<_> = view.terms.into_iter().collect();
    let term_count = terms.len();
    if !(3..=6).contains(&term_count) {
        return None;
    }

    let full_mask = (1_u32 << term_count) - 1;
    for mask in 1..full_mask {
        let passthrough_mask = full_mask ^ mask;
        if passthrough_mask == 0 {
            continue;
        }

        let residual_candidate = build_residual_subset_candidate(ctx, &terms, mask);
        let Some((_zero, required_conditions)) = try_any_diff_integral_residual_wrapped_root_zero(
            ctx,
            residual_candidate,
            depth,
            direct_root_zero,
        ) else {
            continue;
        };

        let passthrough = build_signed_add_subset(ctx, &terms, mask, false);
        if !is_total_real_polynomial_passthrough(ctx, passthrough, 5) {
            continue;
        }

        return Some(required_conditions);
    }

    None
}

pub(crate) fn try_diff_integral_residual_product_zero_factor_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if !contains_diff_call_bounded(ctx, expr, 8) {
        return None;
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let [left, right] = factors.as_slice() else {
        return None;
    };

    for (zero_factor, residual_factor) in [(*left, *right), (*right, *left)] {
        if !exact_zero_noise_expr(ctx, zero_factor) {
            continue;
        }

        let required_conditions = residual_with_total_real_additive_passthrough_conditions(
            ctx,
            residual_factor,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )?;
        return Some((ctx.num(0), required_conditions));
    }

    None
}

fn compact_nonzero_residual_additive_passthrough_denominator_factor(
    ctx: &mut Context,
    denominator: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if !contains_diff_call_bounded(ctx, denominator, 8) {
        return None;
    }

    let view = cas_math::expr_nary::AddView::from_expr(ctx, denominator);
    let terms: Vec<_> = view.terms.into_iter().collect();
    let term_count = terms.len();
    if !(3..=6).contains(&term_count) {
        return None;
    }

    let full_mask = (1_u32 << term_count) - 1;
    for mask in 1..full_mask {
        let passthrough_mask = full_mask ^ mask;
        if passthrough_mask == 0 {
            continue;
        }

        let residual_candidate = build_residual_subset_candidate(ctx, &terms, mask);
        let Some((_zero, mut required_conditions)) =
            try_any_diff_integral_residual_wrapped_root_zero(
                ctx,
                residual_candidate,
                depth,
                direct_root_zero,
            )
        else {
            continue;
        };

        let passthrough = build_signed_add_subset(ctx, &terms, mask, false);
        if is_zero_constant(ctx, passthrough) {
            continue;
        }

        required_conditions.push(crate::ImplicitCondition::NonZero(passthrough));
        return Some((passthrough, required_conditions));
    }

    None
}

fn compact_nonzero_residual_passthrough_div_denominator(
    ctx: &mut Context,
    scale: &BigRational,
    denominator: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator, inner_denominator) = ctx.get(denominator).clone() else {
        return None;
    };
    if cas_math::numeric_eval::as_rational_const(ctx, inner_denominator)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }

    let (numerator, mut required_conditions) =
        compact_nonzero_residual_passthrough_denominator_factor(
            ctx,
            numerator,
            depth,
            direct_root_zero,
        )?;
    if !cas_math::numeric_eval::as_rational_const(ctx, inner_denominator)
        .is_some_and(|value| value.is_one())
    {
        required_conditions.push(crate::ImplicitCondition::NonZero(inner_denominator));
    }

    let numerator = apply_compacted_denominator_scale(ctx, scale, numerator)?;
    Some((
        ctx.add(Expr::Div(numerator, inner_denominator)),
        required_conditions,
    ))
}

fn apply_compacted_denominator_scale(
    ctx: &mut Context,
    scale: &BigRational,
    denominator: ExprId,
) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }
    if let Some(denominator_constant) = cas_math::numeric_eval::as_rational_const(ctx, denominator)
    {
        return Some(rational_expr(ctx, &(scale * denominator_constant)));
    }
    if scale.is_one() {
        return Some(denominator);
    }
    let scale_expr = rational_expr(ctx, scale);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[scale_expr, denominator],
    ))
}

fn nonzero_conditions_for_denominator_factors(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<Vec<crate::ImplicitCondition>> {
    let mut conditions = Vec::new();
    for factor in factors {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, *factor) {
            if value.is_zero() {
                return None;
            }
        } else {
            conditions.push(crate::ImplicitCondition::NonZero(*factor));
        }
    }
    Some(conditions)
}

fn compact_nonzero_residual_passthrough_product_denominator(
    ctx: &mut Context,
    scale: BigRational,
    denominator: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, denominator);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().enumerate() {
        let Some((compacted_factor, mut required_conditions)) =
            compact_nonzero_residual_passthrough_denominator_factor(
                ctx,
                *factor,
                depth,
                direct_root_zero,
            )
        else {
            continue;
        };

        let external_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(*factor))
            .collect();
        required_conditions.extend(nonzero_conditions_for_denominator_factors(
            ctx,
            &external_factors,
        )?);

        let compacted_factor = apply_compacted_denominator_scale(ctx, &scale, compacted_factor)?;
        let mut rebuilt_factors = Vec::with_capacity(factors.len());
        if !is_one_constant(ctx, compacted_factor) {
            rebuilt_factors.push(compacted_factor);
        }
        rebuilt_factors.extend(external_factors);
        let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &rebuilt_factors);
        return Some((denominator, required_conditions));
    }

    None
}

pub(crate) fn shifted_integral_residual_passthrough_nonzero_required_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let expr = strip_exact_zero_additive_noise_bounded(ctx, expr, 3);
    let (scale, expr) =
        strip_single_nonzero_constant_factor(ctx, expr).unwrap_or((BigRational::one(), expr));
    if let Some((_denominator, required_conditions)) =
        compact_nonzero_residual_passthrough_denominator_factor(
            ctx,
            expr,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )
    {
        return Some(required_conditions);
    }

    if let Some((_denominator, required_conditions)) =
        compact_nonzero_residual_additive_passthrough_denominator_factor(
            ctx,
            expr,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )
    {
        return Some(required_conditions);
    }

    if let Some((_denominator, required_conditions)) =
        compact_nonzero_residual_passthrough_div_denominator(
            ctx,
            &scale,
            expr,
            3,
            try_diff_integral_quadratic_exp_residual_direct_root_zero,
        )
    {
        return Some(required_conditions);
    }

    compact_nonzero_residual_passthrough_product_denominator(
        ctx,
        scale,
        expr,
        3,
        try_diff_integral_quadratic_exp_residual_direct_root_zero,
    )
    .map(|(_denominator, required_conditions)| required_conditions)
}

fn try_diff_integral_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> IntegralResidualRootResult {
    try_diff_integral_residual_constant_passthrough_quotient_with_nested_depth(
        ctx,
        expr,
        depth,
        1,
        direct_root_zero,
    )
}

fn try_diff_integral_residual_constant_passthrough_quotient_with_nested_depth(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    nested_quotient_depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> IntegralResidualRootResult {
    let expr = strip_exact_zero_additive_noise_bounded(ctx, expr, 3);
    let (scale, quotient_expr) =
        strip_single_nonzero_constant_factor(ctx, expr).unwrap_or((BigRational::one(), expr));
    let quotient_expr = strip_exact_zero_additive_noise_bounded(ctx, quotient_expr, 3);
    let Expr::Div(numerator, denominator) = ctx.get(quotient_expr).clone() else {
        return None;
    };
    if cas_math::numeric_eval::as_rational_const(ctx, denominator)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }

    if nested_quotient_depth > 0 {
        if let Some((inner_result, mut required_conditions)) =
            try_diff_integral_residual_constant_passthrough_quotient_with_nested_depth(
                ctx,
                numerator,
                depth,
                nested_quotient_depth - 1,
                direct_root_zero,
            )
        {
            let (denominator, denominator_conditions, compacted_denominator) =
                match compact_nonzero_residual_passthrough_denominator(
                    ctx,
                    denominator,
                    depth,
                    direct_root_zero,
                ) {
                    Some((denominator, required_conditions)) => {
                        (denominator, required_conditions, true)
                    }
                    None => (denominator, Vec::new(), false),
                };
            required_conditions.extend(denominator_conditions);
            if !compacted_denominator
                && cas_math::numeric_eval::as_rational_const(ctx, denominator).is_none()
            {
                required_conditions.push(crate::ImplicitCondition::NonZero(denominator));
            }
            let result = multiply_quotient_denominator(ctx, inner_result, denominator, &scale)?;
            return Some((result, required_conditions));
        }
    }

    let (numerator_scale, numerator) = strip_single_nonzero_constant_factor(ctx, numerator)
        .unwrap_or((BigRational::one(), numerator));
    let passthrough = strip_constant_passthrough(ctx, numerator)?;
    let (_zero, mut required_conditions) = try_diff_integral_residual_wrapped_root_zero(
        ctx,
        passthrough.core,
        depth,
        direct_root_zero,
    )?;
    let (denominator, denominator_conditions, compacted_denominator) =
        match compact_nonzero_residual_passthrough_denominator(
            ctx,
            denominator,
            depth,
            direct_root_zero,
        ) {
            Some((denominator, required_conditions)) => (denominator, required_conditions, true),
            None => (denominator, Vec::new(), false),
        };
    required_conditions.extend(denominator_conditions);
    if !compacted_denominator
        && cas_math::numeric_eval::as_rational_const(ctx, denominator).is_none()
    {
        required_conditions.push(crate::ImplicitCondition::NonZero(denominator));
    }

    let result_constant = passthrough.constant * numerator_scale * scale;
    let result = if result_constant.is_zero() {
        ctx.num(0)
    } else if let Some(denominator_constant) =
        cas_math::numeric_eval::as_rational_const(ctx, denominator)
    {
        if denominator_constant.is_zero() {
            return None;
        }
        rational_expr(ctx, &(result_constant / denominator_constant))
    } else if is_one_constant(ctx, denominator) {
        rational_expr(ctx, &result_constant)
    } else {
        let (result_constant, denominator) = if let Some(stripped_denominator) =
            strip_negative_one_product_factor(ctx, denominator)
        {
            (-result_constant, stripped_denominator)
        } else {
            (result_constant, denominator)
        };
        if let Some(result) = constant_over_fraction_denominator(ctx, &result_constant, denominator)
        {
            return Some((result, required_conditions));
        }
        let constant = rational_expr(ctx, &result_constant);
        ctx.add(Expr::Div(constant, denominator))
    };
    Some((result, required_conditions))
}

fn constant_over_fraction_denominator(
    ctx: &mut Context,
    constant: &BigRational,
    denominator: ExprId,
) -> Option<ExprId> {
    let Expr::Div(denominator_num, denominator_den) = ctx.get(denominator).clone() else {
        return None;
    };
    let denominator_num = cas_math::numeric_eval::as_rational_const(ctx, denominator_num)?;
    if denominator_num.is_zero() {
        return None;
    }
    let result_scale = constant / denominator_num;
    if let Some(denominator_den) = cas_math::numeric_eval::as_rational_const(ctx, denominator_den) {
        return Some(rational_expr(ctx, &(result_scale * denominator_den)));
    }
    if result_scale.is_one() {
        return Some(denominator_den);
    }
    let scale = rational_expr(ctx, &result_scale);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[scale, denominator_den],
    ))
}

fn strip_negative_one_product_factor(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let negative_one = -BigRational::one();
    let mut removed_negative_one = false;
    let mut remaining = Vec::with_capacity(factors.len() - 1);
    for factor in factors {
        if !removed_negative_one
            && cas_math::numeric_eval::as_rational_const(ctx, factor)
                .is_some_and(|value| value == negative_one)
        {
            removed_negative_one = true;
            continue;
        }
        remaining.push(factor);
    }

    removed_negative_one.then(|| cas_math::expr_nary::build_balanced_mul(ctx, &remaining))
}

fn multiply_quotient_denominator(
    ctx: &mut Context,
    quotient: ExprId,
    denominator: ExprId,
    scale: &BigRational,
) -> Option<ExprId> {
    let Expr::Div(numerator, inner_denominator) = ctx.get(quotient).clone() else {
        return None;
    };
    if scale.is_zero() {
        return None;
    }
    if is_one_constant(ctx, denominator) {
        return if scale.is_one() {
            Some(quotient)
        } else {
            let scale_expr = rational_expr(ctx, scale);
            Some(ctx.add(Expr::Mul(scale_expr, quotient)))
        };
    }
    let numerator = if scale.is_one() {
        numerator
    } else {
        let scale_expr = rational_expr(ctx, scale);
        ctx.add(Expr::Mul(scale_expr, numerator))
    };
    let combined_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[inner_denominator, denominator]);
    Some(ctx.add(Expr::Div(numerator, combined_denominator)))
}

fn strip_single_nonzero_constant_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut non_constant = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= factor_scale;
            continue;
        }
        if non_constant.replace(factor).is_some() {
            return None;
        }
    }

    (!scale.is_zero()).then_some((scale, non_constant?))
}

pub(crate) fn try_diff_integral_plain_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_plain_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_plain_trig_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_plain_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_plain_trig_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_integral_plain_trig_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_diff_integral_plain_trig_residual_direct_root_zero,
    )
}

pub(crate) fn try_diff_integral_inverse_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_inverse_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_inverse_trig_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_inverse_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_inverse_trig_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_integral_inverse_trig_residual_constant_passthrough_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_constant_passthrough_quotient(
        ctx,
        expr,
        2,
        try_diff_integral_inverse_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_inverse_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_polynomial_arctan_affine_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| {
                integrated_arcsin_polynomial_substitution_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                integrated_arctan_sqrt_var_reciprocal_diff_matches(ctx, diff_expr, divisor, right)
            })?;
    Some((ctx.num(0), required_conditions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn require_match<T>(value: Option<T>) -> T {
        match value {
            Some(value) => value,
            None => panic!("expected residual helper to match"),
        }
    }

    fn root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_hyperbolic_residual_root_zero(&mut ctx, expr).map(|result| render(&ctx, result))
    }

    fn sqrt_acosh_split_radical_residual_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_sqrt_acosh_split_radical_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn acosh_affine_reciprocal_sqrt_residual_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_acosh_affine_reciprocal_sqrt_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn ln_sqrt_polynomial_gap_residual_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_ln_sqrt_polynomial_gap_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn reciprocal_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_reciprocal_trig_residual_root_zero(&mut ctx, expr)
            .map(|result| render(&ctx, result))
    }

    fn inverse_trig_root_residual_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_inverse_reciprocal_trig_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn reciprocal_half_power_shared_denominator_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_reciprocal_half_power_shared_denominator_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn explicit_log_abs_antiderivative_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_log_abs_antiderivative_residual_root_zero(&mut ctx, expr)
            .map(|(result, _conditions)| render(&ctx, result))
    }

    #[test]
    fn reciprocal_half_power_shared_denominator_residual_cancels_function_base() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "ln(x)^(-1/2)/(2*x) - ln(x)^(1/2)/(2*x*ln(x))"
            ),
            Some(("0".to_string(), vec![]))
        );
    }

    #[test]
    fn reciprocal_half_power_shared_denominator_residual_cancels_shifted_power_denominator() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(x^2+x+1)^(1/2)*(2*x+1)/(x^2+x+1)^3 - (2*x+1)/(x^2+x+1)^(5/2)"
            ),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(x^2-1)^(1/2)*2*x/(x^2-1)^3 - 2*x/(x^2-1)^(5/2)"
            ),
            Some(("0".to_string(), vec!["x < -1 or x > 1".to_string()]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(2*x^2+2*x-3)^(1/2)*(4*x+2)/(2*x^2+2*x-3)^3 - (4*x+2)/(2*x^2+2*x-3)^(5/2)"
            ),
            Some((
                "0".to_string(),
                vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()]
            ))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(2*x^2+2*x-3)^(3/2)*(4*x+2)/(2*x^2+2*x-3)^5 - (4*x+2)/(2*x^2+2*x-3)^(7/2)"
            ),
            Some((
                "0".to_string(),
                vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()]
            ))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(8*x^3+12*x^2-8*x-6)/(2*x^2+2*x-3)^(7/2) - (4*x+2)/(2*x^2+2*x-3)^(5/2)"
            ),
            Some((
                "0".to_string(),
                vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()]
            ))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(8*x^3+12*x^2-8*x-6)/((2*x^2+2*x-3)^(3/2)*(2*x^2+2*x-3)^2) - (4*x+2)/(2*x^2+2*x-3)^(5/2)"
            ),
            Some((
                "0".to_string(),
                vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()]
            ))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(x^2+x+1)^(1/2)*(50*x^5+125*x^4+200*x^3+175*x^2+100*x+25) / \
                 ((x^2+x+1)*(25*(x^4+2*x^3+3*x^2+2*x+1)^2+25*x*(x^4+2*x^3+3*x^2+2*x+1)^2+25*x^2*(x^4+2*x^3+3*x^2+2*x+1)^2)) \
                 - (2*x+1)/(x^2+x+1)^(7/2)"
            ),
            Some(("0".to_string(), vec![]))
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_scaled_polynomial_radicand_content() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "2*(2*x*(2*x-2))^(-1/2) - (x^2-x)^(-1/2)"
            ),
            // Proportional-base normalization shares one base, so the
            // previously duplicated interval condition reports once.
            Some(("0".to_string(), vec!["x < 0 or x > 1".to_string()]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(x^2-x)^(-1/2) - 2*(-2*x*(2-2*x))^(-1/2)"
            ),
            // Proportional-base normalization shares one base, so the
            // previously duplicated interval condition reports once.
            Some(("0".to_string(), vec!["x < 0 or x > 1".to_string()]))
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_scaled_sqrt_product_denominator_content() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "1/(sqrt(2)*sqrt(x)*sqrt(2*x+6)) - 1/(sqrt(2*x)*sqrt(2*x+6))"
            ),
            Some(("0".to_string(), vec![]))
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_scaled_sqrt_rationalized_positive_root() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "2*sqrt(x^2+x+1) - 2*(x^2+x+1)/sqrt(x^2+x+1)"
            ),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result("2*sqrt(x) - 2*x/sqrt(x)"),
            Some(("0".to_string(), vec!["x > 0".to_string()]))
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_three_term_common_denominator_sum() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "3/2*x^(1/2) - (3*x^2-1)/(2*x^(3/2)) - 1/2*x^(-3/2)"
            ),
            Some(("0".to_string(), vec!["x > 0".to_string()]))
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_quotient_wrapped_scaled_sqrt_rationalized_root() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(2*sqrt(x^2+x+1) - 2*(x^2+x+1)/sqrt(x^2+x+1))/(x+2)"
            ),
            Some(("0".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "(2*sqrt(x^2+x+1) - 3*(x^2+x+1)/sqrt(x^2+x+1))/(x+2)"
            ),
            None
        );
    }

    #[test]
    fn reciprocal_half_power_residual_cancels_shifted_reciprocal_difference() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "1/(2*sqrt(x^2+x+1) - 2*(x^2+x+1)/sqrt(x^2+x+1) + x + 2) - 1/(x+2)"
            ),
            Some(("0".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "1/(2*sqrt(x^2+x+1) - 3*(x^2+x+1)/sqrt(x^2+x+1) + x + 2) - 1/(x+2)"
            ),
            None
        );
    }

    #[test]
    fn reciprocal_half_power_integrate_resolved_residual_cancels_shifted_reciprocal_difference() {
        let mut ctx = Context::new();
        let input =
            "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let result = try_integrate_resolved_reciprocal_half_power_residual_root_zero(
            &mut ctx, expr,
        )
        .map(|(result, required_conditions)| {
            (
                render(&ctx, result),
                required_conditions
                    .into_iter()
                    .map(|condition| condition.display(&ctx))
                    .collect::<Vec<_>>(),
            )
        });

        assert_eq!(result, Some(("0".to_string(), vec!["x ≠ -2".to_string()])));
    }

    #[test]
    fn reciprocal_half_power_integrate_resolved_mismatch_compacts_shifted_reciprocal_difference() {
        let mut ctx = Context::new();
        let input =
            "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 3*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let result =
            try_integrate_resolved_reciprocal_half_power_residual_compact_mismatch(&mut ctx, expr)
                .map(|(result, required_conditions)| {
                    (
                        render(&ctx, result),
                        required_conditions
                            .into_iter()
                            .map(|condition| condition.display(&ctx))
                            .collect::<Vec<_>>(),
                    )
                });

        let Some((rendered, required_conditions)) = result else {
            panic!("expected compact mismatch");
        };
        assert_ne!(rendered, "0");
        assert!(!rendered.contains("integrate"));
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition == "x ≠ -2"),
            "expected passthrough denominator condition, got {required_conditions:?}"
        );
    }

    #[test]
    fn explicit_log_abs_antiderivative_residual_cancels_reordered_partial_fraction() {
        assert_eq!(
            explicit_log_abs_antiderivative_residual_result(
                "diff((-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1)), x) - (3*x+5)/(x^3-x^2-x+1)"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn reciprocal_half_power_shared_denominator_residual_rejects_mismatched_scale() {
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "ln(x)^(-1/2)/(2*x) - ln(x)^(1/2)/(3*x*ln(x))"
            ),
            None
        );
        assert_eq!(
            reciprocal_half_power_shared_denominator_result(
                "2*sqrt(x^2+x+1) - 3*(x^2+x+1)/sqrt(x^2+x+1)"
            ),
            None
        );
    }

    fn diff_sqrt_log_plus_constant_root_residual_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_sqrt_log_plus_constant_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    #[test]
    fn diff_sqrt_log_plus_constant_residual_cancels_direct_chain_rule_pair() {
        assert_eq!(
            diff_sqrt_log_plus_constant_root_residual_result(
                "diff(sqrt(ln(x)+1), x) - 1/(2*x*sqrt(ln(x)+1))"
            ),
            Some((
                "0".to_string(),
                vec!["ln(x) + 1 > 0".to_string(), "x > 0".to_string()]
            ))
        );
    }

    #[test]
    fn diff_sqrt_log_plus_constant_residual_rejects_mismatched_variable() {
        assert_eq!(
            diff_sqrt_log_plus_constant_root_residual_result(
                "diff(sqrt(ln(x)+1), y) - 1/(2*x*sqrt(ln(x)+1))"
            ),
            None
        );
    }

    #[test]
    fn diff_sqrt_log_plus_constant_residual_rejects_mismatched_scale() {
        assert_eq!(
            diff_sqrt_log_plus_constant_root_residual_result(
                "diff(sqrt(ln(x)+1), x) - 1/(3*x*sqrt(ln(x)+1))"
            ),
            None
        );
    }

    fn integral_reciprocal_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_reciprocal_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_plain_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_plain_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_exp_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_exp_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_exp_trig_compact_mismatch_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_exp_trig_residual_compact_mismatch(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_inverse_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_inverse_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_plain_trig_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_plain_trig_residual_constant_passthrough_quotient(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_reciprocal_trig_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_reciprocal_trig_residual_constant_passthrough_quotient(&mut ctx, expr)
            .map(|(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            })
    }

    fn integral_residual_passthrough_quotient_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_residual_passthrough_quotient_root_one(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_inverse_trig_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_inverse_trig_residual_constant_passthrough_quotient(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_quadratic_exp_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_quadratic_exp_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_hyperbolic_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_hyperbolic_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_hyperbolic_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_hyperbolic_residual_constant_passthrough_quotient(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_hyperbolic_reciprocal_shifted_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_integral_hyperbolic_reciprocal_shifted_difference_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_rational_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_rational_quadratic_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_rational_passthrough_quotient_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_rational_quadratic_residual_constant_passthrough_quotient(&mut ctx, expr)
            .map(|(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            })
    }

    fn integral_rational_reciprocal_shifted_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_rational_quadratic_residual_reciprocal_shifted_difference_root_zero(
            &mut ctx, expr,
        )
        .map(|(result, required_conditions)| {
            (
                render(&ctx, result),
                required_conditions
                    .into_iter()
                    .map(|condition| condition.display(&ctx))
                    .collect(),
            )
        })
    }

    fn integral_residual_reciprocal_shifted_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_residual_reciprocal_shifted_difference_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_residual_shifted_quotient_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_residual_shifted_quotient_difference_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn integral_residual_product_zero_factor_result(input: &str) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_residual_product_zero_factor_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn explicit_positive_quadratic_cube_antiderivative_residual_result(
        input: &str,
    ) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_positive_quadratic_cube_antiderivative_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn explicit_positive_quadratic_square_antiderivative_residual_result(
        input: &str,
    ) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_positive_quadratic_square_antiderivative_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn explicit_positive_quadratic_cube_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_positive_quadratic_cube_antiderivative_residual_constant_passthrough_quotient(
            &mut ctx, expr,
        )
        .map(|(result, required_conditions)| {
            (
                render(&ctx, result),
                required_conditions
                    .into_iter()
                    .map(|condition| condition.display(&ctx))
                    .collect(),
            )
        })
    }

    fn explicit_positive_quadratic_square_passthrough_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_positive_quadratic_square_antiderivative_residual_constant_passthrough_quotient(
            &mut ctx, expr,
        )
        .map(|(result, required_conditions)| {
            (
                render(&ctx, result),
                required_conditions
                    .into_iter()
                    .map(|condition| condition.display(&ctx))
                    .collect(),
            )
        })
    }

    fn explicit_high_log_power_product_antiderivative_residual_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_high_log_power_product_antiderivative_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn explicit_quadratic_positive_quadratic_log_antiderivative_residual_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_root_zero(
            &mut ctx, expr,
        )
        .map(|(result, required_conditions)| {
            (
                render(&ctx, result),
                required_conditions
                    .into_iter()
                    .map(|condition| condition.display(&ctx))
                    .collect(),
            )
        })
    }

    fn explicit_quadratic_affine_log_antiderivative_residual_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_explicit_quadratic_affine_log_antiderivative_residual_root_zero(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|condition| condition.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    #[test]
    fn explicit_positive_quadratic_cube_antiderivative_residual_cancels_scaled_affine_quartic() {
        assert_eq!(
            explicit_positive_quadratic_cube_antiderivative_residual_result(
                "diff(3/16*arctan(2*x+1)+(6*x^3+9*x^2+7*x+2)/(4*(4*x^2+4*x+2)^2)+(-2*x-1)/(2*(4*x^2+4*x+2)),x)-(2*x+1)^4/(((2*x+1)^2+1)^3)"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn explicit_positive_quadratic_square_antiderivative_residual_cancels_quotient_wrapper() {
        let residual = "diff(1/2*arctan(x)+x/(2*(x^2+1)),x)-1/(x^2+1)^2";
        let quotient = format!("({residual})/(x+1)");
        let mut ctx = Context::new();
        let expr = parse(&quotient, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {quotient}: {err:?}"));
        let Some((result, required_conditions)) =
            try_explicit_positive_quadratic_square_antiderivative_residual_root_zero(
                &mut ctx, expr,
            )
        else {
            panic!("wrapped quotient residual should cancel");
        };
        assert_eq!(render(&ctx, result), "0");
        assert_eq!(required_conditions.len(), 1);
        let crate::ImplicitCondition::NonZero(required_denominator) = required_conditions[0] else {
            panic!("expected wrapped quotient to add only a nonzero denominator condition");
        };
        assert_eq!(render(&ctx, required_denominator), "x + 1");
        assert_eq!(simplify_text(&quotient), "0");

        assert_eq!(
            explicit_positive_quadratic_square_antiderivative_residual_result(residual),
            Some("0".to_string())
        );
    }

    #[test]
    fn explicit_positive_quadratic_square_antiderivative_residual_compacts_passthrough_quotient() {
        let residual = "diff(1/2*arctan(x)+x/(2*(x^2+1)),x)-1/(x^2+1)^2";
        let input = format!("(({residual}) + 1)/(x+2)");
        assert_eq!(
            explicit_positive_quadratic_square_passthrough_quotient_result(&input),
            Some(("1 / (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "1 / (x + 2)");
    }

    #[test]
    fn explicit_positive_quadratic_cube_antiderivative_residual_cancels_wrappers() {
        let residual = "diff(3/16*arctan(2*x+1)+(6*x^3+9*x^2+7*x+2)/(4*(4*x^2+4*x+2)^2)+(-2*x-1)/(2*(4*x^2+4*x+2)),x)-(2*x+1)^4/(((2*x+1)^2+1)^3)";
        let cases = [
            format!("0 - ({residual})"),
            format!("({residual}) + 0"),
            format!("2*({residual})"),
        ];

        for input in cases {
            assert_eq!(
                explicit_positive_quadratic_cube_antiderivative_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }

        let quotient = format!("({residual})/(x+1)");
        let mut ctx = Context::new();
        let expr = parse(&quotient, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {quotient}: {err:?}"));
        let Some((result, required_conditions)) =
            try_explicit_positive_quadratic_cube_antiderivative_residual_root_zero(&mut ctx, expr)
        else {
            panic!("wrapped quotient residual should cancel");
        };
        assert_eq!(render(&ctx, result), "0");
        assert_eq!(required_conditions.len(), 1);
        let crate::ImplicitCondition::NonZero(required_denominator) = required_conditions[0] else {
            panic!("expected wrapped quotient to add only a nonzero denominator condition");
        };
        assert_eq!(render(&ctx, required_denominator), "x + 1");
        assert_eq!(simplify_text(&quotient), "0");
    }

    #[test]
    fn explicit_positive_quadratic_cube_antiderivative_residual_compacts_passthrough_quotient() {
        let residual = "diff(3/8*arctan(x)+(3*x^3+5*x)/(8*(x^2+1)^2),x)-1/(x^2+1)^3";
        let input = format!("(1 - ({residual}))/(x+2)");
        assert_eq!(
            explicit_positive_quadratic_cube_passthrough_quotient_result(&input),
            Some(("1 / (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "1 / (x + 2)");
    }

    #[test]
    fn explicit_high_log_power_product_antiderivative_residual_cancels_shifted_positive_quadratic()
    {
        let residual = "diff(integrate((2*x+1)*ln(x^2+x+1)^4, x), x) - (2*x+1)*ln(x^2+x+1)^4";
        assert_eq!(
            explicit_high_log_power_product_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(simplify_text(residual), "0");
    }

    #[test]
    fn explicit_high_log_power_product_antiderivative_residual_cancels_conditional_quadratic() {
        let residual = "diff(integrate(2*x*ln(x^2-1)^4, x), x) - 2*x*ln(x^2-1)^4";
        assert_eq!(
            explicit_high_log_power_product_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec!["x < -1 or x > 1".to_string()]))
        );
        assert_eq!(simplify_text(residual), "0");
    }

    #[test]
    fn explicit_log_power_product_antiderivative_residual_cancels_constant_base_square() {
        let residual = "diff(integrate(2*x*log(2,x^2+1)^2, x), x) - 2*x*log(2,x^2+1)^2";
        assert_eq!(
            explicit_high_log_power_product_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(simplify_text(residual), "0");

        let rendered_residual = "diff((x^2+1)*(log(2,x^2+1)^2 + 2/ln(2)^2 - 2*log(2,x^2+1)/ln(2)), x) - 2*x*log(2,x^2+1)^2";
        assert_eq!(
            explicit_high_log_power_product_antiderivative_residual_result(rendered_residual),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(simplify_text(rendered_residual), "0");
    }

    #[test]
    fn explicit_quadratic_affine_log_antiderivative_residual_cancels_negative_slope() {
        let residual = "diff(integrate(x^2*ln(1-2*x), x), x) - x^2*ln(1-2*x)";
        assert_eq!(
            explicit_quadratic_affine_log_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec!["x < 1/2".to_string()]))
        );
        assert_eq!(simplify_text(residual), "0");
    }

    #[test]
    fn explicit_quadratic_positive_quadratic_log_antiderivative_residual_cancels_shifted_argument()
    {
        let residual = "diff(integrate(x^2*ln(x^2+x+1), x), x) - x^2*ln(x^2+x+1)";
        assert_eq!(
            explicit_quadratic_positive_quadratic_log_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(simplify_text(residual), "0");
    }

    #[test]
    fn explicit_quadratic_positive_quadratic_log_antiderivative_residual_cancels_rendered_linear() {
        let residual = "diff(1/2*x^2*ln(x^2+x+1) - 3/2*arctan((2*x+1)/sqrt(3))/sqrt(3) - 1/2*x^2 + 1/4*ln(x^2+x+1) + 1/2*x, x) - x*ln(x^2+x+1)";
        assert_eq!(
            explicit_quadratic_positive_quadratic_log_antiderivative_residual_result(residual),
            Some(("0".to_string(), vec![]))
        );
        assert_eq!(simplify_text(residual), "0");
    }

    fn diff_arctan_sqrt_positive_quotient_shifted_one_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_one_root(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_compact_mismatch(&mut ctx, expr)
            .map(|(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            })
    }

    fn diff_inverse_reciprocal_trig_shifted_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_inverse_reciprocal_trig_shifted_quotient_root_one(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_inverse_reciprocal_trig_shifted_quotient_compact_mismatch(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn simplify_text(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::new();
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _steps) = simplifier.simplify(expr);
        render(&simplifier.context, result)
    }

    fn simplify_text_with_default_rules(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::with_default_rules();
        simplifier.disable_rule("Double Angle Identity");
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _steps) = simplifier.simplify(expr);
        render(&simplifier.context, result)
    }

    #[test]
    fn diff_log_abs_hyperbolic_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(ln(abs(sinh(2*x+1))), x)/2 - 1/tanh(2*x+1)",
            "diff(ln(abs(sinh(2*x+1))), x)/2 - cosh(2*x+1)/sinh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - tanh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - sinh(2*x+1)/cosh(2*x+1)",
            "diff(ln(cosh(sqrt(2*x))), x) - tanh(sqrt(2*x))/sqrt(2*x)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
            assert_eq!(simplify_text_with_default_rules(input), "0", "{input}");
        }
    }

    #[test]
    fn hyperbolic_tanh_common_factor_residual_root_cancels_scaled_sqrt_forms() {
        let cases = [
            "sinh((2*x)^(1/2)) * (2*x)^(-1/2) / cosh((2*x)^(1/2)) - tanh((2*x)^(1/2)) * (2*x)^(-1/2)",
            "tanh((2*x)^(1/2)) * (2*x)^(-1/2) - sinh((2*x)^(1/2)) * (2*x)^(-1/2) / cosh((2*x)^(1/2))",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_hyperbolic_cubic_residual_root_cancels_negative_targets() {
        let cases = [
            "diff(-1/3*sinh(2*x+1)^3, x) - (-2*cosh(2*x+1)*sinh(2*x+1)^2)",
            "diff(1/3*cosh(2*x+1)^3, x) - 2*sinh(2*x+1)*cosh(2*x+1)^2",
            "diff(-1/3*cosh(x)^3, x) - (-sinh(x)*cosh(x)^2)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
            assert_eq!(simplify_text_with_default_rules(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_hyperbolic_square_power_reduction_residual_root_cancels_compact_primitives() {
        let cases = [
            "diff(1/8*sinh(4*x)-x/2, x) - sinh(2*x)^2",
            "diff(1/8*sinh(4*x)+x/2, x) - cosh(2*x)^2",
            "diff(1/4*sinh(2*x)-x/2, x) - sinh(x)^2",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
            assert_eq!(simplify_text_with_default_rules(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_hyperbolic_square_product_power_reduction_residual_root_cancels_compact_primitives() {
        let cases = [
            "diff(1/32*sinh(4*x)-x/8, x) - sinh(x)^2*cosh(x)^2",
            "diff(1/8*sinh(4*x)-x/2, x) - 4*sinh(x)^2*cosh(x)^2",
            "diff(1/64*sinh(4*(2*x+1))-x/8, x) - sinh(2*x+1)^2*cosh(2*x+1)^2",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
            assert_eq!(simplify_text_with_default_rules(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_log_abs_reciprocal_trig_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(ln(abs((sin(2*x+1)+1)/cos(2*x+1))), x)/2 - sec(2*x+1)",
            "diff(ln(abs((sin(2*x+1)+1)/cos(2*x+1))), x)/2 - 1/cos(2*x+1)",
            "diff(ln(abs((cos(2*x+1)-1)/sin(2*x+1))), x)/2 - csc(2*x+1)",
            "diff(ln(abs((cos(2*x+1)-1)/sin(2*x+1))), x)/2 - 1/sin(2*x+1)",
            "diff(-2/3*ln(abs((cos((2-3*x)/2)-1)/sin((2-3*x)/2))), x) - csc((2-3*x)/2)",
            "diff(-2/3*ln(abs((cos((2-3*x)/2)-1)/sin((2-3*x)/2))), x) - 1/sin((2-3*x)/2)",
        ];

        for input in cases {
            assert_eq!(
                reciprocal_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_bounded_inverse_trig_sqrt_rationalized_residual_root_cancels() {
        let (result, required_conditions) = require_match(inverse_trig_root_residual_result(
            "diff(arcsin(sqrt(x)), x) - sqrt(x)*sqrt(1-x)/(2*x*(1-x))",
        ));
        assert_eq!(result, "0");
        assert_eq!(
            required_conditions,
            vec!["x > 0".to_string(), "x < 1".to_string()]
        );
        assert!(inverse_trig_root_residual_result(
            "diff(arcsin(sqrt(x)), x) + sqrt(x)*sqrt(1-x)/(2*x*(1-x))"
        )
        .is_none());
    }

    #[test]
    fn diff_bounded_inverse_trig_sqrt_affine_quotient_residual_root_cancels() {
        let (result, required_conditions) = require_match(inverse_trig_root_residual_result(
            "diff(arccos(sqrt((x+1)/(x+3))), x) + sqrt(2)/(2*(x+3)*sqrt(x+1))",
        ));
        assert_eq!(result, "0");
        assert_eq!(required_conditions, vec!["x > -1".to_string()]);

        assert!(inverse_trig_root_residual_result(
            "diff(arccos(sqrt((x+1)/(x+3))), x) - sqrt(2)/(2*(x+3)*sqrt(x+1))"
        )
        .is_none());
    }

    #[test]
    fn diff_integral_reciprocal_trig_residual_root_verifies_antiderivative_first() {
        let cases = [
            "diff(integrate(sec((3*x+2)/2), x), x) - sec((3*x+2)/2)",
            "diff(integrate(1/cos((3*x+2)/2), x), x) - 1/cos((3*x+2)/2)",
            "diff(integrate(csc((2-3*x)/2), x), x) - csc((2-3*x)/2)",
            "diff(integrate(csc((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "diff(integrate(1/sin((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "diff(integrate(2*x*sec(x^2)*tan(x^2), x), x) - 2*x*sec(x^2)*tan(x^2)",
            "diff(integrate(2*x*csc(x^2)*cot(x^2), x), x) - 2*x*csc(x^2)*cot(x^2)",
            "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)",
            "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)",
            "diff(integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x), x) - (2*x+1)*sin(x^2+x)/cos(x^2+x)^2",
            "diff(integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x), x) - (2*x+1)*cos(x^2+x)/sin(x^2+x)^2",
        ];

        for input in cases {
            assert_eq!(
                integral_reciprocal_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_reciprocal_trig_root_residual_result(
                "diff(integrate(csc((2-3*x)/2), x), y) - csc((2-3*x)/2)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_reciprocal_trig_residual_root_cancels_algebraic_wrappers() {
        let sec_residual = "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)";
        let csc_residual = "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)";
        let cases = [
            format!("({sec_residual}) + 0"),
            format!("0 - ({sec_residual})"),
            format!("2*({sec_residual})"),
            format!("({sec_residual})/(x+1)"),
            format!("-({csc_residual})"),
            format!("(3/2)*(({csc_residual}) + 0)"),
        ];

        for input in cases {
            assert_eq!(
                integral_reciprocal_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_cancels_algebraic_wrappers() {
        let tan_residual =
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)";
        let cot_residual =
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)";
        let cases = [
            format!("({tan_residual}) + 0"),
            format!("0 - ({tan_residual})"),
            format!("2*({tan_residual})"),
            format!("({tan_residual})/(x+1)"),
            format!("-({cot_residual})"),
            format!("(3/2)*(({cot_residual}) + 0)"),
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_antiderivative_first() {
        let cases = [
            "diff(integrate(tan(2*x+1), x), x) - tan(2*x+1)",
            "diff(integrate(sin(2*x+1)/cos(2*x+1), x), x) - sin(2*x+1)/cos(2*x+1)",
            "diff(integrate(cot(2*x+1), x), x) - cot(2*x+1)",
            "diff(integrate(cos(2*x+1)/sin(2*x+1), x), x) - cos(2*x+1)/sin(2*x+1)",
            "diff(integrate(2*x*tan(x^2), x), x) - 2*x*tan(x^2)",
            "diff(integrate(3*x^2*cot(x^3), x), x) - 3*x^2*cot(x^3)",
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)",
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(cot(2*x+1), x), y) - cot(2*x+1)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_fifth_power_reduction() {
        let cases = [
            "diff(integrate(sin(x)^5, x), x) - sin(x)^5",
            "diff(integrate(cos(x)^5, x), x) - cos(x)^5",
            "diff(integrate(sin(2*x+1)^5, x), x) - sin(2*x+1)^5",
            "diff(integrate(cos(2*x+1)^5, x), x) - cos(2*x+1)^5",
            "diff(1/5*(10/3*cos(x)^3 - cos(x)^5 - 5*cos(x)), x) - sin(x)^5",
            "diff(1/5*(sin(x)^5 + 5*sin(x) - 10/3*sin(x)^3), x) - cos(x)^5",
            "diff(1/10*(10/3*cos(2*x+1)^3 - cos(2*x+1)^5 - 5*cos(2*x+1)), x) - sin(2*x+1)^5",
            "diff(1/10*(sin(2*x+1)^5 + 5*sin(2*x+1) - 10/3*sin(2*x+1)^3), x) - cos(2*x+1)^5",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(sin(2*x+1)^5, x), y) - sin(2*x+1)^5"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_seventh_power_reduction() {
        let cases = [
            "diff(integrate(sin(x)^7, x), x) - sin(x)^7",
            "diff(integrate(cos(x)^7, x), x) - cos(x)^7",
            "diff(integrate(sin(2*x+1)^7, x), x) - sin(2*x+1)^7",
            "diff(integrate(cos(2*x+1)^7, x), x) - cos(2*x+1)^7",
            "diff(integrate(sin(1-2*x)^7, x), x) - sin(1-2*x)^7",
            "diff(integrate(cos(1-2*x)^7, x), x) - cos(1-2*x)^7",
            "diff(1/2*(-cos(2*x+1)+cos(2*x+1)^3-3/5*cos(2*x+1)^5+1/7*cos(2*x+1)^7), x) - sin(2*x+1)^7",
            "diff(1/2*(sin(2*x+1)-sin(2*x+1)^3+3/5*sin(2*x+1)^5-1/7*sin(2*x+1)^7), x) - cos(2*x+1)^7",
            "diff(-1/2*(-cos(1-2*x)+cos(1-2*x)^3-3/5*cos(1-2*x)^5+1/7*cos(1-2*x)^7), x) - sin(1-2*x)^7",
            "diff(-1/2*(sin(1-2*x)-sin(1-2*x)^3+3/5*sin(1-2*x)^5-1/7*sin(1-2*x)^7), x) - cos(1-2*x)^7",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(sin(2*x+1)^7, x), y) - sin(2*x+1)^7"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_explicit_fourth_power_primitive() {
        let cases = [
            "diff(3*x/8 - sin(2*x)/4 + sin(4*x)/32, x) - sin(x)^4",
            "diff(3*x/8 + sin(2*x)/4 + sin(4*x)/32, x) - cos(x)^4",
            "diff(3*x/8 - sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - sin(2*x+1)^4",
            "diff(3*x/8 + sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - cos(2*x+1)^4",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_tan_fourth_without_global_detour() {
        let cases = [
            "diff(integrate(tan(x)^4, x), x) - tan(x)^4",
            "diff(tan(x)^3/3 + x - tan(x), x) - tan(x)^4",
            "diff(integrate(tan(2*x+1)^4, x), x) - tan(2*x+1)^4",
            "diff(-tan(2*x+1)/2 + tan(2*x+1)^3/6 + x, x) - tan(2*x+1)^4",
            "diff(-tan(2*x+1)/2 + tan(2*x+1)^3/6 + x, x) - sin(2*x+1)^4/cos(2*x+1)^4",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_cot_fourth_without_global_detour() {
        let cases = [
            "diff(integrate(cot(x)^4, x), x) - cot(x)^4",
            "diff(x + cot(x) - cot(x)^3/3, x) - cot(x)^4",
            "diff(integrate(cos(2*x+1)^4/sin(2*x+1)^4, x), x) - cos(2*x+1)^4/sin(2*x+1)^4",
            "diff(x + cot(2*x+1)/2 - cot(2*x+1)^3/6, x) - cos(2*x+1)^4/sin(2*x+1)^4",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_tan_sixth_without_global_detour() {
        let cases = [
            "diff(integrate(tan(x)^6, x), x) - tan(x)^6",
            "diff(tan(x)^5/5 - tan(x)^3/3 + tan(x) - x, x) - tan(x)^6",
            "diff(integrate(tan(2*x+1)^6, x), x) - tan(2*x+1)^6",
            "diff(tan(2*x+1)^5/10 - tan(2*x+1)^3/6 + tan(2*x+1)/2 - x, x) - tan(2*x+1)^6",
            "diff(-tan(1-2*x)/2 - tan(1-2*x)^5/10 + tan(1-2*x)^3/6 - x, x) - tan(1-2*x)^6",
            "diff(tan(2*x+1)^5/10 - tan(2*x+1)^3/6 + tan(2*x+1)/2 - x, x) - sin(2*x+1)^6/cos(2*x+1)^6",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_cot_sixth_without_global_detour() {
        let cases = [
            "diff(integrate(cot(x)^6, x), x) - cot(x)^6",
            "diff(-cot(x)^5/5 + cot(x)^3/3 - cot(x) - x, x) - cot(x)^6",
            "diff(integrate(cot(2*x+1)^6, x), x) - cot(2*x+1)^6",
            "diff(-cot(2*x+1)^5/10 + cot(2*x+1)^3/6 - cot(2*x+1)/2 - x, x) - cot(2*x+1)^6",
            "diff(cot(1-2*x)/2 - cot(1-2*x)^3/6 + cot(1-2*x)^5/10 - x, x) - cot(1-2*x)^6",
            "diff(-cot(2*x+1)^5/10 + cot(2*x+1)^3/6 - cot(2*x+1)/2 - x, x) - cos(2*x+1)^6/sin(2*x+1)^6",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_sec_fourth_without_global_detour() {
        let cases = [
            "diff(integrate(sec(x)^4, x), x) - sec(x)^4",
            "diff(tan(x) + tan(x)^3/3, x) - sec(x)^4",
            "diff(integrate(1/cos(2*x+1)^4, x), x) - 1/cos(2*x+1)^4",
            "diff(tan(2*x+1)/2 + tan(2*x+1)^3/6, x) - 1/cos(2*x+1)^4",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_csc_fourth_without_global_detour() {
        let cases = [
            "diff(integrate(csc(x)^4, x), x) - csc(x)^4",
            "diff(-cot(x) - cot(x)^3/3, x) - csc(x)^4",
            "diff(integrate(1/sin(2*x+1)^4, x), x) - 1/sin(2*x+1)^4",
            "diff(-cot(2*x+1)/2 - cot(2*x+1)^3/6, x) - 1/sin(2*x+1)^4",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_sec_sixth_without_global_detour() {
        let cases = [
            "diff(integrate(sec(x)^6, x), x) - sec(x)^6",
            "diff(tan(x) + 2*tan(x)^3/3 + tan(x)^5/5, x) - sec(x)^6",
            "diff(integrate(1/cos(2*x+1)^6, x), x) - 1/cos(2*x+1)^6",
            "diff(tan(2*x+1)/2 + tan(2*x+1)^3/3 + tan(2*x+1)^5/10, x) - 1/cos(2*x+1)^6",
            "diff(-tan(1-2*x)/2 - tan(1-2*x)^3/3 - tan(1-2*x)^5/10, x) - sec(1-2*x)^6",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_csc_sixth_without_global_detour() {
        let cases = [
            "diff(integrate(csc(x)^6, x), x) - csc(x)^6",
            "diff(-cot(x) - 2*cot(x)^3/3 - cot(x)^5/5, x) - csc(x)^6",
            "diff(integrate(1/sin(2*x+1)^6, x), x) - 1/sin(2*x+1)^6",
            "diff(-cot(2*x+1)/2 - cot(2*x+1)^3/3 - cot(2*x+1)^5/10, x) - 1/sin(2*x+1)^6",
            "diff(cot(1-2*x)/2 + cot(1-2*x)^3/3 + cot(1-2*x)^5/10, x) - csc(1-2*x)^6",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_tan_eighth_without_global_detour() {
        let cases = [
            "diff(integrate(tan(x)^8, x), x) - tan(x)^8",
            "diff(tan(x)^7/7 - tan(x)^5/5 + tan(x)^3/3 - tan(x) + x, x) - tan(x)^8",
            "diff(integrate(tan(2*x+1)^8, x), x) - tan(2*x+1)^8",
            "diff(tan(2*x+1)^7/14 - tan(2*x+1)^5/10 + tan(2*x+1)^3/6 - tan(2*x+1)/2 + x, x) - tan(2*x+1)^8",
            "diff(-tan(1-2*x)^7/14 + tan(1-2*x)^5/10 - tan(1-2*x)^3/6 + tan(1-2*x)/2 + x, x) - tan(1-2*x)^8",
            "diff(tan(2*x+1)^7/14 - tan(2*x+1)^5/10 + tan(2*x+1)^3/6 - tan(2*x+1)/2 + x, x) - sin(2*x+1)^8/cos(2*x+1)^8",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_cot_eighth_without_global_detour() {
        let cases = [
            "diff(integrate(cot(x)^8, x), x) - cot(x)^8",
            "diff(-cot(x)^7/7 + cot(x)^5/5 - cot(x)^3/3 + cot(x) + x, x) - cot(x)^8",
            "diff(integrate(cot(2*x+1)^8, x), x) - cot(2*x+1)^8",
            "diff(-cot(2*x+1)^7/14 + cot(2*x+1)^5/10 - cot(2*x+1)^3/6 + cot(2*x+1)/2 + x, x) - cot(2*x+1)^8",
            "diff(cot(1-2*x)^7/14 - cot(1-2*x)^5/10 + cot(1-2*x)^3/6 - cot(1-2*x)/2 + x, x) - cot(1-2*x)^8",
            "diff(-cot(2*x+1)^7/14 + cot(2*x+1)^5/10 - cot(2*x+1)^3/6 + cot(2*x+1)/2 + x, x) - cos(2*x+1)^8/sin(2*x+1)^8",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_sec_eighth_without_global_detour() {
        let cases = [
            "diff(integrate(sec(x)^8, x), x) - sec(x)^8",
            "diff(tan(x) + tan(x)^3 + 3*tan(x)^5/5 + tan(x)^7/7, x) - sec(x)^8",
            "diff(integrate(1/cos(2*x+1)^8, x), x) - 1/cos(2*x+1)^8",
            "diff(tan(2*x+1)/2 + tan(2*x+1)^3/2 + 3*tan(2*x+1)^5/10 + tan(2*x+1)^7/14, x) - 1/cos(2*x+1)^8",
            "diff(-tan(1-2*x)/2 - tan(1-2*x)^3/2 - 3*tan(1-2*x)^5/10 - tan(1-2*x)^7/14, x) - sec(1-2*x)^8",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_csc_eighth_without_global_detour() {
        let cases = [
            "diff(integrate(csc(x)^8, x), x) - csc(x)^8",
            "diff(-cot(x) - cot(x)^3 - 3*cot(x)^5/5 - cot(x)^7/7, x) - csc(x)^8",
            "diff(integrate(1/sin(2*x+1)^8, x), x) - 1/sin(2*x+1)^8",
            "diff(-cot(2*x+1)/2 - cot(2*x+1)^3/2 - 3*cot(2*x+1)^5/10 - cot(2*x+1)^7/14, x) - 1/sin(2*x+1)^8",
            "diff(cot(1-2*x)/2 + cot(1-2*x)^3/2 + 3*cot(1-2*x)^5/10 + cot(1-2*x)^7/14, x) - csc(1-2*x)^8",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_sixth_power_reduction() {
        let cases = [
            "diff(integrate(sin(x)^6, x), x) - sin(x)^6",
            "diff(integrate(cos(x)^6, x), x) - cos(x)^6",
            "diff(integrate(sin(2*x+1)^6, x), x) - sin(2*x+1)^6",
            "diff(integrate(cos(2*x+1)^6, x), x) - cos(2*x+1)^6",
            "diff(5*x/16 - 15*sin(2*x)/64 + 3*sin(4*x)/64 - sin(6*x)/192, x) - sin(x)^6",
            "diff(5*x/16 + 15*sin(2*x)/64 + 3*sin(4*x)/64 + sin(6*x)/192, x) - cos(x)^6",
            "diff(5*x/16 - 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 - sin(6*(2*x+1))/384, x) - sin(2*x+1)^6",
            "diff(5*x/16 + 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 + sin(6*(2*x+1))/384, x) - cos(2*x+1)^6",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(sin(2*x+1)^6, x), y) - sin(2*x+1)^6"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_eighth_power_reduction() {
        let cases = [
            "diff(integrate(sin(x)^8, x), x) - sin(x)^8",
            "diff(integrate(cos(x)^8, x), x) - cos(x)^8",
            "diff(integrate(sin(2*x+1)^8, x), x) - sin(2*x+1)^8",
            "diff(integrate(cos(2*x+1)^8, x), x) - cos(2*x+1)^8",
            "diff(35*x/128 - 7*sin(2*x)/32 + 7*sin(4*x)/128 - sin(6*x)/96 + sin(8*x)/1024, x) - sin(x)^8",
            "diff(35*x/128 + 7*sin(2*x)/32 + 7*sin(4*x)/128 + sin(6*x)/96 + sin(8*x)/1024, x) - cos(x)^8",
            "diff(35*x/128 - 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 - sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - sin(2*x+1)^8",
            "diff(35*x/128 + 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 + sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - cos(2*x+1)^8",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(sin(2*x+1)^8, x), y) - sin(2*x+1)^8"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_polynomial_by_parts() {
        let cases = [
            "diff(integrate(x*sin(x), x), x) - x*sin(x)",
            "diff(integrate(x*cos(x), x), x) - x*cos(x)",
            "diff(integrate(x^5*sin(x), x), x) - x^5*sin(x)",
            "diff(integrate(x^5*cos(x), x), x) - x^5*cos(x)",
            "diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)",
            "diff(integrate(x^6*cos(x), x), x) - x^6*cos(x)",
            "diff(integrate((2*x+3)*sin(2*x+1), x), x) - (2*x+3)*sin(2*x+1)",
            "diff(integrate((2*x+3)*cos(2*x+1), x), x) - (2*x+3)*cos(2*x+1)",
            "diff(integrate((x^3+x)*sin(2*x+1), x), x) - (x^3+x)*sin(2*x+1)",
            "diff(integrate((x^3+x)*cos(2*x+1), x), x) - (x^3+x)*cos(2*x+1)",
            "diff(integrate((x^5+x^2)*sin(2*x+1), x), x) - (x^5+x^2)*sin(2*x+1)",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(x^5*cos(x), y), y) - x^5*cos(x)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_cancels_polynomial_by_parts_wrappers() {
        let residual = "diff(integrate(x^5*cos(x), x), x) - x^5*cos(x)";
        let cases = [
            format!("({residual}) + 0"),
            format!("0 - ({residual})"),
            format!("2*({residual})"),
            format!("({residual})/(x+1)"),
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_compacts_constant_passthrough_quotient() {
        let residual = "diff(integrate((x^5+x^2)*sin(2*x+1), x), x) - (x^5+x^2)*sin(2*x+1)";
        let input = format!("(({residual}) + 1)/(x+2)");
        assert_eq!(
            integral_plain_trig_passthrough_quotient_result(&input),
            Some(("1 / (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "1 / (x + 2)");
    }

    #[test]
    fn diff_integral_inverse_trig_residual_root_verifies_polynomial_arctan_by_parts() {
        let cases = [
            "diff(integrate(x*arctan(x), x), x) - x*arctan(x)",
            "diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)",
        ];

        for input in cases {
            assert_eq!(
                integral_inverse_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_inverse_trig_root_residual_result(
                "diff(integrate(x^6*arctan(x), y), y) - x^6*arctan(x)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_inverse_trig_residual_root_cancels_polynomial_arctan_wrappers() {
        let residual = "diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)";
        let cases = [
            format!("({residual}) + 0"),
            format!("0 - ({residual})"),
            format!("2*({residual})"),
            format!("({residual})/(x+1)"),
        ];

        for input in cases {
            assert_eq!(
                integral_inverse_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_inverse_trig_residual_compacts_constant_passthrough_quotient() {
        let residual = "diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)";
        let input = format!("(({residual}) + 1)/(x+2)");
        assert_eq!(
            integral_inverse_trig_passthrough_quotient_result(&input),
            Some(("1 / (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "1 / (x + 2)");
    }

    #[test]
    fn diff_integral_quadratic_exp_residual_root_cancels_supported_target() {
        let input = "diff(integrate((x^2+x+1)*exp(2*x+1), x), x) - (x^2+x+1)*exp(2*x+1)";
        assert_eq!(
            integral_quadratic_exp_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        let quartic_input = "diff(integrate(x^4*exp(2*x+1), x), x) - x^4*exp(2*x+1)";
        assert_eq!(
            integral_quadratic_exp_root_residual_result(quartic_input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(quartic_input), "0");

        assert_eq!(
            integral_quadratic_exp_root_residual_result(
                "diff(integrate((x^2+x+1)*exp(2*x+1), y), y) - (x^2+x+1)*exp(2*x+1)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_exp_trig_residual_accepts_additive_constant_in_explicit_primitive() {
        let primitive = "1/13*exp(2*x)*(2*cos(3*x)+3*sin(3*x))";
        let target = "exp(2*x)*cos(3*x)";

        for input in [
            format!("diff({primitive} + 7, x) - {target}"),
            format!("diff({primitive} + C, x) - {target}"),
            format!("diff(7 + {primitive}, x) - {target}"),
        ] {
            assert_eq!(
                integral_exp_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }

        assert_eq!(
            integral_exp_trig_root_residual_result(&format!("diff(7 - {primitive}, x) - {target}")),
            None
        );
    }

    #[test]
    fn diff_integral_exp_trig_residual_compacts_negative_antiderivative_mismatch() {
        let primitive = "1/13*exp(2*x)*(2*cos(3*x)+3*sin(3*x))";
        let target = "exp(2*x)*cos(3*x)";

        assert_eq!(
            integral_exp_trig_compact_mismatch_result(&format!(
                "diff(7 - {primitive}, x) - {target}"
            )),
            Some("-2 * cos(3 * x) * e^(2 * x)".to_string())
        );
        assert_eq!(
            integral_exp_trig_compact_mismatch_result(&format!(
                "{target} - diff(7 - {primitive}, x)"
            )),
            Some("2 * cos(3 * x) * e^(2 * x)".to_string())
        );
        assert_eq!(
            integral_exp_trig_compact_mismatch_result(&format!(
                "diff({primitive} + 7, x) - {target}"
            )),
            None
        );
    }

    #[test]
    fn diff_integral_log_reciprocal_power_residual_root_cancels_shifted_quadratic() {
        let input = "diff(integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3)";
        assert_eq!(
            integral_quadratic_exp_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        let rendered_primitive =
            "diff(-1/(2*ln(x^2+x-1)^2), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3)";
        assert_eq!(
            integral_quadratic_exp_root_residual_result(rendered_primitive),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(rendered_primitive), "0");
    }

    #[test]
    fn diff_integral_hyperbolic_residual_root_cancels_supported_target() {
        let input = "diff(integrate((x^2+x)*sinh(2*x+1), x), x) - (x^2+x)*sinh(2*x+1)";
        assert_eq!(
            integral_hyperbolic_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        let high_degree_input =
            "diff(integrate((x^5+x^2)*sinh(2*x+1), x), x) - (x^5+x^2)*sinh(2*x+1)";
        assert_eq!(
            integral_hyperbolic_root_residual_result(high_degree_input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(high_degree_input), "0");

        assert_eq!(
            integral_hyperbolic_root_residual_result(
                "diff(integrate((x^2+x)*sinh(2*x+1), y), y) - (x^2+x)*sinh(2*x+1)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_hyperbolic_residual_root_cancels_negative_affine_linear_by_parts() {
        for input in [
            "diff(integrate((2*x+3)*sinh(1-2*x), x), x) - (2*x+3)*sinh(1-2*x)",
            "diff(integrate((2*x+3)*cosh(1-2*x), x), x) - (2*x+3)*cosh(1-2*x)",
        ] {
            assert_eq!(
                integral_hyperbolic_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_hyperbolic_residual_root_cancels_csch_square_without_global_detour() {
        let input = "diff(integrate(1/sinh(2*x+1)^2, x), x) - 1/sinh(2*x+1)^2";
        assert_eq!(
            integral_hyperbolic_root_residual_result(input),
            Some("0".to_string())
        );
    }

    #[test]
    fn diff_integral_hyperbolic_residual_compacts_constant_passthrough_quotient() {
        let residual = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        for input in [
            format!("(({residual}) + 1)/(x+2)"),
            format!("(1 - ({residual}))/(x+2)"),
            format!("2*((({residual}) + 1)/(x+2))"),
            format!("((1 - ({residual}))/(x+2))*2"),
            format!("((({residual}) + 1)/(x+2))/(x+3)"),
            format!("(({residual}) - 1)/(x+2)"),
            format!("((({residual}) + 1)/(x+2)) + (x-x)"),
            format!("((((({residual}) + 1)/(x+2)) + (x-x)) + (y-y))"),
            format!("(((((({residual}) + 1)/(x+2)) + (x-x)) + (y-y)) + (z-z))"),
            format!("(((((({residual}) + 1)/(x+2)) + (x-x)) + (y-y)) + (z-z))*1"),
        ] {
            let (expected, required_conditions) = if input.contains("x+3") {
                (
                    "1 / ((x + 2) * (x + 3))",
                    vec!["x ≠ -2".to_string(), "x ≠ -3".to_string()],
                )
            } else if input.contains("*2") || input.starts_with("2*") {
                ("2 / (x + 2)", vec!["x ≠ -2".to_string()])
            } else if input.contains("- 1") {
                ("-1 / (x + 2)", vec!["x ≠ -2".to_string()])
            } else {
                ("1 / (x + 2)", vec!["x ≠ -2".to_string()])
            };
            assert_eq!(
                integral_hyperbolic_passthrough_quotient_result(&input),
                Some((expected.to_string(), required_conditions)),
                "{input}"
            );
            assert_eq!(simplify_text(&input), expected, "{input}");
        }
    }

    #[test]
    fn diff_integral_hyperbolic_residual_compacts_deeper_passthrough_quotient_shortcut() {
        let residual = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let input = format!("((((({residual}) + 1)/(x+2))/(x+3))/(x+4))");
        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some((
                "1 / ((x + 2) * (x + 3) * (x + 4))".to_string(),
                vec![
                    "x ≠ -2".to_string(),
                    "x ≠ -3".to_string(),
                    "x ≠ -4".to_string(),
                ],
            ))
        );
        assert_eq!(simplify_text(&input), "1 / ((x + 2) * (x + 3) * (x + 4))");
    }

    #[test]
    fn diff_integral_residual_compacts_shifted_residual_denominator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("(({numerator}) + 1)/(({denominator}) + 1)");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("1".to_string(), vec![]))
        );
        assert_eq!(simplify_text(&input), "1");
    }

    #[test]
    fn diff_integral_residual_compacts_negative_shifted_residual_denominator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("(({numerator}) + 1)/(({denominator}) - 1)");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("-1".to_string(), vec![]))
        );
        assert_eq!(simplify_text(&input), "-1");
    }

    #[test]
    fn diff_integral_residual_compacts_scaled_shifted_residual_denominator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("(({numerator}) + 1)/(2*(({denominator}) + 1))");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("1/2".to_string(), vec![]))
        );
        assert_eq!(simplify_text(&input), "1/2");
    }

    #[test]
    fn diff_integral_residual_compacts_scaled_numerator_and_denominator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("3*(({numerator}) + 1)/(2*(({denominator}) + 1))");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("3/2".to_string(), vec![]))
        );
        assert_eq!(simplify_text(&input), "3/2");
    }

    #[test]
    fn diff_integral_residual_compacts_product_denominator_residual_factor() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("3*(({numerator}) + 1)/((({denominator}) + 1)*(x+2))");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("3 / (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "3 / (x + 2)");
    }

    #[test]
    fn diff_integral_residual_moves_negative_product_denominator_sign_to_numerator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("3*(({numerator}) + 1)/((x+2)*(({denominator}) - 1)*(x+3))");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some((
                "-3 / ((x + 2) * (x + 3))".to_string(),
                vec!["x ≠ -2".to_string(), "x ≠ -3".to_string()]
            ))
        );
        assert_eq!(simplify_text(&input), "-3 / ((x + 2) * (x + 3))");
    }

    #[test]
    fn diff_integral_residual_compacts_fraction_denominator_residual_numerator() {
        let numerator = "diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)";
        let denominator = "diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)";
        let input = format!("3*(({numerator}) + 1)/((({denominator}) - 1)/(x+2))");

        assert_eq!(
            integral_hyperbolic_passthrough_quotient_result(&input),
            Some(("-3 * (x + 2)".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(&input), "-3 * (x + 2)");
    }

    #[test]
    fn integral_hyperbolic_residual_cancels_reciprocal_shifted_difference() {
        let input = "1/((integrate(x^2*sinh(x),x))+c) - 1/((x^2*cosh(x)-2*x*sinh(x)+2*cosh(x))+c)";
        assert_eq!(
            integral_hyperbolic_reciprocal_shifted_result(input),
            Some((
                "0".to_string(),
                vec!["2 * cosh(x) + cosh(x) * x^2 - 2 * x * sinh(x) + c ≠ 0".to_string()]
            ))
        );
        assert_eq!(simplify_text(input), "0");

        assert_eq!(
            integral_hyperbolic_reciprocal_shifted_result(
                "1/((integrate(x^2*sinh(x),y))+c) - 1/((x^2*cosh(x)-2*x*sinh(x)+2*cosh(x))+c)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_residual_reciprocal_shifted_compacts_asinh_substitution_denominator() {
        let input =
            "1/((diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)) + x + 2) - 1/(x+2)";
        assert_eq!(
            integral_residual_reciprocal_shifted_result(input),
            Some(("0".to_string(), vec!["x ≠ -2".to_string()]))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_residual_reciprocal_shifted_compacts_reciprocal_trig_denominator() {
        let input = "1/((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2) - 1/(x+2)";
        assert_eq!(
            integral_residual_reciprocal_shifted_result(input),
            Some((
                "0".to_string(),
                vec!["sin(2 * x + 1) ≠ 0".to_string(), "x ≠ -2".to_string()]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_residual_shifted_quotient_compacts_asinh_substitution_denominator() {
        let residual = "diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)";
        for input in [
            format!("(({residual}) + x + 2)/(x+2) - 1"),
            format!("1 - (x+2)/(({residual}) + x + 2)"),
        ] {
            assert_eq!(
                integral_residual_shifted_quotient_result(&input),
                Some(("0".to_string(), vec!["x ≠ -2".to_string()])),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_residual_shifted_quotient_compacts_reciprocal_trig_denominator() {
        let residual = "diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)";
        for input in [
            format!("(({residual}) + x + 2)/(x+2) - 1"),
            format!("1 - (x+2)/(({residual}) + x + 2)"),
        ] {
            assert_eq!(
                integral_residual_shifted_quotient_result(&input),
                Some((
                    "0".to_string(),
                    vec!["sin(2 * x + 1) ≠ 0".to_string(), "x ≠ -2".to_string()]
                )),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_reciprocal_trig_residual_compacts_cross_family_shifted_denominator() {
        let numerator = "diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)";
        let simple_denominator = "diff(integrate(1/(x^2+1), x), x) - 1/(x^2+1)";
        let repeated_pole_denominator =
            "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";

        for (input, expected_conditions) in [
            (
                format!("(({numerator}) + 1)/(({simple_denominator}) + 1)"),
                vec!["sin(2 * x + 1) ≠ 0".to_string()],
            ),
            (
                format!("(({numerator}) + 1)/(({repeated_pole_denominator}) + 1)"),
                vec!["sin(2 * x + 1) ≠ 0", "x ≠ -1"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            ),
        ] {
            assert_eq!(
                integral_reciprocal_trig_passthrough_quotient_result(&input),
                Some(("1".to_string(), expected_conditions)),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "1", "{input}");
        }
    }

    #[test]
    fn diff_integral_reciprocal_trig_log_sqrt_chain_passthrough_quotient_preserves_denominator() {
        let residual = "diff(integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))), x), x) - 3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1)))";
        let input = format!("(({residual}) + 1)/(x+2)");

        assert_eq!(
            integral_reciprocal_trig_passthrough_quotient_result(&input),
            Some((
                "1 / (x + 2)".to_string(),
                vec![
                    "cos(sqrt(3 * x + 1)) ≠ 0".to_string(),
                    "x > -1/3".to_string(),
                    "x ≠ -2".to_string(),
                ]
            )),
            "{input}"
        );
        assert_eq!(simplify_text(&input), "1 / (x + 2)", "{input}");
    }

    #[test]
    fn diff_integral_residual_product_zero_factor_compacts_passthrough_wrapper() {
        let residual = "diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)";
        for input in [
            format!("(({residual}) + x + 2)*(y-y)"),
            format!("(y-y)*(({residual}) + x + 2)"),
        ] {
            assert_eq!(
                integral_residual_product_zero_factor_result(&input),
                Some(("0".to_string(), vec!["sin(2 * x + 1) ≠ 0".to_string()])),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_residual_product_zero_factor_rejects_domain_sensitive_passthrough() {
        let residual = "diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)";
        let input = format!("(({residual}) + 1/z)*(y-y)");
        assert_eq!(integral_residual_product_zero_factor_result(&input), None);
    }

    #[test]
    fn shifted_integral_residual_condition_filter_handles_product_denominator_factor() {
        let mut ctx = Context::new();
        let input = "(diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1) + 1)*(x+2)";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let conditions =
            shifted_integral_residual_passthrough_nonzero_required_conditions(&mut ctx, expr)
                .unwrap_or_else(|| panic!("expected product denominator condition to compact"));
        let rendered: Vec<String> = conditions
            .into_iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();

        assert_eq!(rendered, vec!["x ≠ -2"]);
    }

    #[test]
    fn shifted_integral_residual_condition_filter_handles_fraction_denominator_numerator() {
        let mut ctx = Context::new();
        let input = "(diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1) - 1)/(x+2)";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let conditions =
            shifted_integral_residual_passthrough_nonzero_required_conditions(&mut ctx, expr)
                .unwrap_or_else(|| panic!("expected fractional denominator condition to compact"));
        let rendered: Vec<String> = conditions
            .into_iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();

        assert_eq!(rendered, vec!["x ≠ -2"]);
    }

    #[test]
    fn shifted_integral_residual_condition_filter_compacts_additive_passthrough_denominator() {
        let mut ctx = Context::new();
        let input = "diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2 + x + 1";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let conditions =
            shifted_integral_residual_passthrough_nonzero_required_conditions(&mut ctx, expr)
                .unwrap_or_else(|| {
                    panic!("expected additive passthrough denominator condition to compact")
                });
        let rendered: Vec<String> = conditions
            .into_iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();

        assert_eq!(rendered, vec!["x ≠ -1"]);
    }

    #[test]
    fn shifted_integral_residual_condition_filter_compacts_arctan_sqrt_reciprocal_passthrough() {
        let mut ctx = Context::new();
        let input = "diff(integrate(1/(sqrt(x)*(x+1)), x), x) - 1/(sqrt(x)*(x+1)) + x + 2";
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let conditions =
            shifted_integral_residual_passthrough_nonzero_required_conditions(&mut ctx, expr)
                .unwrap_or_else(|| {
                    panic!("expected arctan sqrt reciprocal passthrough condition to compact")
                });
        let rendered: Vec<String> = conditions
            .into_iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();

        assert_eq!(rendered, vec!["x > 0", "x ≠ -2"]);
    }

    #[test]
    fn diff_integral_rational_residual_root_cancels_shifted_quadratic_repeated_pole() {
        let input = "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";
        assert_eq!(
            integral_rational_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        assert_eq!(
            integral_rational_root_residual_result(
                "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), y), y) - 1/((x+1)^3*(x^2+2*x+2))"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_rational_residual_root_cancels_linear_partial_fraction() {
        let input =
            "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x) - 1/((x-2)*(x-1)*x*(x+1)*(x+2))";
        assert_eq!(
            integral_rational_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        assert_eq!(
            integral_rational_root_residual_result(
                "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), y), y) - 1/((x-2)*(x-1)*x*(x+1)*(x+2))"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_rational_residual_compacts_constant_passthrough_quotient() {
        let residual = "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";
        for input in [
            format!("(({residual}) + 1)/(x+2)"),
            format!("(1 - ({residual}))/(x+2)"),
        ] {
            assert_eq!(
                integral_rational_passthrough_quotient_result(&input),
                Some((
                    "1 / (x + 2)".to_string(),
                    vec!["x ≠ -1".to_string(), "x ≠ -2".to_string()]
                )),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "1 / (x + 2)", "{input}");
        }
    }

    #[test]
    fn diff_integral_rational_residual_compacts_positive_quadratic_double_nested_quotient() {
        let residual = "diff(integrate((3*x + 5)/(x^2+x+1), x), x) - ((3*x + 5)/(x^2+x+1))";
        let input = format!("((((({residual}) + 1)/(x+2))/(x+3))/(x+4))");
        assert_eq!(
            integral_rational_passthrough_quotient_result(&input),
            Some((
                "1 / ((x + 2) * (x + 3) * (x + 4))".to_string(),
                vec![
                    "x ≠ -2".to_string(),
                    "x ≠ -3".to_string(),
                    "x ≠ -4".to_string(),
                ],
            ))
        );
        assert_eq!(simplify_text(&input), "1 / ((x + 2) * (x + 3) * (x + 4))");
    }

    #[test]
    fn diff_integral_rational_residual_compacts_cross_family_shifted_denominator() {
        let numerator = "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";
        let denominator = "diff(integrate(1/(x^2+1), x), x) - 1/(x^2+1)";
        let input = format!("(({numerator}) + 1)/(({denominator}) + 1)");

        assert_eq!(
            integral_rational_passthrough_quotient_result(&input),
            Some(("1".to_string(), vec!["x ≠ -1".to_string()]))
        );
        assert_eq!(simplify_text(&input), "1");
    }

    #[test]
    fn diff_integral_rational_residual_cancels_reciprocal_shifted_difference() {
        let input =
            "1/((diff(integrate((3*x+5)/(x^3-x^2-x+1),x),x))+c) - 1/(((3*x+5)/(x^3-x^2-x+1))+c)";
        assert_eq!(
            integral_rational_reciprocal_shifted_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x ≠ 1".to_string(),
                    "x ≠ -1".to_string(),
                    "(3 * x + 5) / (x^3 - x^2 - x + 1) + c ≠ 0".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");

        assert_eq!(
            integral_rational_reciprocal_shifted_result(
                "1/((diff(integrate((3*x+5)/(x^3-x^2-x+1),x),y))+c) - 1/(((3*x+5)/(x^3-x^2-x+1))+c)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_root_product_residual_cancels_reciprocal_shifted_difference() {
        let input =
            "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)-1/(x+2)";
        assert_eq!(
            integral_residual_reciprocal_shifted_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_root_product_residual_cancels_reciprocal_shifted_difference_with_additive_noise(
    ) {
        let input =
            "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)-1/(x+2)+y-y";
        assert_eq!(
            integral_residual_reciprocal_shifted_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_root_product_residual_cancels_reciprocal_shifted_quotient_with_additive_noise()
    {
        let input =
            "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)/(x+2))+y-(1+y)";
        assert_eq!(
            integral_residual_shifted_quotient_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_root_product_residual_compacts_constant_passthrough_quotient() {
        let input =
            "((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)/(x+2)";
        assert_eq!(
            integral_residual_passthrough_quotient_result(input),
            Some((
                "1".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "1");
    }

    #[test]
    fn diff_integral_root_product_residual_cancels_reciprocal_shifted_quotient() {
        let input =
            "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)/(x+2))-1";
        assert_eq!(
            integral_residual_shifted_quotient_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_integral_root_product_residual_cancels_squared_shifted_quotient() {
        let input =
            "(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6)))+x+2)/(x+2))^2-1";
        assert_eq!(
            integral_residual_shifted_quotient_result(input),
            Some((
                "0".to_string(),
                vec![
                    "x < -3 or x > 0".to_string(),
                    "x > 0".to_string(),
                    "x ≠ -2".to_string()
                ]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_one_root_cancels_supported_pair() {
        let input = "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (result, required_conditions) =
            require_match(diff_arctan_sqrt_positive_quotient_shifted_one_result(input));
        assert_eq!(result, "1");
        assert_eq!(
            required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 1 ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), y))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_one_root_cancels_negative_orientation_pair() {
        let input = "(1 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (result, required_conditions) =
            require_match(diff_arctan_sqrt_positive_quotient_shifted_one_result(input));
        assert_eq!(result, "1");
        assert_eq!(
            required_conditions,
            vec![
                "1 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_constant_root_cancels_matching_wrappers() {
        let plus_input = "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (plus_result, plus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_one_result(plus_input),
        );
        assert_eq!(plus_result, "1");
        assert_eq!(
            plus_required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 2 ≠ 0"
                    .to_string()
            ]
        );

        let minus_input = "(2 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_one_result(minus_input),
        );
        assert_eq!(minus_result, "1");
        assert_eq!(
            minus_required_conditions,
            vec![
                "2 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(3 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_mismatch_compacts_diff_side() {
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(x)), x))/(2 + 1/(2*sqrt(x)*(x+1)))"),
            "(1 / (2 * sqrt(x) * (x + 1)) + 1) / (1 / (2 * sqrt(x) * (x + 1)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 - diff(arctan(sqrt(x)), x))/(2 - 1/(2*sqrt(x)*(x+1)))"),
            "(1 - 1 / (2 * sqrt(x) * (x + 1))) / (2 - 1 / (2 * sqrt(x) * (x + 1)))"
        );
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(2*x+3)), x))/(2 + 1/(sqrt(2*x+3)*(2*x+4)))"),
            "(1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 1) / (1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 - diff(arctan(sqrt(2*x+3)), x))/(2 - 1/(sqrt(2*x+3)*(2*x+4)))"),
            "(1 - 1 / (sqrt(2 * x + 3) * (2 * x + 4))) / (2 - 1 / (sqrt(2 * x + 3) * (2 * x + 4)))"
        );
        assert_eq!(
            simplify_text(
                "(1 + diff(arctan(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))"
        );
        assert_eq!(
            simplify_text(
                "(1 - diff(arctan(sqrt(5-3*x)), x))/(2 + 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1) / (3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(5-3*x)), x))/(1 - 3/(2*sqrt(5-3*x)*(6-3*x)))"),
            "1"
        );
        assert_eq!(
            simplify_text(
                "(1 + diff(arccot(sqrt(5-3*x)), x))/(2 + 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1) / (3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2)"
        );
        assert_eq!(
            simplify_text(
                "(1 - diff(arccot(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))"
        );
        assert_eq!(
            simplify_text("(1 + diff(arccot(sqrt(5-3*x)), x))/(1 + 3/(2*sqrt(5-3*x)*(6-3*x)))"),
            "1"
        );

        let plus_input = "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (plus_result, plus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(plus_input),
        );
        assert_ne!(plus_result, "1");
        assert!(!plus_result.contains("diff"), "{plus_result}");
        assert!(plus_result.contains("+ 1"), "{plus_result}");
        assert!(plus_result.contains("+ 2"), "{plus_result}");
        assert_eq!(
            plus_required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 2 ≠ 0"
                    .to_string()
            ]
        );

        let minus_input = "(1 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(minus_input),
        );
        assert_ne!(minus_result, "1");
        assert!(!minus_result.contains("diff"), "{minus_result}");
        assert!(minus_result.contains("1 -"), "{minus_result}");
        assert!(minus_result.contains("2 -"), "{minus_result}");
        assert_eq!(
            minus_required_conditions,
            vec![
                "2 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
        assert!(diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_unit_interval_bounded_inverse_trig_shifted_quotient_compacts_contextual_diff() {
        assert_eq!(
            simplify_text("(1 + diff(1/2*arcsin(2*x-1), x))/(1 + 1/(2*sqrt(x)*sqrt(1-x)))"),
            "1"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arccos(2*x-1), x))/(1 - 1/(2*sqrt(x)*sqrt(1-x)))"),
            "1"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arcsin(2*x-1), x))/(2 + 1/(2*sqrt(x)*sqrt(1-x)))"),
            "(1 / (2 * sqrt(x) * sqrt(1 - x)) + 1) / (1 / (2 * sqrt(x) * sqrt(1 - x)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arccos(2*x-1), x))/(2 - 1/(2*sqrt(x)*sqrt(1-x)))"),
            "(1 - 1 / (2 * sqrt(x) * sqrt(1 - x))) / (2 - 1 / (2 * sqrt(x) * sqrt(1 - x)))"
        );
    }

    #[test]
    fn diff_inverse_reciprocal_trig_shifted_quotient_root_cancels_matching_wrappers() {
        let minus_input = "(1 - diff(arcsec(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_result(minus_input),
        );
        assert_eq!(minus_result, "1");
        assert_eq!(
            minus_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 - 1 / (2 * sqrt(x) * (x + 1)) ≠ 0".to_string()
            ]
        );

        let arccsc_input = "(1 - diff(arccsc(sqrt(x+1)), x))/(1 + 1/(2*(x+1)*sqrt(x)))";
        let (arccsc_result, arccsc_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_result(arccsc_input),
        );
        assert_eq!(arccsc_result, "1");
        assert_eq!(
            arccsc_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 / (2 * sqrt(x) * (x + 1)) + 1 ≠ 0".to_string()
            ]
        );

        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arcsec(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 + diff(arcsec(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arccsc(sqrt(x+1)), x))/(2 + 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arccsc(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
    }

    #[test]
    fn diff_inverse_reciprocal_trig_shifted_quotient_mismatch_compacts_diff_side() {
        let arccsc_input = "(1 - diff(arccsc(sqrt(x+1)), x))/(2 + 1/(2*(x+1)*sqrt(x)))";
        let (arccsc_result, arccsc_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(arccsc_input),
        );
        assert_ne!(arccsc_result, "1");
        assert!(!arccsc_result.contains("diff"), "{arccsc_result}");
        assert!(arccsc_result.contains("+ 1"), "{arccsc_result}");
        assert!(arccsc_result.contains("+ 2"), "{arccsc_result}");
        assert_eq!(
            arccsc_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 / (2 * sqrt(x) * (x + 1)) + 2 ≠ 0".to_string()
            ]
        );

        let arcsec_input = "(1 - diff(arcsec(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))";
        let (arcsec_result, arcsec_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(arcsec_input),
        );
        assert_ne!(arcsec_result, "1");
        assert!(!arcsec_result.contains("diff"), "{arcsec_result}");
        assert!(arcsec_result.contains("1 -"), "{arcsec_result}");
        assert!(arcsec_result.contains("2 -"), "{arcsec_result}");
        assert_eq!(
            arcsec_required_conditions,
            vec![
                "x > 0".to_string(),
                "2 - 1 / (2 * sqrt(x) * (x + 1)) ≠ 0".to_string()
            ]
        );

        let arcsec_negative_affine_input =
            "(1 + diff(arcsec(sqrt(5-3*x)), x))/(2 - 3/(2*(5-3*x)*sqrt(4-3*x)))";
        let (arcsec_negative_affine_result, arcsec_negative_affine_required_conditions) =
            require_match(
                diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                    arcsec_negative_affine_input,
                ),
            );
        assert_ne!(arcsec_negative_affine_result, "1");
        assert!(!arcsec_negative_affine_result.contains("diff"));
        assert_eq!(
            arcsec_negative_affine_result,
            "(1 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x))) / (2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)))"
        );
        assert_eq!(
            arcsec_negative_affine_required_conditions,
            vec![
                "x < 4/3".to_string(),
                "2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) ≠ 0".to_string()
            ]
        );

        let arccsc_negative_affine_input =
            "(1 + diff(arccsc(sqrt(5-3*x)), x))/(2 + 3/(2*(5-3*x)*sqrt(4-3*x)))";
        let (arccsc_negative_affine_result, arccsc_negative_affine_required_conditions) =
            require_match(
                diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                    arccsc_negative_affine_input,
                ),
            );
        assert_ne!(arccsc_negative_affine_result, "1");
        assert!(!arccsc_negative_affine_result.contains("diff"));
        assert_eq!(
            arccsc_negative_affine_result,
            "(3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 1) / (3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 2)"
        );
        assert_eq!(
            arccsc_negative_affine_required_conditions,
            vec![
                "x < 4/3".to_string(),
                "3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 2 ≠ 0".to_string()
            ]
        );

        assert!(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                "(1 - diff(arccsc(sqrt(x+1)), x))/(1 + 1/(2*(x+1)*sqrt(x)))"
            )
            .is_none()
        );
        assert!(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                "(1 - diff(arccsc(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))"
            )
            .is_none()
        );
    }

    #[test]
    fn diff_hyperbolic_reciprocal_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(-1/(2*cosh(2*x+1)), x) - sinh(2*x+1)/cosh(2*x+1)^2",
            "diff(-1/(2*sinh(2*x+1)), x) - cosh(2*x+1)/sinh(2*x+1)^2",
            "sinh(2*x+1)/cosh(2*x+1)^2 - diff(-1/(2*cosh(2*x+1)), x)",
            "cosh(2*x+1)/sinh(2*x+1)^2 - diff(-1/(2*sinh(2*x+1)), x)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_sqrt_acosh_split_radical_residual_cancels_affine_chain() {
        let input = "diff(sqrt(acosh(2*x+3)), x) - 1/(sqrt(2*x+2)*sqrt(2*x+4)*sqrt(acosh(2*x+3)))";
        assert_eq!(
            sqrt_acosh_split_radical_residual_result(input),
            Some((
                "0".to_string(),
                vec!["acosh(2 * x + 3) > 0".to_string(), "x > -1".to_string()]
            ))
        );
        assert_eq!(simplify_text(input), "0");
    }

    #[test]
    fn diff_acosh_affine_fused_reciprocal_sqrt_residual_cancels_before_expanding_diff() {
        let cases = [
            (
                "diff(acosh(2*x-1), x) - 2/sqrt((2*x-1)^2-1)",
                vec!["x > 1".to_string()],
            ),
            (
                "diff(acosh(1-2*x), x) - (-2/sqrt((2*x-1)^2-1))",
                vec!["x < 0".to_string()],
            ),
            (
                "diff(acosh(2*x-1), x) - 2*sqrt((2*x-1)^2-1)/((2*x-1)^2-1)",
                vec!["x > 1".to_string()],
            ),
            (
                "diff(acosh(1-2*x), x) + 2*sqrt((2*x-1)^2-1)/((2*x-1)^2-1)",
                vec!["x < 0".to_string()],
            ),
            (
                "diff(acosh(x^2+x+3), x) - (2*x+1)*sqrt((x^2+x+3)^2-1)/((x^2+x+3)^2-1)",
                vec![],
            ),
        ];

        for (input, expected_conditions) in cases {
            assert_eq!(
                acosh_affine_reciprocal_sqrt_residual_result(input),
                Some(("0".to_string(), expected_conditions)),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_ln_sqrt_polynomial_gap_residual_cancels_affine_branches() {
        let cases = [
            (
                "diff(ln(sqrt((2*x+1)^2-4)+(2*x+1)), x) - 2/sqrt((2*x+1)^2-4)",
                "x > 1/2",
            ),
            (
                "diff(ln(sqrt((2*x+1)^2-4)-(2*x+1)), x) + 2/sqrt((2*x+1)^2-4)",
                "x < -3/2",
            ),
            (
                "diff(ln(sqrt((2*x+1)^2-4)-(2*x+1)), x) + 2*(4*x^2+4*x-3)^(-1/2)",
                "x < -3/2",
            ),
            (
                "diff(ln(sqrt((x^2+x+1)^2-4)+(x^2+x+1)), x) - (2*x+1)/sqrt((x^2+x+1)^2-4)",
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2",
            ),
            (
                "diff(ln(sqrt((x^2+x+1)^2-4)+(x^2+x+1)), x) - (x^4+2*x*x^2+x^2+2*x^2+2*x+1-4)^(-1/2)*(2*x+1)",
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2",
            ),
        ];

        for (input, expected_condition) in cases {
            assert_eq!(
                ln_sqrt_polynomial_gap_residual_result(input),
                Some(("0".to_string(), vec![expected_condition.to_string()])),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }
}
