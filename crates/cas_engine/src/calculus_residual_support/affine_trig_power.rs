//! Residual verification for integrated affine trig power families
//! (tan/cot/sec/csc fourth/sixth/eighth and sin/cos odd/even powers):
//! bounded diff-matching of explicit primitives without broad simplification.

use super::*;

fn affine_trig_power_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    expected_power: i64,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let Expr::Pow(base, exp) = ctx.get(target).clone() else {
        return None;
    };
    let Expr::Number(power) = ctx.get(exp) else {
        return None;
    };
    if *power != BigRational::from_integer(expected_power.into()) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
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

    Some((builtin, args[0], slope))
}

fn companion_builtin_for_odd_power_primitive(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Sin => Some(BuiltinFn::Cos),
        BuiltinFn::Cos => Some(BuiltinFn::Sin),
        _ => None,
    }
}

fn companion_power_factor(
    ctx: &mut Context,
    expr: ExprId,
    companion_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<u32> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Some(companion_arg) = unary_builtin_arg(ctx, expr, companion_builtin) {
        return exprs_match(ctx, companion_arg, arg).then_some(1);
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let companion_arg = unary_builtin_arg(ctx, base, companion_builtin)?;
    if !exprs_match(ctx, companion_arg, arg) {
        return None;
    }
    let Expr::Number(power) = ctx.get(exp) else {
        return None;
    };
    if *power == BigRational::from_integer(3.into()) {
        Some(3)
    } else if *power == BigRational::from_integer(5.into()) {
        Some(5)
    } else if *power == BigRational::from_integer(7.into()) {
        Some(7)
    } else {
        None
    }
}

fn collect_companion_power_terms(
    ctx: &mut Context,
    expr: ExprId,
    companion_builtin: BuiltinFn,
    arg: ExprId,
    scale: BigRational,
    terms: &mut Vec<(u32, BigRational)>,
) -> Option<()> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_companion_power_terms(ctx, left, companion_builtin, arg, scale.clone(), terms)?;
            collect_companion_power_terms(ctx, right, companion_builtin, arg, scale, terms)
        }
        Expr::Sub(left, right) => {
            collect_companion_power_terms(ctx, left, companion_builtin, arg, scale.clone(), terms)?;
            collect_companion_power_terms(ctx, right, companion_builtin, arg, -scale, terms)
        }
        Expr::Neg(inner) => {
            collect_companion_power_terms(ctx, inner, companion_builtin, arg, -scale, terms)
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, den)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_companion_power_terms(
                ctx,
                num,
                companion_builtin,
                arg,
                scale / den_scale,
                terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_companion_power_terms(
                ctx,
                non_numeric?,
                companion_builtin,
                arg,
                term_scale,
                terms,
            )
        }
        _ => {
            let power = companion_power_factor(ctx, expr, companion_builtin, arg)?;
            terms.push((power, scale));
            Some(())
        }
    }
}

fn companion_power_coeff(terms: &[(u32, BigRational)], power: u32) -> BigRational {
    terms
        .iter()
        .filter_map(|(term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

fn odd_power_primitive_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    right: ExprId,
    expected_power: i64,
) -> Option<bool> {
    let (builtin, arg, slope) = affine_trig_power_target(ctx, right, var_name, expected_power)?;
    let companion_builtin = companion_builtin_for_odd_power_primitive(builtin)?;
    let mut terms = Vec::new();
    collect_companion_power_terms(
        ctx,
        target,
        companion_builtin,
        arg,
        BigRational::one(),
        &mut terms,
    )?;

    let expected_terms: &[(u32, i64, i64)] = match (builtin, expected_power) {
        (BuiltinFn::Sin, 5) => &[(1, -1, 1), (3, 2, 3), (5, -1, 5)],
        (BuiltinFn::Cos, 5) => &[(1, 1, 1), (3, -2, 3), (5, 1, 5)],
        (BuiltinFn::Sin, 7) => &[(1, -1, 1), (3, 1, 1), (5, -3, 5), (7, 1, 7)],
        (BuiltinFn::Cos, 7) => &[(1, 1, 1), (3, -1, 1), (5, 3, 5), (7, -1, 7)],
        _ => return None,
    };

    if terms.iter().any(|(power, coeff)| {
        !expected_terms
            .iter()
            .any(|(expected, _, _)| *expected == *power)
            && !coeff.is_zero()
    }) {
        return None;
    }

    let expected = |numerator: i64, denominator: i64| {
        BigRational::new(numerator.into(), denominator.into()) / slope.clone()
    };

    Some(
        expected_terms
            .iter()
            .all(|(power, numerator, denominator)| {
                companion_power_coeff(&terms, *power) == expected(*numerator, *denominator)
            }),
    )
}

fn integrated_affine_trig_odd_power_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
    expected_power: i64,
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
        affine_trig_power_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
            expected_power,
        )?;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(Vec::new());
    }

    odd_power_primitive_target_diff_matches(
        ctx,
        diff_call.target,
        &diff_call.var_name,
        right,
        expected_power,
    )
    .filter(|matched| *matched)
    .map(|_| Vec::new())
}

pub(super) fn integrated_affine_trig_fifth_power_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_affine_trig_odd_power_diff_matches(ctx, diff_expr, divisor, right, 5)
}

pub(super) fn integrated_affine_trig_seventh_power_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_affine_trig_odd_power_diff_matches(ctx, diff_expr, divisor, right, 7)
}

struct EvenPowerPrimitiveTerms {
    linear_slope: BigRational,
    sin_coeffs: Vec<BigRational>,
}

impl EvenPowerPrimitiveTerms {
    fn new(harmonic_count: usize) -> Self {
        Self {
            linear_slope: BigRational::zero(),
            sin_coeffs: vec![BigRational::zero(); harmonic_count],
        }
    }
}

fn collect_even_power_primitive_terms(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
    var_name: &str,
    harmonics: &[i64],
    scale: BigRational,
    terms: &mut EvenPowerPrimitiveTerms,
) -> Option<()> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_even_power_primitive_terms(
                ctx,
                left,
                arg,
                var_name,
                harmonics,
                scale.clone(),
                terms,
            )?;
            collect_even_power_primitive_terms(ctx, right, arg, var_name, harmonics, scale, terms)
        }
        Expr::Sub(left, right) => {
            collect_even_power_primitive_terms(
                ctx,
                left,
                arg,
                var_name,
                harmonics,
                scale.clone(),
                terms,
            )?;
            collect_even_power_primitive_terms(ctx, right, arg, var_name, harmonics, -scale, terms)
        }
        Expr::Neg(inner) => {
            collect_even_power_primitive_terms(ctx, inner, arg, var_name, harmonics, -scale, terms)
        }
        _ => {
            let (term_scale, core) = rational_scaled_term(ctx, expr);
            let term_scale = scale * term_scale;

            if let Ok(poly) = Polynomial::from_expr(ctx, core, var_name) {
                if poly.degree() <= 1 {
                    let slope = poly
                        .coeffs
                        .get(1)
                        .cloned()
                        .unwrap_or_else(BigRational::zero);
                    terms.linear_slope += term_scale * slope;
                    return Some(());
                }
            }

            let (sin_scale, sin_arg) =
                scaled_unary_builtin_rational_target(ctx, core, BuiltinFn::Sin)?;
            for (index, harmonic) in harmonics.iter().enumerate() {
                let harmonic_arg =
                    scale_expr_by_rational(ctx, arg, BigRational::from_integer((*harmonic).into()));
                if exprs_match(ctx, sin_arg, harmonic_arg) {
                    terms.sin_coeffs[index] += term_scale * sin_scale;
                    return Some(());
                }
            }

            None
        }
    }
}

fn expected_harmonic_coeffs(
    builtin: BuiltinFn,
    sin_coeffs: &'static [(i64, i64, i64)],
    cos_coeffs: &'static [(i64, i64, i64)],
) -> Option<&'static [(i64, i64, i64)]> {
    match builtin {
        BuiltinFn::Sin => Some(sin_coeffs),
        BuiltinFn::Cos => Some(cos_coeffs),
        _ => None,
    }
}

struct EvenPowerPrimitiveSpec {
    expected_power: i64,
    expected_linear: BigRational,
    sin_coeffs: &'static [(i64, i64, i64)],
    cos_coeffs: &'static [(i64, i64, i64)],
}

fn even_power_primitive_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    right: ExprId,
    spec: EvenPowerPrimitiveSpec,
) -> Option<bool> {
    let (builtin, arg, slope) =
        affine_trig_power_target(ctx, right, var_name, spec.expected_power)?;
    let harmonic_coeffs = expected_harmonic_coeffs(builtin, spec.sin_coeffs, spec.cos_coeffs)?;
    let harmonics: Vec<i64> = harmonic_coeffs
        .iter()
        .map(|(harmonic, _, _)| *harmonic)
        .collect();
    let mut terms = EvenPowerPrimitiveTerms::new(harmonics.len());
    collect_even_power_primitive_terms(
        ctx,
        target,
        arg,
        var_name,
        &harmonics,
        BigRational::one(),
        &mut terms,
    )?;

    if terms.linear_slope != spec.expected_linear {
        return Some(false);
    }

    for ((_, numerator, denominator), actual) in harmonic_coeffs.iter().zip(&terms.sin_coeffs) {
        let expected = BigRational::new((*numerator).into(), (*denominator).into()) / slope.clone();
        if *actual != expected {
            return Some(false);
        }
    }

    Some(true)
}

fn fourth_power_primitive_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    right: ExprId,
) -> Option<bool> {
    even_power_primitive_target_diff_matches(
        ctx,
        target,
        var_name,
        right,
        EvenPowerPrimitiveSpec {
            expected_power: 4,
            expected_linear: BigRational::new(3.into(), 8.into()),
            sin_coeffs: &[(2, -1, 4), (4, 1, 32)],
            cos_coeffs: &[(2, 1, 4), (4, 1, 32)],
        },
    )
}

pub(super) fn integrated_affine_trig_fourth_power_diff_matches(
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
        affine_trig_power_target(ctx, integrate_call.target, &integrate_call.var_name, 4)?;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(Vec::new());
    }

    fourth_power_primitive_target_diff_matches(ctx, diff_call.target, &diff_call.var_name, right)
        .filter(|matched| *matched)
        .map(|_| Vec::new())
}

fn sixth_power_primitive_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    right: ExprId,
) -> Option<bool> {
    even_power_primitive_target_diff_matches(
        ctx,
        target,
        var_name,
        right,
        EvenPowerPrimitiveSpec {
            expected_power: 6,
            expected_linear: BigRational::new(5.into(), 16.into()),
            sin_coeffs: &[(2, -15, 64), (4, 3, 64), (6, -1, 192)],
            cos_coeffs: &[(2, 15, 64), (4, 3, 64), (6, 1, 192)],
        },
    )
}

fn eighth_power_primitive_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    right: ExprId,
) -> Option<bool> {
    even_power_primitive_target_diff_matches(
        ctx,
        target,
        var_name,
        right,
        EvenPowerPrimitiveSpec {
            expected_power: 8,
            expected_linear: BigRational::new(35.into(), 128.into()),
            sin_coeffs: &[(2, -7, 32), (4, 7, 128), (6, -1, 96), (8, 1, 1024)],
            cos_coeffs: &[(2, 7, 32), (4, 7, 128), (6, 1, 96), (8, 1, 1024)],
        },
    )
}

pub(super) fn integrated_affine_trig_sixth_power_diff_matches(
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
        affine_trig_power_target(ctx, integrate_call.target, &integrate_call.var_name, 6)?;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(Vec::new());
    }

    sixth_power_primitive_target_diff_matches(ctx, diff_call.target, &diff_call.var_name, right)
        .filter(|matched| *matched)
        .map(|_| Vec::new())
}

pub(super) fn integrated_affine_trig_eighth_power_diff_matches(
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
        affine_trig_power_target(ctx, integrate_call.target, &integrate_call.var_name, 8)?;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(Vec::new());
    }

    eighth_power_primitive_target_diff_matches(ctx, diff_call.target, &diff_call.var_name, right)
        .filter(|matched| *matched)
        .map(|_| Vec::new())
}

fn integrated_or_explicit_trig_ratio_power_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
    is_supported_target: fn(&mut Context, ExprId, &str) -> bool,
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
        if !is_supported_target(ctx, integrate_call.target, &integrate_call.var_name) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !is_supported_target(ctx, right, &diff_call.var_name) {
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_tan_fourth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_fourth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_cot_fourth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_fourth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_tan_sixth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_sixth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_cot_sixth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_sixth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_tan_eighth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_eighth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_cot_eighth_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    integrated_or_explicit_trig_ratio_power_diff_matches(
        ctx,
        diff_expr,
        divisor,
        right,
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_eighth_affine_target,
    )
}

pub(super) fn integrated_or_explicit_sec_fourth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_fourth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_fourth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_csc_fourth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_fourth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_fourth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_sec_sixth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_sixth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_sixth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_csc_sixth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_sixth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_sixth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_sec_eighth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_eighth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_sec_eighth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}

pub(super) fn integrated_or_explicit_csc_eighth_diff_matches(
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
        if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_eighth_affine_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return None;
        }
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
        return Some(integral_required_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ));
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_csc_eighth_affine_target(
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

    Some(integral_required_conditions(
        ctx,
        right,
        &diff_call.var_name,
    ))
}
