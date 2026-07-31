//! `focused_rule_substeps`: familia `trigonometric`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn phase_shift_shifted_trig_formula_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let before_trig = single_sin_cos_function_name(ctx, before)?;
    let after_trig = single_sin_cos_function_name(ctx, after)?;

    match (before_trig, after_trig) {
        ("sin", "cos") => Some(schema_substep(
            "Usar sin(u + φ) = cos(u - (π/2 - φ))",
            "sin(u + φ)",
            "cos(u - (π/2 - φ))",
            "\\sin(u+\\varphi)",
            "\\cos(u-(\\pi/2-\\varphi))",
        )),
        ("cos", "sin") => Some(schema_substep(
            "Usar cos(u - φ) = sin(u + (π/2 - φ))",
            "cos(u - φ)",
            "sin(u + (π/2 - φ))",
            "\\cos(u-\\varphi)",
            "\\sin(u+(\\pi/2-\\varphi))",
        )),
        _ => None,
    }
}

fn single_sin_cos_function_name(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let mut found = None;
    if collect_single_sin_cos_function_name(ctx, expr, &mut found) {
        found
    } else {
        None
    }
}

fn collect_single_sin_cos_function_name(
    ctx: &Context,
    expr: ExprId,
    found: &mut Option<&'static str>,
) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
                return record_single_sin_cos_name(found, "sin");
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
                return record_single_sin_cos_name(found, "cos");
            }
            args.iter()
                .copied()
                .all(|arg| collect_single_sin_cos_function_name(ctx, arg, found))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_single_sin_cos_function_name(ctx, *left, found)
                && collect_single_sin_cos_function_name(ctx, *right, found)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_single_sin_cos_function_name(ctx, *inner, found)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .all(|item| collect_single_sin_cos_function_name(ctx, item, found)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => true,
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .all(|arg| collect_single_sin_cos_function_name(ctx, arg, found)),
    }
}

fn record_single_sin_cos_name(found: &mut Option<&'static str>, name: &'static str) -> bool {
    if found.is_some() {
        return false;
    }
    *found = Some(name);
    true
}

pub(super) fn generate_cos_product_telescoping_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (base_multiplier, base_factors, factor_count, expands_morrie) =
        if let Some((base_multiplier, base_factors, factor_count)) =
            cos_product_telescoping_base_and_len(ctx, before)
        {
            (base_multiplier, base_factors, factor_count, false)
        } else if let Some((base_multiplier, base_factors, factor_count)) =
            cos_product_telescoping_base_and_len(ctx, after)
        {
            (base_multiplier, base_factors, factor_count, true)
        } else {
            return Vec::new();
        };

    let power = 1i64 << factor_count;
    let product_plain = (0..factor_count)
        .map(|idx| {
            let coeff = 1i64 << idx;
            if coeff == 1 {
                "cos(u)".to_string()
            } else {
                format!("cos({coeff}u)")
            }
        })
        .collect::<Vec<_>>()
        .join(" · ");
    let quotient_plain = format!("sin({power}u) / ({power} · sin(u))");
    let product_latex = (0..factor_count)
        .map(|idx| {
            let coeff = 1i64 << idx;
            if coeff == 1 {
                "\\cos(u)".to_string()
            } else {
                format!("\\cos({coeff}u)")
            }
        })
        .collect::<Vec<_>>()
        .join("\\cdot ");
    let quotient_latex = format!("\\frac{{\\sin({power}u)}}{{{power}\\cdot \\sin(u)}}");
    let (base_u_plain, _) = render_factor_basis(ctx, &base_factors);
    let u_plain = if base_multiplier == 1 {
        base_u_plain
    } else {
        format!("{base_multiplier} · {base_u_plain}")
    };
    let title = if expands_morrie {
        identity_title_with_optional_u("Expandir la ley de Morrie", &u_plain)
    } else {
        identity_title_with_optional_u("Usar el telescopado de cosenos", &u_plain)
    };
    let (before_plain, after_plain, before_latex, after_latex) = if expands_morrie {
        (
            quotient_plain.as_str(),
            product_plain.as_str(),
            quotient_latex.as_str(),
            product_latex.as_str(),
        )
    } else {
        (
            product_plain.as_str(),
            quotient_plain.as_str(),
            product_latex.as_str(),
            quotient_latex.as_str(),
        )
    };

    vec![formula_substep(
        title,
        before_plain,
        after_plain,
        before_latex,
        after_latex,
    )]
}

pub(super) fn dirichlet_cosine_multiple(
    ctx: &Context,
    expr: ExprId,
) -> Option<(usize, Vec<ExprId>)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    let cosine = if is_integer_literal(ctx, *left, 2) {
        *right
    } else if is_integer_literal(ctx, *right, 2) {
        *left
    } else {
        return None;
    };

    let Expr::Function(fn_id, args) = ctx.get(cosine) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "cos" || args.len() != 1 {
        return None;
    }

    let (multiple, base_u) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple <= 0 {
        return None;
    }
    Some((multiple as usize, base_u.into_vec()))
}

fn cos_product_telescoping_base_and_len(
    ctx: &Context,
    expr: ExprId,
) -> Option<(i64, Vec<ExprId>, usize)> {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_info = Vec::new();
    for &factor in &factors {
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Cos) || args.len() != 1 {
            return None;
        }
        let (multiplier, base_u) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
        cos_info.push((multiplier, base_u.into_vec()));
    }

    let base_u = cos_info.first()?.1.clone();
    let mut multipliers = Vec::with_capacity(cos_info.len());
    for (multiplier, u) in cos_info {
        if !same_factor_basis(ctx, &u, &base_u) {
            return None;
        }
        multipliers.push(multiplier);
    }

    multipliers.sort_unstable();
    let base_multiplier = *multipliers.first()?;
    if base_multiplier <= 0 {
        return None;
    }

    for (idx, multiplier) in multipliers.iter().enumerate() {
        let expected = base_multiplier * (1i64 << idx);
        if *multiplier != expected {
            return None;
        }
    }

    Some((base_multiplier, base_u, multipliers.len()))
}

pub(super) fn generate_inverse_trig_double_angle_expansion_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let (outer, inverse, inverse_call) = inverse_trig_double_angle_parts(ctx, before)?;

    let mut work = ctx.clone();
    let two = work.num(2);
    let one = work.num(1);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![inverse_call]);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![inverse_call]);

    let expanded = match outer {
        BuiltinFn::Sin => build_balanced_mul(&mut work, &[two, sin_u, cos_u]),
        BuiltinFn::Cos => match inverse {
            BuiltinFn::Arcsin => {
                let sin_u_squared = work.add(Expr::Pow(sin_u, two));
                let double_sin_u_squared = build_balanced_mul(&mut work, &[two, sin_u_squared]);
                work.add(Expr::Sub(one, double_sin_u_squared))
            }
            BuiltinFn::Arccos => {
                let cos_u_squared = work.add(Expr::Pow(cos_u, two));
                let double_cos_u_squared = build_balanced_mul(&mut work, &[two, cos_u_squared]);
                work.add(Expr::Sub(double_cos_u_squared, one))
            }
            BuiltinFn::Arctan => {
                let cos_u_squared = work.add(Expr::Pow(cos_u, two));
                let sin_u_squared = work.add(Expr::Pow(sin_u, two));
                work.add(Expr::Sub(cos_u_squared, sin_u_squared))
            }
            _ => return None,
        },
        _ => return None,
    };

    Some(vec![
        temp_ctx_substep(
            "Expandir con la identidad de ángulo doble",
            &work,
            before,
            expanded,
        ),
        temp_ctx_substep(
            "Sustituir las razones trigonométricas inversas",
            &work,
            expanded,
            after,
        ),
    ])
}

fn inverse_trig_double_angle_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }

    let outer = ctx.builtin_of(*outer_fn)?;
    if !matches!(outer, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, outer_args[0]);
    if multiple != 2 {
        return None;
    }
    let base_factors = base_factors.into_vec();
    if base_factors.len() != 1 {
        return None;
    }
    let inverse_call = base_factors[0];

    let Expr::Function(inverse_fn, inverse_args) = ctx.get(inverse_call) else {
        return None;
    };
    if inverse_args.len() != 1 {
        return None;
    }

    let inverse = match ctx.builtin_of(*inverse_fn)? {
        BuiltinFn::Arcsin | BuiltinFn::Asin => BuiltinFn::Arcsin,
        BuiltinFn::Arccos | BuiltinFn::Acos => BuiltinFn::Arccos,
        BuiltinFn::Arctan | BuiltinFn::Atan => BuiltinFn::Arctan,
        _ => return None,
    };

    Some((outer, inverse, inverse_call))
}

pub(super) fn generate_trig_angle_sum_diff_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = recursive_trig_angle_sum_diff_substep(ctx, before, false) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        trig_angle_sum_diff_formula(ctx, before)
    {
        return vec![schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some(substep) = recursive_trig_angle_sum_diff_substep(ctx, after, true) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        trig_angle_sum_diff_formula(ctx, after)
    {
        return vec![schema_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )];
    }

    Vec::new()
}

fn recursive_trig_angle_sum_diff_substep(
    ctx: &Context,
    expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple <= 1 {
        return None;
    }

    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, &base_factors.into_vec());
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let previous = multiple - 1;

    let (title, compact_plain, expanded_plain, compact_latex, expanded_latex) = match ctx
        .builtin_of(*fn_id)
    {
        Some(BuiltinFn::Sin) => {
            let title = format!(
                    "Usar sin({previous}u+u) = sin({previous}u) · cos(u) + cos({previous}u) · sin(u), con u = {base_plain}"
                );
            let compact_plain = format!("sin({multiple}u)");
            let expanded_plain = format!("sin({previous}u) · cos(u) + cos({previous}u) · sin(u)");
            let compact_latex = format!("\\sin({multiple}u)");
            let expanded_latex =
                format!("\\sin({previous}u)\\cdot\\cos(u)+\\cos({previous}u)\\cdot\\sin(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        Some(BuiltinFn::Cos) => {
            let title = format!(
                    "Usar cos({previous}u+u) = cos({previous}u) · cos(u) - sin({previous}u) · sin(u), con u = {base_plain}"
                );
            let compact_plain = format!("cos({multiple}u)");
            let expanded_plain = format!("cos({previous}u) · cos(u) - sin({previous}u) · sin(u)");
            let compact_latex = format!("\\cos({multiple}u)");
            let expanded_latex =
                format!("\\cos({previous}u)\\cdot\\cos(u)-\\sin({previous}u)\\cdot\\sin(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        _ => return None,
    };

    if reverse {
        Some(formula_substep(
            title,
            &expanded_plain,
            &compact_plain,
            &expanded_latex,
            &compact_latex,
        ))
    } else {
        Some(formula_substep(
            title,
            &compact_plain,
            &expanded_plain,
            &compact_latex,
            &expanded_latex,
        ))
    }
}

fn trig_angle_sum_diff_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let is_sum = match ctx.get(args[0]) {
        Expr::Add(_, _) => true,
        Expr::Sub(_, _) => false,
        _ => return None,
    };

    match (ctx.builtin_of(*fn_id), is_sum) {
        (Some(BuiltinFn::Sin), true) => Some((
            "Usar sin(A+B) = sin(A) · cos(B) + cos(A) · sin(B)",
            "sin(A+B)",
            "sin(A) · cos(B) + cos(A) · sin(B)",
            "\\sin(A+B)",
            "\\sin(A)\\cdot\\cos(B)+\\cos(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Sin), false) => Some((
            "Usar sin(A-B) = sin(A) · cos(B) - cos(A) · sin(B)",
            "sin(A-B)",
            "sin(A) · cos(B) - cos(A) · sin(B)",
            "\\sin(A-B)",
            "\\sin(A)\\cdot\\cos(B)-\\cos(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Cos), true) => Some((
            "Usar cos(A+B) = cos(A) · cos(B) - sin(A) · sin(B)",
            "cos(A+B)",
            "cos(A) · cos(B) - sin(A) · sin(B)",
            "\\cos(A+B)",
            "\\cos(A)\\cdot\\cos(B)-\\sin(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Cos), false) => Some((
            "Usar cos(A-B) = cos(A) · cos(B) + sin(A) · sin(B)",
            "cos(A-B)",
            "cos(A) · cos(B) + sin(A) · sin(B)",
            "\\cos(A-B)",
            "\\cos(A)\\cdot\\cos(B)+\\sin(A)\\cdot\\sin(B)",
        )),
        _ => None,
    }
}

pub(super) fn extract_trig_function_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*name, BuiltinFn::Sin) {
        Some("sin")
    } else if ctx.is_builtin(*name, BuiltinFn::Cos) {
        Some("cos")
    } else {
        None
    }
}

pub(super) fn trig_power_reduction_kind_and_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigPowerReductionKind, ExprId)> {
    if let Some((trig_fn, arg)) = even_trig_power_arg(ctx, expr) {
        return match trig_fn {
            BuiltinFn::Sin => Some((TrigPowerReductionKind::SinEvenPower, arg)),
            BuiltinFn::Cos => Some((TrigPowerReductionKind::CosEvenPower, arg)),
            _ => None,
        };
    }

    trig_square_product_same_arg(ctx, expr).map(|arg| (TrigPowerReductionKind::SinCosSquares, arg))
}

fn even_trig_power_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let power = small_positive_integer_value(ctx, *exponent)?;
    if power < 4 || power % 2 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        Some((BuiltinFn::Cos, args[0]))
    } else {
        None
    }
}

pub(super) fn trig_square_product_same_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in expr_nary::mul_leaves(ctx, expr) {
        if let Some(arg) = squared_trig_arg(ctx, factor, BuiltinFn::Sin) {
            if sin_arg.replace(arg).is_some() {
                return None;
            }
        } else if let Some(arg) = squared_trig_arg(ctx, factor, BuiltinFn::Cos) {
            if cos_arg.replace(arg).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    match (sin_arg, cos_arg) {
        (Some(left), Some(right)) if left == right => Some(left),
        _ => None,
    }
}

fn squared_trig_arg(ctx: &Context, expr: ExprId, trig_fn: BuiltinFn) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *exponent, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, trig_fn) {
        Some(args[0])
    } else {
        None
    }
}

pub(super) fn find_nested_trig_triple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigTripleAngleKind, ExprId, Vec<ExprId>)> {
    trig_triple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_nested_trig_triple_angle_call(ctx, *left)
            .or_else(|| find_nested_trig_triple_angle_call(ctx, *right)),
        Expr::Neg(inner) => find_nested_trig_triple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_triple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigTripleAngleKind, ExprId, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        TrigTripleAngleKind::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        TrigTripleAngleKind::Cos
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Tan) {
        TrigTripleAngleKind::Tan
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple != 3 {
        return None;
    }

    Some((kind, expr, base_factors.into_vec()))
}

pub(super) fn build_trig_triple_angle_formula_substep(
    ctx: &Context,
    kind: TrigTripleAngleKind,
    _call_expr: ExprId,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) =
        trig_triple_angle_formula_template(kind);
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        schema_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

fn trig_triple_angle_formula_template(
    kind: TrigTripleAngleKind,
) -> (&'static str, &'static str, &'static str, &'static str) {
    match kind {
        TrigTripleAngleKind::Sin => (
            "sin(3u)",
            "3 · sin(u) - 4 · sin(u)^3",
            "\\sin(3u)",
            "3\\cdot\\sin(u)-4\\cdot\\sin(u)^3",
        ),
        TrigTripleAngleKind::Cos => (
            "cos(3u)",
            "4 · cos(u)^3 - 3 · cos(u)",
            "\\cos(3u)",
            "4\\cdot\\cos(u)^3-3\\cdot\\cos(u)",
        ),
        TrigTripleAngleKind::Tan => (
            "tan(3u)",
            "(3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2)",
            "\\tan(3u)",
            "\\frac{3\\cdot\\tan(u)-\\tan(u)^3}{1-3\\cdot\\tan(u)^2}",
        ),
    }
}

pub(super) fn nested_trig_quadruple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuadrupleAngleKind, Vec<ExprId>)> {
    trig_quadruple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => nested_trig_quadruple_angle_call(ctx, *left)
            .or_else(|| nested_trig_quadruple_angle_call(ctx, *right)),
        Expr::Neg(inner) => nested_trig_quadruple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_quadruple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuadrupleAngleKind, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = match ctx.builtin_of(*fn_id)? {
        BuiltinFn::Sin => TrigQuadrupleAngleKind::Sin,
        BuiltinFn::Cos => TrigQuadrupleAngleKind::Cos,
        _ => return None,
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    (multiple == 4).then(|| (kind, base_factors.into_vec()))
}

pub(super) fn build_trig_quadruple_angle_formula_substep(
    ctx: &Context,
    kind: TrigQuadrupleAngleKind,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) = match kind {
        TrigQuadrupleAngleKind::Sin => (
            "sin(4u)",
            "4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u)",
            "\\sin(4u)",
            "4\\cdot\\sin(u)\\cdot\\cos(u)^3-4\\cdot\\sin(u)^3\\cdot\\cos(u)",
        ),
        TrigQuadrupleAngleKind::Cos => (
            "cos(4u)",
            "8 · cos(u)^4 - 8 · cos(u)^2 + 1",
            "\\cos(4u)",
            "8\\cdot\\cos(u)^4-8\\cdot\\cos(u)^2+1",
        ),
    };
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        schema_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

pub(super) fn nested_trig_quintuple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuintupleAngleKind, Vec<ExprId>)> {
    trig_quintuple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => nested_trig_quintuple_angle_call(ctx, *left)
            .or_else(|| nested_trig_quintuple_angle_call(ctx, *right)),
        Expr::Neg(inner) => nested_trig_quintuple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_quintuple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuintupleAngleKind, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        TrigQuintupleAngleKind::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        TrigQuintupleAngleKind::Cos
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple == 5 {
        Some((kind, base_factors.into_vec()))
    } else {
        None
    }
}

pub(super) fn build_trig_quintuple_angle_formula_substep(
    ctx: &Context,
    kind: TrigQuintupleAngleKind,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) =
        trig_quintuple_angle_formula_template(kind);
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        schema_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

fn trig_quintuple_angle_formula_template(
    kind: TrigQuintupleAngleKind,
) -> (&'static str, &'static str, &'static str, &'static str) {
    match kind {
        TrigQuintupleAngleKind::Sin => (
            "sin(5u)",
            "5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5",
            "\\sin(5u)",
            "5\\cdot\\sin(u)-20\\cdot\\sin(u)^3+16\\cdot\\sin(u)^5",
        ),
        TrigQuintupleAngleKind::Cos => (
            "cos(5u)",
            "16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u)",
            "\\cos(5u)",
            "16\\cdot\\cos(u)^5-20\\cdot\\cos(u)^3+5\\cdot\\cos(u)",
        ),
    }
}

pub(super) fn generate_half_angle_tangent_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description.contains("half-angle tangent")
        || step.rule_name == "Half-Angle Tangent Identity"
    {
        let before = step.before_local().unwrap_or(step.before);
        let after = step.after_local().unwrap_or(step.after);
        let variant = half_angle_tangent_variant(ctx, before)
            .or_else(|| half_angle_tangent_variant(ctx, after));
        // C1.8: the audit's second named lie lived in the `None` branch —
        // «Usar tan(u) = …» from the arm that recognized NO variant. Every
        // branch now publishes only if the pair instantiates the template it
        // cites, σ shared across both sides.
        return match variant {
            Some(HalfAngleTangentVariant::OneMinusCosOverSin) => {
                named_identity_substep(ctx, "(1 - cos(2u)) / sin(2u)", "tan(u)", before, after)
                    .into_iter()
                    .collect()
            }
            Some(HalfAngleTangentVariant::SinOverOnePlusCos) => {
                named_identity_substep(ctx, "sin(2u) / (1 + cos(2u))", "tan(u)", before, after)
                    .into_iter()
                    .collect()
            }
            None => named_identity_substep(ctx, "tan(u)", "(1 - cos(2u)) / sin(2u)", before, after)
                .into_iter()
                .collect(),
        };
    }
    Vec::new()
}

fn half_angle_tangent_variant(ctx: &Context, expr: ExprId) -> Option<HalfAngleTangentVariant> {
    let (num, den) = as_div(ctx, expr)?;

    if is_one_minus_cos_double(ctx, num) && is_sin_double(ctx, den) {
        return Some(HalfAngleTangentVariant::OneMinusCosOverSin);
    }
    if is_sin_double(ctx, num) && is_one_plus_cos_double(ctx, den) {
        return Some(HalfAngleTangentVariant::SinOverOnePlusCos);
    }

    None
}

fn is_sin_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    ctx.is_builtin(*fn_id, BuiltinFn::Sin)
        && args.len() == 1
        && is_double_angle(ctx, args[0]).is_some()
}

fn is_one_minus_cos_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return false;
    };
    is_one(ctx, *lhs) && is_cos_double(ctx, *rhs)
}

fn is_one_plus_cos_double(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            (is_one(ctx, *lhs) && is_cos_double(ctx, *rhs))
                || (is_one(ctx, *rhs) && is_cos_double(ctx, *lhs))
        }
        _ => false,
    }
}

fn is_cos_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    ctx.is_builtin(*fn_id, BuiltinFn::Cos)
        && args.len() == 1
        && is_double_angle(ctx, args[0]).is_some()
}

pub(super) fn build_pythagorean_high_power_sine_substeps(
    ctx: &Context,
    local_before: ExprId,
) -> Option<Vec<SubStep>> {
    let mut work = ctx.clone();
    let arg = first_trig_argument_with_builtin(&work, local_before, BuiltinFn::Sin)?;
    let one = work.num(1);
    let two = work.num(2);
    let three = work.num(3);
    let four = work.num(4);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_sq = work.add(Expr::Pow(sin_u, two));
    let cos_sq = work.add(Expr::Pow(cos_u, two));
    let sin_cubed = work.add(Expr::Pow(sin_u, three));
    let four_sin = work.add(Expr::Mul(four, sin_u));
    let _four_sin_cubed = work.add(Expr::Mul(four, sin_cubed));
    let one_minus_sin_sq = work.add(Expr::Sub(one, sin_sq));
    let factorized = work.add(Expr::Mul(four_sin, one_minus_sin_sq));
    let four_sin_again = work.add(Expr::Mul(four, sin_u));
    let pythagorean = work.add(Expr::Mul(four_sin_again, cos_sq));
    let double_arg = work.add(Expr::Mul(two, arg));
    let sin_2u = work.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let two_sin_2u = work.add(Expr::Mul(two, sin_2u));
    let final_expr = work.add(Expr::Mul(two_sin_2u, cos_u));

    Some(vec![
        mixed_ctx_substep(
            "Sacar factor común 4 · sin(u)",
            ctx,
            local_before,
            &work,
            factorized,
        ),
        temp_ctx_substep(
            "Usar 1 - sin(u)^2 = cos(u)^2",
            &work,
            factorized,
            pythagorean,
        ),
        temp_ctx_substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            &work,
            pythagorean,
            final_expr,
        ),
    ])
}

pub(super) fn build_pythagorean_high_power_cos_substeps(
    ctx: &Context,
    local_before: ExprId,
    negated: bool,
) -> Option<Vec<SubStep>> {
    let mut work = ctx.clone();
    let arg = first_trig_argument_with_builtin(&work, local_before, BuiltinFn::Cos)?;
    let one = work.num(1);
    let two = work.num(2);
    let four = work.num(4);
    let neg_four = work.num(-4);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_sq = work.add(Expr::Pow(cos_u, two));
    let sin_sq = work.add(Expr::Pow(sin_u, two));
    let lead_coeff = if negated { neg_four } else { four };
    let lead_cos = work.add(Expr::Mul(lead_coeff, cos_u));
    let one_minus_cos_sq = work.add(Expr::Sub(one, cos_sq));
    let factorized = work.add(Expr::Mul(lead_cos, one_minus_cos_sq));
    let lead_cos_again = work.add(Expr::Mul(lead_coeff, cos_u));
    let pythagorean = work.add(Expr::Mul(lead_cos_again, sin_sq));
    let double_arg = work.add(Expr::Mul(two, arg));
    let sin_2u = work.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let final_coeff = if negated { work.num(-2) } else { two };
    let final_prefix = work.add(Expr::Mul(final_coeff, sin_2u));
    let final_expr = work.add(Expr::Mul(final_prefix, sin_u));

    Some(vec![
        mixed_ctx_substep(
            if negated {
                "Sacar factor común -4 · cos(u)"
            } else {
                "Sacar factor común 4 · cos(u)"
            },
            ctx,
            local_before,
            &work,
            factorized,
        ),
        temp_ctx_substep(
            "Usar 1 - cos(u)^2 = sin(u)^2",
            &work,
            factorized,
            pythagorean,
        ),
        temp_ctx_substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            &work,
            pythagorean,
            final_expr,
        ),
    ])
}

fn first_trig_argument_with_builtin(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) => {
            Some(args[0])
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            first_trig_argument_with_builtin(ctx, *left, builtin)
                .or_else(|| first_trig_argument_with_builtin(ctx, *right, builtin))
        }
        Expr::Div(left, right) | Expr::Pow(left, right) => {
            first_trig_argument_with_builtin(ctx, *left, builtin)
                .or_else(|| first_trig_argument_with_builtin(ctx, *right, builtin))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            first_trig_argument_with_builtin(ctx, *inner, builtin)
        }
        Expr::Function(_, args) => args
            .iter()
            .find_map(|arg| first_trig_argument_with_builtin(ctx, *arg, builtin)),
        Expr::Matrix { data, .. } => data
            .iter()
            .find_map(|item| first_trig_argument_with_builtin(ctx, *item, builtin)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => None,
    }
}

/// Migrated to the instance↔template matcher after the extended shadow pass
/// (2026-07-27): the old emitter picked its template by SUBSTRING-sniffing the
/// display (`contains("sin(")`), so a mixed pair took the sine branch
/// regardless and the SCALED pair `4·cos(u)² − 2 ⟹ 2·cos(2u)` was cited as
/// «2·cos²−1 = cos(2u)» — not an instance. Scaled pairs now decline honestly
/// (measured residual: matcher coefficient-peeling, named in the shadow pass).
pub(super) fn generate_cos_2x_additive_contraction_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    const COS_2X_TEMPLATES: [(&str, &str); 2] =
        [("1 - 2·sin(u)^2", "cos(2u)"), ("2·cos(u)^2 - 1", "cos(2u)")];
    named_identity_from_table(ctx, &COS_2X_TEMPLATES, before, after)
        .into_iter()
        .collect()
}

/// Migrated to the instance↔template matcher (2026-07-27) after the shadow's
/// «1/1 covered» for these four rules turned out SPURIOUS — the directed mode
/// was matching them via the factorial template ((k+1)·k!/k! = k+1), whose
/// engine-equal instantiation is title-truth-useless; the real census rows
/// did not exist. Each rule states exactly ONE identity, so the RULE NAME is
/// the router (no description-string fragility: the simplify route spells
/// «1 + tan²(x) = sec²(x)» where derive spells «Recognize 1 + tan²(u) as
/// sec²(u)») and the matcher gates the pair.
pub(super) fn generate_sec_csc_squared_expansion_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (lhs, rhs): (&'static str, &'static str) = match step.rule_name.as_str() {
        "Expand Secant Squared" => ("sec(u)^2", "1 + tan(u)^2"),
        "Expand Cosecant Squared" => ("csc(u)^2", "1 + cot(u)^2"),
        _ => return Vec::new(),
    };
    named_identity_substep(ctx, lhs, rhs, before, after)
        .into_iter()
        .collect()
}

/// Mirror of the expansion emitter: recognition orientation, matcher-gated.
pub(super) fn generate_sec_csc_squared_contraction_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (lhs, rhs): (&'static str, &'static str) = match step.rule_name.as_str() {
        "Recognize Secant Squared" => ("1 + tan(u)^2", "sec(u)^2"),
        "Recognize Cosecant Squared" => ("1 + cot(u)^2", "csc(u)^2"),
        _ => return Vec::new(),
    };
    named_identity_substep(ctx, lhs, rhs, before, after)
        .into_iter()
        .collect()
}

pub(super) fn trig_square_term(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (base, exponent) = as_pow(ctx, expr)?;
    if !is_small_positive_integer(ctx, exponent, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        Some((builtin, args[0]))
    } else {
        None
    }
}

pub(super) fn generate_trig_parity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let mut work = ctx.clone();
    let is_odd = cas_math::trig_core_identity_support::try_rewrite_trig_odd_even_parity_expr(
        &mut work, before,
    )
    .map(|rewrite| rewrite.kind == cas_math::trig_core_identity_support::TrigOddEvenParityKind::Odd)
    .unwrap_or_else(|| matches!(ctx.get(after), Expr::Neg(_)));

    if is_odd {
        vec![schema_substep(
            "Usar que una función impar cumple f(-u) = -f(u)",
            "f(-u)",
            "-f(u)",
            "f(-u)",
            "-f(u)",
        )]
    } else {
        vec![schema_substep(
            "Usar que una función par cumple f(-u) = f(u)",
            "f(-u)",
            "f(u)",
            "f(-u)",
            "f(u)",
        )]
    }
}

pub(super) fn generate_trig_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if step.description.contains("tangent to sine over cosine") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar tan(u) = sin(u) / cos(u)",
            before,
            after,
        )];
    }
    Vec::new()
}

/// Migrated to the instance↔template matcher after the derive-route shadow
/// (2026-07-27) measured the rule at 6/6 census-covered. The engine's own
/// application description routes each pair to ITS oriented template — the
/// directional title («Usar sec(u) = 1/cos(u)» for the expansion) is the
/// gesture's phrasing, and the matcher now gates that the pair actually
/// instantiates it; a described-but-non-instance pair declines.
pub(super) fn generate_reciprocal_trig_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let template: Option<(&'static str, &'static str)> = match step.description.as_str() {
        "Expand sec(u) as 1 / cos(u)" => Some(("sec(u)", "1 / cos(u)")),
        "Expand csc(u) as 1 / sin(u)" => Some(("csc(u)", "1 / sin(u)")),
        "Expand cot(u) as cos(u) / sin(u)" => Some(("cot(u)", "cos(u) / sin(u)")),
        "Recognize 1 / cos(u) as sec(u)" => Some(("1 / cos(u)", "sec(u)")),
        "Recognize 1 / sin(u) as csc(u)" => Some(("1 / sin(u)", "csc(u)")),
        "Recognize cos(u) / sin(u) as cot(u)" => Some(("cos(u) / sin(u)", "cot(u)")),
        _ => None,
    };
    let Some((lhs, rhs)) = template else {
        return Vec::new();
    };
    named_identity_substep(ctx, lhs, rhs, before, after)
        .into_iter()
        .collect()
}

/// Migrated to the instance↔template matcher after the shadow pass
/// (2026-07-27) measured the rule: 7 corpus pairs, and the old emitter cited
/// «sin(u) / cos(u) = tan(u)» for ALL of them — including `cos/sin ⟹ cot`,
/// `1/cos ⟹ sec` and `1/sin ⟹ csc`, where that title is simply the wrong
/// identity. Each candidate template is census-adjudicated and the sub-step
/// publishes only for the pair that PROVABLY instantiates it; a pair that
/// instantiates none stays silent (honest, and measured by the shadow pass).
pub(super) fn generate_trig_quotient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    const QUOTIENT_TEMPLATES: [(&str, &str); 4] = [
        ("sin(u) / cos(u)", "tan(u)"),
        ("cos(u) / sin(u)", "cot(u)"),
        ("1 / cos(u)", "sec(u)"),
        ("1 / sin(u)", "csc(u)"),
    ];
    named_identity_from_table(ctx, &QUOTIENT_TEMPLATES, before, after)
        .into_iter()
        .collect()
}

pub(super) fn generate_cos_diff_sin_diff_quotient_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);

    if let Some(tan_arg) = tan_call_arg(ctx, local_after).or_else(|| tan_call_arg(ctx, step.after))
    {
        if let Some((_, _)) = as_div(ctx, local_before).or_else(|| as_div(ctx, step.before)) {
            let mut work = ctx.clone();
            let sin_arg = work.call_builtin(BuiltinFn::Sin, vec![tan_arg]);
            let cos_arg = work.call_builtin(BuiltinFn::Cos, vec![tan_arg]);
            let simplified_quotient = work.add_raw(Expr::Div(sin_arg, cos_arg));
            let tan_expr = work.call_builtin(BuiltinFn::Tan, vec![tan_arg]);
            return vec![
                mixed_ctx_substep(
                    "Cancelar el factor común del numerador y del denominador",
                    ctx,
                    local_before,
                    &work,
                    simplified_quotient,
                ),
                temp_ctx_substep(
                    "Reconocer el patrón sin(u) / cos(u) = tan(u)",
                    &work,
                    simplified_quotient,
                    tan_expr,
                ),
            ];
        }
    }

    let before_div = as_div(ctx, local_before).or_else(|| as_div(ctx, step.before));
    let after_div = as_div(ctx, local_after).or_else(|| as_div(ctx, step.after));
    let (Some((before_num, before_den)), Some((after_num, after_den))) = (before_div, after_div)
    else {
        return Vec::new();
    };

    if before_den == after_den && before_num != after_num {
        return vec![concrete_expr_substep(
            ctx,
            "Usar cos(A) - cos(B) = 2 · sin((A+B)/2) · sin((B-A)/2)",
            before_num,
            after_num,
        )];
    }

    if before_num == after_num && before_den != after_den {
        return vec![concrete_expr_substep(
            ctx,
            "Usar sin(B) - sin(A) = 2 · cos((A+B)/2) · sin((B-A)/2)",
            before_den,
            after_den,
        )];
    }

    Vec::new()
}

fn tan_call_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Tan) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn generate_inverse_trig_sum_relation_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let mut out = Vec::new();
    let pair_before = step.before_local().unwrap_or(step.before);
    let pair_after = step.after_local().unwrap_or(step.after);

    if step.before != pair_before {
        out.push(
            SubStep::new(
                "Juntar la pareja que encaja con la identidad",
                display_expr(ctx, step.before),
                display_expr(ctx, pair_before),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, pair_before)),
        );
    }

    if pair_before != pair_after {
        let pair_value_title = if step.rule_name == "Inverse Trig Sum Identity" {
            "Aquí arcsin(x) y arccos(x) suman pi/2"
        } else {
            "Esa pareja vale pi/2"
        };
        out.push(
            SubStep::new(
                pair_value_title,
                display_expr(ctx, pair_before),
                display_expr(ctx, pair_after),
            )
            .with_before_latex(latex_expr(ctx, pair_before))
            .with_after_latex(latex_expr(ctx, pair_after)),
        );
    }

    out
}

pub(super) fn generate_inverse_trig_composition_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if let Some(substeps) = generate_direct_inverse_trig_composition_substeps(ctx, step) {
        return substeps;
    }

    if step.description.contains("sin(arctan") || step.description.contains("cos(arctan") {
        return generate_arctan_right_triangle_composition_substeps(ctx, step);
    }

    if step.description.contains("cos(arcsin")
        || step.description.contains("sin(arccos")
        || step.description.contains("tan(arcsin")
    {
        return generate_arcsin_arccos_complement_composition_substeps(ctx, step);
    }

    if !step.description.contains("arcsin") || !step.description.contains("arctan") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(arcsin_arg) =
        inverse_trig_unary_arg(ctx, before, &[BuiltinFn::Arcsin, BuiltinFn::Asin])
    else {
        return Vec::new();
    };
    let Some(x) = inverse_trig_unary_arg(ctx, after, &[BuiltinFn::Arctan, BuiltinFn::Atan]) else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let arctan_x = work.call_builtin(BuiltinFn::Arctan, vec![x]);
    let sin_arctan_x = work.call_builtin(BuiltinFn::Sin, vec![arctan_x]);
    let arcsin_sin_arctan_x = work.call_builtin(BuiltinFn::Arcsin, vec![sin_arctan_x]);

    vec![
        temp_ctx_substep(
            "Reconocer x/sqrt(1+x^2) como sin(arctan(x))",
            &work,
            arcsin_arg,
            sin_arctan_x,
        ),
        temp_ctx_substep(
            "Sustituir dentro de arcsin",
            &work,
            before,
            arcsin_sin_arctan_x,
        ),
        temp_ctx_substep(
            "Usar asin(sin(u)) = u en el rango principal",
            &work,
            arcsin_sin_arctan_x,
            arctan_x,
        ),
    ]
}

fn generate_direct_inverse_trig_composition_substeps(
    ctx: &Context,
    step: &Step,
) -> Option<Vec<SubStep>> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (outer_fn, inner_fn, argument) = direct_inverse_trig_composition_parts(ctx, before)?;
    if compare_expr(ctx, argument, after) != Ordering::Equal {
        return None;
    }

    let mut work = ctx.clone();
    let u = work.var("u");
    let inner = work.call_builtin(inner_fn, vec![u]);
    let composition = work.call_builtin(outer_fn, vec![inner]);
    let outer_name = inverse_trig_fn_name(outer_fn);
    let inner_name = inverse_trig_fn_name(inner_fn);

    Some(vec![
        temp_ctx_substep(
            format!("Usar que {outer_name} y {inner_name} son funciones inversas"),
            &work,
            composition,
            u,
        ),
        SubStep::new(
            format!("Aquí u = {}", human_expr(ctx, argument)),
            human_expr(ctx, before),
            human_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
    ])
}

fn direct_inverse_trig_composition_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }

    let outer = ctx.builtin_of(*outer_fn)?;
    let inner = ctx.builtin_of(*inner_fn)?;
    matches!(
        (outer, inner),
        (BuiltinFn::Sin, BuiltinFn::Arcsin)
            | (BuiltinFn::Sin, BuiltinFn::Asin)
            | (BuiltinFn::Cos, BuiltinFn::Arccos)
            | (BuiltinFn::Cos, BuiltinFn::Acos)
            | (BuiltinFn::Tan, BuiltinFn::Arctan)
            | (BuiltinFn::Tan, BuiltinFn::Atan)
    )
    .then_some((outer, inner, inner_args[0]))
}

fn inverse_trig_fn_name(function: BuiltinFn) -> &'static str {
    match function {
        BuiltinFn::Sin => "sin",
        BuiltinFn::Cos => "cos",
        BuiltinFn::Tan => "tan",
        BuiltinFn::Arcsin | BuiltinFn::Asin => "arcsin",
        BuiltinFn::Arccos | BuiltinFn::Acos => "arccos",
        BuiltinFn::Arctan | BuiltinFn::Atan => "arctan",
        _ => "función",
    }
}

pub(super) fn inverse_trig_unary_arg(
    ctx: &Context,
    expr: ExprId,
    accepted_builtins: &[BuiltinFn],
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    accepted_builtins.contains(&builtin).then_some(args[0])
}

pub(super) fn generate_trig_log_table_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let Some(table_match) = trig_log_table_integrand_arg(ctx, args[0], var_name) else {
        return Vec::new();
    };

    let title = match table_match.kind {
        TrigLogTableKind::Tangent => "Usar la regla de tan(u) -> -ln|cos(u)|",
        TrigLogTableKind::Cotangent => "Usar la regla de cot(u) -> ln|sin(u)|",
        TrigLogTableKind::Secant => "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
        TrigLogTableKind::Cosecant => "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }

    match table_match.trace {
        TrigLogTableTrace::AffineArgument => {
            substeps.push(
                SubStep::keyed(
                    "usub.identify_affine_argument",
                    vec![],
                    display_expr(ctx, table_match.arg),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(ctx, table_match.arg))
                .with_after_latex(latex_expr(ctx, after)),
            );
        }
        TrigLogTableTrace::ConstantMultipleAffineArgument {
            cofactor_display,
            cofactor_latex,
            coefficient,
            slope,
        } => {
            substeps.push(
                SubStep::keyed(
                    "usub.identify_affine_argument",
                    vec![],
                    display_expr(ctx, table_match.arg),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(ctx, table_match.arg))
                .with_after_latex(latex_expr(ctx, after)),
            );
            let scale = coefficient / slope.clone();
            substeps.push(
                SubStep::keyed(
                    "usub.adjust_constant_factor",
                    vec![],
                    cofactor_display,
                    format!(
                        "{} · {}",
                        rational_display(&scale),
                        rational_display(&slope)
                    ),
                )
                .with_before_latex(cofactor_latex)
                .with_after_latex(format!(
                    "{}\\cdot {}",
                    rational_latex(&scale),
                    rational_latex(&slope)
                )),
            );
        }
        TrigLogTableTrace::PolynomialCofactor(trace) => {
            let TrigLogPolynomialCofactorTrace {
                cofactor_display,
                cofactor_latex,
                derivative_display,
                derivative_latex,
                scale,
                symbolic_scale_display,
                symbolic_scale_latex,
            } = *trace;
            substeps.push(
                SubStep::keyed(
                    "usub.identify_u_du",
                    vec![],
                    format!("u = {}", display_expr(ctx, table_match.arg)),
                    format!("du = {} dx", derivative_display),
                )
                .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.arg)))
                .with_after_latex(format!("du = {}\\,dx", derivative_latex)),
            );
            push_integration_constant_factor_adjustment_substep(
                &mut substeps,
                IntegrationConstantFactorAdjustment {
                    cofactor_display: &cofactor_display,
                    cofactor_latex: &cofactor_latex,
                    derivative_display: &derivative_display,
                    derivative_latex: &derivative_latex,
                    scale: &scale,
                    symbolic_scale_display: symbolic_scale_display.as_deref(),
                    symbolic_scale_latex: symbolic_scale_latex.as_deref(),
                },
            );
        }
    }

    substeps
}

pub(super) fn trig_log_denominator_factor_kind_and_arg(
    ctx: &Context,
    denominator: ExprId,
) -> Option<(BigRational, TrigLogTableKind, ExprId)> {
    if let Some((kind, arg)) = trig_log_denominator_kind_and_arg(ctx, denominator) {
        return Some((BigRational::one(), kind, arg));
    }

    let factors = expr_nary::mul_leaves(ctx, denominator);
    if factors.len() < 2 {
        return None;
    }

    let mut denominator_scale = BigRational::one();
    let mut matched_denominator: Option<(TrigLogTableKind, ExprId)> = None;
    for factor in factors {
        if let Some(value) = as_rational_const(ctx, factor, 8) {
            denominator_scale *= value;
            continue;
        }
        let parts = trig_log_denominator_kind_and_arg(ctx, factor)?;
        if matched_denominator.replace(parts).is_some() {
            return None;
        }
    }
    if denominator_scale.is_zero() {
        return None;
    }
    let (kind, arg) = matched_denominator?;
    Some((denominator_scale, kind, arg))
}

fn trig_log_denominator_kind_and_arg(
    ctx: &Context,
    denominator: ExprId,
) -> Option<(TrigLogTableKind, ExprId)> {
    let (builtin, arg) = unary_builtin_arg(ctx, denominator)?;
    let kind = match builtin {
        BuiltinFn::Cos => TrigLogTableKind::Secant,
        BuiltinFn::Sin => TrigLogTableKind::Cosecant,
        _ => return None,
    };
    Some((kind, arg))
}

pub(super) fn trig_log_table_factor_kind_and_arg(
    ctx: &Context,
    factor: ExprId,
) -> Option<(TrigLogTableKind, ExprId)> {
    if let Some((builtin, arg)) = unary_builtin_arg(ctx, factor) {
        let kind = match builtin {
            BuiltinFn::Tan => TrigLogTableKind::Tangent,
            BuiltinFn::Cot => TrigLogTableKind::Cotangent,
            BuiltinFn::Sec => TrigLogTableKind::Secant,
            BuiltinFn::Csc => TrigLogTableKind::Cosecant,
            _ => return None,
        };
        return Some((kind, arg));
    }

    let (num, den) = as_div(ctx, factor)?;
    let coefficient = as_rational_const(ctx, num, 8)?;
    if !coefficient.is_one() {
        return None;
    }
    trig_log_denominator_kind_and_arg(ctx, den)
}

pub(super) fn trig_log_sqrt_chain_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    trig_log_sqrt_chain_from_factors(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

pub(super) fn trig_log_sqrt_chain_from_factors(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let denominator_factors = collect_mul_chain_factors_readonly(ctx, den);

    for (trig_index, factor) in numerator_factors.iter().enumerate() {
        let Some((builtin, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        let kind = match builtin {
            BuiltinFn::Tan => TrigLogTableKind::Tangent,
            BuiltinFn::Cot => TrigLogTableKind::Cotangent,
            _ => continue,
        };

        let remaining_numerator = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
            .collect::<Vec<_>>();
        if contains_var_dependent_trig_factor(ctx, &remaining_numerator, var_name) {
            continue;
        }
        let trace = sqrt_chain_cofactor_derivative_trace(
            ctx,
            arg,
            numerator_negative,
            &remaining_numerator,
            &denominator_factors,
            var_name,
        )?;

        return Some(TrigLogTableMatch {
            kind,
            arg,
            trace: TrigLogTableTrace::PolynomialCofactor(Box::new(
                TrigLogPolynomialCofactorTrace {
                    cofactor_display: trace.cofactor_display,
                    cofactor_latex: trace.cofactor_latex,
                    derivative_display: trace.derivative_display,
                    derivative_latex: trace.derivative_latex,
                    scale: trace.scale,
                    symbolic_scale_display: trace.symbolic_scale_display,
                    symbolic_scale_latex: trace.symbolic_scale_latex,
                },
            )),
        });
    }

    for (trig_index, factor) in denominator_factors.iter().enumerate() {
        let Some((builtin, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        let kind = match builtin {
            BuiltinFn::Cos => TrigLogTableKind::Secant,
            BuiltinFn::Sin => TrigLogTableKind::Cosecant,
            _ => continue,
        };
        if sqrt_chain_arg_radicand(ctx, arg).is_none() {
            continue;
        }

        let remaining_denominator = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
            .collect::<Vec<_>>();
        if contains_var_dependent_trig_factor(ctx, numerator_factors, var_name)
            || contains_var_dependent_trig_factor(ctx, &remaining_denominator, var_name)
        {
            continue;
        }
        let trace = sqrt_chain_cofactor_derivative_trace(
            ctx,
            arg,
            numerator_negative,
            numerator_factors,
            &remaining_denominator,
            var_name,
        )?;

        return Some(TrigLogTableMatch {
            kind,
            arg,
            trace: TrigLogTableTrace::PolynomialCofactor(Box::new(
                TrigLogPolynomialCofactorTrace {
                    cofactor_display: trace.cofactor_display,
                    cofactor_latex: trace.cofactor_latex,
                    derivative_display: trace.derivative_display,
                    derivative_latex: trace.derivative_latex,
                    scale: trace.scale,
                    symbolic_scale_display: trace.symbolic_scale_display,
                    symbolic_scale_latex: trace.symbolic_scale_latex,
                },
            )),
        });
    }

    None
}

fn contains_var_dependent_trig_factor(ctx: &Context, factors: &[ExprId], var_name: &str) -> bool {
    factors
        .iter()
        .any(|factor| expr_contains_var_dependent_trig_factor(ctx, *factor, var_name))
}

fn expr_contains_var_dependent_trig_factor(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    if let Some((builtin, arg)) = unary_builtin_arg(ctx, expr) {
        return matches!(
            builtin,
            BuiltinFn::Sin
                | BuiltinFn::Cos
                | BuiltinFn::Tan
                | BuiltinFn::Cot
                | BuiltinFn::Sec
                | BuiltinFn::Csc
        ) && contains_named_var(ctx, arg, var_name);
    }

    match ctx.get(expr) {
        Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Add(left, right)
        | Expr::Sub(left, right) => {
            expr_contains_var_dependent_trig_factor(ctx, *left, var_name)
                || expr_contains_var_dependent_trig_factor(ctx, *right, var_name)
        }
        Expr::Neg(inner) => expr_contains_var_dependent_trig_factor(ctx, *inner, var_name),
        Expr::Pow(base, exponent) => {
            expr_contains_var_dependent_trig_factor(ctx, *base, var_name)
                || expr_contains_var_dependent_trig_factor(ctx, *exponent, var_name)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_var_dependent_trig_factor(ctx, *arg, var_name)),
        _ => false,
    }
}

pub(super) fn trig_log_symbolic_scaled_quotient_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    trig_log_symbolic_scaled_quotient_from_factors(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

pub(super) fn trig_log_symbolic_scaled_quotient_from_factors(
    ctx: &Context,
    negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    let (expected_num_builtin, kind) = match den_builtin {
        BuiltinFn::Cos => (BuiltinFn::Sin, TrigLogTableKind::Tangent),
        BuiltinFn::Sin => (BuiltinFn::Cos, TrigLogTableKind::Cotangent),
        _ => return None,
    };

    for (trig_index, factor) in numerator_factors.iter().enumerate() {
        let Some((num_builtin, num_arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        if num_builtin != expected_num_builtin
            || compare_expr(ctx, num_arg, den_arg) != Ordering::Equal
        {
            continue;
        }

        let cofactor_factors = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
            .collect::<Vec<_>>();
        let trace = polynomial_derivative_cofactor_trace_with_symbolic_scale(
            ctx,
            negative,
            &cofactor_factors,
            den_arg,
            var_name,
        )?;

        return Some(TrigLogTableMatch {
            kind,
            arg: den_arg,
            trace: TrigLogTableTrace::PolynomialCofactor(Box::new(
                TrigLogPolynomialCofactorTrace {
                    cofactor_display: trace.cofactor_display,
                    cofactor_latex: trace.cofactor_latex,
                    derivative_display: trace.derivative_display,
                    derivative_latex: trace.derivative_latex,
                    scale: trace.scale,
                    symbolic_scale_display: trace.symbolic_scale_display,
                    symbolic_scale_latex: trace.symbolic_scale_latex,
                },
            )),
        });
    }

    None
}

pub(super) fn trig_factor_with_polynomial_cofactor(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<TrigPolynomialCofactor> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if let Some((builtin, arg)) = unary_builtin_arg(ctx, *left) {
        if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
            let cofactor_poly = Polynomial::from_expr(ctx, *right, var_name).ok()?;
            return Some(TrigPolynomialCofactor {
                builtin,
                arg,
                expr: *right,
                polynomial: cofactor_poly,
            });
        }
    }

    let (builtin, arg) = unary_builtin_arg(ctx, *right)?;
    if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }
    let cofactor_poly = Polynomial::from_expr(ctx, *left, var_name).ok()?;
    Some(TrigPolynomialCofactor {
        builtin,
        arg,
        expr: *left,
        polynomial: cofactor_poly,
    })
}

pub(super) fn trig_log_symbolic_scaled_polynomial_reciprocal_arg(
    ctx: &Context,
    negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    let kind = match den_builtin {
        BuiltinFn::Cos => TrigLogTableKind::Secant,
        BuiltinFn::Sin => TrigLogTableKind::Cosecant,
        _ => return None,
    };

    let trace = polynomial_derivative_cofactor_trace_with_symbolic_scale(
        ctx,
        negative,
        numerator_factors,
        den_arg,
        var_name,
    )?;

    Some(TrigLogTableMatch {
        kind,
        arg: den_arg,
        trace: TrigLogTableTrace::PolynomialCofactor(Box::new(TrigLogPolynomialCofactorTrace {
            cofactor_display: trace.cofactor_display,
            cofactor_latex: trace.cofactor_latex,
            derivative_display: trace.derivative_display,
            derivative_latex: trace.derivative_latex,
            scale: trace.scale,
            symbolic_scale_display: trace.symbolic_scale_display,
            symbolic_scale_latex: trace.symbolic_scale_latex,
        })),
    })
}

pub(super) fn trig_log_polynomial_reciprocal_arg(
    ctx: &Context,
    numerator: Polynomial,
    den: ExprId,
    var_name: &str,
) -> Option<(TrigLogTableKind, ExprId, Polynomial, BigRational)> {
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    let kind = match den_builtin {
        BuiltinFn::Cos => TrigLogTableKind::Secant,
        BuiltinFn::Sin => TrigLogTableKind::Cosecant,
        _ => return None,
    };

    let arg_poly = polynomial_trace_arg_ignoring_independent_addends(ctx, den_arg, var_name)?;
    if arg_poly.degree() <= 1 {
        return None;
    }
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    (!scale.is_zero()).then_some((kind, den_arg, derivative, scale))
}

pub(super) fn generate_trig_quotient_table_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let mut scratch = ctx.clone();
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_trig_quotient_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) {
        return Vec::new();
    }

    let Some(table_match) = trig_quotient_table_integrand(ctx, args[0], var_name) else {
        return Vec::new();
    };
    let title = match table_match.kind {
        TrigQuotientTableKind::Tangent => "Usar la regla de tan(u) -> -ln|cos(u)|",
        TrigQuotientTableKind::Cotangent => "Usar la regla de cot(u) -> ln|sin(u)|",
        TrigQuotientTableKind::SecantSquare => "Usar la regla de 1/cos(u)^2 -> tan(u)",
        TrigQuotientTableKind::CosecantSquare => "Usar la regla de 1/sin(u)^2 -> -cot(u)",
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, table_match.arg)),
            format!("du = {} dx", table_match.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.arg)))
        .with_after_latex(format!("du = {}\\,dx", table_match.derivative_latex)),
    );
    if !table_match.scale.is_one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                table_match.cofactor_display,
                format!(
                    "{} · {}",
                    rational_display(&table_match.scale),
                    table_match.derivative_display
                ),
            )
            .with_before_latex(table_match.cofactor_latex)
            .with_after_latex(format!(
                "{}\\cdot {}",
                rational_latex(&table_match.scale),
                table_match.derivative_latex
            )),
        );
    }

    substeps
}

pub(super) fn trig_quotient_table_from_product(
    ctx: &Context,
    negative: bool,
    factors: &[ExprId],
    var_name: &str,
) -> Option<TrigQuotientTableMatch> {
    for (kernel_index, factor) in factors.iter().enumerate() {
        let Some((builtin, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        let kind = match builtin {
            BuiltinFn::Tan => TrigQuotientTableKind::Tangent,
            BuiltinFn::Cot => TrigQuotientTableKind::Cotangent,
            _ => continue,
        };
        if !contains_named_var(ctx, arg, var_name) {
            continue;
        }

        let remaining_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = trig_quotient_match_from_cofactor(
            ctx,
            kind,
            arg,
            negative,
            &remaining_factors,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    for (div_index, factor) in factors.iter().enumerate() {
        let Expr::Div(num, den) = ctx.get(*factor) else {
            continue;
        };
        let (div_negative, div_numerator_factors) = signed_mul_factors(ctx, *num);
        let numerator_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != div_index).then_some(*factor))
            .chain(div_numerator_factors.into_iter())
            .collect::<Vec<_>>();
        if let Some(table_match) = trig_quotient_table_from_div_factors(
            ctx,
            negative != div_negative,
            &numerator_factors,
            *den,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    None
}

pub(super) fn trig_quotient_table_from_div(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<TrigQuotientTableMatch> {
    let (negative, numerator_factors) = signed_mul_factors(ctx, num);
    trig_quotient_table_from_div_factors(ctx, negative, &numerator_factors, den, var_name)
}

fn trig_quotient_table_from_div_factors(
    ctx: &Context,
    negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<TrigQuotientTableMatch> {
    if let Some((den_builtin, arg)) = trig_square_denominator_arg(ctx, den) {
        let kind = match den_builtin {
            BuiltinFn::Cos => TrigQuotientTableKind::SecantSquare,
            BuiltinFn::Sin => TrigQuotientTableKind::CosecantSquare,
            _ => return None,
        };
        if !contains_named_var(ctx, arg, var_name) {
            return None;
        }

        return trig_quotient_match_from_cofactor(
            ctx,
            kind,
            arg,
            negative,
            numerator_factors,
            var_name,
        );
    }

    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    let (kind, num_builtin) = match den_builtin {
        BuiltinFn::Cos => (TrigQuotientTableKind::Tangent, BuiltinFn::Sin),
        BuiltinFn::Sin => (TrigQuotientTableKind::Cotangent, BuiltinFn::Cos),
        _ => return None,
    };
    if !contains_named_var(ctx, den_arg, var_name) {
        return None;
    }

    for (kernel_index, factor) in numerator_factors.iter().enumerate() {
        let Some((candidate_builtin, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        if candidate_builtin != num_builtin
            || compare_expr(ctx, candidate_arg, den_arg) != Ordering::Equal
        {
            continue;
        }

        let remaining_factors = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = trig_quotient_match_from_cofactor(
            ctx,
            kind,
            den_arg,
            negative,
            &remaining_factors,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    None
}

fn trig_quotient_match_from_cofactor(
    ctx: &Context,
    kind: TrigQuotientTableKind,
    arg: ExprId,
    negative: bool,
    cofactor_factors: &[ExprId],
    var_name: &str,
) -> Option<TrigQuotientTableMatch> {
    let trace =
        polynomial_derivative_cofactor_trace(ctx, negative, cofactor_factors, arg, var_name)?;

    Some(TrigQuotientTableMatch {
        kind,
        arg,
        cofactor_display: trace.cofactor_display,
        cofactor_latex: trace.cofactor_latex,
        derivative_display: trace.derivative_display,
        derivative_latex: trace.derivative_latex,
        scale: trace.scale,
    })
}

pub(super) fn sqrt_chain_reciprocal_trig_direct_product(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    for (reciprocal_index, factor) in numerator_factors.iter().enumerate() {
        let Some((factor_builtin, factor_arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        let (kind, derivative_builtin, arg) = match (factor_builtin, factor_arg) {
            (BuiltinFn::Sec, arg) => (
                ReciprocalTrigDerivativeProductKind::SecantTangent,
                BuiltinFn::Tan,
                arg,
            ),
            (BuiltinFn::Csc, arg) => (
                ReciprocalTrigDerivativeProductKind::CosecantCotangent,
                BuiltinFn::Cot,
                arg,
            ),
            _ => continue,
        };

        for (derivative_index, factor) in numerator_factors.iter().enumerate() {
            if derivative_index == reciprocal_index {
                continue;
            }
            let Some((candidate_builtin, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
                continue;
            };
            if candidate_builtin != derivative_builtin
                || !same_sqrt_chain_arg(ctx, candidate_arg, arg)
            {
                continue;
            }

            let remaining_numerator = numerator_factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| {
                    (idx != reciprocal_index && idx != derivative_index).then_some(*factor)
                })
                .collect::<Vec<_>>();

            if let Some(table_match) = sqrt_chain_reciprocal_trig_match_from_cofactor(
                ctx,
                kind,
                arg,
                numerator_negative,
                &remaining_numerator,
                denominator_factors,
                var_name,
            ) {
                return Some(table_match);
            }
        }
    }

    None
}

pub(super) fn sqrt_chain_reciprocal_trig_raw_quotient(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((den_builtin, arg)) = trig_square_denominator_arg(ctx, *factor) else {
            continue;
        };
        let (kind, numerator_builtin) = match den_builtin {
            BuiltinFn::Cos => (
                ReciprocalTrigDerivativeProductKind::SecantTangent,
                BuiltinFn::Sin,
            ),
            BuiltinFn::Sin => (
                ReciprocalTrigDerivativeProductKind::CosecantCotangent,
                BuiltinFn::Cos,
            ),
            _ => continue,
        };

        for (numerator_index, factor) in numerator_factors.iter().enumerate() {
            let Some((candidate_builtin, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
                continue;
            };
            if candidate_builtin != numerator_builtin
                || !same_sqrt_chain_arg(ctx, candidate_arg, arg)
            {
                continue;
            }

            let remaining_numerator = numerator_factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
                .collect::<Vec<_>>();
            let remaining_denominator = denominator_factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
                .collect::<Vec<_>>();

            if let Some(table_match) = sqrt_chain_reciprocal_trig_match_from_cofactor(
                ctx,
                kind,
                arg,
                numerator_negative,
                &remaining_numerator,
                &remaining_denominator,
                var_name,
            ) {
                return Some(table_match);
            }
        }
    }

    None
}

fn sqrt_chain_reciprocal_trig_match_from_cofactor(
    ctx: &Context,
    kind: ReciprocalTrigDerivativeProductKind,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let trace = sqrt_chain_cofactor_derivative_trace_with_symbolic_scale(
        ctx,
        arg,
        numerator_negative,
        numerator_factors,
        denominator_factors,
        var_name,
    )?;

    Some(ReciprocalTrigDerivativeProductMatch {
        kind,
        arg,
        cofactor_display: trace.cofactor_display,
        cofactor_latex: trace.cofactor_latex,
        derivative_display: trace.derivative_display,
        derivative_latex: trace.derivative_latex,
        scale: trace.scale,
        symbolic_scale_display: trace.symbolic_scale_display,
        symbolic_scale_latex: trace.symbolic_scale_latex,
    })
}

pub(super) fn trig_square_denominator_arg(
    ctx: &Context,
    den: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some((base, exponent)) = as_pow(ctx, den) {
        let exponent = as_rational_const(ctx, exponent, 8)?;
        if exponent != BigRational::from_integer(2.into()) {
            return None;
        }
        let (builtin, arg) = unary_builtin_arg(ctx, base)?;
        if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
            return Some((builtin, arg));
        }
        return None;
    }

    let Expr::Mul(left, right) = ctx.get(den) else {
        return None;
    };
    let (left_builtin, left_arg) = unary_builtin_arg(ctx, *left)?;
    let (right_builtin, right_arg) = unary_builtin_arg(ctx, *right)?;
    if left_builtin == right_builtin
        && matches!(left_builtin, BuiltinFn::Sin | BuiltinFn::Cos)
        && compare_expr(ctx, left_arg, right_arg) == Ordering::Equal
    {
        return Some((left_builtin, left_arg));
    }
    None
}

pub(super) fn nested_trig_log_factor_arg(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
        return None;
    }

    match ctx.get(args[0]) {
        Expr::Function(inner_fn_id, inner_args) if inner_args.len() == 1 => {
            match ctx.builtin_of(*inner_fn_id)? {
                BuiltinFn::Tan | BuiltinFn::Cot => Some((expr, inner_args[0])),
                _ => None,
            }
        }
        Expr::Div(num, den) => {
            if let (Some((BuiltinFn::Sin, sin_arg)), Some((BuiltinFn::Cos, cos_arg))) =
                (unary_builtin_arg(ctx, *num), unary_builtin_arg(ctx, *den))
            {
                if compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                    return Some((expr, sin_arg));
                }
            }
            if let (Some((BuiltinFn::Cos, cos_arg)), Some((BuiltinFn::Sin, sin_arg))) =
                (unary_builtin_arg(ctx, *num), unary_builtin_arg(ctx, *den))
            {
                if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
                    return Some((expr, cos_arg));
                }
            }
            None
        }
        _ => None,
    }
}
