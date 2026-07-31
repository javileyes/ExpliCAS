//! `focused_rule_substeps`: familia `hyperbolic`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_hyperbolic_angle_sum_diff_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = recursive_hyperbolic_angle_sum_diff_substep(ctx, before, false) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_angle_sum_diff_formula(ctx, before)
    {
        return vec![schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some(substep) = recursive_hyperbolic_angle_sum_diff_substep(ctx, after, true) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_angle_sum_diff_formula(ctx, after)
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

fn recursive_hyperbolic_angle_sum_diff_substep(
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
        Some(BuiltinFn::Sinh) => {
            let title = format!(
                    "Usar sinh({previous}u+u) = sinh({previous}u) · cosh(u) + cosh({previous}u) · sinh(u), con u = {base_plain}"
                );
            let compact_plain = format!("sinh({multiple}u)");
            let expanded_plain =
                format!("sinh({previous}u) · cosh(u) + cosh({previous}u) · sinh(u)");
            let compact_latex = format!("\\sinh({multiple}u)");
            let expanded_latex =
                format!("\\sinh({previous}u)\\cdot\\cosh(u)+\\cosh({previous}u)\\cdot\\sinh(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        Some(BuiltinFn::Cosh) => {
            let title = format!(
                    "Usar cosh({previous}u+u) = cosh({previous}u) · cosh(u) + sinh({previous}u) · sinh(u), con u = {base_plain}"
                );
            let compact_plain = format!("cosh({multiple}u)");
            let expanded_plain =
                format!("cosh({previous}u) · cosh(u) + sinh({previous}u) · sinh(u)");
            let compact_latex = format!("\\cosh({multiple}u)");
            let expanded_latex =
                format!("\\cosh({previous}u)\\cdot\\cosh(u)+\\sinh({previous}u)\\cdot\\sinh(u)");
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

fn hyperbolic_angle_sum_diff_formula(
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
        (Some(BuiltinFn::Sinh), true) => Some((
            "Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)",
            "sinh(A+B)",
            "sinh(A) · cosh(B) + cosh(A) · sinh(B)",
            "\\sinh(A+B)",
            "\\sinh(A)\\cdot\\cosh(B)+\\cosh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Sinh), false) => Some((
            "Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)",
            "sinh(A-B)",
            "sinh(A) · cosh(B) - cosh(A) · sinh(B)",
            "\\sinh(A-B)",
            "\\sinh(A)\\cdot\\cosh(B)-\\cosh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Cosh), true) => Some((
            "Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)",
            "cosh(A+B)",
            "cosh(A) · cosh(B) + sinh(A) · sinh(B)",
            "\\cosh(A+B)",
            "\\cosh(A)\\cdot\\cosh(B)+\\sinh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Cosh), false) => Some((
            "Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)",
            "cosh(A-B)",
            "cosh(A) · cosh(B) - sinh(A) · sinh(B)",
            "\\cosh(A-B)",
            "\\cosh(A)\\cdot\\cosh(B)-\\sinh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Tanh), true) => Some((
            "Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
            "tanh(A+B)",
            "(tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
            "\\tanh(A+B)",
            "\\frac{\\tanh(A)+\\tanh(B)}{1+\\tanh(A)\\cdot\\tanh(B)}",
        )),
        (Some(BuiltinFn::Tanh), false) => Some((
            "Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
            "tanh(A-B)",
            "(tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
            "\\tanh(A-B)",
            "\\frac{\\tanh(A)-\\tanh(B)}{1-\\tanh(A)\\cdot\\tanh(B)}",
        )),
        _ => None,
    }
}

pub(super) fn generate_hyperbolic_product_to_sum_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_product_to_sum_formula(ctx, before)
    {
        return vec![schema_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_product_to_sum_formula(ctx, after)
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

fn hyperbolic_product_to_sum_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    hyperbolic_sum_to_product_formula(ctx, expr)
        .or_else(|| find_hyperbolic_product_to_sum_formula(ctx, expr))
}

fn hyperbolic_sum_to_product_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let (left, right, is_sum) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };
    let left_fn = extract_hyperbolic_function_name(ctx, left)?;
    let right_fn = extract_hyperbolic_function_name(ctx, right)?;
    if left_fn != right_fn {
        return None;
    }

    match (left_fn, is_sum) {
        ("sinh", true) => Some((
            "Usar sinh(A)+sinh(B) = 2·sinh((A+B)/2)·cosh((A-B)/2)",
            "sinh(A)+sinh(B)",
            "2·sinh((A+B)/2)·cosh((A-B)/2)",
            "\\sinh(A)+\\sinh(B)",
            "2\\cdot\\sinh((A+B)/2)\\cdot\\cosh((A-B)/2)",
        )),
        ("sinh", false) => Some((
            "Usar sinh(A)-sinh(B) = 2·cosh((A+B)/2)·sinh((A-B)/2)",
            "sinh(A)-sinh(B)",
            "2·cosh((A+B)/2)·sinh((A-B)/2)",
            "\\sinh(A)-\\sinh(B)",
            "2\\cdot\\cosh((A+B)/2)\\cdot\\sinh((A-B)/2)",
        )),
        ("cosh", true) => Some((
            "Usar cosh(A)+cosh(B) = 2·cosh((A+B)/2)·cosh((A-B)/2)",
            "cosh(A)+cosh(B)",
            "2·cosh((A+B)/2)·cosh((A-B)/2)",
            "\\cosh(A)+\\cosh(B)",
            "2\\cdot\\cosh((A+B)/2)\\cdot\\cosh((A-B)/2)",
        )),
        ("cosh", false) => Some((
            "Usar cosh(A)-cosh(B) = 2·sinh((A+B)/2)·sinh((A-B)/2)",
            "cosh(A)-cosh(B)",
            "2·sinh((A+B)/2)·sinh((A-B)/2)",
            "\\cosh(A)-\\cosh(B)",
            "2\\cdot\\sinh((A+B)/2)\\cdot\\sinh((A-B)/2)",
        )),
        _ => None,
    }
}

fn find_hyperbolic_product_to_sum_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    hyperbolic_product_to_sum_formula_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            find_hyperbolic_product_to_sum_formula(ctx, *left)
                .or_else(|| find_hyperbolic_product_to_sum_formula(ctx, *right))
        }
        _ => None,
    })
}

fn hyperbolic_product_to_sum_formula_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let factors = expr_nary::mul_leaves(ctx, expr);
    let has_double_factor = factors
        .iter()
        .any(|factor| is_integer_number(ctx, *factor, 2));
    if !has_double_factor {
        return None;
    }

    let mut sinh_count = 0;
    let mut cosh_count = 0;
    for factor in factors {
        match extract_hyperbolic_function_name(ctx, factor) {
            Some("sinh") => sinh_count += 1,
            Some("cosh") => cosh_count += 1,
            _ => {}
        }
    }

    match (sinh_count, cosh_count) {
        (2, 0) => Some((
            "Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)",
            "2·sinh(A)·sinh(B)",
            "cosh(A+B) - cosh(A-B)",
            "2\\cdot\\sinh(A)\\cdot\\sinh(B)",
            "\\cosh(A+B)-\\cosh(A-B)",
        )),
        (1, 1) => Some((
            "Usar 2·sinh(A)·cosh(B) = sinh(A+B) + sinh(A-B)",
            "2·sinh(A)·cosh(B)",
            "sinh(A+B) + sinh(A-B)",
            "2\\cdot\\sinh(A)\\cdot\\cosh(B)",
            "\\sinh(A+B)+\\sinh(A-B)",
        )),
        (0, 2) => Some((
            "Usar 2·cosh(A)·cosh(B) = cosh(A+B) + cosh(A-B)",
            "2·cosh(A)·cosh(B)",
            "cosh(A+B) + cosh(A-B)",
            "2\\cdot\\cosh(A)\\cdot\\cosh(B)",
            "\\cosh(A+B)+\\cosh(A-B)",
        )),
        _ => None,
    }
}

fn extract_hyperbolic_function_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*name, BuiltinFn::Sinh) {
        Some("sinh")
    } else if ctx.is_builtin(*name, BuiltinFn::Cosh) {
        Some("cosh")
    } else {
        None
    }
}

pub(super) fn generate_hyperbolic_half_angle_square_substeps(
    _ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let description = step.description.as_str();

    if description.contains("cosh(u/2)^2") && description.contains("(cosh(u) + 1) / 2") {
        return vec![SubStep::new(
            "Usar cosh²(u/2) = (cosh(u) + 1) / 2",
            "cosh²(u/2)",
            "(cosh(u) + 1) / 2",
        )];
    }

    if description.contains("sinh(u/2)^2") && description.contains("(cosh(u) - 1) / 2") {
        return vec![SubStep::new(
            "Usar sinh²(u/2) = (cosh(u) - 1) / 2",
            "sinh²(u/2)",
            "(cosh(u) - 1) / 2",
        )];
    }

    Vec::new()
}

pub(super) fn generate_hyperbolic_triple_angle_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((kind, call_expr, base_factors)) =
        find_nested_hyperbolic_triple_angle_call(ctx, before, false)
    else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let expansion = build_hyperbolic_triple_angle_expansion(&mut work, kind, &base_factors);
    let (before_plain, before_latex) = render_temp_expr(&work, call_expr);
    let (after_plain, after_latex) = render_temp_expr(&work, expansion);
    let title = format!(
        "Usar {} = {}",
        human_formula_title_plain(&before_plain),
        human_formula_title_plain(&after_plain)
    );

    vec![formula_substep(
        title,
        &before_plain,
        &after_plain,
        &before_latex,
        &after_latex,
    )]
}

fn find_nested_hyperbolic_triple_angle_call(
    ctx: &Context,
    expr: ExprId,
    is_nested: bool,
) -> Option<(HyperbolicTripleAngleKind, ExprId, Vec<ExprId>)> {
    if is_nested {
        if let Some(found) = hyperbolic_triple_angle_call_at_expr(ctx, expr) {
            return Some(found);
        }
    }

    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_nested_hyperbolic_triple_angle_call(ctx, *left, true)
            .or_else(|| find_nested_hyperbolic_triple_angle_call(ctx, *right, true)),
        Expr::Neg(inner) => find_nested_hyperbolic_triple_angle_call(ctx, *inner, true),
        _ => None,
    }
}

fn hyperbolic_triple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(HyperbolicTripleAngleKind, ExprId, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        HyperbolicTripleAngleKind::Sinh
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        HyperbolicTripleAngleKind::Cosh
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Tanh) {
        HyperbolicTripleAngleKind::Tanh
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple != 3 {
        return None;
    }
    Some((kind, expr, base_factors.into_vec()))
}

fn build_hyperbolic_triple_angle_expansion(
    ctx: &mut Context,
    kind: HyperbolicTripleAngleKind,
    base_factors: &[ExprId],
) -> ExprId {
    let u = build_balanced_mul(ctx, base_factors);
    let f_u = match kind {
        HyperbolicTripleAngleKind::Sinh => ctx.call_builtin(BuiltinFn::Sinh, vec![u]),
        HyperbolicTripleAngleKind::Cosh => ctx.call_builtin(BuiltinFn::Cosh, vec![u]),
        HyperbolicTripleAngleKind::Tanh => ctx.call_builtin(BuiltinFn::Tanh, vec![u]),
    };
    let square = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(f_u, two))
    };
    let cube = {
        let three = ctx.num(3);
        ctx.add(Expr::Pow(f_u, three))
    };

    match kind {
        HyperbolicTripleAngleKind::Sinh => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let linear = ctx.add(Expr::Mul(three, f_u));
            let cubic = ctx.add(Expr::Mul(four, cube));
            ctx.add(Expr::Add(linear, cubic))
        }
        HyperbolicTripleAngleKind::Cosh => {
            let four = ctx.num(4);
            let three = ctx.num(3);
            let cubic = ctx.add(Expr::Mul(four, cube));
            let linear = ctx.add(Expr::Mul(three, f_u));
            ctx.add(Expr::Sub(cubic, linear))
        }
        HyperbolicTripleAngleKind::Tanh => {
            let three_numerator = ctx.num(3);
            let three_denominator = ctx.num(3);
            let one = ctx.num(1);
            let linear = ctx.add(Expr::Mul(three_numerator, f_u));
            let numerator = ctx.add(Expr::Add(linear, cube));
            let quadratic = ctx.add(Expr::Mul(three_denominator, square));
            let denominator = ctx.add(Expr::Add(one, quadratic));
            ctx.add(Expr::Div(numerator, denominator))
        }
    }
}

pub(super) fn generate_hyperbolic_quotient_substeps(_ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description == "Recognize sinh(u) / cosh(u) as tanh(u)" {
        return Vec::new();
    }

    let (title, before_display, after_display, before_latex, after_latex) =
        if step.description.contains("Expand tanh") {
            (
                "Usar tanh(u) = sinh(u) / cosh(u)",
                "tanh(u)",
                "sinh(u) / cosh(u)",
                "\\tanh(u)",
                "\\frac{\\sinh(u)}{\\cosh(u)}",
            )
        } else {
            (
                "Usar sinh(u) / cosh(u) = tanh(u)",
                "sinh(u) / cosh(u)",
                "tanh(u)",
                "\\frac{\\sinh(u)}{\\cosh(u)}",
                "\\tanh(u)",
            )
        };

    vec![schema_substep(
        title,
        before_display,
        after_display,
        before_latex,
        after_latex,
    )]
}

pub(super) fn generate_inverse_hyperbolic_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(u) = change_of_base_natural_log_argument(ctx, after) else {
        return Vec::new();
    };

    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return Vec::new();
    }

    let ratio = args[0];
    if !matches!(ctx.get(ratio), Expr::Div(_, _)) {
        return Vec::new();
    }

    vec![SubStep::new(
        "Identificar el argumento como (u^2 - 1)/(u^2 + 1)",
        display_expr(ctx, ratio),
        format!("u = {}", display_expr(ctx, u)),
    )
    .with_before_latex(latex_expr(ctx, ratio))
    .with_after_latex(format!("u = {}", latex_expr(ctx, u)))]
}

pub(super) fn generate_hyperbolic_composition_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((outer_fn, inner_fn, argument)) = hyperbolic_composition_parts(ctx, before) else {
        return Vec::new();
    };
    if compare_expr(ctx, argument, after) != Ordering::Equal {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let u = work.var("u");
    let inner = work.call_builtin(inner_fn, vec![u]);
    let composition = work.call_builtin(outer_fn, vec![inner]);
    let outer_name = hyperbolic_fn_name(outer_fn);
    let inner_name = hyperbolic_fn_name(inner_fn);

    vec![
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
    ]
}

fn hyperbolic_composition_parts(
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
        (BuiltinFn::Sinh, BuiltinFn::Asinh)
            | (BuiltinFn::Cosh, BuiltinFn::Acosh)
            | (BuiltinFn::Tanh, BuiltinFn::Atanh)
            | (BuiltinFn::Asinh, BuiltinFn::Sinh)
            | (BuiltinFn::Acosh, BuiltinFn::Cosh)
            | (BuiltinFn::Atanh, BuiltinFn::Tanh)
    )
    .then_some((outer, inner, inner_args[0]))
}

fn hyperbolic_fn_name(function: BuiltinFn) -> &'static str {
    match function {
        BuiltinFn::Sinh => "sinh",
        BuiltinFn::Cosh => "cosh",
        BuiltinFn::Tanh => "tanh",
        BuiltinFn::Asinh => "asinh",
        BuiltinFn::Acosh => "acosh",
        BuiltinFn::Atanh => "atanh",
        _ => "función",
    }
}

pub(super) fn generate_hyperbolic_log_table_integration_substeps(
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
    let Some(table_match) = hyperbolic_log_sqrt_chain_integrand(ctx, args[0], var_name) else {
        return Vec::new();
    };

    let title = match table_match.kind {
        HyperbolicLogTableKind::Tanh => "Usar la regla de tanh(u) -> ln(cosh(u))",
        HyperbolicLogTableKind::ReciprocalTanh => "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
        HyperbolicLogTableKind::CoshOverSinh => "Usar la regla de cosh(u)/sinh(u) -> ln|sinh(u)|",
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
    push_integration_constant_factor_adjustment_substep(
        &mut substeps,
        IntegrationConstantFactorAdjustment {
            cofactor_display: &table_match.cofactor_display,
            cofactor_latex: &table_match.cofactor_latex,
            derivative_display: &table_match.derivative_display,
            derivative_latex: &table_match.derivative_latex,
            scale: &table_match.scale,
            symbolic_scale_display: table_match.symbolic_scale_display.as_deref(),
            symbolic_scale_latex: table_match.symbolic_scale_latex.as_deref(),
        },
    );

    substeps
}

pub(super) fn hyperbolic_log_sqrt_chain_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<HyperbolicLogTableMatch> {
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    hyperbolic_log_sqrt_chain_from_factors(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

pub(super) fn hyperbolic_log_sqrt_chain_from_factors(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<HyperbolicLogTableMatch> {
    let denominator_factors = collect_mul_chain_factors_readonly(ctx, den);

    for (tanh_index, factor) in numerator_factors.iter().enumerate() {
        let Some((BuiltinFn::Tanh, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };

        let remaining_numerator = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = hyperbolic_log_match_from_cofactor(
            ctx,
            HyperbolicLogTableKind::Tanh,
            arg,
            numerator_negative,
            &remaining_numerator,
            &denominator_factors,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    for (tanh_index, factor) in denominator_factors.iter().enumerate() {
        let Some((BuiltinFn::Tanh, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };

        let remaining_denominator = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = hyperbolic_log_match_from_cofactor(
            ctx,
            HyperbolicLogTableKind::ReciprocalTanh,
            arg,
            numerator_negative,
            numerator_factors,
            &remaining_denominator,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((BuiltinFn::Cosh, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };

        for (numerator_index, factor) in numerator_factors.iter().enumerate() {
            let Some((BuiltinFn::Sinh, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
                continue;
            };
            if !same_sqrt_chain_arg(ctx, candidate_arg, arg) {
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

            if let Some(table_match) = hyperbolic_log_match_from_cofactor(
                ctx,
                HyperbolicLogTableKind::Tanh,
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

    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((BuiltinFn::Sinh, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };

        for (numerator_index, factor) in numerator_factors.iter().enumerate() {
            let Some((BuiltinFn::Cosh, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
                continue;
            };
            if !same_sqrt_chain_arg(ctx, candidate_arg, arg) {
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

            if let Some(table_match) = hyperbolic_log_match_from_cofactor(
                ctx,
                HyperbolicLogTableKind::CoshOverSinh,
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

fn hyperbolic_log_match_from_cofactor(
    ctx: &Context,
    kind: HyperbolicLogTableKind,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicLogTableMatch> {
    if denominator_factors.is_empty() {
        if let Some(trace) = polynomial_derivative_cofactor_trace(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicLogTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }
        if let Some(trace) = polynomial_derivative_cofactor_trace_with_symbolic_scale(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicLogTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }
    }

    hyperbolic_log_sqrt_chain_match_from_cofactor(
        ctx,
        kind,
        arg,
        numerator_negative,
        numerator_factors,
        denominator_factors,
        var_name,
    )
}

fn hyperbolic_log_sqrt_chain_match_from_cofactor(
    ctx: &Context,
    kind: HyperbolicLogTableKind,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicLogTableMatch> {
    let trace = sqrt_chain_cofactor_derivative_trace(
        ctx,
        arg,
        numerator_negative,
        numerator_factors,
        denominator_factors,
        var_name,
    )?;

    Some(HyperbolicLogTableMatch {
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

pub(super) fn generate_hyperbolic_reciprocal_table_integration_substeps(
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
    let Some(table_match) = hyperbolic_reciprocal_sqrt_chain_integrand(ctx, args[0], var_name)
    else {
        return Vec::new();
    };

    let title = match table_match.kind {
        HyperbolicReciprocalTableKind::CoshSquare => "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        HyperbolicReciprocalTableKind::CoshFourth => {
            "Usar la regla de 1/cosh(u)^4 -> tanh(u) - tanh(u)^3/3"
        }
        HyperbolicReciprocalTableKind::SinhSquare => "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
        HyperbolicReciprocalTableKind::SinhFourth => {
            "Usar la regla de 1/sinh(u)^4 -> 1/tanh(u) - 1/(3*tanh(u)^3)"
        }
        HyperbolicReciprocalTableKind::SinhOverCoshSquare => {
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)"
        }
        HyperbolicReciprocalTableKind::CoshOverSinhSquare => {
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)"
        }
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
    push_integration_constant_factor_adjustment_substep(
        &mut substeps,
        IntegrationConstantFactorAdjustment {
            cofactor_display: &table_match.cofactor_display,
            cofactor_latex: &table_match.cofactor_latex,
            derivative_display: &table_match.derivative_display,
            derivative_latex: &table_match.derivative_latex,
            scale: &table_match.scale,
            symbolic_scale_display: table_match.symbolic_scale_display.as_deref(),
            symbolic_scale_latex: table_match.symbolic_scale_latex.as_deref(),
        },
    );

    substeps
}

pub(super) fn hyperbolic_reciprocal_sqrt_chain_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    hyperbolic_reciprocal_sqrt_chain_from_factors(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

pub(super) fn hyperbolic_reciprocal_sqrt_chain_from_factors(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    let denominator_factors = collect_mul_chain_factors_readonly(ctx, den);

    hyperbolic_reciprocal_square_match(
        ctx,
        numerator_negative,
        numerator_factors,
        &denominator_factors,
        var_name,
    )
    .or_else(|| {
        hyperbolic_reciprocal_fourth_match(
            ctx,
            numerator_negative,
            numerator_factors,
            &denominator_factors,
            var_name,
        )
    })
    .or_else(|| {
        hyperbolic_reciprocal_derivative_match(
            ctx,
            numerator_negative,
            numerator_factors,
            &denominator_factors,
            var_name,
        )
    })
}

fn hyperbolic_reciprocal_square_match(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((den_builtin, arg)) = hyperbolic_square_denominator_arg(ctx, *factor) else {
            continue;
        };
        let kind = match den_builtin {
            BuiltinFn::Cosh => HyperbolicReciprocalTableKind::CoshSquare,
            BuiltinFn::Sinh => HyperbolicReciprocalTableKind::SinhSquare,
            _ => continue,
        };

        let remaining_denominator = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = hyperbolic_reciprocal_match_from_cofactor(
            ctx,
            kind,
            arg,
            numerator_negative,
            numerator_factors,
            &remaining_denominator,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    None
}

fn hyperbolic_reciprocal_fourth_match(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((den_builtin, arg)) = hyperbolic_fourth_denominator_arg(ctx, *factor) else {
            continue;
        };
        let kind = match den_builtin {
            BuiltinFn::Cosh => HyperbolicReciprocalTableKind::CoshFourth,
            BuiltinFn::Sinh => HyperbolicReciprocalTableKind::SinhFourth,
            _ => continue,
        };

        let remaining_denominator = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(table_match) = hyperbolic_reciprocal_match_from_cofactor(
            ctx,
            kind,
            arg,
            numerator_negative,
            numerator_factors,
            &remaining_denominator,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    None
}

pub(super) fn hyperbolic_reciprocal_match_from_cofactor(
    ctx: &Context,
    kind: HyperbolicReciprocalTableKind,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    if denominator_factors.is_empty() {
        if let Some(trace) = polynomial_derivative_cofactor_trace(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicReciprocalTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }
        if let Some(trace) = polynomial_derivative_cofactor_trace_with_symbolic_scale(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicReciprocalTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }
        if let Some(trace) = symbolic_linear_exact_derivative_cofactor_trace(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicReciprocalTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: None,
                symbolic_scale_latex: None,
            });
        }
        if let Some(trace) = symbolic_linear_scaled_derivative_cofactor_trace(
            ctx,
            numerator_negative,
            numerator_factors,
            arg,
            var_name,
        ) {
            return Some(HyperbolicReciprocalTableMatch {
                kind,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: None,
                symbolic_scale_latex: None,
            });
        }
    }

    let trace = sqrt_chain_cofactor_derivative_trace_with_symbolic_scale(
        ctx,
        arg,
        numerator_negative,
        numerator_factors,
        denominator_factors,
        var_name,
    )?;

    Some(HyperbolicReciprocalTableMatch {
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

pub(super) fn hyperbolic_square_denominator_arg(
    ctx: &Context,
    den: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some((base, exponent)) = as_pow(ctx, den) {
        let exponent = as_rational_const(ctx, exponent, 8)?;
        if exponent != BigRational::from_integer(2.into()) {
            return None;
        }
        let (builtin, arg) = unary_builtin_arg(ctx, base)?;
        if matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) {
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
        && matches!(left_builtin, BuiltinFn::Sinh | BuiltinFn::Cosh)
        && compare_expr(ctx, left_arg, right_arg) == Ordering::Equal
    {
        return Some((left_builtin, left_arg));
    }
    None
}

fn hyperbolic_fourth_denominator_arg(ctx: &Context, den: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (base, exponent) = as_pow(ctx, den)?;
    let exponent = as_rational_const(ctx, exponent, 8)?;
    if exponent != BigRational::from_integer(4.into()) {
        return None;
    }
    let (builtin, arg) = unary_builtin_arg(ctx, base)?;
    matches!(builtin, BuiltinFn::Cosh | BuiltinFn::Sinh).then_some((builtin, arg))
}
