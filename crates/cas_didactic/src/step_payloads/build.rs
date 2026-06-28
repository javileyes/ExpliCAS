mod expr;
mod latex;
mod substeps;

use cas_api_models::StepWire;
use cas_ast::{ordering::compare_expr, BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_nary;

pub(super) use expr::render_human_expr;

pub(super) fn build_step_wire(
    context: &Context,
    index: usize,
    enriched: &crate::didactic::EnrichedStep,
    language: cas_solver_core::eval_option_axes::Language,
) -> StepWire {
    let step = &enriched.base_step;
    let rendered_exprs = expr::render_step_wire_exprs(context, step);
    let rendered_latex = latex::render_step_wire_latex(context, step);
    let substeps = substeps::collect_step_wire_substeps(step, enriched, language);
    let visible_rule = inferred_step_rule(context, step, &substeps).unwrap_or_else(|| {
        crate::didactic::visible_rule_name_for_step(&step.rule_name, &step.description).to_string()
    });

    StepWire {
        index,
        rule: visible_rule,
        rule_latex: rendered_latex.rule_latex,
        before: rendered_exprs.before,
        after: rendered_exprs.after,
        before_latex: rendered_latex.before_latex,
        after_latex: rendered_latex.after_latex,
        substeps,
    }
}

fn inferred_step_rule(
    context: &Context,
    step: &crate::runtime::Step,
    substeps: &[cas_api_models::SubStepWire],
) -> Option<String> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if step.rule_name == "Simplify"
        && log_change_of_base_chain_expansion_len(context, before, after).is_some()
    {
        return Some("Expandir cambio de base".to_string());
    }

    let first_title = substeps.first()?.title.as_str();

    if step.rule_name == "Simplify"
        && matches!(
            first_title,
            "Usar n · ln(|u|) = ln(u^n) cuando n es par" | "Usar n · log_b(u) = log_b(u^n)"
        )
    {
        return Some("Meter el coeficiente dentro del logaritmo".to_string());
    }

    if matches!(step.rule_name.as_str(), "Simplify" | "Factorization")
        && matches!(
            first_title,
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)"
                | "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))"
        )
    {
        return Some("Descomponer en fracciones telescópicas".to_string());
    }

    if step.rule_name == "Simplify"
        && matches!(
            first_title,
            "Usar log_b(c) = log_a(c) · log_b(a)"
                | "Desplegar un logaritmo en una cadena de cambios de base"
        )
    {
        return Some("Expandir cambio de base".to_string());
    }

    if step.rule_name == "Simplify"
        && matches!(
            first_title,
            "Repartir el denominador entre los términos del numerador"
                | "Repartir el mismo denominador sobre cada término del numerador"
        )
    {
        return Some("Repartir el denominador común".to_string());
    }

    if matches!(
        first_title,
        "Usar log_b(a) · log_a(c) = log_b(c)" | "Encadenar los cambios de base intermedios"
    ) {
        return Some("Contraer cadena de logaritmos".to_string());
    }

    if first_title.starts_with("Reescribir el denominador sacando factor común ")
        || first_title.starts_with("Reescribir el numerador sacando factor común ")
    {
        return Some("Simplificar fracción anidada".to_string());
    }

    if matches!(
        step.rule_name.as_str(),
        "Simplify" | "Canonicalize" | "Expand Odd Half Power"
    ) && first_title == "Separar el radicando en una potencia par y un factor"
    {
        return Some("Extraer potencia par de la raíz".to_string());
    }

    let second_title = substeps.get(1)?.title.as_str();

    if matches!(
        step.rule_name.as_str(),
        "Canonicalize Negation"
            | "Rationalize"
            | "Rationalize Denominator"
            | "Rationalize Linear Sqrt Denominator"
    ) && !matches!(context.get(before), Expr::Div(_, _))
        && first_title.starts_with("Reconocer el patrón (a ")
        && second_title.starts_with("Aplicar (a ")
        && second_title.contains("= a^3")
    {
        return Some("Expandir la expresión".to_string());
    }

    if step.rule_name == "Simplify Nested Fraction" {
        if first_title.starts_with("Como ") && first_title.contains("aparece arriba y abajo") {
            return Some("Cancelar factor común".to_string());
        }

        if first_title.starts_with("Si ") && first_title.contains("queda una sola copia") {
            return Some("Cancelar un factor repetido".to_string());
        }
    }

    None
}

fn general_log_base_and_arg(ctx: &Context, expr: ExprId) -> Option<(Option<ExprId>, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Log) {
        return None;
    }
    match args.as_slice() {
        [arg] => Some((None, *arg)),
        [base, arg] => Some((Some(*base), *arg)),
        _ => None,
    }
}

fn log_change_of_base_chain_expansion_len(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<usize> {
    let Some((Some(target_base), target_arg)) = general_log_base_and_arg(ctx, before) else {
        return None;
    };

    let factors = expr_nary::mul_leaves(ctx, after);
    log_change_of_base_chain_len(ctx, &factors, target_base, target_arg)
}

fn log_change_of_base_chain_len(
    ctx: &Context,
    factors: &[ExprId],
    target_base: ExprId,
    target_arg: ExprId,
) -> Option<usize> {
    if factors.len() < 2 {
        return None;
    }

    let chain_nodes: Option<Vec<(ExprId, ExprId)>> = factors
        .iter()
        .map(|factor| {
            let (Some(base), arg) = general_log_base_and_arg(ctx, *factor)? else {
                return None;
            };
            Some((base, arg))
        })
        .collect();
    let chain_nodes = chain_nodes?;

    for start in 0..chain_nodes.len() {
        if compare_expr(ctx, chain_nodes[start].0, target_base) != std::cmp::Ordering::Equal {
            continue;
        }

        let mut used = vec![false; chain_nodes.len()];
        used[start] = true;
        if log_change_of_base_chain_dfs(ctx, &chain_nodes, start, target_arg, 1, &mut used) {
            return Some(chain_nodes.len());
        }
    }

    None
}

fn log_change_of_base_chain_dfs(
    ctx: &Context,
    nodes: &[(ExprId, ExprId)],
    current: usize,
    target_arg: ExprId,
    depth: usize,
    used: &mut [bool],
) -> bool {
    if depth == nodes.len() {
        return compare_expr(ctx, nodes[current].1, target_arg) == std::cmp::Ordering::Equal;
    }

    let current_arg = nodes[current].1;
    for next in 0..nodes.len() {
        if used[next] {
            continue;
        }
        if compare_expr(ctx, current_arg, nodes[next].0) != std::cmp::Ordering::Equal {
            continue;
        }
        used[next] = true;
        if log_change_of_base_chain_dfs(ctx, nodes, next, target_arg, depth + 1, used) {
            return true;
        }
        used[next] = false;
    }

    false
}
