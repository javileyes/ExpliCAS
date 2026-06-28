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
    let visible_rule = inferred_step_rule(context, step, enriched).unwrap_or_else(|| {
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

// The step-naming heuristics match on the SOURCE (Spanish) sub-step descriptions, not the rendered
// wire titles: descriptions are language-independent (a keyed sub-step keeps its Spanish description
// and only the wire title is translated), so the inferred step name is identical across languages
// and the `Es` output stays byte-for-byte unchanged.
fn inferred_step_rule(
    context: &Context,
    step: &crate::runtime::Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Option<String> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if step.rule_name == "Simplify"
        && log_change_of_base_chain_expansion_len(context, before, after).is_some()
    {
        return Some("Expandir cambio de base".to_string());
    }

    // Matrix operations carry no sub-steps, so name them from the `before` expression here (before
    // the sub-step gate below): replace the generic "Operación con matrices" with the specific
    // operation, and split the misleading combined "Potencia o inversa de la matriz".
    if step.rule_name == "Matrix Functions" {
        if let Some(name) = matrix_function_step_rule(context, before) {
            return Some(name.to_string());
        }
    }
    if step.rule_name == "Matrix Reciprocal/Inverse" {
        if let Some(name) = matrix_power_or_inverse_step_rule(context, before) {
            return Some(name.to_string());
        }
    }

    let first_title = enriched.sub_steps.first()?.description.as_str();

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

    let second_title = enriched.sub_steps.get(1)?.description.as_str();

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

/// Specific Spanish name for a matrix-function step (`det`, `trace`, …). Returns `None` for any
/// function we do not name explicitly, so it falls back to the generic "Operación con matrices".
fn matrix_function_step_rule(ctx: &Context, before: ExprId) -> Option<&'static str> {
    let Expr::Function(fn_id, _) = ctx.get(before) else {
        return None;
    };
    Some(match ctx.sym_name(*fn_id) {
        "det" | "determinant" => "Calcular el determinante de la matriz",
        "trace" => "Calcular la traza de la matriz",
        "transpose" => "Transponer la matriz",
        "inverse" | "inv" => "Calcular la inversa de la matriz",
        "charpoly" => "Calcular el polinomio característico",
        "rref" => "Reducir la matriz a forma escalonada",
        "eigenvalues" | "eigvals" => "Calcular los autovalores",
        "eigenvectors" => "Calcular los autovectores",
        "rank" => "Calcular el rango de la matriz",
        "dot" => "Calcular el producto escalar",
        "cross" => "Calcular el producto vectorial",
        "linsolve" => "Resolver el sistema lineal",
        _ => return None,
    })
}

/// Disambiguate the combined matrix power/inverse rule by the actual exponent: a negative power is
/// the inverse, a positive power is repeated multiplication, and `c / M` multiplies by the inverse.
fn matrix_power_or_inverse_step_rule(ctx: &Context, before: ExprId) -> Option<&'static str> {
    match ctx.get(before) {
        Expr::Pow(_, exponent) => {
            let negative = cas_math::numeric_eval::as_rational_const(ctx, *exponent)
                .is_some_and(|value| num_traits::Signed::is_negative(&value));
            Some(if negative {
                "Calcular la inversa de la matriz"
            } else {
                "Elevar la matriz a una potencia"
            })
        }
        Expr::Div(_, _) => Some("Multiplicar por la inversa de la matriz"),
        _ => None,
    }
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

#[cfg(test)]
mod matrix_rule_name_tests {
    use super::{matrix_function_step_rule, matrix_power_or_inverse_step_rule};
    use cas_ast::{Context, ExprId};
    use cas_parser::parse;

    fn name_of(
        src: &str,
        namer: fn(&Context, ExprId) -> Option<&'static str>,
    ) -> Option<&'static str> {
        let mut ctx = Context::new();
        let expr = parse(src, &mut ctx).expect("parse");
        namer(&ctx, expr)
    }

    #[test]
    fn matrix_functions_named_specifically() {
        assert_eq!(
            name_of("det([[1,2],[3,4]])", matrix_function_step_rule),
            Some("Calcular el determinante de la matriz")
        );
        assert_eq!(
            name_of("trace([[1,2],[3,4]])", matrix_function_step_rule),
            Some("Calcular la traza de la matriz")
        );
        assert_eq!(
            name_of("transpose([[1,2],[3,4]])", matrix_function_step_rule),
            Some("Transponer la matriz")
        );
        assert_eq!(name_of("foo(x)", matrix_function_step_rule), None);
    }

    #[test]
    fn matrix_power_split_from_inverse() {
        assert_eq!(
            name_of("([[1,1],[0,1]])^3", matrix_power_or_inverse_step_rule),
            Some("Elevar la matriz a una potencia")
        );
        assert_eq!(
            name_of("([[1,2],[3,4]])^(-1)", matrix_power_or_inverse_step_rule),
            Some("Calcular la inversa de la matriz")
        );
    }
}
