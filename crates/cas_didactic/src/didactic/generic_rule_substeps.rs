use super::SubStep;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(crate) fn generate_generic_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let Some(description) = generic_substep_description(step) else {
        return Vec::new();
    };

    let (before, after) = focused_expr_ids(step);
    vec![build_focus_substep(ctx, description, before, after)]
}

fn generic_substep_description(step: &Step) -> Option<&'static str> {
    match step.rule_name.as_str() {
        "Combine Like Terms" => {
            if step.description.contains("Cancel opposite terms") {
                None
            } else {
                Some("Agrupar términos semejantes y sumar coeficientes")
            }
        }
        "Pre-order Common Factor Cancel" => {
            Some("Identificar y cancelar el factor común en numerador y denominador")
        }
        "Pre-order Difference of Squares Cancel" => {
            Some("Aparece el mismo factor arriba y abajo, así que se cancela")
        }
        "Pythagorean Chain Identity" => Some("Sin²(u) y cos²(u) del mismo ángulo suman 1"),
        "Sqrt Perfect Square" => Some("Reconocer un cuadrado perfecto dentro de la raíz"),
        "Auto Expand Power Sum" => Some("Aplicar la fórmula (A + B)^2 = A^2 + 2AB + B^2"),
        "Inverse Tan Relations" => Some("Estas dos arctangentes suman pi/2"),
        _ => None,
    }
}

fn focused_expr_ids(step: &Step) -> (ExprId, ExprId) {
    match (step.before_local(), step.after_local()) {
        (Some(before), Some(after)) => (before, after),
        _ => (step.before, step.after),
    }
}

fn build_focus_substep(ctx: &Context, description: &str, before: ExprId, after: ExprId) -> SubStep {
    SubStep::new(
        description,
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))
}

fn display_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr,
        }
    )
}

fn latex_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex()
}
