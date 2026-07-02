use crate::DisplayEvalSteps;
use cas_api_models::EngineWireStep;
use cas_ast::hold::strip_all_holds;
use cas_solver_core::rule_names::{
    RULE_CONSERVAR_DERIVADA_RESIDUAL, RULE_CONSERVAR_INTEGRAL_RESIDUAL,
    RULE_CONSERVAR_LIMITE_RESIDUAL,
};

pub(super) fn build_engine_wire_steps(
    ctx: &mut cas_ast::Context,
    steps: &DisplayEvalSteps,
    steps_enabled: bool,
) -> Vec<EngineWireStep> {
    if !steps_enabled {
        return Vec::new();
    }

    steps
        .iter()
        .map(|step| build_engine_wire_step(ctx, step))
        .collect()
}

fn build_engine_wire_step(ctx: &mut cas_ast::Context, step: &crate::Step) -> EngineWireStep {
    let normalize_expr = !matches!(
        step.rule_name.as_str(),
        RULE_CONSERVAR_DERIVADA_RESIDUAL
            | RULE_CONSERVAR_INTEGRAL_RESIDUAL
            | RULE_CONSERVAR_LIMITE_RESIDUAL
    );
    EngineWireStep {
        phase: "Simplify".into(),
        rule: step.rule_name.clone().into(),
        before: render_optional_expr(ctx, step.global_before, normalize_expr).unwrap_or_default(),
        after: render_optional_expr(ctx, step.global_after, normalize_expr).unwrap_or_default(),
        substeps: vec![],
    }
}

fn render_optional_expr(
    ctx: &mut cas_ast::Context,
    expr: Option<cas_ast::ExprId>,
    normalize_expr: bool,
) -> Option<String> {
    expr.map(|id| {
        let mut clean = strip_all_holds(ctx, id);
        if normalize_expr {
            clean = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(ctx, clean);
        }
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: clean
            }
        ))
    })
}
