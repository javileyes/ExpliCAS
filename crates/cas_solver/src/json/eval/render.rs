use crate::{DisplayEvalSteps, EvalResult};
use cas_api_models::EngineJsonStep;
use cas_ast::hold::strip_all_holds;

pub fn render_eval_result(ctx: &mut cas_ast::Context, result: &EvalResult) -> String {
    match result {
        EvalResult::Expr(e) => {
            let clean = strip_all_holds(ctx, *e);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: clean
                }
            )
        }
        EvalResult::Set(v) if !v.is_empty() => {
            let clean = strip_all_holds(ctx, v[0]);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: clean
                }
            )
        }
        EvalResult::SolutionSet(solution_set) => crate::display_solution_set(ctx, solution_set),
        EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    }
}

pub fn build_engine_json_steps(
    ctx: &mut cas_ast::Context,
    steps: &DisplayEvalSteps,
    steps_enabled: bool,
) -> Vec<EngineJsonStep> {
    if !steps_enabled {
        return Vec::new();
    }

    steps
        .iter()
        .map(|step| {
            let before_str = step.global_before.map(|id| {
                let clean = strip_all_holds(ctx, id);
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: clean
                    }
                )
            });
            let after_str = step.global_after.map(|id| {
                let clean = strip_all_holds(ctx, id);
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: clean
                    }
                )
            });
            EngineJsonStep {
                phase: "Simplify".to_string(),
                rule: step.rule_name.clone(),
                before: before_str.unwrap_or_default(),
                after: after_str.unwrap_or_default(),
                substeps: vec![],
            }
        })
        .collect()
}
