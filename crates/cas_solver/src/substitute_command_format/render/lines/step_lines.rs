use crate::substitute_command_types::{SubstituteEvalOutput, SubstituteRenderMode};

use super::super::filter::should_render_substitute_step;

pub(super) fn append_step_lines(
    lines: &mut Vec<String>,
    context: &cas_ast::Context,
    output: &SubstituteEvalOutput,
    mode: SubstituteRenderMode,
) {
    if mode == SubstituteRenderMode::None || output.steps.is_empty() {
        return;
    }

    if mode != SubstituteRenderMode::Succinct {
        lines.push("Steps:".to_string());
    }
    for step in &output.steps {
        if should_render_substitute_step(step, mode) {
            if mode == SubstituteRenderMode::Succinct {
                lines.push(format!(
                    "-> {}",
                    cas_formatter::DisplayExpr {
                        context,
                        id: step.global_after.unwrap_or(step.after)
                    }
                ));
            } else {
                lines.push(format!("  {}  [{}]", step.description, step.rule_name));
            }
        }
    }
}
