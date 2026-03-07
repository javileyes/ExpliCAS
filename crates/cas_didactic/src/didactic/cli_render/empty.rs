use super::super::display_policy::StepDisplayMode;
use super::super::get_standalone_substeps;
use super::super::latex_plain_text::latex_to_plain_text;
use cas_ast::{Context, ExprId};

pub(super) fn render_empty_simplification_lines(
    ctx: &mut Context,
    expr: ExprId,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    let mut lines = Vec::new();
    let standalone_substeps = get_standalone_substeps(ctx, expr);

    if !standalone_substeps.is_empty() && display_mode != StepDisplayMode::Succinct {
        lines.push("Computation:".to_string());
        for sub in &standalone_substeps {
            lines.push(format!("   → {}", sub.description));
            if !sub.before_expr.is_empty() {
                lines.push(format!(
                    "     {} → {}",
                    latex_to_plain_text(&sub.before_expr),
                    latex_to_plain_text(&sub.after_expr)
                ));
            }
        }
    } else if display_mode != StepDisplayMode::Succinct {
        lines.push("No simplification steps needed.".to_string());
    }

    lines
}
