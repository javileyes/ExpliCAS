use cas_ast::Context;
use cas_formatter::display_transforms::ScopedRenderer;

pub(super) fn push_solve_step_lines(
    lines: &mut Vec<String>,
    ctx: &Context,
    renderer: Option<&ScopedRenderer<'_>>,
    index: usize,
    step: &crate::SolveStep,
) {
    lines.push(format!("{}. {}", index + 1, step.description));

    let (lhs_str, rhs_str) = if let Some(renderer) = renderer {
        (
            renderer.render(step.equation_after.lhs),
            renderer.render(step.equation_after.rhs),
        )
    } else {
        (
            cas_formatter::DisplayExpr {
                context: ctx,
                id: step.equation_after.lhs,
            }
            .to_string(),
            cas_formatter::DisplayExpr {
                context: ctx,
                id: step.equation_after.rhs,
            }
            .to_string(),
        )
    };

    lines.push(format!(
        "   -> {} {} {}",
        lhs_str, step.equation_after.op, rhs_str
    ));
}
