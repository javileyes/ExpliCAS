use cas_ast::Context;
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};

pub fn format_solve_steps_lines(
    ctx: &Context,
    solve_steps: &[crate::SolveStep],
    output_scopes: &[ScopeTag],
    include_verbose_substeps: bool,
) -> Vec<String> {
    if solve_steps.is_empty() {
        return Vec::new();
    }

    let registry = DisplayTransformRegistry::with_defaults();
    let style = cas_formatter::root_style::StylePreferences::default();
    let renderer = if output_scopes.is_empty() {
        None
    } else {
        Some(ScopedRenderer::new(ctx, output_scopes, &registry, &style))
    };

    let mut lines = Vec::new();
    for (i, step) in solve_steps.iter().enumerate() {
        lines.push(format!("{}. {}", i + 1, step.description));

        let (lhs_str, rhs_str) = if let Some(ref r) = renderer {
            (
                r.render(step.equation_after.lhs),
                r.render(step.equation_after.rhs),
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

        if include_verbose_substeps && !step.substeps.is_empty() {
            for (j, substep) in step.substeps.iter().enumerate() {
                let sub_lhs = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.lhs,
                }
                .to_string();
                let sub_rhs = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: substep.equation_after.rhs,
                }
                .to_string();
                lines.push(format!(
                    "      {}.{}. {}",
                    i + 1,
                    j + 1,
                    substep.description
                ));
                lines.push(format!(
                    "          -> {} {} {}",
                    sub_lhs, substep.equation_after.op, sub_rhs
                ));
            }
        }
    }

    lines
}
