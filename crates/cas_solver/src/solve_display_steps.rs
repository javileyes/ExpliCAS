mod render;
mod substeps;

use cas_ast::Context;
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};

pub fn format_solve_steps_lines(
    ctx: &Context,
    solve_steps: &[crate::SolveStep],
    output_scopes: &[ScopeTag],
    include_verbose_substeps: bool,
) -> Vec<String> {
    let filtered: Vec<_> = solve_steps
        .iter()
        .filter(|step| step.importance >= crate::ImportanceLevel::Medium)
        .collect();

    if filtered.is_empty() {
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
    for (index, step) in filtered.iter().enumerate() {
        render::push_solve_step_lines(&mut lines, ctx, renderer.as_ref(), index, step);
        if include_verbose_substeps && !step.substeps.is_empty() {
            substeps::push_verbose_substep_lines(&mut lines, ctx, index, &step.substeps);
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::format_solve_steps_lines;
    use cas_ast::{Context, Equation, RelOp};

    #[test]
    fn format_solve_steps_lines_skips_low_importance_noise() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);

        let noisy = crate::SolveStep::new(
            "Canonicalize multiplication",
            Equation {
                lhs: x,
                rhs: one,
                op: RelOp::Eq,
            },
            crate::ImportanceLevel::Low,
        );
        let meaningful = crate::SolveStep::new(
            "Take logarithms",
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            crate::ImportanceLevel::High,
        );

        let lines = format_solve_steps_lines(&ctx, &[noisy, meaningful], &[], false);

        assert!(
            lines.iter().any(|line| line.contains("Take logarithms")),
            "lines: {lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("Canonicalize")),
            "lines: {lines:?}"
        );
    }
}
