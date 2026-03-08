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
    for (index, step) in solve_steps.iter().enumerate() {
        render::push_solve_step_lines(&mut lines, ctx, renderer.as_ref(), index, step);
        if include_verbose_substeps && !step.substeps.is_empty() {
            substeps::push_verbose_substep_lines(&mut lines, ctx, index, &step.substeps);
        }
    }

    lines
}
