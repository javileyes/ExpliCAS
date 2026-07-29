mod conditional;
mod discrete;
mod intervals;
mod residual;

use cas_ast::{Context, SolutionSet};

/// `style` lleva la notación de raíz del ECO (la misma decisión que el texto,
/// `result_root_style`): sin ella este camino renderizaba con el default
/// (radical incondicional) y las dos superficies del MISMO conjunto divergían —
/// `solve(x^(2/3)>2, x)` daba `-(2^(3/2))` en texto y `\sqrt{2^3}` en LaTeX.
pub(crate) fn solution_set_to_output_latex(
    ctx: &Context,
    solution_set: &SolutionSet,
    style: &cas_formatter::StylePreferences,
) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => discrete::render_discrete_solution_set(ctx, exprs, style),
        SolutionSet::Conditional(cases) => {
            conditional::render_conditional_solution_set(ctx, cases, style)
        }
        SolutionSet::Continuous(interval) => {
            intervals::render_continuous_interval(ctx, interval, style)
        }
        SolutionSet::Union(intervals) => intervals::render_interval_union(ctx, intervals, style),
        SolutionSet::Residual(expr) => residual::render_residual_solution(ctx, *expr, style),
        SolutionSet::Periodic { bases, period } => {
            cas_formatter::latex_periodic_family_styled(ctx, bases, *period, style)
        }
        SolutionSet::PeriodicIntervalUnion { windows, period } => {
            cas_formatter::latex_periodic_interval_union_styled(ctx, windows, *period, style)
        }
    }
}
