use super::header::push_solve_header_lines;
use super::sections::{
    push_assumption_and_blocked_sections, push_requires_section, push_solve_steps_section,
};
use super::verify::push_solution_verification_lines;

pub(crate) fn format_solve_command_eval_lines(
    simplifier: &mut crate::Simplifier,
    var: &str,
    original_equation: Option<&cas_ast::Equation>,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    push_solve_header_lines(&mut lines, output, var);
    push_solve_steps_section(&mut lines, simplifier, output, config);

    // `semantics set numeric decimal`: present the solution set decimally
    // at this output boundary (members and interval bounds only — the
    // solve pipeline above ran exact).
    let mut presented_result = output.result.clone();
    if config.numeric_display == crate::NumericDisplayMode::Decimal {
        let complex_enabled =
            config.value_domain == cas_solver_core::value_domain::ValueDomain::ComplexEnabled;
        let ctx = &mut simplifier.context;
        let present = |ctx: &mut cas_ast::Context, id: cas_ast::ExprId| {
            cas_math::numeric_presentation::present_numeric(ctx, id, complex_enabled).unwrap_or(id)
        };
        match &mut presented_result {
            crate::EvalResult::Expr(id) => {
                *id = present(ctx, *id);
            }
            crate::EvalResult::Set(ids) => {
                for id in ids.iter_mut() {
                    *id = present(ctx, *id);
                }
            }
            crate::EvalResult::SolutionSet(set) => match set {
                cas_ast::SolutionSet::Discrete(ids) => {
                    for id in ids.iter_mut() {
                        *id = present(ctx, *id);
                    }
                }
                cas_ast::SolutionSet::Continuous(interval) => {
                    interval.min = present(ctx, interval.min);
                    interval.max = present(ctx, interval.max);
                }
                cas_ast::SolutionSet::Union(intervals) => {
                    for interval in intervals.iter_mut() {
                        interval.min = present(ctx, interval.min);
                        interval.max = present(ctx, interval.max);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    lines.push(crate::format_solve_result_line(
        &simplifier.context,
        &presented_result,
        &output.output_scopes,
    ));

    push_requires_section(&mut lines, simplifier, output, config);
    push_solution_verification_lines(
        &mut lines,
        simplifier,
        original_equation,
        output,
        config,
        var,
    );
    push_assumption_and_blocked_sections(&mut lines, simplifier, output, config);

    lines
}
