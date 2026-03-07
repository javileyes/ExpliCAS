use super::header::push_solve_header_lines;
use super::sections::{
    push_assumption_and_blocked_sections, push_requires_section, push_solve_steps_section,
};
use super::verify::push_solution_verification_lines;

pub fn format_solve_command_eval_lines(
    simplifier: &mut crate::Simplifier,
    var: &str,
    original_equation: Option<&cas_ast::Equation>,
    output: &crate::EvalOutputView,
    config: crate::SolveCommandRenderConfig,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    push_solve_header_lines(&mut lines, output, var);
    push_solve_steps_section(&mut lines, simplifier, output, config);

    lines.push(crate::format_solve_result_line(
        &simplifier.context,
        &output.result,
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
