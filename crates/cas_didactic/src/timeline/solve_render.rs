mod equation;
mod final_result;
mod step_html;

use crate::runtime::SolveStep;
use cas_ast::{Context, Equation, RelOp};

pub(super) fn render_equation_latex(context: &Context, equation: &Equation) -> String {
    equation::render_equation_latex(context, equation)
}

pub(super) fn render_solve_final_result_html(var: &str, solution_latex: &str) -> String {
    final_result::render_solve_final_result_html(var, solution_latex)
}

pub(super) fn render_solve_step_html(
    context: &Context,
    step_number: usize,
    step: &SolveStep,
) -> String {
    step_html::render_solve_step_html(context, step_number, step)
}

pub(super) fn solve_relop_to_latex(op: &RelOp) -> &'static str {
    match op {
        RelOp::Eq => "=",
        RelOp::Neq => "\\neq",
        RelOp::Lt => "<",
        RelOp::Gt => ">",
        RelOp::Leq => "\\leq",
        RelOp::Geq => "\\geq",
    }
}
