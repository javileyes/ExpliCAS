use cas_ast::Context;
use num_rational::BigRational;

use super::super::display_linear_system_solution;

pub(super) fn format_unique_result(
    ctx: &mut Context,
    vars: &[String],
    solution: &[BigRational],
) -> String {
    display_linear_system_solution(ctx, vars, solution)
}
