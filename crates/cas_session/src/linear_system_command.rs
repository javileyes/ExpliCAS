//! Session-level orchestration for `solve_system` command rendering.

use cas_ast::Expr;
use cas_formatter::DisplayExpr;
use num_rational::BigRational;

fn display_linear_system_solution(
    ctx: &mut cas_ast::Context,
    vars: &[String],
    values: &[BigRational],
) -> String {
    let mut pairs = Vec::with_capacity(vars.len().min(values.len()));
    for (var, val) in vars.iter().zip(values.iter()) {
        let expr = ctx.add(Expr::Number(val.clone()));
        let shown = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: expr
            }
        );
        pairs.push(format!("{var} = {shown}"));
    }
    format!("{{ {} }}", pairs.join(", "))
}

fn format_not_linear_reply(message: &str) -> String {
    let with_prefix = |detail: &str| {
        if detail.contains("non-linear") {
            detail.to_string()
        } else {
            format!("non-linear term: {detail}")
        }
    };

    if let Some((eq, detail)) = message
        .strip_prefix("equation ")
        .and_then(|rest| rest.split_once(": "))
    {
        format!(
            "Error in equation {eq}: {}\nsolve_system() only handles linear equations.",
            with_prefix(detail)
        )
    } else {
        format!(
            "Error: {}\nsolve_system() only handles linear equations.",
            with_prefix(message)
        )
    }
}

fn format_linear_system_result_message(
    ctx: &mut cas_ast::Context,
    output: &cas_solver::LinearSystemCommandEvalOutput,
) -> String {
    match &output.result {
        cas_solver::LinSolveResult::Unique(solution) => {
            display_linear_system_solution(ctx, &output.vars, solution)
        }
        cas_solver::LinSolveResult::Infinite => "System has infinitely many solutions.\n\
                 The equations are dependent."
            .to_string(),
        cas_solver::LinSolveResult::Inconsistent => "System has no solution.\n\
                 The equations are inconsistent."
            .to_string(),
    }
}

fn format_linear_system_command_error_message(
    error: &cas_solver::LinearSystemCommandEvalError,
) -> String {
    match error {
        cas_solver::LinearSystemCommandEvalError::Parse(
            cas_solver::LinearSystemSpecError::InvalidPartCount,
        ) => "Usage:\n  \
                         2x2: solve_system(eq1; eq2; x; y)\n  \
                         3x3: solve_system(eq1; eq2; eq3; x; y; z)\n  \
                         nxn: solve_system(eq1; ...; eqn; x1; ...; xn)\n\n\
                         Examples:\n  \
                         solve_system(x+y=3; x-y=1; x; y)\n  \
                         solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)"
            .to_string(),
        cas_solver::LinearSystemCommandEvalError::Parse(
            cas_solver::LinearSystemSpecError::InvalidVariableName { name, .. },
        ) => format!(
            "Invalid variable name: '{name}'\n\
                         Variables must be simple identifiers."
        ),
        cas_solver::LinearSystemCommandEvalError::Parse(
            cas_solver::LinearSystemSpecError::ParseEquation { position, message },
        ) => format!("Error parsing equation {position}: {message}"),
        cas_solver::LinearSystemCommandEvalError::Parse(
            cas_solver::LinearSystemSpecError::ExpectedEquation { position, input },
        ) => format!(
            "Expected equation, got expression in position {position}: '{input}'\n\
                         Use '=' to create an equation."
        ),
        cas_solver::LinearSystemCommandEvalError::Parse(
            cas_solver::LinearSystemSpecError::UnsupportedRelation { .. },
        ) => "solve_system(): only '=' equations supported\n\
                         Inequalities (<, >, <=, >=, !=) are not supported."
            .to_string(),
        cas_solver::LinearSystemCommandEvalError::Solve(
            cas_solver::LinearSystemError::NotLinear(message),
        ) => format_not_linear_reply(message),
        cas_solver::LinearSystemCommandEvalError::Solve(e) => {
            format!("Error solving system: {e}")
        }
    }
}

/// Evaluate full `solve_system ...` invocation and return CLI-ready message lines.
pub fn evaluate_linear_system_command_message(ctx: &mut cas_ast::Context, line: &str) -> String {
    let spec = cas_solver::parse_linear_system_invocation_input(line);
    match cas_solver::evaluate_linear_system_command_input(ctx, &spec) {
        Ok(output) => format_linear_system_result_message(ctx, &output),
        Err(error) => format_linear_system_command_error_message(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::display_linear_system_solution;
    use num_rational::BigRational;

    #[test]
    fn evaluate_linear_system_command_message_solves_2x2() {
        let mut ctx = cas_ast::Context::new();
        let shown = super::evaluate_linear_system_command_message(
            &mut ctx,
            "solve_system(x+y=3; x-y=1; x; y)",
        );
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }

    #[test]
    fn evaluate_linear_system_command_message_reports_usage() {
        let mut ctx = cas_ast::Context::new();
        let shown = super::evaluate_linear_system_command_message(&mut ctx, "solve_system x+y=3");
        assert!(shown.contains("Usage:"));
    }

    #[test]
    fn display_linear_system_solution_formats_pairs() {
        let mut ctx = cas_ast::Context::new();
        let vars = vec!["x".to_string(), "y".to_string()];
        let values = vec![
            BigRational::from_integer(2.into()),
            BigRational::from_integer(1.into()),
        ];
        let shown = display_linear_system_solution(&mut ctx, &vars, &values);
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }
}
