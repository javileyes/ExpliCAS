use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;
use num_rational::BigRational;

use crate::linear_system_types::{
    LinearSystemCommandEvalError, LinearSystemCommandEvalOutput, LinearSystemSpecError,
};

pub(crate) fn display_linear_system_solution(
    ctx: &mut Context,
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

pub(crate) fn format_linear_system_result_message(
    ctx: &mut Context,
    output: &LinearSystemCommandEvalOutput,
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

pub(crate) fn format_linear_system_command_error_message(
    error: &LinearSystemCommandEvalError,
) -> String {
    match error {
        LinearSystemCommandEvalError::Parse(LinearSystemSpecError::InvalidPartCount) => {
            "Usage:\n  \
                         2x2: solve_system(eq1; eq2; x; y)\n  \
                         3x3: solve_system(eq1; eq2; eq3; x; y; z)\n  \
                         nxn: solve_system(eq1; ...; eqn; x1; ...; xn)\n\n\
                         Examples:\n  \
                         solve_system(x+y=3; x-y=1; x; y)\n  \
                         solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)"
                .to_string()
        }
        LinearSystemCommandEvalError::Parse(LinearSystemSpecError::InvalidVariableName {
            name,
        }) => {
            format!(
                "Invalid variable name: '{name}'\n\
                         Variables must be simple identifiers."
            )
        }
        LinearSystemCommandEvalError::Parse(LinearSystemSpecError::ParseEquation {
            position,
            message,
        }) => format!("Error parsing equation {position}: {message}"),
        LinearSystemCommandEvalError::Parse(LinearSystemSpecError::ExpectedEquation {
            position,
            input,
        }) => format!(
            "Expected equation, got expression in position {position}: '{input}'\n\
                         Use '=' to create an equation."
        ),
        LinearSystemCommandEvalError::Parse(LinearSystemSpecError::UnsupportedRelation) => {
            "solve_system(): only '=' equations supported\n\
                         Inequalities (<, >, <=, >=, !=) are not supported."
                .to_string()
        }
        LinearSystemCommandEvalError::Solve(cas_solver::LinearSystemError::NotLinear(message)) => {
            format_not_linear_reply(message)
        }
        LinearSystemCommandEvalError::Solve(e) => {
            format!("Error solving system: {e}")
        }
    }
}
