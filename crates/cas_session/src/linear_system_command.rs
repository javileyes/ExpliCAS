//! Session-level orchestration for `solve_system` command rendering.

use cas_ast::{Context, Expr, ExprId, RelOp};
use cas_formatter::DisplayExpr;
use num_rational::BigRational;

#[derive(Debug, Clone)]
struct LinearSystemSpec {
    exprs: Vec<ExprId>,
    vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LinearSystemSpecError {
    InvalidPartCount,
    InvalidVariableName { name: String },
    ParseEquation { position: usize, message: String },
    ExpectedEquation { position: usize, input: String },
    UnsupportedRelation,
}

#[derive(Debug)]
struct LinearSystemCommandEvalOutput {
    vars: Vec<String>,
    result: cas_solver::LinSolveResult,
}

#[derive(Debug)]
enum LinearSystemCommandEvalError {
    Parse(LinearSystemSpecError),
    Solve(cas_solver::LinearSystemError),
}

fn is_valid_linear_system_var(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '_')
}

fn split_semicolon_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_i32;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            ';' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn parse_linear_system_spec(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemSpec, LinearSystemSpecError> {
    let parts: Vec<&str> = split_semicolon_top_level(input)
        .into_iter()
        .map(str::trim)
        .collect();

    if parts.len() < 4 || !parts.len().is_multiple_of(2) {
        return Err(LinearSystemSpecError::InvalidPartCount);
    }

    let n = parts.len() / 2;
    let eq_parts = &parts[0..n];
    let var_parts = &parts[n..2 * n];

    let mut vars = Vec::with_capacity(n);
    for var in var_parts {
        if !is_valid_linear_system_var(var) {
            return Err(LinearSystemSpecError::InvalidVariableName {
                name: (*var).to_string(),
            });
        }
        vars.push((*var).to_string());
    }

    let mut exprs = Vec::with_capacity(n);
    for (i, eq_str) in eq_parts.iter().enumerate() {
        match cas_parser::parse_statement(eq_str, ctx) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                if eq.op != RelOp::Eq {
                    return Err(LinearSystemSpecError::UnsupportedRelation);
                }
                exprs.push(ctx.add(Expr::Sub(eq.lhs, eq.rhs)));
            }
            Ok(cas_parser::Statement::Expression(_)) => {
                return Err(LinearSystemSpecError::ExpectedEquation {
                    position: i + 1,
                    input: (*eq_str).to_string(),
                });
            }
            Err(e) => {
                return Err(LinearSystemSpecError::ParseEquation {
                    position: i + 1,
                    message: e.to_string(),
                });
            }
        }
    }

    Ok(LinearSystemSpec { exprs, vars })
}

fn solve_linear_system_spec(
    ctx: &Context,
    spec: &LinearSystemSpec,
) -> Result<cas_solver::LinSolveResult, cas_solver::LinearSystemError> {
    let n = spec.vars.len();
    if spec.exprs.len() != n {
        return Err(cas_solver::LinearSystemError::NotLinear(
            "equation/variable count mismatch".to_string(),
        ));
    }

    match n {
        2 => {
            let (x, y) = match cas_solver::solve_2x2_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                &spec.vars[0],
                &spec.vars[1],
            ) {
                Ok(pair) => pair,
                Err(cas_solver::LinearSystemError::InfiniteSolutions) => {
                    return Ok(cas_solver::LinSolveResult::Infinite);
                }
                Err(cas_solver::LinearSystemError::NoSolution) => {
                    return Ok(cas_solver::LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(cas_solver::LinSolveResult::Unique(vec![x, y]))
        }
        3 => {
            let (x, y, z) = match cas_solver::solve_3x3_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                spec.exprs[2],
                &spec.vars[0],
                &spec.vars[1],
                &spec.vars[2],
            ) {
                Ok(triple) => triple,
                Err(cas_solver::LinearSystemError::InfiniteSolutions) => {
                    return Ok(cas_solver::LinSolveResult::Infinite);
                }
                Err(cas_solver::LinearSystemError::NoSolution) => {
                    return Ok(cas_solver::LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(cas_solver::LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = spec.vars.iter().map(String::as_str).collect();
            cas_solver::solve_nxn_linear_system(ctx, &spec.exprs, &var_refs)
        }
    }
}

fn evaluate_linear_system_command_input(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemCommandEvalOutput, LinearSystemCommandEvalError> {
    let spec = parse_linear_system_spec(ctx, input).map_err(LinearSystemCommandEvalError::Parse)?;
    let result =
        solve_linear_system_spec(ctx, &spec).map_err(LinearSystemCommandEvalError::Solve)?;
    Ok(LinearSystemCommandEvalOutput {
        vars: spec.vars,
        result,
    })
}

fn parse_linear_system_invocation_input(line: &str) -> String {
    let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();
    let inner = if rest.starts_with('(') && rest.ends_with(')') && rest.len() >= 2 {
        &rest[1..rest.len() - 1]
    } else {
        rest
    };
    inner.trim().to_string()
}

fn display_linear_system_solution(
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

fn format_linear_system_result_message(
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

fn format_linear_system_command_error_message(error: &LinearSystemCommandEvalError) -> String {
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

/// Evaluate full `solve_system ...` invocation and return CLI-ready message lines.
pub fn evaluate_linear_system_command_message(ctx: &mut Context, line: &str) -> String {
    let spec = parse_linear_system_invocation_input(line);
    match evaluate_linear_system_command_input(ctx, &spec) {
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

    #[test]
    fn parse_linear_system_invocation_input_accepts_parenthesized_form() {
        assert_eq!(
            super::parse_linear_system_invocation_input("solve_system(x+y=3; x-y=1; x; y)"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }

    #[test]
    fn parse_linear_system_invocation_input_accepts_space_form() {
        assert_eq!(
            super::parse_linear_system_invocation_input("solve_system x+y=3; x-y=1; x; y"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }
}
