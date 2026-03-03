use cas_ast::{Context, Expr, ExprId, RelOp};

use crate::{LinSolveResult, LinearSystemError};

#[derive(Debug, Clone)]
struct LinearSystemSpec {
    exprs: Vec<ExprId>,
    vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearSystemSpecError {
    InvalidPartCount,
    InvalidVariableName { position: usize, name: String },
    ParseEquation { position: usize, message: String },
    ExpectedEquation { position: usize, input: String },
    UnsupportedRelation { position: usize, relation: String },
}

#[derive(Debug)]
pub struct LinearSystemCommandEvalOutput {
    pub vars: Vec<String>,
    pub result: LinSolveResult,
}

#[derive(Debug)]
pub enum LinearSystemCommandEvalError {
    Parse(LinearSystemSpecError),
    Solve(LinearSystemError),
}

fn is_valid_linear_system_var(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '_')
}

fn split_semicolon_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
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
    for (i, var) in var_parts.iter().enumerate() {
        if !is_valid_linear_system_var(var) {
            return Err(LinearSystemSpecError::InvalidVariableName {
                position: i + 1,
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
                    return Err(LinearSystemSpecError::UnsupportedRelation {
                        position: i + 1,
                        relation: format!("{}", eq.op),
                    });
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
) -> Result<LinSolveResult, LinearSystemError> {
    let n = spec.vars.len();
    if spec.exprs.len() != n {
        return Err(LinearSystemError::NotLinear(
            "equation/variable count mismatch".to_string(),
        ));
    }

    match n {
        2 => {
            let (x, y) = match crate::solve_2x2_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                &spec.vars[0],
                &spec.vars[1],
            ) {
                Ok(pair) => pair,
                Err(LinearSystemError::InfiniteSolutions) => {
                    return Ok(LinSolveResult::Infinite);
                }
                Err(LinearSystemError::NoSolution) => {
                    return Ok(LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(LinSolveResult::Unique(vec![x, y]))
        }
        3 => {
            let (x, y, z) = match crate::solve_3x3_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                spec.exprs[2],
                &spec.vars[0],
                &spec.vars[1],
                &spec.vars[2],
            ) {
                Ok(triple) => triple,
                Err(LinearSystemError::InfiniteSolutions) => {
                    return Ok(LinSolveResult::Infinite);
                }
                Err(LinearSystemError::NoSolution) => {
                    return Ok(LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = spec.vars.iter().map(String::as_str).collect();
            crate::solve_nxn_linear_system(ctx, &spec.exprs, &var_refs)
        }
    }
}

pub fn evaluate_linear_system_command_input(
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

pub fn parse_linear_system_invocation_input(line: &str) -> String {
    let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();
    let inner = if rest.starts_with('(') && rest.ends_with(')') && rest.len() >= 2 {
        &rest[1..rest.len() - 1]
    } else {
        rest
    };
    inner.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_linear_system_command_input, parse_linear_system_invocation_input,
        LinearSystemCommandEvalError, LinearSystemSpecError,
    };
    use num_rational::BigRational;

    #[test]
    fn parse_linear_system_invocation_input_accepts_parenthesized_form() {
        assert_eq!(
            parse_linear_system_invocation_input("solve_system(x+y=3; x-y=1; x; y)"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }

    #[test]
    fn parse_linear_system_invocation_input_accepts_space_form() {
        assert_eq!(
            parse_linear_system_invocation_input("solve_system x+y=3; x-y=1; x; y"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }

    #[test]
    fn evaluate_linear_system_command_input_rejects_inequality() {
        let mut ctx = cas_ast::Context::new();
        let err = evaluate_linear_system_command_input(&mut ctx, "x+y<3; x-y=1; x; y")
            .expect_err("expected error");
        assert!(matches!(
            err,
            LinearSystemCommandEvalError::Parse(LinearSystemSpecError::UnsupportedRelation {
                position: 1,
                ..
            })
        ));
    }

    #[test]
    fn evaluate_linear_system_command_input_solves_2x2() {
        let mut ctx = cas_ast::Context::new();
        let result =
            evaluate_linear_system_command_input(&mut ctx, "x+y=3; x-y=1; x; y").expect("solve");
        assert_eq!(result.vars, vec!["x".to_string(), "y".to_string()]);
        match result.result {
            crate::LinSolveResult::Unique(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], BigRational::from_integer(2.into()));
                assert_eq!(values[1], BigRational::from_integer(1.into()));
            }
            _ => panic!("expected unique solution"),
        }
    }
}
