use cas_ast::{Context, Expr};

use crate::linear_system_types::{
    ensure_equation_relation, LinearSystemSpec, LinearSystemSpecError,
};

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

pub(crate) fn parse_linear_system_spec(
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
                ensure_equation_relation(eq.op)?;
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

pub(crate) fn parse_linear_system_invocation_input(line: &str) -> String {
    let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();
    let inner = if rest.starts_with('(') && rest.ends_with(')') && rest.len() >= 2 {
        &rest[1..rest.len() - 1]
    } else {
        rest
    };
    inner.trim().to_string()
}
