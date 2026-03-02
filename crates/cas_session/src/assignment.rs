use crate::SessionState;
use cas_ast::{Expr, ExprId};

/// Errors returned when applying a `let` assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentError {
    EmptyName,
    InvalidNameStart,
    ReservedName(String),
    Parse(String),
}

/// Parsed form of `let` command assignment input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedLetAssignment<'a> {
    pub name: &'a str,
    pub expr: &'a str,
    pub lazy: bool,
}

/// Error while parsing `let` command assignment syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LetAssignmentParseError {
    MissingAssignmentOperator,
}

/// Parse `let` tail input:
/// - `name := expr` -> lazy
/// - `name = expr` -> eager
pub fn parse_let_assignment_input(
    rest: &str,
) -> Result<ParsedLetAssignment<'_>, LetAssignmentParseError> {
    if let Some(idx) = rest.find(":=") {
        Ok(ParsedLetAssignment {
            name: rest[..idx].trim(),
            expr: rest[idx + 2..].trim(),
            lazy: true,
        })
    } else if let Some(eq_idx) = rest.find('=') {
        Ok(ParsedLetAssignment {
            name: rest[..eq_idx].trim(),
            expr: rest[eq_idx + 1..].trim(),
            lazy: false,
        })
    } else {
        Err(LetAssignmentParseError::MissingAssignmentOperator)
    }
}

fn unwrap_hold_top(ctx: &cas_ast::Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if ctx.is_builtin(*name, cas_ast::BuiltinFn::Hold) && args.len() == 1 {
            return args[0];
        }
    }
    expr
}

/// Apply a session assignment:
/// - `lazy = false` behaves like `let a = ...` (eager simplify + unwrap top-level hold).
/// - `lazy = true` behaves like `let a := ...` (store unresolved formula after ref/env substitution).
pub fn apply_assignment(
    state: &mut SessionState,
    simplifier: &mut cas_engine::Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<ExprId, AssignmentError> {
    if name.is_empty() {
        return Err(AssignmentError::EmptyName);
    }

    let starts_with_letter = name
        .chars()
        .next()
        .map(|c| c.is_alphabetic())
        .unwrap_or(false);
    if !starts_with_letter && !name.starts_with('_') {
        return Err(AssignmentError::InvalidNameStart);
    }

    if crate::env::is_reserved(name) {
        return Err(AssignmentError::ReservedName(name.to_string()));
    }

    let rhs_expr = cas_parser::parse(expr_str, &mut simplifier.context)
        .map_err(|e| AssignmentError::Parse(e.to_string()))?;

    // Temporarily remove this binding to prevent self-reference in substitution.
    let old_binding = state.get_binding(name);
    state.unset_binding(name);

    let rhs_substituted = match state.resolve_state_refs(&mut simplifier.context, rhs_expr) {
        Ok(r) => r,
        Err(_) => rhs_expr,
    };

    let result = if lazy {
        rhs_substituted
    } else {
        let (simplified, _steps) = simplifier.simplify(rhs_substituted);
        unwrap_hold_top(&simplifier.context, simplified)
    };

    state.set_binding(name.to_string(), result);
    let _ = old_binding;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_assignment, parse_let_assignment_input, AssignmentError, LetAssignmentParseError,
        ParsedLetAssignment,
    };

    #[test]
    fn apply_assignment_validates_name() {
        let mut state = crate::SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let err = apply_assignment(&mut state, &mut simplifier, "", "1", false).expect_err("err");
        assert_eq!(err, AssignmentError::EmptyName);
    }

    #[test]
    fn apply_assignment_stores_eager_value() {
        let mut state = crate::SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let id = apply_assignment(&mut state, &mut simplifier, "a", "x + x", false).expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id
            }
        );
        assert_eq!(rendered.replace(' ', ""), "2*x");
    }

    #[test]
    fn apply_assignment_stores_lazy_formula() {
        let mut state = crate::SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let id = apply_assignment(&mut state, &mut simplifier, "a", "x + x", true).expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id
            }
        );
        assert_eq!(rendered, "x + x");
    }

    #[test]
    fn parse_let_assignment_input_supports_lazy_and_eager() {
        let lazy = parse_let_assignment_input("a := x + 1").expect("lazy");
        assert_eq!(
            lazy,
            ParsedLetAssignment {
                name: "a",
                expr: "x + 1",
                lazy: true
            }
        );

        let eager = parse_let_assignment_input("b = x + 2").expect("eager");
        assert_eq!(
            eager,
            ParsedLetAssignment {
                name: "b",
                expr: "x + 2",
                lazy: false
            }
        );
    }

    #[test]
    fn parse_let_assignment_input_requires_operator() {
        let err = parse_let_assignment_input("abc").expect_err("missing operator");
        assert_eq!(err, LetAssignmentParseError::MissingAssignmentOperator);
    }
}
