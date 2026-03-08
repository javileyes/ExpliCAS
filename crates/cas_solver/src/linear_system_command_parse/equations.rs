use cas_ast::{Context, Expr};

use crate::linear_system_command_types::{ensure_equation_relation, LinearSystemSpecError};

pub(super) fn parse_linear_system_exprs(
    ctx: &mut Context,
    eq_parts: &[&str],
) -> Result<Vec<cas_ast::ExprId>, LinearSystemSpecError> {
    let mut exprs = Vec::with_capacity(eq_parts.len());
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
    Ok(exprs)
}
