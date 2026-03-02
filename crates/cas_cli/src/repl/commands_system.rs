//! Linear system solving commands.
//!
//! Syntax: solve_system(eq1; eq2; x; y)
//! Example: solve_system(x+y=3; x-y=1; x; y) → { x = 2, y = 1 }

use super::output::{reply_output, ReplReply};
use super::Repl;
use cas_ast::{Expr, ExprId};
use num_rational::BigRational;

fn bigrational_to_expr(ctx: &mut cas_ast::Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

fn format_not_linear_reply(message: &str) -> String {
    let with_prefix = |detail: &str| {
        if detail.contains("non-linear") {
            detail.to_string()
        } else {
            format!("non-linear term: {}", detail)
        }
    };

    if let Some((eq, detail)) = message
        .strip_prefix("equation ")
        .and_then(|rest| rest.split_once(": "))
    {
        format!(
            "Error in equation {}: {}\nsolve_system() only handles linear equations.",
            eq,
            with_prefix(detail)
        )
    } else {
        format!(
            "Error: {}\nsolve_system() only handles linear equations.",
            with_prefix(message)
        )
    }
}

impl Repl {
    /// Wrapper: handle_solve_system (prints result)
    pub(crate) fn handle_solve_system(&mut self, line: &str) {
        let reply = self.handle_solve_system_core(line);
        self.print_reply(reply);
    }

    /// Core: handle_solve_system_core (returns ReplReply, no I/O)
    fn handle_solve_system_core(&mut self, line: &str) -> ReplReply {
        // Strip command prefix: "solve_system" or "solve_system("
        let rest = line.strip_prefix("solve_system").unwrap_or(line).trim();

        // Handle parenthesized form: solve_system(...)
        let inner = if rest.starts_with('(') && rest.ends_with(')') {
            &rest[1..rest.len() - 1]
        } else {
            rest
        };

        let spec = {
            let ctx = &mut self.core.engine.simplifier.context;
            match cas_solver::parse_linear_system_spec(ctx, inner) {
                Ok(spec) => spec,
                Err(cas_solver::LinearSystemSpecError::InvalidPartCount) => {
                    return reply_output(
                        "Usage:\n  \
                         2×2: solve_system(eq1; eq2; x; y)\n  \
                         3×3: solve_system(eq1; eq2; eq3; x; y; z)\n  \
                         n×n: solve_system(eq1; ...; eqn; x1; ...; xn)\n\n\
                         Examples:\n  \
                         solve_system(x+y=3; x-y=1; x; y)\n  \
                         solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)",
                    );
                }
                Err(cas_solver::LinearSystemSpecError::InvalidVariableName { name, .. }) => {
                    return reply_output(format!(
                        "Invalid variable name: '{}'\n\
                         Variables must be simple identifiers.",
                        name
                    ));
                }
                Err(cas_solver::LinearSystemSpecError::ParseEquation { position, message }) => {
                    return reply_output(format!(
                        "Error parsing equation {}: {}",
                        position, message
                    ));
                }
                Err(cas_solver::LinearSystemSpecError::ExpectedEquation { position, input }) => {
                    return reply_output(format!(
                        "Expected equation, got expression in position {}: '{}'\n\
                         Use '=' to create an equation.",
                        position, input
                    ));
                }
                Err(cas_solver::LinearSystemSpecError::UnsupportedRelation { .. }) => {
                    return reply_output(
                        "solve_system(): only '=' equations supported\n\
                         Inequalities (<, >, <=, >=, !=) are not supported.",
                    );
                }
            }
        };

        match cas_solver::solve_linear_system_spec(&self.core.engine.simplifier.context, &spec) {
            Ok(cas_solver::LinSolveResult::Unique(solution)) => {
                let pairs: Vec<String> = spec
                    .vars
                    .iter()
                    .zip(solution.iter())
                    .map(|(var, val)| {
                        let val_str = self.display_rational(val);
                        format!("{} = {}", var, val_str)
                    })
                    .collect();
                reply_output(format!("{{ {} }}", pairs.join(", ")))
            }
            Ok(cas_solver::LinSolveResult::Infinite) => reply_output(
                "System has infinitely many solutions.\n\
                 The equations are dependent.",
            ),
            Ok(cas_solver::LinSolveResult::Inconsistent) => reply_output(
                "System has no solution.\n\
                 The equations are inconsistent.",
            ),
            Err(cas_solver::LinearSystemError::NotLinear(message)) => {
                reply_output(format_not_linear_reply(&message))
            }
            Err(e) => reply_output(format!("Error solving system: {}", e)),
        }
    }

    /// Display a BigRational as a string.
    fn display_rational(&mut self, r: &BigRational) -> String {
        let ctx = &mut self.core.engine.simplifier.context;
        let expr = bigrational_to_expr(ctx, r);
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: expr
            }
        )
    }
}
