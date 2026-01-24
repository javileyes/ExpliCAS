//! Linear system solving commands (2x2)
//!
//! Syntax: solve_system(eq1; eq2; x; y)
//! Example: solve_system(x+y=3; x-y=1; x; y) → { x = 2, y = 1 }

use super::output::{reply_output, ReplReply};
use super::Repl;
use cas_ast::{Context, Expr, ExprId, RelOp};
use cas_engine::multipoly::{multipoly_from_expr, PolyBudget, PolyError};
use num_rational::BigRational;
use num_traits::Zero;

/// Error type for linear system solving
#[derive(Debug)]
pub enum LinearSystemError {
    NotLinear(String),
    DegenerateSystem,
    PolyConversion(PolyError),
}

impl std::fmt::Display for LinearSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearSystemError::NotLinear(msg) => write!(f, "non-linear term: {}", msg),
            LinearSystemError::DegenerateSystem => {
                write!(f, "determinant is 0; system has no unique solution")
            }
            LinearSystemError::PolyConversion(e) => write!(f, "polynomial conversion: {}", e),
        }
    }
}

/// Linear coefficients for equation ax + by + c = 0
#[derive(Debug, Clone)]
pub struct LinearCoeffs {
    pub a: BigRational, // coefficient of x
    pub b: BigRational, // coefficient of y
    pub c: BigRational, // constant term
}

/// Extract linear coefficients from a polynomial expression.
/// Returns (a, b, c) where the equation is ax + by + c = 0
fn extract_linear_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<LinearCoeffs, LinearSystemError> {
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 2, // Allow check, we verify <= 1 below
        max_pow_exp: 2,
    };

    let poly =
        multipoly_from_expr(ctx, expr, &budget).map_err(LinearSystemError::PolyConversion)?;

    // Fast check: if total degree > 1, not linear
    if poly.total_degree() > 1 {
        return Err(LinearSystemError::NotLinear(
            "degree > 1 in the system".to_string(),
        ));
    }

    // Find variable indices in the polynomial's vars list
    let idx_x = poly.vars.iter().position(|v| v == var_x);
    let idx_y = poly.vars.iter().position(|v| v == var_y);
    let num_vars = poly.vars.len();

    // Extract coefficients by iterating terms
    let mut a = BigRational::zero();
    let mut b = BigRational::zero();
    let mut c = BigRational::zero();

    for (coef, mono) in &poly.terms {
        // Check if this term is linear or constant
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            // Constant term
            c = &c + coef;
        } else if total_exp == 1 {
            // Linear term - find which variable
            let mut found = false;
            for (i, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    if Some(i) == idx_x {
                        a = &a + coef;
                        found = true;
                    } else if Some(i) == idx_y {
                        b = &b + coef;
                        found = true;
                    } else {
                        // Variable not in our system
                        return Err(LinearSystemError::NotLinear(format!(
                            "unexpected variable '{}'",
                            poly.vars[i]
                        )));
                    }
                }
            }
            if !found && num_vars == 0 {
                c = &c + coef; // Edge case: constant
            }
        } else {
            return Err(LinearSystemError::NotLinear(format!(
                "non-linear term with degree {}",
                total_exp
            )));
        }
    }

    Ok(LinearCoeffs { a, b, c })
}

/// Solve 2x2 linear system using Cramer's rule.
/// System: a1*x + b1*y = d1, a2*x + b2*y = d2
/// where d1 = -c1, d2 = -c2 (from ax + by + c = 0)
fn solve_2x2_cramer(
    coeffs1: &LinearCoeffs,
    coeffs2: &LinearCoeffs,
) -> Result<(BigRational, BigRational), LinearSystemError> {
    let a1 = &coeffs1.a;
    let b1 = &coeffs1.b;
    let d1 = -coeffs1.c.clone(); // from ax + by + c = 0 → ax + by = -c

    let a2 = &coeffs2.a;
    let b2 = &coeffs2.b;
    let d2 = -coeffs2.c.clone();

    // det = a1*b2 - a2*b1
    let det = a1 * b2 - a2 * b1;

    if det.is_zero() {
        return Err(LinearSystemError::DegenerateSystem);
    }

    // x = (d1*b2 - b1*d2) / det
    // y = (a1*d2 - d1*a2) / det
    let x = (&d1 * b2 - b1 * &d2) / &det;
    let y = (a1 * &d2 - &d1 * a2) / &det;

    Ok((x, y))
}

/// Split a string by `;` at top level (respecting parentheses)
fn split_by_semicolon(s: &str) -> Vec<&str> {
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

/// Convert BigRational to ExprId
fn bigrational_to_expr(ctx: &mut Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
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

        // Split by semicolon
        let parts: Vec<&str> = split_by_semicolon(inner)
            .into_iter()
            .map(|s| s.trim())
            .collect();

        if parts.len() != 4 {
            return reply_output(
                "Usage: solve_system(eq1; eq2; x; y)\n\
                 Example: solve_system(x+y=3; x-y=1; x; y)",
            );
        }

        let eq1_str = parts[0];
        let eq2_str = parts[1];
        let var_x = parts[2];
        let var_y = parts[3];

        // Validate variable names (simple identifiers)
        if !var_x.chars().all(|c| c.is_alphabetic() || c == '_')
            || !var_y.chars().all(|c| c.is_alphabetic() || c == '_')
        {
            return reply_output(format!(
                "Invalid variable names: '{}', '{}'\n\
                 Variables must be simple identifiers (letters only).",
                var_x, var_y
            ));
        }

        // Parse equations
        let eq1 =
            match cas_parser::parse_statement(eq1_str, &mut self.core.engine.simplifier.context) {
                Ok(cas_parser::Statement::Equation(eq)) => eq,
                Ok(cas_parser::Statement::Expression(_)) => {
                    return reply_output(format!(
                        "Expected equation, got expression: '{}'\n\
                     Use '=' to create an equation, e.g., 'x+y=3'",
                        eq1_str
                    ));
                }
                Err(e) => {
                    return reply_output(format!("Error parsing equation 1: {}", e));
                }
            };

        let eq2 =
            match cas_parser::parse_statement(eq2_str, &mut self.core.engine.simplifier.context) {
                Ok(cas_parser::Statement::Equation(eq)) => eq,
                Ok(cas_parser::Statement::Expression(_)) => {
                    return reply_output(format!(
                        "Expected equation, got expression: '{}'\n\
                     Use '=' to create an equation, e.g., 'x-y=1'",
                        eq2_str
                    ));
                }
                Err(e) => {
                    return reply_output(format!("Error parsing equation 2: {}", e));
                }
            };

        // Validate equation operators (only '=' supported)
        if eq1.op != RelOp::Eq || eq2.op != RelOp::Eq {
            return reply_output(
                "solve_system(): only '=' equations supported\n\
                 Inequalities (<, >, <=, >=, !=) are not supported.",
            );
        }

        // Normalize equations: expr = lhs - rhs
        let ctx = &mut self.core.engine.simplifier.context;
        let expr1 = ctx.add(Expr::Sub(eq1.lhs, eq1.rhs));
        let expr2 = ctx.add(Expr::Sub(eq2.lhs, eq2.rhs));

        // Extract linear coefficients
        let coeffs1 = match extract_linear_coeffs(ctx, expr1, var_x, var_y) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 1: {}\n\
                     solve_system() only handles linear equations.",
                    e
                ));
            }
        };

        let coeffs2 = match extract_linear_coeffs(ctx, expr2, var_x, var_y) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 2: {}\n\
                     solve_system() only handles linear equations.",
                    e
                ));
            }
        };

        // Solve using Cramer's rule
        match solve_2x2_cramer(&coeffs1, &coeffs2) {
            Ok((x_val, y_val)) => {
                // Convert to expressions for pretty printing
                let x_expr = bigrational_to_expr(ctx, &x_val);
                let y_expr = bigrational_to_expr(ctx, &y_val);

                let x_str = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: x_expr
                    }
                );
                let y_str = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: y_expr
                    }
                );

                reply_output(format!(
                    "{{ {} = {}, {} = {} }}",
                    var_x, x_str, var_y, y_str
                ))
            }
            Err(LinearSystemError::DegenerateSystem) => reply_output(
                "determinant is 0; system has no unique solution\n\
                     The system may have infinitely many solutions or none.",
            ),
            Err(e) => reply_output(format!("Error solving system: {}", e)),
        }
    }
}
