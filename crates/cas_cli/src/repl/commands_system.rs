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

/// Linear coefficients for equation ax + by + c = 0 (2 variables)
#[derive(Debug, Clone)]
pub struct LinearCoeffs {
    pub a: BigRational, // coefficient of x
    pub b: BigRational, // coefficient of y
    pub c: BigRational, // constant term
}

/// Linear coefficients for equation ax + by + cz + d = 0 (3 variables)
#[derive(Debug, Clone)]
pub struct LinearCoeffs3 {
    pub a: BigRational, // coefficient of x
    pub b: BigRational, // coefficient of y
    pub c: BigRational, // coefficient of z
    pub d: BigRational, // constant term
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

/// Extract linear coefficients from a polynomial for 3 variables.
/// Returns coeffs where the equation is ax + by + cz + d = 0
fn extract_linear_coeffs_3(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
    var_z: &str,
) -> Result<LinearCoeffs3, LinearSystemError> {
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 2,
        max_pow_exp: 2,
    };

    let poly =
        multipoly_from_expr(ctx, expr, &budget).map_err(LinearSystemError::PolyConversion)?;

    if poly.total_degree() > 1 {
        return Err(LinearSystemError::NotLinear(
            "degree > 1 in the system".to_string(),
        ));
    }

    let idx_x = poly.vars.iter().position(|v| v == var_x);
    let idx_y = poly.vars.iter().position(|v| v == var_y);
    let idx_z = poly.vars.iter().position(|v| v == var_z);

    let mut a = BigRational::zero();
    let mut b = BigRational::zero();
    let mut c = BigRational::zero();
    let mut d = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            d = &d + coef;
        } else if total_exp == 1 {
            let mut found = false;
            for (i, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    if Some(i) == idx_x {
                        a = &a + coef;
                        found = true;
                    } else if Some(i) == idx_y {
                        b = &b + coef;
                        found = true;
                    } else if Some(i) == idx_z {
                        c = &c + coef;
                        found = true;
                    } else {
                        return Err(LinearSystemError::NotLinear(format!(
                            "unexpected variable '{}'",
                            poly.vars[i]
                        )));
                    }
                }
            }
            if !found {
                d = &d + coef;
            }
        } else {
            return Err(LinearSystemError::NotLinear(format!(
                "non-linear term with degree {}",
                total_exp
            )));
        }
    }

    Ok(LinearCoeffs3 { a, b, c, d })
}

/// Compute 3x3 determinant
/// | a1 b1 c1 |
/// | a2 b2 c2 |
/// | a3 b3 c3 |
fn det3x3(
    a1: &BigRational,
    b1: &BigRational,
    c1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    c2: &BigRational,
    a3: &BigRational,
    b3: &BigRational,
    c3: &BigRational,
) -> BigRational {
    // Sarrus/cofactor expansion
    a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2)
}

/// Solve 3x3 linear system using Cramer's rule.
/// System: a*x + b*y + c*z = e  (where e = -d from ax + by + cz + d = 0)
fn solve_3x3_cramer(
    c1: &LinearCoeffs3,
    c2: &LinearCoeffs3,
    c3: &LinearCoeffs3,
) -> Result<(BigRational, BigRational, BigRational), LinearSystemError> {
    // Convert to Ax = E form where E = -d
    let e1 = -c1.d.clone();
    let e2 = -c2.d.clone();
    let e3 = -c3.d.clone();

    // Coefficient matrix determinant
    let det_a = det3x3(
        &c1.a, &c1.b, &c1.c, &c2.a, &c2.b, &c2.c, &c3.a, &c3.b, &c3.c,
    );

    if det_a.is_zero() {
        return Err(LinearSystemError::DegenerateSystem);
    }

    // Cramer's rule: replace column i with E vector
    let det_x = det3x3(&e1, &c1.b, &c1.c, &e2, &c2.b, &c2.c, &e3, &c3.b, &c3.c);

    let det_y = det3x3(&c1.a, &e1, &c1.c, &c2.a, &e2, &c2.c, &c3.a, &e3, &c3.c);

    let det_z = det3x3(&c1.a, &c1.b, &e1, &c2.a, &c2.b, &e2, &c3.a, &c3.b, &e3);

    let x = det_x / &det_a;
    let y = det_y / &det_a;
    let z = det_z / &det_a;

    Ok((x, y, z))
}

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

        // Dispatch based on number of parts
        match parts.len() {
            4 => self.solve_2x2_system(&parts),
            6 => self.solve_3x3_system(&parts),
            _ => reply_output(
                "Usage:\n  \
                 2x2: solve_system(eq1; eq2; x; y)\n  \
                 3x3: solve_system(eq1; eq2; eq3; x; y; z)\n\n\
                 Examples:\n  \
                 solve_system(x+y=3; x-y=1; x; y)\n  \
                 solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)",
            ),
        }
    }

    /// Solve a 2x2 linear system
    fn solve_2x2_system(&mut self, parts: &[&str]) -> ReplReply {
        let eq1_str = parts[0];
        let eq2_str = parts[1];
        let var_x = parts[2];
        let var_y = parts[3];

        // Validate variable names
        if !is_valid_var(var_x) || !is_valid_var(var_y) {
            return reply_output(format!(
                "Invalid variable names: '{}', '{}'\n\
                 Variables must be simple identifiers.",
                var_x, var_y
            ));
        }

        // Parse and validate equations
        let eq1 = match self.parse_equation(eq1_str, 1) {
            Ok(eq) => eq,
            Err(reply) => return reply,
        };
        let eq2 = match self.parse_equation(eq2_str, 2) {
            Ok(eq) => eq,
            Err(reply) => return reply,
        };

        // Normalize equations: expr = lhs - rhs
        let ctx = &mut self.core.engine.simplifier.context;
        let expr1 = ctx.add(Expr::Sub(eq1.lhs, eq1.rhs));
        let expr2 = ctx.add(Expr::Sub(eq2.lhs, eq2.rhs));

        // Extract linear coefficients
        let coeffs1 = match extract_linear_coeffs(ctx, expr1, var_x, var_y) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 1: {}\nsolve_system() only handles linear equations.",
                    e
                ))
            }
        };
        let coeffs2 = match extract_linear_coeffs(ctx, expr2, var_x, var_y) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 2: {}\nsolve_system() only handles linear equations.",
                    e
                ))
            }
        };

        // Solve using Cramer's rule
        match solve_2x2_cramer(&coeffs1, &coeffs2) {
            Ok((x_val, y_val)) => {
                let x_str = self.display_rational(&x_val);
                let y_str = self.display_rational(&y_val);
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

    /// Solve a 3x3 linear system
    fn solve_3x3_system(&mut self, parts: &[&str]) -> ReplReply {
        let eq1_str = parts[0];
        let eq2_str = parts[1];
        let eq3_str = parts[2];
        let var_x = parts[3];
        let var_y = parts[4];
        let var_z = parts[5];

        // Validate variable names
        if !is_valid_var(var_x) || !is_valid_var(var_y) || !is_valid_var(var_z) {
            return reply_output(format!(
                "Invalid variable names: '{}', '{}', '{}'\n\
                 Variables must be simple identifiers.",
                var_x, var_y, var_z
            ));
        }

        // Parse and validate equations
        let eq1 = match self.parse_equation(eq1_str, 1) {
            Ok(eq) => eq,
            Err(reply) => return reply,
        };
        let eq2 = match self.parse_equation(eq2_str, 2) {
            Ok(eq) => eq,
            Err(reply) => return reply,
        };
        let eq3 = match self.parse_equation(eq3_str, 3) {
            Ok(eq) => eq,
            Err(reply) => return reply,
        };

        // Normalize equations: expr = lhs - rhs
        let ctx = &mut self.core.engine.simplifier.context;
        let expr1 = ctx.add(Expr::Sub(eq1.lhs, eq1.rhs));
        let expr2 = ctx.add(Expr::Sub(eq2.lhs, eq2.rhs));
        let expr3 = ctx.add(Expr::Sub(eq3.lhs, eq3.rhs));

        // Extract linear coefficients
        let coeffs1 = match extract_linear_coeffs_3(ctx, expr1, var_x, var_y, var_z) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 1: {}\nsolve_system() only handles linear equations.",
                    e
                ))
            }
        };
        let coeffs2 = match extract_linear_coeffs_3(ctx, expr2, var_x, var_y, var_z) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 2: {}\nsolve_system() only handles linear equations.",
                    e
                ))
            }
        };
        let coeffs3 = match extract_linear_coeffs_3(ctx, expr3, var_x, var_y, var_z) {
            Ok(c) => c,
            Err(e) => {
                return reply_output(format!(
                    "Error in equation 3: {}\nsolve_system() only handles linear equations.",
                    e
                ))
            }
        };

        // Solve using Cramer's rule
        match solve_3x3_cramer(&coeffs1, &coeffs2, &coeffs3) {
            Ok((x_val, y_val, z_val)) => {
                let x_str = self.display_rational(&x_val);
                let y_str = self.display_rational(&y_val);
                let z_str = self.display_rational(&z_val);
                reply_output(format!(
                    "{{ {} = {}, {} = {}, {} = {} }}",
                    var_x, x_str, var_y, y_str, var_z, z_str
                ))
            }
            Err(LinearSystemError::DegenerateSystem) => reply_output(
                "determinant is 0; system has no unique solution\n\
                 The system may have infinitely many solutions or none.",
            ),
            Err(e) => reply_output(format!("Error solving system: {}", e)),
        }
    }

    /// Parse an equation string and validate it
    fn parse_equation(&mut self, eq_str: &str, num: usize) -> Result<cas_ast::Equation, ReplReply> {
        match cas_parser::parse_statement(eq_str, &mut self.core.engine.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                if eq.op != RelOp::Eq {
                    return Err(reply_output(
                        "solve_system(): only '=' equations supported\n\
                         Inequalities (<, >, <=, >=, !=) are not supported.",
                    ));
                }
                Ok(eq)
            }
            Ok(cas_parser::Statement::Expression(_)) => Err(reply_output(format!(
                "Expected equation, got expression in position {}: '{}'\n\
                 Use '=' to create an equation.",
                num, eq_str
            ))),
            Err(e) => Err(reply_output(format!(
                "Error parsing equation {}: {}",
                num, e
            ))),
        }
    }

    /// Display a BigRational as a string
    fn display_rational(&mut self, r: &BigRational) -> String {
        let ctx = &mut self.core.engine.simplifier.context;
        let expr = bigrational_to_expr(ctx, r);
        format!(
            "{}",
            cas_ast::DisplayExpr {
                context: ctx,
                id: expr
            }
        )
    }
}

/// Check if a string is a valid variable name
fn is_valid_var(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '_')
}
