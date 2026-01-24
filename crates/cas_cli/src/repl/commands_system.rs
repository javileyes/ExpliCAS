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
    /// System has infinitely many solutions (dependent equations)
    InfiniteSolutions,
    /// System has no solution (inconsistent equations)
    NoSolution,
    PolyConversion(PolyError),
}

impl std::fmt::Display for LinearSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearSystemError::NotLinear(msg) => write!(f, "non-linear term: {}", msg),
            LinearSystemError::InfiniteSolutions => {
                write!(
                    f,
                    "system has infinitely many solutions (dependent equations)"
                )
            }
            LinearSystemError::NoSolution => {
                write!(f, "system has no solution (inconsistent equations)")
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
        // Determine if infinite solutions or no solution
        // Check if (a1, b1, d1) is proportional to (a2, b2, d2)
        return Err(classify_degenerate_2x2(a1, b1, &d1, a2, b2, &d2));
    }

    // x = (d1*b2 - b1*d2) / det
    // y = (a1*d2 - d1*a2) / det
    let x = (&d1 * b2 - b1 * &d2) / &det;
    let y = (a1 * &d2 - &d1 * a2) / &det;

    Ok((x, y))
}

/// Classify a degenerate 2x2 system as infinite solutions or no solution
fn classify_degenerate_2x2(
    a1: &BigRational,
    b1: &BigRational,
    d1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    d2: &BigRational,
) -> LinearSystemError {
    // When det = 0, we have a1*b2 = a2*b1 (parallel lines)
    // Need to check if d1*b2 = d2*b1 and d1*a2 = d2*a1 (same line)
    // If yes: infinite solutions. If no: no solution.

    // Check cross products for proportionality
    let lhs_consistent = d1 * b2 == d2 * b1;
    let rhs_consistent = d1 * a2 == d2 * a1;

    if lhs_consistent && rhs_consistent {
        LinearSystemError::InfiniteSolutions
    } else {
        LinearSystemError::NoSolution
    }
}

/// Check if two 3-variable equations are proportional (consistent when parallel)
/// (a1, b1, c1, e1) vs (a2, b2, c2, e2) representing a1*x + b1*y + c1*z = e1
#[allow(clippy::too_many_arguments)]
fn check_proportional_3(
    a1: &BigRational,
    b1: &BigRational,
    c1: &BigRational,
    e1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    c2: &BigRational,
    e2: &BigRational,
) -> bool {
    // Find a non-zero coefficient to use as reference ratio
    // Then check if all coefficients (including e) follow the same ratio

    // Cross-multiply checks for each pair of components
    // (a1, a2) should have same ratio as (b1, b2), (c1, c2), (e1, e2)

    // Check: a1*b2 = a2*b1, a1*c2 = a2*c1, a1*e2 = a2*e1
    // And:   b1*c2 = b2*c1, b1*e2 = b2*e1
    // And:   c1*e2 = c2*e1

    let ab = a1 * b2 == a2 * b1;
    let ac = a1 * c2 == a2 * c1;
    let ae = a1 * e2 == a2 * e1;
    let bc = b1 * c2 == b2 * c1;
    let be = b1 * e2 == b2 * e1;
    let ce = c1 * e2 == c2 * e1;

    ab && ac && ae && bc && be && ce
}

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
#[allow(clippy::too_many_arguments)]
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
        // For 3x3 with det=0, check consistency by examining sub-systems
        // If any pair of equations is inconsistent, the whole system is inconsistent

        // Check each pair using 2x2 consistency logic
        // Two equations ax + by + cz = e are consistent if they're proportional

        let pair1_consistent =
            check_proportional_3(&c1.a, &c1.b, &c1.c, &e1, &c2.a, &c2.b, &c2.c, &e2);
        let pair2_consistent =
            check_proportional_3(&c1.a, &c1.b, &c1.c, &e1, &c3.a, &c3.b, &c3.c, &e3);
        let pair3_consistent =
            check_proportional_3(&c2.a, &c2.b, &c2.c, &e2, &c3.a, &c3.b, &c3.c, &e3);

        if pair1_consistent && pair2_consistent && pair3_consistent {
            return Err(LinearSystemError::InfiniteSolutions);
        } else {
            return Err(LinearSystemError::NoSolution);
        }
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

// =============================================================================
// N×N Gaussian Elimination Solver
// =============================================================================

/// Result of solving a linear system
#[derive(Debug)]
pub enum LinSolveResult {
    /// Unique solution: values for each variable in order
    Unique(Vec<BigRational>),
    /// Infinitely many solutions (dependent equations)
    Infinite,
    /// No solution (inconsistent equations)
    Inconsistent,
}

/// Extract linear coefficients from an equation for n variables.
/// Returns row of coefficients [a1, a2, ..., an] and constant term b
/// where equation is: a1*x1 + a2*x2 + ... + an*xn = b
fn extract_linear_row(
    ctx: &Context,
    expr: ExprId,
    vars: &[&str],
) -> Result<(Vec<BigRational>, BigRational), LinearSystemError> {
    let budget = PolyBudget {
        max_terms: 200,
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

    // Find variable indices in the MultiPoly
    let var_indices: Vec<Option<usize>> = vars
        .iter()
        .map(|v| poly.vars.iter().position(|pv| pv == *v))
        .collect();

    // Initialize coefficient vector and constant
    let n = vars.len();
    let mut coeffs = vec![BigRational::zero(); n];
    let mut constant = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            // Constant term
            constant = &constant + coef;
        } else if total_exp == 1 {
            // Linear term - find which variable
            let mut found = false;
            for (mono_idx, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    // Check if this variable is one of our target variables
                    for (var_idx, opt_idx) in var_indices.iter().enumerate() {
                        if *opt_idx == Some(mono_idx) {
                            coeffs[var_idx] = &coeffs[var_idx] + coef;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        // Variable not in our list
                        return Err(LinearSystemError::NotLinear(format!(
                            "unexpected variable '{}'",
                            poly.vars[mono_idx]
                        )));
                    }
                }
            }
        } else {
            return Err(LinearSystemError::NotLinear(format!(
                "non-linear term with degree {}",
                total_exp
            )));
        }
    }

    // Return (coefficients, -constant) since equation is sum = -constant
    Ok((coeffs, -constant))
}

/// Build the augmented matrix [A|b] from equations
fn build_augmented_matrix(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<Vec<Vec<BigRational>>, LinearSystemError> {
    let n = vars.len();
    let m = exprs.len();

    let mut matrix = Vec::with_capacity(m);

    for (i, &expr) in exprs.iter().enumerate() {
        let (coeffs, b) = extract_linear_row(ctx, expr, vars).map_err(|e| match e {
            LinearSystemError::NotLinear(msg) => {
                LinearSystemError::NotLinear(format!("equation {}: {}", i + 1, msg))
            }
            other => other,
        })?;

        if coeffs.len() != n {
            return Err(LinearSystemError::NotLinear(format!(
                "equation {} has wrong number of coefficients",
                i + 1
            )));
        }

        // Create augmented row: [a1, a2, ..., an, b]
        let mut row = coeffs;
        row.push(b);
        matrix.push(row);
    }

    Ok(matrix)
}

/// Gaussian elimination with partial pivoting on augmented matrix
/// Returns LinSolveResult
#[allow(clippy::needless_range_loop)] // Complex borrow pattern: reads pivot_row while modifying row
fn gauss_solve(mut matrix: Vec<Vec<BigRational>>, n: usize) -> LinSolveResult {
    let m = matrix.len(); // number of equations

    if m < n {
        // Underdetermined system - could have infinite or no solutions
        // For now, do elimination and check
    }

    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new(); // Track which columns have pivots

    // Forward elimination
    for col in 0..n {
        // Find pivot (first non-zero in column from pivot_row down)
        let mut pivot_found = None;
        for (row, mat_row) in matrix.iter().enumerate().skip(pivot_row) {
            if !mat_row[col].is_zero() {
                pivot_found = Some(row);
                break;
            }
        }

        let Some(pivot_idx) = pivot_found else {
            // No pivot in this column, continue to next
            continue;
        };

        // Swap rows if needed
        if pivot_idx != pivot_row {
            matrix.swap(pivot_row, pivot_idx);
        }

        pivot_cols.push(col);

        // Scale pivot row to make pivot = 1
        let pivot_val = matrix[pivot_row][col].clone();
        for cell in matrix[pivot_row].iter_mut().take(n + 1) {
            *cell = &*cell / &pivot_val;
        }

        // Eliminate below
        for row in (pivot_row + 1)..m {
            if !matrix[row][col].is_zero() {
                let factor = matrix[row][col].clone();
                for j in 0..=n {
                    let subtrahend = &factor * &matrix[pivot_row][j];
                    matrix[row][j] -= subtrahend;
                }
            }
        }

        pivot_row += 1;
        if pivot_row >= m {
            break;
        }
    }

    let rank = pivot_cols.len();

    // Check for inconsistency: any row [0, 0, ..., 0 | c] where c ≠ 0
    for mat_row in matrix.iter().take(m).skip(rank) {
        let all_zero = (0..n).all(|j| mat_row[j].is_zero());
        if all_zero && !mat_row[n].is_zero() {
            return LinSolveResult::Inconsistent;
        }
    }

    // If rank < n, infinite solutions
    if rank < n {
        return LinSolveResult::Infinite;
    }

    // Back substitution for unique solution
    let mut solution = vec![BigRational::zero(); n];

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let mut val = matrix[i][n].clone();

        for j in (col + 1)..n {
            val -= &matrix[i][j] * &solution[j];
        }

        solution[col] = val;
    }

    LinSolveResult::Unique(solution)
}

/// Solve n×n system using Gaussian elimination
fn solve_nxn_gauss(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    let n = vars.len();
    let matrix = build_augmented_matrix(ctx, exprs, vars)?;
    Ok(gauss_solve(matrix, n))
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
        // parts = [eq1, eq2, ..., eqn, var1, var2, ..., varn]
        // For n×n: total parts = 2*n
        if parts.len() < 4 || !parts.len().is_multiple_of(2) {
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

        let n = parts.len() / 2;

        match n {
            2 => self.solve_2x2_system(&parts),
            3 => self.solve_3x3_system(&parts),
            _ => self.solve_nxn_system(&parts, n),
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
            Err(LinearSystemError::InfiniteSolutions) => reply_output(
                "System has infinitely many solutions.\n\
                 The equations are dependent (same line).",
            ),
            Err(LinearSystemError::NoSolution) => reply_output(
                "System has no solution.\n\
                 The equations are inconsistent (parallel lines).",
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
            Err(LinearSystemError::InfiniteSolutions) => reply_output(
                "System has infinitely many solutions.\n\
                 The equations are dependent.",
            ),
            Err(LinearSystemError::NoSolution) => reply_output(
                "System has no solution.\n\
                 The equations are inconsistent.",
            ),
            Err(e) => reply_output(format!("Error solving system: {}", e)),
        }
    }

    /// Solve an n×n linear system using Gaussian elimination
    fn solve_nxn_system(&mut self, parts: &[&str], n: usize) -> ReplReply {
        // First n parts are equations, last n parts are variables
        let eq_strs = &parts[0..n];
        let var_strs = &parts[n..2 * n];

        // Validate variable names
        for (i, var) in var_strs.iter().enumerate() {
            if !is_valid_var(var) {
                return reply_output(format!(
                    "Invalid variable name at position {}: '{}'\n\
                     Variables must be simple identifiers.",
                    i + 1,
                    var
                ));
            }
        }

        // Parse all equations
        let mut equations = Vec::with_capacity(n);
        for (i, eq_str) in eq_strs.iter().enumerate() {
            match self.parse_equation(eq_str, i + 1) {
                Ok(eq) => equations.push(eq),
                Err(reply) => return reply,
            }
        }

        // Normalize equations: expr = lhs - rhs
        let ctx = &mut self.core.engine.simplifier.context;
        let exprs: Vec<ExprId> = equations
            .iter()
            .map(|eq| ctx.add(Expr::Sub(eq.lhs, eq.rhs)))
            .collect();

        // Solve using Gaussian elimination
        match solve_nxn_gauss(ctx, &exprs, var_strs) {
            Ok(LinSolveResult::Unique(solution)) => {
                // Format output: { x = val1, y = val2, ... }
                let pairs: Vec<String> = var_strs
                    .iter()
                    .zip(solution.iter())
                    .map(|(var, val)| {
                        let val_str = self.display_rational(val);
                        format!("{} = {}", var, val_str)
                    })
                    .collect();
                reply_output(format!("{{ {} }}", pairs.join(", ")))
            }
            Ok(LinSolveResult::Infinite) => reply_output(
                "System has infinitely many solutions.\n\
                 The equations are dependent.",
            ),
            Ok(LinSolveResult::Inconsistent) => reply_output(
                "System has no solution.\n\
                 The equations are inconsistent.",
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
