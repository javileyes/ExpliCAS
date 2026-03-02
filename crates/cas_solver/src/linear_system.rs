use cas_ast::{Context, Expr, ExprId, RelOp};
use cas_math::multipoly::{multipoly_from_expr, PolyBudget, PolyError};
use num_rational::BigRational;
use num_traits::Zero;

/// Error type for linear system solving.
#[derive(Debug)]
pub enum LinearSystemError {
    NotLinear(String),
    /// System has infinitely many solutions (dependent equations).
    InfiniteSolutions,
    /// System has no solution (inconsistent equations).
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

/// Result of solving a linear system.
#[derive(Debug)]
pub enum LinSolveResult {
    /// Unique solution: values for each variable in order.
    Unique(Vec<BigRational>),
    /// Infinitely many solutions (dependent equations).
    Infinite,
    /// No solution (inconsistent equations).
    Inconsistent,
}

/// Parsed and normalized linear system specification from textual input.
#[derive(Debug, Clone)]
pub struct LinearSystemSpec {
    /// Expressions normalized as `lhs - rhs = 0`.
    pub exprs: Vec<ExprId>,
    /// Variables in the order requested by the caller.
    pub vars: Vec<String>,
}

/// Input parse/validation errors for linear system commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearSystemSpecError {
    /// The input does not contain `n` equations followed by `n` variables.
    InvalidPartCount,
    /// One of the variable identifiers is invalid.
    InvalidVariableName { position: usize, name: String },
    /// Equation failed to parse.
    ParseEquation { position: usize, message: String },
    /// A non-equation statement was found where an equation is required.
    ExpectedEquation { position: usize, input: String },
    /// Only `=` equations are supported.
    UnsupportedRelation { position: usize, relation: String },
}

impl std::fmt::Display for LinearSystemSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearSystemSpecError::InvalidPartCount => write!(
                f,
                "expected n equations followed by n variables (minimum 2 equations/2 variables)"
            ),
            LinearSystemSpecError::InvalidVariableName { position, name } => {
                write!(f, "invalid variable at position {}: '{}'", position, name)
            }
            LinearSystemSpecError::ParseEquation { position, message } => {
                write!(f, "error parsing equation {}: {}", position, message)
            }
            LinearSystemSpecError::ExpectedEquation { position, input } => {
                write!(f, "expected equation in position {}: '{}'", position, input)
            }
            LinearSystemSpecError::UnsupportedRelation { position, relation } => write!(
                f,
                "unsupported relation '{}' in equation {}",
                relation, position
            ),
        }
    }
}

/// Validate variable name used in `solve_system`.
pub fn is_valid_linear_system_var(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic() || c == '_')
}

/// Split by semicolon while ignoring separators inside parentheses/brackets/braces.
pub fn split_semicolon_top_level(s: &str) -> Vec<&str> {
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

/// Parse a textual linear system specification and normalize equations to `lhs-rhs=0`.
///
/// Input format:
/// - `eq1; eq2; ...; eqn; var1; var2; ...; varn`
pub fn parse_linear_system_spec(
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

/// Solve a parsed linear-system specification.
///
/// Returns a unified `LinSolveResult` for 2x2, 3x3, and generic n×n systems.
pub fn solve_linear_system_spec(
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
            let (x, y) = match solve_2x2_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                &spec.vars[0],
                &spec.vars[1],
            ) {
                Ok(pair) => pair,
                Err(LinearSystemError::InfiniteSolutions) => return Ok(LinSolveResult::Infinite),
                Err(LinearSystemError::NoSolution) => return Ok(LinSolveResult::Inconsistent),
                Err(e) => return Err(e),
            };
            Ok(LinSolveResult::Unique(vec![x, y]))
        }
        3 => {
            let (x, y, z) = match solve_3x3_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                spec.exprs[2],
                &spec.vars[0],
                &spec.vars[1],
                &spec.vars[2],
            ) {
                Ok(triple) => triple,
                Err(LinearSystemError::InfiniteSolutions) => return Ok(LinSolveResult::Infinite),
                Err(LinearSystemError::NoSolution) => return Ok(LinSolveResult::Inconsistent),
                Err(e) => return Err(e),
            };
            Ok(LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = spec.vars.iter().map(String::as_str).collect();
            solve_nxn_linear_system(ctx, &spec.exprs, &var_refs)
        }
    }
}

/// Solve 2x2 linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_2x2_linear_system(
    ctx: &Context,
    expr1: ExprId,
    expr2: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<(BigRational, BigRational), LinearSystemError> {
    let coeffs1 =
        extract_linear_coeffs(ctx, expr1, var_x, var_y).map_err(|e| with_equation_index(e, 1))?;
    let coeffs2 =
        extract_linear_coeffs(ctx, expr2, var_x, var_y).map_err(|e| with_equation_index(e, 2))?;
    solve_2x2_cramer(&coeffs1, &coeffs2)
}

/// Solve 3x3 linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_3x3_linear_system(
    ctx: &Context,
    expr1: ExprId,
    expr2: ExprId,
    expr3: ExprId,
    var_x: &str,
    var_y: &str,
    var_z: &str,
) -> Result<(BigRational, BigRational, BigRational), LinearSystemError> {
    let coeffs1 = extract_linear_coeffs_3(ctx, expr1, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 1))?;
    let coeffs2 = extract_linear_coeffs_3(ctx, expr2, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 2))?;
    let coeffs3 = extract_linear_coeffs_3(ctx, expr3, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 3))?;
    solve_3x3_cramer(&coeffs1, &coeffs2, &coeffs3)
}

/// Solve n×n linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_nxn_linear_system(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    solve_nxn_gauss(ctx, exprs, vars)
}

fn with_equation_index(error: LinearSystemError, index: usize) -> LinearSystemError {
    match error {
        LinearSystemError::NotLinear(message) => {
            LinearSystemError::NotLinear(format!("equation {}: {}", index, message))
        }
        other => other,
    }
}

/// Linear coefficients for equation `a*x + b*y + c = 0` (2 variables).
#[derive(Debug, Clone)]
struct LinearCoeffs {
    a: BigRational,
    b: BigRational,
    c: BigRational,
}

/// Linear coefficients for equation `a*x + b*y + c*z + d = 0` (3 variables).
#[derive(Debug, Clone)]
struct LinearCoeffs3 {
    a: BigRational,
    b: BigRational,
    c: BigRational,
    d: BigRational,
}

fn extract_linear_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<LinearCoeffs, LinearSystemError> {
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
    let num_vars = poly.vars.len();

    let mut a = BigRational::zero();
    let mut b = BigRational::zero();
    let mut c = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            c = &c + coef;
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
                    } else {
                        return Err(LinearSystemError::NotLinear(format!(
                            "unexpected variable '{}'",
                            poly.vars[i]
                        )));
                    }
                }
            }
            if !found && num_vars == 0 {
                c = &c + coef;
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

fn solve_2x2_cramer(
    coeffs1: &LinearCoeffs,
    coeffs2: &LinearCoeffs,
) -> Result<(BigRational, BigRational), LinearSystemError> {
    let a1 = &coeffs1.a;
    let b1 = &coeffs1.b;
    let d1 = -coeffs1.c.clone();

    let a2 = &coeffs2.a;
    let b2 = &coeffs2.b;
    let d2 = -coeffs2.c.clone();

    let det = a1 * b2 - a2 * b1;
    if det.is_zero() {
        return Err(classify_degenerate_2x2(a1, b1, &d1, a2, b2, &d2));
    }

    let x = (&d1 * b2 - b1 * &d2) / &det;
    let y = (a1 * &d2 - &d1 * a2) / &det;
    Ok((x, y))
}

fn classify_degenerate_2x2(
    a1: &BigRational,
    b1: &BigRational,
    d1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    d2: &BigRational,
) -> LinearSystemError {
    let lhs_consistent = d1 * b2 == d2 * b1;
    let rhs_consistent = d1 * a2 == d2 * a1;

    if lhs_consistent && rhs_consistent {
        LinearSystemError::InfiniteSolutions
    } else {
        LinearSystemError::NoSolution
    }
}

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
    let ab = a1 * b2 == a2 * b1;
    let ac = a1 * c2 == a2 * c1;
    let ae = a1 * e2 == a2 * e1;
    let bc = b1 * c2 == b2 * c1;
    let be = b1 * e2 == b2 * e1;
    let ce = c1 * e2 == c2 * e1;

    ab && ac && ae && bc && be && ce
}

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
    a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2)
}

fn solve_3x3_cramer(
    c1: &LinearCoeffs3,
    c2: &LinearCoeffs3,
    c3: &LinearCoeffs3,
) -> Result<(BigRational, BigRational, BigRational), LinearSystemError> {
    let e1 = -c1.d.clone();
    let e2 = -c2.d.clone();
    let e3 = -c3.d.clone();

    let det_a = det3x3(
        &c1.a, &c1.b, &c1.c, &c2.a, &c2.b, &c2.c, &c3.a, &c3.b, &c3.c,
    );

    if det_a.is_zero() {
        let pair1_consistent =
            check_proportional_3(&c1.a, &c1.b, &c1.c, &e1, &c2.a, &c2.b, &c2.c, &e2);
        let pair2_consistent =
            check_proportional_3(&c1.a, &c1.b, &c1.c, &e1, &c3.a, &c3.b, &c3.c, &e3);
        let pair3_consistent =
            check_proportional_3(&c2.a, &c2.b, &c2.c, &e2, &c3.a, &c3.b, &c3.c, &e3);

        if pair1_consistent && pair2_consistent && pair3_consistent {
            return Err(LinearSystemError::InfiniteSolutions);
        }
        return Err(LinearSystemError::NoSolution);
    }

    let det_x = det3x3(&e1, &c1.b, &c1.c, &e2, &c2.b, &c2.c, &e3, &c3.b, &c3.c);
    let det_y = det3x3(&c1.a, &e1, &c1.c, &c2.a, &e2, &c2.c, &c3.a, &e3, &c3.c);
    let det_z = det3x3(&c1.a, &c1.b, &e1, &c2.a, &c2.b, &e2, &c3.a, &c3.b, &e3);

    let x = det_x / &det_a;
    let y = det_y / &det_a;
    let z = det_z / &det_a;
    Ok((x, y, z))
}

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

    let var_indices: Vec<Option<usize>> = vars
        .iter()
        .map(|v| poly.vars.iter().position(|pv| pv == *v))
        .collect();

    let n = vars.len();
    let mut coeffs = vec![BigRational::zero(); n];
    let mut constant = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            constant = &constant + coef;
        } else if total_exp == 1 {
            let mut found = false;
            for (mono_idx, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    for (var_idx, opt_idx) in var_indices.iter().enumerate() {
                        if *opt_idx == Some(mono_idx) {
                            coeffs[var_idx] = &coeffs[var_idx] + coef;
                            found = true;
                            break;
                        }
                    }
                    if !found {
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

    Ok((coeffs, -constant))
}

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

        let mut row = coeffs;
        row.push(b);
        matrix.push(row);
    }

    Ok(matrix)
}

#[allow(clippy::needless_range_loop)]
fn gauss_solve(mut matrix: Vec<Vec<BigRational>>, n: usize) -> LinSolveResult {
    let m = matrix.len();
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    for col in 0..n {
        let mut pivot_found = None;
        for (row, mat_row) in matrix.iter().enumerate().skip(pivot_row) {
            if !mat_row[col].is_zero() {
                pivot_found = Some(row);
                break;
            }
        }

        let Some(pivot_idx) = pivot_found else {
            continue;
        };

        if pivot_idx != pivot_row {
            matrix.swap(pivot_row, pivot_idx);
        }

        pivot_cols.push(col);

        let pivot_val = matrix[pivot_row][col].clone();
        for cell in matrix[pivot_row].iter_mut().take(n + 1) {
            *cell = &*cell / &pivot_val;
        }

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

    for mat_row in matrix.iter().take(m).skip(rank) {
        let all_zero = (0..n).all(|j| mat_row[j].is_zero());
        if all_zero && !mat_row[n].is_zero() {
            return LinSolveResult::Inconsistent;
        }
    }

    if rank < n {
        return LinSolveResult::Infinite;
    }

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

fn solve_nxn_gauss(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    let n = vars.len();
    let matrix = build_augmented_matrix(ctx, exprs, vars)?;
    Ok(gauss_solve(matrix, n))
}

#[cfg(test)]
mod tests {
    use super::{
        is_valid_linear_system_var, parse_linear_system_spec, solve_2x2_linear_system,
        solve_linear_system_spec, solve_nxn_linear_system, split_semicolon_top_level,
        LinSolveResult, LinearSystemSpecError,
    };
    use cas_ast::Expr;
    use num_rational::BigRational;

    #[test]
    fn split_semicolon_top_level_respects_parentheses() {
        let parts = split_semicolon_top_level("x+y=3; solve(a;b); x; y");
        assert_eq!(parts, vec!["x+y=3", " solve(a;b)", " x", " y"]);
    }

    #[test]
    fn valid_linear_system_var_rules() {
        assert!(is_valid_linear_system_var("x"));
        assert!(is_valid_linear_system_var("var_name"));
        assert!(!is_valid_linear_system_var(""));
        assert!(!is_valid_linear_system_var("x1"));
    }

    #[test]
    fn solve_2x2_linear_system_unique_solution() {
        let mut ctx = cas_ast::Context::new();
        let eq1 = match cas_parser::parse_statement("x+y=3", &mut ctx).expect("eq1 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };
        let eq2 = match cas_parser::parse_statement("x-y=1", &mut ctx).expect("eq2 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };

        let expr1 = ctx.add(Expr::Sub(eq1.lhs, eq1.rhs));
        let expr2 = ctx.add(Expr::Sub(eq2.lhs, eq2.rhs));

        let (x, y) = solve_2x2_linear_system(&ctx, expr1, expr2, "x", "y")
            .expect("linear solve should succeed");

        assert_eq!(x, BigRational::from_integer(2.into()));
        assert_eq!(y, BigRational::from_integer(1.into()));
    }

    #[test]
    fn solve_nxn_linear_system_detects_inconsistent() {
        let mut ctx = cas_ast::Context::new();
        let eq1 = match cas_parser::parse_statement("x+y=1", &mut ctx).expect("eq1 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };
        let eq2 = match cas_parser::parse_statement("x+y=2", &mut ctx).expect("eq2 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };

        let exprs = vec![
            ctx.add(Expr::Sub(eq1.lhs, eq1.rhs)),
            ctx.add(Expr::Sub(eq2.lhs, eq2.rhs)),
        ];

        let result = solve_nxn_linear_system(&ctx, &exprs, &["x", "y"]).expect("solver result");
        match result {
            LinSolveResult::Inconsistent => {}
            _ => panic!("expected inconsistent system"),
        }
    }

    #[test]
    fn parse_linear_system_spec_builds_normalized_spec() {
        let mut ctx = cas_ast::Context::new();
        let spec = parse_linear_system_spec(&mut ctx, "x+y=3; x-y=1; x; y").expect("spec parse");
        assert_eq!(spec.vars, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(spec.exprs.len(), 2);
    }

    #[test]
    fn parse_linear_system_spec_rejects_inequality() {
        let mut ctx = cas_ast::Context::new();
        let err =
            parse_linear_system_spec(&mut ctx, "x+y<3; x-y=1; x; y").expect_err("expected error");
        assert!(matches!(
            err,
            LinearSystemSpecError::UnsupportedRelation { position: 1, .. }
        ));
    }

    #[test]
    fn solve_linear_system_spec_2x2_unique() {
        let mut ctx = cas_ast::Context::new();
        let spec = parse_linear_system_spec(&mut ctx, "x+y=3; x-y=1; x; y").expect("spec parse");
        let result = solve_linear_system_spec(&ctx, &spec).expect("solve");

        match result {
            LinSolveResult::Unique(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], BigRational::from_integer(2.into()));
                assert_eq!(values[1], BigRational::from_integer(1.into()));
            }
            _ => panic!("expected unique solution"),
        }
    }
}
