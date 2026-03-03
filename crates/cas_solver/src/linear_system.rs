use cas_ast::{Context, ExprId};
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
