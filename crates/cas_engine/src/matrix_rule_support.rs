//! Matrix rule support helpers shared with engine rule layers.
//!
//! These helpers evaluate matrix operation patterns directly from expression
//! nodes and return structured metadata for caller-owned narration.

use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::matrix::{Matrix, MatrixInverseOutcome};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixShape {
    fn from_matrix(matrix: &Matrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatrixBinaryEval {
    pub left: MatrixShape,
    pub right: MatrixShape,
    pub result: Matrix,
}

impl MatrixBinaryEval {
    pub fn add_desc(&self) -> String {
        format!(
            "Matrix addition: {}×{} + {}×{}",
            self.left.rows, self.left.cols, self.right.rows, self.right.cols
        )
    }

    pub fn sub_desc(&self) -> String {
        format!(
            "Matrix subtraction: {}×{} - {}×{}",
            self.left.rows, self.left.cols, self.right.rows, self.right.cols
        )
    }

    pub fn mul_desc(&self) -> String {
        format!(
            "Matrix multiplication: {}×{} × {}×{} = {}×{}",
            self.left.rows,
            self.left.cols,
            self.right.rows,
            self.right.cols,
            self.result.rows,
            self.result.cols
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarMatrixSide {
    ScalarLeft,
    ScalarRight,
}

#[derive(Debug, Clone)]
pub struct ScalarMatrixEval {
    pub side: ScalarMatrixSide,
    pub matrix: MatrixShape,
    pub result: Matrix,
}

impl ScalarMatrixEval {
    pub fn desc(&self) -> String {
        match self.side {
            ScalarMatrixSide::ScalarLeft => format!(
                "Scalar multiplication: scalar × {}×{} matrix",
                self.matrix.rows, self.matrix.cols
            ),
            ScalarMatrixSide::ScalarRight => format!(
                "Scalar multiplication: {}×{} matrix × scalar",
                self.matrix.rows, self.matrix.cols
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MatrixFunctionEval {
    Determinant {
        shape: MatrixShape,
        value: ExprId,
    },
    Transpose {
        from: MatrixShape,
        to: MatrixShape,
        matrix: Matrix,
    },
    Trace {
        shape: MatrixShape,
        value: ExprId,
    },
    Rank {
        shape: MatrixShape,
        value: ExprId,
    },
    Rref {
        shape: MatrixShape,
        value: ExprId,
    },
    CharPoly {
        shape: MatrixShape,
        value: ExprId,
    },
    Eigenvalues {
        shape: MatrixShape,
        value: ExprId,
    },
    Eigenvectors {
        shape: MatrixShape,
        value: ExprId,
    },
    Nullspace {
        shape: MatrixShape,
        value: ExprId,
    },
    Norm {
        shape: MatrixShape,
        value: ExprId,
    },
    Adjugate {
        shape: MatrixShape,
        value: ExprId,
    },
    Inverse {
        shape: MatrixShape,
        /// `Some(inverse)` for an invertible matrix; `None` when the matrix is provably
        /// singular (determinant zero ⇒ no inverse, rewritten to `undefined`).
        matrix: Option<Matrix>,
    },
}

#[derive(Debug, Clone)]
pub struct MatrixFunctionRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

/// Human-readable description for matrix function rewrites.
pub(crate) fn format_matrix_function_desc(eval: &MatrixFunctionEval) -> String {
    match eval {
        MatrixFunctionEval::Determinant { shape, .. } => {
            format!("det({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Transpose { from, to, .. } => {
            format!(
                "transpose({}×{}) = {}×{}",
                from.rows, from.cols, to.rows, to.cols
            )
        }
        MatrixFunctionEval::Trace { shape, .. } => {
            format!("trace({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Rank { shape, .. } => {
            format!("rank({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Rref { shape, .. } => {
            format!("rref({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::CharPoly { shape, .. } => {
            format!(
                "charpoly({}×{} matrix) = det(λI − A)",
                shape.rows, shape.cols
            )
        }
        MatrixFunctionEval::Eigenvalues { shape, .. } => {
            format!("eigenvalues({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Eigenvectors { shape, .. } => {
            format!("eigenvectors({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Nullspace { shape, .. } => {
            format!("nullspace({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Norm { shape, .. } => {
            format!("norm({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Adjugate { shape, .. } => {
            format!("adjugate({}×{} matrix)", shape.rows, shape.cols)
        }
        MatrixFunctionEval::Inverse { shape, matrix } => {
            if matrix.is_some() {
                format!("inverse({}×{} matrix)", shape.rows, shape.cols)
            } else {
                format!(
                    "inverse({}×{} matrix): singular (determinant 0, no inverse)",
                    shape.rows, shape.cols
                )
            }
        }
    }
}

pub(crate) fn try_eval_matrix_add_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixBinaryEval> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    let left = *left;
    let right = *right;

    let m1 = Matrix::from_expr(ctx, left)?;
    let m2 = Matrix::from_expr(ctx, right)?;
    let result = m1.add(&m2, ctx)?;
    Some(MatrixBinaryEval {
        left: MatrixShape::from_matrix(&m1),
        right: MatrixShape::from_matrix(&m2),
        result,
    })
}

pub(crate) fn try_eval_matrix_sub_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixBinaryEval> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    let left = *left;
    let right = *right;

    let m1 = Matrix::from_expr(ctx, left)?;
    let m2 = Matrix::from_expr(ctx, right)?;
    let result = m1.sub(&m2, ctx)?;
    Some(MatrixBinaryEval {
        left: MatrixShape::from_matrix(&m1),
        right: MatrixShape::from_matrix(&m2),
        result,
    })
}

pub(crate) fn try_eval_matrix_mul_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixBinaryEval> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let left = *left;
    let right = *right;

    let m1 = Matrix::from_expr(ctx, left)?;
    let m2 = Matrix::from_expr(ctx, right)?;
    let result = m1.multiply(&m2, ctx)?;
    Some(MatrixBinaryEval {
        left: MatrixShape::from_matrix(&m1),
        right: MatrixShape::from_matrix(&m2),
        result,
    })
}

/// A literal scalar (number or numeric constant) — never a matrix. Used to recognise
/// a matrix±scalar combination, which has no value (matrices and scalars do not add).
/// A symbolic variable is deliberately NOT treated as a scalar here: it could later bind
/// to a matrix, so `[[…]] + y` is left for downstream rules instead of declared undefined.
fn is_concrete_scalar(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_) | Expr::Constant(_))
}

/// Detect a matrix operation whose operand SHAPES do not conform and return the `undefined`
/// sentinel. A non-conforming matrix op has no mathematical value, so leaving it as an echoed
/// residual would dishonestly report success (`ok:true`) over a non-result. Covers:
/// matrix `±` matrix of different dimensions, matrix `±` scalar (no broadcast), matrix `·`
/// matrix with mismatched inner dimensions, and a NON-square matrix raised to an integer `≥ 2`
/// (including the `M·M → M^2` form the engine manufactures). Returns `None` for well-formed
/// matrix ops (handled by the evaluation rules) and for non-matrix expressions.
pub(crate) fn try_matrix_shape_mismatch_undefined(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            match (Matrix::from_expr(ctx, left), Matrix::from_expr(ctx, right)) {
                // Two matrices: add/sub is defined only for identical dimensions.
                (Some(m1), Some(m2)) => {
                    if m1.rows != m2.rows || m1.cols != m2.cols {
                        return Some(ctx.add(Expr::Constant(Constant::Undefined)));
                    }
                    None
                }
                // Matrix ± concrete scalar: no broadcasting convention ⇒ undefined.
                (Some(_), None) if is_concrete_scalar(ctx, right) => {
                    Some(ctx.add(Expr::Constant(Constant::Undefined)))
                }
                (None, Some(_)) if is_concrete_scalar(ctx, left) => {
                    Some(ctx.add(Expr::Constant(Constant::Undefined)))
                }
                _ => None,
            }
        }
        // Matrix · matrix is defined only when inner dimensions agree (l.cols == r.rows).
        // Scalar · matrix is valid and handled elsewhere, so only guard two literal matrices.
        Expr::Mul(left, right) => {
            if let (Some(m1), Some(m2)) =
                (Matrix::from_expr(ctx, left), Matrix::from_expr(ctx, right))
            {
                if m1.cols != m2.rows {
                    return Some(ctx.add(Expr::Constant(Constant::Undefined)));
                }
            }
            None
        }
        // A matrix power is defined only for a SQUARE base. A non-square base raised to an
        // integer `≥ 2` (repeated self-multiplication) has no value.
        Expr::Pow(base, exp) => {
            let m = Matrix::from_expr(ctx, base)?;
            if m.rows != m.cols {
                let e = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
                if e.is_integer() && e >= num_rational::BigRational::from_integer(2.into()) {
                    return Some(ctx.add(Expr::Constant(Constant::Undefined)));
                }
            }
            None
        }
        _ => None,
    }
}

/// Whether `expr` evaluates to a MATRIX — a literal, a matrix-returning function call
/// (`inverse`/`transpose`/`adjugate`), or a structural combination of such. Used to keep
/// scalar-matrix multiplication from broadcasting a matrix-valued operand as if it were a scalar.
pub(crate) fn is_matrix_valued(ctx: &Context, expr: ExprId) -> bool {
    if Matrix::from_expr(ctx, expr).is_some() {
        return true;
    }
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && matches!(
                    ctx.sym_name(*fn_id),
                    "inverse" | "inv" | "transpose" | "T" | "adjugate" | "adj"
                )
                && is_matrix_valued(ctx, args[0])
        }
        Expr::Neg(inner) => is_matrix_valued(ctx, *inner),
        Expr::Pow(base, _) => is_matrix_valued(ctx, *base),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            is_matrix_valued(ctx, *l) || is_matrix_valued(ctx, *r)
        }
        _ => false,
    }
}

/// Route `M^(-1)` and `c / M` (matrix `M`) to the matrix INVERSE instead of letting scalar
/// arithmetic fabricate `1/[[…]]`: `M^(-1) → inverse(M)`, `c / M → c · inverse(M)`. The
/// `inverse(…)` call is evaluated soundly downstream (numeric → inverse matrix, singular →
/// undefined, non-square / symbolic → honest residual).
/// Identity matrix `Iₙ` as a matrix expression.
fn identity_matrix_expr(ctx: &mut Context, n: usize) -> ExprId {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data.push(if i == j { ctx.num(1) } else { ctx.num(0) });
        }
    }
    ctx.matrix(n, n, data.clone()).unwrap_or_else(|_| {
        ctx.add(Expr::Matrix {
            rows: n,
            cols: n,
            data,
        })
    })
}

/// Integer matrix power `Mⁿ` of a SQUARE matrix. `n = 0 → I`, `n = 1 → M` (any
/// entries); `|n| ≥ 2` is restricted to ALL-NUMERIC matrices, where the repeated
/// product folds exactly and (for `n < 0`) the inverse is exact with a decidable
/// `det ≠ 0` — a symbolic `Mⁿ` is left a residual rather than committing a form
/// without its domain condition. A singular matrix to a negative power is
/// `undefined`.
fn try_matrix_power_expr(ctx: &mut Context, m: &Matrix, n: i64) -> Option<ExprId> {
    const MATRIX_POWER_MAX_EXPONENT: u64 = 64;
    if m.rows != m.cols {
        return None;
    }
    if n == 0 {
        return Some(identity_matrix_expr(ctx, m.rows));
    }
    if n == 1 {
        return Some(m.to_expr(ctx));
    }
    let all_numeric = m
        .data
        .iter()
        .all(|&e| matches!(ctx.get(e), Expr::Number(_) | Expr::Constant(_)));
    if !all_numeric || n.unsigned_abs() > MATRIX_POWER_MAX_EXPONENT {
        return None;
    }
    let (base, repetitions) = if n > 0 {
        (m.clone(), n as usize)
    } else {
        match m.inverse(ctx)? {
            MatrixInverseOutcome::Inverse(inverse) => (inverse, n.unsigned_abs() as usize),
            MatrixInverseOutcome::Singular => {
                return Some(ctx.add(Expr::Constant(Constant::Undefined)))
            }
        }
    };
    let mut result = base.clone();
    for _ in 1..repetitions {
        result = result.multiply(&base, ctx)?;
    }
    Some(result.to_expr(ctx))
}

pub(crate) fn try_rewrite_matrix_reciprocal_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            let matrix = Matrix::from_expr(ctx, base)?;
            let exp_val = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
            if !exp_val.is_integer() {
                return None;
            }
            // Preserve the inverse-CALL form for `-1`: its symbolic case must stay the
            // `inverse(M)` residual (which withholds the `det ≠ 0` condition), not an echoed `M^(-1)`.
            if exp_val == num_rational::BigRational::from_integer((-1).into()) {
                return Some(ctx.call("inverse", vec![base]));
            }
            let n = num_traits::ToPrimitive::to_i64(&exp_val.to_integer())?;
            try_matrix_power_expr(ctx, &matrix, n)
        }
        Expr::Div(num, den) => {
            // `c / M`: a scalar numerator over a matrix denominator. `M / N` (matrix over matrix)
            // is not scalar division and is left alone.
            if Matrix::from_expr(ctx, den).is_none() || is_matrix_valued(ctx, num) {
                return None;
            }
            let inv = ctx.call("inverse", vec![den]);
            Some(ctx.add(Expr::Mul(num, inv)))
        }
        _ => None,
    }
}

pub(crate) fn try_eval_scalar_matrix_mul_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ScalarMatrixEval> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let left = *left;
    let right = *right;

    // Scalar-matrix multiplication requires EXACTLY ONE operand to be a matrix. When BOTH are
    // matrices this is matrix-matrix multiplication (handled by MatrixMultiplyRule); treating one
    // matrix as a scalar broadcasts it over the other's entries, corrupting a valid product into a
    // malformed matrix-of-matrices (`[[1,2],[3,4]] * [[5,6,7],[8,9,10]]`) and even fabricating a
    // finite result for a dimension-mismatched product. Decline so the proper rule (or an honest
    // residual on mismatch) is used.
    let left_is_matrix = Matrix::from_expr(ctx, left).is_some();
    let right_is_matrix = Matrix::from_expr(ctx, right).is_some();
    if left_is_matrix && right_is_matrix {
        return None;
    }
    // The would-be SCALAR operand must be a genuine scalar, not a MATRIX-VALUED expression that
    // simply has not reduced to a literal yet (e.g. `inverse(M)` for a symbolic `M`). Broadcasting
    // such an operand over the other matrix fabricates a matrix-of-matrices
    // (`inverse([[a,b],[c,d]]) * I` → `[[inverse(M), 0], [0, inverse(M)]]`). Decline so it stays an
    // honest residual; a numeric `inverse(...)` evaluates to a literal first and goes via matmul.
    if right_is_matrix && is_matrix_valued(ctx, left) {
        return None;
    }
    if left_is_matrix && is_matrix_valued(ctx, right) {
        return None;
    }

    if let Some(matrix) = Matrix::from_expr(ctx, right) {
        let result = matrix.scalar_mul(left, ctx);
        return Some(ScalarMatrixEval {
            side: ScalarMatrixSide::ScalarLeft,
            matrix: MatrixShape::from_matrix(&matrix),
            result,
        });
    }

    if let Some(matrix) = Matrix::from_expr(ctx, left) {
        let result = matrix.scalar_mul(right, ctx);
        return Some(ScalarMatrixEval {
            side: ScalarMatrixSide::ScalarRight,
            matrix: MatrixShape::from_matrix(&matrix),
            result,
        });
    }

    None
}

/// Eigenvalues of a NUMERIC square matrix: the roots of `charpoly(A)`, returned
/// as a 1×n row of values. Reuses the exact rational-root finder and the
/// quadratic formula (both `cas_solver_core`). Peels every rational eigenvalue
/// exactly, then closes a degree-1 or degree-2 deflated factor; an irreducible
/// factor of degree ≥ 3 (needing Cardano/Galois machinery) declines the WHOLE
/// computation to an honest residual rather than reporting a partial spectrum.
/// Symbolic matrices decline (the characteristic polynomial is not numeric).
pub(crate) fn try_matrix_eigenvalues(ctx: &mut Context, matrix: &Matrix) -> Option<ExprId> {
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    use num_rational::BigRational;

    if matrix.rows != matrix.cols {
        return None;
    }
    let var = "x";
    let charpoly = matrix.charpoly(ctx, var)?;
    let poly = cas_math::polynomial::Polynomial::from_expr(ctx, charpoly, var).ok()?;
    if poly.degree() == 0 {
        return None;
    }

    let (rational_roots, remaining) = find_rational_roots(poly.coeffs.clone(), 4096);
    let mut eigenvalues: Vec<ExprId> = rational_roots
        .iter()
        .map(|r| rational_to_expr(ctx, r))
        .collect();

    match remaining.len().saturating_sub(1) {
        0 => {}
        1 => {
            // c1·x + c0 = 0 ⇒ x = −c0/c1.
            let root = -&remaining[0] / &remaining[1];
            let root_expr = rational_to_expr(ctx, &root);
            eigenvalues.push(root_expr);
        }
        2 => {
            // c2·x² + c1·x + c0: x = (−c1 ± √(c1² − 4·c2·c0)) / (2·c2).
            let c0 = remaining[0].clone();
            let c1 = remaining[1].clone();
            let c2 = remaining[2].clone();
            let four = BigRational::from_integer(4.into());
            let two = BigRational::from_integer(2.into());
            let discriminant = &c1 * &c1 - &four * &c2 * &c0;
            // Real-domain engine: a negative discriminant means this factor's eigenvalues are a
            // complex-conjugate pair (no REAL eigenvalues). Decline the whole spectrum as an honest
            // residual rather than emit non-real values outside the real-domain scope.
            if num_traits::Signed::is_negative(&discriminant) {
                return None;
            }
            let neg_c1 = rational_to_expr(ctx, &(-c1));
            let two_c2 = rational_to_expr(ctx, &(&two * &c2));
            let disc_expr = rational_to_expr(ctx, &discriminant);
            let sqrt_disc = sqrt_expr(ctx, disc_expr);
            let sum = ctx.add(Expr::Add(neg_c1, sqrt_disc));
            let plus = ctx.add(Expr::Div(sum, two_c2));
            let diff = ctx.add(Expr::Sub(neg_c1, sqrt_disc));
            let minus = ctx.add(Expr::Div(diff, two_c2));
            eigenvalues.push(plus);
            eigenvalues.push(minus);
        }
        _ => return None,
    }

    if eigenvalues.is_empty() {
        return None;
    }
    let count = eigenvalues.len();
    Some(
        ctx.matrix(1, count, eigenvalues.clone())
            .unwrap_or_else(|_| {
                ctx.add(Expr::Matrix {
                    rows: 1,
                    cols: count,
                    data: eigenvalues,
                })
            }),
    )
}

/// Reduce a rational matrix to RREF in place and return its pivot columns.
fn rational_rref_in_place(m: &mut [Vec<num_rational::BigRational>], cols: usize) -> Vec<usize> {
    use num_traits::Zero;
    let rows = m.len();
    let mut pivots = Vec::new();
    let (mut pr, mut pc) = (0usize, 0usize);
    while pr < rows && pc < cols {
        match (pr..rows).find(|&i| !m[i][pc].is_zero()) {
            None => pc += 1,
            Some(p) => {
                m.swap(pr, p);
                let pivot_val = m[pr][pc].clone();
                for entry in &mut m[pr] {
                    *entry = &*entry / &pivot_val;
                }
                let pivot_row = m[pr].clone();
                for (i, row) in m.iter_mut().enumerate() {
                    if i != pr && !row[pc].is_zero() {
                        let factor = row[pc].clone();
                        for (entry, pivot_entry) in row.iter_mut().zip(&pivot_row) {
                            *entry -= &factor * pivot_entry;
                        }
                    }
                }
                pivots.push(pc);
                pr += 1;
                pc += 1;
            }
        }
    }
    pivots
}

/// Basis of the null space of a rational matrix, from its RREF: one vector per
/// free column (free var = 1, pivot vars = −rref[row][free]).
fn rational_null_space(
    mut m: Vec<Vec<num_rational::BigRational>>,
    cols: usize,
) -> Vec<Vec<num_rational::BigRational>> {
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    let pivots = rational_rref_in_place(&mut m, cols);
    let pivot_cols: std::collections::HashSet<usize> = pivots.iter().copied().collect();
    let mut basis = Vec::new();
    for free in (0..cols).filter(|c| !pivot_cols.contains(c)) {
        let mut v = vec![BigRational::zero(); cols];
        v[free] = BigRational::one();
        for (row, &pivot_col) in pivots.iter().enumerate() {
            v[pivot_col] = -&m[row][free];
        }
        basis.push(v);
    }
    basis
}

/// Eigenvectors of a NUMERIC square matrix with ALL-RATIONAL eigenvalues: for
/// each distinct rational eigenvalue λ, the null-space basis of `A − λI` (exact
/// rational RREF). Output is a matrix whose ROWS are the eigenvectors. Declines
/// (honest residual) when any entry is non-numeric, or any eigenvalue is
/// irrational/complex — a surd eigenvalue would need a surd-coefficient RREF,
/// out of this exact-rational scope.
pub(crate) fn try_matrix_eigenvectors(ctx: &mut Context, matrix: &Matrix) -> Option<ExprId> {
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    use num_rational::BigRational;

    if matrix.rows != matrix.cols {
        return None;
    }
    let n = matrix.rows;
    // Read A as an exact rational matrix; bail on any symbolic entry.
    let mut a: Vec<Vec<BigRational>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            row.push(cas_math::numeric_eval::as_rational_const(
                ctx,
                matrix.data[i * n + j],
            )?);
        }
        a.push(row);
    }

    let charpoly = matrix.charpoly(ctx, "x")?;
    let poly = cas_math::polynomial::Polynomial::from_expr(ctx, charpoly, "x").ok()?;

    // Gather ALL RATIONAL eigenvalues: `find_rational_roots` peels rational roots only down to a
    // degree-≤2 factor (it stops at degree 2), so the deflated quadratic/linear remainder must be
    // solved here. The quadratic's roots are rational iff its discriminant is a perfect square; an
    // irrational or complex factor declines the whole computation (a surd eigenvalue would need a
    // surd-coefficient RREF, out of this exact-rational scope).
    let (mut rational_eigenvalues, remaining) = find_rational_roots(poly.coeffs.clone(), 4096);
    match remaining.len().saturating_sub(1) {
        0 => {}
        1 => rational_eigenvalues.push(-&remaining[0] / &remaining[1]),
        2 => {
            let four = BigRational::from_integer(4.into());
            let two = BigRational::from_integer(2.into());
            let discriminant =
                &remaining[1] * &remaining[1] - &four * &remaining[2] * &remaining[0];
            let root = cas_math::perfect_square_support::rational_sqrt(&discriminant)?;
            let two_c2 = &two * &remaining[2];
            rational_eigenvalues.push((-&remaining[1] + &root) / &two_c2);
            rational_eigenvalues.push((-&remaining[1] - &root) / &two_c2);
        }
        _ => return None,
    }

    let mut eigenvectors: Vec<Vec<BigRational>> = Vec::new();
    let mut processed: Vec<BigRational> = Vec::new();
    for lambda in &rational_eigenvalues {
        if processed.contains(lambda) {
            continue; // one eigenspace per distinct eigenvalue
        }
        processed.push(lambda.clone());
        let mut shifted = a.clone();
        for (i, row) in shifted.iter_mut().enumerate() {
            row[i] -= lambda;
        }
        for basis_vector in rational_null_space(shifted, n) {
            eigenvectors.push(basis_vector);
        }
    }

    if eigenvectors.is_empty() {
        return None;
    }
    let rows = eigenvectors.len();
    let data: Vec<ExprId> = eigenvectors
        .into_iter()
        .flatten()
        .map(|value| rational_to_expr(ctx, &value))
        .collect();
    Some(ctx.matrix(rows, n, data.clone()).unwrap_or_else(|_| {
        ctx.add(Expr::Matrix {
            rows,
            cols: n,
            data,
        })
    }))
}

/// Null space (kernel) of a NUMERIC matrix `A`: a basis of `{x : A·x = 0}` by exact
/// rational RREF, returned as a matrix whose ROWS are the basis vectors. A trivial
/// kernel (full column rank) is represented by the single zero vector. Declines
/// (honest residual) on any symbolic entry.
pub(crate) fn try_matrix_nullspace(ctx: &mut Context, matrix: &Matrix) -> Option<ExprId> {
    use cas_solver_core::rational_roots::rational_to_expr;
    use num_rational::BigRational;

    let (rows, cols) = (matrix.rows, matrix.cols);
    let mut a: Vec<Vec<BigRational>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(cas_math::numeric_eval::as_rational_const(
                ctx,
                matrix.data[i * cols + j],
            )?);
        }
        a.push(row);
    }

    let basis = rational_null_space(a, cols);
    let (out_rows, data): (usize, Vec<ExprId>) = if basis.is_empty() {
        (1, (0..cols).map(|_| ctx.num(0)).collect())
    } else {
        let out_rows = basis.len();
        let data = basis
            .into_iter()
            .flatten()
            .map(|value| rational_to_expr(ctx, &value))
            .collect();
        (out_rows, data)
    };
    Some(
        ctx.matrix(out_rows, cols, data.clone())
            .unwrap_or_else(|_| {
                ctx.add(Expr::Matrix {
                    rows: out_rows,
                    cols,
                    data,
                })
            }),
    )
}

/// Wronskian `W(f₁,…,fₙ)(x) = det[ dⁱ/dxⁱ fⱼ ]` — the determinant of the matrix
/// whose rows are the 0th…(n−1)th derivatives of the functions. `[sin(x), cos(x)]
/// → −1`. Reuses symbolic differentiation and the (symbolic) determinant; a
/// function that cannot be differentiated declines to a residual.
pub(crate) fn try_wronskian_expr(
    ctx: &mut Context,
    list_expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;

    let functions = Matrix::from_expr(ctx, list_expr)?;
    let n = functions.data.len();
    if n == 0 {
        return None;
    }
    let mut data: Vec<ExprId> = Vec::with_capacity(n * n);
    let mut current = functions.data.clone();
    data.extend_from_slice(&current);
    for _ in 1..n {
        let mut next = Vec::with_capacity(n);
        for &f in &current {
            next.push(differentiate_symbolic_expr(ctx, f, var)?);
        }
        current = next;
        data.extend_from_slice(&current);
    }
    Matrix {
        rows: n,
        cols: n,
        data,
    }
    .determinant(ctx)
}

/// Cell cap for componentwise maps over a `Matrix` node (jacobian precedent:
/// 8×8 = 64). Beyond it the map declines to an honest residual instead of
/// requesting an unbounded budget exemption.
pub(crate) const COMPONENTWISE_MAX_CELLS: usize = 64;

/// Map `f` over every component of a `Matrix` expression, ALL-OR-NOTHING: if any
/// component maps to `None` the whole map declines (honest residual — never a
/// half-transformed matrix). This is the reusable primitive of the vectorial
/// verbs (Fase 2 V1): componentwise diff/integrate and the verb assemblers all
/// build on it.
pub(crate) fn map_matrix_components(
    ctx: &mut Context,
    matrix_expr: ExprId,
    mut f: impl FnMut(&mut Context, ExprId) -> Option<ExprId>,
) -> Option<ExprId> {
    let m = Matrix::from_expr(ctx, matrix_expr)?;
    if m.data.is_empty() || m.data.len() > COMPONENTWISE_MAX_CELLS {
        return None;
    }
    let mut data = Vec::with_capacity(m.data.len());
    for &entry in &m.data {
        data.push(f(ctx, entry)?);
    }
    Some(
        Matrix {
            rows: m.rows,
            cols: m.cols,
            data,
        }
        .to_expr(ctx),
    )
}

/// Var cap for the vectorial verbs (scoping D8: vars ≤ 8; output cells stay
/// within `COMPONENTWISE_MAX_CELLS`). Beyond it a verb declines honestly.
pub(crate) const VERB_MAX_VARS: usize = 8;

/// Gradient of a SCALAR field: `∇f = [∂f/∂v₁, …, ∂f/∂vₙ]` as an n×1 COLUMN
/// (Fase 2 V3 — the parser's own `[x,y]` convention). A Matrix field is
/// jacobian territory (V4) and declines here; a component that cannot be
/// differentiated declines the whole call (all-or-nothing).
pub(crate) fn try_gradient_expr(
    ctx: &mut Context,
    field: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    if var_names.is_empty() || var_names.len() > VERB_MAX_VARS {
        return None;
    }
    if matches!(ctx.get(field), Expr::Matrix { .. }) {
        return None;
    }
    let mut data = Vec::with_capacity(var_names.len());
    for var in var_names {
        data.push(differentiate_symbolic_expr(ctx, field, var)?);
    }
    Some(
        Matrix {
            rows: data.len(),
            cols: 1,
            data,
        }
        .to_expr(ctx),
    )
}

/// Jacobian of a VECTOR field `[f₁,…,f_m]` w.r.t. `[v₁,…,v_n]`: the m×n matrix
/// with ROWS = functions and COLUMNS = variables — `J[i][j] = ∂fᵢ/∂vⱼ` (Fase 2
/// V4, the standard orientation, pinned in fixtures BEFORE coding). A scalar
/// target declines (gradient owns scalars); a general matrix target declines;
/// all-or-nothing; caps: functions ≤ 8, vars ≤ `VERB_MAX_VARS` (≤ 64 cells).
pub(crate) fn try_jacobian_expr(
    ctx: &mut Context,
    fields: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    if var_names.is_empty() || var_names.len() > VERB_MAX_VARS {
        return None;
    }
    let m = Matrix::from_expr(ctx, fields)?;
    if m.rows != 1 && m.cols != 1 {
        return None;
    }
    let funcs = m.data.clone();
    if funcs.is_empty() || funcs.len() > VERB_MAX_VARS {
        return None;
    }
    let mut data = Vec::with_capacity(funcs.len() * var_names.len());
    for &f in &funcs {
        for var in var_names {
            data.push(differentiate_symbolic_expr(ctx, f, var)?);
        }
    }
    Some(
        Matrix {
            rows: funcs.len(),
            cols: var_names.len(),
            data,
        }
        .to_expr(ctx),
    )
}

/// Hessian of a SCALAR field: the n×n matrix `H[i][j] = ∂²f/∂vᵢ∂vⱼ` (Fase 2
/// V4) — computed DIRECTLY as the jacobian of the internal gradient, without
/// re-entering the rewrite pipeline. Matrix targets decline (vector fields have
/// no curriculum hessian here); all-or-nothing.
pub(crate) fn try_hessian_expr(
    ctx: &mut Context,
    field: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    if var_names.is_empty() || var_names.len() > VERB_MAX_VARS {
        return None;
    }
    if matches!(ctx.get(field), Expr::Matrix { .. }) {
        return None;
    }
    let mut gradient = Vec::with_capacity(var_names.len());
    for var in var_names {
        gradient.push(differentiate_symbolic_expr(ctx, field, var)?);
    }
    let n = var_names.len();
    let mut data = Vec::with_capacity(n * n);
    for &g in &gradient {
        for var in var_names {
            data.push(differentiate_symbolic_expr(ctx, g, var)?);
        }
    }
    Some(
        Matrix {
            rows: n,
            cols: n,
            data,
        }
        .to_expr(ctx),
    )
}

/// Divergence of a vector field: `∇·F = Σ ∂Fᵢ/∂vᵢ` — REQUIRES #components ==
/// #vars (a mismatch is an honest residual, never `undefined`: that verdict is
/// reserved for malformed ALGEBRA, Fase 2 V5 policy). Scalar output, computed
/// directly (no jacobian assembly); all-or-nothing.
pub(crate) fn try_divergence_expr(
    ctx: &mut Context,
    fields: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    if var_names.is_empty() || var_names.len() > VERB_MAX_VARS {
        return None;
    }
    let m = Matrix::from_expr(ctx, fields)?;
    if m.rows != 1 && m.cols != 1 {
        return None;
    }
    if m.data.len() != var_names.len() {
        return None;
    }
    let comps = m.data.clone();
    let mut sum = ctx.num(0);
    for (f, var) in comps.iter().zip(var_names) {
        let d = differentiate_symbolic_expr(ctx, *f, var)?;
        sum = ctx.add(Expr::Add(sum, d));
    }
    Some(sum)
}

/// Laplacian of a SCALAR field: `Δf = Σ ∂²f/∂vᵢ²` (div ∘ grad computed
/// internally, no pipeline re-entry). A vector field declines — the
/// vector-laplacian is a named future rung (scope-out), not an error.
pub(crate) fn try_laplacian_expr(
    ctx: &mut Context,
    field: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    if var_names.is_empty() || var_names.len() > VERB_MAX_VARS {
        return None;
    }
    if matches!(ctx.get(field), Expr::Matrix { .. }) {
        return None;
    }
    let mut sum = ctx.num(0);
    for var in var_names {
        let d1 = differentiate_symbolic_expr(ctx, field, var)?;
        let d2 = differentiate_symbolic_expr(ctx, d1, var)?;
        sum = ctx.add(Expr::Add(sum, d2));
    }
    Some(sum)
}

/// Curl of a vector field (Fase 2 V6). With `F = [F₀,F₁,F₂]` over `[v₀,v₁,v₂]`:
/// `∇×F = [∂F₂/∂v₁ − ∂F₁/∂v₂, ∂F₀/∂v₂ − ∂F₂/∂v₀, ∂F₁/∂v₀ − ∂F₀/∂v₁]` as a 3×1
/// COLUMN. The 2D case is the SCALAR `∂Q/∂v₀ − ∂P/∂v₁` (curriculum convention —
/// NEVER zero-padded to 3D: that would assert a shape the input does not have).
/// Any other component/var pairing declines honestly; all-or-nothing.
pub(crate) fn try_curl_expr(
    ctx: &mut Context,
    fields: ExprId,
    var_names: &[String],
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    let m = Matrix::from_expr(ctx, fields)?;
    if m.rows != 1 && m.cols != 1 {
        return None;
    }
    let f = m.data.clone();
    match (f.len(), var_names.len()) {
        (2, 2) => {
            let dq_dx = differentiate_symbolic_expr(ctx, f[1], &var_names[0])?;
            let dp_dy = differentiate_symbolic_expr(ctx, f[0], &var_names[1])?;
            Some(ctx.add(Expr::Sub(dq_dx, dp_dy)))
        }
        (3, 3) => {
            let a = differentiate_symbolic_expr(ctx, f[2], &var_names[1])?;
            let b = differentiate_symbolic_expr(ctx, f[1], &var_names[2])?;
            let c1 = ctx.add(Expr::Sub(a, b));
            let a = differentiate_symbolic_expr(ctx, f[0], &var_names[2])?;
            let b = differentiate_symbolic_expr(ctx, f[2], &var_names[0])?;
            let c2 = ctx.add(Expr::Sub(a, b));
            let a = differentiate_symbolic_expr(ctx, f[1], &var_names[0])?;
            let b = differentiate_symbolic_expr(ctx, f[0], &var_names[1])?;
            let c3 = ctx.add(Expr::Sub(a, b));
            Some(
                Matrix {
                    rows: 3,
                    cols: 1,
                    data: vec![c1, c2, c3],
                }
                .to_expr(ctx),
            )
        }
        _ => None,
    }
}

/// Componentwise derivative of a vector/matrix target: `d/dx [f₁, …] = [f₁′, …]`
/// (Fase 2 V1). A component that cannot be differentiated declines the WHOLE
/// call (all-or-nothing), keeping `diff([...], x)` an honest residual.
pub(crate) fn try_componentwise_diff_matrix(
    ctx: &mut Context,
    matrix_expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    map_matrix_components(ctx, matrix_expr, |ctx, entry| {
        differentiate_symbolic_expr(ctx, entry, var)
    })
}

/// Binary (2-argument) matrix/vector operations: `dot`, `cross`, `linsolve`.
/// Returns the result expression plus a description, or `None` to leave the call
/// as an honest residual (mismatched shapes, a singular/inconsistent system, etc.).
pub(crate) fn try_rewrite_matrix_binary_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixFunctionRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let name = ctx.sym_name(*fn_id).to_string();
    let args = args.clone();
    if args.len() != 2 {
        return None;
    }
    let u = Matrix::from_expr(ctx, args[0])?;
    let v = Matrix::from_expr(ctx, args[1])?;
    let (rewritten, desc) = match name.as_str() {
        "dot" => (matrix_dot(ctx, &u, &v)?, "dot product".to_string()),
        "cross" => (matrix_cross(ctx, &u, &v)?, "cross product".to_string()),
        // `matmul` sat in the eval gate with no dispatch arm (silent residual while `A*B`
        // evaluated) — the live reproduction of the gate-without-rule gotcha. Same math as
        // the `*` operator; mismatched shapes decline to an honest residual.
        "matmul" => {
            let product = u.multiply(&v, ctx)?;
            (
                product.to_expr(ctx),
                format!(
                    "matrix multiplication ({}×{} · {}×{})",
                    u.rows, u.cols, v.rows, v.cols
                ),
            )
        }
        "linsolve" => (
            matrix_linsolve(ctx, &u, &v)?,
            "solve linear system A·x = b".to_string(),
        ),
        "proj" | "projection" => (
            matrix_proj(ctx, &u, &v)?,
            "vector projection of u onto v".to_string(),
        ),
        "angle" => (
            matrix_angle(ctx, &u, &v)?,
            "angle between two vectors".to_string(),
        ),
        _ => return None,
    };
    Some(MatrixFunctionRewrite { rewritten, desc })
}

/// Vector projection of `u` onto `v`: `proj_v(u) = (⟨u,v⟩/⟨v,v⟩)·v`, returned in
/// `v`'s shape. The direction `v` must be a KNOWN non-zero numeric vector (so the
/// `⟨v,v⟩≠0` divisor is a verified constant). A zero or symbolic direction declines
/// to an honest residual; symbolic `u` also stays residual (the rule only commits
/// the unfolded projection when both operands are numeric, per the anti-worsen gate).
fn matrix_proj(ctx: &mut Context, u: &Matrix, v: &Matrix) -> Option<ExprId> {
    use num_traits::Zero;
    if u.data.is_empty() || u.data.len() != v.data.len() {
        return None;
    }
    let dot_uv = matrix_dot(ctx, u, v)?;
    let dot_vv = matrix_dot(ctx, v, v)?;
    let divisor = cas_math::numeric_eval::as_rational_const(ctx, dot_vv)?;
    if divisor.is_zero() {
        return None;
    }
    let coeff = ctx.add(Expr::Div(dot_uv, dot_vv));
    let data: Vec<ExprId> = v
        .data
        .iter()
        .map(|&e| ctx.add(Expr::Mul(coeff, e)))
        .collect();
    Some(
        ctx.matrix(v.rows, v.cols, data.clone())
            .unwrap_or_else(|_| {
                ctx.add(Expr::Matrix {
                    rows: v.rows,
                    cols: v.cols,
                    data,
                })
            }),
    )
}

/// Angle between two vectors: `arccos(⟨u,v⟩ / (‖u‖·‖v‖))` with
/// `‖u‖·‖v‖ = sqrt(⟨u,u⟩·⟨v,v⟩)`. Both vectors must be KNOWN non-zero numeric
/// vectors (the formula is undefined for a zero vector and the norm needs a real
/// value); the engine folds `arccos` at the standard cosines (0→π/2, 1→0, ½→π/3,
/// √2/2→π/4, √3/2→π/6). Symbolic / zero vectors decline.
fn matrix_angle(ctx: &mut Context, u: &Matrix, v: &Matrix) -> Option<ExprId> {
    use num_traits::Zero;
    if u.data.is_empty() || u.data.len() != v.data.len() {
        return None;
    }
    let dot_uv = matrix_dot(ctx, u, v)?;
    let dot_uu = matrix_dot(ctx, u, u)?;
    let dot_vv = matrix_dot(ctx, v, v)?;
    let norm_u_sq = cas_math::numeric_eval::as_rational_const(ctx, dot_uu)?;
    let norm_v_sq = cas_math::numeric_eval::as_rational_const(ctx, dot_vv)?;
    if norm_u_sq.is_zero() || norm_v_sq.is_zero() {
        return None;
    }
    let product = ctx.add(Expr::Mul(dot_uu, dot_vv));
    let denom = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![product]);
    let ratio = ctx.add(Expr::Div(dot_uv, denom));
    Some(ctx.call_builtin(cas_ast::BuiltinFn::Arccos, vec![ratio]))
}

/// Dot product `Σ uᵢ·vᵢ` of two equal-length vectors (entries read row-major, so a
/// column `[a,b]` and a row `[[a,b]]` both work). Folds numerically, stays exact
/// symbolically. Returns `None` on a length mismatch.
fn matrix_dot(ctx: &mut Context, u: &Matrix, v: &Matrix) -> Option<ExprId> {
    if u.data.is_empty() || u.data.len() != v.data.len() {
        return None;
    }
    let mut sum = ctx.num(0);
    for (&a, &b) in u.data.iter().zip(&v.data) {
        let product = ctx.add(Expr::Mul(a, b));
        sum = ctx.add(Expr::Add(sum, product));
    }
    Some(sum)
}

/// Cross product of two 3-vectors, returned as a 3×1 column.
fn matrix_cross(ctx: &mut Context, u: &Matrix, v: &Matrix) -> Option<ExprId> {
    if u.data.len() != 3 || v.data.len() != 3 {
        return None;
    }
    let (u1, u2, u3) = (u.data[0], u.data[1], u.data[2]);
    let (v1, v2, v3) = (v.data[0], v.data[1], v.data[2]);
    let component = |ctx: &mut Context, a, b, c, d| -> ExprId {
        let ad = ctx.add(Expr::Mul(a, b));
        let bc = ctx.add(Expr::Mul(c, d));
        ctx.add(Expr::Sub(ad, bc))
    };
    let c1 = component(ctx, u2, v3, u3, v2);
    let c2 = component(ctx, u3, v1, u1, v3);
    let c3 = component(ctx, u1, v2, u2, v1);
    let data = vec![c1, c2, c3];
    Some(ctx.matrix(3, 1, data.clone()).unwrap_or_else(|_| {
        ctx.add(Expr::Matrix {
            rows: 3,
            cols: 1,
            data,
        })
    }))
}

/// Solve the NUMERIC square system `A·x = b` by exact rational RREF of the
/// augmented matrix `[A|b]`. Returns the solution as an n×1 column ONLY when it is
/// UNIQUE (A invertible); a singular or inconsistent system declines to an honest
/// residual rather than guessing. Symbolic entries decline.
fn matrix_linsolve(ctx: &mut Context, a: &Matrix, b: &Matrix) -> Option<ExprId> {
    use cas_solver_core::rational_roots::rational_to_expr;
    use num_rational::BigRational;

    if a.rows != a.cols || a.rows == 0 || b.data.len() != a.rows {
        return None;
    }
    let n = a.rows;
    let mut augmented: Vec<Vec<BigRational>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        for j in 0..n {
            row.push(cas_math::numeric_eval::as_rational_const(
                ctx,
                a.data[i * n + j],
            )?);
        }
        row.push(cas_math::numeric_eval::as_rational_const(ctx, b.data[i])?);
        augmented.push(row);
    }

    let pivots = rational_rref_in_place(&mut augmented, n + 1);
    // Unique iff there are n pivots and none lands in the augmented (b) column — i.e. A has full
    // rank and the system is consistent. Otherwise the solution set is empty or infinite: decline.
    if pivots.len() != n || pivots.contains(&n) {
        return None;
    }
    let data: Vec<ExprId> = (0..n)
        .map(|i| rational_to_expr(ctx, &augmented[i][n]))
        .collect();
    Some(ctx.matrix(n, 1, data.clone()).unwrap_or_else(|_| {
        ctx.add(Expr::Matrix {
            rows: n,
            cols: 1,
            data,
        })
    }))
}

pub(crate) fn try_eval_matrix_function_expr(
    ctx: &mut Context,
    expr: ExprId,
    complex_enabled: bool,
) -> Option<MatrixFunctionEval> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let name = ctx.sym_name(*fn_id).to_string();
    let args = args.clone();
    if args.len() != 1 {
        return None;
    }

    let matrix = Matrix::from_expr(ctx, args[0])?;
    let shape = MatrixShape::from_matrix(&matrix);

    match name.as_str() {
        "det" | "determinant" => matrix
            .determinant(ctx)
            .map(|value| MatrixFunctionEval::Determinant { shape, value }),
        "transpose" | "T" => {
            let transposed = matrix.transpose();
            let to = MatrixShape::from_matrix(&transposed);
            Some(MatrixFunctionEval::Transpose {
                from: shape,
                to,
                matrix: transposed,
            })
        }
        "trace" | "tr" => matrix
            .trace(ctx)
            .map(|value| MatrixFunctionEval::Trace { shape, value }),
        "rank" => matrix
            .rank(ctx)
            .map(|value| MatrixFunctionEval::Rank { shape, value }),
        "rref" => matrix
            .rref(ctx)
            .map(|value| MatrixFunctionEval::Rref { shape, value }),
        "charpoly" => matrix
            .charpoly(ctx, "lambda")
            .map(|value| MatrixFunctionEval::CharPoly { shape, value }),
        "eigenvalues" | "eigvals" | "eig" => try_matrix_eigenvalues(ctx, &matrix)
            .map(|value| MatrixFunctionEval::Eigenvalues { shape, value }),
        "eigenvectors" | "eigvecs" => try_matrix_eigenvectors(ctx, &matrix)
            .map(|value| MatrixFunctionEval::Eigenvectors { shape, value }),
        "nullspace" | "null" | "kernel" => try_matrix_nullspace(ctx, &matrix)
            .map(|value| MatrixFunctionEval::Nullspace { shape, value }),
        "norm" => matrix
            .norm_in_domain(ctx, complex_enabled)
            .map(|value| MatrixFunctionEval::Norm { shape, value }),
        // |v| of a VECTOR is its Euclidean norm (Fase 2 V7a) — inheriting V0's domain
        // decision wholesale via `norm_in_domain` (never re-deciding it). A general
        // matrix keeps the honest residual: the matrix modulus is NOT the Frobenius
        // norm, and claiming it would assert semantics we do not have.
        "abs" => {
            if shape.rows == 1 || shape.cols == 1 {
                matrix
                    .norm_in_domain(ctx, complex_enabled)
                    .map(|value| MatrixFunctionEval::Norm { shape, value })
            } else {
                None
            }
        }
        "adjugate" | "adj" => matrix
            .adjugate(ctx)
            .map(|value| MatrixFunctionEval::Adjugate { shape, value }),
        "inverse" | "inv" => match matrix.inverse(ctx)? {
            MatrixInverseOutcome::Inverse(inv) => Some(MatrixFunctionEval::Inverse {
                shape,
                matrix: Some(inv),
            }),
            MatrixInverseOutcome::Singular => Some(MatrixFunctionEval::Inverse {
                shape,
                matrix: None,
            }),
        },
        _ => None,
    }
}

/// Evaluate matrix function calls and materialize a direct rewrite payload.
pub(crate) fn try_rewrite_matrix_function_rule_expr(
    ctx: &mut Context,
    expr: ExprId,
    complex_enabled: bool,
) -> Option<MatrixFunctionRewrite> {
    match try_eval_matrix_function_expr(ctx, expr, complex_enabled)? {
        MatrixFunctionEval::Determinant { shape, value } => {
            let desc =
                format_matrix_function_desc(&MatrixFunctionEval::Determinant { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Transpose { from, to, matrix } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Transpose {
                from,
                to,
                matrix: matrix.clone(),
            });
            Some(MatrixFunctionRewrite {
                rewritten: matrix.to_expr(ctx),
                desc,
            })
        }
        MatrixFunctionEval::Trace { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Trace { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Rank { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Rank { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Rref { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Rref { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::CharPoly { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::CharPoly { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Eigenvalues { shape, value } => {
            let desc =
                format_matrix_function_desc(&MatrixFunctionEval::Eigenvalues { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Eigenvectors { shape, value } => {
            let desc =
                format_matrix_function_desc(&MatrixFunctionEval::Eigenvectors { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Nullspace { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Nullspace { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Norm { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Norm { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Adjugate { shape, value } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Adjugate { shape, value });
            Some(MatrixFunctionRewrite {
                rewritten: value,
                desc,
            })
        }
        MatrixFunctionEval::Inverse { shape, matrix } => {
            let desc = format_matrix_function_desc(&MatrixFunctionEval::Inverse {
                shape,
                matrix: matrix.clone(),
            });
            // Invertible ⇒ the inverse matrix; provably singular ⇒ `undefined`.
            let rewritten = match matrix {
                Some(m) => m.to_expr(ctx),
                None => ctx.add(Expr::Constant(Constant::Undefined)),
            };
            Some(MatrixFunctionRewrite { rewritten, desc })
        }
    }
}

/// Rewrite helper for transpose-of-product identity:
/// `transpose(matmul(A, B)) -> matmul(transpose(B), transpose(A))`.
pub(crate) fn try_rewrite_transpose_product_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let fn_id = *fn_id;
    let args = args.clone();
    let name = ctx.sym_name(fn_id);
    if (name != "transpose" && name != "T") || args.len() != 1 {
        return None;
    }

    let Expr::Function(inner_fn_id, inner_args) = ctx.get(args[0]) else {
        return None;
    };
    let inner_fn_id = *inner_fn_id;
    let inner_args = inner_args.clone();
    if ctx.sym_name(inner_fn_id) != "matmul" || inner_args.len() != 2 {
        return None;
    }

    let a = inner_args[0];
    let b = inner_args[1];
    let transposed_b = ctx.call("transpose", vec![b]);
    let transposed_a = ctx.call("transpose", vec![a]);
    Some(ctx.call("matmul", vec![transposed_b, transposed_a]))
}

/// Rewrite helper for transpose-of-product identity with canonical description.
pub(crate) fn try_rewrite_transpose_product_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixFunctionRewrite> {
    let rewritten = try_rewrite_transpose_product_expr(ctx, expr)?;
    Some(MatrixFunctionRewrite {
        rewritten,
        desc: "(AB)^T = B^T·A^T".to_string(),
    })
}

#[cfg(test)]
#[path = "matrix_rule_support_tests.rs"]
mod tests;
