use cas_ast::{Context, Expr, ExprId};

/// Matrix wrapper for operations
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<ExprId>,
}

#[inline]
fn mul2_raw(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    ctx.add_raw(Expr::Mul(a, b))
}

impl Matrix {
    /// Create a Matrix from an Expr::Matrix
    pub fn from_expr(ctx: &Context, id: ExprId) -> Option<Self> {
        if let Expr::Matrix { rows, cols, data } = ctx.get(id) {
            Some(Matrix {
                rows: *rows,
                cols: *cols,
                data: data.clone(),
            })
        } else {
            None
        }
    }

    /// Convert Matrix back to Expr::Matrix
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        // INVARIANT: Matrix struct always has valid dimensions from construction.
        // Fallback to raw add in the unreachable error case.
        ctx.matrix(self.rows, self.cols, self.data.clone())
            .unwrap_or_else(|_| {
                ctx.add(Expr::Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    data: self.data.clone(),
                })
            })
    }

    /// Check if dimensions match for addition
    pub fn can_add(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols
    }

    /// Check if dimensions are compatible for multiplication
    /// self (m×n) * other (p×q) requires n == p
    pub fn can_multiply(&self, other: &Self) -> bool {
        self.cols == other.rows
    }

    /// Add two matrices element-wise
    /// Returns None if dimensions don't match
    pub fn add(&self, other: &Self, ctx: &mut Context) -> Option<Self> {
        if !self.can_add(other) {
            return None;
        }

        let mut result_data = Vec::with_capacity(self.data.len());
        for (&a, &b) in self.data.iter().zip(other.data.iter()) {
            let sum = ctx.add(Expr::Add(a, b));
            result_data.push(sum);
        }

        Some(Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        })
    }

    /// Subtract two matrices element-wise
    pub fn sub(&self, other: &Self, ctx: &mut Context) -> Option<Self> {
        if !self.can_add(other) {
            return None;
        }

        let mut result_data = Vec::with_capacity(self.data.len());
        for (&a, &b) in self.data.iter().zip(other.data.iter()) {
            let diff = ctx.add(Expr::Sub(a, b));
            result_data.push(diff);
        }

        Some(Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        })
    }

    /// Multiply matrix by a scalar
    pub fn scalar_mul(&self, scalar: ExprId, ctx: &mut Context) -> Self {
        let mut result_data = Vec::with_capacity(self.data.len());
        for &elem in &self.data {
            let product = mul2_raw(ctx, scalar, elem);
            result_data.push(product);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    /// Multiply two matrices
    /// self (m×n) * other (n×p) → result (m×p)
    pub fn multiply(&self, other: &Self, ctx: &mut Context) -> Option<Self> {
        if !self.can_multiply(other) {
            return None;
        }

        let m = self.rows;
        let n = self.cols; // == other.rows
        let p = other.cols;

        let mut result_data = Vec::with_capacity(m * p);

        for i in 0..m {
            for j in 0..p {
                // Compute dot product of row i of self with column j of other
                let mut sum = ctx.num(0);
                for k in 0..n {
                    let a_ik = self.data[i * self.cols + k];
                    let b_kj = other.data[k * other.cols + j];
                    let product = mul2_raw(ctx, a_ik, b_kj);
                    sum = ctx.add(Expr::Add(sum, product));
                }
                result_data.push(sum);
            }
        }

        Some(Matrix {
            rows: m,
            cols: p,
            data: result_data,
        })
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut result_data = Vec::with_capacity(self.data.len());

        for j in 0..self.cols {
            for i in 0..self.rows {
                result_data.push(self.data[i * self.cols + j]);
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: result_data,
        }
    }

    /// Compute the trace (sum of diagonal elements)
    /// Only defined for square matrices
    pub fn trace(&self, ctx: &mut Context) -> Option<ExprId> {
        if self.rows != self.cols {
            return None;
        }

        let mut sum = ctx.num(0);
        for i in 0..self.rows {
            let diag_elem = self.data[i * self.cols + i];
            sum = ctx.add(Expr::Add(sum, diag_elem));
        }

        Some(sum)
    }

    /// Exact rank of a NUMERIC matrix by Gaussian elimination over `BigRational`
    /// (works for any shape — rank is not restricted to square matrices). Returns
    /// `None` (⇒ honest residual) when any entry is not a rational constant, since
    /// the rank of a symbolic/parametrized matrix is not a single number. Every
    /// pivot test is EXACT rational arithmetic — never f64 — so a near-zero pivot
    /// can never be misjudged nonzero.
    pub fn rank(&self, ctx: &mut Context) -> Option<ExprId> {
        use num_rational::BigRational;
        use num_traits::Zero;

        let mut m: Vec<Vec<BigRational>> = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(crate::numeric_eval::as_rational_const(
                    ctx,
                    self.data[i * self.cols + j],
                )?);
            }
            m.push(row);
        }

        let mut rank = 0usize;
        let mut pivot_col = 0usize;
        let mut row = 0usize;
        while row < self.rows && pivot_col < self.cols {
            let pivot = (row..self.rows).find(|&i| !m[i][pivot_col].is_zero());
            match pivot {
                None => pivot_col += 1,
                Some(p) => {
                    m.swap(row, p);
                    let pivot_val = m[row][pivot_col].clone();
                    let pivot_row = m[row].clone();
                    for target in m.iter_mut().skip(row + 1) {
                        if !target[pivot_col].is_zero() {
                            let factor = &target[pivot_col] / &pivot_val;
                            for (entry, pivot_entry) in
                                target.iter_mut().zip(&pivot_row).skip(pivot_col)
                            {
                                *entry -= &factor * pivot_entry;
                            }
                        }
                    }
                    rank += 1;
                    row += 1;
                    pivot_col += 1;
                }
            }
        }
        Some(ctx.num(rank as i64))
    }

    /// Euclidean / Frobenius norm `√(Σ entryᵢ²)`. For a vector (n×1 or 1×n) this
    /// is the usual length; for a matrix it is the Frobenius norm. Works
    /// symbolically too — `norm([a,b]) = √(a²+b²)` — and the engine folds the
    /// numeric case (`norm([3,4]) = 5`).
    pub fn norm(&self, ctx: &mut Context) -> Option<ExprId> {
        use num_traits::Zero;
        let two = ctx.num(2);
        let mut sum_of_squares = ctx.num(0);
        for &entry in &self.data {
            // The Euclidean norm squares the MAGNITUDE of each component, `|a+bi|^2 = a^2 + b^2`,
            // NOT `(a+bi)^2` — squaring the raw component makes a complex entry go imaginary or
            // negative (`norm([3,4i])` must be `5`, not `sqrt(9+(4i)^2) = i·sqrt(7)`; `norm([1,i])`
            // must be `sqrt(2)`, not `sqrt(1+i^2) = 0`). A real entry keeps the `entry^2` form
            // (RealOnly-identical); a recognized Gaussian `a+bi` with `b != 0` folds to the exact
            // rational `a^2 + b^2`.
            let square = match crate::complex_support::extract_gaussian(ctx, entry) {
                Some(g) if !g.imag.is_zero() => {
                    let magnitude_squared = &g.real * &g.real + &g.imag * &g.imag;
                    ctx.add(Expr::Number(magnitude_squared))
                }
                _ => ctx.add(Expr::Pow(entry, two)),
            };
            sum_of_squares = ctx.add(Expr::Add(sum_of_squares, square));
        }
        Some(ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![sum_of_squares]))
    }

    /// Reduced row echelon form of a NUMERIC matrix by exact Gauss-Jordan
    /// elimination over `BigRational` (any shape). Each pivot is normalized to 1
    /// and its column is cleared in every other row. Returns `None` (⇒ honest
    /// residual) when any entry is not a rational constant. All arithmetic is
    /// exact — never f64 — so pivot detection is never misjudged.
    pub fn rref(&self, ctx: &mut Context) -> Option<ExprId> {
        use num_rational::BigRational;
        use num_traits::Zero;

        let mut m: Vec<Vec<BigRational>> = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(crate::numeric_eval::as_rational_const(
                    ctx,
                    self.data[i * self.cols + j],
                )?);
            }
            m.push(row);
        }

        let mut pivot_row = 0usize;
        let mut pivot_col = 0usize;
        while pivot_row < self.rows && pivot_col < self.cols {
            match (pivot_row..self.rows).find(|&i| !m[i][pivot_col].is_zero()) {
                None => pivot_col += 1,
                Some(p) => {
                    m.swap(pivot_row, p);
                    // Normalize the pivot row so the pivot becomes 1.
                    let pivot_val = m[pivot_row][pivot_col].clone();
                    for entry in &mut m[pivot_row] {
                        *entry = &*entry / &pivot_val;
                    }
                    // Clear the pivot column in every OTHER row (above and below).
                    let pivot_row_values = m[pivot_row].clone();
                    for (i, row) in m.iter_mut().enumerate() {
                        if i != pivot_row && !row[pivot_col].is_zero() {
                            let factor = row[pivot_col].clone();
                            for (entry, pivot_entry) in row.iter_mut().zip(&pivot_row_values) {
                                *entry -= &factor * pivot_entry;
                            }
                        }
                    }
                    pivot_row += 1;
                    pivot_col += 1;
                }
            }
        }

        let data: Vec<ExprId> = m
            .into_iter()
            .flatten()
            .map(|value| ctx.add(Expr::Number(value)))
            .collect();
        Some(
            ctx.matrix(self.rows, self.cols, data.clone())
                .unwrap_or_else(|_| {
                    ctx.add(Expr::Matrix {
                        rows: self.rows,
                        cols: self.cols,
                        data,
                    })
                }),
        )
    }

    /// The submatrix with `remove_row` and `remove_col` deleted (its minor base).
    fn minor_submatrix(&self, remove_row: usize, remove_col: usize) -> Matrix {
        let n = self.cols;
        let mut data = Vec::with_capacity((self.rows - 1) * (self.cols - 1));
        for r in 0..self.rows {
            if r == remove_row {
                continue;
            }
            for c in 0..self.cols {
                if c == remove_col {
                    continue;
                }
                data.push(self.data[r * n + c]);
            }
        }
        Matrix {
            rows: self.rows - 1,
            cols: self.cols - 1,
            data,
        }
    }

    /// Adjugate (classical adjoint) `adj(A)`: the transpose of the cofactor matrix,
    /// `adj(A)[i][j] = (−1)^{i+j}·det(minor removing row j, col i)`. Unlike the
    /// inverse it is a polynomial in the entries — ALWAYS defined (no `det ≠ 0`
    /// condition) — so it works symbolically (`[[a,b],[c,d]] → [[d,-b],[-c,a]]`) as
    /// well as numerically, and satisfies `A·adj(A) = det(A)·I`. Square only.
    pub fn adjugate(&self, ctx: &mut Context) -> Option<ExprId> {
        if self.rows != self.cols || self.rows == 0 {
            return None;
        }
        let n = self.rows;
        if n == 1 {
            let one = ctx.num(1);
            return Some(ctx.matrix(1, 1, vec![one]).unwrap_or_else(|_| {
                ctx.add(Expr::Matrix {
                    rows: 1,
                    cols: 1,
                    data: vec![one],
                })
            }));
        }
        let zero = ctx.num(0);
        let mut data = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                let minor_det = self.minor_submatrix(j, i).determinant(ctx)?;
                let cofactor = if (i + j) % 2 == 0 {
                    minor_det
                } else {
                    ctx.add(Expr::Neg(minor_det))
                };
                data[i * n + j] = cofactor;
            }
        }
        Some(ctx.matrix(n, n, data.clone()).unwrap_or_else(|_| {
            ctx.add(Expr::Matrix {
                rows: n,
                cols: n,
                data,
            })
        }))
    }

    /// Characteristic polynomial `det(λI − A)` in the variable `var` (monic,
    /// degree n). Reuses the symbolic cofactor determinant, which auto-expands,
    /// so `[[2,1],[1,2]]` yields `λ² − 4λ + 3`. Returns `None` for a non-square
    /// matrix. Symbolic entries are fine — the determinant stays symbolic.
    pub fn charpoly(&self, ctx: &mut Context, var: &str) -> Option<ExprId> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let lambda = ctx.var(var);
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let entry = self.data[i * n + j];
                // λI − A: diagonal λ − a_ii, off-diagonal −a_ij.
                let cell = if i == j {
                    ctx.add(Expr::Sub(lambda, entry))
                } else {
                    ctx.add(Expr::Neg(entry))
                };
                data.push(cell);
            }
        }
        Matrix {
            rows: n,
            cols: n,
            data,
        }
        .determinant(ctx)
    }

    /// Compute determinant for 2×2 matrix
    /// det([[a, b], [c, d]]) = ad - bc
    fn det_2x2(&self, ctx: &mut Context) -> ExprId {
        let a = self.data[0];
        let b = self.data[1];
        let c = self.data[2];
        let d = self.data[3];

        let ad = mul2_raw(ctx, a, d);
        let bc = mul2_raw(ctx, b, c);
        ctx.add(Expr::Sub(ad, bc))
    }

    /// Compute determinant for 3×3 matrix using Sarrus rule
    fn det_3x3(&self, ctx: &mut Context) -> ExprId {
        // Rule of Sarrus for 3×3:
        // det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32
        //     - a13*a22*a31 - a11*a23*a32 - a12*a21*a33

        let get = |r: usize, c: usize| self.data[r * 3 + c];

        let a11 = get(0, 0);
        let a12 = get(0, 1);
        let a13 = get(0, 2);
        let a21 = get(1, 0);
        let a22 = get(1, 1);
        let a23 = get(1, 2);
        let a31 = get(2, 0);
        let a32 = get(2, 1);
        let a33 = get(2, 2);

        // Positive terms - split to avoid multiple mutable borrows
        let a22_a33 = mul2_raw(ctx, a22, a33);
        let t1 = mul2_raw(ctx, a11, a22_a33);

        let a23_a31 = mul2_raw(ctx, a23, a31);
        let t2 = mul2_raw(ctx, a12, a23_a31);

        let a21_a32 = mul2_raw(ctx, a21, a32);
        let t3 = mul2_raw(ctx, a13, a21_a32);

        // Negative terms
        let a22_a31 = mul2_raw(ctx, a22, a31);
        let t4 = mul2_raw(ctx, a13, a22_a31);

        let a23_a32 = mul2_raw(ctx, a23, a32);
        let t5 = mul2_raw(ctx, a11, a23_a32);

        let a21_a33 = mul2_raw(ctx, a21, a33);
        let t6 = mul2_raw(ctx, a12, a21_a33);

        // Sum positive
        let t2_t3 = ctx.add(Expr::Add(t2, t3));
        let pos = ctx.add(Expr::Add(t1, t2_t3));

        // Sum negative
        let t5_t6 = ctx.add(Expr::Add(t5, t6));
        let neg = ctx.add(Expr::Add(t4, t5_t6));

        ctx.add(Expr::Sub(pos, neg))
    }

    /// Compute determinant
    /// Supports any square matrix size using cofactor expansion for n≥4
    pub fn determinant(&self, ctx: &mut Context) -> Option<ExprId> {
        if self.rows != self.cols {
            return None; // Not square
        }

        match self.rows {
            1 => Some(self.data[0]),
            2 => Some(self.det_2x2(ctx)),
            3 => Some(self.det_3x3(ctx)),
            _ => {
                // Use cofactor expansion for matrices 4×4 and larger
                Self::determinant_cofactor(ctx, self.rows, &self.data)
            }
        }
    }

    /// Compute determinant using cofactor expansion along the first row
    /// For n×n matrices where n ≥ 4
    fn determinant_cofactor(ctx: &mut Context, n: usize, data: &[ExprId]) -> Option<ExprId> {
        // det(M) = Σ(j=0 to n-1) (-1)^j * M[0][j] * det(Minor[0][j])

        let mut terms = Vec::new();

        for j in 0..n {
            let element = data[j]; // Element M[0][j]

            // Compute minor: submatrix without row 0 and column j
            let minor_data = Self::get_minor(data, n, 0, j);

            // Recursively compute determinant of the (n-1)×(n-1) minor
            let minor_det = if n - 1 == 1 {
                // Base case: 1×1 minor
                Some(minor_data[0])
            } else if n - 1 == 2 {
                // 2×2 minor
                let a = minor_data[0];
                let b = minor_data[1];
                let c = minor_data[2];
                let d = minor_data[3];
                let ad = mul2_raw(ctx, a, d);
                let bc = mul2_raw(ctx, b, c);
                Some(ctx.add(Expr::Sub(ad, bc)))
            } else if n - 1 == 3 {
                // 3×3 minor
                // This calls the private det_3x3 function, but it's a method of Matrix.
                // We need a standalone function or to pass a Matrix instance.
                // For now, let's assume a helper `det_3x3_from_slice` or similar.
                // Given the context, it's likely the user intends to call a helper that operates on a slice.
                // For now, I'll assume a placeholder or a direct implementation if it's simple enough.
                // The original det_3x3 takes `&self` and `ctx`.
                // Let's implement it directly here for the slice.
                let get = |r: usize, c: usize| minor_data[r * 3 + c];

                let a11 = get(0, 0);
                let a12 = get(0, 1);
                let a13 = get(0, 2);
                let a21 = get(1, 0);
                let a22 = get(1, 1);
                let a23 = get(1, 2);
                let a31 = get(2, 0);
                let a32 = get(2, 1);
                let a33 = get(2, 2);

                let a22_a33 = mul2_raw(ctx, a22, a33);
                let t1 = mul2_raw(ctx, a11, a22_a33);

                let a23_a31 = mul2_raw(ctx, a23, a31);
                let t2 = mul2_raw(ctx, a12, a23_a31);

                let a21_a32 = mul2_raw(ctx, a21, a32);
                let t3 = mul2_raw(ctx, a13, a21_a32);

                let a22_a31 = mul2_raw(ctx, a22, a31);
                let t4 = mul2_raw(ctx, a13, a22_a31);

                let a23_a32 = mul2_raw(ctx, a23, a32);
                let t5 = mul2_raw(ctx, a11, a23_a32);

                let a21_a33 = mul2_raw(ctx, a21, a33);
                let t6 = mul2_raw(ctx, a12, a21_a33);

                let t2_t3 = ctx.add(Expr::Add(t2, t3));
                let pos = ctx.add(Expr::Add(t1, t2_t3));

                let t5_t6 = ctx.add(Expr::Add(t5, t6));
                let neg = ctx.add(Expr::Add(t4, t5_t6));

                Some(ctx.add(Expr::Sub(pos, neg)))
            } else {
                // Recursive case for larger minors
                Self::determinant_cofactor(ctx, n - 1, &minor_data)
            }?;

            // Cofactor = (-1)^(0+j) * element * det(minor)
            let cofactor = if j % 2 == 0 {
                // Positive: element * minor_det
                mul2_raw(ctx, element, minor_det)
            } else {
                // Negative: -(element * minor_det)
                let product = mul2_raw(ctx, element, minor_det);
                ctx.add(Expr::Neg(product))
            };

            terms.push(cofactor);
        }

        // Sum all cofactors
        if terms.is_empty() {
            return Some(ctx.num(0));
        }

        let mut result = terms[0];
        for term in terms.iter().skip(1) {
            result = ctx.add(Expr::Add(result, *term));
        }

        Some(result)
    }

    /// Extract minor matrix: remove row i and column j from n×n matrix
    fn get_minor(data: &[ExprId], n: usize, row: usize, col: usize) -> Vec<ExprId> {
        let mut minor = Vec::new();

        for r in 0..n {
            if r == row {
                continue; // Skip row
            }
            for c in 0..n {
                if c == col {
                    continue; // Skip column
                }
                minor.push(data[r * n + c]);
            }
        }

        minor
    }

    /// Determinant of an `n×n` matrix given as a row-major slice, by recursive cofactor
    /// expansion along the first row. `n = 0` is the empty product `1` (the determinant
    /// of the `0×0` matrix), which lets the `1×1` cofactors of a `1×1` inverse bottom out
    /// cleanly. Builds a symbolic expression, so it works for numeric and symbolic
    /// entries alike; numeric entries fold to an exact rational under simplification.
    fn det_of_slice(ctx: &mut Context, n: usize, data: &[ExprId]) -> ExprId {
        match n {
            0 => ctx.num(1),
            1 => data[0],
            2 => {
                let ad = mul2_raw(ctx, data[0], data[3]);
                let bc = mul2_raw(ctx, data[1], data[2]);
                ctx.add(Expr::Sub(ad, bc))
            }
            _ => {
                let mut result: Option<ExprId> = None;
                for j in 0..n {
                    let minor = Self::get_minor(data, n, 0, j);
                    let minor_det = Self::det_of_slice(ctx, n - 1, &minor);
                    let prod = mul2_raw(ctx, data[j], minor_det);
                    let term = if j % 2 == 0 {
                        prod
                    } else {
                        ctx.add(Expr::Neg(prod))
                    };
                    result = Some(match result {
                        None => term,
                        Some(acc) => ctx.add(Expr::Add(acc, term)),
                    });
                }
                result.unwrap_or_else(|| ctx.num(0))
            }
        }
    }

    /// Compute the inverse of a square matrix as `adjugate / determinant`, where
    /// `adjugate[i][j] = cofactor[j][i]` and `cofactor[i][j] = (-1)^(i+j)·det(minor(i,j))`.
    ///
    /// Returns `None` for a non-square matrix (the `inverse(...)` call is left symbolic),
    /// [`MatrixInverseOutcome::Singular`] when the determinant is provably zero (no
    /// inverse exists), and otherwise the inverse matrix. Each entry is built as
    /// `cofactor / det`; for a numeric matrix this folds to an exact rational under
    /// simplification (e.g. `inverse([[1,2],[3,4]]) = [[-2, 1], [3/2, -1/2]]`), and for a
    /// symbolic matrix it yields the standard generic inverse (valid where `det ≠ 0`).
    pub fn inverse(&self, ctx: &mut Context) -> Option<MatrixInverseOutcome> {
        if self.rows != self.cols {
            return None; // inverse is only defined for square matrices
        }
        let n = self.rows;
        let det = self.determinant(ctx)?;
        // Provably-singular ⇒ no inverse. The numeric `as_rational_const` check catches a
        // det that folds to literal `0` (`[[1,2],[2,4]]`); `is_provably_zero` additionally
        // catches structural zeros (e.g. two equal rows of a symbolic matrix).
        let det_is_zero = crate::numeric_eval::as_rational_const(ctx, det)
            .map(|r| num_traits::Zero::is_zero(&r))
            .unwrap_or(false)
            || crate::arithmetic_cancel_support::is_provably_zero(ctx, det);
        if det_is_zero {
            return Some(MatrixInverseOutcome::Singular);
        }

        // Fold the determinant to a literal rational when the matrix is numeric, so each
        // inverse entry can be emitted as an exact rational rather than a deep
        // `cofactor / det` quotient (the latter trips the simplifier's node-growth guard
        // and leaves `inverse(...)` unfolded).
        let det_rat = crate::numeric_eval::as_rational_const(ctx, det);

        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                // inverse[i][j] = adjugate[i][j] / det = cofactor(j, i) / det.
                let minor = Self::get_minor(&self.data, n, j, i);
                let minor_det = Self::det_of_slice(ctx, n - 1, &minor);
                let signed = if (i + j) % 2 == 0 {
                    minor_det
                } else {
                    ctx.add(Expr::Neg(minor_det))
                };
                // Numeric matrix ⇒ exact rational entry; symbolic matrix ⇒ the generic
                // `cofactor / det` quotient (valid where `det ≠ 0`).
                let entry = match (
                    &det_rat,
                    crate::numeric_eval::as_rational_const(ctx, signed),
                ) {
                    (Some(den), Some(num)) => ctx.add(Expr::Number(num / den)),
                    _ => ctx.add(Expr::Div(signed, det)),
                };
                data.push(entry);
            }
        }
        Some(MatrixInverseOutcome::Inverse(Matrix {
            rows: n,
            cols: n,
            data,
        }))
    }
}

/// Result of a matrix inversion: either the inverse matrix, or a proof that the matrix
/// is singular (its determinant is provably zero ⇒ no inverse exists).
#[derive(Debug, Clone)]
pub enum MatrixInverseOutcome {
    Inverse(Matrix),
    Singular,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_add() {
        let mut ctx = Context::new();

        // [[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3), ctx.num(4)],
        };

        let m2 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(5), ctx.num(6), ctx.num(7), ctx.num(8)],
        };

        let result = m1.add(&m2, &mut ctx).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_scalar_mul() {
        let mut ctx = Context::new();

        // 2 * [[1, 2], [3, 4]] = [[2, 4], [6, 8]]
        let m = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3), ctx.num(4)],
        };

        let scalar = ctx.num(2);
        let result = m.scalar_mul(scalar, &mut ctx);

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_matrix_multiply() {
        let mut ctx = Context::new();

        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let m1 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3), ctx.num(4)],
        };

        let m2 = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(5), ctx.num(6), ctx.num(7), ctx.num(8)],
        };

        let result = m1.multiply(&m2, &mut ctx).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_transpose() {
        let mut ctx = Context::new();

        // transpose([[1, 2, 3], [4, 5, 6]]) = [[1, 4], [2, 5], [3, 6]]
        let m = Matrix {
            rows: 2,
            cols: 3,
            data: vec![
                ctx.num(1),
                ctx.num(2),
                ctx.num(3),
                ctx.num(4),
                ctx.num(5),
                ctx.num(6),
            ],
        };

        let result = m.transpose();
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
    }

    #[test]
    fn test_det_2x2() {
        let mut ctx = Context::new();

        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        let m = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3), ctx.num(4)],
        };

        let det = m.determinant(&mut ctx).unwrap();
        // The result is an expression tree, would need simplification to verify = -2
        // For now just check it returns something
        assert!(matches!(ctx.get(det), Expr::Sub(_, _)));
    }

    #[test]
    fn test_rank_exact_numeric() {
        let mut ctx = Context::new();
        let rank_of = |ctx: &mut Context, rows: usize, cols: usize, vals: &[i64]| -> i64 {
            let data = vals.iter().map(|&v| ctx.num(v)).collect();
            let m = Matrix { rows, cols, data };
            let value = m.rank(ctx).expect("numeric rank");
            crate::numeric_eval::as_rational_const(ctx, value)
                .expect("rank is a number")
                .to_integer()
                .try_into()
                .unwrap()
        };
        assert_eq!(rank_of(&mut ctx, 2, 2, &[1, 2, 2, 4]), 1); // dependent rows
        assert_eq!(rank_of(&mut ctx, 2, 2, &[1, 2, 3, 4]), 2); // full
        assert_eq!(rank_of(&mut ctx, 3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]), 2);
        assert_eq!(rank_of(&mut ctx, 3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]), 3); // identity
        assert_eq!(rank_of(&mut ctx, 2, 2, &[0, 0, 0, 0]), 0); // zero matrix
        assert_eq!(rank_of(&mut ctx, 2, 3, &[1, 2, 3, 2, 4, 6]), 1); // wide, dependent
        assert_eq!(rank_of(&mut ctx, 3, 2, &[1, 2, 3, 4, 5, 6]), 2); // tall, full column rank
    }

    #[test]
    fn test_rref_exact_and_residual() {
        let mut ctx = Context::new();
        // Full-rank 2×2 reduces to the identity.
        let full = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3), ctx.num(4)],
        };
        let identity = full.rref(&mut ctx).expect("numeric rref");
        if let Expr::Matrix { rows, cols, data } = ctx.get(identity).clone() {
            assert_eq!((rows, cols), (2, 2));
            let nums: Vec<_> = data
                .iter()
                .map(|&e| crate::numeric_eval::as_rational_const(&ctx, e).unwrap())
                .collect();
            let expect = [1, 0, 0, 1];
            for (got, want) in nums.iter().zip(expect) {
                assert_eq!(*got, num_rational::BigRational::from_integer(want.into()));
            }
        } else {
            panic!("rref must return a matrix");
        }
        // Symbolic entries decline (honest residual).
        let a = ctx.var("a");
        let symbolic = Matrix {
            rows: 2,
            cols: 2,
            data: vec![a, ctx.num(2), ctx.num(3), ctx.num(4)],
        };
        assert!(symbolic.rref(&mut ctx).is_none());
    }

    #[test]
    fn test_charpoly_shape_and_residual() {
        let mut ctx = Context::new();
        // Square 2×2 yields a determinant expression; non-square declines.
        let square = Matrix {
            rows: 2,
            cols: 2,
            data: vec![ctx.num(2), ctx.num(1), ctx.num(1), ctx.num(2)],
        };
        assert!(square.charpoly(&mut ctx, "lambda").is_some());
        let non_square = Matrix {
            rows: 1,
            cols: 3,
            data: vec![ctx.num(1), ctx.num(2), ctx.num(3)],
        };
        assert!(
            non_square.charpoly(&mut ctx, "lambda").is_none(),
            "characteristic polynomial is only defined for square matrices"
        );
    }

    #[test]
    fn test_rank_symbolic_entry_is_residual() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let m = Matrix {
            rows: 2,
            cols: 2,
            data: vec![a, ctx.num(2), ctx.num(3), ctx.num(4)],
        };
        assert!(
            m.rank(&mut ctx).is_none(),
            "symbolic rank must decline (honest residual), not guess a number"
        );
    }

    // Build an n×n matrix from a row-major list of integers.
    fn int_matrix(ctx: &mut Context, n: usize, vals: &[i64]) -> Matrix {
        Matrix {
            rows: n,
            cols: n,
            data: vals.iter().map(|&v| ctx.num(v)).collect(),
        }
    }

    // Expect each entry of a folded (numeric) inverse to equal the given rational.
    fn assert_inverse_entries(ctx: &Context, inv: &Matrix, expected: &[(i64, i64)]) {
        use num_rational::BigRational;
        assert_eq!(inv.data.len(), expected.len());
        for (entry, &(num, den)) in inv.data.iter().zip(expected) {
            let got = crate::numeric_eval::as_rational_const(ctx, *entry)
                .unwrap_or_else(|| panic!("inverse entry should fold to a rational"));
            assert_eq!(
                got,
                BigRational::new(num.into(), den.into()),
                "entry value mismatch"
            );
        }
    }

    #[test]
    fn test_inverse_2x2_numeric() {
        let mut ctx = Context::new();
        // inverse([[1,2],[3,4]]) = [[-2, 1], [3/2, -1/2]]
        let m = int_matrix(&mut ctx, 2, &[1, 2, 3, 4]);
        let MatrixInverseOutcome::Inverse(inv) = m.inverse(&mut ctx).unwrap() else {
            panic!("expected an invertible matrix");
        };
        assert_eq!((inv.rows, inv.cols), (2, 2));
        assert_inverse_entries(&ctx, &inv, &[(-2, 1), (1, 1), (3, 2), (-1, 2)]);
    }

    #[test]
    fn test_inverse_diagonal_and_1x1() {
        let mut ctx = Context::new();
        // inverse([[2,0],[0,4]]) = [[1/2, 0], [0, 1/4]]
        let diag = int_matrix(&mut ctx, 2, &[2, 0, 0, 4]);
        let MatrixInverseOutcome::Inverse(inv) = diag.inverse(&mut ctx).unwrap() else {
            panic!("expected invertible");
        };
        assert_inverse_entries(&ctx, &inv, &[(1, 2), (0, 1), (0, 1), (1, 4)]);

        // inverse([[5]]) = [[1/5]]
        let one = int_matrix(&mut ctx, 1, &[5]);
        let MatrixInverseOutcome::Inverse(inv1) = one.inverse(&mut ctx).unwrap() else {
            panic!("expected invertible 1x1");
        };
        assert_inverse_entries(&ctx, &inv1, &[(1, 5)]);
    }

    #[test]
    fn test_inverse_3x3_numeric() {
        let mut ctx = Context::new();
        // det = 1; inverse([[1,2,3],[0,1,4],[5,6,0]]) = [[-24,18,5],[20,-15,-4],[-5,4,1]]
        let m = int_matrix(&mut ctx, 3, &[1, 2, 3, 0, 1, 4, 5, 6, 0]);
        let MatrixInverseOutcome::Inverse(inv) = m.inverse(&mut ctx).unwrap() else {
            panic!("expected invertible 3x3");
        };
        assert_inverse_entries(
            &ctx,
            &inv,
            &[
                (-24, 1),
                (18, 1),
                (5, 1),
                (20, 1),
                (-15, 1),
                (-4, 1),
                (-5, 1),
                (4, 1),
                (1, 1),
            ],
        );
    }

    #[test]
    fn test_inverse_singular_is_detected() {
        let mut ctx = Context::new();
        // det([[1,2],[2,4]]) = 0 -> singular.
        let m = int_matrix(&mut ctx, 2, &[1, 2, 2, 4]);
        assert!(matches!(
            m.inverse(&mut ctx).unwrap(),
            MatrixInverseOutcome::Singular
        ));

        // Two identical symbolic rows are structurally singular for all values.
        let a = ctx.var("a");
        let b = ctx.var("b");
        let sym = Matrix {
            rows: 2,
            cols: 2,
            data: vec![a, b, a, b],
        };
        assert!(matches!(
            sym.inverse(&mut ctx).unwrap(),
            MatrixInverseOutcome::Singular
        ));
    }

    #[test]
    fn test_inverse_non_square_is_none() {
        let mut ctx = Context::new();
        let m = Matrix {
            rows: 2,
            cols: 3,
            data: vec![
                ctx.num(1),
                ctx.num(2),
                ctx.num(3),
                ctx.num(4),
                ctx.num(5),
                ctx.num(6),
            ],
        };
        assert!(m.inverse(&mut ctx).is_none());
    }
}
