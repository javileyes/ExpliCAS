use cas_ast::{Context, Expr, ExprId};

/// Matrix wrapper for operations
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<ExprId>,
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
        ctx.matrix(self.rows, self.cols, self.data.clone())
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
        for i in 0..self.data.len() {
            let sum = ctx.add(Expr::Add(self.data[i], other.data[i]));
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
        for i in 0..self.data.len() {
            let diff = ctx.add(Expr::Sub(self.data[i], other.data[i]));
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
            let product = ctx.add(Expr::Mul(scalar, elem));
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
                    let product = ctx.add(Expr::Mul(a_ik, b_kj));
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

    /// Compute determinant for 2×2 matrix
    /// det([[a, b], [c, d]]) = ad - bc
    fn det_2x2(&self, ctx: &mut Context) -> ExprId {
        let a = self.data[0];
        let b = self.data[1];
        let c = self.data[2];
        let d = self.data[3];

        let ad = ctx.add(Expr::Mul(a, d));
        let bc = ctx.add(Expr::Mul(b, c));
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
        let a22_a33 = ctx.add(Expr::Mul(a22, a33));
        let t1 = ctx.add(Expr::Mul(a11, a22_a33));

        let a23_a31 = ctx.add(Expr::Mul(a23, a31));
        let t2 = ctx.add(Expr::Mul(a12, a23_a31));

        let a21_a32 = ctx.add(Expr::Mul(a21, a32));
        let t3 = ctx.add(Expr::Mul(a13, a21_a32));

        // Negative terms
        let a22_a31 = ctx.add(Expr::Mul(a22, a31));
        let t4 = ctx.add(Expr::Mul(a13, a22_a31));

        let a23_a32 = ctx.add(Expr::Mul(a23, a32));
        let t5 = ctx.add(Expr::Mul(a11, a23_a32));

        let a21_a33 = ctx.add(Expr::Mul(a21, a33));
        let t6 = ctx.add(Expr::Mul(a12, a21_a33));

        // Sum positive
        let t2_t3 = ctx.add(Expr::Add(t2, t3));
        let pos = ctx.add(Expr::Add(t1, t2_t3));

        // Sum negative
        let t5_t6 = ctx.add(Expr::Add(t5, t6));
        let neg = ctx.add(Expr::Add(t4, t5_t6));

        ctx.add(Expr::Sub(pos, neg))
    }

    /// Compute determinant
    /// Currently supports 1×1, 2×2, and 3×3 matrices
    pub fn determinant(&self, ctx: &mut Context) -> Option<ExprId> {
        if self.rows != self.cols {
            return None; // Not square
        }

        match self.rows {
            1 => Some(self.data[0]),
            2 => Some(self.det_2x2(ctx)),
            3 => Some(self.det_3x3(ctx)),
            _ => None, // Not yet implemented for larger matrices
        }
    }
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
}
