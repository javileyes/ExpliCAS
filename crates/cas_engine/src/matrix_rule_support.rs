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
pub fn format_matrix_function_desc(eval: &MatrixFunctionEval) -> String {
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

pub fn try_eval_matrix_add_expr(ctx: &mut Context, expr: ExprId) -> Option<MatrixBinaryEval> {
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

pub fn try_eval_matrix_sub_expr(ctx: &mut Context, expr: ExprId) -> Option<MatrixBinaryEval> {
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

pub fn try_eval_matrix_mul_expr(ctx: &mut Context, expr: ExprId) -> Option<MatrixBinaryEval> {
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
pub fn try_matrix_shape_mismatch_undefined(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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
pub fn is_matrix_valued(ctx: &Context, expr: ExprId) -> bool {
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
pub fn try_rewrite_matrix_reciprocal_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            Matrix::from_expr(ctx, base)?;
            // Only the inverse power `-1`; general matrix powers are out of scope here.
            let exp_val = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
            if exp_val != num_rational::BigRational::from_integer((-1).into()) {
                return None;
            }
            Some(ctx.call("inverse", vec![base]))
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

pub fn try_eval_scalar_matrix_mul_expr(
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

pub fn try_eval_matrix_function_expr(
    ctx: &mut Context,
    expr: ExprId,
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
pub fn try_rewrite_matrix_function_rule_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<MatrixFunctionRewrite> {
    match try_eval_matrix_function_expr(ctx, expr)? {
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
pub fn try_rewrite_transpose_product_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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
pub fn try_rewrite_transpose_product_identity_expr(
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
