//! Matrix rule support helpers shared with engine rule layers.
//!
//! These helpers evaluate matrix operation patterns directly from expression
//! nodes and return structured metadata for caller-owned narration.

use cas_ast::{Context, Expr, ExprId};
use cas_math::matrix::Matrix;

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

pub fn try_eval_scalar_matrix_mul_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ScalarMatrixEval> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let left = *left;
    let right = *right;

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
