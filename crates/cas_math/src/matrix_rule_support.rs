//! Matrix rule support helpers shared with engine rule layers.
//!
//! These helpers evaluate matrix operation patterns directly from expression
//! nodes and return structured metadata for caller-owned narration.

use crate::matrix::Matrix;
use cas_ast::{Context, Expr, ExprId};

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

#[cfg(test)]
mod tests {
    use super::{
        try_eval_matrix_add_expr, try_eval_matrix_function_expr, try_eval_matrix_mul_expr,
        try_eval_scalar_matrix_mul_expr, try_rewrite_transpose_product_expr, MatrixFunctionEval,
        ScalarMatrixSide,
    };
    use cas_ast::Expr;

    #[test]
    fn matrix_add_eval_extracts_shapes_and_result() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let m1 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m1");
        let m2 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m2");
        let expr = ctx.add(Expr::Add(m1, m2));

        let eval = try_eval_matrix_add_expr(&mut ctx, expr).expect("matrix add");
        assert_eq!(eval.left.rows, 2);
        assert_eq!(eval.right.cols, 2);
        assert_eq!(eval.result.rows, 2);
        assert_eq!(eval.result.cols, 2);
    }

    #[test]
    fn matrix_mul_eval_extracts_shapes_and_result() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let m1 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m1");
        let m2 = ctx.matrix(2, 2, vec![one, two, three, four]).expect("m2");
        let expr = ctx.add(Expr::Mul(m1, m2));

        let eval = try_eval_matrix_mul_expr(&mut ctx, expr).expect("matrix mul");
        assert_eq!(eval.result.rows, 2);
        assert_eq!(eval.result.cols, 2);
    }

    #[test]
    fn scalar_matrix_eval_detects_side() {
        let mut ctx = cas_ast::Context::new();
        let scalar = ctx.num(2);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let matrix = ctx
            .matrix(2, 2, vec![one, two, three, four])
            .expect("matrix");
        let expr = ctx.add(Expr::Mul(scalar, matrix));

        let eval = try_eval_scalar_matrix_mul_expr(&mut ctx, expr).expect("scalar*matrix");
        assert_eq!(eval.side, ScalarMatrixSide::ScalarLeft);
        assert_eq!(eval.matrix.rows, 2);
    }

    #[test]
    fn matrix_function_eval_handles_transpose() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let matrix = ctx
            .matrix(2, 2, vec![one, two, three, four])
            .expect("matrix");
        let expr = ctx.call("transpose", vec![matrix]);

        let eval = try_eval_matrix_function_expr(&mut ctx, expr).expect("transpose");
        match eval {
            MatrixFunctionEval::Transpose { from, to, .. } => {
                assert_eq!(from.rows, 2);
                assert_eq!(to.cols, 2);
            }
            _ => panic!("expected transpose"),
        }
    }

    #[test]
    fn transpose_product_rewrite_builds_expected_shape() {
        let mut ctx = cas_ast::Context::new();
        let a = ctx.var("A");
        let b = ctx.var("B");
        let matmul = ctx.call("matmul", vec![a, b]);
        let expr = ctx.call("transpose", vec![matmul]);

        let rewritten =
            try_rewrite_transpose_product_expr(&mut ctx, expr).expect("expected rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewritten) else {
            panic!("expected function");
        };
        assert_eq!(ctx.sym_name(*fn_id), "matmul");
        assert_eq!(args.len(), 2);
    }
}
