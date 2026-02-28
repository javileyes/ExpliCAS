use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, ExprId};
use cas_math::matrix_rule_support::{
    try_eval_matrix_add_expr, try_eval_matrix_function_expr, try_eval_matrix_mul_expr,
    try_eval_matrix_sub_expr, try_eval_scalar_matrix_mul_expr, try_rewrite_transpose_product_expr,
    MatrixFunctionEval, ScalarMatrixSide,
};

/// Rule to add two matrices
pub struct MatrixAddRule;

impl SimpleRule for MatrixAddRule {
    fn name(&self) -> &'static str {
        "Matrix Addition"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let eval = try_eval_matrix_add_expr(ctx, expr)?;
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc_lazy(|| {
            format!(
                "Matrix addition: {}×{} + {}×{}",
                eval.left.rows, eval.left.cols, eval.right.rows, eval.right.cols
            )
        }))
    }
}

/// Rule to subtract two matrices
pub struct MatrixSubRule;

impl SimpleRule for MatrixSubRule {
    fn name(&self) -> &'static str {
        "Matrix Subtraction"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::SUB)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let eval = try_eval_matrix_sub_expr(ctx, expr)?;
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc_lazy(|| {
            format!(
                "Matrix subtraction: {}×{} - {}×{}",
                eval.left.rows, eval.left.cols, eval.right.rows, eval.right.cols
            )
        }))
    }
}

/// Rule to multiply a matrix by a scalar
pub struct ScalarMatrixRule;

impl SimpleRule for ScalarMatrixRule {
    fn name(&self) -> &'static str {
        "Scalar Matrix Multiplication"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let eval = try_eval_scalar_matrix_mul_expr(ctx, expr)?;
        Some(
            Rewrite::new(eval.result.to_expr(ctx)).desc_lazy(|| match eval.side {
                ScalarMatrixSide::ScalarLeft => format!(
                    "Scalar multiplication: scalar × {}×{} matrix",
                    eval.matrix.rows, eval.matrix.cols
                ),
                ScalarMatrixSide::ScalarRight => format!(
                    "Scalar multiplication: {}×{} matrix × scalar",
                    eval.matrix.rows, eval.matrix.cols
                ),
            }),
        )
    }
}

/// Rule to multiply two matrices
pub struct MatrixMultiplyRule;

impl SimpleRule for MatrixMultiplyRule {
    fn name(&self) -> &'static str {
        "Matrix Multiplication"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let eval = try_eval_matrix_mul_expr(ctx, expr)?;
        let result_rows = eval.result.rows;
        let result_cols = eval.result.cols;
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc_lazy(|| {
            format!(
                "Matrix multiplication: {}×{} × {}×{} = {}×{}",
                eval.left.rows,
                eval.left.cols,
                eval.right.rows,
                eval.right.cols,
                result_rows,
                result_cols
            )
        }))
    }
}

/// Rule to evaluate matrix functions: det(), transpose(), trace()
pub struct MatrixFunctionRule;

impl SimpleRule for MatrixFunctionRule {
    fn name(&self) -> &'static str {
        "Matrix Functions"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        match try_eval_matrix_function_expr(ctx, expr)? {
            MatrixFunctionEval::Determinant { shape, value } => Some(
                Rewrite::new(value)
                    .desc_lazy(|| format!("det({}×{} matrix)", shape.rows, shape.cols)),
            ),
            MatrixFunctionEval::Transpose { from, to, matrix } => {
                Some(Rewrite::new(matrix.to_expr(ctx)).desc_lazy(|| {
                    format!(
                        "transpose({}×{}) = {}×{}",
                        from.rows, from.cols, to.rows, to.cols
                    )
                }))
            }
            MatrixFunctionEval::Trace { shape, value } => Some(
                Rewrite::new(value)
                    .desc_lazy(|| format!("trace({}×{} matrix)", shape.rows, shape.cols)),
            ),
        }
    }
}

// TransposeProductRule: transpose(matmul(A, B)) → matmul(transpose(B), transpose(A))
// This is the fundamental identity (AB)^T = B^T·A^T
pub struct TransposeProductRule;

impl SimpleRule for TransposeProductRule {
    fn name(&self) -> &'static str {
        "Transpose of Product"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let result = try_rewrite_transpose_product_expr(ctx, expr)?;
        Some(Rewrite::new(result).desc("(AB)^T = B^T·A^T"))
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(MatrixAddRule));
    simplifier.add_rule(Box::new(MatrixSubRule));
    // IMPORTANT: MatrixMultiplyRule MUST come before ScalarMatrixRule
    // so that matrix-matrix multiplication is checked before scalar-matrix
    simplifier.add_rule(Box::new(MatrixMultiplyRule));
    simplifier.add_rule(Box::new(ScalarMatrixRule));
    simplifier.add_rule(Box::new(MatrixFunctionRule));
    // Algebraic identity: (AB)^T = B^T·A^T
    simplifier.add_rule(Box::new(TransposeProductRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn test_matrix_add_rule() {
        let mut ctx = Context::new();
        let rule = MatrixAddRule;

        // [[1, 2], [3, 4]] + [[5, 6], [7, 8]]
        let n1 = ctx.num(1);
        let n2 = ctx.num(2);
        let n3 = ctx.num(3);
        let n4 = ctx.num(4);
        let n5 = ctx.num(5);
        let n6 = ctx.num(6);
        let n7 = ctx.num(7);
        let n8 = ctx.num(8);

        let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
        let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]).unwrap();
        let add_expr = ctx.add(Expr::Add(m1, m2));

        let result = rule.apply_simple(&mut ctx, add_expr);
        assert!(result.is_some());
    }

    #[test]
    fn test_scalar_matrix_rule() {
        let mut ctx = Context::new();
        let rule = ScalarMatrixRule;

        // 2 * [[1, 2], [3, 4]]
        let scalar = ctx.num(2);
        let n1 = ctx.num(1);
        let n2 = ctx.num(2);
        let n3 = ctx.num(3);
        let n4 = ctx.num(4);

        let matrix = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
        let mul_expr = ctx.add(Expr::Mul(scalar, matrix));

        let result = rule.apply_simple(&mut ctx, mul_expr);
        assert!(result.is_some());
    }

    #[test]
    fn test_matrix_multiply_rule() {
        let mut ctx = Context::new();
        let rule = MatrixMultiplyRule;

        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]]
        let n1 = ctx.num(1);
        let n2 = ctx.num(2);
        let n3 = ctx.num(3);
        let n4 = ctx.num(4);
        let n5 = ctx.num(5);
        let n6 = ctx.num(6);
        let n7 = ctx.num(7);
        let n8 = ctx.num(8);

        let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]).unwrap();
        let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]).unwrap();
        let mul_expr = ctx.add(Expr::Mul(m1, m2));

        let result = rule.apply_simple(&mut ctx, mul_expr);
        assert!(result.is_some());
    }
}
