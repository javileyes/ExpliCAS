use crate::matrix::Matrix;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, Expr, ExprId};

/// Rule to add two matrices
pub struct MatrixAddRule;

impl Rule for MatrixAddRule {
    fn name(&self) -> &'static str {
        "Matrix Addition"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add"])
    }

    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Add(left, right) = ctx.get(expr) {
            let left_id = *left;
            let right_id = *right;

            // Try to interpret both as matrices
            if let (Some(m1), Some(m2)) = (
                Matrix::from_expr(ctx, left_id),
                Matrix::from_expr(ctx, right_id),
            ) {
                // Add matrices
                if let Some(result) = m1.add(&m2, ctx) {
                    return Some(Rewrite {
                        new_expr: result.to_expr(ctx),
                        description: format!(
                            "Matrix addition: {}×{} + {}×{}",
                            m1.rows, m1.cols, m2.rows, m2.cols
                        ),
                    });
                }
            }
        }
        None
    }
}

/// Rule to subtract two matrices
pub struct MatrixSubRule;

impl Rule for MatrixSubRule {
    fn name(&self) -> &'static str {
        "Matrix Subtraction"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Sub"])
    }

    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Sub(left, right) = ctx.get(expr) {
            let left_id = *left;
            let right_id = *right;

            if let (Some(m1), Some(m2)) = (
                Matrix::from_expr(ctx, left_id),
                Matrix::from_expr(ctx, right_id),
            ) {
                if let Some(result) = m1.sub(&m2, ctx) {
                    return Some(Rewrite {
                        new_expr: result.to_expr(ctx),
                        description: format!(
                            "Matrix subtraction: {}×{} - {}×{}",
                            m1.rows, m1.cols, m2.rows, m2.cols
                        ),
                    });
                }
            }
        }
        None
    }
}

/// Rule to multiply a matrix by a scalar
pub struct ScalarMatrixRule;

impl Rule for ScalarMatrixRule {
    fn name(&self) -> &'static str {
        "Scalar Matrix Multiplication"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Mul(left, right) = ctx.get(expr) {
            let left_id = *left;
            let right_id = *right;

            // Try scalar * matrix
            if let Some(matrix) = Matrix::from_expr(ctx, right_id) {
                // left is the scalar
                let result = matrix.scalar_mul(left_id, ctx);
                return Some(Rewrite {
                    new_expr: result.to_expr(ctx),
                    description: format!(
                        "Scalar multiplication: scalar × {}×{} matrix",
                        matrix.rows, matrix.cols
                    ),
                });
            }

            // Try matrix * scalar
            if let Some(matrix) = Matrix::from_expr(ctx, left_id) {
                // right is the scalar
                let result = matrix.scalar_mul(right_id, ctx);
                return Some(Rewrite {
                    new_expr: result.to_expr(ctx),
                    description: format!(
                        "Scalar multiplication: {}×{} matrix × scalar",
                        matrix.rows, matrix.cols
                    ),
                });
            }
        }
        None
    }
}

/// Rule to multiply two matrices
pub struct MatrixMultiplyRule;

impl Rule for MatrixMultiplyRule {
    fn name(&self) -> &'static str {
        "Matrix Multiplication"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Mul(left, right) = ctx.get(expr) {
            let left_id = *left;
            let right_id = *right;

            // Check if both are matrices (and not scalar-matrix case)
            if let (Some(m1), Some(m2)) = (
                Matrix::from_expr(ctx, left_id),
                Matrix::from_expr(ctx, right_id),
            ) {
                // Verify neither is a scalar (1×1 matrix)
                // Actually, 1×1 matrices should be treated as matrices for multiplication
                // So we proceed with matrix multiplication
                if let Some(result) = m1.multiply(&m2, ctx) {
                    return Some(Rewrite {
                        new_expr: result.to_expr(ctx),
                        description: format!(
                            "Matrix multiplication: {}×{} × {}×{} = {}×{}",
                            m1.rows, m1.cols, m2.rows, m2.cols, result.rows, result.cols
                        ),
                    });
                }
            }
        }
        None
    }
}

/// Rule to evaluate matrix functions: det(), transpose(), trace()
pub struct MatrixFunctionRule;

impl Rule for MatrixFunctionRule {
    fn name(&self) -> &'static str {
        "Matrix Functions"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        if let Expr::Function(name, args) = ctx.get(expr) {
            let name = name.clone();
            let args = args.clone();

            match name.as_str() {
                "det" | "determinant" => {
                    if args.len() == 1 {
                        if let Some(matrix) = Matrix::from_expr(ctx, args[0]) {
                            if let Some(det_value) = matrix.determinant(ctx) {
                                return Some(Rewrite {
                                    new_expr: det_value,
                                    description: format!(
                                        "det({}×{} matrix)",
                                        matrix.rows, matrix.cols
                                    ),
                                });
                            }
                        }
                    }
                }
                "transpose" | "T" => {
                    if args.len() == 1 {
                        if let Some(matrix) = Matrix::from_expr(ctx, args[0]) {
                            let transposed = matrix.transpose();
                            return Some(Rewrite {
                                new_expr: transposed.to_expr(ctx),
                                description: format!(
                                    "transpose({}×{}) = {}×{}",
                                    matrix.rows, matrix.cols, transposed.rows, transposed.cols
                                ),
                            });
                        }
                    }
                }
                "trace" | "tr" => {
                    if args.len() == 1 {
                        if let Some(matrix) = Matrix::from_expr(ctx, args[0]) {
                            if let Some(trace_value) = matrix.trace(ctx) {
                                return Some(Rewrite {
                                    new_expr: trace_value,
                                    description: format!(
                                        "trace({}×{} matrix)",
                                        matrix.rows, matrix.cols
                                    ),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]);
        let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]);
        let add_expr = ctx.add(Expr::Add(m1, m2));

        let result = rule.apply(&mut ctx, add_expr);
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

        let matrix = ctx.matrix(2, 2, vec![n1, n2, n3, n4]);
        let mul_expr = ctx.add(Expr::Mul(scalar, matrix));

        let result = rule.apply(&mut ctx, mul_expr);
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

        let m1 = ctx.matrix(2, 2, vec![n1, n2, n3, n4]);
        let m2 = ctx.matrix(2, 2, vec![n5, n6, n7, n8]);
        let mul_expr = ctx.add(Expr::Mul(m1, m2));

        let result = rule.apply(&mut ctx, mul_expr);
        assert!(result.is_some());
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
}
