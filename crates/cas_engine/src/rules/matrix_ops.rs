use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, ExprId};
use cas_math::matrix_rule_support::{
    try_eval_matrix_add_expr, try_eval_matrix_mul_expr, try_eval_matrix_sub_expr,
    try_eval_scalar_matrix_mul_expr, try_rewrite_matrix_function_rule_expr,
    try_rewrite_transpose_product_identity_expr,
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
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc(eval.add_desc()))
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
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc(eval.sub_desc()))
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
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc(eval.desc()))
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
        Some(Rewrite::new(eval.result.to_expr(ctx)).desc(eval.mul_desc()))
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
        let rewrite = try_rewrite_matrix_function_rule_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
        let rewrite = try_rewrite_transpose_product_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
