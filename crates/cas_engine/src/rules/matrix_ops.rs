use crate::matrix_rule_support::{
    try_eval_matrix_add_expr, try_eval_matrix_mul_expr, try_eval_matrix_sub_expr,
    try_eval_scalar_matrix_mul_expr, try_rewrite_matrix_function_rule_expr,
    try_rewrite_transpose_product_identity_expr,
};
use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, ExprId};

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
        let rewrite = Rewrite::new(eval.result.to_expr(ctx)).desc(eval.mul_desc());

        // Matrix multiplication is a definitional evaluation: each output entry is a
        // sum of `inner_dim` products, so the result can transiently exceed the
        // anti-worsen node budget (unfolded `a*c + b*d` sums) before the recursive
        // pass folds the arithmetic. Exempt it so valid products — including
        // non-square ones whose entries are large unfolded sums — are committed
        // instead of being rejected and falling through to a (wrong) scalar broadcast.
        //
        // The exemption is BOUNDED (so it cannot be abused to explode the tree): it
        // applies only when every dimension stays within MAX_N, the output cell count
        // stays within the output-size cap, and the contracted (inner) dimension stays
        // within the input-size cap. A product larger than that keeps the normal budget
        // — if it then worsens too much it stays an honest unevaluated residual rather
        // than blowing up. The bounds comfortably cover educational matrices (≤16×16).
        const MATRIX_MUL_MAX_N: usize = 16; // MAX_N: per-dimension cap
        const MATRIX_MUL_MAX_OUTPUT_CELLS: usize = 256; // output-size cap: result entries
        const MATRIX_MUL_MAX_INNER: usize = 16; // input-size cap: contracted dimension
        let rows = eval.result.rows;
        let cols = eval.result.cols;
        let inner = eval.left.cols; // == eval.right.rows (the contracted dimension)
        let within_caps = rows <= MATRIX_MUL_MAX_N
            && cols <= MATRIX_MUL_MAX_N
            && inner <= MATRIX_MUL_MAX_INNER
            && rows.saturating_mul(cols) <= MATRIX_MUL_MAX_OUTPUT_CELLS;

        Some(if within_caps {
            rewrite.budget_exempt()
        } else {
            rewrite
        })
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
