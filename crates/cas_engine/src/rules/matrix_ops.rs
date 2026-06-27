use crate::matrix_rule_support::{
    try_eval_matrix_add_expr, try_eval_matrix_mul_expr, try_eval_matrix_sub_expr,
    try_eval_scalar_matrix_mul_expr, try_matrix_shape_mismatch_undefined,
    try_rewrite_matrix_binary_function_expr, try_rewrite_matrix_function_rule_expr,
    try_rewrite_matrix_reciprocal_expr, try_rewrite_transpose_product_identity_expr,
};
use crate::rule::{Rewrite, SimpleRule};
use cas_ast::{Context, Expr, ExprId};
use cas_math::matrix::Matrix;

/// Rule that maps a shape-INCOMPATIBLE matrix operation to `undefined`.
///
/// `[[1,2],[3,4]] + [[1,2,3],[4,5,6]]`, `M·N` with mismatched inner dimensions, a non-square
/// `M^2`, and `matrix ± scalar` have no value; without this guard the evaluation rules simply
/// decline (returning `None`) and the malformed operation is echoed back as if it were a valid
/// result with `ok:true`. Routing it to the engine's `undefined` sentinel makes the dishonesty
/// explicit. Runs at high priority so it fires before the (declining) evaluation rules.
pub struct MatrixShapeGuardRule;

impl SimpleRule for MatrixShapeGuardRule {
    fn name(&self) -> &'static str {
        "Matrix Shape Guard"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(
            crate::target_kind::TargetKindSet::ADD
                | crate::target_kind::TargetKindSet::SUB
                | crate::target_kind::TargetKindSet::MUL
                | crate::target_kind::TargetKindSet::POW,
        )
    }

    fn priority(&self) -> i32 {
        // Above the matrix evaluation rules and the reciprocal rule (20) so a malformed op
        // becomes `undefined` before anything else inspects it. Well-formed ops return `None`
        // here and fall through unchanged.
        25
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let undefined = try_matrix_shape_mismatch_undefined(ctx, expr)?;
        Some(Rewrite::new(undefined).desc("Incompatible matrix shapes ⇒ undefined"))
    }
}

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
        // `charpoly`/`det` of a NUMERIC matrix expand to an UNFOLDED cofactor sum that
        // transiently exceeds the anti-worsen node budget before the recursive pass folds
        // it to a number / polynomial; with a tiny numeric input it is rejected, leaving a
        // residual (`charpoly([[1,0,0],[0,2,0],[0,0,3]])`). Exempt bounded-size matrix
        // functions — like MatrixMultiplyRule — so they commit.
        //
        // GATED TO ALL-NUMERIC operands on purpose: a SYMBOLIC `inverse([[a,b],[c,d]])`
        // must stay an honest residual (its `1/(ad−bc)` formula is undefined when the
        // determinant is zero, a condition the residual deliberately withholds). Symbolic
        // `charpoly` already commits under the normal budget because its input is large, so
        // it needs no exemption — only the small-numeric case does.
        const MATRIX_FN_MAX_N: usize = 6;
        let bounded_all_numeric = |ctx: &Context, arg: ExprId| -> bool {
            Matrix::from_expr(ctx, arg).is_some_and(|m| {
                m.rows <= MATRIX_FN_MAX_N
                    && m.cols <= MATRIX_FN_MAX_N
                    && m.data
                        .iter()
                        .all(|&e| matches!(ctx.get(e), Expr::Number(_) | Expr::Constant(_)))
            })
        };
        let operands: Vec<ExprId> = if let Expr::Function(_, args) = ctx.get(expr) {
            args.clone()
        } else {
            Vec::new()
        };

        if let Some(rewrite) = try_rewrite_matrix_function_rule_expr(ctx, expr) {
            let base = Rewrite::new(rewrite.rewritten).desc(rewrite.desc);
            let within_caps = operands
                .first()
                .is_some_and(|&arg| bounded_all_numeric(ctx, arg));
            return Some(if within_caps {
                base.budget_exempt()
            } else {
                base
            });
        }

        // Binary (2-arg) matrix/vector operations: dot, cross, linsolve.
        if let Some(rewrite) = try_rewrite_matrix_binary_function_expr(ctx, expr) {
            let base = Rewrite::new(rewrite.rewritten).desc(rewrite.desc);
            let within_caps =
                operands.len() == 2 && operands.iter().all(|&arg| bounded_all_numeric(ctx, arg));
            return Some(if within_caps {
                base.budget_exempt()
            } else {
                base
            });
        }

        None
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

/// MatrixReciprocalRule: `M^(-1)` and `c / M` route to the matrix inverse instead of falling to
/// scalar arithmetic, which fabricated `1/[[…]]` (a non-square matrix has NO inverse, and a
/// symbolic one is not elementwise `1/entry`).
pub struct MatrixReciprocalRule;

impl SimpleRule for MatrixReciprocalRule {
    fn name(&self) -> &'static str {
        "Matrix Reciprocal/Inverse"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW | crate::target_kind::TargetKindSet::DIV)
    }

    fn priority(&self) -> i32 {
        // Beat the scalar reciprocal/power rules so `M^(-1)` / `c/M` never fabricate `1/[[…]]`.
        20
    }

    fn apply_simple(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
        let rewritten = try_rewrite_matrix_reciprocal_expr(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc("M^(-1) = inverse(M); c/M = c·inverse(M)"))
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(MatrixShapeGuardRule));
    simplifier.add_rule(Box::new(MatrixReciprocalRule));
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
