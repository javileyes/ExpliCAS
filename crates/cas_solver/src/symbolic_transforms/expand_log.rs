use cas_ast::ExprId;

/// Recursively expand logarithmic identities where the expansion rule applies.
pub fn expand_log_recursive(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    cas_math::logarithm_inverse_support::expand_logs_collect_positive_assumptions(ctx, expr)
        .rewritten
}
