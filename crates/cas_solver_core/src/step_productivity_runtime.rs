//! Shared runtime adapter for filtering non-productive simplification steps.

use cas_ast::{Context, Expr, ExprId};

/// Filter out steps that do not materially change the global expression.
///
/// This keeps core behavior shared while letting runtime crates provide:
/// - how to read step fields,
/// - which rules are always kept,
/// - how to rebuild `Mul` nodes in the host runtime.
#[allow(clippy::too_many_arguments)]
pub fn filter_non_productive_steps_with_runtime_recompose_mul<
    T,
    FAlwaysKeep,
    FBefore,
    FAfter,
    FPath,
    FDisplayNoop,
    FRecomposeMul,
>(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<T>,
    always_keep: FAlwaysKeep,
    before_of: FBefore,
    after_of: FAfter,
    path_of: FPath,
    is_display_noop: FDisplayNoop,
    mut recompose_mul: FRecomposeMul,
) -> Vec<T>
where
    FAlwaysKeep: FnMut(&T) -> bool,
    FBefore: FnMut(&T) -> ExprId,
    FAfter: FnMut(&T) -> ExprId,
    FPath: FnMut(&T) -> Vec<u8>,
    FDisplayNoop: FnMut(&Context, ExprId, ExprId) -> bool,
    FRecomposeMul: FnMut(&mut Context, ExprId, ExprId) -> ExprId,
{
    cas_math::step_productivity::filter_non_productive_steps_with(
        ctx,
        original,
        steps,
        always_keep,
        before_of,
        after_of,
        path_of,
        is_display_noop,
        |ctx, root, path, replacement| {
            cas_math::expr_path_rewrite::rewrite_at_expr_path_with(
                ctx,
                root,
                path,
                replacement,
                &mut |ctx, expr| match expr {
                    Expr::Mul(l, r) => recompose_mul(ctx, l, r),
                    other => ctx.add(other),
                },
            )
        },
    )
}

/// `Step`-specialized productivity filter used by runtime orchestrators.
///
/// Keeps behavior stable with the historical engine implementation:
/// - always-keep rules from `cas_math::step_rules`
/// - display-noop detection via formatted before/after string equality
/// - global rewrite via expr-path replacement.
pub fn filter_non_productive_solver_steps_with_runtime_recompose_mul<FRecomposeMul>(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<crate::step_model::Step>,
    recompose_mul: FRecomposeMul,
) -> Vec<crate::step_model::Step>
where
    FRecomposeMul: FnMut(&mut Context, ExprId, ExprId) -> ExprId,
{
    filter_non_productive_steps_with_runtime_recompose_mul(
        ctx,
        original,
        steps,
        |step: &crate::step_model::Step| {
            cas_math::step_rules::is_always_keep_step_rule_name(&step.rule_name)
        },
        |step: &crate::step_model::Step| step.before,
        |step: &crate::step_model::Step| step.after,
        |step: &crate::step_model::Step| crate::step_types::pathsteps_to_expr_path(step.path()),
        |ctx, before, after| {
            let before_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: before
                }
            );
            let after_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: after
                }
            );
            before_str == after_str
        },
        recompose_mul,
    )
}
