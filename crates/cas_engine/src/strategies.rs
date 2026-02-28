use crate::build::mul2_raw;
use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use cas_math::step_productivity::filter_non_productive_steps_with;
use cas_math::step_rules::is_always_keep_step_rule_name;

/// Filter out steps that don't change the global expression state
pub fn filter_non_productive_steps(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<Step>,
) -> Vec<Step> {
    filter_non_productive_steps_with(
        ctx,
        original,
        steps,
        |step| is_always_keep_step_rule_name(&step.rule_name),
        |step| step.before,
        |step| step.after,
        |step| crate::step::pathsteps_to_expr_path(step.path()),
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
        |ctx, root, path, replacement| {
            cas_math::expr_path_rewrite::rewrite_at_expr_path_with(
                ctx,
                root,
                path,
                replacement,
                &mut |ctx, expr| match expr {
                    Expr::Mul(l, r) => mul2_raw(ctx, l, r),
                    other => ctx.add(other),
                },
            )
        },
    )
}
