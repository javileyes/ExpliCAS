use cas_ast::{Context, ExprId};

use super::{detect::detect_substitute_strategy, SubstituteOptions, SubstituteStrategy};

pub(super) fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    cas_math::substitute::substitute_power_aware(ctx, root, target, replacement, opts)
}

pub(super) fn substitute_auto(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    match detect_substitute_strategy(ctx, target) {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    }
}

pub(super) fn substitute_auto_with_strategy(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> (ExprId, SubstituteStrategy) {
    let strategy = detect_substitute_strategy(ctx, target);
    let expr = match strategy {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    };
    (expr, strategy)
}
