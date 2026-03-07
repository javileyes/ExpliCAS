use cas_ast::{Context, Expr, ExprId};

use super::SubstituteStrategy;

pub(super) fn detect_substitute_strategy(ctx: &Context, target: ExprId) -> SubstituteStrategy {
    match ctx.get(target) {
        Expr::Variable(_) => SubstituteStrategy::Variable,
        _ => SubstituteStrategy::PowerAware,
    }
}
