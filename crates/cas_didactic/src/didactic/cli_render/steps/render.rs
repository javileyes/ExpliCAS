pub(super) mod lines;
pub(super) mod visibility;

use crate::runtime::Step;
use cas_ast::ExprId;

pub(super) fn local_rule_expr_ids(step: &Step) -> (ExprId, ExprId) {
    match (step.before_local(), step.after_local()) {
        (Some(bl), Some(al)) => (bl, al),
        _ => (step.before, step.after),
    }
}
