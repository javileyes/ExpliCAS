use super::super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn generate_inner_fraction_substeps(
    ctx: &Context,
    inner_frac: ExprId,
    num_str: &str,
    before_str: &str,
    after_str: &str,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let _ = (ctx, inner_frac, num_str, before_str, after_str, hints);
    Vec::new()
}
