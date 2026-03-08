use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;

use super::{SubstituteOptions, SubstituteResult, SubstituteStep};

/// Perform power-aware substitution with step collection.
pub fn substitute_with_steps(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> SubstituteResult {
    let trace = cas_math::substitute::substitute_with_trace(ctx, root, target, replacement, opts);
    let steps = trace
        .steps
        .into_iter()
        .map(|step| SubstituteStep {
            rule: step.rule,
            before: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.before,
                }
            ),
            after: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.after,
                }
            ),
            note: step.note,
        })
        .collect();

    SubstituteResult {
        expr: trace.expr,
        steps,
    }
}
