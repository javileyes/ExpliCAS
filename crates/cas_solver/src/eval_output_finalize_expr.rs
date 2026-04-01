use cas_ast::{Context, ExprId};
use cas_formatter::ParseStyleSignals;

use crate::eval_output_finalize::{build_eval_output, EvalOutputResultPayload, EvalOutputWire};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

fn build_expr_result_payload(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    style_signals: &ParseStyleSignals,
) -> EvalOutputResultPayload {
    let (result_str, truncated, char_count) =
        crate::eval_output_stats::format_limited_output_expr(ctx, result_expr, max_chars);
    let stats = crate::eval_output_stats::expr_output_stats(ctx, result_expr);
    let hash = if truncated {
        Some(crate::eval_output_stats::expr_output_hash(ctx, result_expr))
    } else {
        None
    };

    let result_latex = if !truncated {
        Some(crate::eval_output_latex_style::render_expr_latex_for_eval(
            ctx,
            result_expr,
            style_signals,
            crate::eval_output_latex_style::EvalLatexRenderIntent::Result,
        ))
    } else {
        None
    };

    EvalOutputResultPayload {
        result: result_str,
        result_truncated: truncated,
        result_chars: char_count,
        result_latex,
        stats,
        hash,
    }
}

pub(crate) fn finalize_expr_like_eval_output(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    let payload = build_expr_result_payload(ctx, result_expr, max_chars, &shared.style_signals);
    let steps_count = shared.primary_steps_count();

    build_eval_output(payload, steps_count, shared)
}
