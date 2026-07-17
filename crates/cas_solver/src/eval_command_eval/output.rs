use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::{EvalCommandEvalView, EvalCommandOutput};
use crate::eval_command_format_metadata::format_eval_metadata_lines;
use crate::eval_command_format_result::{format_eval_result_line, format_eval_stored_entry_line};

pub(super) fn build_eval_command_output(
    context: &mut cas_ast::Context,
    eval_options: crate::EvalOptions,
    mut eval_view: EvalCommandEvalView,
    style_signals: ParseStyleSignals,
    debug_mode: bool,
) -> EvalCommandOutput {
    // `semantics set numeric decimal`: OUTPUT-BOUNDARY presentation only —
    // the evaluation above ran exact and symbolic; here the result maps its
    // maximal closed numeric subtrees to decimal display nodes (the same
    // walker as the one-shot `--numeric-display` axis).
    if eval_options.numeric_display == crate::NumericDisplayMode::Decimal {
        let complex_enabled = eval_options.shared.semantics.value_domain
            == cas_solver_core::value_domain::ValueDomain::ComplexEnabled;
        let present = |ctx: &mut cas_ast::Context, id: cas_ast::ExprId| {
            cas_math::numeric_presentation::present_numeric(ctx, id, complex_enabled).unwrap_or(id)
        };
        match &mut eval_view.result {
            crate::EvalResult::Expr(id) => {
                *id = present(context, *id);
            }
            crate::EvalResult::Set(ids) => {
                for id in ids.iter_mut() {
                    *id = present(context, *id);
                }
            }
            crate::EvalResult::SolutionSet(set) => match set {
                cas_ast::SolutionSet::Discrete(ids) => {
                    for id in ids.iter_mut() {
                        *id = present(context, *id);
                    }
                }
                cas_ast::SolutionSet::Continuous(interval) => {
                    interval.min = present(context, interval.min);
                    interval.max = present(context, interval.max);
                }
                cas_ast::SolutionSet::Union(intervals) => {
                    for interval in intervals.iter_mut() {
                        interval.min = present(context, interval.min);
                        interval.max = present(context, interval.max);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    let metadata = format_eval_metadata_lines(
        context,
        &eval_view,
        eval_options.requires_display,
        debug_mode,
        eval_options.hints_enabled,
        eval_options.shared.semantics.domain_mode,
        eval_options.shared.assumption_reporting,
    );
    let stored_entry_line = format_eval_stored_entry_line(context, &eval_view);
    let result_line =
        format_eval_result_line(context, eval_view.parsed, &eval_view.result, &style_signals);

    EvalCommandOutput {
        resolved_expr: eval_view.resolved,
        style_signals,
        steps: eval_view.steps,
        stored_entry_line,
        metadata,
        result_line,
    }
}
