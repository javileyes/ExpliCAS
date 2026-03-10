use cas_api_models::{EvalJsonOutput, ExprStatsJson};
use cas_ast::{Context, SolutionSet};

use crate::eval_output_finalize::{build_eval_output, EvalOutputResultPayload};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;
use crate::eval_output_presentation::{format_output_solution_set, solution_set_to_output_latex};

fn build_nonexpr_result_payload(
    result: String,
    result_latex: Option<String>,
) -> EvalOutputResultPayload {
    EvalOutputResultPayload {
        result_chars: result.len(),
        result,
        result_truncated: false,
        result_latex,
        stats: ExprStatsJson::default(),
        hash: None,
    }
}

pub(crate) fn finalize_solution_set_output(
    ctx: &Context,
    solution_set: &SolutionSet,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalJsonOutput {
    let result_str = format_output_solution_set(ctx, solution_set);
    let result_latex = solution_set_to_output_latex(ctx, solution_set);
    let steps_count = shared.combined_steps_count();
    build_eval_output(
        build_nonexpr_result_payload(result_str, Some(result_latex)),
        steps_count,
        shared,
    )
}

pub(crate) fn finalize_bool_output(
    value: bool,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalJsonOutput {
    let result_str = value.to_string();
    let steps_count = shared.primary_steps_count();
    build_eval_output(
        build_nonexpr_result_payload(result_str, None),
        steps_count,
        shared,
    )
}
