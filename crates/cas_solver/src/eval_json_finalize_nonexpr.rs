use cas_api_models::{EvalJsonOutput, ExprStatsJson};
use cas_ast::{Context, SolutionSet};

use crate::eval_json_finalize::{build_eval_json_output, EvalJsonResultPayload};
use crate::eval_json_finalize_input::EvalJsonFinalizeShared;
use crate::eval_json_presentation::{
    format_solution_set_eval_json, solution_set_to_latex_eval_json,
};

fn build_nonexpr_result_payload(
    result: String,
    result_latex: Option<String>,
) -> EvalJsonResultPayload {
    EvalJsonResultPayload {
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
    shared: EvalJsonFinalizeShared<'_>,
) -> EvalJsonOutput {
    let result_str = format_solution_set_eval_json(ctx, solution_set);
    let result_latex = solution_set_to_latex_eval_json(ctx, solution_set);
    let steps_count = shared.combined_steps_count();
    build_eval_json_output(
        build_nonexpr_result_payload(result_str, Some(result_latex)),
        steps_count,
        shared,
    )
}

pub(crate) fn finalize_bool_output(
    value: bool,
    shared: EvalJsonFinalizeShared<'_>,
) -> EvalJsonOutput {
    let result_str = value.to_string();
    let steps_count = shared.primary_steps_count();
    build_eval_json_output(
        build_nonexpr_result_payload(result_str, None),
        steps_count,
        shared,
    )
}
