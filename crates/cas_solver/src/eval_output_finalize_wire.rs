use cas_api_models::{wire::build_eval_wire_reply, WarningJson};

pub(crate) fn build_eval_output_wire_value(
    warnings: &[WarningJson],
    required_display: &[String],
    result: &str,
    result_latex: Option<&str>,
    steps_count: usize,
    steps_mode: &str,
) -> Option<serde_json::Value> {
    serde_json::to_value(build_eval_wire_reply(
        warnings,
        required_display,
        result,
        result_latex,
        steps_count,
        steps_mode,
    ))
    .ok()
}
