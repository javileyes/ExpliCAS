use crate::eval_command_types::EvalCommandEvalView;

pub(super) fn format_assumption_lines(
    output: &EvalCommandEvalView,
    assumption_reporting: crate::AssumptionReporting,
) -> Vec<String> {
    if assumption_reporting == crate::AssumptionReporting::Off {
        return Vec::new();
    }

    let assumed_conditions = crate::collect_assumed_conditions_from_steps(&output.steps);
    if assumed_conditions.is_empty() {
        Vec::new()
    } else {
        crate::format_assumed_conditions_report_lines(&assumed_conditions)
    }
}
