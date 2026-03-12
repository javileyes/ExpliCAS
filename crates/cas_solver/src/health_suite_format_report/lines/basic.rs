use cas_solver_core::health_suite_models::HealthCaseResult;

use super::failure::append_failure_details;

pub(crate) fn push_basic_result_line(
    report: &mut String,
    result: &HealthCaseResult,
    passed: &mut usize,
    failed: &mut usize,
) {
    let name_padded = format!("{:25}", result.case.name);

    if result.passed {
        *passed += 1;
        let status = if result.warning.is_some() {
            "⚠"
        } else {
            "✔"
        };
        report.push_str(&format!(
            "{} {}  rewrites={:3} growth={:+4} transform={:2}",
            status, name_padded, result.total_rewrites, result.growth, result.transform_rewrites
        ));
        if let Some(ref warn) = result.warning {
            report.push_str(&format!(" [{}]", warn));
        }
        report.push('\n');
    } else {
        *failed += 1;
        report.push_str(&format!(
            "✘ {}  FAILED: {}\n",
            name_padded,
            result
                .failure_reason
                .as_ref()
                .unwrap_or(&"unknown".to_string())
        ));
        append_failure_details(report, result, false);
    }
}
