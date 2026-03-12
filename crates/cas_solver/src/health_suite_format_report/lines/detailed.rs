use cas_solver_core::health_suite_models::HealthCaseResult;

use super::failure::append_failure_details;

pub(crate) fn push_detailed_result_line(
    report: &mut String,
    result: &HealthCaseResult,
    passed: &mut usize,
    failed: &mut usize,
) {
    let name_padded = format!("{:25}", result.case.name);
    let phases = format!(
        "(c={} t={} r={} p={})",
        result.core_rewrites,
        result.transform_rewrites,
        result.rationalize_rewrites,
        result.post_rewrites
    );
    let growth_str = if result.shrink > 0 {
        format!("+{:3}/-{}", result.growth, result.shrink)
    } else {
        format!("+{:3}", result.growth)
    };

    if result.passed {
        *passed += 1;
        let status = if result.warning.is_some() {
            "⚠"
        } else {
            "✔"
        };
        report.push_str(&format!(
            "{} {}  rw={:3} {} {}\n",
            status, name_padded, result.total_rewrites, phases, growth_str
        ));
        if let Some(ref warn) = result.warning {
            report.push_str(&format!("    └─ {}\n", warn));
        }
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
        report.push_str(&format!(
            "    rw={} {} {}\n",
            result.total_rewrites, phases, growth_str
        ));
        append_failure_details(report, result, true);
    }
}
