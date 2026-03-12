use cas_solver_core::health_suite_models::HealthCaseResult;

use super::super::lines::push_basic_result_line;
use super::super::summary::append_summary;
use super::header::push_report_header;

/// Format suite results as a human-readable report.
#[allow(dead_code)]
pub fn format_report(results: &[HealthCaseResult]) -> String {
    let mut report = String::new();
    push_report_header(&mut report, None);

    let mut passed = 0;
    let mut failed = 0;

    for result in results {
        push_basic_result_line(&mut report, result, &mut passed, &mut failed);
    }

    append_summary(&mut report, passed, failed);
    report
}
