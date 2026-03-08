use crate::health_suite_types::{Category, HealthCaseResult};

use super::super::lines::push_detailed_result_line;
use super::super::summary::append_summary;
use super::header::push_report_header;

/// Format report with category header.
pub fn format_report_filtered(results: &[HealthCaseResult], category: Option<Category>) -> String {
    let mut report = String::new();
    push_report_header(&mut report, category);

    let mut passed = 0;
    let mut failed = 0;

    for result in results {
        push_detailed_result_line(&mut report, result, &mut passed, &mut failed);
    }

    append_summary(&mut report, passed, failed);
    report
}
