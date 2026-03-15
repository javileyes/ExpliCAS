mod lines;
mod render;
mod summary;

use crate::health_category::Category;
use crate::health_suite_models::HealthCaseResult;

#[allow(dead_code)]
pub fn format_report(results: &[HealthCaseResult]) -> String {
    self::render::format_report(results)
}

pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    self::summary::count_results(results)
}

pub fn format_report_filtered(results: &[HealthCaseResult], category: Option<Category>) -> String {
    self::render::format_report_filtered(results, category)
}
