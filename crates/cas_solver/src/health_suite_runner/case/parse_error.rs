use crate::health_suite_types::{HealthCase, HealthCaseResult};

pub(super) fn parse_error_result(case: &HealthCase, error: String) -> HealthCaseResult {
    HealthCaseResult {
        case: case.clone(),
        passed: false,
        total_rewrites: 0,
        core_rewrites: 0,
        transform_rewrites: 0,
        rationalize_rewrites: 0,
        post_rewrites: 0,
        growth: 0,
        shrink: 0,
        cycle_detected: None,
        top_rules: vec![],
        failure_reason: Some(format!("Parse error: {}", error)),
        warning: None,
    }
}
