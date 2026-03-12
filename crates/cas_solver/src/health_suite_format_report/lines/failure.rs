use cas_solver_core::health_suite_models::HealthCaseResult;

pub(super) fn append_failure_details(
    report: &mut String,
    result: &HealthCaseResult,
    include_rw_line: bool,
) {
    if include_rw_line {
        // detailed report already printed the rw/phases/growth line above
    }
    if let Some((phase, period)) = &result.cycle_detected {
        report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
    }
    if !result.top_rules.is_empty() {
        let rules: Vec<_> = result
            .top_rules
            .iter()
            .map(|(n, c)| format!("{}={}", n, c))
            .collect();
        report.push_str(&format!("    Top Transform: {}\n", rules.join(", ")));
    }
}
