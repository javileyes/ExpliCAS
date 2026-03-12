use cas_solver_core::health_suite_models::HealthCaseResult;

pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    (passed, failed)
}

pub(super) fn append_summary(report: &mut String, passed: usize, failed: usize) {
    report.push_str("═══════════════════════════════════════════════════════════════\n");
    let total = passed + failed;
    if failed == 0 {
        report.push_str(&format!("PASSED: {}/{} cases ✓\n", passed, total));
    } else {
        report.push_str(&format!("FAILED: {}/{} cases\n", failed, total));
    }

    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");
}
