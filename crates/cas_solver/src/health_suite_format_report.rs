use crate::health_suite_types::{Category, HealthCaseResult};

/// Format suite results as a human-readable report.
#[allow(dead_code)]
pub fn format_report(results: &[HealthCaseResult]) -> String {
    let mut report = String::new();
    report.push_str("Health Status Suite\n");
    report.push_str("═══════════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    for r in results {
        let name_padded = format!("{:25}", r.case.name);

        if r.passed {
            passed += 1;
            let status = if r.warning.is_some() { "⚠" } else { "✔" };
            report.push_str(&format!(
                "{} {}  rewrites={:3} growth={:+4} transform={:2}",
                status, name_padded, r.total_rewrites, r.growth, r.transform_rewrites
            ));
            if let Some(ref warn) = r.warning {
                report.push_str(&format!(" [{}]", warn));
            }
            report.push('\n');
        } else {
            failed += 1;
            report.push_str(&format!(
                "✘ {}  FAILED: {}\n",
                name_padded,
                r.failure_reason.as_ref().unwrap_or(&"unknown".to_string())
            ));
            if let Some((phase, period)) = &r.cycle_detected {
                report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
            }
            if !r.top_rules.is_empty() {
                let rules: Vec<_> = r
                    .top_rules
                    .iter()
                    .map(|(n, c)| format!("{}={}", n, c))
                    .collect();
                report.push_str(&format!("    Top Transform: {}\n", rules.join(", ")));
            }
        }
    }

    report.push_str("═══════════════════════════════════════════════════════════════\n");
    let total = passed + failed;
    if failed == 0 {
        report.push_str(&format!("PASSED: {}/{} cases ✓\n", passed, total));
    } else {
        report.push_str(&format!("FAILED: {}/{} cases\n", failed, total));
    }

    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");
    report
}

/// Count passed/failed for summary.
pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    (passed, failed)
}

/// Format report with category header.
pub fn format_report_filtered(results: &[HealthCaseResult], category: Option<Category>) -> String {
    let mut report = String::new();

    let header = match category {
        Some(cat) => format!("Health Status Suite [category={}]\n", cat),
        None => "Health Status Suite\n".to_string(),
    };
    report.push_str(&header);
    report.push_str("═══════════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    for r in results {
        let name_padded = format!("{:25}", r.case.name);
        let phases = format!(
            "(c={} t={} r={} p={})",
            r.core_rewrites, r.transform_rewrites, r.rationalize_rewrites, r.post_rewrites
        );
        let growth_str = if r.shrink > 0 {
            format!("+{:3}/-{}", r.growth, r.shrink)
        } else {
            format!("+{:3}", r.growth)
        };

        if r.passed {
            passed += 1;
            let status = if r.warning.is_some() { "⚠" } else { "✔" };
            report.push_str(&format!(
                "{} {}  rw={:3} {} {}\n",
                status, name_padded, r.total_rewrites, phases, growth_str
            ));
            if let Some(ref warn) = r.warning {
                report.push_str(&format!("    └─ {}\n", warn));
            }
        } else {
            failed += 1;
            report.push_str(&format!(
                "✘ {}  FAILED: {}\n",
                name_padded,
                r.failure_reason.as_ref().unwrap_or(&"unknown".to_string())
            ));
            report.push_str(&format!(
                "    rw={} {} {}\n",
                r.total_rewrites, phases, growth_str
            ));
            if let Some((phase, period)) = &r.cycle_detected {
                report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
            }
            if !r.top_rules.is_empty() {
                let rules: Vec<_> = r
                    .top_rules
                    .iter()
                    .map(|(n, c)| format!("{}={}", n, c))
                    .collect();
                report.push_str(&format!("    Top Transform: {}\n", rules.join(", ")));
            }
        }
    }

    report.push_str("═══════════════════════════════════════════════════════════════\n");
    let total = passed + failed;
    if failed == 0 {
        report.push_str(&format!("PASSED: {}/{} cases ✓\n", passed, total));
    } else {
        report.push_str(&format!("FAILED: {}/{} cases\n", failed, total));
    }

    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");
    report
}
