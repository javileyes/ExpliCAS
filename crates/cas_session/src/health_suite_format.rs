use crate::health_suite_catalog::default_suite;
use crate::health_suite_types::{Category, HealthCaseResult};

/// Format suite results as a human-readable report
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
            // Show warning indicator if present
            let status = if r.warning.is_some() { "⚠" } else { "✔" };
            report.push_str(&format!(
                "{} {}  rewrites={:3} growth={:+4} transform={:2}",
                status, name_padded, r.total_rewrites, r.growth, r.transform_rewrites
            ));
            // Append warning message
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
            // Show cycle info if present
            if let Some((phase, period)) = &r.cycle_detected {
                report.push_str(&format!("    Cycle: {:?} period={}\n", phase, period));
            }
            // Show top rules
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

    // Add legend
    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");

    report
}

/// Count passed/failed for summary
pub fn count_results(results: &[HealthCaseResult]) -> (usize, usize) {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    (passed, failed)
}

/// List all available test cases
pub fn list_cases() -> String {
    let suite = default_suite();
    let mut output = format!("Available health cases ({}):\n", suite.len());
    output.push_str("─────────────────────────────────────────\n");

    // Group by category
    for cat in Category::all() {
        let cases_in_cat: Vec<_> = suite.iter().filter(|c| c.category == *cat).collect();
        if !cases_in_cat.is_empty() {
            for case in cases_in_cat {
                output.push_str(&format!("[{:14}] {}\n", cat.as_str(), case.name));
            }
        }
    }
    output
}

/// Get all available category names for autocomplete
pub fn category_names() -> Vec<&'static str> {
    Category::all().iter().map(|c| c.as_str()).collect()
}

/// Format report with category header
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
        // Format per-phase rewrites compactly
        let phases = format!(
            "(c={} t={} r={} p={})",
            r.core_rewrites, r.transform_rewrites, r.rationalize_rewrites, r.post_rewrites
        );
        // Format growth/shrink
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

    // Add legend
    report.push_str("\nLegend: rw=total rewrites, c=Core, t=Transform, r=Rationalize, p=Post\n");

    report
}
