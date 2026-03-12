use cas_solver_core::health_category::Category;

pub(super) fn push_report_header(report: &mut String, category: Option<Category>) {
    let header = match category {
        Some(cat) => format!("Health Status Suite [category={}]\n", cat),
        None => "Health Status Suite\n".to_string(),
    };
    report.push_str(&header);
    report.push_str("═══════════════════════════════════════════════════════════════\n");
}
