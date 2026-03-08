use crate::health_suite_types::{Category, HealthCase};

pub(super) fn list_suite_cases(suite: Vec<HealthCase>) -> String {
    let mut output = format!("Available health cases ({}):\n", suite.len());
    output.push_str("─────────────────────────────────────────\n");

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
