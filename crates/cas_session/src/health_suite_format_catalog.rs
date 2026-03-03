use crate::health_suite_catalog::default_suite;
use crate::health_suite_types::Category;

/// List all available test cases.
pub fn list_cases() -> String {
    let suite = default_suite();
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

/// Get all available category names for autocomplete.
pub fn category_names() -> Vec<&'static str> {
    Category::all().iter().map(|c| c.as_str()).collect()
}
