mod list;
mod names;

use crate::health_suite_catalog::default_suite;

/// List all available test cases.
pub fn list_cases() -> String {
    list::list_suite_cases(default_suite())
}

/// Get all available category names for autocomplete.
pub fn category_names() -> Vec<&'static str> {
    names::category_name_list()
}
