pub(super) fn format_health_usage_message(category_names: &str) -> String {
    format!(
        "Usage: health [on|off|reset|status]\n\
         \n\
               health               Show last health report\n\
               health on            Enable health tracking\n\
               health off           Disable health tracking\n\
               health reset         Clear health statistics\n\
               health status        Run diagnostic test suite\n\
               health status --list List available test cases\n\
               health status --category <cat>  Run only category\n\
                                            Categories: {}",
        category_names
    )
}

pub(super) fn health_usage_message() -> String {
    format_health_usage_message(&crate::health_suite_format_catalog::category_names().join(", "))
}

pub(super) fn format_health_missing_category_arg_message(category_names: &str) -> String {
    format!(
        "Error: --category requires an argument\nAvailable categories: {}",
        category_names
    )
}

pub(super) fn format_health_invalid_category_message(error: &str, category_names: &str) -> String {
    format!("Error: {}\nAvailable categories: {}", error, category_names)
}

pub(super) fn format_health_status_running_message(category_label: &str) -> String {
    format!(
        "Running health status suite [category={}]...\n",
        category_label
    )
}

pub(super) fn format_health_failed_tests_warning_line(failed: usize) -> Option<String> {
    if failed == 0 {
        None
    } else {
        Some(format!(
            "\n⚠ {} tests failed. Check Transform rules for churn.",
            failed
        ))
    }
}
