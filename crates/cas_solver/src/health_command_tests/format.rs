use super::{
    format_health_failed_tests_warning_line, format_health_missing_category_arg_message,
    format_health_report_lines, format_health_status_running_message, format_health_usage_message,
    health_usage_message, resolve_health_category_filter, HealthStatusInput,
};

#[test]
fn format_health_usage_message_mentions_status() {
    let text = format_health_usage_message("a,b,c");
    assert!(text.contains("health status"));
    assert!(text.contains("Categories: a,b,c"));
}

#[test]
fn health_usage_message_includes_categories() {
    let text = health_usage_message();
    assert!(text.contains("Categories:"));
}

#[test]
fn format_health_missing_category_arg_message_mentions_argument() {
    let text = format_health_missing_category_arg_message("a,b,c");
    assert!(text.contains("--category requires an argument"));
}

#[test]
fn format_health_report_lines_without_report_includes_hint() {
    let lines = format_health_report_lines(None, None);
    assert!(lines
        .iter()
        .any(|line| line.contains("No health report available.")));
    assert!(lines
        .iter()
        .any(|line| line.contains("Enable with: health on")));
}

#[test]
fn resolve_health_category_filter_handles_all_and_missing() {
    let names = "algebra,analysis";
    let status_all = HealthStatusInput {
        list_only: false,
        category: Some("all".to_string()),
        category_missing_arg: false,
    };
    let all = resolve_health_category_filter::<u8, String, _>(&status_all, names, |_raw| {
        Err("unused".to_string())
    })
    .expect("all");
    assert!(all.is_none());

    let status_missing = HealthStatusInput {
        list_only: false,
        category: None,
        category_missing_arg: true,
    };
    let err = resolve_health_category_filter::<u8, String, _>(&status_missing, names, |_raw| {
        Err("unused".to_string())
    })
    .expect_err("missing");
    assert!(err.contains("--category requires an argument"));
}

#[test]
fn format_health_status_running_message_includes_category() {
    let line = format_health_status_running_message("algebra");
    assert!(line.contains("category=algebra"));
}

#[test]
fn format_health_failed_tests_warning_line_only_for_nonzero() {
    assert!(format_health_failed_tests_warning_line(0).is_none());
    let line = format_health_failed_tests_warning_line(2).expect("warning");
    assert!(line.contains("2 tests failed"));
}
