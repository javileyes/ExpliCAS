use crate::{
    clear_health_profiler, evaluate_health_command, evaluate_health_command_input,
    evaluate_health_status_lines, format_health_failed_tests_warning_line,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    parse_health_command_input, resolve_health_category_filter, HealthCommandInput,
    HealthStatusInput,
};

#[test]
fn parse_health_command_input_status_list() {
    assert_eq!(
        parse_health_command_input("health status --list"),
        HealthCommandInput::Status(HealthStatusInput {
            list_only: true,
            category: None,
            category_missing_arg: false,
        })
    );
}

#[test]
fn parse_health_command_input_status_category() {
    assert_eq!(
        parse_health_command_input("health status --category algebra"),
        HealthCommandInput::Status(HealthStatusInput {
            list_only: false,
            category: Some("algebra".to_string()),
            category_missing_arg: false,
        })
    );
}

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

#[test]
fn evaluate_health_command_input_maps_invalid_to_usage() {
    let err = evaluate_health_command_input("health nope").expect_err("invalid");
    assert!(err.contains("Usage: health"));
}

#[test]
fn evaluate_health_status_lines_list_only() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lines = evaluate_health_status_lines(
        &mut simplifier,
        &HealthStatusInput {
            list_only: true,
            category: None,
            category_missing_arg: false,
        },
    )
    .expect("list");
    assert!(lines
        .first()
        .is_some_and(|line| line.contains("Available health cases")));
}

#[test]
fn evaluate_health_status_lines_invalid_category_returns_error() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let err = evaluate_health_status_lines(
        &mut simplifier,
        &HealthStatusInput {
            list_only: false,
            category: Some("unknown".to_string()),
            category_missing_arg: false,
        },
    )
    .expect_err("invalid category");
    assert!(err.contains("Available categories"));
}

#[test]
fn clear_health_profiler_resets_profiler_state() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    clear_health_profiler(&mut simplifier);
}

#[test]
fn evaluate_health_command_show_last_uses_report_lines() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let out = evaluate_health_command(&mut simplifier, "health", None, None).expect("health");
    assert!(out.set_enabled.is_none());
    assert!(!out.clear_last_report);
    assert!(out
        .lines
        .iter()
        .any(|line| line.contains("No health report available.")));
}

#[test]
fn evaluate_health_command_enable_sets_flag() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let out = evaluate_health_command(&mut simplifier, "health on", None, None).expect("health on");
    assert_eq!(out.set_enabled, Some(true));
    assert!(!out.clear_last_report);
    assert!(out
        .lines
        .first()
        .is_some_and(|line| line.contains("ENABLED")));
}

#[test]
fn evaluate_health_command_clear_requests_report_clear() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let out =
        evaluate_health_command(&mut simplifier, "health clear", None, None).expect("health clear");
    assert!(out.set_enabled.is_none());
    assert!(out.clear_last_report);
    assert!(out
        .lines
        .first()
        .is_some_and(|line| line.contains("Health statistics cleared")));
}
