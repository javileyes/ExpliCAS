use super::{
    clear_health_profiler, evaluate_health_command, evaluate_health_command_input,
    evaluate_health_status_lines, HealthStatusInput,
};

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
