pub(super) fn fallback_health_report_lines() -> Vec<String> {
    vec![
        "No health report available.".to_string(),
        "Run a simplification first (health is captured when debug mode or health mode is on)."
            .to_string(),
        "Enable with: health on".to_string(),
    ]
}
