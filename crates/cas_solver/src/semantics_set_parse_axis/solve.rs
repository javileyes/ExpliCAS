pub(super) fn set_solve_axis(value: &str) -> Option<String> {
    match value {
        "check" => Some(
            "ERROR: Use 'semantics set solve check on' or 'semantics set solve check off'"
                .to_string(),
        ),
        _ => Some(format!(
            "ERROR: Invalid value '{}' for axis 'solve'\nAllowed: 'check on', 'check off'",
            value
        )),
    }
}
