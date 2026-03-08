use super::{parse_health_command_input, HealthCommandInput, HealthStatusInput};

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
