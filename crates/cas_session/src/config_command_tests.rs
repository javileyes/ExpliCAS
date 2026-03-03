#[cfg(test)]
mod tests {
    use crate::config_command_eval::evaluate_config_command;
    use crate::config_command_parse::parse_config_command_input;
    use crate::config_command_types::{ConfigCommandInput, ConfigCommandResult};
    use crate::{evaluate_and_apply_config_command, ConfigCommandApplyOutput};

    #[test]
    fn parse_config_command_input_reads_enable_rule() {
        assert_eq!(
            parse_config_command_input("config enable distribute"),
            ConfigCommandInput::SetRule {
                rule: "distribute".to_string(),
                enable: true,
            }
        );
    }

    #[test]
    fn parse_config_command_input_detects_missing_rule() {
        assert_eq!(
            parse_config_command_input("config disable"),
            ConfigCommandInput::MissingRuleArg {
                action: "disable".to_string(),
            }
        );
    }

    #[test]
    fn parse_config_command_input_detects_unknown_subcommand() {
        assert_eq!(
            parse_config_command_input("config nope"),
            ConfigCommandInput::UnknownSubcommand {
                subcommand: "nope".to_string(),
            }
        );
    }

    #[test]
    fn evaluate_config_command_set_rule_applies_toggle_config() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config enable distribute", toggles);
        match result {
            ConfigCommandResult::ApplyToggleConfig { toggles, message } => {
                assert!(toggles.distribute);
                assert!(message.contains("Rule 'distribute' set to true."));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn evaluate_config_command_list_formats_message() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config list", toggles);
        match result {
            ConfigCommandResult::ShowList { message } => {
                assert!(message.contains("distribute:"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn evaluate_config_command_invalid_usage_returns_error() {
        let toggles = crate::SimplifierToggleConfig::default();
        let result = evaluate_config_command("config", toggles);
        assert!(matches!(result, ConfigCommandResult::Error { .. }));
    }

    #[test]
    fn evaluate_and_apply_config_command_updates_config_and_syncs() {
        let mut config = crate::CasConfig::default();
        let out = evaluate_and_apply_config_command("config enable distribute", &mut config);
        assert_eq!(
            out,
            ConfigCommandApplyOutput {
                message: "Rule 'distribute' set to true.".to_string(),
                sync_simplifier: true,
            }
        );
        assert!(config.distribute);
    }
}
