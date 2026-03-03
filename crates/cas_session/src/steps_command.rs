#![allow(unused_imports)]

pub use crate::steps_command_eval::{apply_steps_command_update, evaluate_steps_command_input};
pub use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
pub use crate::steps_command_parse::parse_steps_command_input;
pub use crate::steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};

#[cfg(test)]
mod tests {
    use super::{
        apply_steps_command_update, evaluate_steps_command_input, format_steps_current_message,
        format_steps_unknown_mode_message, parse_steps_command_input, StepsCommandApplyEffects,
        StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
    };

    #[test]
    fn parse_steps_command_input_reads_compact() {
        assert_eq!(
            parse_steps_command_input("steps compact"),
            StepsCommandInput::SetCollectionMode(cas_solver::StepsMode::Compact)
        );
    }

    #[test]
    fn parse_steps_command_input_reads_verbose_display_mode() {
        assert_eq!(
            parse_steps_command_input("steps verbose"),
            StepsCommandInput::SetDisplayMode(StepsDisplayMode::Verbose)
        );
    }

    #[test]
    fn format_steps_current_message_reports_modes() {
        let text =
            format_steps_current_message(cas_solver::StepsMode::On, StepsDisplayMode::Normal);
        assert!(text.contains("Steps collection: on"));
        assert!(text.contains("Steps display: normal"));
    }

    #[test]
    fn format_steps_unknown_mode_message_mentions_usage() {
        let text = format_steps_unknown_mode_message("oops");
        assert!(text.contains("Unknown steps mode: 'oops'"));
        assert!(text.contains("Usage: steps"));
    }

    #[test]
    fn evaluate_steps_command_input_off_disables_collection_and_display() {
        let state = StepsCommandState {
            steps_mode: cas_solver::StepsMode::On,
            display_mode: StepsDisplayMode::Normal,
        };
        let out = evaluate_steps_command_input("steps off", state);
        assert_eq!(
            out,
            StepsCommandResult::Update {
                set_steps_mode: Some(cas_solver::StepsMode::Off),
                set_display_mode: Some(StepsDisplayMode::None),
                message: "Steps: off\n  ⚡ Steps disabled (faster). Warnings still enabled."
                    .to_string(),
            }
        );
    }

    #[test]
    fn evaluate_steps_command_input_show_current_renders_message() {
        let state = StepsCommandState {
            steps_mode: cas_solver::StepsMode::Compact,
            display_mode: StepsDisplayMode::Succinct,
        };
        let out = evaluate_steps_command_input("steps", state);
        match out {
            StepsCommandResult::ShowCurrent { message } => {
                assert!(message.contains("Steps collection: compact"));
                assert!(message.contains("Steps display: succinct"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn apply_steps_command_update_sets_mode_when_changed() {
        let mut eval_options = cas_solver::EvalOptions {
            steps_mode: cas_solver::StepsMode::On,
            ..cas_solver::EvalOptions::default()
        };
        let effects = apply_steps_command_update(
            Some(cas_solver::StepsMode::Compact),
            Some(StepsDisplayMode::Succinct),
            &mut eval_options,
        );
        assert_eq!(eval_options.steps_mode, cas_solver::StepsMode::Compact);
        assert_eq!(
            effects,
            StepsCommandApplyEffects {
                set_steps_mode: Some(cas_solver::StepsMode::Compact),
                set_display_mode: Some(StepsDisplayMode::Succinct),
            }
        );
    }

    #[test]
    fn apply_steps_command_update_skips_mode_when_unchanged() {
        let mut eval_options = cas_solver::EvalOptions {
            steps_mode: cas_solver::StepsMode::On,
            ..cas_solver::EvalOptions::default()
        };
        let effects = apply_steps_command_update(
            Some(cas_solver::StepsMode::On),
            Some(StepsDisplayMode::Verbose),
            &mut eval_options,
        );
        assert_eq!(
            effects,
            StepsCommandApplyEffects {
                set_steps_mode: None,
                set_display_mode: Some(StepsDisplayMode::Verbose),
            }
        );
    }
}
