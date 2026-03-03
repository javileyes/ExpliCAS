#[cfg(test)]
mod tests {
    use crate::{
        apply_context_mode_to_options, evaluate_and_apply_context_command,
        evaluate_context_command_input, format_context_current_message, format_context_set_message,
        parse_context_command_input, ContextCommandInput, ContextCommandResult,
    };

    #[test]
    fn parse_context_command_input_reads_solve() {
        assert_eq!(
            parse_context_command_input("context solve"),
            ContextCommandInput::SetMode(cas_solver::ContextMode::Solve)
        );
    }

    #[test]
    fn parse_context_command_input_unknown_value() {
        assert_eq!(
            parse_context_command_input("context weird"),
            ContextCommandInput::UnknownMode("weird".to_string())
        );
    }

    #[test]
    fn format_context_messages_include_expected_words() {
        let current = format_context_current_message(cas_solver::ContextMode::Auto);
        assert!(current.contains("Current context: auto"));

        let set = format_context_set_message(cas_solver::ContextMode::IntegratePrep);
        assert!(set.contains("integrate-prep"));
    }

    #[test]
    fn evaluate_context_command_input_set_mode_contains_message() {
        let out = evaluate_context_command_input("context solve", cas_solver::ContextMode::Auto);
        match out {
            ContextCommandResult::SetMode { mode, message } => {
                assert_eq!(mode, cas_solver::ContextMode::Solve);
                assert!(message.contains("Context: solve"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn apply_context_mode_to_options_reports_change() {
        let mut eval_options = cas_solver::EvalOptions::default();
        eval_options.shared.context_mode = cas_solver::ContextMode::Auto;

        assert!(!apply_context_mode_to_options(
            cas_solver::ContextMode::Auto,
            &mut eval_options
        ));
        assert!(apply_context_mode_to_options(
            cas_solver::ContextMode::Solve,
            &mut eval_options
        ));
        assert_eq!(
            eval_options.shared.context_mode,
            cas_solver::ContextMode::Solve
        );
    }

    #[test]
    fn evaluate_and_apply_context_command_sets_rebuild_flag_when_changed() {
        let mut eval_options = cas_solver::EvalOptions::default();
        eval_options.shared.context_mode = cas_solver::ContextMode::Auto;

        let out = evaluate_and_apply_context_command("context solve", &mut eval_options);
        assert!(out.rebuild_simplifier);
        assert!(out.message.contains("Context: solve"));
        assert_eq!(
            eval_options.shared.context_mode,
            cas_solver::ContextMode::Solve
        );
    }

    #[test]
    fn evaluate_and_apply_context_command_show_current_does_not_rebuild() {
        let mut eval_options = cas_solver::EvalOptions::default();
        let out = evaluate_and_apply_context_command("context", &mut eval_options);
        assert!(!out.rebuild_simplifier);
        assert!(out.message.contains("Current context"));
    }
}
