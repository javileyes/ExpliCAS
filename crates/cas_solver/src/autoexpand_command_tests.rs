#[cfg(test)]
mod tests {
    use crate::session_api::environment::{
        apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
        evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
        format_autoexpand_current_message, format_autoexpand_unknown_mode_message,
        parse_autoexpand_command_input, AutoexpandBudgetView, AutoexpandCommandInput,
        AutoexpandCommandResult, AutoexpandCommandState,
    };

    fn budget() -> AutoexpandBudgetView {
        AutoexpandBudgetView {
            max_pow_exp: 6,
            max_base_terms: 3,
            max_generated_terms: 100,
            max_vars: 2,
        }
    }

    #[test]
    fn parse_autoexpand_command_input_reads_on_mode() {
        assert_eq!(
            parse_autoexpand_command_input("autoexpand on"),
            AutoexpandCommandInput::SetPolicy(crate::ExpandPolicy::Auto)
        );
    }

    #[test]
    fn format_autoexpand_current_message_contains_budget() {
        let text = format_autoexpand_current_message(crate::ExpandPolicy::Off, budget());
        assert!(text.contains("Auto-expand: off"));
        assert!(text.contains("pow<=6"));
    }

    #[test]
    fn format_autoexpand_unknown_mode_message_mentions_usage() {
        let text = format_autoexpand_unknown_mode_message("bad");
        assert!(text.contains("Unknown autoexpand mode: 'bad'"));
        assert!(text.contains("Usage: autoexpand"));
    }

    #[test]
    fn autoexpand_budget_view_from_options_reads_budget() {
        let mut eval_options = crate::EvalOptions::default();
        eval_options.shared.expand_budget.max_pow_exp = 9;
        let budget = autoexpand_budget_view_from_options(&eval_options);
        assert_eq!(budget.max_pow_exp, 9);
    }

    #[test]
    fn evaluate_autoexpand_command_input_set_policy_returns_message() {
        let state = AutoexpandCommandState {
            policy: crate::ExpandPolicy::Off,
            budget: budget(),
        };
        let out = evaluate_autoexpand_command_input("autoexpand on", state);
        match out {
            AutoexpandCommandResult::SetPolicy { policy, message } => {
                assert_eq!(policy, crate::ExpandPolicy::Auto);
                assert!(message.contains("Auto-expand: on"));
            }
            other => panic!("unexpected output: {other:?}"),
        }
    }

    #[test]
    fn apply_autoexpand_policy_to_options_reports_change() {
        let mut eval_options = crate::EvalOptions::default();
        eval_options.shared.expand_policy = crate::ExpandPolicy::Off;

        assert!(!apply_autoexpand_policy_to_options(
            crate::ExpandPolicy::Off,
            &mut eval_options
        ));
        assert!(apply_autoexpand_policy_to_options(
            crate::ExpandPolicy::Auto,
            &mut eval_options
        ));
        assert_eq!(eval_options.shared.expand_policy, crate::ExpandPolicy::Auto);
    }

    #[test]
    fn evaluate_and_apply_autoexpand_command_sets_rebuild_flag_when_changed() {
        let mut eval_options = crate::EvalOptions::default();
        eval_options.shared.expand_policy = crate::ExpandPolicy::Off;

        let out = evaluate_and_apply_autoexpand_command("autoexpand on", &mut eval_options);
        assert!(out.rebuild_simplifier);
        assert!(out.message.contains("Auto-expand: on"));
        assert_eq!(eval_options.shared.expand_policy, crate::ExpandPolicy::Auto);
    }

    #[test]
    fn evaluate_and_apply_autoexpand_command_show_current_does_not_rebuild() {
        let mut eval_options = crate::EvalOptions::default();
        let out = evaluate_and_apply_autoexpand_command("autoexpand", &mut eval_options);
        assert!(!out.rebuild_simplifier);
        assert!(out.message.contains("Auto-expand:"));
    }
}
