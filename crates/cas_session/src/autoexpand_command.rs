/// View-only budget values used for displaying auto-expand settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoexpandBudgetView {
    pub max_pow_exp: u32,
    pub max_base_terms: u32,
    pub max_generated_terms: u32,
    pub max_vars: u32,
}

/// Runtime state needed to evaluate an `autoexpand` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoexpandCommandState {
    pub policy: cas_engine::ExpandPolicy,
    pub budget: AutoexpandBudgetView,
}

/// Build an autoexpand budget view from eval options.
pub fn autoexpand_budget_view_from_options(
    eval_options: &cas_engine::EvalOptions,
) -> AutoexpandBudgetView {
    let budget = &eval_options.shared.expand_budget;
    AutoexpandBudgetView {
        max_pow_exp: budget.max_pow_exp,
        max_base_terms: budget.max_base_terms,
        max_generated_terms: budget.max_generated_terms,
        max_vars: budget.max_vars,
    }
}

/// Parsed input for the `autoexpand` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoexpandCommandInput {
    ShowCurrent,
    SetPolicy(cas_engine::ExpandPolicy),
    UnknownMode(String),
}

/// Normalized result for `autoexpand` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoexpandCommandResult {
    ShowCurrent {
        message: String,
    },
    SetPolicy {
        policy: cas_engine::ExpandPolicy,
        message: String,
    },
    Invalid {
        message: String,
    },
}

/// Result from evaluating + applying an `autoexpand` command to runtime options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoexpandCommandApplyOutput {
    pub message: String,
    pub rebuild_simplifier: bool,
}

/// Parse raw `autoexpand ...` command input.
pub fn parse_autoexpand_command_input(line: &str) -> AutoexpandCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => AutoexpandCommandInput::ShowCurrent,
        Some(&"on") => AutoexpandCommandInput::SetPolicy(cas_engine::ExpandPolicy::Auto),
        Some(&"off") => AutoexpandCommandInput::SetPolicy(cas_engine::ExpandPolicy::Off),
        Some(other) => AutoexpandCommandInput::UnknownMode((*other).to_string()),
    }
}

/// Evaluate an `autoexpand` command into policy changes + message.
pub fn evaluate_autoexpand_command_input(
    line: &str,
    state: AutoexpandCommandState,
) -> AutoexpandCommandResult {
    match parse_autoexpand_command_input(line) {
        AutoexpandCommandInput::ShowCurrent => AutoexpandCommandResult::ShowCurrent {
            message: format_autoexpand_current_message(state.policy, state.budget),
        },
        AutoexpandCommandInput::SetPolicy(policy) => AutoexpandCommandResult::SetPolicy {
            policy,
            message: format_autoexpand_set_message(policy, state.budget),
        },
        AutoexpandCommandInput::UnknownMode(mode) => AutoexpandCommandResult::Invalid {
            message: format_autoexpand_unknown_mode_message(&mode),
        },
    }
}

/// Apply autoexpand policy into eval options, returning whether policy changed.
pub fn apply_autoexpand_policy_to_options(
    policy: cas_engine::ExpandPolicy,
    eval_options: &mut cas_engine::EvalOptions,
) -> bool {
    if eval_options.shared.expand_policy == policy {
        return false;
    }
    eval_options.shared.expand_policy = policy;
    true
}

/// Evaluate and apply an `autoexpand` command directly to runtime options.
pub fn evaluate_and_apply_autoexpand_command(
    line: &str,
    eval_options: &mut cas_engine::EvalOptions,
) -> AutoexpandCommandApplyOutput {
    let state = AutoexpandCommandState {
        policy: eval_options.shared.expand_policy,
        budget: autoexpand_budget_view_from_options(eval_options),
    };
    match evaluate_autoexpand_command_input(line, state) {
        AutoexpandCommandResult::ShowCurrent { message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
        AutoexpandCommandResult::SetPolicy { policy, message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: apply_autoexpand_policy_to_options(policy, eval_options),
        },
        AutoexpandCommandResult::Invalid { message } => AutoexpandCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
    }
}

/// Format status output for `autoexpand`.
pub fn format_autoexpand_current_message(
    policy: cas_engine::ExpandPolicy,
    budget: AutoexpandBudgetView,
) -> String {
    let policy_str = match policy {
        cas_engine::ExpandPolicy::Off => "off",
        cas_engine::ExpandPolicy::Auto => "on",
    };
    format!(
        "Auto-expand: {}\n\
           Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
           (use 'autoexpand on|off' to change)",
        policy_str,
        budget.max_pow_exp,
        budget.max_base_terms,
        budget.max_generated_terms,
        budget.max_vars
    )
}

/// Format feedback after applying an auto-expand policy.
pub fn format_autoexpand_set_message(
    policy: cas_engine::ExpandPolicy,
    budget: AutoexpandBudgetView,
) -> String {
    match policy {
        cas_engine::ExpandPolicy::Auto => format!(
            "Auto-expand: on\n\
               Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
               ⚠️ Expands small (sum)^n patterns automatically.",
            budget.max_pow_exp, budget.max_base_terms, budget.max_generated_terms, budget.max_vars
        ),
        cas_engine::ExpandPolicy::Off => {
            "Auto-expand: off\n  Polynomial expansions require explicit expand().".to_string()
        }
    }
}

/// Format unknown-mode error for `autoexpand`.
pub fn format_autoexpand_unknown_mode_message(mode: &str) -> String {
    format!(
        "Unknown autoexpand mode: '{}'\n\
             Usage: autoexpand [on | off]\n\
               on  - Auto-expand cheap polynomial powers\n\
               off - Only expand when explicitly requested (default)",
        mode
    )
}

#[cfg(test)]
mod tests {
    use super::{
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
            AutoexpandCommandInput::SetPolicy(cas_engine::ExpandPolicy::Auto)
        );
    }

    #[test]
    fn format_autoexpand_current_message_contains_budget() {
        let text = format_autoexpand_current_message(cas_engine::ExpandPolicy::Off, budget());
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
        let mut eval_options = cas_engine::EvalOptions::default();
        eval_options.shared.expand_budget.max_pow_exp = 9;
        let budget = autoexpand_budget_view_from_options(&eval_options);
        assert_eq!(budget.max_pow_exp, 9);
    }

    #[test]
    fn evaluate_autoexpand_command_input_set_policy_returns_message() {
        let state = AutoexpandCommandState {
            policy: cas_engine::ExpandPolicy::Off,
            budget: budget(),
        };
        let out = evaluate_autoexpand_command_input("autoexpand on", state);
        match out {
            AutoexpandCommandResult::SetPolicy { policy, message } => {
                assert_eq!(policy, cas_engine::ExpandPolicy::Auto);
                assert!(message.contains("Auto-expand: on"));
            }
            other => panic!("unexpected output: {other:?}"),
        }
    }

    #[test]
    fn apply_autoexpand_policy_to_options_reports_change() {
        let mut eval_options = cas_engine::EvalOptions::default();
        eval_options.shared.expand_policy = cas_engine::ExpandPolicy::Off;

        assert!(!apply_autoexpand_policy_to_options(
            cas_engine::ExpandPolicy::Off,
            &mut eval_options
        ));
        assert!(apply_autoexpand_policy_to_options(
            cas_engine::ExpandPolicy::Auto,
            &mut eval_options
        ));
        assert_eq!(
            eval_options.shared.expand_policy,
            cas_engine::ExpandPolicy::Auto
        );
    }

    #[test]
    fn evaluate_and_apply_autoexpand_command_sets_rebuild_flag_when_changed() {
        let mut eval_options = cas_engine::EvalOptions::default();
        eval_options.shared.expand_policy = cas_engine::ExpandPolicy::Off;

        let out = evaluate_and_apply_autoexpand_command("autoexpand on", &mut eval_options);
        assert!(out.rebuild_simplifier);
        assert!(out.message.contains("Auto-expand: on"));
        assert_eq!(
            eval_options.shared.expand_policy,
            cas_engine::ExpandPolicy::Auto
        );
    }

    #[test]
    fn evaluate_and_apply_autoexpand_command_show_current_does_not_rebuild() {
        let mut eval_options = cas_engine::EvalOptions::default();
        let out = evaluate_and_apply_autoexpand_command("autoexpand", &mut eval_options);
        assert!(!out.rebuild_simplifier);
        assert!(out.message.contains("Auto-expand:"));
    }
}
