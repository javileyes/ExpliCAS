/// Parsed input for the `context` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommandInput {
    ShowCurrent,
    SetMode(crate::ContextMode),
    UnknownMode(String),
}

/// Normalized result for `context` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommandResult {
    ShowCurrent {
        message: String,
    },
    SetMode {
        mode: crate::ContextMode,
        message: String,
    },
    Invalid {
        message: String,
    },
}

/// Result from evaluating + applying a `context` command to runtime options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCommandApplyOutput {
    pub message: String,
    pub rebuild_simplifier: bool,
}

/// Parse raw `context ...` command input.
pub fn parse_context_command_input(line: &str) -> ContextCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => ContextCommandInput::ShowCurrent,
        Some(&"auto") => ContextCommandInput::SetMode(crate::ContextMode::Auto),
        Some(&"standard") => ContextCommandInput::SetMode(crate::ContextMode::Standard),
        Some(&"solve") => ContextCommandInput::SetMode(crate::ContextMode::Solve),
        Some(&"integrate") => ContextCommandInput::SetMode(crate::ContextMode::IntegratePrep),
        Some(other) => ContextCommandInput::UnknownMode((*other).to_string()),
    }
}

/// Evaluate a `context` command into mode changes + message.
pub fn evaluate_context_command_input(
    line: &str,
    current_mode: crate::ContextMode,
) -> ContextCommandResult {
    match parse_context_command_input(line) {
        ContextCommandInput::ShowCurrent => ContextCommandResult::ShowCurrent {
            message: format_context_current_message(current_mode),
        },
        ContextCommandInput::SetMode(mode) => ContextCommandResult::SetMode {
            mode,
            message: format_context_set_message(mode),
        },
        ContextCommandInput::UnknownMode(mode) => ContextCommandResult::Invalid {
            message: format_context_unknown_message(&mode),
        },
    }
}

/// Apply context mode into eval options, returning whether mode changed.
pub fn apply_context_mode_to_options(
    mode: crate::ContextMode,
    eval_options: &mut crate::EvalOptions,
) -> bool {
    if eval_options.shared.context_mode == mode {
        return false;
    }
    eval_options.shared.context_mode = mode;
    true
}

/// Evaluate and apply a `context` command directly to runtime options.
pub fn evaluate_and_apply_context_command(
    line: &str,
    eval_options: &mut crate::EvalOptions,
) -> ContextCommandApplyOutput {
    match evaluate_context_command_input(line, eval_options.shared.context_mode) {
        ContextCommandResult::ShowCurrent { message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
        ContextCommandResult::SetMode { mode, message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: apply_context_mode_to_options(mode, eval_options),
        },
        ContextCommandResult::Invalid { message } => ContextCommandApplyOutput {
            message,
            rebuild_simplifier: false,
        },
    }
}

/// Format current context status line.
pub fn format_context_current_message(mode: crate::ContextMode) -> String {
    let context = match mode {
        crate::ContextMode::Auto => "auto",
        crate::ContextMode::Standard => "standard",
        crate::ContextMode::Solve => "solve",
        crate::ContextMode::IntegratePrep => "integrate",
    };
    format!(
        "Current context: {}\n  (use 'context auto|standard|solve|integrate' to change)",
        context
    )
}

/// Format confirmation message after setting context.
pub fn format_context_set_message(mode: crate::ContextMode) -> String {
    match mode {
        crate::ContextMode::Auto => "Context: auto (infers from expression)".to_string(),
        crate::ContextMode::Standard => {
            "Context: standard (safe simplification only)".to_string()
        }
        crate::ContextMode::Solve => "Context: solve (preserves solver-friendly forms)".to_string(),
        crate::ContextMode::IntegratePrep => "Context: integrate-prep\n  ⚠️ Enables transforms for integration (telescoping, product→sum)".to_string(),
    }
}

/// Format unknown-context error message.
pub fn format_context_unknown_message(mode: &str) -> String {
    format!(
        "Unknown context: '{}'\nUsage: context [auto | standard | solve | integrate]",
        mode
    )
}

#[cfg(test)]
mod tests {
    use super::{
        apply_context_mode_to_options, evaluate_and_apply_context_command,
        evaluate_context_command_input, format_context_current_message, format_context_set_message,
        parse_context_command_input, ContextCommandInput, ContextCommandResult,
    };

    #[test]
    fn parse_context_command_input_reads_solve() {
        assert_eq!(
            parse_context_command_input("context solve"),
            ContextCommandInput::SetMode(crate::ContextMode::Solve)
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
        let current = format_context_current_message(crate::ContextMode::Auto);
        assert!(current.contains("Current context: auto"));

        let set = format_context_set_message(crate::ContextMode::IntegratePrep);
        assert!(set.contains("integrate-prep"));
    }

    #[test]
    fn evaluate_context_command_input_set_mode_contains_message() {
        let out = evaluate_context_command_input("context solve", crate::ContextMode::Auto);
        match out {
            ContextCommandResult::SetMode { mode, message } => {
                assert_eq!(mode, crate::ContextMode::Solve);
                assert!(message.contains("Context: solve"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn apply_context_mode_to_options_reports_change() {
        let mut eval_options = crate::EvalOptions::default();
        eval_options.shared.context_mode = crate::ContextMode::Auto;

        assert!(!apply_context_mode_to_options(
            crate::ContextMode::Auto,
            &mut eval_options
        ));
        assert!(apply_context_mode_to_options(
            crate::ContextMode::Solve,
            &mut eval_options
        ));
        assert_eq!(eval_options.shared.context_mode, crate::ContextMode::Solve);
    }

    #[test]
    fn evaluate_and_apply_context_command_sets_rebuild_flag_when_changed() {
        let mut eval_options = crate::EvalOptions::default();
        eval_options.shared.context_mode = crate::ContextMode::Auto;

        let out = evaluate_and_apply_context_command("context solve", &mut eval_options);
        assert!(out.rebuild_simplifier);
        assert!(out.message.contains("Context: solve"));
        assert_eq!(eval_options.shared.context_mode, crate::ContextMode::Solve);
    }

    #[test]
    fn evaluate_and_apply_context_command_show_current_does_not_rebuild() {
        let mut eval_options = crate::EvalOptions::default();
        let out = evaluate_and_apply_context_command("context", &mut eval_options);
        assert!(!out.rebuild_simplifier);
        assert!(out.message.contains("Current context"));
    }
}
