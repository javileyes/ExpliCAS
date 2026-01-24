//! Core-only tests for ReplCore contract validation.
//! These tests verify that core methods return correct messages
//! without any I/O, purely through the ReplReply/CoreResult types.

#[cfg(test)]
mod tests {
    use crate::repl::{CoreResult, Repl, ReplMsg, Verbosity};

    /// Test: help command returns non-empty Output
    #[test]
    fn test_core_help_returns_output() {
        let repl = Repl::new();
        let reply = repl.handle_help_core("help");

        // Should have at least one message
        assert!(!reply.is_empty(), "help should return non-empty reply");

        // Should contain Output (help text)
        let has_output = reply.iter().any(|m| matches!(m, ReplMsg::Output(_)));
        assert!(has_output, "help should return Output message");
    }

    /// Test: eval of "2+2" returns Output containing "4"
    #[test]
    fn test_core_eval_simple_arithmetic() {
        let mut repl = Repl::new();
        let reply = repl.handle_eval_core("2+2");

        // Should have at least one message
        assert!(!reply.is_empty(), "eval should return non-empty reply");

        // Should contain output with "4"
        let has_result = reply.iter().any(|m| {
            if let ReplMsg::Output(s) = m {
                s.contains("4")
            } else {
                false
            }
        });
        assert!(has_result, "eval 2+2 should return output containing '4'");
    }

    /// Test: set verbose command returns UiDelta with verbosity change
    #[test]
    fn test_core_set_verbose_returns_ui_delta() {
        let mut repl = Repl::new();
        let result: CoreResult = repl.handle_set_command_core("set steps verbose");

        // Should have verbosity delta
        assert!(
            result.ui_delta.verbosity.is_some(),
            "set steps verbose should return verbosity delta"
        );
        assert_eq!(
            result.ui_delta.verbosity,
            Some(Verbosity::Verbose),
            "delta should be Verbose"
        );

        // Should have reply message
        assert!(
            !result.reply.is_empty(),
            "set should return non-empty reply"
        );
    }

    /// Test: set steps off returns UiDelta with None verbosity
    #[test]
    fn test_core_set_steps_off_returns_ui_delta() {
        let mut repl = Repl::new();
        let result: CoreResult = repl.handle_set_command_core("set steps off");

        assert!(
            result.ui_delta.verbosity.is_some(),
            "set steps off should return verbosity delta"
        );
        assert_eq!(
            result.ui_delta.verbosity,
            Some(Verbosity::None),
            "delta should be None"
        );
    }
}
