#[cfg(test)]
mod tests {
    use crate::{
        evaluate_det_command_message_on_runtime, evaluate_equiv_invocation_message_on_runtime,
        evaluate_explain_invocation_message_on_runtime,
        evaluate_profile_command_message_on_runtime, evaluate_rationalize_command_lines_on_runtime,
        evaluate_substitute_invocation_user_message_on_runtime,
        evaluate_telescope_invocation_message_on_runtime, ReplSimplifierRuntimeContext,
        SetDisplayMode, Simplifier,
    };

    struct MockReplSimplifierRuntime {
        simplifier: Simplifier,
    }

    impl MockReplSimplifierRuntime {
        fn new() -> Self {
            Self {
                simplifier: Simplifier::with_default_rules(),
            }
        }
    }

    impl ReplSimplifierRuntimeContext for MockReplSimplifierRuntime {
        fn simplifier_mut(&mut self) -> &mut Simplifier {
            &mut self.simplifier
        }
    }

    #[test]
    fn evaluate_det_command_message_on_runtime_works() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let message = evaluate_det_command_message_on_runtime(
            &mut runtime,
            "det [[1,2],[3,4]]",
            SetDisplayMode::Normal,
        )
        .expect("message");
        assert!(!message.is_empty());
    }

    #[test]
    fn evaluate_telescope_invocation_message_on_runtime_invalid_input_errors() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let error = evaluate_telescope_invocation_message_on_runtime(&mut runtime, "telescope")
            .expect_err("error");
        assert!(!error.is_empty());
    }

    #[test]
    fn evaluate_explain_invocation_message_on_runtime_invalid_input_errors() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let error = evaluate_explain_invocation_message_on_runtime(&mut runtime, "explain")
            .expect_err("error");
        assert!(!error.is_empty());
    }

    #[test]
    fn evaluate_equiv_invocation_message_on_runtime_works() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let message = evaluate_equiv_invocation_message_on_runtime(&mut runtime, "equiv x+1,1+x")
            .expect("message");
        assert!(message.contains("True"));
    }

    #[test]
    fn evaluate_substitute_invocation_user_message_on_runtime_works() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let message = evaluate_substitute_invocation_user_message_on_runtime(
            &mut runtime,
            "subst x+1, x, 2",
            SetDisplayMode::Normal,
        )
        .expect("message");
        assert!(!message.is_empty());
    }

    #[test]
    fn evaluate_rationalize_command_lines_on_runtime_works() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let lines = evaluate_rationalize_command_lines_on_runtime(
            &mut runtime,
            "rationalize 1/(1+sqrt(2))",
        )
        .expect("lines");
        assert!(!lines.is_empty());
    }

    #[test]
    fn evaluate_profile_command_message_on_runtime_works() {
        let mut runtime = MockReplSimplifierRuntime::new();
        let message = evaluate_profile_command_message_on_runtime(&mut runtime, "profile");
        assert!(!message.trim().is_empty());
    }
}
