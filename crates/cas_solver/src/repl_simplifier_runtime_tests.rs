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

    /// Runtime whose session is `semantics set value complex` — the shim must
    /// carry that axis into the comparator (fichas S5-002/S5-003).
    struct ComplexMockRuntime {
        simplifier: Simplifier,
    }

    impl ReplSimplifierRuntimeContext for ComplexMockRuntime {
        fn simplifier_mut(&mut self) -> &mut Simplifier {
            &mut self.simplifier
        }

        fn session_value_domain(&self) -> cas_engine::ValueDomain {
            cas_engine::ValueDomain::ComplexEnabled
        }
    }

    #[test]
    fn repl_equiv_confirms_eulers_identity_under_complex_session() {
        // SOUNDNESS (auditoría 2026-07-30, ficha S5-003): under
        // `semantics set value complex` the REPL published «False /
        // Residual: undefined» for the TRUE identity ln(−1) = i·π — the
        // comparator ran real-only (collapsing the difference to
        // `undefined`) and the probe read NaN as a counterexample. With the
        // session axis armed and the finiteness guards in place the verdict
        // must never be False; today it lands True (conditional, principal
        // branch), matching the eval route.
        let mut runtime = ComplexMockRuntime {
            simplifier: Simplifier::with_default_rules(),
        };
        let message =
            evaluate_equiv_invocation_message_on_runtime(&mut runtime, "equiv ln(-1), i*pi")
                .expect("message");
        assert!(
            message.contains("True") && !message.contains("False"),
            "Euler's identity must not be refuted under a complex session: {message}"
        );
        assert!(
            !message.contains("undefined"),
            "no undefined residual may be printed: {message}"
        );
    }

    #[test]
    fn repl_equiv_indeterminate_forms_stay_unknown_without_undefined_residual() {
        // The NaN⟹False mechanism was INDEPENDENT of the complex axis
        // (verificador de S5-003): indeterminate forms in the DEFAULT real
        // session published «False / Residual: undefined». An undefined
        // residual is absence of value — the honest verdict is Unknown and
        // no residual line at all.
        for input in ["equiv 0*infinity, 0", "equiv infinity-infinity, 0"] {
            let mut runtime = MockReplSimplifierRuntime::new();
            let message =
                evaluate_equiv_invocation_message_on_runtime(&mut runtime, input).expect("message");
            assert!(
                !message.contains("False"),
                "{input}: an undefined evaluation must not refute: {message}"
            );
            assert!(
                !message.contains("undefined"),
                "{input}: no undefined residual may be printed: {message}"
            );
        }
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
