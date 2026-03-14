#[cfg(test)]
mod tests {
    use std::{cell::Cell, thread, time::Duration};

    use cas_api_models::{
        EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
        EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
        EvalValueDomain,
    };

    use crate::eval::evaluate_eval_text_command_with_session;
    use crate::eval::EvalCommandConfig;
    use crate::state_core::SessionState;
    use tempfile::tempdir;

    #[test]
    fn evaluate_eval_with_session_runs() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        let out = cas_solver::session_api::eval::evaluate_eval_with_session(
            &mut engine,
            &mut session,
            EvalCommandConfig {
                expr: "x + x",
                auto_store: false,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        )
        .expect("eval");

        assert!(out.ok);
        assert!(out.result.contains("2 * x"));
    }

    #[test]
    fn evaluate_eval_with_session_keeps_steps_when_steps_on() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        let step_count = Cell::new(0usize);

        let out = cas_solver::session_api::eval::evaluate_eval_with_session(
            &mut engine,
            &mut session,
            EvalCommandConfig {
                expr: "x + 0",
                auto_store: false,
                max_chars: 2000,
                steps_mode: EvalStepsMode::On,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |steps, _events, _context, _steps_mode| {
                step_count.set(steps.len());
                Vec::new()
            },
        )
        .expect("eval");

        assert!(out.ok);
        assert!(
            step_count.get() > 0,
            "steps-on eval should still expose raw steps"
        );
    }

    #[test]
    fn evaluate_eval_with_session_applies_const_fold_mode_from_config() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();

        let out = cas_solver::session_api::eval::evaluate_eval_with_session(
            &mut engine,
            &mut session,
            EvalCommandConfig {
                expr: "sqrt(-1)",
                auto_store: false,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::On,
                const_fold: EvalConstFoldMode::Safe,
                value_domain: EvalValueDomain::Complex,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        )
        .expect("eval");

        assert!(out.ok);
        assert_eq!(out.result, "i");
    }

    #[test]
    fn persisted_no_store_eval_does_not_resave_snapshot() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("session.bin");

        let first = evaluate_eval_text_command_with_session(Some(&path), "generic", "x + 1", true);
        assert!(first.0.is_ok());
        assert!(path.exists());

        let before = std::fs::metadata(&path)
            .expect("metadata before")
            .modified()
            .expect("modified before");

        thread::sleep(Duration::from_millis(25));

        let second =
            evaluate_eval_text_command_with_session(Some(&path), "generic", "x + 1", false);
        assert!(second.0.is_ok());

        let after = std::fs::metadata(&path)
            .expect("metadata after")
            .modified()
            .expect("modified after");

        assert_eq!(
            before, after,
            "persisted no-store eval should skip snapshot rewrite"
        );
    }

    #[test]
    fn persisted_cache_hit_eval_does_not_resave_snapshot() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("session.bin");

        let first = evaluate_eval_text_command_with_session(Some(&path), "generic", "x + 1", true);
        assert!(first.0.is_ok());
        assert!(path.exists());

        let before = std::fs::metadata(&path)
            .expect("metadata before")
            .modified()
            .expect("modified before");

        thread::sleep(Duration::from_millis(25));

        let second = evaluate_eval_text_command_with_session(Some(&path), "generic", "#1", false);
        assert!(second.0.is_ok());

        let after = std::fs::metadata(&path)
            .expect("metadata after")
            .modified()
            .expect("modified after");

        assert_eq!(
            before, after,
            "persisted cache-hit eval should skip snapshot rewrite"
        );
    }

    #[test]
    fn persisted_cache_hit_eval_preserves_result() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("session.bin");

        let first = crate::eval::evaluate_eval_command_with_session(
            Some(&path),
            EvalCommandConfig {
                expr: "x/x",
                auto_store: true,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );
        assert!(first.0.is_ok(), "seed eval should succeed");

        let second = crate::eval::evaluate_eval_command_with_session(
            Some(&path),
            EvalCommandConfig {
                expr: "#1",
                auto_store: false,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let output = second.0.expect("cache-hit eval should succeed");
        assert_eq!(output.result, "1");
    }

    #[test]
    fn persisted_cache_hit_eval_preserves_required_conditions_for_direct_ref() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("session.bin");

        let first = crate::eval::evaluate_eval_command_with_session(
            Some(&path),
            EvalCommandConfig {
                expr: "sqrt(x)",
                auto_store: true,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );
        assert!(first.0.is_ok(), "seed eval should succeed");

        let second = crate::eval::evaluate_eval_command_with_session(
            Some(&path),
            EvalCommandConfig {
                expr: "#1",
                auto_store: false,
                max_chars: 2000,
                steps_mode: EvalStepsMode::Off,
                budget_preset: EvalBudgetPreset::Standard,
                strict: false,
                domain: EvalDomainMode::Generic,
                context_mode: EvalContextMode::Auto,
                branch_mode: EvalBranchMode::Strict,
                expand_policy: EvalExpandPolicy::Off,
                complex_mode: EvalComplexMode::Auto,
                const_fold: EvalConstFoldMode::Off,
                value_domain: EvalValueDomain::Real,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                inv_trig: EvalInvTrigPolicy::Strict,
                assume_scope: EvalAssumeScope::Real,
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );

        let output = second.0.expect("cache-hit eval should succeed");
        assert!(
            output.result == "sqrt(x)" || output.result == "x^(1/2)",
            "expected direct cached ref to preserve sqrt semantics, got {:?}",
            output.result
        );
        assert!(
            output
                .required_display
                .iter()
                .any(|cond| cond.contains("x") && (cond.contains(">= 0") || cond.contains("≥ 0"))),
            "expected inherited required condition for sqrt(x), got {:?}",
            output.required_display
        );
    }
}
