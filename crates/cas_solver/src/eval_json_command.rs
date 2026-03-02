use cas_api_models::{EvalJsonOutput, StepJson};

/// CLI-facing config for `eval-json` command execution.
#[derive(Debug, Clone, Copy)]
pub struct EvalJsonCommandConfig<'a> {
    pub expr: &'a str,
    pub auto_store: bool,
    pub max_chars: usize,
    pub steps_mode: &'a str,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub context_mode: &'a str,
    pub branch_mode: &'a str,
    pub expand_policy: &'a str,
    pub complex_mode: &'a str,
    pub const_fold: &'a str,
    pub value_domain: &'a str,
    pub complex_branch: &'a str,
    pub inv_trig: &'a str,
    pub assume_scope: &'a str,
}

/// Evaluate `eval-json` command with an existing engine/session.
pub fn evaluate_eval_json_command_with_session<S, F>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
    F: Fn(&crate::EvalOutput, &crate::Engine, &str) -> Vec<StepJson>,
{
    crate::json::evaluate_eval_json_with_session(
        engine,
        session,
        crate::json::EvalJsonSessionRunConfig {
            expr: config.expr,
            auto_store: config.auto_store,
            max_chars: config.max_chars,
            steps_mode: config.steps_mode,
            budget_preset: config.budget_preset,
            strict: config.strict,
            domain: config.domain,
            context_mode: config.context_mode,
            branch_mode: config.branch_mode,
            expand_policy: config.expand_policy,
            complex_mode: config.complex_mode,
            const_fold: config.const_fold,
            value_domain: config.value_domain,
            complex_branch: config.complex_branch,
            inv_trig: config.inv_trig,
            assume_scope: config.assume_scope,
        },
        collect_steps,
    )
}

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_json_command_with_session, EvalJsonCommandConfig};

    #[test]
    fn evaluate_eval_json_command_with_session_runs() {
        let mut engine = crate::Engine::new();
        let mut state = cas_session::SessionState::new();

        let out = evaluate_eval_json_command_with_session(
            &mut engine,
            &mut state,
            EvalJsonCommandConfig {
                expr: "x + x",
                auto_store: false,
                max_chars: 2000,
                steps_mode: "off",
                budget_preset: "standard",
                strict: false,
                domain: "generic",
                context_mode: "auto",
                branch_mode: "strict",
                expand_policy: "off",
                complex_mode: "auto",
                const_fold: "off",
                value_domain: "real",
                complex_branch: "principal",
                inv_trig: "strict",
                assume_scope: "real",
            },
            |_output, _engine, _steps_mode| Vec::new(),
        )
        .expect("eval-json command");

        assert!(out.ok);
    }
}
