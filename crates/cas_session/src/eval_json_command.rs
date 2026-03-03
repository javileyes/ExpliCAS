//! Session-backed eval-json command orchestration.

use std::path::Path;

/// Session-backed config for `eval-json` command orchestration.
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

impl<'a> EvalJsonCommandConfig<'a> {
    fn to_solver(self) -> cas_solver::json::EvalJsonSessionRunConfig<'a> {
        cas_solver::json::EvalJsonSessionRunConfig {
            expr: self.expr,
            auto_store: self.auto_store,
            max_chars: self.max_chars,
            steps_mode: self.steps_mode,
            budget_preset: self.budget_preset,
            strict: self.strict,
            domain: self.domain,
            context_mode: self.context_mode,
            branch_mode: self.branch_mode,
            expand_policy: self.expand_policy,
            complex_mode: self.complex_mode,
            const_fold: self.const_fold,
            value_domain: self.value_domain,
            complex_branch: self.complex_branch,
            inv_trig: self.inv_trig,
            assume_scope: self.assume_scope,
        }
    }
}

/// Evaluate `eval-json` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_json_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> (
    Result<cas_solver::json::EvalJsonOutput, String>,
    Option<String>,
    Option<String>,
)
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_solver::json::StepJson>,
{
    let solver_config = config.to_solver();
    crate::run_with_domain_session(session_path, config.domain, |engine, state| {
        cas_solver::json::evaluate_eval_json_with_session(
            engine,
            state,
            solver_config,
            |steps, ctx, mode| collect_steps(steps, ctx, mode),
        )
    })
}

/// Evaluate `eval-json` and always return a pretty JSON string.
///
/// Successful runs return canonical JSON payload. Errors are normalized into
/// canonical JSON error output.
pub fn evaluate_eval_json_command_pretty_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> String
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_solver::json::StepJson>,
{
    let input = config.expr.to_string();
    let (output, _, _) =
        evaluate_eval_json_command_with_session(session_path, config, collect_steps);
    match output {
        Ok(payload) => payload.to_json_pretty(),
        Err(error) => {
            cas_solver::json::build_eval_json_error_output(&error, &input).to_json_pretty()
        }
    }
}
