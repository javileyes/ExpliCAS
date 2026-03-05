use crate::{
    evaluate_assignment_command_message_with_context, evaluate_clear_bindings_command_lines,
    evaluate_delete_history_command_message,
    evaluate_history_command_lines_from_history_with_context,
    evaluate_let_assignment_command_message_with_context, evaluate_profile_cache_command_lines,
    evaluate_show_command_lines, evaluate_solve_budget_command_message,
    evaluate_vars_command_lines_from_bindings_with_context, AssignmentApplyContext,
    BindingsContext, Engine, HistoryDeleteContext, HistoryOverviewContext, ShowCommandContext,
    Simplifier, SolveBudgetContext,
};

/// Runtime context needed by session-centric REPL command adapters.
pub trait ReplSessionRuntimeContext {
    type State: AssignmentApplyContext
        + BindingsContext
        + HistoryDeleteContext
        + HistoryOverviewContext
        + ShowCommandContext
        + SolveBudgetContext;

    fn state(&self) -> &Self::State;
    fn state_mut(&mut self) -> &mut Self::State;
    fn simplifier_context(&self) -> &cas_ast::Context;
    fn engine_mut(&mut self) -> &mut Engine;
    fn with_engine_and_state<R>(&mut self, f: impl FnOnce(&mut Engine, &mut Self::State) -> R)
        -> R;
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R;
}

/// Render `vars` command output using runtime state/context.
pub fn evaluate_vars_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &C,
) -> String {
    evaluate_vars_command_lines_from_bindings_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}

/// Render `history` command output using runtime state/context.
pub fn evaluate_history_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &C,
) -> String {
    evaluate_history_command_lines_from_history_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}

/// Evaluate `show` command lines against runtime state/engine.
pub fn evaluate_show_command_lines_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<Vec<String>, String> {
    context.with_engine_and_state(|engine, state| evaluate_show_command_lines(state, engine, line))
}

/// Evaluate `clear` command lines against runtime state.
pub fn evaluate_clear_command_lines_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Vec<String> {
    evaluate_clear_bindings_command_lines(context.state_mut(), line)
}

/// Evaluate `del` command message against runtime state.
pub fn evaluate_delete_history_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    evaluate_delete_history_command_message(context.state_mut(), line)
}

/// Evaluate profile `cache` command lines against runtime engine.
pub fn evaluate_profile_cache_command_lines_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Vec<String> {
    evaluate_profile_cache_command_lines(context.engine_mut(), line)
}

/// Evaluate `budget ...` command message against runtime session state.
pub fn evaluate_solve_budget_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    evaluate_solve_budget_command_message(context.state_mut(), line)
}

/// Evaluate `let ...` command against runtime and return user-facing message.
pub fn evaluate_let_assignment_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    input: &str,
) -> Result<String, String> {
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_let_assignment_command_message_with_context(state, simplifier, input)
    })
}

/// Evaluate assignment command against runtime and return user-facing message.
pub fn evaluate_assignment_command_message_on_runtime<C: ReplSessionRuntimeContext>(
    context: &mut C,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_assignment_command_message_with_context(state, simplifier, name, expr_str, lazy)
    })
}
