use crate::profile_cache_command::ProfileCacheStore;
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
    type State;

    fn state(&self) -> &Self::State;
}

/// Runtime context extension for commands that render via AST display context.
pub trait ReplSessionViewRuntimeContext: ReplSessionRuntimeContext {
    fn simplifier_context(&self) -> &cas_ast::Context;
}

/// Runtime context extension for commands that mutate session state directly.
pub trait ReplSessionStateMutRuntimeContext: ReplSessionRuntimeContext {
    fn state_mut(&mut self) -> &mut Self::State;
}

/// Runtime context extension for commands that need mutable session + simplifier.
pub trait ReplSessionSimplifierRuntimeContext: ReplSessionRuntimeContext {
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R;
}

/// Runtime context extension for commands that only need direct `Engine` access.
pub trait ReplEngineRuntimeContext {
    type Engine: ProfileCacheStore;

    fn with_engine_mut<R>(&mut self, f: impl FnOnce(&mut Self::Engine) -> R) -> R;
}

/// Runtime context extension for session REPL commands that still require
/// direct `Engine` access.
pub trait ReplSessionEngineRuntimeContext: ReplSessionRuntimeContext {
    fn with_engine_and_state<R>(&mut self, f: impl FnOnce(&mut Engine, &mut Self::State) -> R)
        -> R;
}

/// Render `vars` command output using runtime state/context.
pub fn evaluate_vars_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: BindingsContext,
{
    evaluate_vars_command_lines_from_bindings_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}

/// Render `history` command output using runtime state/context.
pub fn evaluate_history_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: HistoryOverviewContext,
{
    evaluate_history_command_lines_from_history_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}

/// Evaluate `show` command lines against runtime state/engine.
pub fn evaluate_show_command_lines_on_runtime<C>(
    context: &mut C,
    line: &str,
) -> Result<Vec<String>, String>
where
    C: ReplSessionEngineRuntimeContext,
    C::State: ShowCommandContext,
{
    context.with_engine_and_state(|engine, state| evaluate_show_command_lines(state, engine, line))
}

/// Evaluate `clear` command lines against runtime state.
pub fn evaluate_clear_command_lines_on_runtime<C>(context: &mut C, line: &str) -> Vec<String>
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: BindingsContext,
{
    evaluate_clear_bindings_command_lines(context.state_mut(), line)
}

/// Evaluate `del` command message against runtime state.
pub fn evaluate_delete_history_command_message_on_runtime<C>(context: &mut C, line: &str) -> String
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: HistoryDeleteContext,
{
    evaluate_delete_history_command_message(context.state_mut(), line)
}

/// Evaluate profile `cache` command lines against runtime engine.
pub fn evaluate_profile_cache_command_lines_on_runtime<C: ReplEngineRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Vec<String> {
    context.with_engine_mut(|engine| evaluate_profile_cache_command_lines(engine, line))
}

/// Evaluate `budget ...` command message against runtime session state.
pub fn evaluate_solve_budget_command_message_on_runtime<C>(context: &mut C, line: &str) -> String
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: SolveBudgetContext,
{
    evaluate_solve_budget_command_message(context.state_mut(), line)
}

/// Evaluate `let ...` command against runtime and return user-facing message.
pub fn evaluate_let_assignment_command_message_on_runtime<C>(
    context: &mut C,
    input: &str,
) -> Result<String, String>
where
    C: ReplSessionSimplifierRuntimeContext,
    C::State: AssignmentApplyContext,
{
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_let_assignment_command_message_with_context(state, simplifier, input)
    })
}

/// Evaluate assignment command against runtime and return user-facing message.
pub fn evaluate_assignment_command_message_on_runtime<C>(
    context: &mut C,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String>
where
    C: ReplSessionSimplifierRuntimeContext,
    C::State: AssignmentApplyContext,
{
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_assignment_command_message_with_context(state, simplifier, name, expr_str, lazy)
    })
}
