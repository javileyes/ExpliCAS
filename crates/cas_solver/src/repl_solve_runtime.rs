use crate::{Engine, EvalOptions, EvalSession, EvalStore, SetDisplayMode, Simplifier};

fn map_solve_display_mode(mode: SetDisplayMode) -> crate::SolveDisplayMode {
    match mode {
        SetDisplayMode::None => crate::SolveDisplayMode::None,
        SetDisplayMode::Succinct => crate::SolveDisplayMode::Succinct,
        SetDisplayMode::Normal => crate::SolveDisplayMode::Normal,
        SetDisplayMode::Verbose => crate::SolveDisplayMode::Verbose,
    }
}

fn map_full_simplify_display_mode(mode: SetDisplayMode) -> crate::FullSimplifyDisplayMode {
    match mode {
        SetDisplayMode::None => crate::FullSimplifyDisplayMode::None,
        SetDisplayMode::Succinct => crate::FullSimplifyDisplayMode::Succinct,
        SetDisplayMode::Normal => crate::FullSimplifyDisplayMode::Normal,
        SetDisplayMode::Verbose => crate::FullSimplifyDisplayMode::Verbose,
    }
}

/// Runtime context needed by solve/full-simplify REPL command adapters.
pub trait ReplSolveRuntimeContext {
    type State: EvalSession<Options = EvalOptions, Diagnostics = crate::Diagnostics>;

    fn debug_mode(&self) -> bool;
    fn with_engine_and_state<R>(&mut self, f: impl FnOnce(&mut Engine, &mut Self::State) -> R)
        -> R;
    fn with_state_and_simplifier_mut<R>(
        &mut self,
        f: impl FnOnce(&mut Self::State, &mut Simplifier) -> R,
    ) -> R;
}

/// Evaluate `solve ...` invocation against runtime engine/session state.
pub fn evaluate_solve_command_message_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String>
where
    C::State: EvalSession<Options = EvalOptions, Diagnostics = crate::Diagnostics>,
    <C::State as EvalSession>::Store: EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    let debug_mode = context.debug_mode();
    context.with_state_and_simplifier_mut(|state, simplifier| {
        let options = state.options().clone();
        crate::evaluate_solve_command_message_with_session(
            simplifier,
            state,
            line,
            &options,
            map_solve_display_mode(display_mode),
            debug_mode,
        )
    })
}

/// Evaluate `simplify ...` invocation against runtime simplifier/session state.
pub fn evaluate_full_simplify_command_lines_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<Vec<String>, String> {
    context.with_state_and_simplifier_mut(|state, simplifier| {
        let simplify_options = state.options().to_simplify_options();
        crate::evaluate_full_simplify_command_lines_with_resolver(
            simplifier,
            line,
            map_full_simplify_display_mode(display_mode),
            simplify_options,
            |ctx, parsed_expr| {
                state
                    .resolve_all_with_diagnostics(ctx, parsed_expr)
                    .map(|(resolved_expr, _diag, _cache_hits)| resolved_expr)
                    .map_err(|error| error.to_string())
            },
        )
    })
}
