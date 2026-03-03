use crate::{ReplCore, SetCommandApplyEffects, SetCommandPlan, SetCommandState, SetDisplayMode};

/// Build `set` command state snapshot from REPL runtime.
pub fn set_command_state_for_repl_core(
    core: &ReplCore,
    display_mode: SetDisplayMode,
) -> SetCommandState {
    SetCommandState {
        transform: core.simplify_options().enable_transform,
        rationalize: core.simplify_options().rationalize.auto_level,
        heuristic_poly: core.simplify_options().shared.heuristic_poly,
        autoexpand_binomials: core.simplify_options().shared.autoexpand_binomials,
        steps_mode: core.eval_options().steps_mode,
        display_mode,
        max_rewrites: core.simplify_options().budgets.max_total_rewrites,
        debug_mode: core.debug_mode(),
    }
}

/// Apply a `set` mutation plan directly on REPL runtime.
pub fn apply_set_command_plan_on_repl_core(
    core: &mut ReplCore,
    plan: &SetCommandPlan,
) -> SetCommandApplyEffects {
    let mut debug_mode = core.debug_mode();
    let effects = core.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
        crate::apply_set_command_plan(plan, simplify_options, eval_options, &mut debug_mode)
    });
    core.set_debug_mode(debug_mode);

    if let Some(mode) = effects.set_steps_mode {
        core.simplifier_mut().set_steps_mode(mode);
    }

    effects
}
