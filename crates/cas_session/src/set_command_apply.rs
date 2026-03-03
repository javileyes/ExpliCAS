use crate::{SetCommandApplyEffects, SetCommandPlan};

/// Apply a normalized `set` plan to runtime options.
///
/// Returns UI-adjacent effects (`steps_mode`/`display_mode`) so frontends can
/// update local renderer/runtime state without duplicating option mutation logic.
pub fn apply_set_command_plan(
    plan: &SetCommandPlan,
    simplify_options: &mut cas_solver::SimplifyOptions,
    eval_options: &mut cas_solver::EvalOptions,
    debug_mode: &mut bool,
) -> SetCommandApplyEffects {
    if let Some(enabled) = plan.set_transform {
        simplify_options.enable_transform = enabled;
    }
    if let Some(level) = plan.set_rationalize {
        simplify_options.rationalize.auto_level = level;
    }
    if let Some(mode) = plan.set_heuristic_poly {
        eval_options.shared.heuristic_poly = mode;
        simplify_options.shared.heuristic_poly = mode;
    }
    if let Some(mode) = plan.set_autoexpand_binomials {
        eval_options.shared.autoexpand_binomials = mode;
        simplify_options.shared.autoexpand_binomials = mode;
    }
    if let Some(max_rewrites) = plan.set_max_rewrites {
        simplify_options.budgets.max_total_rewrites = max_rewrites;
    }
    if let Some(value) = plan.set_debug_mode {
        *debug_mode = value;
    }
    if let Some(mode) = plan.set_steps_mode {
        eval_options.steps_mode = mode;
    }

    SetCommandApplyEffects {
        set_steps_mode: plan.set_steps_mode,
        set_display_mode: plan.set_display_mode,
    }
}
