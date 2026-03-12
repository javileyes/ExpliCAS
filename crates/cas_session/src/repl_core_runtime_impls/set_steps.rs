use crate::repl_core::ReplCore;
use cas_solver::session_api::options::{apply_set_command_plan, apply_steps_command_update};
use cas_solver::session_api::options::{
    ReplSetRuntimeContext, ReplStepsRuntimeContext, SetCommandApplyEffects, SetCommandPlan,
    SetCommandState, SetDisplayMode, StepsCommandApplyEffects, StepsDisplayMode,
};
use cas_solver_core::eval_option_axes::StepsMode;

impl ReplSetRuntimeContext for ReplCore {
    fn set_command_state(&self, display_mode: SetDisplayMode) -> SetCommandState {
        SetCommandState {
            transform: self.simplify_options().enable_transform,
            rationalize: self.simplify_options().rationalize.auto_level,
            heuristic_poly: self.simplify_options().shared.heuristic_poly,
            autoexpand_binomials: self.simplify_options().shared.autoexpand_binomials,
            steps_mode: self.eval_options().steps_mode,
            display_mode,
            max_rewrites: self.simplify_options().budgets.max_total_rewrites,
            debug_mode: self.debug_mode(),
        }
    }

    fn apply_set_command_plan(&mut self, plan: &SetCommandPlan) -> SetCommandApplyEffects {
        let mut debug_mode = self.debug_mode();
        let effects = self.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
            apply_set_command_plan(plan, simplify_options, eval_options, &mut debug_mode)
        });
        self.set_debug_mode(debug_mode);

        if let Some(mode) = effects.set_steps_mode {
            self.simplifier_mut().set_steps_mode(mode);
        }

        effects
    }
}

impl ReplStepsRuntimeContext for ReplCore {
    fn steps_mode_current(&self) -> StepsMode {
        self.eval_options().steps_mode
    }

    fn apply_steps_effects_to_eval_options(
        &mut self,
        set_steps_mode: Option<StepsMode>,
        set_display_mode: Option<StepsDisplayMode>,
    ) -> StepsCommandApplyEffects {
        apply_steps_command_update(set_steps_mode, set_display_mode, self.eval_options_mut())
    }

    fn set_simplifier_steps_mode(&mut self, mode: StepsMode) {
        self.simplifier_mut().set_steps_mode(mode);
    }
}
