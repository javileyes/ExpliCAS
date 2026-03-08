use super::super::super::display_policy::CliSubstepsRenderState;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) struct StepLoopState {
    current_root: ExprId,
    step_count: usize,
    cli_substeps_state: CliSubstepsRenderState,
}

impl StepLoopState {
    pub(super) fn new(expr: ExprId) -> Self {
        Self {
            current_root: expr,
            step_count: 0,
            cli_substeps_state: CliSubstepsRenderState::default(),
        }
    }

    pub(super) fn current_root(&self) -> ExprId {
        self.current_root
    }

    pub(super) fn next_step_number(&mut self) -> usize {
        self.step_count += 1;
        self.step_count
    }

    pub(super) fn cli_substeps_state_mut(&mut self) -> &mut CliSubstepsRenderState {
        &mut self.cli_substeps_state
    }

    pub(super) fn advance(&mut self, ctx: &mut Context, step: &Step) {
        self.current_root = if let Some(global_after) = step.global_after {
            global_after
        } else {
            cas_solver::reconstruct_global_expr(ctx, self.current_root, step.path(), step.after)
        };
    }
}
