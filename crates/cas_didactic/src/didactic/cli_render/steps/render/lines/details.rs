mod assumptions;
mod engine_substeps;

use crate::cas_solver::Step;

pub(super) fn render_engine_substeps_lines(step: &Step) -> Vec<String> {
    engine_substeps::render_engine_substeps_lines(step)
}

pub(super) fn render_assumption_lines(step: &Step) -> Vec<String> {
    assumptions::render_assumption_lines(step)
}
