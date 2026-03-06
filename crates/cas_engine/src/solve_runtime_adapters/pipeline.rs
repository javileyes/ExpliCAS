pub(crate) use pipeline_preflight::{build_solve_preflight_state, prepare_equation_for_strategy};
pub(crate) use pipeline_strategy::{apply_strategy, execute_strategy_pipeline};

#[path = "pipeline_preflight.rs"]
mod pipeline_preflight;
#[path = "pipeline_strategy.rs"]
mod pipeline_strategy;
