//! Solver command/output types re-exported for session clients.

pub use crate::command_api::eval::{EvalCommandError, EvalCommandOutput, EvalCommandRenderPlan};
pub use crate::solve_command_eval_core::SolveCommandEvalOutput;
pub use crate::timeline_types::{
    TimelineCommandEvalOutput, TimelineSimplifyEvalOutput, TimelineSolveEvalOutput,
};
pub use cas_solver_core::health_runtime::{
    HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
pub use cas_solver_core::history_models::{
    DeleteHistoryError, DeleteHistoryResult, HistoryEntryDetails, HistoryEntryInspection,
    HistoryExprInspection, HistoryOverviewEntry, HistoryOverviewKind,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use cas_solver_core::repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};
pub use cas_solver_core::session_runtime::{BindingOverviewEntry, ClearBindingsResult};
pub use cas_solver_core::solve_command_types::{
    SolveCommandEvalError, SolveCommandInput, SolvePrepareError, TimelineCommandEvalError,
    TimelineCommandInput, TimelineSimplifyEvalError, TimelineSolveEvalError,
};
