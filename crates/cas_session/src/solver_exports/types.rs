//! Solver command/output types re-exported for session clients.

pub use cas_solver::{BindingOverviewEntry, ClearBindingsResult};
pub use cas_solver::{
    DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
};
pub use cas_solver::{HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput};
pub use cas_solver::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use cas_solver::{ReplSetCommandOutput, ReplSetMessageKind};
pub use cas_solver::{
    SolveCommandEvalError, SolveCommandEvalOutput, SolveCommandInput, SolvePrepareError,
    TimelineCommandEvalError, TimelineCommandEvalOutput, TimelineCommandInput,
    TimelineSimplifyEvalError, TimelineSimplifyEvalOutput, TimelineSolveEvalError,
    TimelineSolveEvalOutput,
};
