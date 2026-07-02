//! Health command/session-facing API re-exported for session clients.

pub use crate::health_category::Category as HealthSuiteCategory;
pub use crate::repl_health_runtime::{
    evaluate_health_command_message_on_runtime as evaluate_health_command_message_on_repl_core,
    update_health_report_on_runtime as update_health_report_on_repl_core, ReplHealthRuntimeContext,
};
pub use cas_solver_core::health_runtime::{
    HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
