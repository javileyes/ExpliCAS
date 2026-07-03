use crate::health_command_format::format_health_report_lines;
use crate::health_command_messages::clear_health_profiler;
use crate::health_command_parse::evaluate_health_command_input;
use cas_solver_core::health_runtime::{HealthCommandEvalOutput, HealthCommandInput};

use super::{output, status::evaluate_health_status_lines};

/// Evaluate full `health ...` command and return both lines and side-effect intents.
///
/// The caller applies `set_enabled` and `clear_last_report` to its UI/session state.
pub(crate) fn evaluate_health_command(
    simplifier: &mut crate::Simplifier,
    line: &str,
    last_stats: Option<&crate::PipelineStats>,
    last_health_report: Option<&str>,
) -> Result<HealthCommandEvalOutput, String> {
    match evaluate_health_command_input(line)? {
        HealthCommandInput::ShowLast => Ok(output::build_show_last_output(
            format_health_report_lines(last_stats, last_health_report),
        )),
        HealthCommandInput::SetEnabled { enabled } => {
            // Keep the engine profiler in lockstep with the session flag —
            // health metrics (incl. domain-assumption hits) are recorded by
            // the simplifier's own profiler, not by session state.
            simplifier.profiler.set_health_tracking(enabled);
            Ok(output::build_set_enabled_output(enabled))
        }
        HealthCommandInput::Clear => {
            clear_health_profiler(simplifier);
            Ok(output::build_clear_output())
        }
        HealthCommandInput::Status(status) => {
            let lines = evaluate_health_status_lines(simplifier, &status)?;
            Ok(output::build_status_output(lines))
        }
        HealthCommandInput::Invalid => {
            unreachable!("invalid is handled in evaluate_health_command_input")
        }
    }
}
