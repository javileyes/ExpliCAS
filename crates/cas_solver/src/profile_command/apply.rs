use super::eval::evaluate_profile_command_input;
use cas_solver_core::profile_command_types::ProfileCommandResult;

/// Apply a `profile` command directly to a simplifier and return user-facing text.
pub fn apply_profile_command(simplifier: &mut crate::Simplifier, line: &str) -> String {
    match evaluate_profile_command_input(line) {
        ProfileCommandResult::ShowReport => simplifier.profiler.report(),
        ProfileCommandResult::SetEnabled { enabled, message } => {
            if enabled {
                simplifier.profiler.enable();
            } else {
                simplifier.profiler.disable();
            }
            message
        }
        ProfileCommandResult::Clear { message } => {
            simplifier.profiler.clear();
            message
        }
        ProfileCommandResult::Invalid { message } => message,
    }
}
