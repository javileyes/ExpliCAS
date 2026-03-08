use crate::{SetCommandPlan, SetCommandResult};

pub(super) fn evaluate_debug_option(value: &str) -> SetCommandResult {
    match value {
        "on" | "true" | "1" => {
            let mut plan = SetCommandPlan::with_message(
                "Debug mode ENABLED (pipeline diagnostics after each operation)",
            );
            plan.set_debug_mode = Some(true);
            SetCommandResult::Apply { plan }
        }
        "off" | "false" | "0" => {
            let mut plan = SetCommandPlan::with_message("Debug mode DISABLED");
            plan.set_debug_mode = Some(false);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set debug <on|off>".to_string(),
        },
    }
}
