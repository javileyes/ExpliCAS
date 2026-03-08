use crate::{SetCommandPlan, SetCommandResult};

pub(super) fn evaluate_transform_option(value: &str) -> SetCommandResult {
    match value {
        "on" | "true" | "1" => {
            let mut plan =
                SetCommandPlan::with_message("Transform phase ENABLED (distribution, expansion)");
            plan.set_transform = Some(true);
            SetCommandResult::Apply { plan }
        }
        "off" | "false" | "0" => {
            let mut plan = SetCommandPlan::with_message(
                "Transform phase DISABLED (no distribution/expansion)",
            );
            plan.set_transform = Some(false);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set transform <on|off>".to_string(),
        },
    }
}
