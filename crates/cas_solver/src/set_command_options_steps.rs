use crate::{SetCommandPlan, SetCommandResult, SetDisplayMode};

pub(crate) fn evaluate_set_steps(value: &str) -> SetCommandResult {
    match value {
        "on" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: on (full collection, normal display)");
            plan.set_steps_mode = Some(crate::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Normal);
            SetCommandResult::Apply { plan }
        }
        "off" => {
            let mut plan = SetCommandPlan::with_message("Steps: off");
            plan.set_steps_mode = Some(crate::StepsMode::Off);
            plan.set_display_mode = Some(SetDisplayMode::None);
            SetCommandResult::Apply { plan }
        }
        "compact" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: compact (no before/after snapshots)");
            plan.set_steps_mode = Some(crate::StepsMode::Compact);
            SetCommandResult::Apply { plan }
        }
        "verbose" => {
            let mut plan = SetCommandPlan::with_message("Steps: verbose (all rules, full detail)");
            plan.set_steps_mode = Some(crate::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Verbose);
            SetCommandResult::Apply { plan }
        }
        "succinct" => {
            let mut plan =
                SetCommandPlan::with_message("Steps: succinct (compact 1-line per step)");
            plan.set_steps_mode = Some(crate::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Succinct);
            SetCommandResult::Apply { plan }
        }
        "normal" => {
            let mut plan = SetCommandPlan::with_message("Steps: normal (default display)");
            plan.set_steps_mode = Some(crate::StepsMode::On);
            plan.set_display_mode = Some(SetDisplayMode::Normal);
            SetCommandResult::Apply { plan }
        }
        "none" => {
            let mut plan =
                SetCommandPlan::with_message("Steps display: none (collection still active)");
            plan.set_display_mode = Some(SetDisplayMode::None);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set steps <on|off|compact|verbose|succinct|normal|none>".to_string(),
        },
    }
}
