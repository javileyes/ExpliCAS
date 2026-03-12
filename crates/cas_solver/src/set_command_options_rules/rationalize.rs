use crate::{SetCommandPlan, SetCommandResult};
use cas_solver_core::rationalize_policy::AutoRationalizeLevel;

pub(crate) fn evaluate_rationalize_option(value: &str) -> SetCommandResult {
    match value {
        "on" | "true" | "auto" => {
            let mut plan = SetCommandPlan::with_message("Rationalization ENABLED (Level 1.5)");
            plan.set_rationalize = Some(AutoRationalizeLevel::Level15);
            SetCommandResult::Apply { plan }
        }
        "off" | "false" => {
            let mut plan = SetCommandPlan::with_message("Rationalization DISABLED");
            plan.set_rationalize = Some(AutoRationalizeLevel::Off);
            SetCommandResult::Apply { plan }
        }
        "0" | "level0" => {
            let mut plan =
                SetCommandPlan::with_message("Rationalization set to Level 0 (single sqrt)");
            plan.set_rationalize = Some(AutoRationalizeLevel::Level0);
            SetCommandResult::Apply { plan }
        }
        "1" | "level1" => {
            let mut plan =
                SetCommandPlan::with_message("Rationalization set to Level 1 (binomial conjugate)");
            plan.set_rationalize = Some(AutoRationalizeLevel::Level1);
            SetCommandResult::Apply { plan }
        }
        "1.5" | "level15" => {
            let mut plan = SetCommandPlan::with_message(
                "Rationalization set to Level 1.5 (same-surd products)",
            );
            plan.set_rationalize = Some(AutoRationalizeLevel::Level15);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set rationalize <on|off|0|1|1.5>".to_string(),
        },
    }
}
