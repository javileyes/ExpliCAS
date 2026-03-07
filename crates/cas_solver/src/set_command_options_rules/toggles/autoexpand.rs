use crate::{AutoExpandBinomials, SetCommandPlan, SetCommandResult};

pub(super) fn evaluate_autoexpand_option(value: &str) -> SetCommandResult {
    match value {
        "on" | "true" | "1" => {
            let mut plan = SetCommandPlan::with_message(
                "Autoexpand binomials: ON (always expand)\n  (x+1)^5 will now expand to x⁵+5x⁴+10x³+10x²+5x+1",
            );
            plan.set_autoexpand_binomials = Some(AutoExpandBinomials::On);
            SetCommandResult::Apply { plan }
        }
        "off" | "false" | "0" => {
            let mut plan = SetCommandPlan::with_message(
                "Autoexpand binomials: OFF (default, keep factored form)",
            );
            plan.set_autoexpand_binomials = Some(AutoExpandBinomials::Off);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set autoexpand <off|on>".to_string(),
        },
    }
}
