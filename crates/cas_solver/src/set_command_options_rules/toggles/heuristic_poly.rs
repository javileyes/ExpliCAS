use crate::{HeuristicPoly, SetCommandPlan, SetCommandResult};

pub(super) fn evaluate_heuristic_poly_option(value: &str) -> SetCommandResult {
    match value {
        "on" | "true" | "1" => {
            let mut plan = SetCommandPlan::with_message(
                "Heuristic polynomial simplification: ON\n  - Extract common factors in Add/Sub\n  - Poly normalize if no factor found\n  Example: (x+1)^4 + 4·(x+1)^3 → (x+1)³·(x+5)",
            );
            plan.set_heuristic_poly = Some(HeuristicPoly::On);
            SetCommandResult::Apply { plan }
        }
        "off" | "false" | "0" => {
            let mut plan =
                SetCommandPlan::with_message("Heuristic polynomial simplification: OFF (default)");
            plan.set_heuristic_poly = Some(HeuristicPoly::Off);
            SetCommandResult::Apply { plan }
        }
        _ => SetCommandResult::Invalid {
            message: "Usage: set heuristic_poly <off|on>".to_string(),
        },
    }
}
