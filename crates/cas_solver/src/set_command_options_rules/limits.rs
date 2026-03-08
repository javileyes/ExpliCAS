use crate::{SetCommandPlan, SetCommandResult};

pub(crate) fn evaluate_max_rewrites_option(value: &str) -> SetCommandResult {
    match value.parse::<usize>() {
        Ok(n) => {
            let mut plan = SetCommandPlan::with_message(format!("Max rewrites set to {}", n));
            plan.set_max_rewrites = Some(n);
            SetCommandResult::Apply { plan }
        }
        Err(_) => SetCommandResult::Invalid {
            message: "Usage: set max-rewrites <number>".to_string(),
        },
    }
}
