use cas_ast::{Case, Context};

use super::super::is_pure_residual_otherwise;
use super::display_solution_set;

pub(super) fn display_conditional_solution_set(ctx: &Context, cases: &[Case]) -> String {
    let case_strs: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            if is_pure_residual_otherwise(case) {
                return None;
            }
            let sol_str = display_solution_set(ctx, &case.then.solutions);
            if case.when.is_otherwise() {
                Some(format!("  otherwise: {}", sol_str))
            } else {
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                Some(format!("  if {}: {}", cond_str, sol_str))
            }
        })
        .collect();
    if case_strs.len() == 1 {
        case_strs[0].trim().to_string()
    } else {
        format!("Conditional:\n{}", case_strs.join("\n"))
    }
}
