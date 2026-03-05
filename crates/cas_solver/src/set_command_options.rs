use crate::set_command_options_rules::{
    evaluate_autoexpand_option, evaluate_debug_option, evaluate_heuristic_poly_option,
    evaluate_max_rewrites_option, evaluate_rationalize_option, evaluate_transform_option,
};
use crate::set_command_options_steps::evaluate_set_steps;
use crate::{format_set_help_text, SetCommandResult, SetCommandState};

pub(crate) fn evaluate_set_option(
    option: &str,
    value: &str,
    state: SetCommandState,
) -> SetCommandResult {
    match option {
        "transform" => evaluate_transform_option(value),
        "autoexpand" | "autoexpand_binomials" => evaluate_autoexpand_option(value),
        "heuristic_poly" => evaluate_heuristic_poly_option(value),
        "rationalize" => evaluate_rationalize_option(value),
        "max-rewrites" => evaluate_max_rewrites_option(value),
        "steps" => evaluate_set_steps(value),
        "debug" => evaluate_debug_option(value),
        _ => SetCommandResult::ShowHelp {
            message: format_set_help_text(state),
        },
    }
}
