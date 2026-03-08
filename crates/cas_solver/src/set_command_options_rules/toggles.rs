mod autoexpand;
mod debug;
mod heuristic_poly;
mod transform;

use crate::SetCommandResult;

pub(crate) fn evaluate_transform_option(value: &str) -> SetCommandResult {
    transform::evaluate_transform_option(value)
}

pub(crate) fn evaluate_autoexpand_option(value: &str) -> SetCommandResult {
    autoexpand::evaluate_autoexpand_option(value)
}

pub(crate) fn evaluate_heuristic_poly_option(value: &str) -> SetCommandResult {
    heuristic_poly::evaluate_heuristic_poly_option(value)
}

pub(crate) fn evaluate_debug_option(value: &str) -> SetCommandResult {
    debug::evaluate_debug_option(value)
}
