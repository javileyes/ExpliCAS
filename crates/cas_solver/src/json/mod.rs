//! Canonical JSON API entry points for solver responses.
//!
//! This module provides stable, serializable JSON entry points for CLI and FFI consumers.
//! All callsites should use these types to ensure consistent JSON schema.
//!
//! # Schema Version
//!
//! Current schema version: **1**
//!
//! # Stability Contract
//!
//! - `schema_version`, `ok`, `kind`, `code` are **stable** - do not change
//! - `message` is human-readable and may change between versions
//! - `details` is extensible (new keys may be added)

mod config;
mod envelope;
mod eval;
mod eval_flow;
mod input_parse;
mod limit;
mod mappers;
mod output;
mod request;
mod solve_render;
mod substitute;

pub use cas_api_models::{EnvelopeEvalOptions, EvalJsonOutput, StepJson};
pub use config::{
    apply_eval_json_options, build_budget_json_eval, build_domain_json_eval,
    build_options_json_eval, build_semantics_json_eval,
};
pub use envelope::eval_str_to_output_envelope;
pub use eval::eval_str_to_json;
pub use eval_flow::{evaluate_eval_json_with_session, EvalJsonSessionRunConfig};
pub use input_parse::{parse_eval_json_special_command, EvalJsonSpecialCommand};
pub use limit::{eval_limit_from_str, limit_str_to_json, LimitEvalError, LimitEvalResult};
pub use output::{
    build_eval_json_error_output, build_eval_json_output, build_eval_wire_value,
    expr_hash_eval_json, expr_stats_eval_json, finalize_eval_json_output, format_eval_result_text,
    format_expr_limited_eval_json, EvalJsonFinalizeInput, EvalJsonOutputBuild,
};
pub use request::{build_eval_request_for_input, format_eval_input_latex};
pub use solve_render::{
    collect_required_conditions_eval_json, collect_required_display_eval_json,
    collect_solve_steps_eval_json, collect_warnings_eval_json, detect_solve_variable_eval_json,
    format_solution_set_eval_json, solution_set_to_latex_eval_json,
};
pub use substitute::{
    eval_substitute_from_str, evaluate_substitute_subcommand_json,
    evaluate_substitute_subcommand_text_lines, evaluate_substitute_subcommand_text_lines_with_mode,
    format_substitute_subcommand_text_lines, substitute_str_to_json,
    substitute_str_to_json_with_options, SubstituteEvalError, SubstituteEvalMode,
    SubstituteEvalResult, SubstituteEvalStep,
};
