use cas_api_models::SubstituteJsonOptions;
mod eval;
mod parse;
mod response;

use self::response::substitute_str_to_json_impl;

/// Substitute an expression and return JSON response.
///
/// This is the **solver-level canonical entry point** for JSON-returning
/// stateless substitution. Frontends should normally go through
/// `cas_session::evaluate_substitute_json_canonical`.
///
/// # Arguments
/// * `expr_str` - Expression string to substitute in
/// * `target_str` - Target expression to replace
/// * `with_str` - Replacement expression
/// * `opts_json` - Options JSON string (optional, see `SubstituteJsonOptions`)
///
/// # Returns
/// JSON string with `SubstituteJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
pub fn substitute_str_to_json(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String {
    let opts = SubstituteJsonOptions::parse_optional_json(opts_json);
    substitute_str_to_json_impl(expr_str, target_str, with_str, opts)
}

#[cfg(test)]
mod tests;
