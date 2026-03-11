use cas_api_models::SubstituteRunOptions;
mod eval;
mod parse;
mod response;

use self::response::substitute_str_to_wire_impl;

/// Substitute an expression and return wire response.
///
/// This is the **solver-level direct entry point** for wire-returning
/// stateless substitution. Frontends should normally go through
/// Stateless wire entrypoint for substitute-style callers.
///
/// # Arguments
/// * `expr_str` - Expression string to substitute in
/// * `target_str` - Target expression to replace
/// * `with_str` - Replacement expression
/// * `opts_json` - Options JSON string (optional, see `SubstituteRunOptions`)
///
/// # Returns
/// Wire payload string with `SubstituteWireResponse` (schema v1).
/// Always returns valid JSON, even on errors.
pub fn substitute_str_to_wire(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String {
    let opts = SubstituteRunOptions::parse_optional_json(opts_json);
    substitute_str_to_wire_impl(expr_str, target_str, with_str, opts)
}

#[cfg(test)]
mod tests;
