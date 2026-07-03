//! Parse error rendering with source location caret.
//!
//! Keeps parse error formatting reusable across frontends while preserving
//! the existing REPL visual contract.

use cas_ast::Span;
use cas_parser::ParseError;

/// Render parse error with caret indicator.
///
/// # Example Output
/// ```text
/// x + * 3
///     ^ unexpected token
/// ```
///
/// If span is out of bounds, it is clamped to input bounds.
pub(crate) fn render_error_with_caret(input: &str, span: Span, message: &str) -> String {
    let start = span.start.min(input.len());
    let end = span.end.min(input.len()).max(start);

    let mut result = String::new();
    result.push_str(input);
    result.push('\n');

    result.push_str(&" ".repeat(start));
    result.push('^');
    let underline_len = end.saturating_sub(start + 1);
    if underline_len > 0 {
        result.push_str(&"~".repeat(underline_len));
    }

    result.push(' ');
    result.push_str(message);
    result
}

/// Render a ParseError, using caret if span is available.
pub(crate) fn render_parse_error(input: &str, error: &ParseError) -> String {
    if let Some(span) = error.span() {
        render_error_with_caret(input, span, error.message())
    } else {
        parse_error_message(error)
    }
}

/// Chokepoint E: single owner of the "Parse error: " message prefix.
///
/// The parser's `Display` for the `Syntax` variant already self-describes
/// ("Parse error at <span>: ..."), so blindly prepending produced
/// "Parse error: Parse error at ...". Callers building a parse-error MESSAGE
/// string go through here and never prepend themselves.
pub(crate) fn parse_error_message(error: impl std::fmt::Display) -> String {
    let s = error.to_string();
    if s.starts_with("Parse error") {
        s
    } else {
        format!("Parse error: {s}")
    }
}
