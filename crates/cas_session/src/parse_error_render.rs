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
pub fn render_error_with_caret(input: &str, span: Span, message: &str) -> String {
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
pub fn render_parse_error(input: &str, error: &ParseError) -> String {
    if let Some(span) = error.span() {
        render_error_with_caret(input, span, error.message())
    } else {
        format!("Parse error: {}", error)
    }
}
