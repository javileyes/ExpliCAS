//! Parse error rendering with source location caret.
//!
//! Provides visual error rendering with caret indicators for parse errors.

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
/// If span is out of bounds or unavailable, returns plain message.
pub fn render_error_with_caret(input: &str, span: Span, message: &str) -> String {
    // Clamp span to input bounds
    let start = span.start.min(input.len());
    let end = span.end.min(input.len()).max(start);

    let mut result = String::new();
    result.push_str(input);
    result.push('\n');

    // Add leading spaces
    result.push_str(&" ".repeat(start));

    // Add caret and underline
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caret_at_middle() {
        let input = "x + * 3";
        let span = Span::new(4, 5);
        let rendered = render_error_with_caret(input, span, "unexpected");
        assert!(rendered.contains("x + * 3"));
        assert!(rendered.contains("    ^ unexpected"));
    }

    #[test]
    fn test_caret_at_start() {
        let input = "* x + 3";
        let span = Span::new(0, 1);
        let rendered = render_error_with_caret(input, span, "bad start");
        assert!(rendered.contains("^ bad start"));
    }

    #[test]
    fn test_caret_multichar() {
        let input = "x + foo bar";
        let span = Span::new(4, 7); // "foo"
        let rendered = render_error_with_caret(input, span, "unknown");
        assert!(rendered.contains("    ^~~ unknown"));
    }

    #[test]
    fn test_caret_out_of_bounds_clamped() {
        let input = "x + y";
        let span = Span::new(100, 200); // Way out of bounds
        let rendered = render_error_with_caret(input, span, "error");
        // Should clamp to end of input
        assert!(rendered.contains("x + y"));
    }

    #[test]
    fn test_render_parse_error_with_span() {
        let error = ParseError::syntax_at("bad token", Span::new(2, 3));
        let rendered = render_parse_error("a + b", &error);
        assert!(rendered.contains("  ^ bad token"));
    }

    #[test]
    fn test_render_parse_error_without_span() {
        let error = ParseError::syntax("something wrong");
        let rendered = render_parse_error("a + b", &error);
        assert!(rendered.contains("Parse error:"));
    }
}
