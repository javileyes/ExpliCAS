//! Parse error rendering with source location caret.
//!
//! Provides visual error rendering with caret indicators for parse errors.

use cas_ast::Span;
use cas_parser::ParseError;

const SUBSTITUTE_USAGE_MESSAGE: &str = "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              → 12\n\
                       subst x^4 + x^2 + 1, x^2, y      → y² + y + 1\n\
                       subst x^3, x^2, y                → y·x";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParseExprPairError {
    MissingDelimiter,
    FirstArg(String),
    SecondArg(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParseSubstituteArgsError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}

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

pub fn format_expr_pair_parse_error_message(error: &ParseExprPairError, command: &str) -> String {
    match error {
        ParseExprPairError::MissingDelimiter => {
            format!("Usage: {} <expr1>, <expr2>", command)
        }
        ParseExprPairError::FirstArg(e) => format!("Error parsing first arg: {}", e),
        ParseExprPairError::SecondArg(e) => {
            format!("Error parsing second arg: {}", e)
        }
    }
}

pub fn format_substitute_parse_error_message(error: &ParseSubstituteArgsError) -> String {
    match error {
        ParseSubstituteArgsError::InvalidArity => SUBSTITUTE_USAGE_MESSAGE.to_string(),
        ParseSubstituteArgsError::Expression(e) => {
            format!("Error parsing expression: {}", e)
        }
        ParseSubstituteArgsError::Target(e) => {
            format!("Error parsing target: {}", e)
        }
        ParseSubstituteArgsError::Replacement(e) => {
            format!("Error parsing replacement: {}", e)
        }
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
