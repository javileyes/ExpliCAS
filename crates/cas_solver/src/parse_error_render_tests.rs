#[cfg(test)]
mod tests {
    use cas_ast::Span;
    use cas_parser::ParseError;

    use crate::{render_error_with_caret, render_parse_error};

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
        let span = Span::new(4, 7);
        let rendered = render_error_with_caret(input, span, "unknown");
        assert!(rendered.contains("    ^~~ unknown"));
    }

    #[test]
    fn test_caret_out_of_bounds_clamped() {
        let input = "x + y";
        let span = Span::new(100, 200);
        let rendered = render_error_with_caret(input, span, "error");
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
        // Chokepoint E contract: exactly ONE "Parse error" prefix (the
        // parser's own Display), never the doubled "Parse error: Parse
        // error at ..." this test used to pin.
        assert!(rendered.starts_with("Parse error"));
        assert_eq!(rendered.matches("Parse error").count(), 1);
        assert!(rendered.contains("something wrong"));
    }

    #[test]
    fn parse_error_message_owns_the_prefix_exactly_once() {
        use super::super::parse_error_render::parse_error_message;
        // Self-prefixed parser Display passes through untouched.
        let doubled_before = parse_error_message(ParseError::syntax("x"));
        assert_eq!(doubled_before.matches("Parse error").count(), 1);
        // Bare messages get the prefix added exactly once.
        assert_eq!(parse_error_message("bad input"), "Parse error: bad input");
    }
}
