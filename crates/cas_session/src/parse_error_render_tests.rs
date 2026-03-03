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
        assert!(rendered.contains("Parse error:"));
    }
}
