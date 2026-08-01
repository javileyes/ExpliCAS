/// Escape HTML special characters
pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Prepare string for LaTeX rendering in MathJax
pub fn latex_escape(s: &str) -> String {
    // For MathJax, we mostly just need the string as-is
    // But escape backslashes that aren't part of LaTeX commands
    s.to_string()
}

/// Escape a VARIABLE NAME for math-mode LaTeX. Deliberately narrower than
/// [`latex_escape`]: only `#` (TeX's macro-parameter character — MathJax
/// errors on it raw) is escaped, so conventional names like `x_1` keep their
/// subscript rendering. The bare session-ref shorthand parses as a variable
/// literally named `#N`, which reaches every renderer through this path.
pub fn latex_variable_name(s: &str) -> String {
    // Spelled Greek names render as real Greek (`alpha` → `\alpha`), so the
    // α-typed and alpha-typed spellings of the SAME symbol both display as
    // the letter. Exact whole-name match only (`beta1` stays as-is).
    if let Some(cmd) = cas_ast::greek_name_latex(s) {
        return cmd.to_string();
    }
    if s.contains('#') {
        s.replace('#', "\\#")
    } else {
        s.to_string()
    }
}

/// Escape arbitrary display text for the BODY of a `\text{…}` group.
///
/// The caller MUST place the result literally between `\text{` and `}` — the
/// backslash case closes and reopens the group, because no text-mode spelling
/// of a backslash renders in MathJax.
///
/// Text mode is not math mode, and the difference is the whole point of this
/// table. Everything below was measured against MathJax 3 `tex-svg` itself —
/// the renderer the web loads (`web/index.html:18`) — by typesetting each
/// candidate and LOOKING at the glyphs it painted:
///
/// * `^ _ ~ & # %` are ordinary characters inside `\text{…}` and render as
///   themselves. Their LaTeX escapes are the trap: `\_`, `\&`, `\#` and `\%`
///   are not implemented by MathJax's text-mode parser, which echoes the
///   SOURCE — `root_sum` escaped to `root\_sum` displayed as `root\_sum`,
///   backslash and all. Raw is both simpler and the only correct spelling.
/// * `\unicode{xNN}` is math-mode-only for the same reason. It IS in MathJax's
///   autoload map — which is what made it look verified — but inside `\text{…}`
///   nothing expands it, so `x\unicode{x5E}2` reached the reader verbatim
///   instead of `x^2`.
/// * `{`, `}` and `$` are the three that genuinely must be escaped, and their
///   escapes are the three text-mode ones MathJax does implement. Unescaped
///   they do not degrade a name, they take down the row: an unbalanced brace
///   leaves the whole `$…$` unrendered and a bare `$` closes the delimiter
///   early («Math not terminated in text box»).
pub fn escape_for_text_mode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            // No text-mode spelling of `\` renders (`\textbackslash`,
            // `\backslash`, `\char92` all echo their source), so step OUT of
            // the group, emit the glyph in math mode, and step back in.
            '\\' => out.push_str("}\\unicode{x5C}\\text{"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            _ => out.push(ch),
        }
    }
    out
}

/// Escape a FUNCTION NAME for use inside `\text{…}`.
///
/// Unknown/engine-internal functions are rendered as `\text{<name>}(args)`, and
/// several of them carry an underscore (`root_sum`, the RootSum closure of the
/// rational-integration frontier) — see [`escape_for_text_mode`] for why that
/// underscore must travel RAW.
pub fn latex_text_name(s: &str) -> String {
    escape_for_text_mode(s)
}

#[cfg(test)]
mod tests {
    use super::{escape_for_text_mode, latex_text_name};

    /// Replaces `latex_text_name_escapes_the_underscore_of_root_sum`, which
    /// pinned the wrong spelling: `\_` is not implemented in MathJax's text
    /// mode, so the escape it asserted was displayed literally as `root\_sum`.
    #[test]
    fn latex_text_name_lets_the_underscore_of_root_sum_travel_raw() {
        assert_eq!(latex_text_name("root_sum"), "root_sum");
    }

    #[test]
    fn latex_text_name_leaves_ordinary_names_alone() {
        assert_eq!(latex_text_name("cbrt"), "cbrt");
    }

    /// The characters that are ordinary in text mode stay untouched: escaping
    /// them is what made the escape itself visible.
    #[test]
    fn escape_for_text_mode_keeps_text_mode_ordinary_characters_raw() {
        assert_eq!(
            escape_for_text_mode("y·2·x^(2 - 1) & 50% #1 a~b"),
            "y·2·x^(2 - 1) & 50% #1 a~b"
        );
    }

    /// The three that must be escaped — and only those.
    #[test]
    fn escape_for_text_mode_escapes_braces_and_dollar() {
        assert_eq!(escape_for_text_mode("{1, 2} $x"), "\\{1, 2\\} \\$x");
    }

    /// A backslash leaves the group, renders in math mode, and re-enters it, so
    /// the result is still the body of a well-formed `\text{…}`.
    #[test]
    fn escape_for_text_mode_renders_a_backslash_through_math_mode() {
        assert_eq!(
            format!("\\text{{{}}}", escape_for_text_mode("a\\b")),
            "\\text{a}\\unicode{x5C}\\text{b}"
        );
    }
}
