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

/// Escape a FUNCTION NAME for use inside `\text{…}`.
///
/// Unknown/engine-internal functions are rendered as `\text{<name>}(args)`, and
/// several of them carry an underscore (`root_sum`, the RootSum closure of the
/// rational-integration frontier). Raw, that underscore is a HARD MathJax error
/// —`'_' allowed only in math mode`— which does not degrade the name, it kills
/// the rendering of the entire expression: `integrate(1/(x^5-x-1), x)` published
/// a broken row on the web. Verified against MathJax 3 itself, the renderer the
/// web loads.
pub fn latex_text_name(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\unicode{x5C}"),
            '^' => out.push_str("\\unicode{x5E}"),
            '~' => out.push_str("\\unicode{x7E}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '%' => out.push_str("\\%"),
            '_' => out.push_str("\\_"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::latex_text_name;

    #[test]
    fn latex_text_name_escapes_the_underscore_of_root_sum() {
        assert_eq!(latex_text_name("root_sum"), "root\\_sum");
    }

    #[test]
    fn latex_text_name_leaves_ordinary_names_alone() {
        assert_eq!(latex_text_name("cbrt"), "cbrt");
    }
}
