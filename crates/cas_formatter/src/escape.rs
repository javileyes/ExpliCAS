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
    if s.contains('#') {
        s.replace('#', "\\#")
    } else {
        s.to_string()
    }
}
