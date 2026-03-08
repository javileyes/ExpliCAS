/// Clean identity patterns from LaTeX strings (removes redundant ·1 patterns)
/// This is the LaTeX equivalent of clean_display_string for CLI output.
/// Patterns like "\cdot 1" and "1 \cdot x" are cleaned up for better display.
pub fn clean_latex_identities(latex: &str) -> String {
    use regex::Regex;
    use std::sync::LazyLock;

    // Compiled once, reused across all calls
    static RE_MULT_UNIT_FRAC: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(\d+)\s*\\cdot\s*\\frac\{1\}\{([^}]+)\}").expect("valid regex literal")
    });

    let mut result = latex.to_string();
    let mut changed = true;

    // Iterate until no more changes (handles nested patterns)
    while changed {
        let before = result.clone();

        // "\cdot 1}" at end of group → "}" (remove trailing ·1)
        result = result.replace("\\cdot 1}", "}");

        // "\cdot 1$" at end of math (inline) → "$"
        result = result.replace("\\cdot 1$", "$");

        // "\cdot 1 " (with space after) → " "
        result = result.replace("\\cdot 1 ", " ");

        // "\cdot 1+" → "+"
        result = result.replace("\\cdot 1+", "+");

        // "\cdot 1-" → "-"
        result = result.replace("\\cdot 1-", "-");

        // "1 \cdot " at start or after operators → ""
        result = result.replace("{1 \\cdot ", "{");

        // Handle pattern at start of string
        if result.starts_with("1 \\cdot ") {
            result = result[8..].to_string();
        }

        // "\cdot 1\" (before another LaTeX command) → "\"
        result = result.replace("\\cdot 1\\", "\\");

        // Handle "\frac{1}{1}" → "1"
        result = result.replace("\\frac{1}{1}", "1");

        // Standalone "\cdot 1" at the very end
        if result.ends_with("\\cdot 1") {
            result = result[..result.len() - 7].to_string();
        }

        // KEY FIX: Convert "n \cdot \frac{1}{expr}" → "\frac{n}{expr}"
        // This handles cases like "2 \cdot \frac{1}{x}" → "\frac{2}{x}"
        result = RE_MULT_UNIT_FRAC
            .replace_all(&result, r"\frac{$1}{$2}")
            .to_string();

        changed = before != result;
    }

    result
}
