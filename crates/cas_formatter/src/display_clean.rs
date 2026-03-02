use regex::Regex;
use std::sync::LazyLock;

/// Clean verbose artifacts from display strings for didactic CLI output.
pub fn clean_display_string(s: &str) -> String {
    let mut result = s.to_string();
    let mut changed = true;

    // Iterate until no more changes (handles nested patterns).
    while changed {
        let before = result.clone();

        if result.starts_with("1 * ") {
            result = result[4..].to_string();
        }
        if result.ends_with(" * 1") && !result.ends_with("0 * 1") {
            result = result[..result.len() - 4].to_string();
        }

        result = result.replace("(1 * ", "(");
        result = result.replace(" * 1)", ")");
        result = result.replace("1 * 1", "1");
        result = result.replace(" + 1 * ", " + ");
        result = result.replace(" - 1 * ", " - ");
        result = result.replace("/ (1 * ", "/ (");
        result = result.replace(" * 1 +", " +");
        result = result.replace(" * 1 -", " -");
        result = result.replace(" * 1 /", " /");
        result = result.replace(" * 1 *", " *");

        if result.starts_with("1 * -") {
            result = result[4..].to_string();
        }

        if result.starts_with("1 * ") && result.len() > 4 {
            let next_char = result.chars().nth(4);
            if let Some(c) = next_char {
                if c.is_ascii_digit() || c == '-' || c == 'x' || c == 'y' || c == '(' {
                    result = result[4..].to_string();
                }
            }
        }

        result = result.replace("/ (1 * x)", "/ x");
        result = result.replace("/ (1 * y)", "/ y");
        result = result.replace(" * -1 * ", " * -");

        // Also handle middot patterns used in display output.
        result = result.replace("(1·", "(");
        result = result.replace("·1)", ")");
        result = result.replace(" + 1·", " + ");
        result = result.replace(" - 1·", " - ");
        result = result.replace("·1 +", " +");
        result = result.replace("·1 -", " -");
        result = result.replace("·1 /", " /");
        result = result.replace("·1·", "·");
        result = result.replace("·1)", ")");
        if result.ends_with("·1") {
            result = result[..result.len() - 3].to_string();
        }
        if result.starts_with("1·") && result.len() > 3 {
            result = result[3..].to_string();
        }

        changed = before != result;
    }

    // Remove __hold(...) wrapper.
    while let Some(start) = result.find("__hold(") {
        let content_start = start + 7;
        let mut depth = 1;
        let mut end = content_start;
        for (i, c) in result[content_start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = content_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let inner = &result[content_start..end];
            result = format!("{}{}{}", &result[..start], inner, &result[end + 1..]);
        } else {
            break;
        }
    }

    clean_sign_patterns(result)
}

/// Clean sign patterns from display strings.
///
/// Converts `+ -` to `-` and `- -` to `+` only when followed by a digit or
/// variable, NOT when followed by `(` (to preserve grouped subexpressions).
pub fn clean_sign_patterns(s: String) -> String {
    static RE_PLUS_MINUS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\+ -([0-9a-zA-Z√^])").expect("valid regex literal"));
    static RE_MINUS_MINUS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"- -([0-9a-zA-Z√^])").expect("valid regex literal"));
    static RE_PLUS_MINUS_COMPACT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\+-([0-9a-zA-Z])").expect("valid regex literal"));
    static RE_MINUS_MINUS_COMPACT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"--([0-9a-zA-Z])").expect("valid regex literal"));

    let mut result = s;
    result = RE_PLUS_MINUS.replace_all(&result, "- $1").to_string();
    result = RE_MINUS_MINUS.replace_all(&result, "+ $1").to_string();
    result = RE_PLUS_MINUS_COMPACT
        .replace_all(&result, "-$1")
        .to_string();
    result = RE_MINUS_MINUS_COMPACT
        .replace_all(&result, "+$1")
        .to_string();
    result
}

#[cfg(test)]
mod tests {
    use super::clean_display_string;

    #[test]
    fn removes_hold_wrapper() {
        assert_eq!(clean_display_string("__hold(x+1)"), "x+1");
    }

    #[test]
    fn cleans_sign_patterns() {
        assert_eq!(clean_display_string("x + -y"), "x - y");
        assert_eq!(clean_display_string("x - -y"), "x + y");
    }
}
