use cas_ast::hold::hold_name;

/// Clean verbose artifacts from display strings for didactic CLI output.
pub fn clean_display_string(s: &str) -> String {
    if !may_need_display_clean(s) {
        return s.to_owned();
    }

    let mut result = s.to_owned();
    let hold_pattern = hold_pattern();

    if may_need_unit_cleanup(&result) {
        clean_unit_patterns_in_place(&mut result);
    }
    if result.contains(&hold_pattern) {
        strip_hold_wrappers_in_place(&mut result, &hold_pattern);
    }
    if may_need_sign_cleanup(&result) {
        result = clean_sign_patterns(result);
    }

    result
}

fn may_need_display_clean(s: &str) -> bool {
    may_need_unit_cleanup(s) || s.contains(&hold_pattern()) || may_need_sign_cleanup(s)
}

fn may_need_unit_cleanup(s: &str) -> bool {
    s.contains("1 * ")
        || s.contains(" * 1")
        || s.contains("(1·")
        || s.contains("1·")
        || s.contains("·1")
        || s.contains(" * -1 * ")
}

fn may_need_sign_cleanup(s: &str) -> bool {
    s.contains("+ -") || s.contains("- -") || s.contains("+-") || s.contains("--")
}

fn replace_in_place(result: &mut String, from: &str, to: &str) -> bool {
    if !result.contains(from) {
        return false;
    }
    *result = result.replace(from, to);
    true
}

fn clean_unit_patterns_in_place(result: &mut String) {
    loop {
        let mut changed = false;

        if result.starts_with("1 * ") {
            result.drain(..4);
            changed = true;
        }
        if result.ends_with(" * 1") && !result.ends_with("0 * 1") {
            result.truncate(result.len() - 4);
            changed = true;
        }

        changed |= replace_in_place(result, "(1 * ", "(");
        changed |= replace_in_place(result, " * 1)", ")");
        changed |= replace_in_place(result, "1 * 1", "1");
        changed |= replace_in_place(result, " + 1 * ", " + ");
        changed |= replace_in_place(result, " - 1 * ", " - ");
        changed |= replace_in_place(result, "/ (1 * ", "/ (");
        changed |= replace_in_place(result, " * 1 +", " +");
        changed |= replace_in_place(result, " * 1 -", " -");
        changed |= replace_in_place(result, " * 1 /", " /");
        changed |= replace_in_place(result, " * 1 *", " *");

        if result.starts_with("1 * -") {
            result.drain(..4);
            changed = true;
        }

        if result.starts_with("1 * ") && result.len() > 4 {
            let next = result.as_bytes()[4];
            if next.is_ascii_digit() || matches!(next, b'-' | b'x' | b'y' | b'(') {
                result.drain(..4);
                changed = true;
            }
        }

        changed |= replace_in_place(result, "/ (1 * x)", "/ x");
        changed |= replace_in_place(result, "/ (1 * y)", "/ y");
        changed |= replace_in_place(result, " * -1 * ", " * -");

        changed |= replace_in_place(result, "(1·", "(");
        changed |= replace_in_place(result, "·1)", ")");
        changed |= replace_in_place(result, " + 1·", " + ");
        changed |= replace_in_place(result, " - 1·", " - ");
        changed |= replace_in_place(result, "·1 +", " +");
        changed |= replace_in_place(result, "·1 -", " -");
        changed |= replace_in_place(result, "·1 /", " /");
        changed |= replace_in_place(result, "·1·", "·");
        changed |= replace_in_place(result, "·1)", ")");

        if result.ends_with("·1") {
            result.truncate(result.len() - 3);
            changed = true;
        }
        if result.starts_with("1·") && result.len() > 3 {
            result.drain(..3);
            changed = true;
        }

        if !changed {
            break;
        }
    }
}

fn hold_pattern() -> String {
    format!("{}(", hold_name())
}

fn strip_hold_wrappers_in_place(result: &mut String, hold_pattern: &str) {
    while let Some(start) = result.find(hold_pattern) {
        let content_start = start + hold_pattern.len();
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
            *result = format!("{}{}{}", &result[..start], inner, &result[end + 1..]);
        } else {
            break;
        }
    }
}

/// Clean sign patterns from display strings.
///
/// Converts `+ -` to `-` and `- -` to `+` only when followed by a digit or
/// variable, NOT when followed by `(` (to preserve grouped subexpressions).
pub fn clean_sign_patterns(s: String) -> String {
    if !may_need_sign_cleanup(&s) {
        return s;
    }

    let chars: Vec<char> = s.chars().collect();
    let mut result = String::with_capacity(s.len());
    let mut i = 0;
    let mut changed = false;

    while i < chars.len() {
        if i + 3 < chars.len() && chars[i] == '+' && chars[i + 1] == ' ' && chars[i + 2] == '-' {
            let next = chars[i + 3];
            if matches_sign_follow(next) {
                result.push('-');
                result.push(' ');
                result.push(next);
                i += 4;
                changed = true;
                continue;
            }
        }

        if i + 3 < chars.len() && chars[i] == '-' && chars[i + 1] == ' ' && chars[i + 2] == '-' {
            let next = chars[i + 3];
            if matches_sign_follow(next) {
                result.push('+');
                result.push(' ');
                result.push(next);
                i += 4;
                changed = true;
                continue;
            }
        }

        if i + 2 < chars.len() && chars[i] == '+' && chars[i + 1] == '-' {
            let next = chars[i + 2];
            if next.is_ascii_alphanumeric() {
                result.push('-');
                result.push(next);
                i += 3;
                changed = true;
                continue;
            }
        }

        if i + 2 < chars.len() && chars[i] == '-' && chars[i + 1] == '-' {
            let next = chars[i + 2];
            if next.is_ascii_alphanumeric() {
                result.push('+');
                result.push(next);
                i += 3;
                changed = true;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    if changed {
        result
    } else {
        s
    }
}

fn matches_sign_follow(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '√' | '^')
}

#[cfg(test)]
mod tests {
    use super::{clean_display_string, clean_sign_patterns};

    #[test]
    fn removes_hold_wrapper() {
        assert_eq!(clean_display_string("__hold(x+1)"), "x+1");
    }

    #[test]
    fn cleans_sign_patterns() {
        assert_eq!(clean_display_string("x + -y"), "x - y");
        assert_eq!(clean_display_string("x - -y"), "x + y");
    }

    #[test]
    fn leaves_clean_strings_unchanged() {
        assert_eq!(clean_display_string("x + 1"), "x + 1");
        assert_eq!(
            clean_display_string("sin(x)^2 + cos(x)^2"),
            "sin(x)^2 + cos(x)^2"
        );
    }

    #[test]
    fn cleans_compact_sign_patterns_without_regex() {
        assert_eq!(clean_sign_patterns("x+-y".to_string()), "x-y");
        assert_eq!(clean_sign_patterns("x--y".to_string()), "x+y");
    }
}
