pub(super) fn replace_last_fraction(
    value: &str,
    find_balanced_braces: fn(&str) -> Option<(String, usize)>,
) -> Option<String> {
    let start = value.rfind("\\frac{")?;
    let rest = &value[start + 5..];
    let (numerator, numerator_end) = find_balanced_braces(rest)?;
    let after_numerator = &rest[numerator_end + 1..];
    if !after_numerator.starts_with('{') {
        return None;
    }
    let (denominator, denominator_end) = find_balanced_braces(after_numerator)?;
    let total_end = start + 5 + numerator_end + 1 + denominator_end + 1;
    let replacement = format!(
        "{}/{}",
        format_fraction_side(&numerator),
        format_fraction_side(&denominator)
    );
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}

fn format_fraction_side(side: &str) -> String {
    let trimmed = side.trim();
    if needs_fraction_parentheses(trimmed) {
        format!("({trimmed})")
    } else {
        trimmed.to_string()
    }
}

fn needs_fraction_parentheses(value: &str) -> bool {
    let mut depth = 0usize;
    let chars: Vec<(usize, char)> = value.char_indices().collect();

    let mut index = 0usize;
    while index < chars.len() {
        let (_, ch) = chars[index];
        match ch {
            '{' | '(' => depth += 1,
            '}' | ')' => depth = depth.saturating_sub(1),
            '+' if depth == 0 => return true,
            '-' if depth == 0 && index > 0 => return true,
            '/' if depth == 0 => return true,
            '\\' if depth == 0 => {
                let tail = &value[chars[index].0..];
                if tail.starts_with("\\cdot") {
                    return true;
                }
            }
            _ => {}
        }
        index += 1;
    }

    false
}
