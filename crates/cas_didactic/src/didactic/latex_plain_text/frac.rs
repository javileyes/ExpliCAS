pub(super) fn replace_last_fraction(value: &str) -> Option<String> {
    let start = value.rfind("\\frac{")?;
    let rest = &value[start + 5..];
    let (numerator, numerator_end) = find_balanced_braces(rest)?;
    let after_numerator = &rest[numerator_end + 1..];
    if !after_numerator.starts_with('{') {
        return None;
    }
    let (denominator, denominator_end) = find_balanced_braces(after_numerator)?;
    let total_end = start + 5 + numerator_end + 1 + denominator_end + 1;
    let replacement = format!("({}/{})", numerator, denominator);
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}

fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0;
    let mut content = String::new();
    for (i, c) in s.chars().enumerate() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(c);
            }
            _ => {
                if depth > 0 {
                    content.push(c);
                }
            }
        }
    }
    None
}
