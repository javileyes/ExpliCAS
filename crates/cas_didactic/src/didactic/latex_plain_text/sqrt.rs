pub(super) fn replace_last_sqrt(value: &str) -> Option<String> {
    let start = value.rfind("\\sqrt{")?;
    let rest = &value[start + 5..];
    let (radicand, radicand_end) = find_balanced_braces(rest)?;
    let total_end = start + 5 + radicand_end + 1;
    let replacement = format!("sqrt({})", radicand);
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[total_end..]
    ))
}

fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut content = String::new();

    for (i, c) in s.char_indices() {
        match c {
            '{' => {
                if depth > 0 {
                    content.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.checked_sub(1)?;
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
