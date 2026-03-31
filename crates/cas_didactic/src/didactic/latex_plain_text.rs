mod frac;
mod sqrt;
mod text;

/// Convert LaTeX-like notation into plain-text form for CLI display.
pub fn latex_to_plain_text(s: &str) -> String {
    let mut result = s.to_string();

    result = text::strip_text_wrappers(result);

    let mut iterations = 0;
    while result.contains("\\frac{") && iterations < 10 {
        iterations += 1;
        let Some(next_result) = frac::replace_last_fraction(&result) else {
            break;
        };
        result = next_result;
    }

    let mut sqrt_iterations = 0;
    while result.contains("\\sqrt{") && sqrt_iterations < 10 {
        sqrt_iterations += 1;
        let Some(next_result) = sqrt::replace_last_sqrt(&result) else {
            break;
        };
        result = next_result;
    }

    let mut power_iterations = 0;
    while result.contains("}^{") && power_iterations < 10 {
        power_iterations += 1;
        let Some(next_result) = replace_next_parenthesized_power_base(&result) else {
            break;
        };
        result = next_result;
    }

    let mut fractional_exponent_iterations = 0;
    while result.contains("^{") && fractional_exponent_iterations < 10 {
        fractional_exponent_iterations += 1;
        let Some(next_result) = replace_next_fractional_exponent(&result) else {
            break;
        };
        result = next_result;
    }

    result = result.replace("\\cdot", " · ");
    result = result.replace("\\left", "");
    result = result.replace("\\right", "");
    result = result.replace('{', "");
    result = result.replace('}', "");

    result.replace("\\", "")
}

fn replace_next_parenthesized_power_base(value: &str) -> Option<String> {
    let mut open_braces = Vec::new();
    let chars: Vec<(usize, char)> = value.char_indices().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let (byte_index, ch) = chars[index];
        match ch {
            '{' => open_braces.push(byte_index),
            '}' => {
                let Some(base_start) = open_braces.pop() else {
                    index += 1;
                    continue;
                };
                let after_close = &value[byte_index + 1..];
                if !after_close.starts_with("^{") {
                    index += 1;
                    continue;
                }

                let exponent_rest = &value[byte_index + 2..];
                let (_, exponent_end) = find_balanced_braces(exponent_rest)?;

                let base = &value[base_start + 1..byte_index];
                if !needs_parenthesized_power_base(base) {
                    index += 1;
                    continue;
                }

                let exponent_close = byte_index + 2 + exponent_end;
                let replacement = format!("({base})^{}", &value[byte_index + 2..=exponent_close]);
                return Some(format!(
                    "{}{}{}",
                    &value[..base_start],
                    replacement,
                    &value[exponent_close + 1..]
                ));
            }
            _ => {}
        }
        index += 1;
    }

    None
}

fn replace_next_fractional_exponent(value: &str) -> Option<String> {
    let start = value.find("^{")?;
    let exponent_rest = &value[start + 1..];
    let (exponent, exponent_end) = find_balanced_braces(exponent_rest)?;
    if !exponent.contains('/') {
        return None;
    }
    let exponent_close = start + 1 + exponent_end;
    let replacement = format!("^({exponent})");
    Some(format!(
        "{}{}{}",
        &value[..start],
        replacement,
        &value[exponent_close + 1..]
    ))
}

fn needs_parenthesized_power_base(base: &str) -> bool {
    base.chars()
        .any(|ch| matches!(ch, ' ' | '+' | '-' | '·' | '/' | '\\'))
}

fn find_balanced_braces(s: &str) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut content = String::new();

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => {
                if depth > 0 {
                    content.push(ch);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some((content, i));
                }
                content.push(ch);
            }
            _ => {
                if depth > 0 {
                    content.push(ch);
                }
            }
        }
    }

    None
}
