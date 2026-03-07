/// Convert LaTeX-like notation into plain-text form for CLI display.
pub fn latex_to_plain_text(s: &str) -> String {
    let mut result = s.to_string();

    result = result.replace("\\cdot", " · ");

    while let Some(start) = result.find("\\text{") {
        if let Some(end) = result[start + 6..].find('}') {
            let content = &result[start + 6..start + 6 + end];
            result = format!(
                "{}{}{}",
                &result[..start],
                content,
                &result[start + 7 + end..]
            );
        } else {
            break;
        }
    }

    let mut iterations = 0;
    while result.contains("\\frac{") && iterations < 10 {
        iterations += 1;
        if let Some(start) = result.rfind("\\frac{") {
            let rest = &result[start + 5..];
            if let Some((numer, numer_end)) = find_balanced_braces(rest) {
                let after_numer = &rest[numer_end + 1..];
                if after_numer.starts_with('{') {
                    if let Some((denom, denom_end)) = find_balanced_braces(after_numer) {
                        let total_end = start + 5 + numer_end + 1 + denom_end + 1;
                        let replacement = format!("({}/{})", numer, denom);
                        result = format!(
                            "{}{}{}",
                            &result[..start],
                            replacement,
                            &result[total_end..]
                        );
                        continue;
                    }
                }
            }
        }
        break;
    }

    result.replace("\\", "")
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
