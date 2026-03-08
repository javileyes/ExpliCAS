mod frac;
mod text;

/// Convert LaTeX-like notation into plain-text form for CLI display.
pub fn latex_to_plain_text(s: &str) -> String {
    let mut result = s.to_string();

    result = result.replace("\\cdot", " · ");
    result = text::strip_text_wrappers(result);

    let mut iterations = 0;
    while result.contains("\\frac{") && iterations < 10 {
        iterations += 1;
        let Some(next_result) = frac::replace_last_fraction(&result) else {
            break;
        };
        result = next_result;
    }

    result.replace("\\", "")
}
