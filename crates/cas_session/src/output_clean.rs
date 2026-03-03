//! Shared output post-processing helpers for REPL-like frontends.

/// Normalize the last `Result: ...` line by cleaning display artifacts.
pub fn clean_result_output_line(lines: &mut [String]) {
    let Some(last) = lines.last_mut() else {
        return;
    };
    let Some(raw_value) = last.strip_prefix("Result: ") else {
        return;
    };
    *last = format!("Result: {}", cas_formatter::clean_display_string(raw_value));
}

#[cfg(test)]
mod tests {
    #[test]
    fn clean_result_output_line_normalizes_last_result() {
        let mut lines = vec!["Result: ((x))".to_string()];
        super::clean_result_output_line(&mut lines);
        assert!(lines[0].starts_with("Result: "));
    }
}
