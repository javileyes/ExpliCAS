#[cfg(test)]
mod tests {
    use crate::clean_result_output_line;

    #[test]
    fn clean_result_output_line_normalizes_last_result() {
        let mut lines = vec!["Result: ((x))".to_string()];
        clean_result_output_line(&mut lines);
        assert!(lines[0].starts_with("Result: "));
    }
}
