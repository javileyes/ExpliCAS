#[cfg(test)]
mod tests {
    use crate::history_metadata_format::format_history_eval_metadata_sections;

    #[test]
    fn format_history_eval_metadata_sections_returns_empty_when_no_payloads() {
        let ctx = cas_ast::Context::new();
        let lines = format_history_eval_metadata_sections(&ctx, &[], &[], &[]);
        assert!(lines.is_empty());
    }
}
