#[cfg(test)]
mod tests {
    use crate::format_history_eval_metadata_sections;

    #[test]
    fn format_history_eval_metadata_sections_returns_empty_when_no_payloads() {
        let ctx = cas_ast::Context::new();
        let lines = format_history_eval_metadata_sections(&ctx, &[], &[], &[]);
        assert!(lines.is_empty());
    }

    #[test]
    fn format_history_eval_metadata_sections_groups_repeated_blocked_conditions() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let hints = vec![
            crate::BlockedHint {
                key: cas_solver_core::assumption_model::AssumptionKey::nonzero_key(&ctx, x),
                expr_id: x,
                rule: "Cancel Identical Numerator/Denominator".to_string(),
                suggestion: "use `domain generic` to allow definability assumptions",
            },
            crate::BlockedHint {
                key: cas_solver_core::assumption_model::AssumptionKey::nonzero_key(&ctx, x),
                expr_id: x,
                rule: "Simplify Nested Fraction".to_string(),
                suggestion: "use `domain generic` to allow definability assumptions",
            },
            crate::BlockedHint {
                key: cas_solver_core::assumption_model::AssumptionKey::nonzero_key(&ctx, x),
                expr_id: x,
                rule: "Cancel Common Factors".to_string(),
                suggestion: "use `domain generic` to allow definability assumptions",
            },
        ];

        let lines = format_history_eval_metadata_sections(&ctx, &[], &[], &hints);

        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains("requires x ≠ 0"))
                .count(),
            1,
            "history blocked metadata should group repeated conditions: {lines:?}"
        );
        let grouped_line = lines
            .iter()
            .find(|line| line.contains("requires x ≠ 0"))
            .expect("grouped blocked hint line");
        assert!(grouped_line.contains("Cancel Identical Numerator/Denominator"));
        assert!(grouped_line.contains("Simplify Nested Fraction"));
        assert!(grouped_line.contains("Cancel Common Factors"));
        assert_eq!(
            lines
                .iter()
                .filter(|line| {
                    line.contains("use `domain generic` to allow definability assumptions")
                })
                .count(),
            1,
            "history blocked metadata should show the repeated tip once: {lines:?}"
        );
    }
}
