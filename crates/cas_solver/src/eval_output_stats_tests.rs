#[cfg(test)]
mod tests {
    use crate::eval_output_stats::{
        expr_output_hash, expr_output_stats, format_limited_output_expr,
    };

    #[test]
    fn format_limited_output_expr_truncates_when_needed() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + x + x + x + x", &mut ctx).expect("parse");
        let (rendered, truncated, original_len) = format_limited_output_expr(&ctx, expr, 5);
        assert!(truncated);
        assert!(rendered.contains("<truncated>"));
        assert!(original_len > 5);
    }

    #[test]
    fn expr_output_hash_is_stable_for_same_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x^2 + 1", &mut ctx).expect("parse");
        let h1 = expr_output_hash(&ctx, expr);
        let h2 = expr_output_hash(&ctx, expr);
        assert_eq!(h1, h2);

        let stats = expr_output_stats(&ctx, expr);
        assert!(stats.node_count >= 1);
    }
}
