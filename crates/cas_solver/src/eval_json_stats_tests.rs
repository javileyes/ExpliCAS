#[cfg(test)]
mod tests {
    use crate::eval_json_stats::{
        expr_hash_eval_json, expr_stats_eval_json, format_expr_limited_eval_json,
    };

    #[test]
    fn format_expr_limited_eval_json_truncates_when_needed() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + x + x + x + x", &mut ctx).expect("parse");
        let (rendered, truncated, original_len) = format_expr_limited_eval_json(&ctx, expr, 5);
        assert!(truncated);
        assert!(rendered.contains("<truncated>"));
        assert!(original_len > 5);
    }

    #[test]
    fn expr_hash_eval_json_is_stable_for_same_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x^2 + 1", &mut ctx).expect("parse");
        let h1 = expr_hash_eval_json(&ctx, expr);
        let h2 = expr_hash_eval_json(&ctx, expr);
        assert_eq!(h1, h2);

        let stats = expr_stats_eval_json(&ctx, expr);
        assert!(stats.node_count >= 1);
    }
}
