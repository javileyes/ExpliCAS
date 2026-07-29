#[cfg(test)]
mod tests {
    use crate::eval_output_stats::{
        expr_output_hash, expr_output_stats, format_limited_output_expr,
    };

    #[test]
    fn format_limited_output_expr_truncates_when_needed() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + x + x + x + x", &mut ctx).expect("parse");
        let signals = cas_formatter::ParseStyleSignals::from_input_string("x + x + x + x + x");
        let (rendered, truncated, original_len) =
            format_limited_output_expr(&ctx, expr, 5, &signals);
        assert!(truncated);
        assert!(rendered.contains("<truncated>"));
        assert!(original_len > 5);
    }

    /// El texto plano del resultado usa la MISMA regla de eco que su LaTeX: quien
    /// escribe una raíz recibe una raíz también aquí. Era la superficie que
    /// quedaba diciendo `(3 + 4·i)^(1/2)` mientras el LaTeX decía `\sqrt{3 + 4i}`.
    #[test]
    fn format_limited_output_expr_echoes_the_root_notation_of_the_input() {
        // El resultado es siempre el mismo árbol (`x^(1/2)`); lo que cambia es lo
        // que el usuario ESCRIBIÓ, y con ello la notación que recibe de vuelta.
        for input in [
            "sqrt(x)",
            "root(x + 1, 3)",
            // Sin notación en la entrada: radical, como el LaTeX.
            "integrate(e^(-x^2), x, -oo, oo)",
        ] {
            let mut ctx = cas_ast::Context::new();
            let expr = cas_parser::parse("x^(1/2)", &mut ctx).expect("parse");
            let signals = cas_formatter::ParseStyleSignals::from_input_string(input);
            let (rendered, _, _) = format_limited_output_expr(&ctx, expr, 200, &signals);
            assert_eq!(rendered, "sqrt(x)", "eco roto para el input {input}");
        }

        // Y quien escribe potencias fraccionarias las recibe.
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x^(1/2)", &mut ctx).expect("parse");
        let signals = cas_formatter::ParseStyleSignals::from_input_string("x^(1/2)");
        let (rendered, _, _) = format_limited_output_expr(&ctx, expr, 200, &signals);
        assert_eq!(rendered, "x^(1 / 2)");
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
