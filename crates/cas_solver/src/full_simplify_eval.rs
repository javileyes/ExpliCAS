mod error;
mod eval;
mod types;

pub use error::{format_full_simplify_eval_error_message, FullSimplifyEvalError};
pub use eval::evaluate_full_simplify_input_with_resolver;
pub use types::FullSimplifyEvalOutput;

#[cfg(test)]
mod tests {
    use super::{evaluate_full_simplify_input_with_resolver, FullSimplifyEvalError};

    #[test]
    fn evaluate_full_simplify_input_parse_error_is_typed() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let simplify_options = crate::SimplifyOptions::default();
        let err = evaluate_full_simplify_input_with_resolver(
            &mut simplifier,
            "x+",
            true,
            simplify_options,
            |_ctx, expr| Ok(expr),
        )
        .expect_err("parse error");
        assert!(matches!(err, FullSimplifyEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_full_simplify_input_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let simplify_options = crate::SimplifyOptions::default();
        let out = evaluate_full_simplify_input_with_resolver(
            &mut simplifier,
            "x + 0",
            true,
            simplify_options,
            |_ctx, expr| Ok(expr),
        )
        .expect("full simplify");
        let shown = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert_eq!(shown, "x");
    }
}
