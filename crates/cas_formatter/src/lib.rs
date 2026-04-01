//! Formatting facade for AST rendering concerns.
//!
//! This crate centralizes display/LaTeX APIs so callers can avoid importing
//! presentation modules directly from `cas_ast`.

pub use cas_ast::{eq, expr_path, hold, ordering, views};
pub use cas_ast::{Constant, Context, Expr, ExprId};

pub mod conditions;
pub mod display;
pub mod display_clean;
pub mod display_context;
pub mod display_hint_builder;
pub mod display_transforms;
pub mod escape;
pub mod latex;
pub mod latex_clean;
pub mod latex_core;
pub mod latex_highlight;
pub mod latex_no_roots;
pub mod path;
pub mod root_style;
pub mod rule_scope;
pub mod visualizer;

pub use conditions::{
    condition_predicate_to_display, condition_predicate_to_latex, condition_set_to_display,
    condition_set_to_latex,
};
pub use display::{DisplayExpr, DisplayExprStyled, DisplayExprWithHints, RawDisplayExpr};
pub use display_clean::{clean_display_string, clean_sign_patterns};
pub use display_context::{DisplayContext, DisplayHint};
pub use display_hint_builder::{
    build_display_context, build_display_context_with_result, DisplayStepLike,
};
pub use display_transforms::{
    DisplayTransform, DisplayTransformRegistry, ScopeTag, ScopedRenderer,
};
pub use escape::{html_escape, latex_escape};
pub use latex::{LaTeXExpr, LaTeXExprStyled, LaTeXExprWithHints};
pub use latex_clean::clean_latex_identities;
pub use latex_core::PathHighlightedLatexRenderer;
pub use latex_highlight::{
    HighlightColor, HighlightConfig, LaTeXExprHighlighted, LaTeXExprHighlightedWithHints,
    PathHighlightConfig,
};
pub use latex_no_roots::LatexNoRoots;
pub use path::{
    diff_find_all_paths_to_expr, diff_find_path_to_expr, diff_find_paths_by_structure,
    extract_add_terms, find_path_to_expr, navigate_to_subexpr,
};
pub use root_style::{detect_root_style, ParseStyleSignals, RootStyle, StylePreferences};
pub use rule_scope::render_with_rule_scope;
pub use visualizer::AstVisualizer;

/// Render one expression id to display text.
pub fn render_expr(context: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context, id })
}

#[cfg(test)]
mod tests {
    use super::render_expr;
    use cas_ast::{Context, Expr};

    #[test]
    fn render_expr_renders_basic_expression() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert_eq!(render_expr(&ctx, x), "x");
    }

    #[test]
    fn render_expr_displays_mul_with_negative_factor_as_subtraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let three = ctx.num(3);
        let a3 = ctx.add(Expr::Pow(a, three));
        let neg_c = ctx.add(Expr::Neg(c));
        let left = ctx.add(Expr::Mul(a3, b));
        let right = ctx.add(Expr::Mul(a3, neg_c));
        let expr = ctx.add(Expr::Add(left, right));
        assert_eq!(render_expr(&ctx, expr), "b * a^3 - c * a^3");
    }

    #[test]
    fn render_expr_does_not_group_multiple_negative_products() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let three = ctx.num(3);
        let a3 = ctx.add(Expr::Pow(a, three));
        let b3 = ctx.add(Expr::Pow(b, three));
        let c3 = ctx.add(Expr::Pow(c, three));
        let positives = [
            ctx.add(Expr::Mul(a, c3)),
            ctx.add(Expr::Mul(b, a3)),
            ctx.add(Expr::Mul(c, b3)),
        ];
        let neg_ab3_inner = ctx.add(Expr::Mul(a, b3));
        let neg_ca3_inner = ctx.add(Expr::Mul(c, a3));
        let neg_c3b_inner = ctx.add(Expr::Mul(c3, b));
        let negatives = [
            ctx.add(Expr::Neg(neg_ab3_inner)),
            ctx.add(Expr::Neg(neg_ca3_inner)),
            ctx.add(Expr::Neg(neg_c3b_inner)),
        ];
        let pos_tail = ctx.add(Expr::Add(positives[1], positives[2]));
        let pos_sum = ctx.add(Expr::Add(positives[0], pos_tail));
        let neg_tail = ctx.add(Expr::Add(negatives[1], negatives[2]));
        let neg_sum = ctx.add(Expr::Add(negatives[0], neg_tail));
        let expr = ctx.add(Expr::Add(pos_sum, neg_sum));
        let rendered = render_expr(&ctx, expr);
        assert!(
            !rendered.contains(" - ("),
            "unexpected grouped subtraction: {rendered}"
        );
        assert!(
            rendered.contains(" - a * b^3"),
            "missing direct subtraction: {rendered}"
        );
        assert!(
            rendered.contains(" - c * a^3"),
            "missing direct subtraction: {rendered}"
        );
        assert!(
            rendered.contains(" - b * c^3") || rendered.contains(" - c^3 * b"),
            "missing direct subtraction: {rendered}"
        );
    }
}
