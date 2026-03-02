use crate::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};
use crate::root_style::StylePreferences;
use crate::{Context, ExprId};

/// Render an expression with rule-scoped display transforms.
///
/// Used for per-step rendering where certain rules should alter presentation
/// (e.g. showing square roots instead of fractional exponents).
pub fn render_with_rule_scope(
    context: &Context,
    id: ExprId,
    rule_name: &str,
    style_prefs: &StylePreferences,
) -> String {
    let scopes: Vec<ScopeTag> = match rule_name {
        "Quadratic Formula" => vec![ScopeTag::Rule("QuadraticFormula")],
        _ => vec![],
    };

    let registry = DisplayTransformRegistry::with_defaults();
    let renderer = ScopedRenderer::new(context, &scopes, &registry, style_prefs);
    renderer.render(id)
}

#[cfg(test)]
mod tests {
    use super::render_with_rule_scope;
    use crate::root_style::{ParseStyleSignals, StylePreferences};
    use crate::{Context, Expr};

    #[test]
    fn render_with_rule_scope_falls_back_for_unknown_rule() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(x, one));
        let signals = ParseStyleSignals::from_input_string("x+1");
        let prefs = StylePreferences::from_expression_with_signals(&ctx, expr, Some(&signals));
        let rendered = render_with_rule_scope(&ctx, expr, "Unknown", &prefs);
        assert!(!rendered.is_empty());
    }
}
