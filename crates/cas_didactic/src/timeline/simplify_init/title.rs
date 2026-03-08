use cas_ast::{Context, ExprId};
use cas_formatter::{PathHighlightConfig, PathHighlightedLatexRenderer};

pub(super) fn build_timeline_title_and_style_prefs(
    context: &mut Context,
    original_expr: ExprId,
    input_string: Option<&str>,
) -> (String, cas_formatter::root_style::StylePreferences) {
    let signals = input_string.map(cas_formatter::root_style::ParseStyleSignals::from_input_string);
    let style_prefs = cas_formatter::root_style::StylePreferences::from_expression_with_signals(
        context,
        original_expr,
        signals.as_ref(),
    );

    let title = PathHighlightedLatexRenderer {
        context,
        id: original_expr,
        path_highlights: &PathHighlightConfig::new(),
        hints: None,
        style_prefs: Some(&style_prefs),
    }
    .to_latex();

    (title, style_prefs)
}
