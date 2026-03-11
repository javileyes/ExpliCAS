use crate::cas_solver::Step;
use cas_ast::Context;
use cas_formatter::{
    DisplayContext, HighlightColor, HighlightConfig, LaTeXExprHighlightedWithHints,
    StylePreferences,
};

pub(super) fn render_local_change_latex(
    context: &Context,
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let focus_before = step.before_local().unwrap_or(step.before);
    let focus_after = step.after_local().unwrap_or(step.after);

    let mut rule_before_config = HighlightConfig::new();
    rule_before_config.add(focus_before, HighlightColor::Red);
    let local_before_colored = LaTeXExprHighlightedWithHints {
        context,
        id: focus_before,
        highlights: &rule_before_config,
        hints: display_hints,
        style_prefs: Some(style_prefs),
    }
    .to_latex();

    let mut rule_after_config = HighlightConfig::new();
    rule_after_config.add(focus_after, HighlightColor::Green);
    let local_after_colored = LaTeXExprHighlightedWithHints {
        context,
        id: focus_after,
        highlights: &rule_after_config,
        hints: display_hints,
        style_prefs: Some(style_prefs),
    }
    .to_latex();

    format!(
        "{} \\rightarrow {}",
        local_before_colored, local_after_colored
    )
}
