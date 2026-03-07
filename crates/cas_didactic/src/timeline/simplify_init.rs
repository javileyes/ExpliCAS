use cas_ast::{Context, ExprId};
use cas_formatter::{PathHighlightConfig, PathHighlightedLatexRenderer};
use cas_solver::{infer_implicit_domain, ImplicitCondition, Step, ValueDomain};

pub(super) struct SimplifyTimelineInit {
    pub title: String,
    pub global_requires: Vec<ImplicitCondition>,
    pub style_prefs: cas_formatter::root_style::StylePreferences,
}

pub(super) fn build_simplify_timeline_init(
    context: &mut Context,
    _steps: &[Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    input_string: Option<&str>,
) -> SimplifyTimelineInit {
    let signals = input_string.map(cas_formatter::root_style::ParseStyleSignals::from_input_string);
    let style_prefs = cas_formatter::root_style::StylePreferences::from_expression_with_signals(
        context,
        original_expr,
        signals.as_ref(),
    );

    let empty_config = PathHighlightConfig::new();
    let title = PathHighlightedLatexRenderer {
        context,
        id: original_expr,
        path_highlights: &empty_config,
        hints: None,
        style_prefs: Some(&style_prefs),
    }
    .to_latex();

    let domain_expr = simplified_result.unwrap_or(original_expr);
    let input_domain = infer_implicit_domain(context, domain_expr, ValueDomain::RealOnly);
    let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();

    SimplifyTimelineInit {
        title,
        global_requires,
        style_prefs,
    }
}
