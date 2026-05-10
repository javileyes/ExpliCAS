use cas_ast::{Context, ExprId};
use cas_formatter::{LaTeXExprStyled, ParseStyleSignals, RootStyle, StylePreferences};

#[derive(Clone, Copy)]
pub(crate) enum EvalLatexRenderIntent {
    Input,
    Result,
}

fn style_for_eval_intent(
    ctx: &Context,
    id: ExprId,
    signals: &ParseStyleSignals,
    intent: EvalLatexRenderIntent,
) -> StylePreferences {
    let mut style = StylePreferences::from_expression(ctx, id);

    match intent {
        // Preserve explicit typed fractional powers in the input, while keeping
        // explicit sqrt(...) calls as roots because they remain Function nodes.
        EvalLatexRenderIntent::Input => {
            if signals.saw_caret_fraction > 0 {
                style.root_style = RootStyle::Exponential;
            } else if signals.saw_sqrt_token > 0 {
                style.root_style = RootStyle::Radical;
            }
        }
        // Normalize mixed root/power inputs to radical style in the result so
        // the output does not keep a mixed "smell".
        EvalLatexRenderIntent::Result => {
            if signals.saw_sqrt_token > 0 {
                style.root_style = RootStyle::Radical;
            } else if signals.saw_caret_fraction > 0 {
                style.root_style = RootStyle::Exponential;
            }
        }
    }

    style
}

pub(crate) fn render_expr_latex_for_eval(
    ctx: &Context,
    id: ExprId,
    signals: &ParseStyleSignals,
    intent: EvalLatexRenderIntent,
) -> String {
    let style = style_for_eval_intent(ctx, id, signals, intent);
    let latex = LaTeXExprStyled {
        context: ctx,
        id,
        style_prefs: &style,
    }
    .to_latex();

    match intent {
        EvalLatexRenderIntent::Input => latex,
        EvalLatexRenderIntent::Result => {
            crate::pipeline_display::compact_subtracted_difference_display(latex)
        }
    }
}
