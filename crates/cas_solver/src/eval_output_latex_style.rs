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
        // El RESULTADO va siempre en radical (decisión del usuario 2026-07-29,
        // por consistencia). Antes ecoaba el estilo del input, y eso hacía que
        // resultados que el alumno reconoce de memoria salieran en la forma
        // menos reconocible cuando NADIE había escrito una raíz:
        // `integrate(e^(-x^2), x, -oo, oo)` daba `pi^(1/2)` en vez de `sqrt(pi)`
        // y la fórmula cuadrática, `(b^2 - 4·a·c)^(1/2)`.
        // El eco fiel del input se conserva donde toca: en la CABECERA.
        EvalLatexRenderIntent::Result => {
            style.root_style = RootStyle::Radical;
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
