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
        // El RESULTADO ECOA la notación del INPUT (decisión del usuario 2026-07-29):
        // raíces si el usuario escribió raíces, potencias fraccionarias si escribió
        // potencias fraccionarias. En la MEZCLA gana la raíz — al usuario le da igual
        // cuál, y así el caso mixto tiene una respuesta y no dos.
        //
        // Cuando el input no trae NINGUNA de las dos notaciones no hay nada que ecoar
        // y se presenta en RADICAL: son los casos que el usuario señaló como los peor
        // impresos — `integrate(e^(-x^2), x, -oo, oo)` es `√π`, no `pi^(1/2)`, y la
        // fórmula cuadrática lleva `√(b²−4ac)`.
        //
        // Ese tercer caso es justo el que hay que decidir a mano: preguntarle al ÁRBOL
        // no vale, porque el resultado ya está convertido a `Pow` y responde «potencia»
        // pase lo que pase en la entrada. Sniffearlo era lo que imprimía la integral de
        // Gauss como potencia.
        EvalLatexRenderIntent::Result => {
            style.root_style = if signals.saw_caret_fraction > 0 && signals.saw_sqrt_token == 0 {
                RootStyle::Exponential
            } else {
                RootStyle::Radical
            };
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
