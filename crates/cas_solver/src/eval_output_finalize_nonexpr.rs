use cas_api_models::ExprStatsWire;
use cas_ast::{Context, SolutionSet};

use crate::eval_output_finalize::{build_eval_output, EvalOutputResultPayload, EvalOutputWire};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;
use crate::eval_output_presentation::{format_output_solution_set, solution_set_to_output_latex};

fn build_nonexpr_result_payload(
    result: String,
    result_latex: Option<String>,
) -> EvalOutputResultPayload {
    EvalOutputResultPayload {
        result_chars: result.len(),
        result,
        result_truncated: false,
        result_latex,
        stats: ExprStatsWire::default(),
        hash: None,
    }
}

pub(crate) fn finalize_solution_set_output(
    ctx: &Context,
    solution_set: &SolutionSet,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    // El texto del conjunto solución ecoa la notación del input igual que el de una
    // expresión suelta y que su propio LaTeX: `solve(a·x²+b·x+c=0,x)` imprimía la
    // fórmula cuadrática con `(b^2 - 4·a·c)^(1/2)` mientras su LaTeX ya decía
    // `\sqrt{…}`. La reescritura se hace sobre un contexto de SCRATCH y sobre el
    // conjunto ENTERO (`map_exprs`), para que soluciones, extremos de intervalo,
    // periodos, residuales y condiciones de cada caso hablen la misma notación.
    let (scratch, rewritten);
    let (render_ctx, render_set) =
        match crate::eval_output_latex_style::result_root_style(&shared.style_signals) {
            cas_formatter::RootStyle::Exponential => (ctx, solution_set),
            cas_formatter::RootStyle::Radical | cas_formatter::RootStyle::Auto => {
                let mut work = ctx.clone();
                let mapped = solution_set.map_exprs(&mut |id| {
                    cas_formatter::root_display_rewrite::rewrite_fractional_powers_as_roots(
                        &mut work, id,
                    )
                });
                scratch = work;
                rewritten = mapped;
                (&scratch, &rewritten)
            }
        };

    let result_str = format_output_solution_set(render_ctx, render_set);
    // El LaTeX recibe el MISMO estilo de eco que acaba de decidir el texto: sin
    // enhebrarlo, este camino renderizaba con el default (radical incondicional)
    // y las dos superficies del mismo conjunto divergían.
    let style = cas_formatter::StylePreferences::with_root_style(
        crate::eval_output_latex_style::result_root_style(&shared.style_signals),
    );
    let result_latex = solution_set_to_output_latex(ctx, solution_set, &style);
    let steps_count = shared.primary_steps_count();
    build_eval_output(
        build_nonexpr_result_payload(result_str, Some(result_latex)),
        steps_count,
        shared,
    )
}

pub(crate) fn finalize_bool_output(
    value: bool,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    let result_str = value.to_string();
    let steps_count = shared.primary_steps_count();
    build_eval_output(
        build_nonexpr_result_payload(result_str, None),
        steps_count,
        shared,
    )
}

pub(crate) fn finalize_text_output(
    plain: &str,
    latex: Option<&str>,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    let steps_count = shared.primary_steps_count();
    build_eval_output(
        build_nonexpr_result_payload(plain.to_string(), latex.map(str::to_string)),
        steps_count,
        shared,
    )
}
