use super::{rationalization_display, rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_product_rationalization_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before) {
        let num_plain = rationalization_display(ctx, hints, *num);
        let num_latex = rationalization_latex(ctx, hints, *num);
        let den_plain = rationalization_display(ctx, hints, *den);
        let den_latex = rationalization_latex(ctx, hints, *den);

        // The audit found this emitter publishing LaTeX in the PLAIN holes and
        // leaving the latex holes empty — the JSON layer's fallback made the
        // web render fine while the CLI reader saw `\frac{...}`. Each surface
        // now fills its own hole, and the titles go through their i18n keys
        // (they existed in the locale table all along; the emitter predated
        // `SubStep::keyed`).
        sub_steps.push(
            SubStep::keyed(
                "rationalize.denominator_radical_product",
                vec![],
                format!("({num_plain})/({den_plain})"),
                "a/(k · sqrt(n))",
            )
            .with_before_latex(format!("\\frac{{{num_latex}}}{{{den_latex}}}"))
            .with_after_latex("\\frac{a}{k \\cdot \\sqrt{n}}"),
        );

        if let Expr::Div(new_num, new_den) = ctx.get(after) {
            let after_num_plain = rationalization_display(ctx, hints, *new_num);
            let after_num_latex = rationalization_latex(ctx, hints, *new_num);
            let after_den_plain = rationalization_display(ctx, hints, *new_den);
            let after_den_latex = rationalization_latex(ctx, hints, *new_den);
            sub_steps.push(
                SubStep::keyed(
                    "rationalize.multiply_by_root_over_root",
                    vec![],
                    format!("({num_plain} · sqrt(n))/({den_plain} · sqrt(n))"),
                    format!("({after_num_plain})/({after_den_plain})"),
                )
                .with_before_latex(format!(
                    "\\frac{{{num_latex} \\cdot \\sqrt{{n}}}}{{{den_latex} \\cdot \\sqrt{{n}}}}"
                ))
                .with_after_latex(format!("\\frac{{{after_num_latex}}}{{{after_den_latex}}}")),
            );
        }
    }

    sub_steps
}
