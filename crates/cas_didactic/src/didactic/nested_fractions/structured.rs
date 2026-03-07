use super::super::nested_fraction_analysis::{
    extract_combined_fraction_str, NestedFractionPattern,
};
use super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_structured_nested_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    pattern: NestedFractionPattern,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    match pattern {
        NestedFractionPattern::OneOverSumWithUnitFraction
        | NestedFractionPattern::OneOverSumWithFraction => {
            generate_one_over_sum_substeps(ctx, before_expr, after_expr, hints)
        }
        NestedFractionPattern::FractionOverSumWithFraction => {
            generate_fraction_over_sum_substeps(ctx, before_expr, after_expr, hints)
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            generate_sum_over_scalar_substeps(ctx, before_expr, after_expr, hints)
        }
        NestedFractionPattern::General => Vec::new(),
    }
}

fn generate_one_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(_, den) = ctx.get(before_expr) {
        let den_str = nested_fraction_latex(ctx, hints, *den);
        let intermediate_str = extract_combined_fraction_str(ctx, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del denominador (denominador común)".to_string(),
            before_expr: den_str.clone(),
            after_expr: intermediate_str.clone(),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: "Invertir la fracción: 1/(a/b) = b/a".to_string(),
            before_expr: format!("\\frac{{1}}{{{}}}", intermediate_str),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}

fn generate_fraction_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);
        let den_str = nested_fraction_latex(ctx, hints, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del denominador (denominador común)".to_string(),
            before_expr: den_str,
            after_expr: extract_combined_fraction_str(ctx, *den),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: format!("Multiplicar {} por el denominador interno", num_str),
            before_expr: nested_fraction_latex(ctx, hints, before_expr),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}

fn generate_sum_over_scalar_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);
        let den_str = nested_fraction_latex(ctx, hints, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del numerador (denominador común)".to_string(),
            before_expr: num_str,
            after_expr: "(numerador combinado) / B".to_string(),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: format!("Dividir por {}: multiplicar denominadores", den_str),
            before_expr: nested_fraction_latex(ctx, hints, before_expr),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
