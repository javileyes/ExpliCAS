mod analysis;

use self::analysis::{analyze_root_denesting, compute_denesting_delta};
use super::SubStep;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

/// Generate sub-steps explaining root denesting process.
pub(crate) fn generate_root_denesting_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_formatter::DisplayContext;
    use cas_formatter::LaTeXExprWithHints;

    let mut sub_steps = Vec::new();
    let before_expr = step.before_local().unwrap_or(step.before);
    let hints = DisplayContext::with_root_index(2);

    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    let analysis = match analyze_root_denesting(ctx, before_expr) {
        Some(analysis) => analysis,
        None => return sub_steps,
    };

    let a_str = to_latex(analysis.a_expr);
    let d_str = to_latex(analysis.d_expr);
    let c_str = format_rational_latex(&analysis.c_coeff);

    sub_steps.push(SubStep {
        description: "Identificar la forma √(a ± c·√d)".to_string(),
        before_expr: to_latex(before_expr),
        after_expr: if analysis.is_add {
            format!("a = {}, \\quad c = {}, \\quad d = {}", a_str, c_str, d_str)
        } else {
            format!("a = {}, \\quad c = -{}, \\quad d = {}", a_str, c_str, d_str)
        },
        before_latex: None,
        after_latex: None,
    });

    if let Some(delta) = compute_denesting_delta(ctx, &analysis) {
        let delta_str = format_rational_latex(&delta);

        sub_steps.push(SubStep {
            description: "Calcular Δ = a² - c²d".to_string(),
            before_expr: format!("({})^2 - ({})^2 \\cdot {}", a_str, c_str, d_str),
            after_expr: delta_str,
            before_latex: None,
            after_latex: None,
        });

        if delta.is_integer() && delta.to_integer() >= num_bigint::BigInt::from(0) {
            sub_steps.push(SubStep {
                description: "Δ es cuadrado perfecto: aplicar desanidación".to_string(),
                before_expr: format!("\\sqrt{{{}}}", to_latex(analysis.inner_expr)),
                after_expr: to_latex(step.after_local().unwrap_or(step.after)),
                before_latex: None,
                after_latex: None,
            });
        }
    }

    sub_steps
}

fn format_rational_latex(value: &num_rational::BigRational) -> String {
    if value.is_integer() {
        format!("{}", value.to_integer())
    } else {
        format!("\\frac{{{}}}{{{}}}", value.numer(), value.denom())
    }
}
