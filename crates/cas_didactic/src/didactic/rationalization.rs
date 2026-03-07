use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;
use num_traits::Signed;

use super::SubStep;

/// Generate sub-steps explaining rationalization process.
/// Uses LaTeXExprWithHints for proper sqrt notation rendering.
pub(crate) fn generate_rationalization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_formatter::DisplayContext;
    use cas_formatter::LaTeXExprWithHints;

    let mut sub_steps = Vec::new();
    let hints = DisplayContext::with_root_index(2);

    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
        let mut terms = Vec::new();
        collect_add_terms_recursive(ctx, expr, &mut terms);
        terms
    }

    fn collect_add_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                collect_add_terms_recursive(ctx, *l, terms);
                collect_add_terms_recursive(ctx, *r, terms);
            }
            _ => terms.push(expr),
        }
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if step.description.contains("group") {
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);
            let den_terms = collect_add_terms(ctx, *den);

            if den_terms.len() >= 3 {
                let group_terms: Vec<String> = den_terms[..den_terms.len() - 1]
                    .iter()
                    .map(|t| to_latex(*t))
                    .collect();
                let last_term = to_latex(den_terms[den_terms.len() - 1]);
                let group_str = group_terms.join(" + ");

                sub_steps.push(SubStep {
                    description: "Agrupar términos del denominador".to_string(),
                    before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                    after_expr: if group_terms.len() > 1 {
                        format!("\\frac{{{}}}{{({}) + {}}}", num_latex, group_str, last_term)
                    } else {
                        format!("\\frac{{{}}}{{{} + {}}}", num_latex, group_str, last_term)
                    },
                    before_latex: None,
                    after_latex: None,
                });

                let conjugate = if group_terms.len() > 1 {
                    format!("({}) - {}", group_str, last_term)
                } else {
                    format!("{} - {}", group_str, last_term)
                };

                sub_steps.push(SubStep {
                    description: "Multiplicar por el conjugado".to_string(),
                    before_expr: if group_terms.len() > 1 {
                        format!("({}) + {}", group_str, last_term)
                    } else {
                        format!("{} + {}", group_str, last_term)
                    },
                    after_expr: conjugate.clone(),
                    before_latex: None,
                    after_latex: None,
                });

                if let Expr::Div(_new_num, new_den) = ctx.get(after) {
                    let after_den_latex = to_latex(*new_den);
                    sub_steps.push(SubStep {
                        description: "Diferencia de cuadrados".to_string(),
                        before_expr: if group_terms.len() > 1 {
                            format!("({})^2 - ({})^2", group_str, last_term)
                        } else {
                            format!("{}^2 - {}^2", group_str, last_term)
                        },
                        after_expr: after_den_latex,
                        before_latex: None,
                        after_latex: None,
                    });
                }
            }
        }
    } else if step.description.contains("product") {
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            sub_steps.push(SubStep {
                description: "Denominador con producto de radical".to_string(),
                before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                after_expr: "\\frac{a}{k \\cdot \\sqrt{n}}".to_string(),
                before_latex: None,
                after_latex: None,
            });

            if let Expr::Div(new_num, new_den) = ctx.get(after) {
                let after_num_latex = to_latex(*new_num);
                let after_den_latex = to_latex(*new_den);
                sub_steps.push(SubStep {
                    description: "Multiplicar por \\sqrt{n}/\\sqrt{n}".to_string(),
                    before_expr: format!(
                        "\\frac{{{} \\cdot \\sqrt{{n}}}}{{{} \\cdot \\sqrt{{n}}}}",
                        num_latex, den_latex
                    ),
                    after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                    before_latex: None,
                    after_latex: None,
                });
            }
        }
    } else if let Expr::Div(num, den) = ctx.get(before) {
        let num_latex = to_latex(*num);
        let den_latex = to_latex(*den);

        let (term_a, term_b, is_original_minus) = match ctx.get(*den) {
            Expr::Add(l, r) => match ctx.get(*r) {
                Expr::Neg(inner) => (to_latex(*l), to_latex(*inner), true),
                Expr::Number(n) if n.is_negative() => {
                    let abs_n = -n;
                    let abs_str = if abs_n.is_integer() {
                        format!("{}", abs_n.numer())
                    } else {
                        format!("\\frac{{{}}}{{{}}}", abs_n.numer(), abs_n.denom())
                    };
                    (to_latex(*l), abs_str, true)
                }
                _ => (to_latex(*l), to_latex(*r), false),
            },
            Expr::Sub(l, r) => (to_latex(*l), to_latex(*r), true),
            _ => (den_latex.clone(), String::new(), false),
        };

        let conjugate = if term_b.is_empty() {
            den_latex.clone()
        } else if is_original_minus {
            format!("{} + {}", term_a, term_b)
        } else {
            format!("{} - {}", term_a, term_b)
        };

        sub_steps.push(SubStep {
            description: "Denominador binomial con radical".to_string(),
            before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
            after_expr: format!("\\text{{Conjugado: }} {}", conjugate),
            before_latex: None,
            after_latex: None,
        });

        if let Expr::Div(new_num, new_den) = ctx.get(after) {
            let after_num_latex = to_latex(*new_num);
            let after_den_latex = to_latex(*new_den);

            sub_steps.push(SubStep {
                description: "(a+b)(a-b) = a² - b²".to_string(),
                before_expr: format!(
                    "\\frac{{({}) \\cdot ({})}}{{{}  \\cdot ({})}}",
                    num_latex, conjugate, den_latex, conjugate
                ),
                after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                before_latex: None,
                after_latex: None,
            });
        }
    }

    sub_steps
}
