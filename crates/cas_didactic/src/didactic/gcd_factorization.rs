use super::SubStep;
use cas_ast::{Context, Expr};
use cas_solver::Step;

/// Generate sub-steps explaining polynomial factorization and GCD cancellation.
/// For example: `(x^2 - 4) / (2 + x)` shows:
///   1. Factor numerator: `x^2 - 4 -> (x-2)(x+2)`
///   2. Cancel common factor: `(x-2)(x+2) / (x+2) -> x-2`
pub(crate) fn generate_gcd_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_formatter::DisplayExpr;

    let mut sub_steps = Vec::new();

    let gcd_start = "Simplified fraction by GCD: ";
    if !step.description.starts_with(gcd_start) {
        return sub_steps;
    }
    let gcd_str = &step.description[gcd_start.len()..];

    if let Expr::Div(num, den) = ctx.get(step.before) {
        let num_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *num
            }
        );
        let den_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *den
            }
        );
        let after_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: step.after
            }
        );

        // If the rule display already exposes the factored form, surface it as
        // an educational sub-step before the cancellation.
        if let Some(local_before) = step.before_local() {
            let local_before_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: local_before
                }
            );

            if local_before_str.contains(gcd_str) && local_before_str.contains('*') {
                sub_steps.push(SubStep {
                    description: format!("Factor: {} contains factor {}", num_str, gcd_str),
                    before_expr: num_str.clone(),
                    after_expr: local_before_str
                        .split('/')
                        .next()
                        .unwrap_or(&local_before_str)
                        .trim()
                        .to_string(),
                    before_latex: None,
                    after_latex: None,
                });
            }
        }

        let needs_parens_num =
            num_str.contains('+') || num_str.contains('-') || num_str.contains(' ');
        let needs_parens_den =
            den_str.contains('+') || den_str.contains('-') || den_str.contains(' ');
        let before_formatted = format!(
            "{} / {}",
            if needs_parens_num {
                format!("({})", num_str)
            } else {
                num_str.clone()
            },
            if needs_parens_den {
                format!("({})", den_str)
            } else {
                den_str.clone()
            }
        );

        sub_steps.push(SubStep {
            description: format!("Cancel common factor: {}", gcd_str),
            before_expr: before_formatted,
            after_expr: after_str,
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
