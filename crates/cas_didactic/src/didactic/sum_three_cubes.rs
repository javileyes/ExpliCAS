use super::SubStep;
use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;
use num_bigint::BigInt;

/// Generate sub-steps explaining the Sum of Three Cubes identity.
pub(crate) fn generate_sum_three_cubes_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_formatter::DisplayExpr;

    let mut sub_steps = Vec::new();
    let before_expr = step.before;
    let terms = cas_math::expr_nary::add_leaves(ctx, before_expr);

    if terms.len() != 3 {
        return sub_steps;
    }

    let mut bases: Vec<ExprId> = Vec::new();
    for &term in &terms {
        let base = match ctx.get(term).clone() {
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e).clone() {
                    if n.is_integer() && n.to_integer() == BigInt::from(3) {
                        Some(b)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Expr::Neg(inner) => {
                if let Expr::Pow(_b, e) = ctx.get(inner).clone() {
                    if let Expr::Number(n) = ctx.get(e).clone() {
                        if n.is_integer() && n.to_integer() == BigInt::from(3) {
                            Some(inner)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(b) = base {
            bases.push(b);
        } else {
            return sub_steps;
        }
    }

    if bases.len() != 3 {
        return sub_steps;
    }

    let fmt = |id: ExprId| -> String { format!("{}", DisplayExpr { context: ctx, id }) };

    let x_str = fmt(bases[0]);
    let y_str = fmt(bases[1]);
    let z_str = fmt(bases[2]);

    sub_steps.push(SubStep {
        description: "Definimos las bases de los cubos".to_string(),
        before_expr: format!("x = {}, \\quad y = {}, \\quad z = {}", x_str, y_str, z_str),
        after_expr: "x^3 + y^3 + z^3".to_string(),
        before_latex: None,
        after_latex: None,
    });

    sub_steps.push(SubStep {
        description: "Verificamos que x + y + z = 0".to_string(),
        before_expr: format!("({}) + ({}) + ({})", x_str, y_str, z_str),
        after_expr: "0 \\quad \\checkmark".to_string(),
        before_latex: None,
        after_latex: None,
    });

    sub_steps.push(SubStep {
        description: "Aplicamos la identidad: si x+y+z=0, entonces x³+y³+z³=3xyz".to_string(),
        before_expr: format!("{}^3 + {}^3 + {}^3", x_str, y_str, z_str),
        after_expr: format!("3 \\cdot ({}) \\cdot ({}) \\cdot ({})", x_str, y_str, z_str),
        before_latex: None,
        after_latex: None,
    });

    sub_steps
}
