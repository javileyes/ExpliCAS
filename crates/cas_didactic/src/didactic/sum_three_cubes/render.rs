use super::super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn render_sum_three_cubes_substeps(
    context: &Context,
    bases: &[ExprId; 3],
) -> Vec<SubStep> {
    let [x, y, z] = *bases;
    let x_str = render_expr(context, x);
    let y_str = render_expr(context, y);
    let z_str = render_expr(context, z);

    vec![
        SubStep {
            description: "Definimos las bases de los cubos".to_string(),
            before_expr: format!("x = {}, \\quad y = {}, \\quad z = {}", x_str, y_str, z_str),
            after_expr: "x^3 + y^3 + z^3".to_string(),
            before_latex: None,
            after_latex: None,
        },
        SubStep {
            description: "Verificamos que x + y + z = 0".to_string(),
            before_expr: format!("({}) + ({}) + ({})", x_str, y_str, z_str),
            after_expr: "0 \\quad \\checkmark".to_string(),
            before_latex: None,
            after_latex: None,
        },
        SubStep {
            description: "Aplicamos la identidad: si x+y+z=0, entonces x³+y³+z³=3xyz".to_string(),
            before_expr: format!("{}^3 + {}^3 + {}^3", x_str, y_str, z_str),
            after_expr: format!("3 \\cdot ({}) \\cdot ({}) \\cdot ({})", x_str, y_str, z_str),
            before_latex: None,
            after_latex: None,
        },
    ]
}

fn render_expr(context: &Context, expr: ExprId) -> String {
    format!("{}", cas_formatter::DisplayExpr { context, id: expr })
}
