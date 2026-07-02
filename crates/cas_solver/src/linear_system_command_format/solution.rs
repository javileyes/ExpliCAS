use cas_ast::{Context, Expr};
use cas_formatter::{DisplayExpr, LaTeXExpr};
use num_rational::BigRational;

pub(crate) fn display_linear_system_solution(
    ctx: &mut Context,
    vars: &[String],
    values: &[BigRational],
) -> String {
    let mut pairs = Vec::with_capacity(vars.len().min(values.len()));
    for (var, val) in vars.iter().zip(values.iter()) {
        let expr = ctx.add(Expr::Number(val.clone()));
        let shown = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: expr
            }
        );
        pairs.push(format!("{var} = {shown}"));
    }
    format!("{{ {} }}", pairs.join(", "))
}

pub(crate) fn display_linear_system_solution_latex(
    ctx: &mut Context,
    vars: &[String],
    values: &[BigRational],
) -> String {
    let mut pairs = Vec::with_capacity(vars.len().min(values.len()));
    for (var, val) in vars.iter().zip(values.iter()) {
        let expr = ctx.add(Expr::Number(val.clone()));
        let shown = LaTeXExpr {
            context: ctx,
            id: expr,
        }
        .to_latex();
        pairs.push(format!("{var} = {shown}"));
    }
    format!("\\left\\{{ {} \\right\\}}", pairs.join(",\\ "))
}
