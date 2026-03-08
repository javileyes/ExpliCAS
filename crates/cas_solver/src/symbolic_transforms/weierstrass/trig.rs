use cas_ast::{Context, Expr, ExprId};

pub(super) fn rewrite_trig_function(ctx: &mut Context, expr_data: &Expr) -> Option<ExprId> {
    let Expr::Function(name_id, args) = expr_data else {
        return None;
    };

    let name = ctx.sym_name(*name_id).to_string();
    if !matches!(name.as_str(), "sin" | "cos" | "tan") || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let two_num = ctx.num(2);
    let half_arg = ctx.add(Expr::Div(arg, two_num));
    let sin_half = ctx.call("sin", vec![half_arg]);
    let cos_half = ctx.call("cos", vec![half_arg]);
    let t = ctx.add(Expr::Div(sin_half, cos_half));

    Some(match name.as_str() {
        "sin" => {
            let two = ctx.num(2);
            let one = ctx.num(1);
            let t_squared = ctx.add(Expr::Pow(t, two));
            let numerator = ctx.add(Expr::Mul(two, t));
            let denominator = ctx.add(Expr::Add(one, t_squared));
            ctx.add(Expr::Div(numerator, denominator))
        }
        "cos" => {
            let one = ctx.num(1);
            let two = ctx.num(2);
            let t_squared = ctx.add(Expr::Pow(t, two));
            let numerator = ctx.add(Expr::Sub(one, t_squared));
            let denominator = ctx.add(Expr::Add(one, t_squared));
            ctx.add(Expr::Div(numerator, denominator))
        }
        "tan" => {
            let two = ctx.num(2);
            let one = ctx.num(1);
            let t_squared = ctx.add(Expr::Pow(t, two));
            let numerator = ctx.add(Expr::Mul(two, t));
            let denominator = ctx.add(Expr::Sub(one, t_squared));
            ctx.add(Expr::Div(numerator, denominator))
        }
        _ => unreachable!(),
    })
}
