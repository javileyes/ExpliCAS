use cas_ast::{Expr, ExprId};

/// Recursively apply Weierstrass substitution:
/// `t = tan(x/2) = sin(x/2)/cos(x/2)`.
pub fn apply_weierstrass_recursive(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    if let Expr::Function(name_id, ref args) = expr_data {
        let name = ctx.sym_name(name_id).to_string();
        if matches!(name.as_str(), "sin" | "cos" | "tan") && args.len() == 1 {
            let arg = args[0];

            let two_num = ctx.num(2);
            let half_arg = ctx.add(Expr::Div(arg, two_num));
            let sin_half = ctx.call("sin", vec![half_arg]);
            let cos_half = ctx.call("cos", vec![half_arg]);
            let t = ctx.add(Expr::Div(sin_half, cos_half));

            return match name.as_str() {
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
                _ => expr,
            };
        }
    }

    match expr_data {
        Expr::Add(l, r) => {
            let new_l = apply_weierstrass_recursive(ctx, l);
            let new_r = apply_weierstrass_recursive(ctx, r);
            ctx.add(Expr::Add(new_l, new_r))
        }
        Expr::Sub(l, r) => {
            let new_l = apply_weierstrass_recursive(ctx, l);
            let new_r = apply_weierstrass_recursive(ctx, r);
            ctx.add(Expr::Sub(new_l, new_r))
        }
        Expr::Mul(l, r) => {
            let new_l = apply_weierstrass_recursive(ctx, l);
            let new_r = apply_weierstrass_recursive(ctx, r);
            ctx.add(Expr::Mul(new_l, new_r))
        }
        Expr::Div(l, r) => {
            let new_l = apply_weierstrass_recursive(ctx, l);
            let new_r = apply_weierstrass_recursive(ctx, r);
            ctx.add(Expr::Div(new_l, new_r))
        }
        Expr::Pow(base, exp) => {
            let new_base = apply_weierstrass_recursive(ctx, base);
            let new_exp = apply_weierstrass_recursive(ctx, exp);
            ctx.add(Expr::Pow(new_base, new_exp))
        }
        Expr::Neg(e) => {
            let new_e = apply_weierstrass_recursive(ctx, e);
            ctx.add(Expr::Neg(new_e))
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|&a| apply_weierstrass_recursive(ctx, a))
                .collect();
            ctx.add(Expr::Function(name, new_args))
        }
        _ => expr,
    }
}
