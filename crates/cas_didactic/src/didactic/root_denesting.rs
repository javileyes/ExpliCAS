use super::SubStep;
use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;
use num_bigint::BigInt;
use num_rational::BigRational;

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

    let get_sqrt_inner = |id: ExprId| -> Option<ExprId> {
        match ctx.get(id) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                Some(args[0])
            }
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n.numer() == BigInt::from(1) && *n.denom() == BigInt::from(2) {
                        return Some(*base);
                    }
                }
                None
            }
            _ => None,
        }
    };

    let inner = match get_sqrt_inner(before_expr) {
        Some(id) => id,
        None => return sub_steps,
    };

    let (a_term, b_term, is_add) = match ctx.get(inner) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return sub_steps,
    };

    fn analyze_surd(ctx: &Context, e: ExprId) -> Option<(BigRational, ExprId)> {
        match ctx.get(e) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                Some((BigRational::from_integer(BigInt::from(1)), args[0]))
            }
            Expr::Mul(l, r) => match (ctx.get(*l), ctx.get(*r)) {
                (Expr::Number(coeff), Expr::Function(fn_id, args))
                    if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
                {
                    Some((coeff.clone(), args[0]))
                }
                (Expr::Function(fn_id, args), Expr::Number(coeff))
                    if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
                {
                    Some((coeff.clone(), args[0]))
                }
                _ => None,
            },
            _ => None,
        }
    }

    let (a_expr, c_coeff, d_expr) = if let Some((coeff, rad)) = analyze_surd(ctx, a_term) {
        (b_term, coeff, rad)
    } else if let Some((coeff, rad)) = analyze_surd(ctx, b_term) {
        (a_term, coeff, rad)
    } else {
        return sub_steps;
    };

    let a_str = to_latex(a_expr);
    let d_str = to_latex(d_expr);
    let c_str = if c_coeff.is_integer() {
        format!("{}", c_coeff.to_integer())
    } else {
        format!("\\frac{{{}}}{{{}}}", c_coeff.numer(), c_coeff.denom())
    };

    sub_steps.push(SubStep {
        description: "Identificar la forma √(a ± c·√d)".to_string(),
        before_expr: to_latex(before_expr),
        after_expr: if is_add {
            format!("a = {}, \\quad c = {}, \\quad d = {}", a_str, c_str, d_str)
        } else {
            format!("a = {}, \\quad c = -{}, \\quad d = {}", a_str, c_str, d_str)
        },
        before_latex: None,
        after_latex: None,
    });

    if let Expr::Number(a_num) = ctx.get(a_expr) {
        if let Expr::Number(d_num) = ctx.get(d_expr) {
            let c_sq = &c_coeff * &c_coeff;
            let delta = a_num * a_num - &c_sq * d_num;
            let delta_str = if delta.is_integer() {
                format!("{}", delta.to_integer())
            } else {
                format!("\\frac{{{}}}{{{}}}", delta.numer(), delta.denom())
            };

            sub_steps.push(SubStep {
                description: "Calcular Δ = a² - c²d".to_string(),
                before_expr: format!("({})^2 - ({})^2 \\cdot {}", a_str, c_str, d_str),
                after_expr: delta_str.clone(),
                before_latex: None,
                after_latex: None,
            });

            if delta.is_integer() {
                let delta_int = delta.to_integer();
                if delta_int >= BigInt::from(0) {
                    sub_steps.push(SubStep {
                        description: "Δ es cuadrado perfecto: aplicar desanidación".to_string(),
                        before_expr: format!("\\sqrt{{{}}}", to_latex(inner)),
                        after_expr: to_latex(step.after_local().unwrap_or(step.after)),
                        before_latex: None,
                        after_latex: None,
                    });
                }
            }
        }
    }

    sub_steps
}
