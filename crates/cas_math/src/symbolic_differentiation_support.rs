//! Symbolic differentiation helpers shared by differentiation-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_predicates::contains_named_var;
use cas_ast::{ordering::compare_expr, BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn add_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

fn sub_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        ctx.add(Expr::Neg(right))
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

fn mul_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) || is_zero(ctx, right) {
        ctx.num(0)
    } else if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        mul2_raw(ctx, left, right)
    }
}

fn one_minus_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![inner])
}

fn unit_reciprocal_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if is_one(ctx, *num) => Some(*den),
        Expr::Pow(base, exp)
            if matches!(
                ctx.get(*exp),
                Expr::Number(n) if n.is_integer() && n.to_integer() == (-1).into()
            ) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn one_minus_reciprocal_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> (ExprId, ExprId) {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let reciprocal_arg_sq = ctx.add(Expr::Div(one, arg_sq));
    let inner = ctx.add(Expr::Sub(one, reciprocal_arg_sq));
    (ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]), arg_sq)
}

fn arcsec_like_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId, sign: i64) -> ExprId {
    let one = ctx.num(1);
    let (sqrt_gap, arg_sq) = one_minus_reciprocal_arg_square_sqrt(ctx, arg);
    let numerator = mul_pruned(ctx, d_arg, sqrt_gap);
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(numerator))
    } else {
        numerator
    };
    let denominator = ctx.add(Expr::Sub(arg_sq, one));
    ctx.add(Expr::Div(numerator, denominator))
}

fn arccot_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let denominator = ctx.add(Expr::Add(arg_sq, one));
    let numerator = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(numerator, denominator))
}

fn acosh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let minus_one = ctx.num(-1);
    let two = ctx.num(2);
    let neg_half = ctx.add(Expr::Div(minus_one, two));
    let left = ctx.add(Expr::Sub(arg, one));
    let right = ctx.add(Expr::Add(arg, one));
    let left_inv_sqrt = ctx.add(Expr::Pow(left, neg_half));
    let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
    let numerator = mul_pruned(ctx, d_arg, left_inv_sqrt);
    ctx.add(Expr::Div(numerator, sqrt_right))
}

fn atanh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let minus_two = ctx.num(-2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    let sqrt_inner = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
    let inv_square = ctx.add(Expr::Pow(sqrt_inner, minus_two));
    mul_pruned(ctx, d_arg, inv_square)
}

fn constant_base_log_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    base: i64,
) -> ExprId {
    let base = ctx.num(base);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let den = mul_pruned(ctx, arg, ln_base);
    ctx.add(Expr::Div(d_arg, den))
}

#[derive(Clone, Copy)]
enum ReciprocalTrigDerivativeKind {
    Sec,
    Csc,
    Cot,
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn canonical_reciprocal_trig_div_kind(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ReciprocalTrigDerivativeKind, ExprId)> {
    if is_one(ctx, num) {
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Cos) {
            return Some((ReciprocalTrigDerivativeKind::Sec, arg));
        }
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Sin) {
            return Some((ReciprocalTrigDerivativeKind::Csc, arg));
        }
    }

    let cos_arg = unary_builtin_arg(ctx, num, BuiltinFn::Cos)?;
    let sin_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sin)?;
    if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
        Some((ReciprocalTrigDerivativeKind::Cot, cos_arg))
    } else {
        None
    }
}

fn squared_builtin_call(ctx: &mut Context, builtin: BuiltinFn, arg: ExprId) -> ExprId {
    let call = ctx.call_builtin(builtin, vec![arg]);
    let two = ctx.num(2);
    ctx.add(Expr::Pow(call, two))
}

fn secant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_sq = squared_builtin_call(ctx, BuiltinFn::Cos, arg);
    let numerator = mul_pruned(ctx, sin_u, d_arg);
    ctx.add(Expr::Div(numerator, cos_sq))
}

fn cosecant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let neg_cos = ctx.add(Expr::Neg(cos_u));
    let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
    let numerator = mul_pruned(ctx, neg_cos, d_arg);
    ctx.add(Expr::Div(numerator, sin_sq))
}

fn cotangent_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
    let neg_d_arg = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(neg_d_arg, sin_sq))
}

fn reciprocal_trig_derivative(
    ctx: &mut Context,
    kind: ReciprocalTrigDerivativeKind,
    arg: ExprId,
    d_arg: ExprId,
) -> ExprId {
    match kind {
        ReciprocalTrigDerivativeKind::Sec => secant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Csc => cosecant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Cot => cotangent_derivative(ctx, arg, d_arg),
    }
}

/// Differentiate `expr` with respect to variable `var`.
pub fn differentiate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if !contains_named_var(ctx, expr, var) {
        return Some(ctx.num(0));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) => {
            if ctx.sym_name(*sym_id) == var {
                Some(ctx.num(1))
            } else {
                Some(ctx.num(0))
            }
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(add_pruned(ctx, dl, dr))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(sub_pruned(ctx, dl, dr))
        }
        Expr::Neg(inner) => {
            let inner = *inner;
            let d_inner = differentiate_symbolic_expr(ctx, inner, var)?;
            Some(ctx.add(Expr::Neg(d_inner)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            Some(add_pruned(ctx, term1, term2))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            if let Some((kind, arg)) = canonical_reciprocal_trig_div_kind(ctx, l, r) {
                let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
                return Some(reciprocal_trig_derivative(ctx, kind, arg, d_arg));
            }

            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            let num = sub_pruned(ctx, term1, term2);
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let db = differentiate_symbolic_expr(ctx, base, var)?;
            let de = differentiate_symbolic_expr(ctx, exp, var)?;

            if !contains_named_var(ctx, exp, var) {
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = mul_pruned(ctx, exp, pow_term);
                Some(mul_pruned(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                let ln_a = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term = mul_pruned(ctx, expr, ln_a);
                Some(mul_pruned(ctx, term, de))
            } else {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term1 = mul_pruned(ctx, de, ln_base);
                let term2_num = mul_pruned(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = add_pruned(ctx, term1, term2);
                Some(mul_pruned(ctx, expr, inner))
            }
        }
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Log)) && args.len() == 2 {
                let base = args[0];
                let arg = args[1];
                if contains_named_var(ctx, base, var) {
                    let db = differentiate_symbolic_expr(ctx, base, var)?;
                    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
                    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                    let two = ctx.num(2);
                    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));

                    if contains_named_var(ctx, arg, var) {
                        let da = differentiate_symbolic_expr(ctx, arg, var)?;
                        let arg_ratio = ctx.add(Expr::Div(da, arg));
                        let base_ratio = ctx.add(Expr::Div(db, base));
                        let term_arg = mul_pruned(ctx, arg_ratio, ln_base);
                        let term_base = mul_pruned(ctx, ln_arg, base_ratio);
                        let numerator = sub_pruned(ctx, term_arg, term_base);
                        return Some(ctx.add(Expr::Div(numerator, ln_base_sq)));
                    }

                    let denominator = mul_pruned(ctx, base, ln_base_sq);
                    let numerator = mul_pruned(ctx, ln_arg, db);
                    let neg_numerator = ctx.add(Expr::Neg(numerator));
                    return Some(ctx.add(Expr::Div(neg_numerator, denominator)));
                }

                let da = differentiate_symbolic_expr(ctx, arg, var)?;
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let den = mul_pruned(ctx, arg, ln_base);
                return Some(ctx.add(Expr::Div(da, den)));
            }

            if args.len() != 1 {
                return None;
            }
            let arg = args[0];

            if let Some(recip_base) = unit_reciprocal_base(ctx, arg) {
                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, 1));
                    }
                    Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, -1));
                    }
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arccot_derivative(ctx, recip_base, d_base));
                    }
                    _ => {}
                }
            }

            let da = differentiate_symbolic_expr(ctx, arg, var)?;

            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    Some(mul_pruned(ctx, cos_u, da))
                }
                Some(BuiltinFn::Cos) => {
                    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let neg_sin = ctx.add(Expr::Neg(sin_u));
                    Some(mul_pruned(ctx, neg_sin, da))
                }
                Some(BuiltinFn::Tan) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    let two = ctx.num(2);
                    let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                    let one = ctx.num(1);
                    let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                    Some(mul_pruned(ctx, sec_sq, da))
                }
                Some(BuiltinFn::Sec) => Some(secant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Csc) => Some(cosecant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Cot) => Some(cotangent_derivative(ctx, arg, da)),
                Some(BuiltinFn::Sinh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    Some(mul_pruned(ctx, cosh_u, da))
                }
                Some(BuiltinFn::Cosh) => {
                    let sinh_u = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    Some(mul_pruned(ctx, sinh_u, da))
                }
                Some(BuiltinFn::Tanh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    let two = ctx.num(2);
                    let cosh_sq = ctx.add(Expr::Pow(cosh_u, two));
                    let one = ctx.num(1);
                    let sech_sq = ctx.add(Expr::Div(one, cosh_sq));
                    Some(mul_pruned(ctx, sech_sq, da))
                }
                Some(BuiltinFn::Sqrt) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    let minus_one = ctx.num(-1);
                    let neg_half = ctx.add(Expr::Div(minus_one, two));
                    let pow_term = ctx.add(Expr::Pow(arg, neg_half));
                    let term = mul_pruned(ctx, half, pow_term);
                    Some(mul_pruned(ctx, term, da))
                }
                Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let den = ctx.add(Expr::Add(one, arg_sq));
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    let neg_da = ctx.add(Expr::Neg(da));
                    Some(ctx.add(Expr::Div(neg_da, den)))
                }
                Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => {
                    Some(arcsec_like_derivative(ctx, arg, da, 1))
                }
                Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
                    Some(arcsec_like_derivative(ctx, arg, da, -1))
                }
                Some(BuiltinFn::Arccot | BuiltinFn::Acot) => Some(arccot_derivative(ctx, arg, da)),
                Some(BuiltinFn::Asinh) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let inner = ctx.add(Expr::Add(arg_sq, one));
                    let den = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Acosh) => Some(acosh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Atanh) => Some(atanh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Exp) => Some(mul_pruned(ctx, expr, da)),
                Some(BuiltinFn::Ln) => Some(ctx.add(Expr::Div(da, arg))),
                Some(BuiltinFn::Log2) => Some(constant_base_log_derivative(ctx, arg, da, 2)),
                Some(BuiltinFn::Log10) => Some(constant_base_log_derivative(ctx, arg, da, 10)),
                Some(BuiltinFn::Abs) => {
                    let term = ctx.add(Expr::Div(arg, expr));
                    Some(mul_pruned(ctx, term, da))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::differentiate_symbolic_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn differentiates_product_rule() {
        let mut ctx = Context::new();
        let expr = parse("x*sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("sin(x)"));
        assert!(text.contains("cos(x)"));
    }

    #[test]
    fn differentiates_chain_rule_exp() {
        let mut ctx = Context::new();
        let expr = parse("exp(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("exp(x^2)") || text.contains("e^(x^2)"));
    }

    #[test]
    fn prunes_zero_product_terms_from_polynomial_derivative() {
        let mut ctx = Context::new();
        let expr = parse("x^3 + 2*x^2 - 5*x + 1", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(!text.contains("0 ·"), "unexpected zero product in {text}");
        assert!(!text.contains("· 1"), "unexpected unit factor in {text}");
        assert!(text.contains("3"));
        assert!(text.contains("- 5"));
    }

    #[test]
    fn prunes_unit_chain_factor_for_sine() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cos(x)");
    }

    #[test]
    fn differentiates_reciprocal_trig_functions_directly() {
        let cases = [
            ("sec(x)", "sin(x) / cos(x)^2"),
            ("csc(x)", "-cos(x) / sin(x)^2"),
            ("cot(x)", "-1 / sin(x)^2"),
            ("1/cos(x)", "sin(x) / cos(x)^2"),
            ("1/sin(x)", "-cos(x) / sin(x)^2"),
            ("cos(x)/sin(x)", "-1 / sin(x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_reciprocal_trig_chain_rule_directly() {
        let cases = [
            ("sec(2*x + 1)", "sin(2 * x + 1) * 2 / cos(2 * x + 1)^2"),
            ("csc(2*x + 1)", "-cos(2 * x + 1) * 2 / sin(2 * x + 1)^2"),
            ("cot(2*x + 1)", "-2 / sin(2 * x + 1)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_functions() {
        let cases = [
            ("sinh(x)", "cosh(x)"),
            ("cosh(x)", "sinh(x)"),
            ("tanh(x)", "1 / cosh(x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("cosh(x^2)"), "{text}");
        assert!(text.contains("2 * x"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_total_domain_inverse_functions() {
        let cases = [("arctan(x)", "x^2 + 1"), ("asinh(x)", "x^2 + 1")];

        for (input, expected_core) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(text.contains(expected_core), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }
    }

    #[test]
    fn differentiates_constant_base_unary_logs() {
        let cases = [
            ("log2(x)", "1 / (x * ln(2))"),
            ("log10(x)", "1 / (x * ln(10))"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_variable_base_constant_argument_logs_conservatively() {
        let cases = [("log(x, 2)", "ln(2)"), ("log(x, y)", "ln(y)")];

        for (input, expected_log_arg) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                text.contains(expected_log_arg),
                "input: {input}, got: {text}"
            );
            assert!(text.contains("ln(x)^2"), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }

        let mut ctx = Context::new();
        let expr = parse("log(x, x + 1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("ln(x)"), "got: {text}");
        assert!(text.contains("ln(x + 1)"), "got: {text}");
        assert!(text.contains("ln(x)^2"), "got: {text}");
        assert!(!text.contains("diff("), "got: {text}");
    }

    #[test]
    fn differentiates_inverse_reciprocal_trig_directly() {
        let cases = [
            ("arcsec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccot(x)", "-1 / (x^2 + 1)"),
            ("asec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acot(x)", "-1 / (x^2 + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn inverse_reciprocal_trig_chain_rule_keeps_compact_core() {
        let mut ctx = Context::new();
        let expr = parse("arcsec((x^2 + 1)^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("sqrt(1 - 1 / (x^2 + 1)^2^2)"), "{text}");
        assert!(
            !text.contains("+") || !text.contains("x^3"),
            "unexpected quotient-rule expansion in {text}"
        );
    }

    #[test]
    fn reciprocal_inverse_trig_rewrite_targets_keep_compact_derivative_core() {
        let cases = [
            "arccos(1/(x^2 + 1)^2)",
            "arcsin(1/(x^2 + 1)^2)",
            "arctan(1/(x^2 + 1)^2)",
        ];

        for input in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("/sqrt("),
                "unexpected quotient-rule reciprocal derivative in {text}"
            );
            assert!(
                !text.contains("x^3"),
                "unexpected expanded chain factor in {text}"
            );
        }
    }

    #[test]
    fn differentiates_acosh_with_domain_safe_radicals() {
        let cases = [
            ("acosh(x)", "(x - 1)^(-1 / 2) / sqrt(x + 1)"),
            (
                "acosh(2*x + 1)",
                "2 * (2 * x + 1 - 1)^(-1 / 2) / sqrt(2 * x + 1 + 1)",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_atanh_with_open_unit_interval_witness() {
        let cases = [
            ("atanh(x)", "sqrt(1 - x^2)^(-2)"),
            ("atanh(x^2)", "(x^(2 - 1) * 2)/(sqrt(1 - x^2^2)^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_sqrt_chain_rule_without_power_presimplification() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(2 - x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(-1 / 2)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_power_chain_rule_through_canonical_negation() {
        let mut ctx = Context::new();
        let expr = parse("(2 + (-x))^(1/2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(1 / 2 - 1)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_bounded_inverse_trig_functions() {
        let cases = [
            ("arcsin(x)", "1 / sqrt(1 - x^2)"),
            ("arccos(x)", "-1 / sqrt(1 - x^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_bounded_inverse_trig_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("asin(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("sqrt(1 - x^2^2)"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_inverse_function_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("arctan(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("x^2^2 + 1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_log_with_constant_base() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("x * ln(2)") || text.contains("ln(2) * x"),
            "{text}"
        );
        assert!(!text.contains("diff("), "{text}");
    }
}
