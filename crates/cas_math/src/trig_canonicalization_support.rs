use crate::build::mul2_raw;
use crate::expr_predicates::{is_one_expr, is_two_expr};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::collections::HashSet;

fn is_reciprocal_trig_builtin(b: BuiltinFn) -> bool {
    matches!(
        b,
        BuiltinFn::Tan | BuiltinFn::Cot | BuiltinFn::Sec | BuiltinFn::Csc
    )
}

fn is_reciprocal_trig_name(name: &str) -> bool {
    matches!(name, "tan" | "cot" | "sec" | "csc")
}

fn is_inverse_trig_builtin(b: BuiltinFn) -> bool {
    matches!(
        b,
        BuiltinFn::Asin
            | BuiltinFn::Acos
            | BuiltinFn::Atan
            | BuiltinFn::Acot
            | BuiltinFn::Asec
            | BuiltinFn::Acsc
            | BuiltinFn::Arcsin
            | BuiltinFn::Arccos
            | BuiltinFn::Arctan
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigCanonicalRewritePlan {
    pub rewritten: ExprId,
    pub desc: String,
}

/// Check if expression is a composition like `tan(arctan(x))`.
pub fn is_trig_of_inverse_trig(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(outer_fn_id, outer_args) if outer_args.len() == 1 => {
            let inner = outer_args[0];
            match ctx.get(inner) {
                Expr::Function(inner_fn_id, _) => {
                    let outer_b = ctx.builtin_of(*outer_fn_id);
                    let inner_b = ctx.builtin_of(*inner_fn_id);
                    match (outer_b, inner_b) {
                        (Some(o), Some(i)) => {
                            is_reciprocal_trig_builtin(o) && is_inverse_trig_builtin(i)
                        }
                        _ => false,
                    }
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn collect_trig_recursive(ctx: &Context, expr: ExprId, funcs: &mut HashSet<String>) {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            if let Some(b) = ctx.builtin_of(*fn_id) {
                if matches!(
                    b,
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Tan
                        | BuiltinFn::Cot
                        | BuiltinFn::Sec
                        | BuiltinFn::Csc
                ) {
                    funcs.insert(b.name().to_string());
                }
            }
            for &arg in args {
                collect_trig_recursive(ctx, arg, funcs);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_trig_recursive(ctx, *l, funcs);
            collect_trig_recursive(ctx, *r, funcs);
        }
        Expr::Pow(base, exp) => {
            collect_trig_recursive(ctx, *base, funcs);
            collect_trig_recursive(ctx, *exp, funcs);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_trig_recursive(ctx, *inner, funcs);
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                collect_trig_recursive(ctx, *elem, funcs);
            }
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

fn collect_trig_functions(ctx: &Context, expr: ExprId) -> HashSet<String> {
    let mut funcs = HashSet::new();
    collect_trig_recursive(ctx, expr, &mut funcs);
    funcs
}

fn has_multiple_trig_types(funcs: &HashSet<String>) -> bool {
    funcs.len() >= 2
}

fn is_any_trig_function_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args) if args.len() == 1 => {
                        if let Some(b) = ctx.builtin_of(*fn_id) {
                            if matches!(
                                b,
                                BuiltinFn::Sin
                                    | BuiltinFn::Cos
                                    | BuiltinFn::Tan
                                    | BuiltinFn::Cot
                                    | BuiltinFn::Sec
                                    | BuiltinFn::Csc
                            ) {
                                return Some(args[0]);
                            }
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

fn is_pythagorean_style(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            let l_is_squared = is_any_trig_function_squared(ctx, *l).is_some();
            let r_is_squared = is_any_trig_function_squared(ctx, *r).is_some();
            let l_is_one = is_one_expr(ctx, *l);
            let r_is_one = is_one_expr(ctx, *r);
            (r_is_squared && (l_is_one || l_is_squared)) || (l_is_squared && r_is_one)
        }
        _ => false,
    }
}

/// Check if should trigger mixed-fraction conversion to `sin/cos`.
pub fn is_mixed_trig_fraction(ctx: &Context, num: ExprId, den: ExprId) -> bool {
    if is_pythagorean_style(ctx, num) || is_pythagorean_style(ctx, den) {
        return false;
    }

    let num_funcs = collect_trig_functions(ctx, num);
    let den_funcs = collect_trig_functions(ctx, den);

    if num_funcs.is_empty() && den_funcs.is_empty() {
        return false;
    }

    let num_has_mixed = has_multiple_trig_types(&num_funcs);
    let den_has_mixed = has_multiple_trig_types(&den_funcs);
    let has_reciprocal = num_funcs.iter().any(|n| is_reciprocal_trig_name(n))
        || den_funcs.iter().any(|n| is_reciprocal_trig_name(n));

    (num_has_mixed || den_has_mixed) && has_reciprocal
}

/// Recursively convert reciprocal trig calls into `sin/cos` forms.
pub fn convert_trig_to_sincos(ctx: &mut Context, expr: ExprId) -> ExprId {
    enum TrigOp {
        Function(usize, Vec<ExprId>),
        Binary(ExprId, ExprId, u8), // 0=Add, 1=Sub, 2=Mul, 3=Div
        Pow(ExprId, ExprId),
        Neg(ExprId),
        Leaf,
    }
    let op = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => TrigOp::Function(*fn_id, args.clone()),
        Expr::Add(l, r) => TrigOp::Binary(*l, *r, 0),
        Expr::Sub(l, r) => TrigOp::Binary(*l, *r, 1),
        Expr::Mul(l, r) => TrigOp::Binary(*l, *r, 2),
        Expr::Div(l, r) => TrigOp::Binary(*l, *r, 3),
        Expr::Pow(base, exp) => TrigOp::Pow(*base, *exp),
        Expr::Neg(inner) => TrigOp::Neg(*inner),
        _ => TrigOp::Leaf,
    };

    match op {
        TrigOp::Function(fn_id, args) => {
            let arg = args[0];
            let converted_arg = convert_trig_to_sincos(ctx, arg);
            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Tan) => {
                    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![converted_arg]);
                    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![converted_arg]);
                    ctx.add(Expr::Div(sin_x, cos_x))
                }
                Some(BuiltinFn::Cot) => {
                    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![converted_arg]);
                    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![converted_arg]);
                    ctx.add(Expr::Div(cos_x, sin_x))
                }
                Some(BuiltinFn::Sec) => {
                    let one = ctx.num(1);
                    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![converted_arg]);
                    ctx.add(Expr::Div(one, cos_x))
                }
                Some(BuiltinFn::Csc) => {
                    let one = ctx.num(1);
                    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![converted_arg]);
                    ctx.add(Expr::Div(one, sin_x))
                }
                _ => ctx.add(Expr::Function(fn_id, vec![converted_arg])),
            }
        }
        TrigOp::Binary(l, r, op_code) => {
            let new_l = convert_trig_to_sincos(ctx, l);
            let new_r = convert_trig_to_sincos(ctx, r);
            match op_code {
                0 => ctx.add(Expr::Add(new_l, new_r)),
                1 => ctx.add(Expr::Sub(new_l, new_r)),
                2 => mul2_raw(ctx, new_l, new_r),
                _ => ctx.add(Expr::Div(new_l, new_r)),
            }
        }
        TrigOp::Pow(base, exp) => {
            let new_base = convert_trig_to_sincos(ctx, base);
            ctx.add(Expr::Pow(new_base, exp))
        }
        TrigOp::Neg(inner) => {
            let new_inner = convert_trig_to_sincos(ctx, inner);
            ctx.add(Expr::Neg(new_inner))
        }
        TrigOp::Leaf => expr,
    }
}

/// Check if `expr` is `fname(arg)^2`.
pub fn is_function_squared(ctx: &Context, expr: ExprId, fname: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args)
                        if ctx.builtin_of(*fn_id).is_some_and(|b| b.name() == fname)
                            && args.len() == 1 =>
                    {
                        Some(args[0])
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if two expressions form a reciprocal trig pair with shared argument.
pub fn check_reciprocal_pair(
    ctx: &Context,
    expr1: ExprId,
    expr2: ExprId,
) -> (bool, Option<ExprId>) {
    match (ctx.get(expr1), ctx.get(expr2)) {
        (Expr::Function(name1, args1), Expr::Function(name2, args2))
            if args1.len() == 1 && args2.len() == 1 && args1[0] == args2[0] =>
        {
            let arg = args1[0];
            let b1 = ctx.builtin_of(*name1);
            let b2 = ctx.builtin_of(*name2);
            let is_pair = matches!(
                (b1, b2),
                (Some(BuiltinFn::Tan), Some(BuiltinFn::Cot))
                    | (Some(BuiltinFn::Cot), Some(BuiltinFn::Tan))
                    | (Some(BuiltinFn::Sec), Some(BuiltinFn::Cos))
                    | (Some(BuiltinFn::Cos), Some(BuiltinFn::Sec))
                    | (Some(BuiltinFn::Csc), Some(BuiltinFn::Sin))
                    | (Some(BuiltinFn::Sin), Some(BuiltinFn::Csc))
            );
            (is_pair, if is_pair { Some(arg) } else { None })
        }
        _ => (false, None),
    }
}

pub fn try_rewrite_trig_function_name_canonicalization_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigCanonicalRewritePlan> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };

    let canonical_name = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Asin) => Some("arcsin"),
        Some(BuiltinFn::Acos) => Some("arccos"),
        Some(BuiltinFn::Atan) => Some("arctan"),
        Some(BuiltinFn::Asec) => Some("arcsec"),
        Some(BuiltinFn::Acsc) => Some("arccsc"),
        Some(BuiltinFn::Acot) => Some("arccot"),
        _ => None,
    }?;

    let old_name = ctx.builtin_of(*fn_id)?.name();
    let rewritten = ctx.call(canonical_name, args.clone());
    Some(TrigCanonicalRewritePlan {
        rewritten,
        desc: format!("{} -> {}", old_name, canonical_name),
    })
}

pub fn try_rewrite_sec_tan_pythagorean_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(l, r) = ctx.get(expr) else {
        return None;
    };

    let (Some(sec_arg), Some(tan_arg)) = (
        is_function_squared(ctx, *l, "sec"),
        is_function_squared(ctx, *r, "tan"),
    ) else {
        return None;
    };

    if sec_arg == tan_arg {
        Some(ctx.num(1))
    } else {
        None
    }
}

pub fn try_rewrite_csc_cot_pythagorean_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(l, r) = ctx.get(expr) else {
        return None;
    };

    let (Some(csc_arg), Some(cot_arg)) = (
        is_function_squared(ctx, *l, "csc"),
        is_function_squared(ctx, *r, "cot"),
    ) else {
        return None;
    };

    if csc_arg == cot_arg {
        Some(ctx.num(1))
    } else {
        None
    }
}

pub fn try_rewrite_tan_to_sec_pythagorean_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };

    let tan_arg = if is_one_expr(ctx, *l) {
        is_function_squared(ctx, *r, "tan")
    } else if is_one_expr(ctx, *r) {
        is_function_squared(ctx, *l, "tan")
    } else {
        None
    }?;

    let sec_expr = ctx.call_builtin(BuiltinFn::Sec, vec![tan_arg]);
    let two = ctx.num(2);
    Some(ctx.add(Expr::Pow(sec_expr, two)))
}

pub fn try_rewrite_cot_to_csc_pythagorean_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };

    let cot_arg = if is_one_expr(ctx, *l) {
        is_function_squared(ctx, *r, "cot")
    } else if is_one_expr(ctx, *r) {
        is_function_squared(ctx, *l, "cot")
    } else {
        None
    }?;

    let csc_expr = ctx.call_builtin(BuiltinFn::Csc, vec![cot_arg]);
    let two = ctx.num(2);
    Some(ctx.add(Expr::Pow(csc_expr, two)))
}

pub fn try_rewrite_sec_tan_minus_one_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !is_one_expr(ctx, *right) {
        return None;
    }

    let Expr::Sub(ll, lr) = ctx.get(*left) else {
        return None;
    };

    let (Some(sec_arg), Some(tan_arg)) = (
        is_function_squared(ctx, *ll, "sec"),
        is_function_squared(ctx, *lr, "tan"),
    ) else {
        return None;
    };

    if sec_arg == tan_arg {
        Some(ctx.num(0))
    } else {
        None
    }
}

pub fn try_rewrite_csc_cot_minus_one_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !is_one_expr(ctx, *right) {
        return None;
    }

    let Expr::Sub(ll, lr) = ctx.get(*left) else {
        return None;
    };

    let (Some(csc_arg), Some(cot_arg)) = (
        is_function_squared(ctx, *ll, "csc"),
        is_function_squared(ctx, *lr, "cot"),
    ) else {
        return None;
    };

    if csc_arg == cot_arg {
        Some(ctx.num(0))
    } else {
        None
    }
}

pub fn try_rewrite_reciprocal_product_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let (is_reciprocal_pair, _) = check_reciprocal_pair(ctx, *l, *r);
    if is_reciprocal_pair {
        Some(ctx.num(1))
    } else {
        None
    }
}

pub fn try_rewrite_mixed_fraction_to_sincos_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    if !is_mixed_trig_fraction(ctx, num, den) {
        return None;
    }

    let new_num = convert_trig_to_sincos(ctx, num);
    let new_den = convert_trig_to_sincos(ctx, den);
    Some(ctx.add(Expr::Div(new_num, new_den)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn contains_named_call(ctx: &Context, expr: ExprId, name: &str) -> bool {
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            match ctx.get(id) {
                Expr::Function(fn_id, args) => {
                    if ctx.builtin_of(*fn_id).is_some_and(|b| b.name() == name) {
                        return true;
                    }
                    stack.extend(args.iter().copied());
                }
                Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Mul(l, r)
                | Expr::Div(l, r)
                | Expr::Pow(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
                Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            }
        }
        false
    }

    #[test]
    fn detects_trig_inverse_composition() {
        let mut ctx = Context::new();
        let expr = parse("tan(arctan(x))", &mut ctx).expect("expr");
        assert!(is_trig_of_inverse_trig(&ctx, expr));
    }

    #[test]
    fn detects_reciprocal_pair() {
        let mut ctx = Context::new();
        let l = parse("tan(x)", &mut ctx).expect("l");
        let r = parse("cot(x)", &mut ctx).expect("r");
        let (ok, _) = check_reciprocal_pair(&ctx, l, r);
        assert!(ok);
    }

    #[test]
    fn converts_reciprocal_calls_to_sin_cos() {
        let mut ctx = Context::new();
        let expr = parse("tan(x) * sec(x)", &mut ctx).expect("expr");
        let converted = convert_trig_to_sincos(&mut ctx, expr);
        assert!(!contains_named_call(&ctx, converted, "tan"));
        assert!(!contains_named_call(&ctx, converted, "sec"));
        assert!(contains_named_call(&ctx, converted, "sin"));
        assert!(contains_named_call(&ctx, converted, "cos"));
    }

    #[test]
    fn mixed_fraction_detection_requires_reciprocal_component() {
        let mut ctx = Context::new();
        let mixed = parse("(sin(x)+cos(x))/tan(x)", &mut ctx).expect("mixed");
        let plain = parse("(sin(x)+cos(x))/sin(x)", &mut ctx).expect("plain");

        let (mixed_num, mixed_den) = match ctx.get(mixed) {
            Expr::Div(n, d) => (*n, *d),
            _ => panic!("expected div"),
        };
        let (plain_num, plain_den) = match ctx.get(plain) {
            Expr::Div(n, d) => (*n, *d),
            _ => panic!("expected div"),
        };

        assert!(is_mixed_trig_fraction(&ctx, mixed_num, mixed_den));
        assert!(!is_mixed_trig_fraction(&ctx, plain_num, plain_den));
    }

    #[test]
    fn rewrites_sec_tan_identity_to_one() {
        let mut ctx = Context::new();
        let expr = parse("sec(x)^2 - tan(x)^2", &mut ctx).expect("parse");
        let rewritten = try_rewrite_sec_tan_pythagorean_expr(&mut ctx, expr).expect("rewrite");
        assert!(is_one_expr(&ctx, rewritten));
    }

    #[test]
    fn rewrites_reciprocal_product_to_one() {
        let mut ctx = Context::new();
        let expr = parse("tan(x) * cot(x)", &mut ctx).expect("parse");
        let rewritten = try_rewrite_reciprocal_product_expr(&mut ctx, expr).expect("rewrite");
        assert!(is_one_expr(&ctx, rewritten));
    }

    #[test]
    fn rewrites_mixed_fraction_to_sincos_form() {
        let mut ctx = Context::new();
        let expr = parse("(sin(x)+cos(x))/tan(x)", &mut ctx).expect("parse");
        let rewritten = try_rewrite_mixed_fraction_to_sincos_expr(&mut ctx, expr).expect("rewrite");
        assert!(!contains_named_call(&ctx, rewritten, "tan"));
        assert!(contains_named_call(&ctx, rewritten, "sin"));
        assert!(contains_named_call(&ctx, rewritten, "cos"));
    }

    #[test]
    fn canonicalizes_short_inverse_trig_name() {
        let mut ctx = Context::new();
        let expr = parse("acos(x)", &mut ctx).expect("parse");
        let plan = try_rewrite_trig_function_name_canonicalization_expr(&mut ctx, expr)
            .expect("canonicalization");
        assert!(plan.desc.contains("acos"));
        assert!(contains_named_call(&ctx, plan.rewritten, "arccos"));
    }
}
