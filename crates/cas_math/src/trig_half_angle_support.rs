use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Check if `arg` represents `u/2` and return `u`.
/// Supports `Mul(1/2, u)`, `Mul(u, 1/2)` and `Div(u, 2)`.
pub fn is_half_angle(ctx: &Context, arg: ExprId) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Mul(l, r) => {
            let half = num_rational::BigRational::new(1.into(), 2.into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == half {
                    return Some(*r);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == half {
                    return Some(*l);
                }
            }
            None
        }
        Expr::Div(num, den) => {
            if let Expr::Number(d) = ctx.get(*den) {
                if *d == num_rational::BigRational::from_integer(2.into()) {
                    return Some(*num);
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if `expr` is `tan(u/2)` and return `u`.
pub fn extract_tan_half_angle(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            return is_half_angle(ctx, args[0]);
        }
    }
    None
}

/// If `expr` is `sin(u/2)` or `cos(u/2)`, returns `(u, is_sin)`.
pub fn extract_trig_half_angle(ctx: &Context, expr: ExprId) -> Option<(ExprId, bool)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 {
            let builtin = ctx.builtin_of(*fn_id);
            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if is_sin || is_cos {
                if let Some(full_angle) = is_half_angle(ctx, args[0]) {
                    return Some((full_angle, is_sin));
                }
            }
        }
    }
    None
}

/// Extract coefficient and cot argument from a term.
/// Returns `(coefficient_opt, cot_arg, is_positive)` where `coefficient_opt=None` means `1`.
pub fn extract_cot_term(ctx: &Context, term: ExprId) -> Option<(Option<ExprId>, ExprId, bool)> {
    let term_data = ctx.get(term);

    let (inner_term, is_positive) = match term_data {
        Expr::Neg(inner) => (*inner, false),
        _ => (term, true),
    };

    let inner_data = ctx.get(inner_term);

    if let Expr::Function(fn_id, args) = inner_data {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
            return Some((None, args[0], is_positive));
        }
    }

    if let Expr::Mul(l, r) = inner_data {
        if let Expr::Function(fn_id, args) = ctx.get(*r) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*l), args[0], is_positive));
            }
        }
        if let Expr::Function(fn_id, args) = ctx.get(*l) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*r), args[0], is_positive));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn is_half_angle_recognizes_mul_and_div_forms() {
        let mut ctx = Context::new();
        let div = parse("x/2", &mut ctx).expect("x/2");
        let x = parse("x", &mut ctx).expect("x");
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let mul = ctx.add(Expr::Mul(half, x));

        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                is_half_angle(&ctx, div).expect("div half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                is_half_angle(&ctx, mul).expect("mul half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_tan_half_angle_matches_tan_of_half_arg() {
        let mut ctx = Context::new();
        let expr = parse("tan(x/2)", &mut ctx).expect("tan(x/2)");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                extract_tan_half_angle(&ctx, expr).expect("tan half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_trig_half_angle_distinguishes_sin_and_cos() {
        let mut ctx = Context::new();
        let sin_expr = parse("sin(x/2)", &mut ctx).expect("sin(x/2)");
        let cos_expr = parse("cos(x/2)", &mut ctx).expect("cos(x/2)");

        let sin = extract_trig_half_angle(&ctx, sin_expr).expect("sin half-angle");
        let cos = extract_trig_half_angle(&ctx, cos_expr).expect("cos half-angle");

        assert!(sin.1);
        assert!(!cos.1);
    }

    #[test]
    fn extract_cot_term_handles_plain_negated_and_scaled_terms() {
        let mut ctx = Context::new();
        let plain = parse("cot(x)", &mut ctx).expect("cot(x)");
        let neg = parse("-cot(y)", &mut ctx).expect("-cot(y)");
        let scaled = parse("3*cot(z)", &mut ctx).expect("3*cot(z)");

        assert!(extract_cot_term(&ctx, plain).is_some());
        assert!(extract_cot_term(&ctx, neg).is_some_and(|(_, _, positive)| !positive));
        assert!(extract_cot_term(&ctx, scaled).is_some_and(|(c, _, _)| c.is_some()));
    }
}
