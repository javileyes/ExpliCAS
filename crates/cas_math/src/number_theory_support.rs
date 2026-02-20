use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Check if expression contains a `poly_result(...)` reference.
pub fn contains_poly_result(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return true;
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && !args.is_empty() {
                return contains_poly_result(ctx, args[0]);
            }
            args.iter().any(|&arg| contains_poly_result(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_poly_result(ctx, *l) || contains_poly_result(ctx, *r)
        }
        Expr::Pow(base, exp) => contains_poly_result(ctx, *base) || contains_poly_result(ctx, *exp),
        Expr::Neg(inner) => contains_poly_result(ctx, *inner),
        _ => false,
    }
}

/// Extract integer exponent from expression.
pub fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Check if expression has large unexpanded powers (exponent > 2) over non-atomic bases.
pub fn has_large_unexpanded_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Some(n) = get_integer_exponent(ctx, *exp) {
                if n > 2 && !matches!(ctx.get(*base), Expr::Variable(_) | Expr::Number(_)) {
                    return true;
                }
            }
            has_large_unexpanded_power(ctx, *base) || has_large_unexpanded_power(ctx, *exp)
        }
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) {
                return false;
            }
            args.iter().any(|&arg| has_large_unexpanded_power(ctx, arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            has_large_unexpanded_power(ctx, *l) || has_large_unexpanded_power(ctx, *r)
        }
        Expr::Neg(inner) => has_large_unexpanded_power(ctx, *inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn detects_poly_result_even_inside_hold() {
        let mut ctx = Context::new();
        let poly = parse("poly_result(7)", &mut ctx).expect("poly");
        let held = ctx.call_builtin(BuiltinFn::Hold, vec![poly]);
        let one = ctx.num(1);
        let wrapped = ctx.add(Expr::Add(one, held));
        assert!(contains_poly_result(&ctx, wrapped));
    }

    #[test]
    fn large_unexpanded_power_detection_matches_intent() {
        let mut ctx = Context::new();
        let large_compound = parse("(x+1)^3", &mut ctx).expect("compound");
        let large_atomic = parse("x^3", &mut ctx).expect("atomic");
        let poly_power = parse("(poly_result(3))^8", &mut ctx).expect("poly");

        assert!(has_large_unexpanded_power(&ctx, large_compound));
        assert!(!has_large_unexpanded_power(&ctx, large_atomic));
        assert!(has_large_unexpanded_power(&ctx, poly_power));
    }

    #[test]
    fn integer_exponent_extraction_handles_negation() {
        let mut ctx = Context::new();
        let exp = parse("-5", &mut ctx).expect("exp");
        assert_eq!(get_integer_exponent(&ctx, exp), Some(-5));
    }
}
