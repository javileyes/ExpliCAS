//! Expression classification helpers shared by higher-level rules.

use cas_ast::{Context, ExprId};

/// Check if a function name is trigonometric (including inverse/hyperbolic forms).
pub fn is_trig_function_name(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "csc"
            | "sec"
            | "cot"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
    )
}

/// Check if a function (by symbol id) is trigonometric.
pub fn is_trig_function(ctx: &Context, fn_id: usize) -> bool {
    ctx.builtin_of(fn_id)
        .is_some_and(|builtin| is_trig_function_name(builtin.name()))
}

/// Check if expression is a rational multiple of `pi` (e.g. `pi`, `pi/9`, `2*pi/3`).
pub fn is_pi_constant(ctx: &Context, id: ExprId) -> bool {
    crate::pi_helpers::extract_rational_pi_multiple(ctx, id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn trig_builtin_detection() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)", &mut ctx).expect("parse");
        let fn_id = match ctx.get(expr) {
            Expr::Function(id, _) => *id,
            _ => panic!("expected function"),
        };
        assert!(is_trig_function(&ctx, fn_id));
    }

    #[test]
    fn non_trig_function_is_rejected() {
        let mut ctx = Context::new();
        let expr = parse("f(x)", &mut ctx).expect("parse");
        let fn_id = match ctx.get(expr) {
            Expr::Function(id, _) => *id,
            _ => panic!("expected function"),
        };
        assert!(!is_trig_function(&ctx, fn_id));
    }

    #[test]
    fn pi_constant_detection() {
        let mut ctx = Context::new();
        let pi = parse("pi", &mut ctx).expect("parse pi");
        let two_pi_over_three = parse("2*pi/3", &mut ctx).expect("parse 2pi/3");
        let x = parse("x", &mut ctx).expect("parse x");

        assert!(is_pi_constant(&ctx, pi));
        assert!(is_pi_constant(&ctx, two_pi_over_three));
        assert!(!is_pi_constant(&ctx, x));
    }
}
