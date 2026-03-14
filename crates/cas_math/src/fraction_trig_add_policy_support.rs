//! Policy helpers for fraction addition inside trigonometric arguments.

use crate::expr_classify::is_pi_constant;
use crate::expr_predicates::is_constant_expr;
use cas_ast::{Context, Expr, ExprId};

/// Decide whether to block combining symbolic + pi-constant additions inside trig args.
///
/// Mirrors the policy:
/// - If not inside trig, never block.
/// - If both sides are constants, never block.
/// - Otherwise, block symbolic + pi-constant mixes to preserve identity-friendly form.
pub fn should_block_symbolic_plus_pi_add_inside_trig(
    inside_trig: bool,
    l_is_const: bool,
    r_is_const: bool,
    l_is_pi: bool,
    r_is_pi: bool,
) -> bool {
    if !inside_trig {
        return false;
    }
    if l_is_const && r_is_const {
        return false;
    }
    (l_is_pi && !r_is_const) || (r_is_pi && !l_is_const)
}

/// True when any addition rewrite inside trig should be blocked to preserve
/// a visible `+ k*pi/2` or `+ pi` shape for downstream trig identities.
pub fn should_block_general_fraction_add_rewrite(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    inside_trig: bool,
) -> bool {
    let l_is_const = is_constant_expr(ctx, l);
    let r_is_const = is_constant_expr(ctx, r);
    let l_is_pi = is_pi_constant(ctx, l);
    let r_is_pi = is_pi_constant(ctx, r);
    should_block_symbolic_plus_pi_add_inside_trig(
        inside_trig,
        l_is_const,
        r_is_const,
        l_is_pi,
        r_is_pi,
    )
}

/// True when a numeric-denominator addition rewrite is allowed.
///
/// Returns `false` when denominators are not both numeric, or when trig-aware
/// blocking policy says to preserve the original structure.
pub fn should_allow_numeric_fraction_add_rewrite(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    d1: ExprId,
    d2: ExprId,
    inside_trig: bool,
) -> bool {
    let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));
    if !is_numeric(d1) || !is_numeric(d2) {
        return false;
    }

    let l_is_const = is_constant_expr(ctx, l);
    let r_is_const = is_constant_expr(ctx, r);
    let l_is_pi = is_pi_constant(ctx, l);
    let r_is_pi = is_pi_constant(ctx, r);
    if should_block_symbolic_plus_pi_add_inside_trig(
        inside_trig,
        l_is_const,
        r_is_const,
        l_is_pi,
        r_is_pi,
    ) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::{
        should_allow_numeric_fraction_add_rewrite, should_block_general_fraction_add_rewrite,
        should_block_symbolic_plus_pi_add_inside_trig,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn outside_trig_never_blocks() {
        assert!(!should_block_symbolic_plus_pi_add_inside_trig(
            false, false, false, true, false
        ));
    }

    #[test]
    fn inside_trig_both_constants_allowed() {
        assert!(!should_block_symbolic_plus_pi_add_inside_trig(
            true, true, true, true, true
        ));
    }

    #[test]
    fn inside_trig_symbol_plus_pi_blocks() {
        assert!(should_block_symbolic_plus_pi_add_inside_trig(
            true, false, false, true, false
        ));
        assert!(should_block_symbolic_plus_pi_add_inside_trig(
            true, false, false, false, true
        ));
    }

    #[test]
    fn numeric_denominators_outside_trig_are_allowed() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("pi/6", &mut ctx).expect("parse");
        let d1 = parse("2", &mut ctx).expect("parse");
        let d2 = parse("3", &mut ctx).expect("parse");
        assert!(should_allow_numeric_fraction_add_rewrite(
            &ctx, l, r, d1, d2, false
        ));
    }

    #[test]
    fn numeric_denominators_inside_trig_symbol_plus_pi_blocks() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("pi/6", &mut ctx).expect("parse");
        let d1 = parse("2", &mut ctx).expect("parse");
        let d2 = parse("3", &mut ctx).expect("parse");
        assert!(!should_allow_numeric_fraction_add_rewrite(
            &ctx, l, r, d1, d2, true
        ));
    }

    #[test]
    fn general_inside_trig_symbol_plus_pi_blocks_even_with_symbolic_denominator() {
        let mut ctx = Context::new();
        let l = parse("1/x + 1/(x+1)", &mut ctx).expect("parse");
        let r = parse("pi", &mut ctx).expect("parse");
        assert!(should_block_general_fraction_add_rewrite(&ctx, l, r, true));
    }
}
