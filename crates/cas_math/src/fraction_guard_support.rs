//! Shared guard predicates for fraction-combination rules.

use crate::expr_predicates::{contains_root_term, is_trivial_denom_one};
use cas_ast::{Context, ExprId};

/// Guard: skip when mixing function-containing side with constant-fraction side.
pub fn should_block_function_constant_mix(
    l_has_func: bool,
    r_has_func: bool,
    l_is_const_frac: bool,
    r_is_const_frac: bool,
) -> bool {
    (l_has_func && r_is_const_frac) || (r_has_func && l_is_const_frac)
}

/// Guard: skip when one side is root-containing with trivial denominator and
/// the other side is a non-trivial fraction.
pub fn should_block_root_trivial_fraction_mix(
    ctx: &Context,
    l: ExprId,
    d1: ExprId,
    is_frac1: bool,
    r: ExprId,
    d2: ExprId,
    is_frac2: bool,
) -> bool {
    let l_has_root_trivial = !is_frac1 && contains_root_term(ctx, l);
    let r_has_root_trivial = !is_frac2 && contains_root_term(ctx, r);
    let l_has_real_frac = is_frac1 && !is_trivial_denom_one(ctx, d1);
    let r_has_real_frac = is_frac2 && !is_trivial_denom_one(ctx, d2);

    (l_has_root_trivial && r_has_real_frac) || (r_has_root_trivial && l_has_real_frac)
}

#[cfg(test)]
mod tests {
    use super::{should_block_function_constant_mix, should_block_root_trivial_fraction_mix};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn function_constant_mix_blocks() {
        assert!(should_block_function_constant_mix(true, false, false, true));
        assert!(should_block_function_constant_mix(false, true, true, false));
        assert!(!should_block_function_constant_mix(
            true, false, false, false
        ));
    }

    #[test]
    fn root_trivial_vs_real_fraction_blocks() {
        let mut ctx = Context::new();
        let l = parse("sqrt(2)", &mut ctx).expect("parse");
        let r = parse("1/x", &mut ctx).expect("parse");
        let d1 = parse("1", &mut ctx).expect("parse");
        let d2 = parse("x", &mut ctx).expect("parse");
        assert!(should_block_root_trivial_fraction_mix(
            &ctx, l, d1, false, r, d2, true
        ));
    }

    #[test]
    fn non_blocking_case() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("1/x", &mut ctx).expect("parse");
        let d1 = parse("1", &mut ctx).expect("parse");
        let d2 = parse("x", &mut ctx).expect("parse");
        assert!(!should_block_root_trivial_fraction_mix(
            &ctx, l, d1, false, r, d2, true
        ));
    }
}
