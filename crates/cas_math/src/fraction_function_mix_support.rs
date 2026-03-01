//! Guards for function/constant-fraction mixing in fraction add/sub rules.

use crate::expr_predicates::{contains_function, is_constant_expr, is_constant_fraction};
use crate::fraction_guard_support::should_block_function_constant_mix;
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct AddFunctionConstantMixInput {
    pub l: ExprId,
    pub r: ExprId,
    pub n1: ExprId,
    pub d1: ExprId,
    pub is_frac1: bool,
    pub n2: ExprId,
    pub d2: ExprId,
    pub is_frac2: bool,
}

/// Add-rule guard: block function-containing term combined with constant fraction.
pub fn should_block_add_function_constant_mix(
    ctx: &Context,
    input: AddFunctionConstantMixInput,
) -> bool {
    let l_has_func = contains_function(ctx, input.l);
    let r_has_func = contains_function(ctx, input.r);
    let l_is_const_frac = input.is_frac1 && is_constant_fraction(ctx, input.n1, input.d1);
    let r_is_const_frac = input.is_frac2 && is_constant_fraction(ctx, input.n2, input.d2);
    should_block_function_constant_mix(l_has_func, r_has_func, l_is_const_frac, r_is_const_frac)
}

/// Sub-rule guard: both sides are already fraction terms.
pub fn should_block_sub_function_constant_mix(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    n1: ExprId,
    d1: ExprId,
    n2: ExprId,
    d2: ExprId,
) -> bool {
    let l_has_func = contains_function(ctx, l);
    let r_has_func = contains_function(ctx, r);
    let l_is_const_frac = is_constant_expr(ctx, n1) && is_constant_expr(ctx, d1);
    let r_is_const_frac = is_constant_expr(ctx, n2) && is_constant_expr(ctx, d2);
    should_block_function_constant_mix(l_has_func, r_has_func, l_is_const_frac, r_is_const_frac)
}

#[cfg(test)]
mod tests {
    use super::{
        should_block_add_function_constant_mix, should_block_sub_function_constant_mix,
        AddFunctionConstantMixInput,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn add_blocks_function_plus_constant_fraction_mix() {
        let mut ctx = Context::new();
        let l = parse("arctan(x)", &mut ctx).expect("parse");
        let r = parse("pi/2", &mut ctx).expect("parse");
        let n1 = parse("arctan(x)", &mut ctx).expect("parse");
        let d1 = parse("1", &mut ctx).expect("parse");
        let n2 = parse("pi", &mut ctx).expect("parse");
        let d2 = parse("2", &mut ctx).expect("parse");
        assert!(should_block_add_function_constant_mix(
            &ctx,
            AddFunctionConstantMixInput {
                l,
                r,
                n1,
                d1,
                is_frac1: false,
                n2,
                d2,
                is_frac2: true,
            }
        ));
    }

    #[test]
    fn sub_allows_nonconstant_fraction_pair() {
        let mut ctx = Context::new();
        let l = parse("f(x)/y", &mut ctx).expect("parse");
        let r = parse("g(x)/z", &mut ctx).expect("parse");
        let n1 = parse("f(x)", &mut ctx).expect("parse");
        let d1 = parse("y", &mut ctx).expect("parse");
        let n2 = parse("g(x)", &mut ctx).expect("parse");
        let d2 = parse("z", &mut ctx).expect("parse");
        assert!(!should_block_sub_function_constant_mix(
            &ctx, l, r, n1, d1, n2, d2
        ));
    }
}
