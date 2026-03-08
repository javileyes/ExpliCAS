//! Guard policies for Add/Sub fraction-pair rewrites.

use crate::fraction_function_mix_support::{
    should_block_add_function_constant_mix, should_block_sub_function_constant_mix,
    AddFunctionConstantMixInput,
};
use crate::fraction_guard_support::should_block_root_trivial_fraction_mix;
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct AddFractionPairGuardInput {
    pub l: ExprId,
    pub r: ExprId,
    pub n1: ExprId,
    pub d1: ExprId,
    pub is_frac1: bool,
    pub n2: ExprId,
    pub d2: ExprId,
    pub is_frac2: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SubFractionPairGuardInput {
    pub l: ExprId,
    pub r: ExprId,
    pub n1: ExprId,
    pub d1: ExprId,
    pub is_frac1: bool,
    pub n2: ExprId,
    pub d2: ExprId,
    pub is_frac2: bool,
    pub inside_trig: bool,
}

/// True when AddFractions-style rewrite should be blocked.
pub fn should_block_add_fraction_pair(ctx: &Context, input: AddFractionPairGuardInput) -> bool {
    if !input.is_frac1 && !input.is_frac2 {
        return true;
    }

    if should_block_add_function_constant_mix(
        ctx,
        AddFunctionConstantMixInput {
            l: input.l,
            r: input.r,
            n1: input.n1,
            d1: input.d1,
            is_frac1: input.is_frac1,
            n2: input.n2,
            d2: input.d2,
            is_frac2: input.is_frac2,
        },
    ) {
        return true;
    }

    should_block_root_trivial_fraction_mix(
        ctx,
        input.l,
        input.d1,
        input.is_frac1,
        input.r,
        input.d2,
        input.is_frac2,
    )
}

/// True when SubFractions-style rewrite should be blocked.
pub fn should_block_sub_fraction_pair(ctx: &Context, input: SubFractionPairGuardInput) -> bool {
    if !input.is_frac1 || !input.is_frac2 {
        return true;
    }
    if input.inside_trig {
        return true;
    }
    if should_block_sub_function_constant_mix(
        ctx, input.l, input.r, input.n1, input.d1, input.n2, input.d2,
    ) {
        return true;
    }
    should_block_root_trivial_fraction_mix(
        ctx,
        input.l,
        input.d1,
        input.is_frac1,
        input.r,
        input.d2,
        input.is_frac2,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        should_block_add_fraction_pair, should_block_sub_fraction_pair, AddFractionPairGuardInput,
        SubFractionPairGuardInput,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn add_blocks_when_neither_side_is_fraction() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("y", &mut ctx).expect("parse");
        let n1 = parse("x", &mut ctx).expect("parse");
        let d1 = parse("1", &mut ctx).expect("parse");
        let n2 = parse("y", &mut ctx).expect("parse");
        let d2 = parse("1", &mut ctx).expect("parse");
        assert!(should_block_add_fraction_pair(
            &ctx,
            AddFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1: false,
                n2,
                d2,
                is_frac2: false,
            }
        ));
    }

    #[test]
    fn add_allows_regular_fraction_pair() {
        let mut ctx = Context::new();
        let l = parse("a/b", &mut ctx).expect("parse");
        let r = parse("c/d", &mut ctx).expect("parse");
        let n1 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("b", &mut ctx).expect("parse");
        let n2 = parse("c", &mut ctx).expect("parse");
        let d2 = parse("d", &mut ctx).expect("parse");
        assert!(!should_block_add_fraction_pair(
            &ctx,
            AddFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1: true,
                n2,
                d2,
                is_frac2: true,
            }
        ));
    }

    #[test]
    fn sub_blocks_inside_trig_context() {
        let mut ctx = Context::new();
        let l = parse("a/b", &mut ctx).expect("parse");
        let r = parse("c/d", &mut ctx).expect("parse");
        let n1 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("b", &mut ctx).expect("parse");
        let n2 = parse("c", &mut ctx).expect("parse");
        let d2 = parse("d", &mut ctx).expect("parse");
        assert!(should_block_sub_fraction_pair(
            &ctx,
            SubFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1: true,
                n2,
                d2,
                is_frac2: true,
                inside_trig: true,
            }
        ));
    }
}
