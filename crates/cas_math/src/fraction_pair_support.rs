//! Shared extraction helpers for binary fraction operations.

use crate::fraction_forms::extract_as_fraction;
use cas_ast::views::FractionParts;
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct FractionPair {
    pub sign1: i8,
    pub n1: ExprId,
    pub d1: ExprId,
    pub is_frac1: bool,
    pub sign2: i8,
    pub n2: ExprId,
    pub d2: ExprId,
    pub is_frac2: bool,
}

/// Extract fraction-shaped `(num, den)` components from two operands.
///
/// Uses `FractionParts` first (captures direct and multiplicative fraction
/// shapes), then falls back to `extract_as_fraction` for extra patterns.
pub fn extract_fraction_pair(ctx: &mut Context, l: ExprId, r: ExprId) -> FractionPair {
    let fp_l = FractionParts::from(&*ctx, l);
    let fp_r = FractionParts::from(&*ctx, r);

    let (n1, d1, is_frac1_fp) = fp_l.to_num_den(ctx);
    let (n2, d2, is_frac2_fp) = fp_r.to_num_den(ctx);

    let (n1, d1, is_frac1) = if is_frac1_fp {
        (n1, d1, true)
    } else {
        extract_as_fraction(ctx, l)
    };
    let (n2, d2, is_frac2) = if is_frac2_fp {
        (n2, d2, true)
    } else {
        extract_as_fraction(ctx, r)
    };

    FractionPair {
        sign1: fp_l.sign,
        n1,
        d1,
        is_frac1,
        sign2: fp_r.sign,
        n2,
        d2,
        is_frac2,
    }
}

#[cfg(test)]
mod tests {
    use super::extract_fraction_pair;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn extracts_direct_div_pair() {
        let mut ctx = Context::new();
        let l = parse("a/b", &mut ctx).expect("parse");
        let r = parse("c/d", &mut ctx).expect("parse");
        let pair = extract_fraction_pair(&mut ctx, l, r);
        assert!(pair.is_frac1);
        assert!(pair.is_frac2);
    }

    #[test]
    fn extracts_fallback_mixed_pair() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("1/y", &mut ctx).expect("parse");
        let pair = extract_fraction_pair(&mut ctx, l, r);
        assert!(!pair.is_frac1);
        assert!(pair.is_frac2);
    }

    #[test]
    fn preserves_num_den_ids_for_simple_fraction() {
        let mut ctx = Context::new();
        let l = parse("2/(x+1)", &mut ctx).expect("parse");
        let r = parse("z", &mut ctx).expect("parse");
        let pair = extract_fraction_pair(&mut ctx, l, r);
        assert!(pair.is_frac1);
        assert!(!pair.is_frac2);
    }
}
