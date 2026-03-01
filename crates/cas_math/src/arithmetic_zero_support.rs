//! Pattern helpers for arithmetic zero identities.

use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulZeroPattern {
    pub other: ExprId,
    pub zero_on_lhs: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivZeroNumeratorPattern {
    pub denominator: ExprId,
    pub denominator_is_literal_zero: bool,
}

/// Match `0 * e` or `e * 0`.
pub fn match_mul_zero_pattern(ctx: &Context, expr: ExprId) -> Option<MulZeroPattern> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let lhs = *lhs;
    let rhs = *rhs;

    let lhs_is_zero = matches!(ctx.get(lhs), Expr::Number(n) if n.is_zero());
    let rhs_is_zero = matches!(ctx.get(rhs), Expr::Number(n) if n.is_zero());
    if !(lhs_is_zero || rhs_is_zero) {
        return None;
    }

    let other = if lhs_is_zero { rhs } else { lhs };
    Some(MulZeroPattern {
        other,
        zero_on_lhs: lhs_is_zero,
    })
}

/// Match `0 / d` and expose denominator info.
pub fn match_div_zero_numerator_pattern(
    ctx: &Context,
    expr: ExprId,
) -> Option<DivZeroNumeratorPattern> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };

    let num_is_zero = matches!(ctx.get(*num), Expr::Number(n) if n.is_zero());
    if !num_is_zero {
        return None;
    }

    Some(DivZeroNumeratorPattern {
        denominator: *den,
        denominator_is_literal_zero: matches!(ctx.get(*den), Expr::Number(d) if d.is_zero()),
    })
}

#[cfg(test)]
mod tests {
    use super::{match_div_zero_numerator_pattern, match_mul_zero_pattern};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn matches_mul_zero_both_orders() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let left = parse("0*x", &mut ctx).expect("parse");
        let right = parse("x*0", &mut ctx).expect("parse");

        let p1 = match_mul_zero_pattern(&ctx, left).expect("pattern");
        assert_eq!(p1.other, x);
        let p2 = match_mul_zero_pattern(&ctx, right).expect("pattern");
        assert_eq!(p2.other, x);
    }

    #[test]
    fn matches_div_zero_numerator_and_marks_zero_den() {
        let mut ctx = Context::new();
        let safe = parse("0/x", &mut ctx).expect("parse");
        let unsafe_case = parse("0/0", &mut ctx).expect("parse");

        let p1 = match_div_zero_numerator_pattern(&ctx, safe).expect("pattern");
        assert!(!p1.denominator_is_literal_zero);

        let p2 = match_div_zero_numerator_pattern(&ctx, unsafe_case).expect("pattern");
        assert!(p2.denominator_is_literal_zero);
    }
}
