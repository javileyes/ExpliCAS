use cas_ast::{Constant, Context, Expr, ExprId};
use num_traits::Signed;

/// Sign of infinity: positive (+∞) or negative (−∞).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfSign {
    Pos,
    Neg,
}

/// Classification of an expression's finiteness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finiteness {
    /// Expression is a known finite value (number, π, e, i)
    FiniteLiteral,
    /// Expression is ±∞ with known sign
    Infinity(InfSign),
    /// Unknown finiteness - could be finite, infinite, or undefined
    Unknown,
}

/// Detect if an expression is ±∞.
pub fn inf_sign(ctx: &Context, id: ExprId) -> Option<InfSign> {
    match ctx.get(id) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Constant(Constant::Infinity) => Some(InfSign::Neg),
            _ => None,
        },
        _ => None,
    }
}

/// Construct ±∞.
pub fn mk_infinity(ctx: &mut Context, sign: InfSign) -> ExprId {
    let inf = ctx.add(Expr::Constant(Constant::Infinity));
    match sign {
        InfSign::Pos => inf,
        InfSign::Neg => ctx.add(Expr::Neg(inf)),
    }
}

/// Construct Undefined (for indeterminate forms).
pub fn mk_undefined(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Undefined))
}

/// Check if an expression is a known finite literal.
///
/// Conservative policy: only true for expressions we KNOW are finite:
/// - Numbers (BigRational)
/// - Constants that are not Infinity or Undefined (π, e, i)
pub fn is_finite_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        _ => false,
    }
}

/// Classify an expression's finiteness.
pub fn classify_finiteness(ctx: &Context, id: ExprId) -> Finiteness {
    if let Some(sign) = inf_sign(ctx, id) {
        return Finiteness::Infinity(sign);
    }

    if is_finite_literal(ctx, id) {
        return Finiteness::FiniteLiteral;
    }

    Finiteness::Unknown
}

/// Check if expression is a positive finite literal (for sign determination).
pub fn is_positive_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => n.is_positive(),
        Expr::Constant(c) => matches!(c, Constant::Pi | Constant::E), // π and e are positive
        _ => false,
    }
}

/// Check if expression is a negative finite literal.
pub fn is_negative_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => n.is_negative(),
        Expr::Neg(inner) => is_positive_literal(ctx, *inner),
        _ => false,
    }
}

/// Collect additive terms with their signs (iterative, handles Sub).
pub fn collect_add_terms_with_sign(
    ctx: &Context,
    id: ExprId,
    is_positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    let mut stack = vec![(id, is_positive)];

    while let Some((current, sign)) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, !sign)); // Right side gets inverted sign
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, !sign));
            }
            _ => terms.push((current, sign)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    #[test]
    fn inf_sign_detects_both_orientations() {
        let mut ctx = Context::new();
        let pos = parse_expr(&mut ctx, "infinity");
        let neg = parse_expr(&mut ctx, "-infinity");
        let finite = parse_expr(&mut ctx, "7");

        assert_eq!(inf_sign(&ctx, pos), Some(InfSign::Pos));
        assert_eq!(inf_sign(&ctx, neg), Some(InfSign::Neg));
        assert_eq!(inf_sign(&ctx, finite), None);
    }

    #[test]
    fn classify_finiteness_keeps_unknown_for_undefined_and_vars() {
        let mut ctx = Context::new();
        let n = parse_expr(&mut ctx, "42");
        let inf = parse_expr(&mut ctx, "infinity");
        let x = parse_expr(&mut ctx, "x");
        let undef = parse_expr(&mut ctx, "undefined");

        assert_eq!(classify_finiteness(&ctx, n), Finiteness::FiniteLiteral);
        assert_eq!(
            classify_finiteness(&ctx, inf),
            Finiteness::Infinity(InfSign::Pos)
        );
        assert_eq!(classify_finiteness(&ctx, x), Finiteness::Unknown);
        assert_eq!(classify_finiteness(&ctx, undef), Finiteness::Unknown);
    }

    #[test]
    fn add_term_collection_tracks_signs_through_sub_and_neg() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "a - (b + -c)");
        let mut terms = Vec::new();
        collect_add_terms_with_sign(&ctx, expr, true, &mut terms);

        let mut rendered: Vec<(String, bool)> = terms
            .into_iter()
            .map(|(id, sign)| {
                let name = match ctx.get(id) {
                    Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
                    _ => panic!("expected variable term"),
                };
                (name, sign)
            })
            .collect();
        rendered.sort();

        assert_eq!(
            rendered,
            vec![
                ("a".to_string(), true),
                ("b".to_string(), false),
                ("c".to_string(), true)
            ]
        );
    }

    #[test]
    fn literal_sign_helpers_handle_numbers_and_constants() {
        let mut ctx = Context::new();
        let pos = parse_expr(&mut ctx, "pi");
        let neg = parse_expr(&mut ctx, "-2");
        let imag = parse_expr(&mut ctx, "i");

        assert!(is_positive_literal(&ctx, pos));
        assert!(is_negative_literal(&ctx, neg));
        assert!(!is_positive_literal(&ctx, imag));
        assert!(!is_negative_literal(&ctx, imag));
    }
}
