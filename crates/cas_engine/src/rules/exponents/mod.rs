mod power_rules;
mod rationalization;
mod simplification;

pub use power_rules::{
    EvaluatePowerRule, NegativeExponentNormalizationRule, PowerPowerRule, ProductPowerRule,
    ProductSameExponentRule, QuotientSameExponentRule, RootPowCancelRule,
};
pub use rationalization::{
    CubeRootDenRationalizeRule, PowPowCancelReciprocalRule, RationalizeLinearSqrtDenRule,
    RationalizeSumOfSqrtsDenRule, ReciprocalSqrtCanonRule, RootMergeDivRule, RootMergeMulRule,
};
pub use simplification::{
    EvenPowSubSwapRule, ExpQuotientRule, IdentityPowerRule, MulNaryCombinePowersRule,
    NegativeBasePowerRule, PowerProductRule, PowerQuotientRule,
};

use crate::build::mul2_raw;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

/// Helper: Add two exponents, folding if both are constants
/// This prevents ugly exponents like x^(1+2) and produces x^3 instead
pub(super) fn add_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let sum = n1 + n2;
        ctx.add(Expr::Number(sum))
    } else {
        ctx.add(Expr::Add(e1, e2))
    }
}

/// Helper: Multiply two exponents, folding if both are constants
/// This prevents ugly exponents like x^(2*3) and produces x^6 instead
pub(super) fn mul_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let prod = n1 * n2;
        ctx.add(Expr::Number(prod))
    } else {
        mul2_raw(ctx, e1, e2)
    }
}

/// Check if an expression contains a numeric factor at the top level
pub(super) fn has_numeric_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Mul(l, r) => {
            matches!(ctx.get(*l), Expr::Number(_)) || matches!(ctx.get(*r), Expr::Number(_))
        }
        _ => false,
    }
}

pub(super) fn extract_root_factor(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    if n.is_zero() {
        return (BigInt::zero(), BigInt::one());
    }
    if n.is_one() {
        return (BigInt::one(), BigInt::one());
    }

    let sign = if n.is_negative() { -1 } else { 1 };
    let mut n_abs = n.abs();

    let mut outside = BigInt::one();
    let mut inside = BigInt::one();

    // Trial division - check 2
    let mut count = 0;
    while n_abs.is_even() {
        count += 1;
        n_abs /= 2;
    }
    if count > 0 {
        let out_exp = count / k;
        let in_exp = count % k;
        if out_exp > 0 {
            outside *= BigInt::from(2).pow(out_exp);
        }
        if in_exp > 0 {
            inside *= BigInt::from(2).pow(in_exp);
        }
    }

    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        if (&n_abs % &d).is_zero() {
            let mut count = 0;
            while (&n_abs % &d).is_zero() {
                count += 1;
                n_abs /= &d;
            }
            let out_exp = count / k;
            let in_exp = count % k;
            if out_exp > 0 {
                outside *= d.pow(out_exp);
            }
            if in_exp > 0 {
                inside *= d.pow(in_exp);
            }
        }
        d += 2;
    }

    if n_abs > BigInt::one() {
        inside *= n_abs;
    }

    // Handle sign
    if sign == -1 {
        if !k.is_multiple_of(2) {
            outside = -outside;
        } else {
            inside = -inside;
        }
    }

    (outside, inside)
}

/// Check if distributing a fractional exponent (1/n) over a product is safe.
/// Returns true if:
/// 1. Base is purely numeric (no variables), OR
/// 2. All variable factors have powers that are exact multiples of n
///    (e.g., x^2 under sqrt is safe because 2 % 2 == 0)
pub(super) fn can_distribute_root_safely(
    ctx: &Context,
    expr: ExprId,
    root_index: &num_bigint::BigInt,
) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) => root_index == &num_bigint::BigInt::from(1),
        Expr::Pow(base, exp) => {
            if is_purely_numeric(ctx, *base) {
                return true;
            }
            if let Expr::Number(exp_num) = ctx.get(*exp) {
                if exp_num.is_integer() {
                    let exp_int = exp_num.to_integer();
                    return (&exp_int % root_index).is_zero();
                }
            }
            false
        }
        Expr::Mul(l, r) => {
            can_distribute_root_safely(ctx, *l, root_index)
                && can_distribute_root_safely(ctx, *r, root_index)
        }
        Expr::Div(l, r) => {
            can_distribute_root_safely(ctx, *l, root_index)
                && can_distribute_root_safely(ctx, *r, root_index)
        }
        Expr::Neg(inner) => can_distribute_root_safely(ctx, *inner, root_index),
        _ => false,
    }
}

/// Check if expression is purely numeric (no variables/constants)
fn is_purely_numeric(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_purely_numeric(ctx, *l) && is_purely_numeric(ctx, *r)
        }
        Expr::Neg(inner) => is_purely_numeric(ctx, *inner),
        Expr::Hold(inner) => is_purely_numeric(ctx, *inner),
        Expr::Function(_, args) => args.iter().all(|a| is_purely_numeric(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().all(|e| is_purely_numeric(ctx, *e)),
        Expr::SessionRef(_) => false,
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // N-ary mul combine rule: handles (a*b)*a^2 → a^3*b
    simplifier.add_rule(Box::new(MulNaryCombinePowersRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(ProductSameExponentRule));
    simplifier.add_rule(Box::new(QuotientSameExponentRule)); // a^n / b^n = (a/b)^n
                                                             // V2.14.45: RootPowCancelRule BEFORE PowerPowerRule for (x^n)^(1/n) with parity
    simplifier.add_rule(Box::new(RootPowCancelRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(NegativeExponentNormalizationRule)); // x^(-n) → 1/x^n
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(ExpQuotientRule)); // V2.14.45: e^a/e^b → e^(a-b)

    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(NegativeBasePowerRule));
    simplifier.add_rule(Box::new(EvenPowSubSwapRule)); // (b-a)^even → (a-b)^even
                                                       // Rationalize sqrt denominators: 1/(sqrt(t)+c) → (sqrt(t)-c)/(t-c²)
    simplifier.add_rule(Box::new(RationalizeLinearSqrtDenRule));
    // Rationalize sum of sqrts: 1/(sqrt(p)+sqrt(q)) → (sqrt(p)-sqrt(q))/(p-q)
    simplifier.add_rule(Box::new(RationalizeSumOfSqrtsDenRule));
    // Rationalize cube root: 1/(1+u^(1/3)) → (1-u^(1/3)+u^(2/3))/(1+u)
    simplifier.add_rule(Box::new(CubeRootDenRationalizeRule));
    // Merge sqrt products: sqrt(a)*sqrt(b) → sqrt(a*b) (with requires a≥0, b≥0)
    simplifier.add_rule(Box::new(RootMergeMulRule));
    // Merge sqrt quotients: sqrt(a)/sqrt(b) → sqrt(a/b) (with requires a≥0, b>0)
    simplifier.add_rule(Box::new(RootMergeDivRule));
    // Cancel reciprocal exponents: (u^y)^(1/y) → u (with requires u>0, y≠0)
    simplifier.add_rule(Box::new(PowPowCancelReciprocalRule));
    // Canonicalize reciprocal sqrt: 1/√x → x^(-1/2), √x/x → x^(-1/2)
    simplifier.add_rule(Box::new(ReciprocalSqrtCanonRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_product_power() {
        let mut ctx = Context::new();
        let rule = ProductPowerRule;

        // x^2 * x^3 -> x^(2+3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));
        let expr = ctx.add(Expr::Mul(x2, x3));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^5"
        );

        // x * x -> x^2
        let expr2 = ctx.add(Expr::Mul(x, x));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x^2"
        );
    }

    #[test]
    fn test_power_power() {
        let mut ctx = Context::new();
        let rule = PowerPowerRule;

        // (x^2)^3 -> x^(2*3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let expr = ctx.add(Expr::Pow(x2, three));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^6"
        );
    }

    #[test]
    fn test_zero_one_power() {
        let mut ctx = Context::new();
        let rule = IdentityPowerRule;

        // x^0 -> 1
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Pow(x, zero));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );

        // x^1 -> x
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Pow(x, one));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x"
        );
    }
}
