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
