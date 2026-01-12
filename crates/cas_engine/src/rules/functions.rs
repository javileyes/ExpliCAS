use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use num_integer::Integer;
use num_traits::Signed;

define_rule!(EvaluateAbsRule, "Evaluate Absolute Value", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "abs" && args.len() == 1 {
            let arg = args[0];

            // Case 1: abs(number)
            let arg_data = ctx.get(arg).clone();
            if let Expr::Number(n) = arg_data {
                // Always evaluate to positive number
                let abs_val = ctx.add(Expr::Number(n.abs()));
                return Some(Rewrite::new(abs_val).desc(format!("abs({}) = {}", n, n.abs())));
            }

            // Case 2: abs(-x) -> abs(x)
            if let Expr::Neg(inner) = ctx.get(arg) {
                // If inner is a number, we can simplify fully: abs(-5) -> 5
                let inner_data = ctx.get(*inner).clone();
                if let Expr::Number(n) = inner_data {
                    let abs_val = ctx.add(Expr::Number(n.clone())); // n is already positive if it was inside Neg? No, Neg(5) means -5.
                                                                    // Wait, Expr::Neg(inner) means the expression is -inner.
                                                                    // If inner is 5, then arg is -5.
                                                                    // But we already handled Expr::Number above.
                                                                    // Expr::Number(-5) is a single node.
                                                                    // Expr::Neg(Expr::Number(5)) is also possible depending on parser/simplifier.
                                                                    // Let's handle it.
                    return Some(Rewrite::new(abs_val).desc(format!("abs(-{}) = {}", n, n)));
                }

                let abs_inner = ctx.add(Expr::Function("abs".to_string(), vec![*inner]));
                return Some(Rewrite::new(abs_inner).desc("abs(-x) = abs(x)"));
            }
        }
    }
    None
});

/// V2.14.20: Simplify absolute value under positivity
/// |x| → x when x > 0 is proven or assumed (depending on DomainMode)
pub struct AbsPositiveSimplifyRule;

impl crate::rule::Rule for AbsPositiveSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Positivity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_positive;

        // Match abs(inner)
        let inner = match ctx.get(expr).clone() {
            Expr::Function(name, args) if name == "abs" && args.len() == 1 => args[0],
            _ => return None,
        };

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, inner, vd);

        match dm {
            DomainMode::Strict | DomainMode::Generic => {
                // Only simplify if proven positive or implied by global requires
                let is_implied = if pos != Proof::Proven {
                    // V2.14.21: Check if Positive(inner) is implied by global requires
                    parent_ctx.root_expr().is_some_and(|root| {
                        let id = crate::implicit_domain::infer_implicit_domain(ctx, root, vd);
                        let dc = crate::implicit_domain::DomainContext::new(
                            id.conditions().iter().cloned().collect(),
                        );
                        let cond = crate::implicit_domain::ImplicitCondition::Positive(inner);
                        dc.is_condition_implied(ctx, &cond)
                    })
                } else {
                    true
                };

                if !is_implied {
                    return None;
                }
                // V2.14.20: .local(abs_id, inner_id) to capture correct step focus
                Some(
                    Rewrite::new(inner)
                        .desc("|x| = x for x > 0")
                        .local(expr, inner),
                )
            }
            DomainMode::Assume => {
                // In Assume mode: if proven, no warning; if not, emit assumption
                if pos == Proof::Proven {
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x for x > 0")
                            .local(expr, inner),
                    )
                } else {
                    // Emit positive_assumed warning
                    // V2.14.20: .local(abs_id, inner_id) to capture correct step focus
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x (assuming x > 0)")
                            .local(expr, inner)
                            .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                                ctx, inner,
                            )),
                    )
                }
            }
        }
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    // V2.14.20: Run in POST phase only so |a| created by LogPowerRule exists first
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    // Ensure step is visible - domain simplification is didactically important
    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

define_rule!(
    AbsSquaredRule,
    "Abs Squared Identity",
    Some(vec!["Pow"]),
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE, // Exclude POST to prevent loop with SimplifySqrtOddPowerRule
    |ctx, expr| {
        // abs(x)^2 -> x^2
        // General: abs(x)^(2k) -> x^(2k) for integer k
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let base_data = ctx.get(base).clone();
            if let Expr::Function(name, args) = base_data {
                if name == "abs" && args.len() == 1 {
                    let inner = args[0];

                    // Check if exponent is an even integer
                    let exp_data = ctx.get(exp).clone();
                    if let Expr::Number(n) = exp_data {
                        if n.is_integer() && n.to_integer().is_even() {
                            // abs(x)^even -> x^even
                            let new_expr = ctx.add(Expr::Pow(inner, exp));
                            return Some(
                                Rewrite::new(new_expr).desc(format!("|x|^{} = x^{}", n, n)),
                            );
                        }
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    SimplifySqrtSquareRule,
    "Simplify Square Root of Square",
    |ctx, expr| {
        // sqrt(x^2) -> |x|
        // Also handles Pow(x, 1/2) of Pow(x, 2)

        let inner = if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "sqrt" && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = ctx.get(expr) {
            // Check if exponent is 1/2
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    Some(*b)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(inner) = inner {
            // Check if inner is x^2 (or x^even_integer)
            if let Expr::Pow(base, exp) = ctx.get(inner) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    // Check if exponent is 2
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                        // sqrt(base^2) -> |base|
                        let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![*base]));
                        return Some(Rewrite::new(abs_base).desc("sqrt(x^2) = |x|"));
                    }
                }
            }
        }
        None
    }
);

// SimplifySqrtOddPowerRule: x^(n/2) -> |x|^k * sqrt(x) where n = 2k+1 (odd >= 3)
// Works on canonicalized form: sqrt(x^3) becomes x^(3/2) before reaching this rule
// Examples:
//   x^(3/2) -> |x| * sqrt(x)     (n=3, k=1)
//   x^(5/2) -> |x|^2 * sqrt(x)   (n=5, k=2)
//   x^(7/2) -> |x|^3 * sqrt(x)   (n=7, k=3)
define_rule!(
    SimplifySqrtOddPowerRule,
    "Simplify Odd Half-Integer Power",
    Some(vec!["Pow"]), // Only match Pow expressions
    PhaseMask::POST,   // Run in POST phase after canonicalization is done
    |ctx, expr| {
        use num_traits::ToPrimitive;

        // Match Pow(base, exp) where exp = n/2 with n odd >= 3
        let (base, k) = if let Expr::Pow(b, e) = ctx.get(expr) {
            let base = *b;
            if let Expr::Number(exp) = ctx.get(*e) {
                // Check if exp = n/2 where n is odd integer >= 3
                // That means denom = 2 and numer is odd >= 3
                let numer = exp.numer().to_i64()?;
                let denom = exp.denom().to_i64()?;

                if denom == 2 && numer >= 3 && numer % 2 == 1 {
                    // n = numer, k = (n-1)/2
                    let k = (numer - 1) / 2;
                    (Some(base), Some((numer, k)))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        if let (Some(base), Some((n, k))) = (base, k) {
            // Build: |x|^k * sqrt(x)
            let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![base]));
            let sqrt_base = ctx.add(Expr::Function("sqrt".to_string(), vec![base]));

            let result = if k == 1 {
                // |x| * sqrt(x)
                ctx.add(Expr::Mul(abs_base, sqrt_base))
            } else {
                // |x|^k * sqrt(x)
                let k_expr = ctx.num(k);
                let abs_pow_k = ctx.add(Expr::Pow(abs_base, k_expr));
                ctx.add(Expr::Mul(abs_pow_k, sqrt_base))
            };

            return Some(Rewrite::new(result).desc(format!("x^({}/2) = |x|^{} * √x", n, k)));
        }

        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_ast::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_evaluate_abs() {
        let mut ctx = Context::new();
        let rule = EvaluateAbsRule;

        // abs(-5) -> 5
        // Note: Parser might produce Number(-5) or Neg(Number(5)).
        // Our parser likely produces Number(-5) for literals.
        let expr1 = parse("abs(-5)", &mut ctx).expect("Failed to parse abs(-5)");
        let rewrite1 = rule
            .apply(
                &mut ctx,
                expr1,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite1.new_expr
                }
            ),
            "5"
        );

        // abs(5) -> 5
        let expr2 = parse("abs(5)", &mut ctx).expect("Failed to parse abs(5)");
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "5"
        );

        // abs(-x) -> abs(x)
        let expr3 = parse("abs(-x)", &mut ctx).expect("Failed to parse abs(-x)");
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "|x|"
        );
    }
}

// EvaluateMetaFunctionsRule: Handles meta functions that operate on expressions
// - simplify(expr) → expr (already simplified by bottom-up processing)
// - factor(expr) → expr (factoring is done by other rules during simplification)
// - expand(expr) → expanded version (calls actual expand logic)
define_rule!(
    EvaluateMetaFunctionsRule,
    "Evaluate Meta Functions",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr).clone() {
            if args.len() == 1 {
                let arg = args[0];
                match name.as_str() {
                    // simplify() is transparent - argument already processed
                    "simplify" => {
                        return Some(Rewrite::new(arg).desc("simplify(x) = x (already processed)"));
                    }
                    // factor() calls actual factorization logic
                    "factor" => {
                        let factored = crate::factor::factor(ctx, arg);
                        if factored != arg {
                            return Some(Rewrite::new(factored).desc("factor(x) → factored form"));
                        }
                        // No change - return as-is (irreducible)
                        return Some(Rewrite::new(arg).desc("factor(x) = x (irreducible)"));
                    }
                    // expand() needs to call actual expansion logic
                    "expand" => {
                        let expanded = crate::expand::expand(ctx, arg);
                        return Some(Rewrite::new(expanded).desc("expand(x) → expanded form"));
                    }
                    // expand_log is handled in eval.rs BEFORE simplification to ensure
                    // goal=ExpandedLog is set before any rules run
                    _ => {}
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifySqrtSquareRule)); // Must go BEFORE EvaluateAbsRule to catch sqrt(x^2) early
    simplifier.add_rule(Box::new(SimplifySqrtOddPowerRule)); // sqrt(x^3) -> |x| * sqrt(x)
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(AbsPositiveSimplifyRule)); // V2.14.20: |x| -> x when x > 0
    simplifier.add_rule(Box::new(AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateMetaFunctionsRule)); // Make simplify/factor/expand transparent
}
