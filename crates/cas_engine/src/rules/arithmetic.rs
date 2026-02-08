use crate::define_rule;
use crate::helpers::{as_add, as_div, as_mul, as_sub};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::Expr;
use num_traits::{One, Zero};

define_rule!(
    AddZeroRule,
    "Identity Property of Addition",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let (lhs, rhs) = as_add(ctx, expr)?;
        if let Expr::Number(n) = ctx.get(rhs) {
            if n.is_zero() {
                return Some(Rewrite::new(lhs).desc("x + 0 = x"));
            }
        }
        if let Expr::Number(n) = ctx.get(lhs) {
            if n.is_zero() {
                return Some(Rewrite::new(rhs).desc("0 + x = x"));
            }
        }
        None
    }
);

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let (lhs, rhs) = as_mul(ctx, expr)?;
        if let Expr::Number(n) = ctx.get(rhs) {
            if n.is_one() {
                return Some(Rewrite::new(lhs).desc("x * 1 = x"));
            }
        }
        if let Expr::Number(n) = ctx.get(lhs) {
            if n.is_one() {
                return Some(Rewrite::new(rhs).desc("1 * x = x"));
            }
        }
        None
    }
);

// MulZeroRule: 0*e → 0
// Domain Mode Policy: 0*e → 0 changes the domain of definition if e can be undefined.
// Uses ConditionClass taxonomy:
// - Strict: only apply if other factor has no undefined risk
// - Generic: apply with Defined(e) assumption (Definability class)
// - Assume: apply with Defined(e) assumption
define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::assumptions::ConditionClass;

        // Helper: check if expression contains any Div with non-literal denominator
        // Delegates to canonical implementation that handles Hold/Matrix
        let has_undefined_risk = |ctx: &cas_ast::Context, expr: cas_ast::ExprId| -> bool {
            crate::collect::has_undefined_risk(ctx, expr)
        };

        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            // Check if either side is zero literal
            let lhs_is_zero = matches!(ctx.get(lhs), Expr::Number(n) if n.is_zero());
            let rhs_is_zero = matches!(ctx.get(rhs), Expr::Number(n) if n.is_zero());

            if !(lhs_is_zero || rhs_is_zero) {
                return None;
            }

            // The "other" side: 0 * other
            let other = if lhs_is_zero { rhs } else { lhs };
            let has_risk = has_undefined_risk(ctx, other);

            // Use ConditionClass gate: Defined is Definability class
            let allowed = if has_risk {
                parent_ctx
                    .domain_mode()
                    .allows_unproven(ConditionClass::Definability)
            } else {
                true // No risk = always allowed
            };

            if !allowed {
                return None; // Strict mode: don't simplify if has risk
            }

            // Build assumption events if has risk and allowed
            let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if has_risk {
                smallvec::smallvec![crate::assumptions::AssumptionEvent::defined(ctx, other)]
            } else {
                smallvec::SmallVec::new()
            };

            let description = if lhs_is_zero {
                "0 * x = 0".to_string()
            } else {
                "x * 0 = 0".to_string()
            };

            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc(description).assume_all(assumption_events));
        }
        None
    }
);

// DivZeroRule: 0/d → 0
// Domain Mode Policy: 0/d → 0 changes the domain of definition if d can be 0.
// Uses ConditionClass taxonomy via can_cancel_factor():
// - Strict: only apply if prove_nonzero(d) == Proven
// - Generic: apply with NonZero(d) assumption (Definability class)
// - Assume: apply with NonZero(d) assumption
define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;

        let (num, den) = as_div(ctx, expr)?;

        // Check if numerator is 0
        let num_is_zero = matches!(ctx.get(num), Expr::Number(n) if n.is_zero());
        if !num_is_zero {
            return None;
        }

        // Special case: 0/0 → undefined (all modes)
        if let Expr::Number(d) = ctx.get(den) {
            if d.is_zero() {
                let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                return Some(Rewrite::new(undef).desc("0/0 is undefined"));
            }
        }

        // Use central gate for NonZero condition (Definability class)
        let den_proof = prove_nonzero(ctx, den);
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, den);
        let decision = crate::domain::can_cancel_factor_with_hint(
            parent_ctx.domain_mode(),
            den_proof,
            key,
            den,
            "Zero Property of Division",
        );

        if !decision.allow {
            return None; // Strict mode: don't simplify if not proven
        }

        // Build assumption events if needed
        let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if decision.assumption.is_some() && den_proof != Proof::Proven {
            smallvec::smallvec![crate::assumptions::AssumptionEvent::nonzero(ctx, den)]
        } else {
            smallvec::SmallVec::new()
        };

        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc("0 / d = 0").assume_all(assumption_events))
    }
);

define_rule!(CombineConstantsRule, "Combine Constants", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    // Try Add branch
    if let Some((lhs, rhs)) = as_add(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs).clone(), ctx.get(rhs).clone()) {
            let sum = &n1 + &n2;
            let new_expr = ctx.add(Expr::Number(sum.clone()));
            let description = if n2 < num_rational::BigRational::from_integer(0.into()) {
                let abs_n2 = -&n2;
                format!("{} - {} = {}", n1, abs_n2, sum)
            } else {
                format!("{} + {} = {}", n1, n2, sum)
            };
            return Some(Rewrite::new(new_expr).desc(description));
        }
        // Handle nested: c1 + (c2 + x) -> (c1+c2) + x
        if let Expr::Number(n1) = ctx.get(lhs).clone() {
            if let Some((rl, rr)) = as_add(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(rl).clone() {
                    let sum = &n1 + &n2;
                    let sum_expr = ctx.add(Expr::Number(sum));
                    let new_expr = ctx.add(Expr::Add(sum_expr, rr));
                    return Some(
                        Rewrite::new(new_expr)
                            .desc(format!("Combine nested constants: {} + {}", n1, n2)),
                    );
                }
            }
        }
    }

    // Try Mul branch
    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs).clone(), ctx.get(rhs).clone()) {
            let prod = &n1 * &n2;
            let new_expr = ctx.add(Expr::Number(prod.clone()));
            return Some(Rewrite::new(new_expr).desc(format!("{} * {} = {}", n1, n2, prod)));
        }

        // V2.15.25: Flatten complete Mul chain and combine ALL numeric factors
        let mut factors = Vec::new();
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(id) {
                stack.push(*r);
                stack.push(*l);
            } else {
                factors.push(id);
            }
        }

        let mut numeric_factors: Vec<num_rational::BigRational> = Vec::new();
        let mut non_numeric: Vec<cas_ast::ExprId> = Vec::new();

        for &f in &factors {
            if let Expr::Number(n) = ctx.get(f) {
                numeric_factors.push(n.clone());
            } else {
                non_numeric.push(f);
            }
        }

        if numeric_factors.len() >= 2 {
            let product: num_rational::BigRational = numeric_factors.iter().fold(
                num_rational::BigRational::from_integer(1.into()),
                |acc, n| acc * n
            );

            if product.is_zero() {
                let zero = ctx.num(0);
                return Some(Rewrite::new(zero).desc("0 * x = 0"));
            }

            let mut result: cas_ast::ExprId;
            if product.is_one() && !non_numeric.is_empty() {
                result = non_numeric[0];
                for &f in &non_numeric[1..] {
                    result = smart_mul(ctx, result, f);
                }
            } else if non_numeric.is_empty() {
                result = ctx.add(Expr::Number(product.clone()));
            } else {
                let prod_expr = ctx.add(Expr::Number(product.clone()));
                result = prod_expr;
                for &f in &non_numeric {
                    result = smart_mul(ctx, result, f);
                }
            }

            let nums_str: Vec<String> = numeric_factors.iter().map(|n| format!("{}", n)).collect();
            return Some(Rewrite::new(result).desc(format!(
                "Combine nested constants: {} = {}",
                nums_str.join(" * "),
                product
            )));
        }

        // Fallback: Handle c1 * (c2 * x) -> (c1*c2) * x
        if let Expr::Number(n1) = ctx.get(lhs).clone() {
            if let Some((rl, rr)) = as_mul(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(rl).clone() {
                    let prod = &n1 * &n2;
                    let prod_expr = ctx.add(Expr::Number(prod));
                    let new_expr = smart_mul(ctx, prod_expr, rr);
                    return Some(
                        Rewrite::new(new_expr)
                            .desc(format!("Combine nested constants: {} * {}", n1, n2)),
                    );
                }
            }

            // Handle c1 * (x / c2) -> (c1/c2) * x
            if let Some((num, den)) = as_div(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(den).clone() {
                    if !n2.is_zero() {
                        let ratio = &n1 / &n2;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        let new_expr = smart_mul(ctx, ratio_expr, num);
                        return Some(
                            Rewrite::new(new_expr).desc(format!(
                                "{} * (x / {}) -> ({} / {}) * x",
                                n1, n2, n1, n2
                            )),
                        );
                    }
                }
            }
        }
    }

    // Try Sub branch
    if let Some((lhs, rhs)) = as_sub(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs).clone(), ctx.get(rhs).clone()) {
            let diff = &n1 - &n2;
            let new_expr = ctx.add(Expr::Number(diff.clone()));
            return Some(Rewrite::new(new_expr).desc(format!("{} - {} = {}", n1, n2, diff)));
        }
    }

    // Try Div branch
    if let Some((lhs, rhs)) = as_div(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs).clone(), ctx.get(rhs).clone()) {
            if !n2.is_zero() {
                let quot = &n1 / &n2;
                let new_expr = ctx.add(Expr::Number(quot.clone()));
                return Some(
                    Rewrite::new(new_expr).desc(format!("{} / {} = {}", n1, n2, quot)),
                );
            } else {
                let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                return Some(Rewrite::new(undef).desc("Division by zero"));
            }
        }

        // Handle (c * x) / d -> (c/d) * x
        if let Expr::Number(d) = ctx.get(rhs).clone() {
            if !d.is_zero() {
                if let Some((ml, mr)) = as_mul(ctx, lhs) {
                    // Case 1: (c * x) / d
                    if let Expr::Number(c) = ctx.get(ml).clone() {
                        let ratio = &c / &d;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        let new_expr = smart_mul(ctx, ratio_expr, mr);
                        return Some(
                            Rewrite::new(new_expr)
                                .desc(format!("({} * x) / {} -> ({} / {}) * x", c, d, c, d)),
                        );
                    }

                    // Case 2: (x * c) / d
                    if let Expr::Number(c) = ctx.get(mr).clone() {
                        let ratio = &c / &d;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        let new_expr = smart_mul(ctx, ratio_expr, ml);
                        return Some(
                            Rewrite::new(new_expr)
                                .desc(format!("(x * {}) / {} -> ({} / {}) * x", c, d, c, d)),
                        );
                    }
                }
            }
        }
    }

    None
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

    #[test]
    fn test_add_zero() {
        let mut ctx = Context::new();
        let rule = AddZeroRule;
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Add(x, zero));
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
            "x"
        );
    }

    #[test]
    fn test_mul_one() {
        let mut ctx = Context::new();
        let rule = MulOneRule;
        let one = ctx.num(1);
        let y = ctx.var("y");
        let expr = ctx.add(Expr::Mul(one, y));
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
            "y"
        );
    }

    #[test]
    fn test_combine_constants() {
        let mut ctx = Context::new();
        let rule = CombineConstantsRule;
        let two = ctx.num(2);
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Add(two, three));
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
            "5"
        );
    }
}

// =============================================================================
// SubSelfToZeroRule: a - a = 0 (Short-circuit)
// =============================================================================
//
// V2.14.45: This rule MUST fire before expansion rules like TanToSinCosRule.
// Without this, tan(3x) - tan(3x) would expand both tans and fail to cancel.
// Uses priority 500 to ensure it runs first.
//
// Domain Policy: Same as AddInverseRule - check for undefined subexpressions.
// Uses compare_expr for structural equality (handles tan(3x) == tan(3·x)).
// =============================================================================
define_rule!(
    SubSelfToZeroRule,
    "Subtraction Self-Cancel",
    priority: 500, // High priority: before any expansion rules
    |ctx, expr, parent_ctx| {
        use crate::semantic_equality::SemanticEqualityChecker;

        // Helper: check if expression contains any Div with non-literal denominator
        // Delegates to canonical implementation that handles Hold/Matrix
        let has_undefined_risk = |ctx: &cas_ast::Context, expr: cas_ast::ExprId| -> bool {
            crate::collect::has_undefined_risk(ctx, expr)
        };

        // Match: Sub(lhs, rhs)
        if let Expr::Sub(lhs, rhs) = ctx.get(expr) {
            let lhs = *lhs;
            let rhs = *rhs;

            // Use semantic equality (handles structural equivalence like tan(3x) vs tan(3·x))
            let checker = SemanticEqualityChecker::new(ctx);
            if checker.are_equal(lhs, rhs) {
                let domain_mode = parent_ctx.domain_mode();

                // In Strict mode, check for undefined risk
                if domain_mode == crate::DomainMode::Strict && has_undefined_risk(ctx, lhs) {
                    return None;
                }

                return Some(Rewrite::new(ctx.num(0)).desc("a - a = 0"));
            }
        }
        None
    }
);

// AddInverseRule: a + (-a) = 0
// Domain Mode Policy: Like other cancellation rules, we must respect domain_mode
// because if `a` can be undefined (e.g., x/(x+1) when x=-1), then a + (-a)
// is undefined, not 0.
// - Strict: only if `a` contains no potentially-undefined subexpressions (no variable denominator)
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
//
// V2.12.13: REMOVED redundant "is defined" assumption event.
// The individual Div operations already emit NonZero(denominator) as Requires.
// Showing "a is defined" here is redundant and confusing.
define_rule!(AddInverseRule, "Add Inverse", |ctx, expr, parent_ctx| {
    // Helper: check if expression contains any Div with non-literal denominator
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        crate::collect::has_undefined_risk(ctx, expr)
    }

    // Pattern: a + (-a) = 0 or (-a) + a = 0
    if let Expr::Add(l, r) = ctx.get(expr) {
        let mut matched_inner: Option<cas_ast::ExprId> = None;

        // Check if r = -l or l = -r
        if let Expr::Neg(neg_inner) = ctx.get(*r) {
            if *neg_inner == *l {
                matched_inner = Some(*l);
            }
        }
        if matched_inner.is_none() {
            if let Expr::Neg(neg_inner) = ctx.get(*l) {
                if *neg_inner == *r {
                    matched_inner = Some(*r);
                }
            }
        }

        if let Some(inner) = matched_inner {
            let domain_mode = parent_ctx.domain_mode();

            // In Strict mode, check for undefined risk
            if domain_mode == crate::DomainMode::Strict && has_undefined_risk(ctx, inner) {
                return None;
            }

            // V2.12.13: No assumption events - the division conditions are already
            // tracked as Requires from the original Div operations.
            // Adding "a is defined" here is redundant and clutters the output.
            return Some(Rewrite::new(ctx.num(0)).desc("a + (-a) = 0"));
        }
    }
    None
});

// Simplify sums of fractions in exponents: x^(1/2 + 1/3) → x^(5/6)
// This makes the fraction sum visible as a step in the timeline.
define_rule!(
    SimplifyNumericExponentsRule,
    "Sum Exponents",
    |ctx, expr| {
        // Only match Pow(base, exp) where exp is a sum of numeric terms
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            let base = *base;
            let exp = *exp;

            // Collect all addends from the exponent
            let mut addends: Vec<num_rational::BigRational> = Vec::new();
            let mut stack = vec![exp];
            let mut all_numeric = true;

            while let Some(id) = stack.pop() {
                match ctx.get(id) {
                    Expr::Add(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    Expr::Number(n) => {
                        addends.push(n.clone());
                    }
                    Expr::Div(num, den) => {
                        // Check if it's a numeric fraction
                        if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                            if !d.is_zero() {
                                addends.push(n / d);
                            } else {
                                all_numeric = false;
                            }
                        } else {
                            all_numeric = false;
                        }
                    }
                    _ => {
                        all_numeric = false;
                    }
                }
            }

            // Only simplify if:
            // 1. All terms are numeric
            // 2. There are at least 2 terms (otherwise it's already simplified)
            if all_numeric && addends.len() >= 2 {
                // Sum all fractions
                let sum: num_rational::BigRational = addends.iter().sum();

                // Create the simplified exponent as a Number
                let new_exp = ctx.add(Expr::Number(sum.clone()));
                let new_pow = ctx.add(Expr::Pow(base, new_exp));

                // Generate description showing the sum
                let addend_strs: Vec<String> = addends
                    .iter()
                    .map(|r| {
                        if r.is_integer() {
                            format!("{}", r.numer())
                        } else {
                            format!("({}/{})", r.numer(), r.denom())
                        }
                    })
                    .collect();
                let sum_str = if sum.is_integer() {
                    format!("{}", sum.numer())
                } else {
                    format!("{}/{}", sum.numer(), sum.denom())
                };

                return Some(Rewrite::new(new_pow).desc(format!(
                    "{} = {}",
                    addend_strs.join(" + "),
                    sum_str
                )));
            }
        }
        None
    }
);

// =============================================================================
// NormalizeMulNegRule: Lift Neg out of Mul for canonical form
// =============================================================================
//
// Canonical form: Neg should be at the TOP of Mul, not buried inside.
// This unlocks cancellations in Add like: a*(-b) + (-a)*b → Neg(a*b) + Neg(a*b) → -2*a*b
//
// Rewrites:
// - Mul(Neg(a), b) → Neg(Mul(a, b))
// - Mul(a, Neg(b)) → Neg(Mul(a, b))
// - Mul(Neg(a), Neg(b)) → Mul(a, b)  (double negation cancels)
//
// This is idempotent and always reduces complexity.
// =============================================================================
define_rule!(
    NormalizeMulNegRule,
    "Normalize Negation in Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l = *l;
            let r = *r;

            let l_neg = if let Expr::Neg(inner) = ctx.get(l) { Some(*inner) } else { None };
            let r_neg = if let Expr::Neg(inner) = ctx.get(r) { Some(*inner) } else { None };

            match (l_neg, r_neg) {
                // Mul(Neg(a), Neg(b)) → Mul(a, b) (double negation)
                (Some(a), Some(b)) => {
                    let new_mul = crate::build::mul2_raw(ctx, a, b);
                    return Some(Rewrite::new(new_mul).desc("(-a) * (-b) = a * b"));
                }
                // Mul(Neg(a), b) → Neg(Mul(a, b))
                (Some(a), None) => {
                    let new_mul = crate::build::mul2_raw(ctx, a, r);
                    let result = ctx.add(Expr::Neg(new_mul));
                    return Some(Rewrite::new(result).desc("(-a) * b = -(a * b)"));
                }
                // Mul(a, Neg(b)) → Neg(Mul(a, b))
                (None, Some(b)) => {
                    let new_mul = crate::build::mul2_raw(ctx, l, b);
                    let result = ctx.add(Expr::Neg(new_mul));
                    return Some(Rewrite::new(result).desc("a * (-b) = -(a * b)"));
                }
                _ => {}
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    // High-priority short-circuit rules first
    simplifier.add_rule(Box::new(SubSelfToZeroRule)); // priority 500: before expansion

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(NormalizeMulNegRule)); // Lift Neg out of Mul for canonical form
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(SimplifyNumericExponentsRule));
    simplifier.add_rule(Box::new(AddInverseRule));
}

#[cfg(test)]
mod importance_tests {
    use super::*;
    use crate::rule::SimpleRule;
    use crate::step::ImportanceLevel;

    #[test]
    fn test_mul_one_rule_importance() {
        let rule = MulOneRule;
        assert_eq!(
            rule.importance(),
            ImportanceLevel::Low,
            "MulOneRule should have Low importance (hidden in normal mode)"
        );
    }

    #[test]
    fn test_add_zero_rule_importance() {
        let rule = AddZeroRule;
        assert_eq!(
            rule.importance(),
            ImportanceLevel::Low,
            "AddZeroRule should have Low importance (hidden in normal mode)"
        );
    }
}
