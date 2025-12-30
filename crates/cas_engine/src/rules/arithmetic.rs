use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::Expr;
use num_traits::{One, Zero};

define_rule!(AddZeroRule, "Identity Property of Addition", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Add(lhs, rhs) = expr_data {
        if let Expr::Number(n) = ctx.get(rhs) {
            if n.is_zero() {
                return Some(Rewrite {
                    new_expr: lhs,
                    description: "x + 0 = x".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
        }
        if let Expr::Number(n) = ctx.get(lhs) {
            if n.is_zero() {
                return Some(Rewrite {
                    new_expr: rhs,
                    description: "0 + x = x".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
        }
    }
    None
});

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(rhs) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: lhs,
                        description: "x * 1 = x".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: rhs,
                        description: "1 * x = x".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }
        }
        None
    }
);

// MulZeroRule: 0*e → 0
// Domain Mode Policy: 0*e → 0 changes the domain of definition if e can be undefined.
// In Strict mode, the expression 0*(x/(x+1)) is undefined at x=-1, but 0 is defined everywhere.
// - Strict: only apply if other factor has no undefined risk
// - Assume: apply with domain_assumption warning if risk exists
// - Generic: apply unconditionally (educational mode)
define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    |ctx, expr, parent_ctx| {
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;

        // Helper: check if expression contains any Div with non-literal denominator
        fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
            let mut stack = vec![expr];
            while let Some(e) = stack.pop() {
                match ctx.get(e) {
                    Expr::Div(_, den) => {
                        // If denominator is not a proven nonzero literal, there's risk
                        if prove_nonzero(ctx, *den) != Proof::Proven {
                            return true;
                        }
                        stack.push(*den);
                    }
                    Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    Expr::Neg(inner) => {
                        stack.push(*inner);
                    }
                    Expr::Function(_, args) => {
                        for arg in args {
                            stack.push(*arg);
                        }
                    }
                    _ => {}
                }
            }
            false
        }

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
            let domain_mode = parent_ctx.domain_mode();
            let has_risk = has_undefined_risk(ctx, other);

            // Domain-aware decision for 0*e → 0
            let assumption_events = match domain_mode {
                crate::DomainMode::Strict => {
                    // Only apply if other factor has no undefined risk
                    if has_risk {
                        return None;
                    }
                    Default::default()
                }
                crate::DomainMode::Assume => {
                    // Apply with structured assumption if has risk
                    if has_risk {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::defined(
                            ctx, other
                        )]
                    } else {
                        Default::default()
                    }
                }
                crate::DomainMode::Generic => {
                    // Educational mode: apply with warning for transparency
                    if has_risk {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::defined(
                            ctx, other
                        )]
                    } else {
                        Default::default()
                    }
                }
            };

            let description = if lhs_is_zero {
                "0 * x = 0".to_string()
            } else {
                "x * 0 = 0".to_string()
            };

            let zero = ctx.num(0);
            return Some(Rewrite {
                new_expr: zero,
                description,
                before_local: None,
                after_local: None,
                assumption_events,
            });
        }
        None
    }
);

// DivZeroRule: 0/d → 0
// Domain Mode Policy: 0/d → 0 changes the domain of definition if d can be 0.
// In Strict mode, the expression 0/(x+1) is undefined at x=-1, but 0 is defined everywhere.
// - Strict: only apply if prove_nonzero(d) == Proven
// - Assume: apply with domain_assumption warning "Assuming d ≠ 0"
// - Generic: apply unconditionally (educational mode)
define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    |ctx, expr, parent_ctx| {
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;

        let expr_data = ctx.get(expr).clone();
        if let Expr::Div(num, den) = expr_data {
            // Check if numerator is 0
            let num_is_zero = matches!(ctx.get(num), Expr::Number(n) if n.is_zero());
            if !num_is_zero {
                return None;
            }

            // Special case: 0/0 → undefined (all modes)
            if let Expr::Number(d) = ctx.get(den) {
                if d.is_zero() {
                    let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                    return Some(Rewrite {
                        new_expr: undef,
                        description: "0/0 is undefined".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }

            // Domain-aware decision for 0/d → 0
            let den_nonzero = prove_nonzero(ctx, den);
            let domain_mode = parent_ctx.domain_mode();

            // Structured assumption emission (preferred over legacy string)
            let assumption_events = match domain_mode {
                crate::DomainMode::Strict => {
                    // Only apply if denominator is provably non-zero
                    if den_nonzero != Proof::Proven {
                        return None;
                    }
                    Default::default()
                }
                crate::DomainMode::Assume => {
                    // Apply with structured assumption if not proven
                    if den_nonzero != Proof::Proven {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::nonzero(ctx, den)]
                    } else {
                        Default::default()
                    }
                }
                crate::DomainMode::Generic => {
                    // Educational mode: apply with warning for transparency
                    if den_nonzero != Proof::Proven {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::nonzero(ctx, den)]
                    } else {
                        Default::default()
                    }
                }
            };

            let zero = ctx.num(0);
            return Some(Rewrite {
                new_expr: zero,
                description: "0 / d = 0".to_string(),
                before_local: None,
                after_local: None,
                assumption_events,
            });
        }
        None
    }
);

define_rule!(CombineConstantsRule, "Combine Constants", |ctx, expr| {
    // We need to clone data to avoid borrowing ctx while mutating it later
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(lhs, rhs) => {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();
            if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                let sum = n1 + n2;
                let new_expr = ctx.add(Expr::Number(sum.clone()));
                return Some(Rewrite {
                    new_expr,
                    description: format!("{} + {} = {}", n1, n2, sum),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
            // Handle nested: c1 + (c2 + x) -> (c1+c2) + x
            if let Expr::Number(n1) = lhs_data {
                if let Expr::Add(rl, rr) = rhs_data {
                    let rl_data = ctx.get(rl).clone();
                    if let Expr::Number(n2) = rl_data {
                        let sum = &n1 + &n2;
                        let sum_expr = ctx.add(Expr::Number(sum));
                        let new_expr = ctx.add(Expr::Add(sum_expr, rr));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("Combine nested constants: {} + {}", n1, n2),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                        });
                    }
                }
            }
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();
            if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                let prod = n1 * n2;
                let new_expr = ctx.add(Expr::Number(prod.clone()));
                return Some(Rewrite {
                    new_expr,
                    description: format!("{} * {} = {}", n1, n2, prod),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
            // Handle nested: c1 * (c2 * x) -> (c1*c2) * x
            if let Expr::Number(ref n1) = lhs_data {
                if let Expr::Mul(rl, rr) = rhs_data {
                    let rl_data = ctx.get(rl).clone();
                    if let Expr::Number(n2) = rl_data {
                        let prod = n1 * &n2;
                        let prod_expr = ctx.add(Expr::Number(prod));
                        let new_expr = smart_mul(ctx, prod_expr, rr);
                        return Some(Rewrite {
                            new_expr,
                            description: format!("Combine nested constants: {} * {}", n1, n2),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                        });
                    }
                }
            }

            // Handle c1 * (x / c2) -> (c1/c2) * x
            if let Expr::Number(ref n1) = lhs_data {
                if let Expr::Div(num, den) = rhs_data {
                    let den_data = ctx.get(den).clone();
                    if let Expr::Number(n2) = den_data {
                        if !n2.is_zero() {
                            let ratio = n1 / &n2;
                            let ratio_expr = ctx.add(Expr::Number(ratio));
                            let new_expr = smart_mul(ctx, ratio_expr, num);
                            return Some(Rewrite {
                                new_expr,
                                description: format!(
                                    "{} * (x / {}) -> ({} / {}) * x",
                                    n1, n2, n1, n2
                                ),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                            });
                        }
                    }
                }
            }
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();
            if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                let diff = n1 - n2;
                let new_expr = ctx.add(Expr::Number(diff.clone()));
                return Some(Rewrite {
                    new_expr,
                    description: format!("{} - {} = {}", n1, n2, diff),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
        }
        Expr::Div(lhs, rhs) => {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();
            if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                if !n2.is_zero() {
                    let quot = n1 / n2;
                    let new_expr = ctx.add(Expr::Number(quot.clone()));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("{} / {} = {}", n1, n2, quot),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                } else {
                    let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                    return Some(Rewrite {
                        new_expr: undef,
                        description: "Division by zero".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }

            // Handle (c * x) / d -> (c/d) * x
            if let Expr::Number(d) = rhs_data {
                if !d.is_zero() {
                    if let Expr::Mul(ml, mr) = lhs_data {
                        let ml_data = ctx.get(ml).clone();
                        let mr_data = ctx.get(mr).clone();

                        // Case 1: (c * x) / d
                        if let Expr::Number(c) = ml_data {
                            let ratio = &c / &d;
                            let ratio_expr = ctx.add(Expr::Number(ratio));
                            let new_expr = smart_mul(ctx, ratio_expr, mr);
                            return Some(Rewrite {
                                new_expr,
                                description: format!("({} * x) / {} -> ({} / {}) * x", c, d, c, d),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                            });
                        }

                        // Case 2: (x * c) / d
                        if let Expr::Number(c) = mr_data {
                            let ratio = &c / &d;
                            let ratio_expr = ctx.add(Expr::Number(ratio));
                            let new_expr = smart_mul(ctx, ratio_expr, ml);
                            return Some(Rewrite {
                                new_expr,
                                description: format!("(x * {}) / {} -> ({} / {}) * x", c, d, c, d),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                            });
                        }
                    }
                }
            }
        }
        _ => {}
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

// AddInverseRule: a + (-a) = 0
// Domain Mode Policy: Like other cancellation rules, we must respect domain_mode
// because if `a` can be undefined (e.g., x/(x+1) when x=-1), then a + (-a)
// is undefined, not 0.
// - Strict: only if `a` contains no potentially-undefined subexpressions (no variable denominator)
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AddInverseRule, "Add Inverse", |ctx, expr, parent_ctx| {
    use crate::domain::Proof;
    use crate::helpers::prove_nonzero;

    // Helper: check if expression contains any Div with non-literal denominator
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        let mut stack = vec![expr];
        while let Some(e) = stack.pop() {
            match ctx.get(e) {
                Expr::Div(_, den) => {
                    // If denominator is not a proven nonzero literal, there's risk
                    if prove_nonzero(ctx, *den) != Proof::Proven {
                        return true;
                    }
                    stack.push(*den);
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                Expr::Neg(inner) => {
                    stack.push(*inner);
                }
                Expr::Function(_, args) => {
                    for arg in args {
                        stack.push(*arg);
                    }
                }
                _ => {}
            }
        }
        false
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

            // Determine warning for Assume/Generic modes with undefined risk
            let has_risk = has_undefined_risk(ctx, inner);
            let needs_warning = has_risk && domain_mode != crate::DomainMode::Strict;

            return Some(Rewrite {
                new_expr: ctx.num(0),
                description: "a + (-a) = 0".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: if needs_warning {
                    smallvec::smallvec![crate::assumptions::AssumptionEvent::defined(ctx, inner)]
                } else {
                    Default::default()
                },
            });
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

                return Some(Rewrite {
                    new_expr: new_pow,
                    description: format!("{} = {}", addend_strs.join(" + "), sum_str),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(SimplifyNumericExponentsRule));
    simplifier.add_rule(Box::new(AddInverseRule));
}
