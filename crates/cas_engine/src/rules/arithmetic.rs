use crate::define_rule;
use crate::rule::Rewrite;
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
            });
                }
            }
        }
        None
    }
);

define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(rhs) {
                if n.is_zero() {
                    let zero = ctx.num(0);
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "x * 0 = 0".to_string(),
                before_local: None,
                after_local: None,
            });
                }
            }
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_zero() {
                    let zero = ctx.num(0);
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "0 * x = 0".to_string(),
                before_local: None,
                after_local: None,
            });
                }
            }
        }
        None
    }
);

define_rule!(DivZeroRule, "Zero Property of Division", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(lhs, rhs) = expr_data {
        if let Expr::Number(n) = ctx.get(lhs) {
            if n.is_zero() {
                // Check if denominator is zero?
                // Ideally yes, but for symbolic simplification we often assume non-zero.
                // If denominator is explicitly zero, CombineConstantsRule handles it (or we can check).
                if let Expr::Number(d) = ctx.get(rhs) {
                    if d.is_zero() {
                        return None; // Undefined
                    }
                }

                let zero = ctx.num(0);
                return Some(Rewrite {
                    new_expr: zero,
                    description: "0 / x = 0".to_string(),
                before_local: None,
                after_local: None,
            });
            }
        }
    }
    None
});

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
            });
            }
            // Handle nested: c1 + (c2 + x) -> (c1+c2) + x
            if let Expr::Number(n1) = lhs_data {
                if let Expr::Add(rl, rr) = rhs_data {
                    let rl_data = ctx.get(rl).clone();
                    if let Expr::Number(n2) = rl_data {
                        let sum = &n1 + &n2;
                        let sum_expr = ctx.add(Expr::Number(sum.clone()));
                        let new_expr = ctx.add(Expr::Add(sum_expr, rr));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("Combine nested constants: {} + {}", n1, n2),
                before_local: None,
                after_local: None,
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
            });
            }
            // Handle nested: c1 * (c2 * x) -> (c1*c2) * x
            if let Expr::Number(ref n1) = lhs_data {
                if let Expr::Mul(rl, rr) = rhs_data {
                    let rl_data = ctx.get(rl).clone();
                    if let Expr::Number(n2) = rl_data {
                        let prod = n1 * &n2;
                        let prod_expr = ctx.add(Expr::Number(prod.clone()));
                        let new_expr = ctx.add(Expr::Mul(prod_expr, rr));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("Combine nested constants: {} * {}", n1, n2),
                before_local: None,
                after_local: None,
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
                            let new_expr = ctx.add(Expr::Mul(ratio_expr, num));
                            return Some(Rewrite {
                                new_expr,
                                description: format!(
                                    "{} * (x / {}) -> ({} / {}) * x",
                                    n1, n2, n1, n2
                                ),
                before_local: None,
                after_local: None,
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
            });
                } else {
                    let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                    return Some(Rewrite {
                        new_expr: undef,
                        description: "Division by zero".to_string(),
                before_local: None,
                after_local: None,
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
                            let new_expr = ctx.add(Expr::Mul(ratio_expr, mr));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("({} * x) / {} -> ({} / {}) * x", c, d, c, d),
                before_local: None,
                after_local: None,
            });
                        }

                        // Case 2: (x * c) / d
                        if let Expr::Number(c) = mr_data {
                            let ratio = &c / &d;
                            let ratio_expr = ctx.add(Expr::Number(ratio));
                            let new_expr = ctx.add(Expr::Mul(ratio_expr, ml));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("(x * {}) / {} -> ({} / {}) * x", c, d, c, d),
                before_local: None,
                after_local: None,
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

define_rule!(AddInverseRule, "Add Inverse", |ctx, expr| {
    // Pattern: a + (-a) = 0 or (-a) + a = 0
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check if r = -l or l = -r
        if let Expr::Neg(neg_inner) = ctx.get(*r) {
            if *neg_inner == *l {
                // a + (-a) = 0
                return Some(Rewrite {
                    new_expr: ctx.num(0),
                    description: "a + (-a) = 0".to_string(),
                before_local: None,
                after_local: None,
            });
            }
        }
        if let Expr::Neg(neg_inner) = ctx.get(*l) {
            if *neg_inner == *r {
                // (-a) + a = 0
                return Some(Rewrite {
                    new_expr: ctx.num(0),
                    description: "(-a) + a = 0".to_string(),
                before_local: None,
                after_local: None,
            });
            }
        }
    }
    None
});

/// Simplify sums of fractions in exponents: x^(1/2 + 1/3) → x^(5/6)
/// This makes the fraction sum visible as a step in the timeline.
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
