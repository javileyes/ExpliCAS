use crate::define_rule;
use crate::helpers::{flatten_add_sub_chain, flatten_mul_chain, is_one, is_zero};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use std::collections::HashMap;

define_rule!(CollectRule, "Collect Terms", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "collect" && args.len() == 2 {
            let target_expr = args[0];
            let var_expr = args[1];

            // Ensure second argument is a variable
            let var_name = if let Expr::Variable(v) = ctx.get(var_expr) {
                v.clone()
            } else {
                return None;
            };

            // 1. Flatten terms (using shared helper)
            let terms = flatten_add_sub_chain(ctx, target_expr);

            // 2. Group by degree of var
            // Map: degree -> Vec<ExprId>
            let mut groups: HashMap<i64, Vec<ExprId>> = HashMap::new();

            for term in terms {
                let (coeff, degree) = extract_coeff_degree(ctx, term, &var_name);
                if degree == 0 {
                    // Check if it really is independent or just degree 0 (constant wrt var)
                    // extract_coeff_degree returns degree 0 for terms not containing var.
                    // We treat them as independent terms to be appended at the end,
                    // or we can group them under x^0.
                    // Let's group them under x^0 for now, effectively "collecting constants".
                    groups.entry(0).or_default().push(coeff);
                } else {
                    groups.entry(degree).or_default().push(coeff);
                }
            }

            // 3. Reconstruct expression
            let mut new_terms = Vec::new();

            // Sort degrees descending
            let mut degrees: Vec<i64> = groups.keys().cloned().collect();
            degrees.sort_by(|a, b| b.cmp(a));

            for deg in degrees {
                let coeffs = groups.get(&deg).unwrap();
                if coeffs.is_empty() {
                    continue;
                }

                // Sum coefficients: (c1 + c2 + ...)
                let combined_coeff = if coeffs.len() == 1 {
                    coeffs[0]
                } else {
                    let mut sum = coeffs[0];
                    for c in coeffs.iter().skip(1) {
                        sum = ctx.add(Expr::Add(sum, *c));
                    }
                    sum
                };

                // Construct term: coeff * var^deg
                let term = if deg == 0 {
                    combined_coeff
                } else {
                    let var_part = if deg == 1 {
                        ctx.var(&var_name)
                    } else {
                        let deg_expr = ctx.num(deg);
                        let v = ctx.var(&var_name);
                        ctx.add(Expr::Pow(v, deg_expr))
                    };

                    if is_one(ctx, combined_coeff) {
                        var_part
                    } else if is_zero(ctx, combined_coeff) {
                        // 0 * x^n = 0, skip
                        continue;
                    } else {
                        smart_mul(ctx, combined_coeff, var_part)
                    }
                };
                new_terms.push(term);
            }

            if new_terms.is_empty() {
                let zero = ctx.num(0);
                return Some(Rewrite {
                    new_expr: zero,
                    description: format!("collect({}, {})", target_expr, var_name), // Debug format,
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }

            let mut result = new_terms[0];
            for t in new_terms.into_iter().skip(1) {
                result = ctx.add(Expr::Add(result, t));
            }

            return Some(Rewrite {
                new_expr: result,
                description: format!("collect({}, {})", target_expr, var_name),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            });
        }
    }
    None
});

// is_one and is_zero are now imported from crate::helpers

// flatten_add_chain and flatten_mul_chain are now imported from crate::helpers

// Returns (coefficient, degree) for a term with respect to var
fn extract_coeff_degree(ctx: &mut Context, term: ExprId, var: &str) -> (ExprId, i64) {
    // Cases:
    // x -> (1, 1)
    // x^n -> (1, n)
    // a * x -> (a, 1)
    // a * x^n -> (a, n)
    // a -> (a, 0)
    // x * y -> (y, 1) if we treat y as coeff? Yes.

    // We need to traverse Mul chain to find var powers.
    // Assume term is a product of factors.
    let factors = flatten_mul_chain(ctx, term);

    let mut degree = 0;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        let factor_data = ctx.get(factor).clone();
        match factor_data {
            Expr::Variable(v) if v == var => {
                degree += 1;
            }
            Expr::Pow(base, exp) => {
                if let Expr::Variable(v) = ctx.get(base) {
                    if v == var {
                        if let Expr::Number(n) = ctx.get(exp) {
                            if n.is_integer() {
                                degree += n.to_integer().try_into().unwrap_or(0);
                                continue;
                            }
                        }
                    }
                }
                coeff_factors.push(factor);
            }
            _ => {
                coeff_factors.push(factor);
            }
        }
    }

    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut c = coeff_factors[0];
        for f in coeff_factors.into_iter().skip(1) {
            c = smart_mul(ctx, c, f);
        }
        c
    };

    (coeff, degree)
}

// Local flatten_mul_chain removed - using imported version from crate::helpers

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_collect_basic() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(a*x + b*x, x) -> (a+b)*x
        let expr = parse("collect(a*x + b*x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Result could be (a+b)*x or (b+a)*x
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("x"));
        assert!(s.contains("a + b") || s.contains("b + a"));
    }

    #[test]
    fn test_collect_with_constants() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(a*x + 2*x + 5, x) -> (a+2)*x + 5
        let expr = parse("collect(a*x + 2*x + 5, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("a + 2") || s.contains("2 + a"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_collect_powers() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(3*x^2 + y*x^2 + x, x) -> (3+y)*x^2 + x
        let expr = parse("collect(3*x^2 + y*x^2 + x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("3 + y") || s.contains("y + 3"));
        assert!(s.contains("x^2"));
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CollectRule));
    // simplifier.add_rule(Box::new(CollectLikeTermsRule));
}

define_rule!(CollectLikeTermsRule, "Collect Like Terms", |ctx, expr| {
    // Only apply to Add/Sub
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return None,
    }

    // CRITICAL: Do NOT apply to non-commutative expressions (e.g., matrices)
    // Matrix addition/subtraction has dedicated rules
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let new_expr = crate::collect::collect(ctx, expr);
    // Check if structurally different to avoid infinite loops with ID regeneration
    if new_expr != expr
        && crate::ordering::compare_expr(ctx, new_expr, expr) != std::cmp::Ordering::Equal
    {
        // eprintln!("CollectLikeTermsRule applied: {:?} -> {:?}", expr, new_expr);
        Some(Rewrite {
            new_expr,
            description: "Collect like terms".to_string(),
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
        })
    } else {
        None
    }
});
