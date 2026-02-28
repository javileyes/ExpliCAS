//! Heuristics for skipping expensive distributive expansions.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Returns true when distribution across an additive expression should be skipped
/// for performance/stability reasons.
pub fn should_skip_distribution_for_factor(
    ctx: &Context,
    factor: ExprId,
    additive: ExprId,
) -> bool {
    // Pure-constant additive sums always distribute.
    if cas_ast::collect_variables(ctx, additive).is_empty() {
        return false;
    }

    let factor_nodes = cas_ast::count_nodes(ctx, factor);
    let factor_vars = cas_ast::collect_variables(ctx, factor);

    // Case 1: Variable-free complex constant.
    if factor_vars.is_empty() && factor_nodes >= 5 {
        return true;
    }

    // Case 2: Expression with fractional exponents.
    if factor_nodes >= 5 && has_fractional_exponents(ctx, factor) {
        return true;
    }

    // Case 3: Multi-variable fraction-like expression.
    if factor_vars.len() >= 3 && factor_nodes >= 10 {
        return true;
    }

    // Case 4: Non-number factor across a long additive chain.
    let additive_terms = crate::expr_relations::count_additive_terms(ctx, additive);
    if additive_terms >= 4 && !matches!(ctx.get(factor), Expr::Number(_)) {
        return true;
    }

    false
}

/// Check if an expression tree contains any fractional exponents.
pub fn has_fractional_exponents(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if !n.is_integer() {
                        return true;
                    }
                }
                if matches!(ctx.get(*exp), Expr::Div(_, _)) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => {
                for &a in args {
                    stack.push(a);
                }
            }
            _ => {}
        }
    }
    false
}

/// Returns true when `expr` is a 2-term additive binomial (`Add`/`Sub`).
pub fn is_binomial_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

/// Shape-only policy for whether `factor * additive` should distribute.
///
/// Mirrors the engine policy:
/// - Always allow when additive side is variable-free
/// - Allow number/function/add/sub/pow/mul/div factors
/// - Allow variable factor only in multivariable products
pub fn should_distribute_factor_over_additive(
    ctx: &Context,
    factor: ExprId,
    additive: ExprId,
    product_expr: ExprId,
) -> bool {
    let factor_expr = ctx.get(factor);
    let additive_is_constant = cas_ast::collect_variables(ctx, additive).is_empty();
    additive_is_constant
        || matches!(factor_expr, Expr::Number(_))
        || matches!(factor_expr, Expr::Function(_, _))
        || matches!(factor_expr, Expr::Add(_, _))
        || matches!(factor_expr, Expr::Sub(_, _))
        || matches!(factor_expr, Expr::Pow(_, _))
        || matches!(factor_expr, Expr::Mul(_, _))
        || matches!(factor_expr, Expr::Div(_, _))
        || (matches!(factor_expr, Expr::Variable(_))
            && cas_ast::collect_variables(ctx, product_expr).len() > 1)
}

/// Educational guard: avoid distributing a fractional numeric coefficient over a binomial.
pub fn should_block_fractional_coeff_over_binomial(
    ctx: &Context,
    coeff_candidate: ExprId,
    additive_side: ExprId,
) -> bool {
    let Expr::Number(n) = ctx.get(coeff_candidate) else {
        return false;
    };
    !n.is_integer() && is_binomial_expr(ctx, additive_side)
}

/// Estimate simplification reduction when distributing `(num1 + num2 + ...)/den`.
///
/// Returns 0 when no clear cancellation/simplification is detected.
pub fn estimate_division_distribution_simplification_reduction(
    ctx: &Context,
    numerator_term: ExprId,
    denominator: ExprId,
) -> usize {
    if numerator_term == denominator {
        return cas_ast::count_nodes(ctx, numerator_term);
    }

    let numerator_factors = collect_mul_factors(ctx, numerator_term);
    let denominator_factors = collect_mul_factors(ctx, denominator);

    for den_factor in denominator_factors {
        let found_structural = numerator_factors
            .iter()
            .any(|num_factor| compare_expr(ctx, *num_factor, den_factor) == Ordering::Equal);
        if found_structural {
            let factor_size = cas_ast::count_nodes(ctx, den_factor);
            let mut reduction = factor_size * 2;
            if den_factor == denominator {
                reduction += 1;
            }
            return reduction;
        }

        if let Expr::Number(den_number) = ctx.get(den_factor) {
            let found_numeric = numerator_factors.iter().any(|num_factor| {
                if let Expr::Number(num_number) = ctx.get(*num_factor) {
                    if num_number.is_integer() && den_number.is_integer() {
                        let num_int = num_number.to_integer();
                        let den_int = den_number.to_integer();
                        if !num_int.is_zero() && !den_int.is_zero() {
                            let gcd = num_int.gcd(&den_int);
                            return gcd > One::one();
                        }
                    }
                }
                false
            });
            if found_numeric {
                return 1;
            }
        }
    }

    let vars = cas_ast::collect_variables(ctx, numerator_term);
    if vars.is_empty() {
        return 0;
    }

    for var in vars {
        if let (Ok(p_num), Ok(p_den)) = (
            crate::polynomial::Polynomial::from_expr(ctx, numerator_term, &var),
            crate::polynomial::Polynomial::from_expr(ctx, denominator, &var),
        ) {
            if p_den.is_zero() {
                continue;
            }
            let gcd = p_num.gcd(&p_den);
            if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                if gcd.degree() == p_den.degree() {
                    return cas_ast::count_nodes(ctx, denominator) + 1;
                }
                return 1;
            }
        }
    }
    0
}

fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut stack = vec![expr];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(left, right) = ctx.get(curr) {
            stack.push(*left);
            stack.push(*right);
        } else {
            factors.push(curr);
        }
    }
    factors
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_division_distribution_simplification_reduction, has_fractional_exponents,
        is_binomial_expr, should_block_fractional_coeff_over_binomial,
        should_distribute_factor_over_additive, should_skip_distribution_for_factor,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_fractional_exponents() {
        let mut ctx = Context::new();
        let expr = parse("x^(1/3) + 1", &mut ctx).expect("parse");
        assert!(has_fractional_exponents(&ctx, expr));
    }

    #[test]
    fn allows_distribution_for_short_numeric_case() {
        let mut ctx = Context::new();
        let factor = parse("2", &mut ctx).expect("parse");
        let additive = parse("x + 1", &mut ctx).expect("parse");
        assert!(!should_skip_distribution_for_factor(&ctx, factor, additive));
    }

    #[test]
    fn skips_distribution_for_complex_constant_factor() {
        let mut ctx = Context::new();
        let factor = parse("(sqrt(6)+sqrt(2))/4", &mut ctx).expect("parse");
        let additive = parse("x^4+4*x^3+6*x^2+4*x+1", &mut ctx).expect("parse");
        assert!(should_skip_distribution_for_factor(&ctx, factor, additive));
    }

    #[test]
    fn binomial_shape_detection_works() {
        let mut ctx = Context::new();
        let expr = parse("x + 1", &mut ctx).expect("parse");
        assert!(is_binomial_expr(&ctx, expr));
    }

    #[test]
    fn distribute_shape_allows_multivariable_variable_factor() {
        let mut ctx = Context::new();
        let factor = parse("x", &mut ctx).expect("parse");
        let additive = parse("y + 1", &mut ctx).expect("parse");
        let product = parse("x*(y+1)", &mut ctx).expect("parse");
        assert!(should_distribute_factor_over_additive(
            &ctx, factor, additive, product
        ));
    }

    #[test]
    fn fractional_coeff_over_binomial_is_blocked() {
        let mut ctx = Context::new();
        let coeff = parse("0.5", &mut ctx).expect("parse");
        let binomial = parse("sqrt(2) - 1", &mut ctx).expect("parse");
        assert!(should_block_fractional_coeff_over_binomial(
            &ctx, coeff, binomial
        ));
    }

    #[test]
    fn estimate_division_distribution_reduction_detects_structural_cancel() {
        let mut ctx = Context::new();
        let num = parse("x*y", &mut ctx).expect("parse");
        let den = parse("x", &mut ctx).expect("parse");
        assert!(estimate_division_distribution_simplification_reduction(&ctx, num, den) > 0);
    }
}
