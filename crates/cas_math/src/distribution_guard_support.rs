//! Heuristics for skipping expensive distributive expansions.

use crate::expr_destructure::as_mul;
use crate::expr_nary::{build_balanced_add, AddView, Sign};
use crate::expr_rewrite::smart_mul;
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

/// True when distribution should be blocked for a binomial×binomial product,
/// except the sum/difference-of-cubes identity pair.
pub fn should_block_binomial_binomial_distribution(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    is_binomial_expr(ctx, left)
        && is_binomial_expr(ctx, right)
        && !crate::cube_identity_support::is_cube_identity_product(ctx, left, right)
}

/// True when distribution should be blocked because it would break a conjugate product.
///
/// Checks:
/// - direct conjugate pair: `(A+B)*(A-B)`
/// - parent-level conjugate protection where `additive_side` already has a conjugate sibling.
pub fn should_block_conjugate_distribution(
    ctx: &Context,
    factor_side: ExprId,
    additive_side: ExprId,
    parent_mul_terms: Option<(ExprId, ExprId)>,
) -> bool {
    if crate::expr_relations::is_conjugate_add_sub(ctx, factor_side, additive_side) {
        return true;
    }

    if let Some((parent_left, parent_right)) = parent_mul_terms {
        if crate::expr_relations::is_conjugate_add_sub(ctx, additive_side, parent_left)
            || crate::expr_relations::is_conjugate_add_sub(ctx, additive_side, parent_right)
        {
            return true;
        }
    }

    false
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdditiveOperator {
    Add,
    Sub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulDistributionCandidate {
    pub factor: ExprId,
    pub additive: ExprId,
    pub left_term: ExprId,
    pub right_term: ExprId,
    pub additive_operator: AdditiveOperator,
    pub factor_on_left: bool,
}

/// Classify `Mul` expressions of the form:
/// - `a*(b+c)` / `a*(b-c)`
/// - `(b+c)*a` / `(b-c)*a`
pub fn classify_mul_distribution_candidate(
    ctx: &Context,
    expr: ExprId,
) -> Option<MulDistributionCandidate> {
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };

    if let Expr::Add(b, c) = ctx.get(right) {
        return Some(MulDistributionCandidate {
            factor: left,
            additive: right,
            left_term: *b,
            right_term: *c,
            additive_operator: AdditiveOperator::Add,
            factor_on_left: true,
        });
    }
    if let Expr::Sub(b, c) = ctx.get(right) {
        return Some(MulDistributionCandidate {
            factor: left,
            additive: right,
            left_term: *b,
            right_term: *c,
            additive_operator: AdditiveOperator::Sub,
            factor_on_left: true,
        });
    }
    if let Expr::Add(b, c) = ctx.get(left) {
        return Some(MulDistributionCandidate {
            factor: right,
            additive: left,
            left_term: *b,
            right_term: *c,
            additive_operator: AdditiveOperator::Add,
            factor_on_left: false,
        });
    }
    if let Expr::Sub(b, c) = ctx.get(left) {
        return Some(MulDistributionCandidate {
            factor: right,
            additive: left,
            left_term: *b,
            right_term: *c,
            additive_operator: AdditiveOperator::Sub,
            factor_on_left: false,
        });
    }

    None
}

/// Plan multiplicative distribution when all structural guards allow it.
///
/// This helper centralizes shape and anti-explosion checks for:
/// - `a*(b+c)` / `a*(b-c)` and symmetric variants.
///
/// Context-level policy (goal mode, canonical-form exclusions, pattern-mark
/// gates) should be handled by the caller.
pub fn try_plan_mul_distribution_candidate(
    ctx: &Context,
    expr: ExprId,
    parent_mul_terms: Option<(ExprId, ExprId)>,
) -> Option<MulDistributionCandidate> {
    let (left, right) = as_mul(ctx, expr)?;

    // 1*(a+b) -> 1*a + 1*b is a visual no-op and noisy didactically.
    if crate::expr_predicates::is_one_expr(ctx, left)
        || crate::expr_predicates::is_one_expr(ctx, right)
    {
        return None;
    }

    let candidate = classify_mul_distribution_candidate(ctx, expr)?;

    if should_skip_distribution_for_factor(ctx, candidate.factor, candidate.additive) {
        return None;
    }

    if !should_distribute_factor_over_additive(ctx, candidate.factor, candidate.additive, expr) {
        return None;
    }

    if should_block_conjugate_distribution(
        ctx,
        candidate.factor,
        candidate.additive,
        parent_mul_terms,
    ) {
        return None;
    }

    // Don't expand binomial*binomial products unless special identity rules handle it.
    if should_block_binomial_binomial_distribution(ctx, left, right) {
        return None;
    }

    if should_block_fractional_coeff_over_binomial(ctx, candidate.factor, candidate.additive) {
        return None;
    }

    Some(candidate)
}

/// Build the distributed expression from a previously-classified candidate.
pub fn build_mul_distribution_expr(
    ctx: &mut Context,
    candidate: MulDistributionCandidate,
) -> ExprId {
    let first = if candidate.factor_on_left {
        smart_mul(ctx, candidate.factor, candidate.left_term)
    } else {
        smart_mul(ctx, candidate.left_term, candidate.factor)
    };
    let second = if candidate.factor_on_left {
        smart_mul(ctx, candidate.factor, candidate.right_term)
    } else {
        smart_mul(ctx, candidate.right_term, candidate.factor)
    };

    match candidate.additive_operator {
        AdditiveOperator::Add => ctx.add(Expr::Add(first, second)),
        AdditiveOperator::Sub => ctx.add(Expr::Sub(first, second)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivisionDistributionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributionRewritePlan {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite `(a+b+...)/d` into `a/d + b/d + ...` only when estimated simplification
/// gain indicates the transformation is not complexity-worsening.
pub fn try_rewrite_distribute_division_when_simplifying_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivisionDistributionRewrite> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let numerator_view = AddView::from_expr(ctx, numerator);
    if numerator_view.terms.len() <= 1 {
        return None;
    }

    let mut total_reduction: usize = 0;
    let mut any_simplifies = false;
    for &(term, _sign) in &numerator_view.terms {
        let reduction =
            estimate_division_distribution_simplification_reduction(ctx, term, denominator);
        if reduction > 0 {
            any_simplifies = true;
            total_reduction += reduction;
        }
    }
    if !any_simplifies {
        return None;
    }

    let rewritten_terms: Vec<ExprId> = numerator_view
        .terms
        .iter()
        .map(|&(term, sign)| {
            let divided = ctx.add(Expr::Div(term, denominator));
            match sign {
                Sign::Pos => divided,
                Sign::Neg => ctx.add(Expr::Neg(divided)),
            }
        })
        .collect();
    let rewritten = build_balanced_add(ctx, &rewritten_terms);

    let old_complexity = cas_ast::count_nodes(ctx, expr);
    let new_complexity = cas_ast::count_nodes(ctx, rewritten);
    if new_complexity > old_complexity + total_reduction {
        return None;
    }

    Some(DivisionDistributionRewrite {
        rewritten,
        desc: "Distribute division (simplifying)",
    })
}

/// Plan a distributive rewrite for multiplication or simplification-oriented
/// division distribution.
pub fn try_plan_distribution_rewrite_expr(
    ctx: &mut Context,
    expr: ExprId,
    parent_mul_terms: Option<(ExprId, ExprId)>,
) -> Option<DistributionRewritePlan> {
    if let Some(candidate) = try_plan_mul_distribution_candidate(ctx, expr, parent_mul_terms) {
        let rewritten = build_mul_distribution_expr(ctx, candidate);
        return Some(DistributionRewritePlan {
            rewritten,
            desc: "Distribute",
        });
    }

    if let Some(rewrite) = try_rewrite_distribute_division_when_simplifying_expr(ctx, expr) {
        return Some(DistributionRewritePlan {
            rewritten: rewrite.rewritten,
            desc: rewrite.desc,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        build_mul_distribution_expr, classify_mul_distribution_candidate,
        estimate_division_distribution_simplification_reduction, has_fractional_exponents,
        is_binomial_expr, should_block_binomial_binomial_distribution,
        should_block_conjugate_distribution, should_block_fractional_coeff_over_binomial,
        should_distribute_factor_over_additive, should_skip_distribution_for_factor,
        try_plan_distribution_rewrite_expr, try_plan_mul_distribution_candidate,
        try_rewrite_distribute_division_when_simplifying_expr, AdditiveOperator,
    };
    use cas_ast::{Context, Expr};
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

    #[test]
    fn blocks_binomial_binomial_distribution_except_cube_identity() {
        let mut ctx = Context::new();
        let left = parse("x + 1", &mut ctx).expect("parse");
        let right = parse("x - 2", &mut ctx).expect("parse");
        assert!(should_block_binomial_binomial_distribution(
            &ctx, left, right
        ));

        let cube_left = parse("x + 1", &mut ctx).expect("parse");
        let cube_right = parse("x^2 - x + 1", &mut ctx).expect("parse");
        assert!(!should_block_binomial_binomial_distribution(
            &ctx, cube_left, cube_right
        ));
    }

    #[test]
    fn rewrite_distribute_division_when_term_simplifies() {
        let mut ctx = Context::new();
        let expr = parse("(x*y + 1)/x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_distribute_division_when_simplifying_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn block_conjugate_distribution_detects_direct_pair() {
        let mut ctx = Context::new();
        let factor = parse("x + 2", &mut ctx).expect("parse");
        let additive = parse("x - 2", &mut ctx).expect("parse");
        assert!(should_block_conjugate_distribution(
            &ctx, factor, additive, None
        ));
    }

    #[test]
    fn block_conjugate_distribution_detects_parent_pair() {
        let mut ctx = Context::new();
        let additive = parse("x + 2", &mut ctx).expect("parse");
        let parent_left = parse("x - 2", &mut ctx).expect("parse");
        let parent_right = parse("y", &mut ctx).expect("parse");
        let factor = parse("k", &mut ctx).expect("parse");
        assert!(should_block_conjugate_distribution(
            &ctx,
            factor,
            additive,
            Some((parent_left, parent_right))
        ));
    }

    #[test]
    fn classify_mul_distribution_candidate_detects_right_additive() {
        let mut ctx = Context::new();
        let expr = parse("a*(b+c)", &mut ctx).expect("parse");
        let candidate = classify_mul_distribution_candidate(&ctx, expr).expect("candidate");
        assert!(candidate.factor_on_left);
        assert_eq!(candidate.additive_operator, AdditiveOperator::Add);
    }

    #[test]
    fn classify_mul_distribution_candidate_detects_left_subtractive() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let c = ctx.var("c");
        let one = ctx.num(1);
        let matrix = ctx.add(Expr::Matrix {
            rows: 1,
            cols: 1,
            data: vec![one],
        });
        let sub = ctx.add(Expr::Sub(b, c));
        let expr = ctx.add(Expr::Mul(sub, matrix));
        let candidate = classify_mul_distribution_candidate(&ctx, expr).expect("candidate");
        assert!(!candidate.factor_on_left);
        assert_eq!(candidate.additive_operator, AdditiveOperator::Sub);
    }

    #[test]
    fn build_mul_distribution_expr_respects_factor_position() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let c = ctx.var("c");
        let one = ctx.num(1);
        let matrix = ctx.add(Expr::Matrix {
            rows: 1,
            cols: 1,
            data: vec![one],
        });
        let add = ctx.add(Expr::Add(b, c));
        let expr = ctx.add(Expr::Mul(add, matrix));
        let candidate = classify_mul_distribution_candidate(&ctx, expr).expect("candidate");
        let rewritten = build_mul_distribution_expr(&mut ctx, candidate);
        let left = ctx.add(Expr::Mul(b, matrix));
        let right = ctx.add(Expr::Mul(c, matrix));
        let expected = ctx.add(Expr::Add(left, right));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn try_plan_mul_distribution_candidate_accepts_simple_case() {
        let mut ctx = Context::new();
        let expr = parse("x*(y+1)", &mut ctx).expect("parse");
        let planned = try_plan_mul_distribution_candidate(&ctx, expr, None);
        assert!(planned.is_some());
    }

    #[test]
    fn try_plan_mul_distribution_candidate_rejects_one_factor_noop() {
        let mut ctx = Context::new();
        let expr = parse("1*(y+1)", &mut ctx).expect("parse");
        let planned = try_plan_mul_distribution_candidate(&ctx, expr, None);
        assert!(planned.is_none());
    }

    #[test]
    fn try_plan_distribution_rewrite_handles_mul_distribution() {
        let mut ctx = Context::new();
        let expr = parse("x*(y+1)", &mut ctx).expect("parse");
        let planned = try_plan_distribution_rewrite_expr(&mut ctx, expr, None).expect("plan");
        assert_eq!(planned.desc, "Distribute");
        assert!(matches!(ctx.get(planned.rewritten), Expr::Add(_, _)));
    }
}
