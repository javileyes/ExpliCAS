use crate::linear_solution::NonZeroStatus;
use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, Expr, ExprId, SolutionSet};
use cas_math::expr_nary::add_terms_no_sign;
use cas_math::expr_predicates::is_one_expr as is_one;

/// Represents a fraction as (numerator, denominator).
#[derive(Debug, Clone)]
struct Fraction {
    num: ExprId,
    den: ExprId,
}

/// Convert an expression to fraction form.
/// - `a/b` -> (a, b)
/// - `a` -> (a, 1)
fn expr_to_fraction(ctx: &mut Context, expr: ExprId) -> Fraction {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => Fraction { num, den },
        _ => {
            let one = ctx.num(1);
            Fraction {
                num: expr,
                den: one,
            }
        }
    }
}

/// Build scale factor for a fraction: product of all OTHER denominators.
fn build_scale_factor(ctx: &mut Context, fractions: &[Fraction], my_den: ExprId) -> ExprId {
    let other_dens: Vec<ExprId> = fractions
        .iter()
        .filter(|f| f.den != my_den)
        .map(|f| f.den)
        .collect();

    if other_dens.is_empty() {
        ctx.num(1)
    } else if other_dens.len() == 1 {
        other_dens[0]
    } else {
        let mut product = other_dens[0];
        for &d in &other_dens[1..] {
            product = ctx.add(Expr::Mul(product, d));
        }
        product
    }
}

/// Combine multiple fractions into a single fraction (numerator, denominator).
///
/// Uses common denominator `D = ∏ den_i`.
/// Numerator `N = Σ (num_i × (D/den_i))`.
///
/// Returns `(N, D)` after light structural normalization.
pub fn combine_fractions_deterministic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let terms = add_terms_no_sign(ctx, expr);
    if terms.is_empty() {
        return None;
    }

    let fractions: Vec<Fraction> = terms.iter().map(|&t| expr_to_fraction(ctx, t)).collect();

    let common_denom = if fractions.len() == 1 {
        fractions[0].den
    } else {
        let mut denom = fractions[0].den;
        for frac in &fractions[1..] {
            denom = ctx.add(Expr::Mul(denom, frac.den));
        }
        denom
    };

    let mut scaled_nums: Vec<ExprId> = Vec::new();
    for frac in &fractions {
        let scale_factor = build_scale_factor(ctx, &fractions, frac.den);
        let scaled_num = if is_one(ctx, scale_factor) {
            frac.num
        } else {
            ctx.add(Expr::Mul(frac.num, scale_factor))
        };
        scaled_nums.push(scaled_num);
    }

    let numerator = if scaled_nums.len() == 1 {
        scaled_nums[0]
    } else {
        let mut sum = scaled_nums[0];
        for &term in &scaled_nums[1..] {
            sum = ctx.add(Expr::Add(sum, term));
        }
        sum
    };

    Some((numerator, common_denom))
}

/// Build solution set for reciprocal equations `1/x = N/D` where
/// candidate solution is `x = D/N` and the domain requires `N != 0`.
pub fn build_reciprocal_solution_set(
    numerator: ExprId,
    solution: ExprId,
    numerator_status: NonZeroStatus,
) -> SolutionSet {
    if numerator_status == NonZeroStatus::NonZero {
        return SolutionSet::Discrete(vec![solution]);
    }

    let guard = ConditionSet::single(ConditionPredicate::NonZero(numerator));
    let case = Case::new(guard, SolutionSet::Discrete(vec![solution]));
    SolutionSet::Conditional(vec![case])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isolation_utils::contains_var;

    #[test]
    fn test_combine_fractions_simple() {
        let mut ctx = Context::new();
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);

        let frac1 = ctx.add(Expr::Div(one, r1));
        let one2 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one2, r2));
        let sum = ctx.add(Expr::Add(frac1, frac2));

        let result = combine_fractions_deterministic(&mut ctx, sum);
        assert!(result.is_some());

        let (num, denom) = result.expect("must combine into a single fraction");
        assert!(contains_var(&ctx, num, "R1") || contains_var(&ctx, num, "R2"));
        assert!(contains_var(&ctx, denom, "R1"));
        assert!(contains_var(&ctx, denom, "R2"));
    }

    #[test]
    fn reciprocal_solution_nonzero_is_discrete() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::NonZero);
        assert_eq!(set, SolutionSet::Discrete(vec![sol]));
    }

    #[test]
    fn reciprocal_solution_unknown_is_conditional() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::Unknown);
        assert!(matches!(set, SolutionSet::Conditional(_)));
    }
}
