use crate::build::mul2_raw;
use crate::define_rule;
use crate::helpers::{as_div, as_mul, as_neg, as_pow};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

use super::can_distribute_root_safely;

define_rule!(
    IdentityPowerRule,
    "Identity Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_nonzero;

        if let Some((base, exp)) = as_pow(ctx, expr) {
            // x^1 -> x (always safe)
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_one() {
                    return Some(Rewrite::new(base).desc("x^1 -> x"));
                }
                if n.is_zero() {
                    // x^0 -> 1 REQUIRES x ≠ 0 (because 0^0 is undefined)
                    let mode = parent_ctx.domain_mode();
                    let proof = prove_nonzero(ctx, base);

                    // Check if base is literal 0 -> 0^0 = undefined
                    if let Expr::Number(b) = ctx.get(base) {
                        if b.is_zero() {
                            return Some(Rewrite::new(ctx.add(Expr::Constant(cas_ast::Constant::Undefined))).desc("0^0 -> undefined"));
                        }
                    }

                    match mode {
                        DomainMode::Generic => {
                            let needs_assumption = !matches!(ctx.get(base), Expr::Number(_));
                            let mut rewrite = Rewrite::new(ctx.num(1)).desc("x^0 -> 1");
                            if needs_assumption {
                                rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonzero(ctx, base));
                            }
                            return Some(rewrite);
                        }
                        DomainMode::Strict => {
                            if proof == Proof::Proven {
                                return Some(Rewrite::new(ctx.num(1)).desc("x^0 -> 1 (x ≠ 0 proven)"));
                            }
                            return None;
                        }
                        DomainMode::Assume => {
                            return Some(Rewrite::new(ctx.num(1))
                                .desc("x^0 -> 1 (assuming x ≠ 0)")
                                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, base)));
                        }
                    }
                }
            }
            // 1^x -> 1 (always safe)
            if let Expr::Number(n) = ctx.get(base) {
                if n.is_one() {
                    return Some(Rewrite::new(ctx.num(1)).desc("1^x -> 1"));
                }
                // 0^x -> 0 REQUIRES x > 0
                if n.is_zero() {
                    if let Expr::Number(e) = ctx.get(exp) {
                        if *e > num_rational::BigRational::zero() {
                            return Some(Rewrite::new(ctx.num(0)).desc("0^n -> 0 (n > 0)"));
                        }
                        return None;
                    }
                    let mode = parent_ctx.domain_mode();
                    let vd = parent_ctx.value_domain();
                    let proof = crate::helpers::prove_positive(ctx, exp, vd);
                    let key = crate::assumptions::AssumptionKey::positive_key(ctx, exp);
                    let decision = crate::domain::can_apply_analytic_with_hint(
                        mode,
                        proof,
                        key,
                        exp,
                        "Evaluate Power",
                    );

                    if decision.allow {
                        let mut rewrite = Rewrite::new(ctx.num(0)).desc("0^x → 0");
                        for event in decision.assumption_events(ctx, exp) {
                            rewrite = rewrite.assume(event);
                        }
                        return Some(rewrite);
                    }
                    return None;
                }
            }
        }
        None
    }
);

define_rule!(
    PowerProductRule,
    "Power of a Product",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr| {
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // (a * b)^n -> a^n * b^n
        if let Some((base, exp)) = as_pow(ctx, expr) {
            if let Some((a, b)) = as_mul(ctx, base) {
                if let Expr::Number(exp_num) = ctx.get(exp) {
                    let denom = exp_num.denom();
                    if denom > &num_bigint::BigInt::from(1)
                        && !can_distribute_root_safely(ctx, base, denom)
                    {
                        return None;
                    }
                }

                let a_is_num = matches!(ctx.get(a), Expr::Number(_));
                let b_is_num = matches!(ctx.get(b), Expr::Number(_));
                let exp_is_numeric = matches!(ctx.get(exp), Expr::Number(_));

                if (a_is_num || b_is_num) && !exp_is_numeric {
                    return None;
                }

                let a_pow = ctx.add(Expr::Pow(a, exp));
                let b_pow = ctx.add(Expr::Pow(b, exp));
                let new_expr = mul2_raw(ctx, a_pow, b_pow);

                return Some(Rewrite::new(new_expr).desc("Distribute power over product"));
            }
        }
        None
    }
);

define_rule!(PowerQuotientRule, "Power of a Quotient", |ctx, expr| {
    // (a / b)^n -> a^n / b^n
    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Expr::Number(exp_num) = ctx.get(exp) {
            let denom = exp_num.denom();
            if denom > &num_bigint::BigInt::from(1) && !can_distribute_root_safely(ctx, base, denom)
            {
                return None;
            }
        }

        if let Some((num, den)) = as_div(ctx, base) {
            let new_num = ctx.add(Expr::Pow(num, exp));
            let new_den = ctx.add(Expr::Pow(den, exp));
            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr).desc("Distribute power over quotient"));
        }
    }
    None
});

// ============================================================================
// ExpQuotientRule: e^a / e^b → e^(a-b)
// ============================================================================
define_rule!(ExpQuotientRule, "Exp Quotient", |ctx, expr| {
    if let Some((num, den)) = as_div(ctx, expr) {
        let num_pow = as_pow(ctx, num);
        let den_pow = as_pow(ctx, den);

        if let (Some((num_base, num_exp)), Some((den_base, den_exp))) = (num_pow, den_pow) {
            let num_base_is_e = matches!(ctx.get(num_base), Expr::Constant(cas_ast::Constant::E));
            let den_base_is_e = matches!(ctx.get(den_base), Expr::Constant(cas_ast::Constant::E));

            if num_base_is_e && den_base_is_e {
                let diff = ctx.add(Expr::Sub(num_exp, den_exp));
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                let new_expr = ctx.add(Expr::Pow(e, diff));
                return Some(Rewrite::new(new_expr).desc("e^a / e^b = e^(a-b)"));
            }
        }

        // e / e^b → e^(1-b)
        if matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::E)) {
            if let Some((den_base, den_exp)) = den_pow {
                if matches!(ctx.get(den_base), Expr::Constant(cas_ast::Constant::E)) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(one, den_exp));
                    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                    let new_expr = ctx.add(Expr::Pow(e, diff));
                    return Some(Rewrite::new(new_expr).desc("e / e^b = e^(1-b)"));
                }
            }
        }

        // e^a / e → e^(a-1)
        if matches!(ctx.get(den), Expr::Constant(cas_ast::Constant::E)) {
            if let Some((num_base, num_exp)) = num_pow {
                if matches!(ctx.get(num_base), Expr::Constant(cas_ast::Constant::E)) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(num_exp, one));
                    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                    let new_expr = ctx.add(Expr::Pow(e, diff));
                    return Some(Rewrite::new(new_expr).desc("e^a / e = e^(a-1)"));
                }
            }
        }
    }
    None
});

// ============================================================================
// MulNaryCombinePowersRule
// ============================================================================
pub struct MulNaryCombinePowersRule;

impl crate::rule::Rule for MulNaryCombinePowersRule {
    fn name(&self) -> &str {
        "N-ary Mul Combine Powers"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        let factors = crate::nary::mul_leaves(ctx, expr);

        if factors.len() > 12 || factors.len() < 2 {
            return None;
        }

        let mut base_exp_pairs: Vec<(ExprId, Option<BigRational>, bool)> = Vec::new();
        for &factor in &factors {
            match ctx.get(factor) {
                Expr::Pow(base, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        base_exp_pairs.push((*base, Some(n.clone()), true));
                    } else {
                        base_exp_pairs.push((factor, None, false));
                    }
                }
                Expr::Number(_) => {
                    base_exp_pairs.push((factor, Some(BigRational::one()), false));
                }
                _ => {
                    base_exp_pairs.push((factor, Some(BigRational::one()), true));
                }
            }
        }

        let mut combined: Vec<(ExprId, BigRational, usize)> = Vec::new();
        let mut absorbed = vec![false; factors.len()];
        let mut any_combined = false;

        for i in 0..base_exp_pairs.len() {
            if absorbed[i] {
                continue;
            }

            let (base_i, exp_i, is_pow_i) = &base_exp_pairs[i];
            let Some(mut sum_exp) = exp_i.clone() else {
                continue;
            };

            if !is_pow_i {
                continue;
            }

            let mut found_match = false;

            for j in (i + 1)..base_exp_pairs.len() {
                if absorbed[j] {
                    continue;
                }

                let (base_j, exp_j, is_pow_j) = &base_exp_pairs[j];

                if !is_pow_j {
                    continue;
                }

                let Some(exp_j_val) = exp_j else {
                    continue;
                };

                if compare_expr(ctx, *base_i, *base_j) == Ordering::Equal {
                    sum_exp += exp_j_val;
                    absorbed[j] = true;
                    found_match = true;
                }
            }

            if found_match {
                absorbed[i] = true;
                combined.push((*base_i, sum_exp, i));
                any_combined = true;
            }
        }

        if !any_combined {
            return None;
        }

        let mut result_factors: Vec<ExprId> = Vec::new();

        let mut combined_map: std::collections::HashMap<usize, (ExprId, BigRational)> =
            std::collections::HashMap::new();
        for (base, sum_exp, first_idx) in &combined {
            combined_map.insert(*first_idx, (*base, sum_exp.clone()));
        }

        for i in 0..factors.len() {
            if let Some((base, sum_exp)) = combined_map.get(&i) {
                let new_factor = if sum_exp.is_one() {
                    *base
                } else if sum_exp.is_zero() {
                    ctx.num(1)
                } else {
                    let exp_id = ctx.add(Expr::Number(sum_exp.clone()));
                    ctx.add(Expr::Pow(*base, exp_id))
                };
                result_factors.push(new_factor);
            } else if !absorbed[i] {
                result_factors.push(factors[i]);
            }
        }

        if result_factors.is_empty() {
            return Some(Rewrite::new(ctx.num(1)).desc("All factors cancelled"));
        }

        if let Some((&last, rest)) = result_factors.split_last() {
            let mut result = last;
            for &factor in rest.iter().rev() {
                result = mul2_raw(ctx, factor, result);
            }

            if result_factors.len() < factors.len() {
                Some(Rewrite::new(result).desc("Combine powers with same base (n-ary)"))
            } else {
                None
            }
        } else {
            // unreachable: empty case returns at line 350
            None
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

define_rule!(NegativeBasePowerRule, "Negative Base Power", |ctx, expr| {
    // (-x)^n
    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Some(inner) = as_neg(ctx, base) {
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() {
                    if n.to_integer().is_even() {
                        let new_expr = ctx.add(Expr::Pow(inner, exp));
                        return Some(Rewrite::new(new_expr).desc("(-x)^even -> x^even"));
                    } else {
                        let pow = ctx.add(Expr::Pow(inner, exp));
                        let new_expr = ctx.add(Expr::Neg(pow));
                        return Some(Rewrite::new(new_expr).desc("(-x)^odd -> -(x^odd)"));
                    }
                }
            }
        }
    }
    None
});

// Canonicalize bases in even powers: (b-a)^even → (a-b)^even when a < b
define_rule!(
    EvenPowSubSwapRule,
    "Canonicalize Even Power Base",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        let (base, exp) = as_pow(ctx, expr)?;

        let is_even = match ctx.get(exp) {
            Expr::Number(n) => n.is_integer() && n.to_integer().is_even(),
            _ => false,
        };
        if !is_even {
            return None;
        }

        let (pos_term, neg_term) = match ctx.get(base) {
            Expr::Sub(a, b) => (*a, *b),
            Expr::Add(l, r) => {
                if let Expr::Neg(inner) = ctx.get(*r) {
                    (*l, *inner)
                } else if let Expr::Neg(inner) = ctx.get(*l) {
                    (*r, *inner)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        if compare_expr(ctx, neg_term, pos_term) != Ordering::Less {
            return None;
        }

        let new_base = ctx.add(Expr::Sub(neg_term, pos_term));
        let new_expr = ctx.add(Expr::Pow(new_base, exp));

        Some(Rewrite::new(new_expr)
            .desc("For even exponent: (a-b)² = (b-a)², normalize for cancellation")
            .local(base, new_base))
    }
);
