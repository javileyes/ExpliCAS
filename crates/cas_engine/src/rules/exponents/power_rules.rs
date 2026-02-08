use crate::build::mul2_raw;
use crate::define_rule;
use crate::helpers::{as_div, as_mul, as_pow, is_half};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive};
use std::cmp::Ordering;

use super::{add_exp, has_numeric_factor, mul_exp};

define_rule!(ProductPowerRule, "Product of Powers", |ctx, expr| {
    // x^a * x^b -> x^(a+b)
    let should_combine = |ctx: &Context, base: ExprId, e1: ExprId, e2: ExprId| -> bool {
        if let Expr::Number(_) = ctx.get(base) {
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
                let sum = n1 + n2;
                if sum.is_integer() {
                    return true;
                }
                // Check if proper fraction: |num| < den
                let num = sum.numer().abs();
                let den = sum.denom().abs();
                return num < den;
            }
        }
        true
    };

    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        let lhs_pow = as_pow(ctx, lhs);
        let rhs_pow = as_pow(ctx, rhs);

        // Case 1: Both are powers with same base: x^a * x^b
        if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
            if compare_expr(ctx, base1, base2) == Ordering::Equal
                && should_combine(ctx, base1, exp1, exp2)
            {
                let sum_exp = add_exp(ctx, exp1, exp2);
                let new_expr = ctx.add(Expr::Pow(base1, sum_exp));
                return Some(Rewrite::new(new_expr).desc("Combine powers with same base"));
            }
        }
        // Case 2: One is power, one is base: x^a * x -> x^(a+1)
        // Left is power
        if let Some((base1, exp1)) = lhs_pow {
            if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base1, exp1, one) {
                    let sum_exp = add_exp(ctx, exp1, one);
                    let new_expr = ctx.add(Expr::Pow(base1, sum_exp));
                    return Some(Rewrite::new(new_expr).desc("Combine power and base"));
                }
            }
        }
        // Right is power
        if let Some((base2, exp2)) = rhs_pow {
            if compare_expr(ctx, base2, lhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base2, one, exp2) {
                    let sum_exp = add_exp(ctx, one, exp2);
                    let new_expr = ctx.add(Expr::Pow(base2, sum_exp));
                    return Some(Rewrite::new(new_expr).desc("Combine base and power"));
                }
            }
        }
        // Case 3: Both are same base (implicit power 1): x * x -> x^2
        if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
            let two = ctx.num(2);
            let new_expr = ctx.add(Expr::Pow(lhs, two));
            return Some(Rewrite::new(new_expr).desc("Multiply identical terms"));
        }

        // Case 4: Nested Multiplication: x * (x * y) -> x^2 * y
        if let Some((rl, rr)) = as_mul(ctx, rhs) {
            // x * (x * y)
            if compare_expr(ctx, lhs, rl) == Ordering::Equal {
                let two = ctx.num(2);
                let x_squared = ctx.add(Expr::Pow(lhs, two));
                let new_expr = mul2_raw(ctx, x_squared, rr);
                return Some(Rewrite::new(new_expr).desc("Combine nested identical terms"));
            }

            // x^a * (x^b * y) -> x^(a+b) * y
            let rl_pow = as_pow(ctx, rl);

            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rl_pow) {
                if compare_expr(ctx, base1, base2) == Ordering::Equal {
                    let sum_exp = add_exp(ctx, exp1, exp2);
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let new_expr = mul2_raw(ctx, new_pow, rr);
                    return Some(Rewrite::new(new_expr).desc("Combine nested powers"));
                }
            }

            // x * (x^a * y) -> x^(a+1) * y
            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = add_exp(ctx, exp2, one);
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let new_expr = mul2_raw(ctx, new_pow, rr);
                    return Some(Rewrite::new(new_expr).desc("Combine base and nested power"));
                }
            }

            // x^a * (x * y) -> x^(a+1) * y
            if let Some((base1, exp1)) = lhs_pow {
                if compare_expr(ctx, base1, rl) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(exp1, one));
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let new_expr = mul2_raw(ctx, new_pow, rr);
                    return Some(Rewrite::new(new_expr).desc("Combine power and nested base"));
                }
            }

            // (c * x^a) * x^b -> c * x^(a+b)
            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    let lr_pow = as_pow(ctx, lr);

                    if let (Some((base1, exp1)), Some((base2, exp2))) = (lr_pow, rhs_pow) {
                        if compare_expr(ctx, base1, base2) == Ordering::Equal {
                            let sum_exp = add_exp(ctx, exp1, exp2);
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let new_expr = mul2_raw(ctx, ll, new_pow);
                            return Some(
                                Rewrite::new(new_expr).desc("Combine coeff-power and power"),
                            );
                        }
                    }

                    // (c * x^a) * x -> c * x^(a+1)
                    if let Some((base1, exp1)) = lr_pow {
                        if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = ctx.add(Expr::Add(exp1, one));
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let new_expr = mul2_raw(ctx, ll, new_pow);
                            return Some(
                                Rewrite::new(new_expr).desc("Combine coeff-power and base"),
                            );
                        }
                    }
                }
            }

            // (c * x) * x^a -> c * x^(a+1)
            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    if let Some((base2, exp2)) = rhs_pow {
                        if compare_expr(ctx, lr, base2) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = add_exp(ctx, exp2, one);
                            let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                            let new_expr = mul2_raw(ctx, ll, new_pow);
                            return Some(
                                Rewrite::new(new_expr).desc("Combine coeff-base and power"),
                            );
                        }
                    }
                }
            }

            // x * (x^b * y) -> x^(1+b) * y
            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(one, exp2));
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let new_expr = mul2_raw(ctx, new_pow, rr);
                    return Some(Rewrite::new(new_expr).desc("Combine nested base and power"));
                }
            }
        }
    }
    None
});

// a^n * b^n = (ab)^n - combines products of powers with same exponent
// Guard: at least one base must contain a numeric factor to avoid infinite loop with PowerProductRule
define_rule!(
    ProductSameExponentRule,
    "Product Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr| {
        if let Some((lhs, rhs)) = as_mul(ctx, expr) {
            let lhs_pow = as_pow(ctx, lhs);
            let rhs_pow = as_pow(ctx, rhs);

            // Case 1: Both are powers with same exponent: a^n * b^n
            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
                if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                    let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                    let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                    let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                    let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);

                    if !base1_has_num && !base2_has_num {
                        return None;
                    }

                    let new_base = mul2_raw(ctx, base1, base2);
                    let new_expr = ctx.add(Expr::Pow(new_base, exp1));
                    return Some(Rewrite::new(new_expr).desc("Combine powers with same exponent"));
                }
            }

            // Case 2: Nested: a^n * (b^n * c) -> (a*b)^n * c
            if let Some((base1, exp1)) = lhs_pow {
                if let Some((rl, rr)) = as_mul(ctx, rhs) {
                    if let Some((base2, exp2)) = as_pow(ctx, rl) {
                        if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                            let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                            let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                            let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                            let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);

                            if !base1_has_num && !base2_has_num {
                                return None;
                            }

                            let new_base = mul2_raw(ctx, base1, base2);
                            let combined_pow = ctx.add(Expr::Pow(new_base, exp1));
                            let new_expr = mul2_raw(ctx, combined_pow, rr);
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("Combine nested powers with same exponent"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
);

// a^n / b^n = (a/b)^n
define_rule!(
    QuotientSameExponentRule,
    "Quotient Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        if let (Expr::Pow(base_num, exp_num), Expr::Pow(base_den, exp_den)) =
            (ctx.get(num), ctx.get(den))
        {
            let (base_num, exp_num, base_den, exp_den) = (*base_num, *exp_num, *base_den, *exp_den);
            if compare_expr(ctx, exp_num, exp_den) == Ordering::Equal {
                let base_num_is_num = matches!(ctx.get(base_num), Expr::Number(_));
                let base_den_is_num = matches!(ctx.get(base_den), Expr::Number(_));
                let base_num_has_num = base_num_is_num || has_numeric_factor(ctx, base_num);
                let base_den_has_num = base_den_is_num || has_numeric_factor(ctx, base_den);

                if !base_num_has_num && !base_den_has_num {
                    return None;
                }

                let new_base = ctx.add(Expr::Div(base_num, base_den));
                let new_expr = ctx.add(Expr::Pow(new_base, exp_num));
                return Some(Rewrite::new(new_expr).desc("a^n / b^n = (a/b)^n"));
            }
        }

        None
    }
);

// ============================================================================
// RootPowCancelRule: (x^n)^(1/n) → x (odd n) or |x| (even n)
// ============================================================================
pub struct RootPowCancelRule;

impl crate::rule::Rule for RootPowCancelRule {
    fn name(&self) -> &str {
        "Root Power Cancel"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        let (base, outer_exp) = as_pow(ctx, expr)?;

        let (inner_base, inner_exp) = as_pow(ctx, base)?;

        let combined_is_one = match (ctx.get(outer_exp), ctx.get(inner_exp)) {
            (Expr::Number(o), Expr::Number(i)) => {
                let combined = o * i;
                combined.is_one()
            }
            _ => {
                if let Some((num, denom)) = as_div(ctx, outer_exp) {
                    if let Expr::Number(n) = ctx.get(num) {
                        if n.is_one() {
                            crate::ordering::compare_expr(ctx, denom, inner_exp) == Ordering::Equal
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        };

        if !combined_is_one {
            return None;
        }

        let vd = parent_ctx.value_domain();
        if vd == ValueDomain::ComplexEnabled {
            return None;
        }

        if let Expr::Number(n) = ctx.get(inner_exp) {
            if n.is_integer() {
                let n_int = n.to_integer();
                let is_even = n_int.is_even();

                if is_even {
                    let abs_base = ctx.call("abs", vec![inner_base]);
                    return Some(
                        crate::rule::Rewrite::new(abs_base).desc("(x^n)^(1/n) = |x| for even n"),
                    );
                } else {
                    return Some(
                        crate::rule::Rewrite::new(inner_base).desc("(x^n)^(1/n) = x for odd n"),
                    );
                }
            } else {
                return None;
            }
        }

        // n is symbolic
        let dm = parent_ctx.domain_mode();

        match dm {
            crate::domain::DomainMode::Strict | crate::domain::DomainMode::Generic => {
                let hint1 = crate::domain::BlockedHint {
                    key: crate::assumptions::AssumptionKey::positive_key(ctx, inner_base),
                    expr_id: inner_base,
                    rule: "Root Power Cancel".to_string(),
                    suggestion: "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.",
                };
                let hint2 = crate::domain::BlockedHint {
                    key: crate::assumptions::AssumptionKey::nonzero_key(ctx, inner_exp),
                    expr_id: inner_exp,
                    rule: "Root Power Cancel".to_string(),
                    suggestion: "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.",
                };
                crate::domain::register_blocked_hint(hint1);
                crate::domain::register_blocked_hint(hint2);
                None
            }
            crate::domain::DomainMode::Assume => {
                use crate::implicit_domain::ImplicitCondition;
                Some(
                    crate::rule::Rewrite::new(inner_base)
                        .desc("(x^n)^(1/n) = x (assuming x > 0, n ≠ 0)")
                        .requires(ImplicitCondition::Positive(inner_base))
                        .requires(ImplicitCondition::NonZero(inner_exp))
                        .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                            ctx, inner_base,
                        )),
                )
            }
        }
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }

    fn priority(&self) -> i32 {
        15
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

define_rule!(
    PowerPowerRule,
    "Power of a Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
    // (x^a)^b -> x^(a*b)
    if let Some((base, outer_exp)) = as_pow(ctx, expr) {
        if let Some((inner_base, inner_exp)) = as_pow(ctx, base) {
            let is_even_int = |e: ExprId| -> bool {
                if let Expr::Number(n) = ctx.get(e) {
                    n.is_integer() && n.to_integer().is_even()
                } else {
                    false
                }
            };

            if is_even_int(inner_exp) && is_half(ctx, outer_exp) {
                let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                let abs_base = ctx.call("abs", vec![inner_base]);
                let new_expr = ctx.add(Expr::Pow(abs_base, prod_exp));
                return Some(Rewrite::new(new_expr).desc("Power of power with even root: (x^2k)^(1/2) -> |x|^k"));
            }

            let inner_is_even_root = if let Expr::Number(n) = ctx.get(inner_exp) {
                let denom = n.denom();
                denom.is_even()
            } else {
                false
            };

            if inner_is_even_root {
                use crate::domain::{can_apply_analytic_with_hint, Proof};
                use crate::helpers::prove_nonnegative;
                use crate::implicit_domain::{witness_survives_in_context, WitnessKind};

                let mode = parent_ctx.domain_mode();
                let vd = parent_ctx.value_domain();
                let base_proof = prove_nonnegative(ctx, inner_base, vd);

                let proof = if matches!(base_proof, Proof::Unknown) {
                    if let (Some(implicit), Some(root)) = (parent_ctx.implicit_domain(), parent_ctx.root_expr()) {
                        if implicit.contains_nonnegative(inner_base) {
                            let output_candidate = inner_base;
                            if witness_survives_in_context(
                                ctx,
                                inner_base,
                                root,
                                expr,
                                Some(output_candidate),
                                WitnessKind::Sqrt,
                            ) {
                                Proof::ProvenImplicit
                            } else {
                                Proof::Unknown
                            }
                        } else {
                            Proof::Unknown
                        }
                    } else {
                        Proof::Unknown
                    }
                } else {
                    base_proof
                };

                if matches!(proof, Proof::Proven | Proof::ProvenImplicit) {
                    let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                    let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
                    return Some(Rewrite::new(new_expr).desc("Multiply exponents"));
                }

                let key = crate::assumptions::AssumptionKey::nonnegative_key(ctx, inner_base);
                let decision = can_apply_analytic_with_hint(
                    mode,
                    proof,
                    key,
                    inner_base,
                    "Power of a Power",
                );

                if !decision.allow {
                    return None;
                }

                let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));

                let mut rewrite = Rewrite::new(new_expr).desc("Multiply exponents");
                if decision.assumption.is_some() {
                    rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, inner_base));
                }
                return Some(rewrite);
            }

            let is_symbolic_root_cancel = if let Some((_num, denom)) = as_div(ctx, outer_exp) {
                crate::ordering::compare_expr(ctx, denom, inner_exp) == Ordering::Equal
            } else {
                false
            };

            if is_symbolic_root_cancel {
                let dm = parent_ctx.domain_mode();
                let vd = parent_ctx.value_domain();

                if vd == crate::semantics::ValueDomain::RealOnly {
                    match dm {
                        crate::domain::DomainMode::Strict | crate::domain::DomainMode::Generic => {
                            let hint1 = crate::domain::BlockedHint {
                                key: crate::assumptions::AssumptionKey::positive_key(ctx, inner_base),
                                expr_id: inner_base,
                                rule: "Power of a Power".to_string(),
                                suggestion: "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.",
                            };
                            let hint2 = crate::domain::BlockedHint {
                                key: crate::assumptions::AssumptionKey::nonzero_key(ctx, inner_exp),
                                expr_id: inner_exp,
                                rule: "Power of a Power".to_string(),
                                suggestion: "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.",
                            };
                            crate::domain::register_blocked_hint(hint1);
                            crate::domain::register_blocked_hint(hint2);
                            return None;
                        }
                        crate::domain::DomainMode::Assume => {
                            use crate::implicit_domain::ImplicitCondition;
                            let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                            let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("(x^n)^(1/n) = x (assuming x > 0, n ≠ 0)")
                                    .requires(ImplicitCondition::Positive(inner_base))
                                    .requires(ImplicitCondition::NonZero(inner_exp))
                                    .assume(crate::assumptions::AssumptionEvent::positive_assumed(ctx, inner_base))
                            );
                        }
                    }
                }
            }

            // Default case
            let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
            let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
            return Some(Rewrite::new(new_expr).desc("Multiply exponents"));
        }
    }
    None
});

// ============================================================================
// NegativeExponentNormalizationRule: x^(-n) → 1/x^n
// ============================================================================
define_rule!(
    NegativeExponentNormalizationRule,
    "Normalize Negative Exponent",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
    if let Some((base, exp)) = as_pow(ctx, expr) {
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() && n.is_negative() {
                    let pos_n = -n.clone();
                    let pos_exp = ctx.add(Expr::Number(pos_n));
                    let one = ctx.num(1);
                    let pos_pow = ctx.add(Expr::Pow(base, pos_exp));
                    let result = ctx.add(Expr::Div(one, pos_pow));
                    return Some(Rewrite::new(result).desc("x^(-n) → 1/x^n"));
                }
            }
        }
        None
    }
);

define_rule!(EvaluatePowerRule, "Evaluate Numeric Power", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Some(result) = crate::const_eval::try_eval_pow_literal(ctx, base, exp) {
            return Some(Rewrite::new(result).desc("Evaluate literal power"));
        }

        if let (Expr::Number(b), Expr::Number(e)) = (ctx.get(base), ctx.get(exp)) {
            let (b, e) = (b.clone(), e.clone());
            let numer = e.numer();
            let denom = e.denom();

            if let Some(n) = denom.to_u32() {
                let b_num = b.numer();
                let b_den = b.denom();

                let (out_n, in_n) = super::extract_root_factor(b_num, n);
                let (out_d, in_d) = super::extract_root_factor(b_den, n);

                if !out_n.is_one() || !out_d.is_one() {
                    if let Some(pow_num) = numer.to_i32() {
                        use num_rational::BigRational;
                        let outside_rat = BigRational::new(out_n.clone(), out_d.clone());
                        let outside_val = if pow_num == 1 {
                            outside_rat
                        } else {
                            num_traits::Pow::pow(&outside_rat, pow_num.unsigned_abs())
                        };
                        let outside_val = if pow_num < 0 {
                            outside_val.recip()
                        } else {
                            outside_val
                        };

                        let inside_rat = BigRational::new(in_n, in_d);
                        if inside_rat.is_one() {
                            let new_expr = ctx.add(Expr::Number(outside_val));
                            return Some(Rewrite::new(new_expr).desc_lazy(|| format!("Simplify root: {}^{}", b, e)));
                        } else {
                            let outside_expr = ctx.add(Expr::Number(outside_val));
                            let inside_expr = ctx.add(Expr::Number(inside_rat));
                            let exp_expr = ctx.add(Expr::Number(e.clone()));
                            let root_part = ctx.add(Expr::Pow(inside_expr, exp_expr));
                            let new_expr = ctx.add(Expr::Mul(outside_expr, root_part));
                            return Some(
                                Rewrite::new(new_expr).desc_lazy(|| format!("Simplify root: {}^{}", b, e)),
                            );
                        }
                    }
                }
            }
        }
    }
    None
});
