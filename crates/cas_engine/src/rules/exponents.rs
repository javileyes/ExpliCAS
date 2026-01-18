use crate::build::mul2_raw;
use crate::define_rule;
use crate::helpers::is_half;
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

/// Helper: Add two exponents, folding if both are constants
/// This prevents ugly exponents like x^(1+2) and produces x^3 instead
fn add_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let sum = n1 + n2;
        ctx.add(Expr::Number(sum))
    } else {
        ctx.add(Expr::Add(e1, e2))
    }
}

/// Helper: Multiply two exponents, folding if both are constants
/// This prevents ugly exponents like x^(2*3) and produces x^6 instead
fn mul_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let prod = n1 * n2;
        ctx.add(Expr::Number(prod))
    } else {
        mul2_raw(ctx, e1, e2)
    }
}

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

    let expr_data = ctx.get(expr).clone();
    if let Expr::Mul(lhs, rhs) = expr_data {
        let lhs_data = ctx.get(lhs).clone();
        let rhs_data = ctx.get(rhs).clone();

        // Case 1: Both are powers with same base: x^a * x^b
        if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (&lhs_data, &rhs_data) {
            if compare_expr(ctx, *base1, *base2) == Ordering::Equal
                && should_combine(ctx, *base1, *exp1, *exp2)
            {
                let sum_exp = add_exp(ctx, *exp1, *exp2);
                let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                return Some(Rewrite::new(new_expr).desc("Combine powers with same base"));
            }
        }
        // Case 2: One is power, one is base: x^a * x -> x^(a+1)
        // Left is power
        if let Expr::Pow(base1, exp1) = &lhs_data {
            if compare_expr(ctx, *base1, rhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, *base1, *exp1, one) {
                    let sum_exp = add_exp(ctx, *exp1, one);
                    let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                    return Some(Rewrite::new(new_expr).desc("Combine power and base"));
                }
            }
        }
        // Right is power
        if let Expr::Pow(base2, exp2) = &rhs_data {
            if compare_expr(ctx, *base2, lhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, *base2, one, *exp2) {
                    let sum_exp = add_exp(ctx, one, *exp2);
                    let new_expr = ctx.add(Expr::Pow(*base2, sum_exp));
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
        // We rely on CanonicalizeMulRule to have sorted terms, so identical bases are adjacent.
        // Check if rhs is a Mul(rl, rr) and lhs == rl
        if let Expr::Mul(rl, rr) = rhs_data {
            // x * (x * y)
            if compare_expr(ctx, lhs, rl) == Ordering::Equal {
                let two = ctx.num(2);
                let x_squared = ctx.add(Expr::Pow(lhs, two));
                let new_expr = mul2_raw(ctx, x_squared, rr);
                return Some(Rewrite::new(new_expr).desc("Combine nested identical terms"));
            }

            // x^a * (x^b * y) -> x^(a+b) * y
            let lhs_pow = if let Expr::Pow(b, e) = &lhs_data {
                Some((*b, *e))
            } else {
                None
            };
            let rhs_pow = if let Expr::Pow(b, e) = ctx.get(rl) {
                Some((*b, *e))
            } else {
                None
            };

            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
                if compare_expr(ctx, base1, base2) == Ordering::Equal {
                    let sum_exp = add_exp(ctx, exp1, exp2);
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let new_expr = mul2_raw(ctx, new_pow, rr);
                    return Some(Rewrite::new(new_expr).desc("Combine nested powers"));
                }
            }

            // x * (x^a * y) -> x^(a+1) * y
            if let Some((base2, exp2)) = rhs_pow {
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
            // Check if lhs is Mul(c, x^a)
            if let Expr::Mul(ll, lr) = lhs_data {
                // Check if ll is number (coefficient)
                if let Expr::Number(_) = ctx.get(ll) {
                    // lr is x^a ?
                    let lr_pow = if let Expr::Pow(b, e) = ctx.get(lr) {
                        Some((*b, *e))
                    } else {
                        None
                    };

                    // Check rhs is x^b
                    let rhs_pow = if let Expr::Pow(b, e) = &rhs_data {
                        Some((*b, *e))
                    } else {
                        None
                    };

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

                    // Check rhs is x (implicit power 1)
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
            if let Expr::Mul(ll, lr) = lhs_data {
                if let Expr::Number(_) = ctx.get(ll) {
                    // lr is x
                    // rhs is x^a
                    let rhs_pow = if let Expr::Pow(b, e) = &rhs_data {
                        Some((*b, *e))
                    } else {
                        None
                    };

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
            if let Some((base2, exp2)) = rhs_pow {
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
// This allows sqrt(x)*sqrt(4*x) to combine into sqrt(4*x^2) since 4*x has numeric factor
define_rule!(
    ProductSameExponentRule,
    "Product Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE, // Exclude POST to prevent loop with SimplifySqrtOddPowerRule
    |ctx, expr| {
        // a^n * b^n -> (a*b)^n
        // Guard: only apply when at least one base has a numeric factor (to avoid infinite loop)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();

            // Case 1: Both are powers with same exponent: a^n * b^n
            if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (&lhs_data, &rhs_data) {
                if compare_expr(ctx, *exp1, *exp2) == Ordering::Equal {
                    // Guard: at least one base must have a numeric factor to avoid infinite loop
                    // with PowerProductRule ((a*b)^n -> a^n * b^n)
                    let base1_is_num = matches!(ctx.get(*base1), Expr::Number(_));
                    let base2_is_num = matches!(ctx.get(*base2), Expr::Number(_));
                    let base1_has_num = base1_is_num || has_numeric_factor(ctx, *base1);
                    let base2_has_num = base2_is_num || has_numeric_factor(ctx, *base2);

                    if !base1_has_num && !base2_has_num {
                        return None; // Skip if both are purely symbolic (would loop with PowerProductRule)
                    }

                    // Same exponent - combine bases
                    let new_base = mul2_raw(ctx, *base1, *base2);
                    let new_expr = ctx.add(Expr::Pow(new_base, *exp1));
                    return Some(Rewrite::new(new_expr).desc("Combine powers with same exponent"));
                }
            }

            // Case 2: Nested: a^n * (b^n * c) -> (a*b)^n * c
            if let Expr::Pow(base1, exp1) = &lhs_data {
                if let Expr::Mul(rl, rr) = &rhs_data {
                    if let Expr::Pow(base2, exp2) = ctx.get(*rl) {
                        if compare_expr(ctx, *exp1, *exp2) == Ordering::Equal {
                            // Guard: at least one base must have a numeric factor
                            let base1_is_num = matches!(ctx.get(*base1), Expr::Number(_));
                            let base2_is_num = matches!(ctx.get(*base2), Expr::Number(_));
                            let base1_has_num = base1_is_num || has_numeric_factor(ctx, *base1);
                            let base2_has_num = base2_is_num || has_numeric_factor(ctx, *base2);

                            if !base1_has_num && !base2_has_num {
                                return None;
                            }

                            let new_base = mul2_raw(ctx, *base1, *base2);
                            let combined_pow = ctx.add(Expr::Pow(new_base, *exp1));
                            let new_expr = mul2_raw(ctx, combined_pow, *rr);
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

// a^n / b^n = (a/b)^n - combines quotients of powers with same exponent
// This rule is the quotient analog of ProductSameExponentRule
// Example: sqrt(27x)/sqrt(3x) = (27x/3x)^(1/2) = 9^(1/2) = 3
// Guard: at least one base must be a number to avoid infinite loop with PowerQuotientRule
define_rule!(
    QuotientSameExponentRule,
    "Quotient Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM, // Before RATIONALIZE to prevent wrong path
    |ctx, expr| {
        // a^n / b^n -> (a/b)^n
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect division structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Both num and den must be powers with same exponent
        let num_data = ctx.get(num).clone();
        let den_data = ctx.get(den).clone();

        if let (Expr::Pow(base_num, exp_num), Expr::Pow(base_den, exp_den)) = (&num_data, &den_data)
        {
            // Check same exponent
            if compare_expr(ctx, *exp_num, *exp_den) == Ordering::Equal {
                // Guard: at least one base must be a number to avoid infinite loop
                // with PowerQuotientRule (a/b)^n -> a^n/b^n
                let base_num_is_num = matches!(ctx.get(*base_num), Expr::Number(_));
                let base_den_is_num = matches!(ctx.get(*base_den), Expr::Number(_));

                // Also allow if base_num or base_den is a product containing a number
                // This handles sqrt(27x)/sqrt(3x) where bases are Mul(27, x) and Mul(3, x)
                let base_num_has_num = base_num_is_num || has_numeric_factor(ctx, *base_num);
                let base_den_has_num = base_den_is_num || has_numeric_factor(ctx, *base_den);

                if !base_num_has_num && !base_den_has_num {
                    return None; // Skip if both are purely symbolic (would loop)
                }

                // Build new base: base_num / base_den
                let new_base = ctx.add(Expr::Div(*base_num, *base_den));
                let new_expr = ctx.add(Expr::Pow(new_base, *exp_num));
                return Some(Rewrite::new(new_expr).desc("a^n / b^n = (a/b)^n"));
            }
        }

        None
    }
);

/// Check if an expression contains a numeric factor at the top level
fn has_numeric_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Mul(l, r) => {
            matches!(ctx.get(*l), Expr::Number(_)) || matches!(ctx.get(*r), Expr::Number(_))
        }
        _ => false,
    }
}

// ============================================================================
// RootPowCancelRule: (x^n)^(1/n) → x (odd n) or |x| (even n)
// ============================================================================
//
// V2.14.45: Sound handling of root-of-power cancellation in RealOnly.
//
// When canceling (x^n)^(1/n):
// - n odd integer → x (raíz impar preserva signo)
// - n even integer → |x| (raíz par requiere valor absoluto)
// - n symbolic → block in Generic (can't determine parity)
//                 in Assume: x with requires x ≥ 0
//
// Priority: HIGHER than PowerPowerRule to catch this pattern first.
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

        // Match Pow(Pow(x, n), 1/n) where outer_exp = 1/n and inner_exp = n
        let expr_data = ctx.get(expr).clone();
        let Expr::Pow(base, outer_exp) = expr_data else {
            return None;
        };

        let base_data = ctx.get(base).clone();
        let Expr::Pow(inner_base, inner_exp) = base_data else {
            return None;
        };

        // Check if outer_exp = 1/n where n matches inner_exp
        // Also handles case where outer_exp = p/q and inner_exp = q (so p/q * q = p)
        let outer_exp_data = ctx.get(outer_exp).clone();
        let inner_exp_data = ctx.get(inner_exp).clone();

        // Get the combined exponent: inner_exp * outer_exp
        // For true root cancellation, we need combined_exp = 1
        let combined_is_one = match (&outer_exp_data, &inner_exp_data) {
            (Expr::Number(o), Expr::Number(i)) => {
                let combined = o * i;
                combined.is_one()
            }
            // Check if outer_exp = Div(1, inner_exp) structurally
            _ => {
                if let Expr::Div(num, denom) = &outer_exp_data {
                    if let Expr::Number(n) = ctx.get(*num) {
                        if n.is_one() {
                            // outer_exp = 1/denom, check if denom == inner_exp
                            crate::ordering::compare_expr(ctx, *denom, inner_exp) == Ordering::Equal
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

        // We have (x^n)^(1/n) pattern. Now check parity of n.
        let vd = parent_ctx.value_domain();

        // Only apply in RealOnly mode (complex has different branch rules)
        if vd == ValueDomain::ComplexEnabled {
            return None;
        }

        // Check if n is a numeric integer and get its parity
        if let Expr::Number(n) = &inner_exp_data {
            if n.is_integer() {
                let n_int = n.to_integer();
                let is_even = n_int.is_even();

                if is_even {
                    // Even root: (x^(2k))^(1/(2k)) = |x|
                    let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![inner_base]));
                    return Some(
                        crate::rule::Rewrite::new(abs_base).desc("(x^n)^(1/n) = |x| for even n"),
                    );
                } else {
                    // Odd root: (x^(2k+1))^(1/(2k+1)) = x
                    return Some(
                        crate::rule::Rewrite::new(inner_base).desc("(x^n)^(1/n) = x for odd n"),
                    );
                }
            }
        }

        // n is symbolic - can't determine parity
        let dm = parent_ctx.domain_mode();

        match dm {
            crate::domain::DomainMode::Strict | crate::domain::DomainMode::Generic => {
                // Block: can't determine if n is even or odd
                // V2.15.9: Register two BlockedHints to guide user to Assume mode
                // For symbolic n, we need x > 0 (not just ≥ 0) because x^n with negative base
                // and symbolic exponent is problematic in reals. Also n ≠ 0 for 1/n.
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
                // Return x with requires x > 0, n ≠ 0
                // x > 0 ensures x^n is well-defined for any real n
                // n ≠ 0 ensures 1/n is defined
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
        // Higher than PowerPowerRule to catch root cancellation first
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
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, outer_exp) = expr_data {
        let base_data = ctx.get(base).clone();
        if let Expr::Pow(inner_base, inner_exp) = base_data {
            // Check for even root safety: (x^2)^(1/2) -> |x|
            // If inner_exp is even integer and outer_exp is fractional with even denominator?
            // Or just check specific case (x^2)^(1/2).

            let is_even_int = |e: ExprId| -> bool {
                if let Expr::Number(n) = ctx.get(e) {
                    n.is_integer() && n.to_integer().is_even()
                } else {
                    false
                }
            };

            if is_even_int(inner_exp) && is_half(ctx, outer_exp) {
                // (x^(2k))^(1/2) -> |x|^k
                // If k=1, |x|.
                // new_exp = inner_exp * outer_exp = 2k * 1/2 = k.
                let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                // We need to wrap base in abs.
                let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![inner_base]));
                let new_expr = ctx.add(Expr::Pow(abs_base, prod_exp));
                return Some(Rewrite::new(new_expr).desc("Power of power with even root: (x^2k)^(1/2) -> |x|^k"));
            }

            // ================================================================
            // V2.1: Domain check for even roots like (x^(1/2))^2 -> x
            // This requires x >= 0 since sqrt(x) is only defined for non-negatives in reals.
            // We use prove_positive as a conservative check (x > 0 implies x >= 0).
            // ================================================================
            let inner_is_even_root = if let Expr::Number(n) = ctx.get(inner_exp) {
                // Check if inner_exp = p/q where q is even (even root)
                // e.g., 1/2 has denominator 2 (sqrt), 1/4 has denominator 4 (4th root)
                let denom = n.denom();
                denom.is_even()
            } else {
                false
            };

            if inner_is_even_root {
                // This transformation requires inner_base >= 0 (NonNegative condition)
                // First try implicit domain: if sqrt(x) exists elsewhere in the tree,
                // we can use ProvenImplicit without assumptions.
                use crate::domain::{can_apply_analytic_with_hint, Proof};
                use crate::helpers::prove_nonnegative;
                use crate::implicit_domain::{witness_survives_in_context, WitnessKind};

                let mode = parent_ctx.domain_mode();
                let vd = parent_ctx.value_domain();
                let base_proof = prove_nonnegative(ctx, inner_base, vd);

                // Check if we can use implicit domain (witness survives in context)
                let proof = if matches!(base_proof, Proof::Unknown) {
                    if let (Some(implicit), Some(root)) = (parent_ctx.implicit_domain(), parent_ctx.root_expr()) {
                        if implicit.contains_nonnegative(inner_base) {
                            // Build the output candidate (the result of this rewrite)
                            let output_candidate = inner_base;
                            // Check if witness survives in the full tree after replacement
                            if witness_survives_in_context(
                                ctx,
                                inner_base,
                                root,
                                expr,  // The node being replaced
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

                // If proven (explicit or implicit), proceed without assumption
                if matches!(proof, Proof::Proven | Proof::ProvenImplicit) {
                    let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                    let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
                    return Some(Rewrite::new(new_expr).desc("Multiply exponents"));
                }

                // Fall back to normal Analytic gate for Unknown case
                let key = crate::assumptions::AssumptionKey::nonnegative_key(ctx, inner_base);
                let decision = can_apply_analytic_with_hint(
                    mode,
                    proof,
                    key,
                    inner_base,
                    "Power of a Power",
                );

                if !decision.allow {
                    // Blocked: Generic/Strict mode with unproven condition
                    return None;
                }

                // Allowed via Assume mode: proceed with assumption
                let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));

                // Build assumption events
                let mut rewrite = Rewrite::new(new_expr).desc("Multiply exponents");
                if decision.assumption.is_some() {
                    rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, inner_base));
                }
                return Some(rewrite);
            }
            // V2.14.45: Check for symbolic root cancellation (x^n)^(1/n) with n symbolic
            // In this case, we cannot determine parity of n, so in Generic/Strict we must block
            // because even roots require abs() but odd roots don't.
            //
            // Detect pattern: outer_exp is Div(p, denom) and denom structurally equals inner_exp
            let is_symbolic_root_cancel = if let Expr::Div(_num, denom) = ctx.get(outer_exp) {
                // Check if denom == inner_exp (structural equality)
                crate::ordering::compare_expr(ctx, *denom, inner_exp) == Ordering::Equal
            } else {
                false
            };

            if is_symbolic_root_cancel {
                let dm = parent_ctx.domain_mode();
                let vd = parent_ctx.value_domain();

                // Only block in RealOnly domain where parity matters
                if vd == crate::semantics::ValueDomain::RealOnly {
                    match dm {
                        crate::domain::DomainMode::Strict | crate::domain::DomainMode::Generic => {
                            // Block: can't determine if n is even or odd
                            // V2.15.9: Register two BlockedHints to guide user to Assume mode
                            // x > 0 required because x^n with negative base and symbolic exp is problematic
                            // n ≠ 0 required for 1/n to be defined
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
                            // Allow with Positive(x) + NonZero(n) assumption
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

            // Default case: no domain restriction needed
            let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
            let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
            return Some(Rewrite::new(new_expr).desc("Multiply exponents"));
        }
    }
    None
});

define_rule!(EvaluatePowerRule, "Evaluate Numeric Power", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        // Delegate literal integer power to canonical const_eval helper
        // This covers: b^n for ℚ base and ℤ exponent, including edge cases (0^0, 0^-n)
        if let Some(result) = crate::const_eval::try_eval_pow_literal(ctx, base, exp) {
            return Some(Rewrite::new(result).desc("Evaluate literal power"));
        }

        let base_data = ctx.get(base).clone();
        let exp_data = ctx.get(exp).clone();

        // Case 2: Fractional Exponent (Roots) - only when not handled by const_eval
        if let (Expr::Number(b), Expr::Number(e)) = (base_data, exp_data) {
            // Fractional exponents: e = num / den (not handled by const_eval which only does ℤ exp)
            let numer = e.numer();
            let denom = e.denom();

            if let Some(n) = denom.to_u32() {
                let b_num = b.numer();
                let b_den = b.denom();

                let (out_n, in_n) = extract_root_factor(b_num, n);
                let (out_d, in_d) = extract_root_factor(b_den, n);

                // If we extracted anything (outside parts are not 1)
                if !out_n.is_one() || !out_d.is_one() {
                    // b^(num/den) = (out_n/out_d)^num * (in_n/in_d)^(num/den)

                    if let Some(pow_num) = numer.to_i32() {
                        let coeff_num = out_n.pow(pow_num as u32);
                        let coeff_den = out_d.pow(pow_num as u32);
                        let coeff = BigRational::new(coeff_num, coeff_den);

                        let new_base_val = BigRational::new(in_n, in_d);

                        let coeff_expr = ctx.add(Expr::Number(coeff));

                        if new_base_val.is_one() {
                            // Perfect root
                            return Some(
                                Rewrite::new(coeff_expr)
                                    .desc(format!("Evaluate perfect root: {}^{}", b, e)),
                            );
                        } else {
                            // Partial root
                            let new_base = ctx.add(Expr::Number(new_base_val));
                            let new_pow = ctx.add(Expr::Pow(new_base, exp));
                            let new_expr = mul2_raw(ctx, coeff_expr, new_pow);
                            return Some(
                                Rewrite::new(new_expr).desc(format!("Simplify root: {}^{}", b, e)),
                            );
                        }
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
    fn test_product_power() {
        let mut ctx = Context::new();
        let rule = ProductPowerRule;

        // x^2 * x^3 -> x^(2+3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));
        let expr = ctx.add(Expr::Mul(x2, x3));

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
            "x^5"
        );

        // x * x -> x^2
        let expr2 = ctx.add(Expr::Mul(x, x));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x^2"
        );
    }

    #[test]
    fn test_power_power() {
        let mut ctx = Context::new();
        let rule = PowerPowerRule;

        // (x^2)^3 -> x^(2*3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let expr = ctx.add(Expr::Pow(x2, three));

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
            "x^6"
        );
    }

    #[test]
    fn test_zero_one_power() {
        let mut ctx = Context::new();
        let rule = IdentityPowerRule;

        // x^0 -> 1
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Pow(x, zero));
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
            "1"
        );

        // x^1 -> x
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Pow(x, one));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x"
        );
    }
}

define_rule!(
    IdentityPowerRule,
    "Identity Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_nonzero;

        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
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
                            // Generic mode: simplify x^0 → 1 with assumption if base is symbolic
                            // This shows the user the domain restriction (x ≠ 0) even in educational mode
                            let needs_assumption = !matches!(ctx.get(base), Expr::Number(_));
                            let mut rewrite = Rewrite::new(ctx.num(1)).desc("x^0 -> 1");
                            if needs_assumption {
                                rewrite = rewrite.assume(crate::assumptions::AssumptionEvent::nonzero(ctx, base));
                            }
                            return Some(rewrite);
                        }
                        DomainMode::Strict => {
                            // Only simplify if base is provably non-zero
                            if proof == Proof::Proven {
                                return Some(Rewrite::new(ctx.num(1)).desc("x^0 -> 1 (x ≠ 0 proven)"));
                            }
                            // Unknown or Disproven: don't simplify in Strict mode
                            return None;
                        }
                        DomainMode::Assume => {
                            // Simplify with assumption warning
                            return Some(Rewrite::new(ctx.num(1))
                                .desc("x^0 -> 1 (assuming x ≠ 0)")
                                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, base)));
                        }
                    }
                }
            }
            // 1^x -> 1 (always safe - 1 raised to any power is 1)
            if let Expr::Number(n) = ctx.get(base) {
                if n.is_one() {
                    return Some(Rewrite::new(ctx.num(1)).desc("1^x -> 1"));
                }
                // 0^x -> 0 REQUIRES x > 0 (because 0^0 is undefined, 0^(-n) is undefined)
                if n.is_zero() {
                    // If exponent is a literal positive number, always safe
                    if let Expr::Number(e) = ctx.get(exp) {
                        if *e > num_rational::BigRational::zero() {
                            return Some(Rewrite::new(ctx.num(0)).desc("0^n -> 0 (n > 0)"));
                        }
                        // 0^0 and 0^(-n) are handled elsewhere as undefined
                        return None;
                    }
                    // Exponent is variable - check domain_mode
                    // NOTE: 0^x → 0 requires x > 0 (0^0 undefined, 0^(-n) undefined)
                    // This is a Type C "domain pruning" rule - dangerous for solvers
                    // because it destroys the x > 0 constraint needed for correct solution sets.
                    // Use the canonical Analytic gate with pedagogical hint for Generic mode.
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
                        // Add assumption events if any
                        for event in decision.assumption_events(ctx, exp) {
                            rewrite = rewrite.assume(event);
                        }
                        return Some(rewrite);
                    }
                    // Blocked: hint already registered by can_apply_analytic_with_hint
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
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE, // Exclude POST to prevent loop with SimplifySqrtOddPowerRule
    |ctx, expr| {
        // Skip if expression is in canonical (elegant) form
        // e.g., ((x+1)*(x-1))^2 should NOT distribute -> stay as is
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // (a * b)^n -> a^n * b^n
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            let base = *base;
            let exp = *exp;
            if let Expr::Mul(a, b) = ctx.get(base) {
                let a = *a;
                let b = *b;

                // GUARD: Don't distribute fractional exponents over symbolic products UNLESS
                // the symbolic parts have powers that are exact multiples of the root index.
                // This prevents cycles: (3V/4π)^(1/3) ↔ 3^(1/3)*V^(1/3)/(4π)^(1/3)
                // But ALLOWS: (8*x^2)^(1/2) → 8^(1/2) * x (since x^2 is a perfect square)
                if let Expr::Number(exp_num) = ctx.get(exp) {
                    let denom = exp_num.denom();
                    if denom > &num_bigint::BigInt::from(1) {
                        // Fractional exponent - check if distribution is safe
                        if !can_distribute_root_safely(ctx, base, denom) {
                            return None;
                        }
                    }
                }

                // GUARD: Prevent ping-pong with ProductSameExponentRule
                // If one factor is a number and exponent is SYMBOLIC (variable),
                // ProductSameExponentRule would recombine them creating a cycle.
                // Allow distribution when:
                // 1. Exponent is a Number (integer, rational, etc) - will evaluate
                // 2. OR both factors are non-numeric (ProductSameExponentRule won't recombine)
                let a_is_num = matches!(ctx.get(a), Expr::Number(_));
                let b_is_num = matches!(ctx.get(b), Expr::Number(_));
                let exp_is_numeric = matches!(ctx.get(exp), Expr::Number(_));

                // If one is a number and exponent is symbolic, this would ping-pong
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
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        // GUARD: Don't distribute fractional exponents over symbolic quotients UNLESS safe
        // This prevents cycles: (3V/(4π))^(1/3) ↔ 3^(1/3)*V^(1/3)/(4^(1/3)*π^(1/3))
        // But ALLOWS: (8*x^2 / 4)^(1/2) → sqrt(8)*|x| / 2
        if let Expr::Number(exp_num) = ctx.get(exp) {
            let denom = exp_num.denom();
            if denom > &num_bigint::BigInt::from(1) {
                // Fractional exponent - check if distribution is safe
                if !can_distribute_root_safely(ctx, base, denom) {
                    return None;
                }
            }
        }

        let base_data = ctx.get(base).clone();
        if let Expr::Div(num, den) = base_data {
            // Distribute exponent
            let new_num = ctx.add(Expr::Pow(num, exp));
            let new_den = ctx.add(Expr::Pow(den, exp));
            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr).desc("Distribute power over quotient"));
        }
    }
    None
});

/// Check if distributing a fractional exponent (1/n) over a product is safe.
/// Returns true if:
/// 1. Base is purely numeric (no variables), OR
/// 2. All variable factors have powers that are exact multiples of n
///    (e.g., x^2 under sqrt is safe because 2 % 2 == 0)
fn can_distribute_root_safely(
    ctx: &Context,
    expr: ExprId,
    root_index: &num_bigint::BigInt,
) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) => {
            // Bare variable x = x^1, only safe if 1 % root_index == 0 (i.e., root_index == 1)
            root_index == &num_bigint::BigInt::from(1)
        }
        Expr::Pow(base, exp) => {
            // Check if base is symbolic and exponent is a multiple of root_index
            if is_purely_numeric(ctx, *base) {
                return true;
            }
            // Base has variables - check if exponent is integer multiple of root_index
            if let Expr::Number(exp_num) = ctx.get(*exp) {
                if exp_num.is_integer() {
                    let exp_int = exp_num.to_integer();
                    // Safe if exp is divisible by root_index (e.g., x^2 under sqrt(2), x^6 under cbrt(3))
                    return (&exp_int % root_index).is_zero();
                }
            }
            false
        }
        Expr::Mul(l, r) => {
            // Product: both factors must be safe
            can_distribute_root_safely(ctx, *l, root_index)
                && can_distribute_root_safely(ctx, *r, root_index)
        }
        Expr::Div(l, r) => {
            // Quotient: both parts must be safe
            can_distribute_root_safely(ctx, *l, root_index)
                && can_distribute_root_safely(ctx, *r, root_index)
        }
        Expr::Neg(inner) => can_distribute_root_safely(ctx, *inner, root_index),
        _ => false, // Functions, matrices, etc. - be conservative
    }
}

/// Check if expression is purely numeric (no variables/constants)
fn is_purely_numeric(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_purely_numeric(ctx, *l) && is_purely_numeric(ctx, *r)
        }
        Expr::Neg(inner) => is_purely_numeric(ctx, *inner),
        Expr::Function(_, args) => args.iter().all(|a| is_purely_numeric(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().all(|e| is_purely_numeric(ctx, *e)),
        Expr::SessionRef(_) => false,
    }
}

// ============================================================================
// ExpQuotientRule: e^a / e^b → e^(a-b)
// ============================================================================
//
// V2.14.45: This rule simplifies quotients of exponentials with base e.
// Safe: exp(x) > 0 for all x ∈ ℝ or ℂ, so no division by zero.
// No requires needed.
// ============================================================================
define_rule!(ExpQuotientRule, "Exp Quotient", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(num, den) = expr_data {
        let num_data = ctx.get(num).clone();
        let den_data = ctx.get(den).clone();

        // Match Pow(e, a) / Pow(e, b)
        if let (Expr::Pow(num_base, num_exp), Expr::Pow(den_base, den_exp)) = (&num_data, &den_data)
        {
            let num_base_is_e = matches!(ctx.get(*num_base), Expr::Constant(cas_ast::Constant::E));
            let den_base_is_e = matches!(ctx.get(*den_base), Expr::Constant(cas_ast::Constant::E));

            if num_base_is_e && den_base_is_e {
                // e^a / e^b → e^(a - b)
                let diff = ctx.add(Expr::Sub(*num_exp, *den_exp));
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                let new_expr = ctx.add(Expr::Pow(e, diff));
                return Some(Rewrite::new(new_expr).desc("e^a / e^b = e^(a-b)"));
            }
        }

        // Match Div where num is just Constant::E (which is e^1)
        // e / e^b → e^(1-b)
        if matches!(num_data, Expr::Constant(cas_ast::Constant::E)) {
            if let Expr::Pow(den_base, den_exp) = &den_data {
                if matches!(ctx.get(*den_base), Expr::Constant(cas_ast::Constant::E)) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(one, *den_exp));
                    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                    let new_expr = ctx.add(Expr::Pow(e, diff));
                    return Some(Rewrite::new(new_expr).desc("e / e^b = e^(1-b)"));
                }
            }
        }

        // Match e^a / e (where den is just Constant::E, which is e^1)
        // e^a / e → e^(a-1)
        if matches!(den_data, Expr::Constant(cas_ast::Constant::E)) {
            if let Expr::Pow(num_base, num_exp) = &num_data {
                if matches!(ctx.get(*num_base), Expr::Constant(cas_ast::Constant::E)) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(*num_exp, one));
                    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                    let new_expr = ctx.add(Expr::Pow(e, diff));
                    return Some(Rewrite::new(new_expr).desc("e^a / e = e^(a-1)"));
                }
            }
        }
    }
    None
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(ProductSameExponentRule));
    simplifier.add_rule(Box::new(QuotientSameExponentRule)); // a^n / b^n = (a/b)^n
                                                             // V2.14.45: RootPowCancelRule BEFORE PowerPowerRule for (x^n)^(1/n) with parity
    simplifier.add_rule(Box::new(RootPowCancelRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(ExpQuotientRule)); // V2.14.45: e^a/e^b → e^(a-b)

    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(NegativeBasePowerRule));
    simplifier.add_rule(Box::new(EvenPowSubSwapRule)); // (b-a)^even → (a-b)^even
}

define_rule!(NegativeBasePowerRule, "Negative Base Power", |ctx, expr| {
    // (-x)^n
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_data = ctx.get(base).clone();
        if let Expr::Neg(inner) = base_data {
            // Check exponent parity
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() {
                    if n.to_integer().is_even() {
                        // (-x)^even -> x^even
                        let new_expr = ctx.add(Expr::Pow(inner, exp));
                        return Some(Rewrite::new(new_expr).desc("(-x)^even -> x^even"));
                    } else {
                        // (-x)^odd -> -(x^odd)
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
// This allows (x-y)^2 - (y-x)^2 to simplify to 0
// IMPORTANT: Does NOT introduce Neg - just swaps the subtraction order
define_rule!(
    EvenPowSubSwapRule,
    "Canonicalize Even Power Base",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Match Pow(base, exp) where exp is even integer
        let (base, exp) = match ctx.get(expr) {
            Expr::Pow(b, e) => (*b, *e),
            _ => return None,
        };

        // Check if exponent is even integer
        let is_even = match ctx.get(exp) {
            Expr::Number(n) => n.is_integer() && n.to_integer().is_even(),
            _ => false,
        };
        if !is_even {
            return None;
        }

        // Parse base as a binomial subtraction (a - b) in various forms
        // Handle: Sub(a, b), Add(a, Neg(b)), Add(Neg(b), a)
        let (pos_term, neg_term) = match ctx.get(base) {
            Expr::Sub(a, b) => (*a, *b),
            Expr::Add(l, r) => {
                // Check Add(a, Neg(b))
                if let Expr::Neg(inner) = ctx.get(*r) {
                    (*l, *inner)
                } else if let Expr::Neg(inner) = ctx.get(*l) {
                    // Check Add(Neg(b), a) - means a - b
                    (*r, *inner)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Base represents pos_term - neg_term
        // Canonical form: smaller term first
        // If neg_term < pos_term, swap to (neg_term - pos_term)
        if compare_expr(ctx, neg_term, pos_term) != Ordering::Less {
            return None; // Already canonical or equal
        }

        // Swap: build (neg_term - pos_term)^exp
        // Use Sub for cleaner structure
        let new_base = ctx.add(Expr::Sub(neg_term, pos_term));
        let new_expr = ctx.add(Expr::Pow(new_base, exp));

        Some(Rewrite::new(new_expr)
            .desc("For even exponent: (a-b)² = (b-a)², normalize for cancellation")
            .local(base, new_base))
    }
);

fn extract_root_factor(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    if n.is_zero() {
        return (BigInt::zero(), BigInt::one());
    }
    if n.is_one() {
        return (BigInt::one(), BigInt::one());
    }

    let sign = if n.is_negative() { -1 } else { 1 };
    let mut n_abs = n.abs();

    let mut outside = BigInt::one();
    let mut inside = BigInt::one();

    // Trial division
    // Check 2
    let mut count = 0;
    while n_abs.is_even() {
        count += 1;
        n_abs /= 2;
    }
    if count > 0 {
        let out_exp = count / k;
        let in_exp = count % k;
        if out_exp > 0 {
            outside *= BigInt::from(2).pow(out_exp);
        }
        if in_exp > 0 {
            inside *= BigInt::from(2).pow(in_exp);
        }
    }

    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        if (&n_abs % &d).is_zero() {
            let mut count = 0;
            while (&n_abs % &d).is_zero() {
                count += 1;
                n_abs /= &d;
            }
            let out_exp = count / k;
            let in_exp = count % k;
            if out_exp > 0 {
                outside *= d.pow(out_exp);
            }
            if in_exp > 0 {
                inside *= d.pow(in_exp);
            }
        }
        d += 2;
    }

    if n_abs > BigInt::one() {
        inside *= n_abs;
    }

    // Handle sign
    if sign == -1 {
        if !k.is_multiple_of(2) {
            outside = -outside;
        } else {
            inside = -inside;
        }
    }

    (outside, inside)
}
