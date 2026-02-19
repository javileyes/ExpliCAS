//! Rationalization and cancellation rules for fractions.
//!
//! Addition rules (FoldAddIntoFractionRule, AddFractionsRule) have been
//! extracted to `addition_rules.rs`.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_terms::{
    build_sum, collect_additive_terms_flat_add as collect_additive_terms, contains_irrational,
};
use num_traits::One;

/// Recognizes ±1 in various AST forms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignOne {
    PlusOne,
    MinusOne,
}

/// Check if expr is +1 or -1 (in any AST form)
fn sign_one(ctx: &Context, id: ExprId) -> Option<SignOne> {
    use num_rational::BigRational;
    match ctx.get(id) {
        Expr::Number(n) => {
            if n == &BigRational::from_integer((-1).into()) {
                Some(SignOne::MinusOne)
            } else if n.is_one() {
                Some(SignOne::PlusOne)
            } else {
                None
            }
        }
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) if n.is_one() => Some(SignOne::MinusOne),
            _ => None,
        },
        _ => None,
    }
}

/// Normalize binomial denominator: canonicalize Add(l, Neg(1)) to conceptual Sub(l, 1)
/// Returns (left_term, right_term_normalized, is_add_normalized, right_is_abs_one)
fn split_binomial_den(ctx: &mut Context, den: ExprId) -> Option<(ExprId, ExprId, bool, bool)> {
    let one = ctx.num(1);

    // Use zero-clone helpers
    if let Some((l, r)) = crate::helpers::as_add(ctx, den) {
        return match sign_one(ctx, r) {
            Some(SignOne::PlusOne) => Some((l, one, true, true)), // l + 1
            Some(SignOne::MinusOne) => Some((l, one, false, true)), // l + (-1) → l - 1
            None => Some((l, r, true, false)),                    // l + r
        };
    }
    if let Some((l, r)) = crate::helpers::as_sub(ctx, den) {
        return match sign_one(ctx, r) {
            Some(SignOne::PlusOne) => Some((l, one, false, true)), // l - 1
            Some(SignOne::MinusOne) => Some((l, one, true, true)), // l - (-1) → l + 1
            None => Some((l, r, false, false)),                    // l - r
        };
    }
    None
}

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Use split_binomial_den to normalize the denominator
        // This canonicalizes Add(√x, Neg(1)) to conceptual Sub(√x, 1)
        let (l, r, is_add, r_is_abs_one) = split_binomial_den(ctx, den)?;

        // Check for sqrt roots (degree 2 only - diff squares only works for sqrt)
        // For nth roots (n >= 3), use RationalizeNthRootBinomialRule instead
        let is_sqrt_root = |e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Pow(_, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        // Must be 1/2 for diff squares to work
                        if !n.is_integer() && n.denom() == &num_bigint::BigInt::from(2) {
                            return true;
                        }
                    }
                    false
                }
                Expr::Function(fn_id, _) => ctx.is_builtin(*fn_id, BuiltinFn::Sqrt),
                _ => false,
            }
        };

        let l_sqrt = is_sqrt_root(l);
        let r_sqrt = is_sqrt_root(r);

        // Only apply if at least one term is a sqrt (degree 2)
        // For cube roots and higher, skip - they need geometric sum, not conjugate
        if !l_sqrt && !r_sqrt {
            return None;
        }

        // Construct conjugate using normalized terms
        let conjugate = if is_add {
            ctx.add(Expr::Sub(l, r))
        } else {
            ctx.add(Expr::Add(l, r))
        };

        // Multiply num by conjugate
        let new_num = mul2_raw(ctx, num, conjugate);

        // Compute new den = l^2 - r^2
        // Key fix: if r is ±1, use literal 1 instead of Pow(-1, 2)
        let two = ctx.num(2);
        let one = ctx.num(1);
        let l_sq = ctx.add(Expr::Pow(l, two));
        let r_sq = if r_is_abs_one {
            one // 1² = 1, avoid (-1)²
        } else {
            ctx.add(Expr::Pow(r, two))
        };
        let new_den = ctx.add(Expr::Sub(l_sq, r_sq));

        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite::new(new_expr).desc("Rationalize denominator (diff squares)"));
    }
);

// Rationalize binomial denominators with nth roots (n >= 3) using geometric sum.
// For a^(1/n) - r, multiply by sum_{k=0}^{n-1} a^((n-1-k)/n) * r^k
// This gives denominator a - r^n
define_rule!(
    RationalizeNthRootBinomialRule,
    "Rationalize Nth Root Binomial",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;
        use num_traits::ToPrimitive;

        // Use FractionParts to detect fraction
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Match den = t ± r where t = base^(1/n) with n >= 3
        // NOTE: Add is commutative and canonicalization often places numbers first,
        // so we must detect the nth-root term on either side.

        // Helper to extract nth-root info from an expression
        let extract_nth_root = |e: ExprId| -> Option<(ExprId, u32)> {
            if let Expr::Pow(b, exp) = ctx.get(e) {
                if let Expr::Number(ev) = ctx.get(*exp) {
                    // ev must be 1/n with n >= 3
                    if ev.numer() == &num_bigint::BigInt::from(1) {
                        if let Some(denom) = ev.denom().to_u32() {
                            if denom >= 3 {
                                return Some((*b, denom));
                            }
                        }
                    }
                }
            }
            None
        };

        // Track if we need to flip the sign (for r - t case, handle as -(t - r))
        let mut sign_flip = false;

        let (t, r, base, n, is_sub) = if let Some((l, r_side)) = crate::helpers::as_add(ctx, den) {
            // den = l + r_side
            if let Some((base, n)) = extract_nth_root(l) {
                // t is on left: t + r
                (l, r_side, base, n, false)
            } else if let Some((base, n)) = extract_nth_root(r_side) {
                // t is on right: r + t (same as t + r due to commutativity)
                (r_side, l, base, n, false)
            } else {
                return None;
            }
        } else if let Some((l, r_side)) = crate::helpers::as_sub(ctx, den) {
            // den = l - r_side
            if let Some((base, n)) = extract_nth_root(l) {
                // t is on left: t - r
                (l, r_side, base, n, true)
            } else if let Some((base, n)) = extract_nth_root(r_side) {
                // t is on right: r - t => need sign flip: 1/(r - t) = -1/(t - r)
                sign_flip = true;
                (r_side, l, base, n, true)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Limit n to prevent explosion (max 8 terms)
        if n > 8 {
            return None;
        }

        // Build multiplier M = sum_{k=0}^{n-1} t^(n-1-k) * r^k
        // For t - r: M = t^(n-1) + t^(n-2)*r + ... + r^(n-1)
        // For t + r: need alternating signs for sum formula to work
        //   (t + r)(t^(n-1) - t^(n-2)*r + ... + (-1)^(n-1)*r^(n-1)) = t^n - (-r)^n = t^n - (-1)^n * r^n

        let mut m_terms: Vec<ExprId> = Vec::new();

        for k in 0..n {
            let exp_t = n - 1 - k; // exponent for t
            let exp_r = k; // exponent for r

            // Build t^exp_t = base^((n-1-k)/n)
            let t_part = if exp_t == 0 {
                ctx.num(1)
            } else if exp_t == 1 {
                t
            } else {
                let exp_val = num_rational::BigRational::new(
                    num_bigint::BigInt::from(exp_t),
                    num_bigint::BigInt::from(n),
                );
                let exp_node = ctx.add(Expr::Number(exp_val));
                ctx.add(Expr::Pow(base, exp_node))
            };

            // Build r^exp_r
            let r_part = if exp_r == 0 {
                ctx.num(1)
            } else if exp_r == 1 {
                r
            } else {
                let exp_node = ctx.num(exp_r as i64);
                ctx.add(Expr::Pow(r, exp_node))
            };

            // Combine t_part * r_part
            let mut term = mul2_raw(ctx, t_part, r_part);

            // For t + r case, alternate signs: (-1)^k
            if !is_sub && k % 2 == 1 {
                term = ctx.add(Expr::Neg(term));
            }

            m_terms.push(term);
        }

        // Build M as sum of terms
        let multiplier = build_sum(ctx, &m_terms);

        // New numerator: num * M (negate if we had r - t instead of t - r)
        let mut new_num = mul2_raw(ctx, num, multiplier);
        if sign_flip {
            // 1/(r - t) = -1/(t - r), so negate numerator
            new_num = ctx.add(Expr::Neg(new_num));
        }

        // New denominator: base - r^n (for t - r) or base - (-1)^n * r^n (for t + r)
        let r_to_n = {
            let exp_node = ctx.num(n as i64);
            ctx.add(Expr::Pow(r, exp_node))
        };

        let new_den = if is_sub {
            // (t - r) * M = t^n - r^n = base - r^n
            ctx.add(Expr::Sub(base, r_to_n))
        } else {
            // (t + r) * M = t^n - (-r)^n = base - (-1)^n * r^n
            if n % 2 == 0 {
                // Even n: base - r^n
                ctx.add(Expr::Sub(base, r_to_n))
            } else {
                // Odd n: base + r^n (since (-r)^n = -r^n)
                ctx.add(Expr::Add(base, r_to_n))
            }
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(
            Rewrite::new(new_expr)
                .desc_lazy(|| format!("Rationalize {} root binomial (geometric sum)", ordinal(n))),
        )
    }
);

/// Helper to get ordinal string for small numbers
fn ordinal(n: u32) -> &'static str {
    match n {
        3 => "cube",
        4 => "4th",
        5 => "5th",
        6 => "6th",
        7 => "7th",
        8 => "8th",
        _ => "nth",
    }
}

// Cancel nth root binomial factors: (u ± r^n) / (u^(1/n) ± r) = geometric series
// Example: (x + 1) / (x^(1/3) + 1) = x^(2/3) - x^(1/3) + 1
// Uses identity: a^n - b^n = (a-b)(a^(n-1) + a^(n-2)b + ... + b^(n-1))
//            and: a^n + b^n = (a+b)(a^(n-1) - a^(n-2)b + ... ± b^(n-1)) for odd n
define_rule!(
    CancelNthRootBinomialFactorRule,
    "Cancel Nth Root Binomial Factor",
    None,
    PhaseMask::TRANSFORM | PhaseMask::POST,
    |ctx, expr| {
        use cas_ast::views::FractionParts;
        use num_traits::ToPrimitive;

        // Use FractionParts to detect fraction
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Match den = t ± r where t = u^(1/n)
        // Use zero-clone destructuring
        let (left, right, den_is_add) = if let Some((l, r)) = crate::helpers::as_add(ctx, den) {
            (l, r, true)
        } else if let Some((l, r)) = crate::helpers::as_sub(ctx, den) {
            (l, r, false)
        } else {
            return None;
        };

        // Helper to extract (base, n) from u^(1/n)
        let extract_nth_root = |e: ExprId| -> Option<(ExprId, u32)> {
            if let Expr::Pow(base, exp) = ctx.get(e) {
                if let Expr::Number(ev) = ctx.get(*exp) {
                    if ev.numer() == &num_bigint::BigInt::from(1) {
                        if let Some(denom) = ev.denom().to_u32() {
                            if denom >= 2 {
                                return Some((*base, denom));
                            }
                        }
                    }
                }
            }
            None
        };

        // Try both orderings: left is Pow or right is Pow
        let (t, r, u, n) = if let Some((base, denom)) = extract_nth_root(left) {
            // left = u^(1/n), right = r
            (left, right, base, denom)
        } else if let Some((base, denom)) = extract_nth_root(right) {
            // right = u^(1/n), left = r
            (right, left, base, denom)
        } else {
            return None;
        };

        // r must be a number (start with integer support)
        let r_val = match ctx.get(r) {
            Expr::Number(rv) => rv.clone(),
            _ => return None,
        };

        // Limit n to prevent explosion
        if n > 8 {
            return None;
        }

        // Compute r^n
        let r_to_n = r_val.pow(n as i32);

        // Determine expected numerator based on sign pattern
        // For den = t + r (t = u^(1/n)):
        //   If n is odd: num should be u + r^n (sum of odd powers)
        //   If n is even: num should be u - r^n (?)
        // For den = t - r:
        //   num should be u - r^n (diff of powers)

        let (expected_num_is_add, expected_r_val) = if den_is_add {
            // t + r: for sum pattern a^n + b^n with odd n
            if n % 2 == 1 {
                (true, r_to_n.clone()) // expect u + r^n
            } else {
                return None; // Even n: a^n + b^n doesn't factor nicely over reals
            }
        } else {
            // t - r: for diff pattern a^n - b^n
            (false, r_to_n.clone()) // expect u - r^n
        };

        // Check if numerator matches expected pattern
        // Use zero-clone destructuring
        let (num_left, num_right, num_is_add) =
            if let Some((l, rr)) = crate::helpers::as_add(ctx, num) {
                (l, rr, true)
            } else if let Some((l, rr)) = crate::helpers::as_sub(ctx, num) {
                (l, rr, false)
            } else {
                return None;
            };

        if num_is_add != expected_num_is_add {
            return None;
        }

        // Check if num_left = u (structurally equal)
        // or num_right = u (commutative)
        let (actual_u, actual_r_n) = if crate::ordering::compare_expr(ctx, num_left, u)
            == std::cmp::Ordering::Equal
        {
            (num_left, num_right)
        } else if crate::ordering::compare_expr(ctx, num_right, u) == std::cmp::Ordering::Equal {
            (num_right, num_left)
        } else {
            return None;
        };

        let _ = actual_u; // used for verification above

        // Check if actual_r_n = expected_r_val (as number)
        let actual_r_n_val = match ctx.get(actual_r_n) {
            Expr::Number(v) => v.clone(),
            _ => return None,
        };

        if actual_r_n_val != expected_r_val {
            return None;
        }

        // Match confirmed! Build the quotient as geometric series
        // For t - r: Q = t^(n-1) + t^(n-2)*r + ... + r^(n-1)
        // For t + r: Q = t^(n-1) - t^(n-2)*r + t^(n-3)*r^2 - ... (alternating)

        let mut terms: Vec<ExprId> = Vec::new();

        for k in 0..n {
            let exp_t = n - 1 - k;
            let exp_r = k;

            // Build t^exp_t = u^((n-1-k)/n)
            let t_part = if exp_t == 0 {
                ctx.num(1)
            } else if exp_t == 1 {
                t // u^(1/n)
            } else {
                let exp_val = num_rational::BigRational::new(
                    num_bigint::BigInt::from(exp_t),
                    num_bigint::BigInt::from(n),
                );
                let exp_node = ctx.add(Expr::Number(exp_val));
                ctx.add(Expr::Pow(u, exp_node))
            };

            // Build r^exp_r
            let r_part = if exp_r == 0 {
                ctx.num(1)
            } else {
                let r_pow_k = r_val.pow(exp_r as i32);
                ctx.add(Expr::Number(r_pow_k))
            };

            // Combine t_part * r_part
            let mut term = mul2_raw(ctx, t_part, r_part);

            // For t + r case, alternate signs
            if den_is_add && k % 2 == 1 {
                term = ctx.add(Expr::Neg(term));
            }

            terms.push(term);
        }

        // Build result as sum
        let result = build_sum(ctx, &terms);

        Some(
            Rewrite::new(result)
                .desc_lazy(|| format!("Cancel {} root binomial factor", ordinal(n))),
        )
    }
);

// Collapse sqrt(A) * B → sqrt(B) when A and B are conjugates with A*B = 1
// Example: sqrt(x + sqrt(x²-1)) * (x - sqrt(x²-1)) → sqrt(x - sqrt(x²-1))
// This works because (p + s)(p - s) = p² - s² = 1 when s = sqrt(p² - 1)
//
// IMPORTANT: This transformation requires `other` (the conjugate being lifted into sqrt)
// to be non-negative (≥ 0), which is an ANALYTIC condition. In Generic mode, this rule
// should be blocked with a hint. In Assume mode, it proceeds with "Assumed: other ≥ 0".
define_rule!(
    SqrtConjugateCollapseRule,
    "Collapse Sqrt Conjugate Product",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
        use cas_ast::views::MulChainView;
        use num_rational::BigRational;
        use crate::semantics::ValueDomain;

        // Guard: Only apply in RealOnly domain (in Complex, sqrt has branch cuts)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        // Only match Mul expressions
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Use MulChainView to get all factors
        let mv = MulChainView::from(&*ctx, expr);
        if mv.factors.len() != 2 {
            return None; // Only handle exactly 2 factors for now
        }

        // Helper to check if expr is sqrt(A) and return A
        let unwrap_sqrt = |e: ExprId| -> Option<ExprId> {
            match ctx.get(e) {
                Expr::Pow(base, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        let half = BigRational::new(1.into(), 2.into());
                        if n == &half {
                            return Some(*base);
                        }
                    }
                    None
                }
                Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 => Some(args[0]),
                _ => None,
            }
        };

        // Try both orderings: factor[0]=sqrt, factor[1]=other or vice versa
        let (sqrt_arg, other) = if let Some(a) = unwrap_sqrt(mv.factors[0]) {
            (a, mv.factors[1])
        } else if let Some(a) = unwrap_sqrt(mv.factors[1]) {
            (a, mv.factors[0])
        } else {
            return None;
        };

        // Extract binomial terms from A (sqrt_arg) and B (other)
        // Handle both Add(p, s) and Add(p, Neg(s)) and Sub(p, s)
        struct SignedBinomial {
            p: ExprId,
            s: ExprId,
            s_positive: bool, // true if p + s, false if p - s
        }

        let parse_signed_binomial = |e: ExprId| -> Option<SignedBinomial> {
            match ctx.get(e) {
                Expr::Add(l, r) => {
                    // Check if r is Neg(something)
                    if let Expr::Neg(inner) = ctx.get(*r) {
                        Some(SignedBinomial {
                            p: *l,
                            s: *inner,
                            s_positive: false,
                        })
                    } else {
                        Some(SignedBinomial {
                            p: *l,
                            s: *r,
                            s_positive: true,
                        })
                    }
                }
                Expr::Sub(l, r) => Some(SignedBinomial {
                    p: *l,
                    s: *r,
                    s_positive: false,
                }),
                _ => None,
            }
        };

        let a_bin = parse_signed_binomial(sqrt_arg)?;
        let b_bin = parse_signed_binomial(other)?;

        // Check if they're conjugates: same p and s, opposite sign for s
        let p_matches =
            crate::ordering::compare_expr(ctx, a_bin.p, b_bin.p) == std::cmp::Ordering::Equal;
        let s_matches =
            crate::ordering::compare_expr(ctx, a_bin.s, b_bin.s) == std::cmp::Ordering::Equal;
        let signs_opposite = a_bin.s_positive != b_bin.s_positive;

        if !p_matches || !s_matches || !signs_opposite {
            return None;
        }

        // Additional guard: s must be a sqrt (so p² - s² = p² - t for some t)
        unwrap_sqrt(a_bin.s)?;

        // ================================================================
        // Analytic Gate: sqrt(other) requires other ≥ 0 (NonNegative)
        // This is an Analytic condition, blocked in Generic, allowed in Assume
        // ================================================================
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &crate::domain_facts::Predicate::NonNegative(other),
            "Collapse Sqrt Conjugate Product",
        );

        if !decision.allow {
            // Blocked: Generic/Strict mode with unproven NonNegative condition
            return None;
        }

        // All checks passed! Return sqrt(B) = sqrt(other)
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let result = ctx.add(Expr::Pow(other, half));

        // Build assumption event if we assumed NonNegative
        let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if decision.assumption.is_some() {
            smallvec::smallvec![crate::assumptions::AssumptionEvent::nonnegative(ctx, other)]
        } else {
            smallvec::SmallVec::new()
        };

        Some(Rewrite::new(result).desc("Lift conjugate into sqrt").assume_all(assumption_events))
    }
);

// Helper functions collect_additive_terms, contains_irrational, and build_sum
// are imported from super::helpers

define_rule!(
    GeneralizedRationalizationRule,
    "Generalized Rationalization",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let terms = collect_additive_terms(ctx, den);

        // Only apply to 3+ terms (binary case handled by RationalizeDenominatorRule)
        if terms.len() < 3 {
            return None;
        }

        // Check if any term contains a root
        let has_roots = terms.iter().any(|&t| contains_irrational(ctx, t));
        if !has_roots {
            return None;
        }

        // Strategy: Group as (first n-1 terms) + last_term
        // Then apply conjugate: multiply by (group - last) / (group - last)
        let last_term = terms[terms.len() - 1];
        let group_terms = &terms[..terms.len() - 1];
        let group = build_sum(ctx, group_terms);

        // Conjugate: (group - last_term)
        let conjugate = ctx.add(Expr::Sub(group, last_term));

        // New numerator: num * conjugate
        let new_num = mul2_raw(ctx, num, conjugate);

        // New denominator: group^2 - last_term^2 (difference of squares)
        let two = ctx.num(2);
        let group_sq = ctx.add(Expr::Pow(group, two));
        let last_sq = ctx.add(Expr::Pow(last_term, two));
        let new_den = ctx.add(Expr::Sub(group_sq, last_sq));

        // Post-pass: expand the denominator to simplify (1+√2)² → 3+2√2
        // This ensures rationalization results don't leave unexpanded pow-sums
        let new_den_expanded = crate::expand::expand(ctx, new_den);

        let new_expr = ctx.add(Expr::Div(new_num, new_den_expanded));

        Some(Rewrite::new(new_expr).desc_lazy(|| {
            format!(
                "Rationalize: group {} terms and multiply by conjugate",
                terms.len()
            )
        }))
    }
);
