//! Binomial conjugate rationalization and difference canonicalization rules.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Context, DisplayExpr, Expr, ExprId};
use num_traits::Signed;
use std::cmp::Ordering;

use super::helpers::collect_mul_factors;

// ========== Binomial Conjugate Rationalization (Level 1) ==========
// Transforms: num / (A + B√n) → num * (A - B√n) / (A² - B²·n)
// Only applies when:
// - denominator is a binomial with exactly one numeric surd term
// - A, B are rational, n is a positive integer
// - uses closed-form arithmetic (no calls to general simplifier)

define_rule!(
    RationalizeBinomialSurdRule,
    "Rationalize Binomial Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use crate::rationalize_policy::RationalizeReason;
        use cas_ast::views::{as_rational_const, count_distinct_numeric_surds, is_surd_free};
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions - use zero-clone helper
        let (num, den) = match crate::helpers::as_div(ctx, expr) {
            Some((n, d)) => (n, d),
            None => {
                tracing::trace!(target: "rationalize", "skipped: not a division");
                return None;
            }
        };

        // Budget guard: denominator shouldn't be too complex
        let den_nodes = count_nodes(ctx, den);
        if den_nodes > 30 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::BudgetExceeded, 
                            nodes = den_nodes, max = 30, "auto rationalize rejected");
            return None;
        }

        // Multi-surd guard: only rationalize if denominator has exactly 1 distinct surd
        // Level 1.5 blocks multi-surd expressions (reserved for `rationalize` command)
        let distinct_surds = count_distinct_numeric_surds(ctx, den, 50);
        if distinct_surds == 0 {
            tracing::trace!(target: "rationalize", "skipped: no surds found");
            return None;
        }
        if distinct_surds > 1 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::MultiSurdBlocked,
                            surds = distinct_surds, "auto rationalize rejected");
            return None;
        }

        // Try to parse denominator as A ± B√n (binomial surd)
        // Patterns: Add(A, Mul(B, √n)), Add(A, √n), Sub(A, Mul(B, √n)), etc.

        struct BinomialSurd {
            a: BigRational, // Rational constant term
            b: BigRational, // Coefficient of surd
            n: i64,         // Radicand (square-free positive integer)
            is_sub: bool,   // true if A - B√n, false if A + B√n
        }

        fn parse_binomial_surd(ctx: &Context, den: ExprId) -> Option<BinomialSurd> {
            // Helper to check if expression is a numeric √n
            fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
                match ctx.get(id) {
                    Expr::Pow(base, exp) => {
                        let exp_val = as_rational_const(ctx, *exp, 8)?;
                        let half = BigRational::new(1.into(), 2.into());
                        if exp_val != half {
                            return None;
                        }
                        if let Expr::Number(n) = ctx.get(*base) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 => {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }

            // Helper to parse B*√n or √n (B=1), handling negation
            // Returns (signed_coefficient, radicand)
            fn parse_surd_term(ctx: &Context, id: ExprId) -> Option<(BigRational, i64)> {
                // Handle Neg(surd) → -(surd) with negated coefficient
                if let Expr::Neg(inner) = ctx.get(id) {
                    let (b, n) = parse_surd_term(ctx, *inner)?;
                    return Some((-b, n));
                }

                // Try √n directly (B=1)
                if let Some(n) = is_numeric_sqrt(ctx, id) {
                    return Some((BigRational::from_integer(1.into()), n));
                }

                // Try B * √n (including negative B)
                if let Expr::Mul(l, r) = ctx.get(id) {
                    if let Some(n) = is_numeric_sqrt(ctx, *r) {
                        if let Some(b) = as_rational_const(ctx, *l, 8) {
                            return Some((b, n)); // b is already signed
                        }
                    }
                    if let Some(n) = is_numeric_sqrt(ctx, *l) {
                        if let Some(b) = as_rational_const(ctx, *r, 8) {
                            return Some((b, n)); // b is already signed
                        }
                    }
                }
                None
            }

            match ctx.get(den) {
                // A + surd_term
                Expr::Add(l, r) => {
                    // Try l=A (rational), r=B√n
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    // Try l=B√n, r=A
                    if let Some(a) = as_rational_const(ctx, *r, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *l) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    None
                }
                // A - surd_term (or surd_term - A which is -(A - surd_term) with negated a)
                Expr::Sub(l, r) => {
                    // Try l=A (rational), r=B√n
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: true,
                            });
                        }
                    }
                    // Try l=B√n, r=A (symmetric case: surd - rational)
                    // This represents -A + B√n, so we negate a and use is_sub: false
                    if let Some(a) = as_rational_const(ctx, *r, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *l) {
                            return Some(BinomialSurd {
                                a: -a, // negate since it's B√n - A = -A + B√n
                                b,
                                n,
                                is_sub: false, // After negation, this is -A + B√n
                            });
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        // Helper to extract binomial factor from a Mul chain (Level 1.5)
        // Returns (k_factors, binomial) where k_factors are surd-free factors
        fn extract_binomial_from_product(
            ctx: &Context,
            den: ExprId,
        ) -> Option<(Vec<ExprId>, BinomialSurd)> {
            // First try direct binomial (Level 1)
            if let Some(surd) = parse_binomial_surd(ctx, den) {
                return Some((vec![], surd));
            }

            // Try Mul chain (Level 1.5)
            match ctx.get(den) {
                Expr::Mul(_, _) => {
                    // Flatten the Mul chain preserving order
                    fn collect_factors(ctx: &Context, id: ExprId, factors: &mut Vec<ExprId>) {
                        match ctx.get(id) {
                            Expr::Mul(l, r) => {
                                collect_factors(ctx, *l, factors);
                                collect_factors(ctx, *r, factors);
                            }
                            _ => factors.push(id),
                        }
                    }

                    let mut factors = Vec::new();
                    collect_factors(ctx, den, &mut factors);

                    // Find exactly one binomial factor; others must be surd-free
                    let mut binomial_idx = None;
                    for (i, &factor) in factors.iter().enumerate() {
                        if parse_binomial_surd(ctx, factor).is_some() {
                            if binomial_idx.is_some() {
                                // Multiple binomials → not Level 1.5
                                return None;
                            }
                            binomial_idx = Some(i);
                        } else if !is_surd_free(ctx, factor, 20) {
                            // Factor is neither binomial nor surd-free → skip
                            return None;
                        }
                    }

                    let binomial_idx = binomial_idx?;
                    let binomial = parse_binomial_surd(ctx, factors[binomial_idx])?;

                    // Collect K factors (those not the binomial)
                    let k_factors: Vec<_> = factors
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i != binomial_idx)
                        .map(|(_, f)| f)
                        .collect();

                    Some((k_factors, binomial))
                }
                _ => None,
            }
        }

        let (k_factors, surd) = extract_binomial_from_product(ctx, den)?;

        // Build k_factor product (outside the helper to avoid borrow issues)
        let k_factor: Option<ExprId> = if k_factors.is_empty() {
            None
        } else if k_factors.len() == 1 {
            Some(k_factors[0])
        } else {
            let mut k = k_factors[0];
            for &f in &k_factors[1..] {
                k = ctx.add(Expr::Mul(k, f));
            }
            Some(k)
        };

        // Compute conjugate: if A + B√n, conjugate is A - B√n (and vice versa)
        // New denominator = A² - B²·n (always the same)
        let a_sq = &surd.a * &surd.a;
        let b_sq = &surd.b * &surd.b;
        let b_sq_n = &b_sq * BigRational::from_integer(surd.n.into());
        let new_den_val = &a_sq - &b_sq_n;

        // Check denominator is non-zero
        if new_den_val == BigRational::from_integer(0.into()) {
            return None;
        }

        // Build conjugate expression: A ∓ B√n
        let a_expr = ctx.add(Expr::Number(surd.a.clone()));
        let n_expr = ctx.num(surd.n);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        let b_sqrt_n = if surd.b == BigRational::from_integer(1.into()) {
            sqrt_n
        } else if surd.b == BigRational::from_integer((-1).into()) {
            ctx.add(Expr::Neg(sqrt_n))
        } else {
            let b_expr = ctx.add(Expr::Number(surd.b.clone()));
            mul2_raw(ctx, b_expr, sqrt_n)
        };

        // conjugate = A - B√n if original was A + B√n (is_sub=false)
        // conjugate = A + B√n if original was A - B√n (is_sub=true)
        let conjugate = if surd.is_sub {
            ctx.add(Expr::Add(a_expr, b_sqrt_n))
        } else {
            ctx.add(Expr::Sub(a_expr, b_sqrt_n))
        };

        // Build new numerator: num * conjugate
        // But first, handle negative denominator by absorbing sign into conjugate
        let (final_conjugate, final_den_val) = if new_den_val < BigRational::from_integer(0.into())
        {
            // Negative denominator: negate the entire conjugate
            // This produces -(A + B√n) instead of -A - B√n, which is cleaner for display
            let negated_conjugate = ctx.add(Expr::Neg(conjugate));
            (negated_conjugate, -new_den_val.clone())
        } else {
            (conjugate, new_den_val.clone())
        };

        let new_num = mul2_raw(ctx, num, final_conjugate);

        // Build new denominator as Number (now always positive or handled)
        let new_den = ctx.add(Expr::Number(final_den_val.clone()));

        // If denominator is 1, just return numerator (possibly divided by K)
        let new_expr = if final_den_val == BigRational::from_integer(1.into()) {
            // new_den = K (if present) or 1
            match k_factor {
                Some(k) => ctx.add(Expr::Div(new_num, k)),
                None => new_num,
            }
        } else {
            // new_den = K * (A² - B²n) or just (A² - B²n)
            let rationalized_den = ctx.add(Expr::Number(final_den_val.clone()));
            let full_den = match k_factor {
                Some(k) => mul2_raw(ctx, k, rationalized_den),
                None => rationalized_den,
            };
            ctx.add(Expr::Div(new_num, full_den))
        };

        // Verify we actually made progress (denominator is now rational)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 20 {
            return None;
        }

        Some(Rewrite::new(new_expr).desc(format!(
            "{} / {} -> {} / {}",
            DisplayExpr {
                context: ctx,
                id: num
            },
            DisplayExpr {
                context: ctx,
                id: den
            },
            DisplayExpr {
                context: ctx,
                id: new_num
            },
            DisplayExpr {
                context: ctx,
                id: new_den
            }
        )))
    }
);

// ============================================================================
// R1: Absorb Negation Into Difference Factor
// ============================================================================
// -1/((x-y)*...) → 1/((y-x)*...)
// Absorbs the negative sign by flipping one difference in the denominator.
// Differences can be Sub(x,y) or Add(x, Neg(y)) or Add(x, Mul(-1,y)).

/// Check if expression is a difference (x - y) in any canonical form
/// Returns Some((x, y)) if it's a difference
fn extract_difference(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(l, r) => Some((*l, *r)),
        Expr::Add(l, r) => {
            // Check if right is Neg(y)
            if let Expr::Neg(inner) = ctx.get(*r) {
                return Some((*l, *inner));
            }
            // Check if right is Mul(-1, y) or Mul(y, -1) with negative number
            if let Expr::Mul(a, b) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*a) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *b));
                    }
                }
                if let Expr::Number(n) = ctx.get(*b) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *a));
                    }
                }
            }
            // Check if left is Neg(x)
            if let Expr::Neg(inner) = ctx.get(*l) {
                return Some((*r, *inner));
            }
            None
        }
        _ => None,
    }
}

/// Build a difference expression: always use Sub form now that canonicalization
/// works properly with our fixes
fn build_difference(ctx: &mut Context, x: ExprId, y: ExprId) -> ExprId {
    ctx.add(Expr::Sub(x, y))
}

define_rule!(
    AbsorbNegationIntoDifferenceRule,
    "Absorb Negation Into Difference",
    |ctx, expr| {
        // Check for Neg(Div(...)) or Div with negative numerator
        let (is_neg_wrapped, div_num, div_den) = match ctx.get(expr) {
            Expr::Neg(inner) => {
                if let Expr::Div(n, d) = ctx.get(*inner) {
                    (true, *n, *d)
                } else {
                    return None;
                }
            }
            Expr::Div(n, d) => {
                if let Expr::Number(num_val) = ctx.get(*n) {
                    if num_val.is_negative() {
                        (false, *n, *d)
                    } else {
                        return None;
                    }
                } else if let Expr::Neg(_) = ctx.get(*n) {
                    (false, *n, *d)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Collect all factors from denominator
        let mut factors: Vec<ExprId> = collect_mul_factors(ctx, div_den);

        // Find a difference factor to flip
        let mut flip_index: Option<usize> = None;
        let mut diff_pair: Option<(ExprId, ExprId)> = None;
        for (i, &f) in factors.iter().enumerate() {
            if let Some((x, y)) = extract_difference(ctx, f) {
                flip_index = Some(i);
                diff_pair = Some((x, y));
                break;
            }
        }

        let idx = flip_index?;
        let (x, y) = diff_pair?;

        // Flip the difference: (x - y) → (y - x)
        let flipped = build_difference(ctx, y, x);
        factors[idx] = flipped;

        // Rebuild denominator
        let new_den = factors.iter().copied().fold(None, |acc, f| {
            Some(match acc {
                Some(a) => mul2_raw(ctx, a, f),
                None => f,
            })
        })?;

        // Handle numerator: remove the negation
        let new_num = if is_neg_wrapped {
            div_num
        } else if let Expr::Number(n) = ctx.get(div_num) {
            ctx.add(Expr::Number(-n.clone()))
        } else if let Expr::Neg(inner) = ctx.get(div_num) {
            *inner
        } else {
            return None;
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(new_expr).desc("Absorb negation into difference factor"))
    }
);

// ============================================================================
// R2: Canonicalize Products of Same-Tail Differences
// ============================================================================
// 1/((p-t)*(q-t)) → 1/((t-p)*(t-q))
// When two difference factors share the same "tail" (right operand),
// flip both to have that common element first.
// Double-flip preserves the overall sign.

define_rule!(
    CanonicalDifferenceProductRule,
    "Canonicalize Difference Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // Check if denominator is Mul of exactly two Sub expressions
        let (factor1, factor2) = if let Expr::Mul(l, r) = ctx.get(den) {
            (*l, *r)
        } else {
            return None;
        };

        // Both factors must be Sub
        let (p, t1) = if let Expr::Sub(a, b) = ctx.get(factor1) {
            (*a, *b)
        } else {
            return None;
        };
        let (q, t2) = if let Expr::Sub(a, b) = ctx.get(factor2) {
            (*a, *b)
        } else {
            return None;
        };

        // Check if they share the same tail
        if crate::ordering::compare_expr(ctx, t1, t2) != Ordering::Equal {
            return None;
        }

        let t = t1;

        // Only flip if the current form is NOT canonical
        // Canonical: (t - p) * (t - q) where t comes first in both
        // Current is (p - t) * (q - t) - needs flipping
        // Guard: if t already comes first in both, don't flip (avoid loops)
        let t_already_first_1 = if let Expr::Sub(a, _) = ctx.get(factor1) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };
        let t_already_first_2 = if let Expr::Sub(a, _) = ctx.get(factor2) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };

        if t_already_first_1 && t_already_first_2 {
            return None; // Already canonical
        }

        // Flip both: (p-t) → (t-p), (q-t) → (t-q)
        let new_factor1 = ctx.add(Expr::Sub(t, p));
        let new_factor2 = ctx.add(Expr::Sub(t, q));
        let new_den = mul2_raw(ctx, new_factor1, new_factor2);

        let new_expr = ctx.add(Expr::Div(num, new_den));

        Some(Rewrite::new(new_expr).desc("Canonicalize same-tail difference product"))
    }
);
