//! Cancellation and rationalization rules for fractions.
//!
//! This module contains rules for adding fractions, rationalizing denominators,
//! and canceling common factors.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::{are_denominators_opposite, collect_variables};
use cas_ast::{count_nodes, Context, Expr, ExprId};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

// Import helpers from sibling core_rules module
use super::core_rules::{
    check_divisible_denominators, extract_as_fraction, fn_name_is, is_pi_constant,
    is_trig_function, is_trig_function_name,
};

// Import from local helpers module
use super::helpers::{build_sum, collect_additive_terms, contains_irrational};

// =============================================================================
// Fold Add Into Fraction: k + p/q → (k·q + p)/q
// =============================================================================
//
// This rule combines a simple term with a fraction into a single fraction.
// Unlike AddFractionsRule, this always fires when k is "simple enough"
// (Number, Variable, or simple polynomial) to produce canonical rational form.
//
// Examples:
// - 1 + (x+1)/(2x+1) → (3x+2)/(2x+1)
// - x + 1/y → (x·y + 1)/y
// - 2 + 3/x → (2x + 3)/x
//
// Guards:
// - Skip if inside trig arguments (preserve sin(a + pi/9) structure)
// - Skip if k contains functions (preserve arctan(x) + 1/y structure)

define_rule!(
    FoldAddIntoFractionRule,
    "Common Denominator",
    |ctx, expr, parent_ctx| {
        // Match Add(l, r) where one is a fraction and the other is not
        let (l, r) = crate::helpers::as_add(ctx, expr)?;

        // Determine which is the fraction
        let (term, p, q, swapped) = if let Expr::Div(p, q) = ctx.get(r).clone() {
            // l + p/q
            if matches!(ctx.get(l), Expr::Div(_, _)) {
                return None; // Both fractions: let AddFractionsRule handle it
            }
            (l, p, q, false)
        } else if let Expr::Div(p, q) = ctx.get(l).clone() {
            // p/q + r
            if matches!(ctx.get(r), Expr::Div(_, _)) {
                return None;
            }
            (r, p, q, true)
        } else {
            return None; // Neither is a fraction
        };

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        if inside_trig {
            return None;
        }

        // Guard: Skip if this expression is inside a fraction (numerator OR denominator)
        // Let SimplifyComplexFraction handle nested cases properly
        // This prevents preemptive simplification of 1 + x/(x+1) when it's in a complex fraction
        let inside_fraction = parent_ctx
            .has_ancestor_matching(ctx, |c, node_id| matches!(c.get(node_id), Expr::Div(_, _)));
        if inside_fraction {
            return None;
        }

        // Guard: Skip if term is just 1 or -1 AND numerator p is also constant
        // These cases interact badly with SimplifyFractionRule and cause cycles
        // e.g., 1 + 1/(x-1) -> x/(x-1) -> some other form -> cycle
        // But allow: 1 + (x+1)/(2x+1) -> (3x+2)/(2x+1) since numerator has x
        let term_is_unit = match ctx.get(term) {
            Expr::Number(n) => {
                n == &num_rational::BigRational::from_integer(1.into())
                    || n == &num_rational::BigRational::from_integer((-1).into())
            }
            Expr::Neg(inner) => {
                matches!(ctx.get(*inner), Expr::Number(n) if n == &num_rational::BigRational::from_integer(1.into()))
            }
            _ => false,
        };

        // Check if numerator p is purely constant (no variables)
        fn is_constant(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Number(_) | Expr::Constant(_) => true,
                Expr::Neg(inner) => is_constant(ctx, *inner),
                Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Add(l, r) | Expr::Sub(l, r) => {
                    is_constant(ctx, *l) && is_constant(ctx, *r)
                }
                Expr::Pow(base, exp) => is_constant(ctx, *base) && is_constant(ctx, *exp),
                _ => false, // Variables, functions, etc. are NOT constant
            }
        }

        if term_is_unit && is_constant(ctx, p) {
            return None;
        }

        // V2.15.8: Guard: Skip if term is or contains a Div (let AddFractionsRule handle both fractions)
        // This fixes: 1/(x-1) - 1/(x+1) should use AddFractionsRule, not treat -1/(x+1) as scalar k
        fn is_or_contains_div(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Div(_, _) => true,
                Expr::Neg(inner) => is_or_contains_div(ctx, *inner),
                Expr::Mul(l, r) => is_or_contains_div(ctx, *l) || is_or_contains_div(ctx, *r),
                _ => false,
            }
        }
        if is_or_contains_div(ctx, term) {
            return None;
        }

        // Guard: Skip if term contains functions or roots (preserve arctan(x) + 1/y)
        fn contains_function_or_root(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Function(_, _) => true,
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                    contains_function_or_root(ctx, *l) || contains_function_or_root(ctx, *r)
                }
                Expr::Neg(e) => contains_function_or_root(ctx, *e),
                // Detect fractional exponents (roots): x^(1/2), x^(1/3), etc.
                Expr::Pow(base, exp) => {
                    // Check if exponent is a fraction with numerator smaller than denominator
                    let is_fractional = match ctx.get(*exp) {
                        Expr::Number(n) => {
                            !n.is_integer()
                                && n.abs() < num_rational::BigRational::from_integer(1.into())
                        }
                        Expr::Div(_, _) => true, // Any explicit division in exponent is a root
                        _ => false,
                    };
                    is_fractional || contains_function_or_root(ctx, *base)
                }
                _ => false,
            }
        }
        if contains_function_or_root(ctx, term) {
            return None;
        }

        // Guard: Skip if denominator contains functions or roots (sqrt, etc.)
        // These interact badly with rationalization rules and cause cycles
        // e.g., 1/(sqrt(x)-1) should NOT be combined with external terms
        if contains_function_or_root(ctx, q) {
            return None;
        }

        // Guard: Skip if denominator is numeric (let AddFractionsRule handle 1 + 1/2)
        if matches!(ctx.get(q), Expr::Number(_)) {
            return None;
        }

        // Build: (term · q + p) / q
        let term_times_q = mul2_raw(ctx, term, q);
        let new_num = ctx.add(Expr::Add(term_times_q, p));
        let new_expr = ctx.add(Expr::Div(new_num, q));

        // Complexity guard: Only apply if we're not making things worse
        // The result should simplify further (combine like terms) to be worthwhile
        let old_nodes = count_nodes(ctx, expr);
        let new_nodes = count_nodes(ctx, new_expr);

        // Allow slight increase in nodes since combine-like-terms will reduce later
        // But prevent explosion (cap at 1.5x the original)
        if new_nodes > old_nodes * 3 / 2 + 2 {
            return None;
        }

        // Description based on order
        let desc = if swapped {
            "Common denominator: p/q + k → (p + k·q)/q"
        } else {
            "Common denominator: k + p/q → (k·q + p)/q"
        };

        Some(Rewrite::new(new_expr).desc(desc))
    }
);

define_rule!(
    AddFractionsRule,
    "Add Fractions",
    |ctx, expr, parent_ctx| {
        use cas_ast::views::FractionParts;

        // Use zero-clone destructuring
        let (l, r) = crate::helpers::as_add(ctx, expr)?;

        // First try FractionParts (handles direct Div and complex multiplicative patterns)
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
        let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);

        // If FractionParts didn't detect fractions, try extract_as_fraction as fallback
        // This handles Mul(1/n, x) pattern that FractionParts misses
        let (n1, d1, is_frac1) = if !is_frac1 {
            extract_as_fraction(ctx, l)
        } else {
            (n1, d1, is_frac1)
        };
        let (n2, d2, is_frac2) = if !is_frac2 {
            extract_as_fraction(ctx, r)
        } else {
            (n2, d2, is_frac2)
        };

        if !is_frac1 && !is_frac2 {
            return None;
        }

        // Structural guard: Skip combining when one side has functions and other is constant/π
        // This lets inverse trig identity rules fire first (arctan(x) + arctan(1/x) = π/2, etc.)
        // Case to block: arctan(1/3) - pi/2  (function expr + constant fraction)
        // Cases to allow: 1 + 1/2, 1/2 + 1/3, x/2 + x/3, etc.

        // Check if expression contains any function call (not purely algebraic)
        fn contains_function(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Function(_, _) => true,
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                    contains_function(ctx, *l) || contains_function(ctx, *r)
                }
                Expr::Neg(e) | Expr::Pow(e, _) => contains_function(ctx, *e),
                _ => false,
            }
        }

        // Check if expression is a constant (no variables, no functions)
        // Covers: Number, pi, e, Neg(const), Mul(const,const), Div(const,const), Pow(const,int)
        fn is_constant(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Number(_) | Expr::Constant(_) => true,
                Expr::Neg(inner) => is_constant(ctx, *inner),
                Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Add(l, r) | Expr::Sub(l, r) => {
                    is_constant(ctx, *l) && is_constant(ctx, *r)
                }
                Expr::Pow(base, exp) => is_constant(ctx, *base) && is_constant(ctx, *exp),
                // Variables, functions, etc. are NOT constant
                _ => false,
            }
        }

        // Check if fraction n/d is purely constant (both n and d are constants)
        fn is_constant_fraction(ctx: &Context, n: ExprId, d: ExprId) -> bool {
            is_constant(ctx, n) && is_constant(ctx, d)
        }

        // Block case: function expr + constant fraction (like arctan(1/3) + pi/2)
        let l_has_func = contains_function(ctx, l);
        let r_has_func = contains_function(ctx, r);
        let l_is_const_frac = is_frac1 && is_constant_fraction(ctx, n1, d1);
        let r_is_const_frac = is_frac2 && is_constant_fraction(ctx, n2, d2);

        // Skip if mixing function-containing with constant-fraction
        if (l_has_func && r_is_const_frac) || (r_has_func && l_is_const_frac) {
            return None;
        }

        // Guard: Skip if one side contains roots (√a = a^(1/2)) with trivial denominator (=1)
        // and the other side has a non-trivial fraction. This preserves structure for cancellation.
        // Example: (√2 + √3) + 1/(u*(u+2)) should NOT become ((√2+√3)*u*(u+2) + 1) / (u*(u+2))
        fn contains_root(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Function(fn_id, _) if fn_name_is(ctx, *fn_id, "sqrt") => true,
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                    contains_root(ctx, *l) || contains_root(ctx, *r)
                }
                Expr::Neg(e) => contains_root(ctx, *e),
                Expr::Pow(_, exp) => {
                    // Fractional exponent = root
                    match ctx.get(*exp) {
                        Expr::Number(n) => {
                            !n.is_integer()
                                && n.abs() < num_rational::BigRational::from_integer(1.into())
                        }
                        Expr::Div(_, _) => true,
                        _ => false,
                    }
                }
                _ => false,
            }
        }

        fn is_trivial_denom(ctx: &Context, d: ExprId) -> bool {
            matches!(ctx.get(d), Expr::Number(n) if n.is_one())
        }

        // Block: (√a + ...) + p/q where √a has den=1 and q is non-trivial
        let l_has_root_trivial = !is_frac1 && contains_root(ctx, l);
        let r_has_root_trivial = !is_frac2 && contains_root(ctx, r);
        let l_has_real_frac = is_frac1 && !is_trivial_denom(ctx, d1);
        let r_has_real_frac = is_frac2 && !is_trivial_denom(ctx, d2);

        if (l_has_root_trivial && r_has_real_frac) || (r_has_root_trivial && l_has_real_frac) {
            return None;
        }

        // Check if d2 = -d1 or d2 == d1 (semantic comparison for cross-tree equality)
        let (n2, d2, opposite_denom, same_denom) = {
            // Use semantic comparison: denominators from different subexpressions may have same value but different ExprIds
            let cmp = crate::ordering::compare_expr(ctx, d1, d2);
            if d1 == d2 || cmp == Ordering::Equal {
                (n2, d2, false, true)
            } else {
                // Algebraic check: expand and compare (catches u*(u+2) == u²+2u)
                // Only do this if denominators look like polynomials (contain Mul/Add/Pow)
                let worth_expanding = |id: ExprId| {
                    matches!(
                        ctx.get(id),
                        Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _)
                    )
                };
                let algebraically_equal = if worth_expanding(d1) || worth_expanding(d2) {
                    let d1_exp = crate::expand::expand(ctx, d1);
                    let d2_exp = crate::expand::expand(ctx, d2);
                    crate::ordering::compare_expr(ctx, d1_exp, d2_exp) == Ordering::Equal
                } else {
                    false
                };

                if algebraically_equal {
                    (n2, d2, false, true) // same denom algebraically
                } else if are_denominators_opposite(ctx, d1, d2) {
                    // Convert d2 -> d1, n2 -> -n2
                    let minus_n2 = ctx.add(Expr::Neg(n2));
                    (minus_n2, d1, true, false)
                } else {
                    (n2, d2, false, false)
                }
            }
        };

        // Check if one denominator divides the other (d2 = k * d1 or d1 = k * d2)
        // This allows combining 1/2 + 1/(2n) = n/(2n) + 1/(2n) = (n+1)/(2n)
        let (n1, n2, common_den, divisible_denom) =
            check_divisible_denominators(ctx, n1, n2, d1, d2);
        let same_denom = same_denom || divisible_denom;

        // Complexity heuristic
        let old_complexity = count_nodes(ctx, expr);

        // V2.15.8: Detect same-sign fractions for growth allowance
        // (same_sign = both positive or both negative; opposite = one +, one -)
        let same_sign = fp_l.sign == fp_r.sign;

        // V2.15.8: Optimized numerator construction - avoid multiplying by 1 or -1
        fn is_one_val(ctx: &Context, id: ExprId) -> bool {
            matches!(ctx.get(id), Expr::Number(n) if n.is_one())
        }
        fn is_minus_one_val(ctx: &Context, id: ExprId) -> bool {
            use num_rational::BigRational;
            matches!(ctx.get(id), Expr::Number(n) if *n == BigRational::from_integer((-1).into()))
        }

        // a/b + c/d = (ad + bc) / bd
        // Optimize: if n1=1, ad=d2; if n1=-1, ad=-d2; else ad=n1*d2
        let ad = if is_one_val(ctx, n1) {
            d2
        } else if is_minus_one_val(ctx, n1) {
            ctx.add(Expr::Neg(d2))
        } else {
            mul2_raw(ctx, n1, d2)
        };

        let bc = if is_one_val(ctx, n2) {
            d1
        } else if is_minus_one_val(ctx, n2) {
            ctx.add(Expr::Neg(d1))
        } else {
            mul2_raw(ctx, n2, d1)
        };

        let new_num = if opposite_denom || same_denom {
            ctx.add(Expr::Add(n1, n2))
        } else {
            ctx.add(Expr::Add(ad, bc))
        };

        let new_den = if opposite_denom || same_denom {
            common_den
        } else {
            mul2_raw(ctx, d1, d2)
        };

        // Try to simplify common den
        let common_den = if same_denom || opposite_denom {
            common_den
        } else {
            new_den
        };

        let new_expr = ctx.add(Expr::Div(new_num, common_den));
        let new_complexity = count_nodes(ctx, new_expr);

        // If complexity explodes, avoid adding fractions unless denominators are related
        // Exception: if denominators are numbers, always combine: 1/2 + 1/3 = 5/6
        let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));

        // Context-aware gating: avoid combining symbol + pi-const inside trig functions
        // This preserves sin(a + pi/9) structure for identity matching
        // But allows combining pi-fractions like sin(pi/9 + pi/6) -> sin(5*pi/18)
        if is_numeric(d1) && is_numeric(d2) {
            // Check if we're inside a trig function argument
            let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });

            if inside_trig {
                // Both constant? Always combine (e.g., pi/9 + pi/6)
                let l_is_const = is_constant(ctx, l);
                let r_is_const = is_constant(ctx, r);

                if !(l_is_const && r_is_const) {
                    // Mixed: check if we have symbolic + pi-constant pattern
                    let l_is_pi = is_pi_constant(ctx, l);
                    let r_is_pi = is_pi_constant(ctx, r);

                    // Block: symbol + pi-const (e.g., a + pi/9) or pi-const + symbol
                    if (l_is_pi && !r_is_const) || (r_is_pi && !l_is_const) {
                        return None; // Preserve structure for trig identity matching
                    }
                }
            }

            return Some(Rewrite::new(new_expr).desc("Add numeric fractions"));
        }

        let simplifies = |ctx: &Context, num: ExprId, den: ExprId| -> (bool, bool) {
            // Heuristics to see if new fraction simplifies
            // e.g. cancellation of factors
            // or algebraic simplification
            // Just checking if we reduced node count isn't enough,
            // because un-added fractions might be smaller locally but harder to work with.
            // But we don't want to create massive expressions.

            // Factor cancellation check would be good.
            // Is there a factor F in num and den?

            // Check if num is 0
            if let Expr::Number(n) = ctx.get(num) {
                if n.is_zero() {
                    return (true, true);
                }
            }

            // Check negation
            let is_negation = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
                if let Expr::Neg(n) = ctx.get(a) {
                    *n == b
                } else if let Expr::Neg(n) = ctx.get(b) {
                    *n == a
                } else if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(a), ctx.get(b)) {
                    n1 == &-n2
                } else {
                    false
                }
            };

            if is_negation(ctx, num, den) {
                return (true, false);
            }

            // Try Polynomial GCD Check
            let vars = collect_variables(ctx, new_num);
            if vars.len() == 1 {
                if let Some(var) = vars.iter().next() {
                    if let Ok(p_num) = Polynomial::from_expr(ctx, new_num, var) {
                        if let Ok(p_den) = Polynomial::from_expr(ctx, common_den, var) {
                            if !p_den.is_zero() {
                                let gcd = p_num.gcd(&p_den);
                                if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                                    // println!(
                                    //     "  -> Simplifies via GCD! deg={} lc={}",
                                    //     gcd.degree(),
                                    //     gcd.leading_coeff()
                                    // );
                                    let is_proper = p_num.degree() < p_den.degree();
                                    return (true, is_proper);
                                }
                            }
                        }
                    }
                }
            }

            (false, false)
        };

        let (does_simplify, is_proper) = simplifies(ctx, new_num, common_den);

        // println!(
        //     "AddFractions check: old={} new={} simplify={} limit={}",
        //     old_complexity,
        //     new_complexity,
        //     does_simplify,
        //     (old_complexity * 3) / 2
        // );

        // Allow complexity growth if we found a simplification (GCD)
        // BUT strict check against improper fractions to prevent loops with polynomial division
        // (DividePolynomialsRule splits improper fractions, AddFractions combines them -> loop)
        //
        // V2.15.8: Also allow small growth for same-sign fractions with unrelated denominators
        // This enables: 1/x + 1/(x+1) -> (2x+1)/(x*(x+1))
        // But keeps strict behavior for opposite signs to preserve telescoping: 1/x - 1/(x+1)
        //
        // V2.15.8b: Also allow for opposite signs when BOTH numerators are simple (±1)
        // This enables: 1/(x-1) - 1/(x+1) -> 2/((x-1)(x+1))
        // The result simplifies well because the numerator cancellation is straightforward
        let growth_ok = new_complexity <= old_complexity * 3 / 2 + 2;

        fn is_simple_num(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Number(n) => n.abs() <= num_rational::BigRational::from_integer(2.into()),
                Expr::Neg(inner) => is_simple_num(ctx, *inner),
                _ => false,
            }
        }
        let both_simple_numerators = is_simple_num(ctx, n1) && is_simple_num(ctx, n2);
        let allow_growth =
            growth_ok && !opposite_denom && !same_denom && (same_sign || both_simple_numerators);

        if opposite_denom
            || same_denom
            || new_complexity <= old_complexity
            || (does_simplify && is_proper && new_complexity < (old_complexity * 2))
            || allow_growth
        {
            // println!("AddFractions APPLIED: old={} new={} simplify={}", old_complexity, new_complexity, does_simplify);
            return Some(Rewrite::new(new_expr).desc("Add fractions: a/b + c/d -> (ad+bc)/bd"));
        }
        None
    }
);

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
                Expr::Function(fn_id, _) => fn_name_is(ctx, *fn_id, "sqrt"),
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

        Some(Rewrite::new(new_expr).desc(format!(
            "Rationalize {} root binomial (geometric sum)",
            ordinal(n)
        )))
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

        Some(Rewrite::new(result).desc(format!("Cancel {} root binomial factor", ordinal(n))))
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
        use crate::domain::{can_apply_analytic_with_hint, Proof};
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
                Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 => Some(args[0]),
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
        let mode = parent_ctx.domain_mode();
        let key = crate::assumptions::AssumptionKey::nonnegative_key(ctx, other);

        // We don't have a proof for this - it's positivity from structure
        // The conjugate product could be positive or negative depending on x
        let proof = Proof::Unknown;

        let decision = can_apply_analytic_with_hint(
            mode,
            proof,
            key,
            other,
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

        Some(Rewrite::new(new_expr).desc(format!(
            "Rationalize: group {} terms and multiply by conjugate",
            terms.len()
        )))
    }
);

/// Collect all multiplicative factors from an expression
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    collect_factors_recursive(ctx, expr, &mut factors);
    factors
}

fn collect_factors_recursive(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_factors_recursive(ctx, *l, factors);
            collect_factors_recursive(ctx, *r, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

/// Extract root from expression: sqrt(n) or n^(1/k)
/// Returns (radicand, index) where expr = radicand^(1/index)
fn extract_root_base(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    // Check for sqrt(n) function - use zero-clone helper
    if let Some(arg) = crate::helpers::as_fn1(ctx, expr, "sqrt") {
        // sqrt(n) = n^(1/2), return (n, 2)
        let two = ctx.num(2);
        return Some((arg, two));
    }

    // Check for Pow(base, exp) - use zero-clone helper
    if let Some((base, exp)) = crate::helpers::as_pow(ctx, expr) {
        // Check if exp is a Number like 1/k
        if let Some(n) = crate::helpers::as_number(ctx, exp) {
            if !n.is_integer() && n.numer().is_one() {
                // n^(1/k) - return (n, k)
                let k_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                    n.denom().clone(),
                )));
                return Some((base, k_expr));
            }
        }
        // Check if exp is Div(1, k)
        if let Some((num_exp, den_exp)) = crate::helpers::as_div(ctx, exp) {
            if let Some(n) = crate::helpers::as_number(ctx, num_exp) {
                if n.is_one() {
                    return Some((base, den_exp));
                }
            }
        }
    }
    None
}

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Handle fractions with product denominators containing roots
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let factors = collect_mul_factors(ctx, den);

        // Find a root factor
        let mut root_factor = None;
        let mut non_root_factors = Vec::new();

        for &factor in &factors {
            if extract_root_base(ctx, factor).is_some() && root_factor.is_none() {
                root_factor = Some(factor);
            } else {
                non_root_factors.push(factor);
            }
        }

        let root = root_factor?;

        // Don't apply if denominator is ONLY a root (handled elsewhere or simpler)
        if non_root_factors.is_empty() {
            // Just sqrt(n) in denominator - still rationalize
            if let Some((radicand, _index)) = extract_root_base(ctx, root) {
                // Check if radicand is a binomial (Add or Sub) - these can cause infinite loops
                // when both numerator and denominator have binomial radicals like sqrt(x+y)/sqrt(x-y)
                let is_binomial_radical =
                    matches!(ctx.get(radicand), Expr::Add(_, _) | Expr::Sub(_, _));
                if is_binomial_radical && contains_irrational(ctx, num) {
                    return None;
                }

                // Don't rationalize if radicand is a simple number - power rules handle these better
                // e.g., sqrt(2) / 2^(1/3) should simplify via power combination to 2^(1/6)
                if matches!(ctx.get(radicand), Expr::Number(_)) {
                    return None;
                }

                // 1/sqrt(n) -> sqrt(n)/n
                let new_num = mul2_raw(ctx, num, root);
                let new_den = radicand;
                let new_expr = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite::new(new_expr).desc("Rationalize: multiply by √n/√n"));
            }
            return None;
        }

        // Don't apply if radicand is a simple number - power rules can handle these better
        // e.g., 2*sqrt(2) / (2*2^(1/3)) should simplify via power combination, not rationalization
        if let Some((radicand, _index)) = extract_root_base(ctx, root) {
            if matches!(ctx.get(radicand), Expr::Number(_)) {
                return None;
            }
        }

        // We have: num / (other_factors * root) where root = radicand^(1/index)
        // To rationalize, we need to multiply by radicand^((index-1)/index) / radicand^((index-1)/index)
        // This gives: root * radicand^((index-1)/index) = radicand^(1/index + (index-1)/index) = radicand^1 = radicand
        //
        // For sqrt (index=2): multiply by radicand^(1/2) to get radicand^(1/2 + 1/2) = radicand
        // For cbrt (index=3): multiply by radicand^(2/3) to get radicand^(1/3 + 2/3) = radicand

        if let Some((radicand, index)) = extract_root_base(ctx, root) {
            // Compute the conjugate exponent: (index - 1) / index
            // For square root (index=2): conjugate = 1/2, so conjugate_power = radicand^(1/2) = sqrt(radicand)
            // For cube root (index=3): conjugate = 2/3, so conjugate_power = radicand^(2/3)

            // Get index as integer if possible
            let index_val = if let Expr::Number(n) = ctx.get(index) {
                if n.is_integer() {
                    Some(n.to_integer())
                } else {
                    None
                }
            } else {
                None
            };

            // Only handle integer indices for now
            let index_int = index_val?;
            if index_int <= num_bigint::BigInt::from(1) {
                return None; // Not a valid root index
            }

            // Build conjugate exponent (index - 1) / index
            let one = num_bigint::BigInt::from(1);
            let conjugate_num = &index_int - &one;
            let conjugate_exp = num_rational::BigRational::new(conjugate_num, index_int);
            let conjugate_exp_id = ctx.add(Expr::Number(conjugate_exp));

            // conjugate_power = radicand^((index-1)/index)
            let conjugate_power = ctx.add(Expr::Pow(radicand, conjugate_exp_id));

            // New numerator: num * conjugate_power
            let new_num = mul2_raw(ctx, num, conjugate_power);

            // Build new denominator: other_factors * radicand (since root * conjugate_power = radicand)
            let mut new_den = radicand;
            for &factor in &non_root_factors {
                new_den = mul2_raw(ctx, new_den, factor);
            }

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr).desc("Rationalize product denominator"));
        }

        None
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;

        // Capture domain mode once at start
        let domain_mode = parent_ctx.domain_mode();

        // CLONE_OK: Multi-branch match on Div/Pow/Mul requires owned Expr
        let expr_data = ctx.get(expr).clone();

        // Helper to collect factors
        fn collect_factors(ctx: &Context, e: ExprId) -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                if let Expr::Mul(l, r) = ctx.get(curr) {
                    stack.push(*r);
                    stack.push(*l);
                } else {
                    factors.push(curr);
                }
            }
            factors
        }

        let (mut num_factors, mut den_factors) = match expr_data {
            Expr::Div(n, d) => (collect_factors(ctx, n), collect_factors(ctx, d)),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        (vec![ctx.num(1)], collect_factors(ctx, b))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(_, _) => {
                let factors = collect_factors(ctx, expr);
                let mut nf = Vec::new();
                let mut df = Vec::new();
                for f in factors {
                    if let Expr::Pow(b, e) = ctx.get(f) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                df.extend(collect_factors(ctx, *b));
                                continue;
                            }
                        }
                    }
                    nf.push(f);
                }
                if df.is_empty() {
                    return None;
                }
                (nf, df)
            }
            _ => return None,
        };
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let mut changed = false;
        let mut assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = Default::default();
        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            // println!("Processing num factor: {:?}", ctx.get(nf));
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];

                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    // DOMAIN GATE: use canonical helper
                    let proof = prove_nonzero(ctx, nf);
                    let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, nf);
                    let decision = crate::domain::can_cancel_factor_with_hint(
                        domain_mode,
                        proof,
                        key,
                        nf,
                        "Cancel Common Factors",
                    );
                    if !decision.allow {
                        continue; // Skip this pair in strict mode
                    }
                    // Record assumption if made
                    if decision.assumption.is_some() {
                        assumption_events.push(
                            crate::assumptions::AssumptionEvent::nonzero(ctx, nf)
                        );
                    }
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }

                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (integer n only to preserve rationalized forms)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = nf_pow {
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        if let Expr::Number(n) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., sqrt(x)/x should NOT become x^(-1/2) as this undoes rationalization
                            if !n.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = n - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x^1 / x = 1, remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                num_factors[i] = new_term;
                                den_factors.remove(j);
                                found = false; // Modified num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 2: nf = base, df = base^m. (integer m only to preserve rationalized forms)
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = df_pow {
                    if crate::ordering::compare_expr(ctx, nf, b) == std::cmp::Ordering::Equal {
                        if let Expr::Number(m) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., x/sqrt(x) with fractional exp handled by QuotientOfPowersRule
                            if !m.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = m - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x / x^1 = 1, remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                den_factors[j] = new_term;
                                found = true; // Remove num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 3: nf = base^n, df = base^m (integer exponents only)
                // Fractional exponents are handled atomically by QuotientOfPowersRule
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal
                        {
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d))
                            {
                                // Skip fractional exponents - QuotientOfPowersRule handles them
                                if !n.is_integer() || !m.is_integer() {
                                    // Continue to next factor, don't process this pair
                                } else if n > m {
                                    let new_exp = n - m;
                                    let new_term = if new_exp.is_one() {
                                        b_n
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_n, exp_node))
                                    };
                                    num_factors[i] = new_term;
                                    den_factors.remove(j);
                                    found = false;
                                    changed = true;
                                    break;
                                } else if m > n {
                                    let new_exp = m - n;
                                    let new_term = if new_exp.is_one() {
                                        b_d
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_d, exp_node))
                                    };
                                    den_factors[j] = new_term;
                                    found = true;
                                    changed = true;
                                    break;
                                } else {
                                    // x^n / x^n (n == m), remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b_n);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b_n);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b_n,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b_n)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true;
                                    changed = true;
                                    break;
                                } // end else for integer exponents
                            }
                        }
                    }
                }
            }
            if found {
                num_factors.remove(i);
            } else {
                i += 1;
            }
        }

        if changed {
            // Reconstruct
            let new_num = if num_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut n = num_factors[0];
                for &f in num_factors.iter().skip(1) {
                    n = mul2_raw(ctx, n, f);
                }
                n
            };
            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut d = den_factors[0];
                for &f in den_factors.iter().skip(1) {
                    d = mul2_raw(ctx, d, f);
                }
                d
            };

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr)
                .desc("Cancel common factors")
                .local(expr, new_expr)
                .assume_all(assumption_events));
        }

        None
    }
);
