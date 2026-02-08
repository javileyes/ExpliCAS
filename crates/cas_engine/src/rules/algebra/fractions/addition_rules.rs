//! Fraction addition rules.
//!
//! This module contains rules for adding terms with fractions:
//! - `FoldAddIntoFractionRule`: k + p/q → (k·q + p)/q
//! - `AddFractionsRule`: a/b + c/d → (ad+bc)/bd

use crate::build::mul2_raw;
use crate::define_rule;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::{are_denominators_opposite, collect_variables};
use cas_ast::{count_nodes, BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

// Import helpers from sibling core_rules module
use super::core_rules::{
    check_divisible_denominators, extract_as_fraction, is_pi_constant, is_trig_function,
};

// =============================================================================
// Shared helpers (pub(super) so cancel_rules can use them)
// =============================================================================

/// Build a sum from a list of terms.
pub(super) fn build_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

/// Collect all additive terms from an expression.
/// For `a + b + c`, returns [a, b, c].
pub(super) fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_additive_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_additive_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_additive_terms_recursive(ctx, *l, terms);
            collect_additive_terms_recursive(ctx, *r, terms);
        }
        _ => {
            terms.push(expr);
        }
    }
}

/// Check if an expression contains an irrational (root).
pub(super) fn contains_irrational(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer()
            } else {
                false
            }
        }
        Expr::Function(name, _) => ctx.is_builtin(*name, cas_ast::BuiltinFn::Sqrt),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_irrational(ctx, *l) || contains_irrational(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => contains_irrational(ctx, *e),
        _ => false,
    }
}

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
        let (term, p, q, swapped) = if let Expr::Div(p, q) = ctx.get(r) {
            let (p, q) = (*p, *q);
            // l + p/q
            if matches!(ctx.get(l), Expr::Div(_, _)) {
                return None; // Both fractions: let AddFractionsRule handle it
            }
            (l, p, q, false)
        } else if let Expr::Div(p, q) = ctx.get(l) {
            let (p, q) = (*p, *q);
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
                Expr::Hold(inner) => contains_function_or_root(ctx, *inner),
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
                Expr::Neg(e) | Expr::Hold(e) | Expr::Pow(e, _) => contains_function(ctx, *e),
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
                Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) => true,
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
                Expr::Hold(e) => contains_root(ctx, *e),
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
            return Some(Rewrite::new(new_expr).desc("Add fractions: a/b + c/d -> (ad+bc)/bd"));
        }
        None
    }
);
