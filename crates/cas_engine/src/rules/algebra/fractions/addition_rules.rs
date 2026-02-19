//! Fraction addition rules.
//!
//! This module contains rules for adding terms with fractions:
//! - `FoldAddIntoFractionRule`: k + p/q → (k·q + p)/q
//! - `AddFractionsRule`: a/b + c/d → (ad+bc)/bd

use crate::build::mul2_raw;
use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Context, Expr, ExprId};
use cas_math::expr_predicates::{
    contains_div_term, contains_function, contains_function_or_root, contains_root_term,
    is_constant_expr, is_constant_fraction, is_minus_one_expr, is_one_expr,
    is_simple_number_abs_leq, is_trivial_denom_one,
};
use cas_math::fraction_forms::are_denominators_opposite;
use cas_math::polynomial::Polynomial;
use num_traits::{One, Zero};
use std::cmp::Ordering;

// Import helpers from sibling core_rules module
use super::core_rules::{
    check_divisible_denominators, extract_as_fraction, is_pi_constant, is_trig_function,
};

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
        if term_is_unit && is_constant_expr(ctx, p) {
            return None;
        }

        // V2.15.8: Guard: Skip if term is or contains a Div (let AddFractionsRule handle both fractions)
        // This fixes: 1/(x-1) - 1/(x+1) should use AddFractionsRule, not treat -1/(x+1) as scalar k
        if contains_div_term(ctx, term) {
            return None;
        }

        // Guard: Skip if term contains functions or roots (preserve arctan(x) + 1/y)
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

// =============================================================================
// SubTermMatchesDenomRule: a - b/a → (a² - b)/a
// =============================================================================
//
// When the denominator of a subtracted fraction matches the other term,
// combine them into a single fraction. This pattern always reduces nesting
// and is essential for trig simplification:
//   cos(x) - sin²(x)/cos(x) → (cos²(x) - sin²(x))/cos(x) → cos(2x)/cos(x)
//
// This rule complements FoldAddIntoFractionRule (which handles Add only)
// by specifically targeting the Sub case where the denominator matches.
//
// Guard: Skip inside trig arguments and inside fractions (same as FoldAddIntoFraction).

define_rule!(
    SubTermMatchesDenomRule,
    "Combine Same Denominator Sub",
    |ctx, expr, parent_ctx| {
        // After canonicalization, Sub(a, b) becomes Add(a, Neg(b)).
        // So we match Add(term, Neg(Div(p, q))) where q == term.
        // Also handles Add(Neg(Div(p, q)), term) where q == term.
        let (l, r) = crate::helpers::as_add(ctx, expr)?;

        // Try both orderings: Add(term, Neg(Div(p, q))) and Add(Neg(Div(p, q)), term)
        let (term, p, q) = if let Expr::Neg(inner) = ctx.get(r) {
            // r = Neg(Div(p, q))
            if let Expr::Div(p, q) = ctx.get(*inner) {
                (l, *p, *q)
            } else {
                return None;
            }
        } else if let Expr::Neg(inner) = ctx.get(l) {
            // l = Neg(Div(p, q))
            if let Expr::Div(p, q) = ctx.get(*inner) {
                (r, *p, *q)
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Key check: denominator q must structurally match term
        if crate::ordering::compare_expr(ctx, q, term) != std::cmp::Ordering::Equal {
            return None;
        }

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        if inside_trig {
            return None;
        }

        // Build: (term·term - p) / term  = (term² - p) / term
        let term_squared = mul2_raw(ctx, term, term);
        let new_num = ctx.add(Expr::Sub(term_squared, p));
        let new_expr = ctx.add(Expr::Div(new_num, term));

        Some(Rewrite::new(new_expr).desc("Common denominator: a - b/a → (a² - b)/a"))
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
        // Block: (√a + ...) + p/q where √a has den=1 and q is non-trivial
        let l_has_root_trivial = !is_frac1 && contains_root_term(ctx, l);
        let r_has_root_trivial = !is_frac2 && contains_root_term(ctx, r);
        let l_has_real_frac = is_frac1 && !is_trivial_denom_one(ctx, d1);
        let r_has_real_frac = is_frac2 && !is_trivial_denom_one(ctx, d2);

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
                    if crate::ordering::compare_expr(ctx, d1_exp, d2_exp) == Ordering::Equal {
                        true
                    } else {
                        // Fallback: numeric probe — expand(d1 - d2) and check if zero
                        // at several random rational points. This handles raw expanded forms
                        // like u·u + u·1 vs u² + u that structural comparison misses.
                        let diff = ctx.add(Expr::Sub(d1, d2));
                        let diff_exp = crate::expand::expand(ctx, diff);
                        cas_math::numeric_eval::numeric_poly_zero_check(ctx, diff_exp)
                    }
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

        // a/b + c/d = (ad + bc) / bd
        // Optimize: if n1=1, ad=d2; if n1=-1, ad=-d2; else ad=n1*d2
        let ad = if is_one_expr(ctx, n1) {
            d2
        } else if is_minus_one_expr(ctx, n1) {
            ctx.add(Expr::Neg(d2))
        } else {
            mul2_raw(ctx, n1, d2)
        };

        let bc = if is_one_expr(ctx, n2) {
            d1
        } else if is_minus_one_expr(ctx, n2) {
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

        // Early zero-numerator check: expand the numerator and check if it
        // simplifies to 0. Catches partial-fraction recombination and combined-
        // identity cases where the numerator is zero after expansion.
        // Runs for ALL denominator cases (same, opposite, or different) and
        // BEFORE the complexity gate so it can't be blocked by growth.
        {
            // Two-pass expansion: the first expand distributes outer products but may
            // leave Neg(Sum)·factor undistributed. The second pass finishes distribution.
            let num_pass1 = crate::expand::expand(ctx, new_num);
            let num_pass2 = crate::expand::expand(ctx, num_pass1);
            if cas_math::numeric_eval::numeric_poly_zero_check(ctx, num_pass2) {
                let zero = ctx.num(0);
                // Return 0/den when denominator has variables (preserves domain
                // restrictions for strict definedness). Return plain 0 otherwise.
                let den_vars = cas_ast::collect_variables(ctx, common_den);
                if den_vars.is_empty() {
                    return Some(Rewrite::new(zero).desc("Add fractions: numerator cancels to 0"));
                } else {
                    let zero_frac = ctx.add(Expr::Div(zero, common_den));
                    return Some(
                        Rewrite::new(zero_frac).desc("Add fractions: numerator cancels to 0"),
                    );
                }
            }
        }

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
                let l_is_const = is_constant_expr(ctx, l);
                let r_is_const = is_constant_expr(ctx, r);

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
            let vars = cas_ast::collect_variables(ctx, new_num);
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

        let both_simple_numerators =
            is_simple_number_abs_leq(ctx, n1, 2) && is_simple_number_abs_leq(ctx, n2, 2);
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

// =============================================================================
// SubFractionsRule: a/b - c/d → (a·d - c·b) / (b·d)
// =============================================================================
//
// Combines two fractions being subtracted into a single fraction.
// The resulting numerator goes through normal simplification which can prove
// it equals 0 when the fractions were algebraically equal (e.g., different
// representations of the same rational expression).
//
// This handles cases that SubSelfToZeroRule misses because the two fractions
// have structurally different (but algebraically equivalent) numerators/denominators
// from independent simplification paths.
//
// Example: ((u+1)·(u·x+1)+u)/(u·(u+1)) - (u²x+ux+2u+1)/(u²+u)
//        → (cross_product) / (common_den) → 0/den → 0
//
// Guards:
// - Both sides must be fractions (direct Div or FractionParts)
// - Skip if inside trig arguments (preserve sin(a - pi/9) structure)
// - Skip function-containing expressions mixed with constant fractions
// - Same complexity heuristics as AddFractionsRule

define_rule!(
    SubFractionsRule,
    "Subtract Fractions",
    |ctx, expr, parent_ctx| {
        use cas_ast::views::FractionParts;

        let (l, r) = crate::helpers::as_sub(ctx, expr)?;

        // Extract fraction parts
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
        let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);

        // Fallback to extract_as_fraction
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

        // Both sides must be fractions
        if !is_frac1 || !is_frac2 {
            return None;
        }

        // Guard: Skip if contains functions mixed with constant fractions
        let l_has_func = contains_function(ctx, l);
        let r_has_func = contains_function(ctx, r);
        let l_is_const_frac = is_constant_expr(ctx, n1) && is_constant_expr(ctx, d1);
        let r_is_const_frac = is_constant_expr(ctx, n2) && is_constant_expr(ctx, d2);

        if (l_has_func && r_is_const_frac) || (r_has_func && l_is_const_frac) {
            return None;
        }

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        if inside_trig {
            return None;
        }

        // Guard: Skip if one side contains roots with trivial denominator
        let l_has_root_trivial = !is_frac1 && contains_root_term(ctx, l);
        let r_has_root_trivial = !is_frac2 && contains_root_term(ctx, r);
        let l_has_real_frac = is_frac1 && !is_trivial_denom_one(ctx, d1);
        let r_has_real_frac = is_frac2 && !is_trivial_denom_one(ctx, d2);

        if (l_has_root_trivial && r_has_real_frac) || (r_has_root_trivial && l_has_real_frac) {
            return None;
        }

        // Check for same or opposite denominators
        let (n2, d2, same_denom) = {
            let cmp = crate::ordering::compare_expr(ctx, d1, d2);
            if d1 == d2 || cmp == Ordering::Equal {
                (n2, d2, true)
            } else {
                // Algebraic check: expand and compare
                let worth_expanding = |id: ExprId| {
                    matches!(
                        ctx.get(id),
                        Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _)
                    )
                };
                let algebraically_equal = if worth_expanding(d1) || worth_expanding(d2) {
                    let d1_exp = crate::expand::expand(ctx, d1);
                    let d2_exp = crate::expand::expand(ctx, d2);
                    if crate::ordering::compare_expr(ctx, d1_exp, d2_exp) == Ordering::Equal {
                        true
                    } else {
                        // Fallback: numeric probe
                        let diff = ctx.add(Expr::Sub(d1, d2));
                        let diff_exp = crate::expand::expand(ctx, diff);
                        cas_math::numeric_eval::numeric_poly_zero_check(ctx, diff_exp)
                    }
                } else {
                    false
                };

                if algebraically_equal {
                    (n2, d2, true)
                } else {
                    (n2, d2, false)
                }
            }
        };

        // Check divisible denominators
        let (n1, n2, common_den, divisible_denom) =
            check_divisible_denominators(ctx, n1, n2, d1, d2);
        let same_denom = same_denom || divisible_denom;

        // Build: (n1·d2 - n2·d1) / (d1·d2), or (n1 - n2) / common_den for same denom
        let new_num = if same_denom {
            ctx.add(Expr::Sub(n1, n2))
        } else {
            // n1·d2 - n2·d1
            let ad = if is_one_expr(ctx, n1) {
                d2
            } else {
                mul2_raw(ctx, n1, d2)
            };
            let bc = if is_one_expr(ctx, n2) {
                d1
            } else {
                mul2_raw(ctx, n2, d1)
            };
            ctx.add(Expr::Sub(ad, bc))
        };

        // Early zero-numerator check: expand the numerator and check if it
        // simplifies to 0. Catches partial-fraction recombination and combined-
        // identity cases where the numerator is zero after expansion.
        // Runs for ALL denominator cases (same or different).
        {
            let num_pass1 = crate::expand::expand(ctx, new_num);
            let num_pass2 = crate::expand::expand(ctx, num_pass1);
            if cas_math::numeric_eval::numeric_poly_zero_check(ctx, num_pass2) {
                let zero = ctx.num(0);
                // Return 0/den when denominator has variables (preserves domain
                // restrictions for strict definedness). Return plain 0 otherwise.
                let eff_den = if same_denom {
                    common_den
                } else {
                    mul2_raw(ctx, d1, d2)
                };
                let den_vars = cas_ast::collect_variables(ctx, eff_den);
                if den_vars.is_empty() {
                    return Some(
                        Rewrite::new(zero).desc("Subtract fractions: numerator cancels to 0"),
                    );
                } else {
                    let zero_frac = ctx.add(Expr::Div(zero, eff_den));
                    return Some(
                        Rewrite::new(zero_frac).desc("Subtract fractions: numerator cancels to 0"),
                    );
                }
            }
        }

        let final_den = if same_denom {
            common_den
        } else {
            mul2_raw(ctx, d1, d2)
        };

        let new_expr = ctx.add(Expr::Div(new_num, final_den));

        // For numeric denominators, always combine
        let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));
        if is_numeric(d1) && is_numeric(d2) {
            return Some(Rewrite::new(new_expr).desc("Subtract numeric fractions"));
        }

        // Always combine Sub(Div,Div) — the resulting numerator will simplify
        // (often to 0 when fractions were algebraically equal)
        return Some(Rewrite::new(new_expr).desc("Subtract fractions: a/b - c/d -> (ad-bc)/bd"));
    }
);
