//! Polynomial manipulation rules: distribution, annihilation, combining like terms,
//! expansion, and factoring.
//!
//! This module is split into submodules:
//! - `expansion`: Binomial/multinomial expansion, auto-expand, polynomial identity detection
//! - `factoring`: Heuristic common factor extraction

mod expansion;
mod expansion_normalize;
mod factoring;
pub(crate) mod polynomial_helpers;

pub use expansion::{AutoExpandPowSumRule, AutoExpandSubCancelRule, BinomialExpansionRule};
pub use expansion_normalize::{
    ExpandSmallBinomialPowRule, HeuristicPolyNormalizeAddRule, PolynomialIdentityZeroRule,
};
pub use factoring::{ExtractCommonMulFactorRule, HeuristicExtractCommonFactorAddRule};

// Re-export helpers used within this module
use polynomial_helpers::{
    count_additive_terms, flatten_additive_terms, is_conjugate, poly_equal, select_best_focus,
    unwrap_hold,
};

use crate::define_rule;
use crate::nary::{build_balanced_add, AddView, Sign};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Check if an expression is a binomial (sum or difference of exactly 2 terms)
/// Examples: (a + b), (a - b), (x + (-y))
fn is_binomial(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Add(_, _) | Expr::Sub(_, _))
}

/// PERFORMANCE: Check if distributing `factor` across `additive` would be
/// computationally expensive and should be skipped.
///
/// Returns true (skip distribution) when ALL of:
///   - The additive side contains variables (pure-constant sums always OK)
///   - The factor matches one of these expensive patterns:
///
/// | Pattern | Example | Why expensive |
/// |---------|---------|---------------|
/// | Variable-free complex constant | `(√6+√2)/4` (≥5 nodes) | Nested radical × polynomial |
/// | Fractional exponents | `(1-x^(1/3)+x^(2/3))/(1+x)` | Cube-root rationalization residual |
/// | Multi-variable high-node fraction | `(-b+√(b²-4ac))/(2a)` | Quadratic formula × polynomial |
///
/// Harmless factors are always allowed through:
///   - Simple numbers: `3`, `-1/2`
///   - Simple surds: `√2`, `√3/2` (< 5 nodes)
///   - Single variables: `x`
fn is_expensive_factor(ctx: &Context, factor: ExprId, additive: ExprId) -> bool {
    // Pure-constant additive sums always distribute (e.g. x*(√3-2) → √3·x - 2·x)
    let additive_vars = cas_ast::collect_variables(ctx, additive);
    if additive_vars.is_empty() {
        return false;
    }

    let factor_nodes = cas_ast::count_nodes(ctx, factor);
    let factor_vars = cas_ast::collect_variables(ctx, factor);

    // Case 1: Variable-free complex constant (≥5 nodes)
    // e.g. (√6+√2)/4, √(10+2√5)/4
    if factor_vars.is_empty() && factor_nodes >= 5 {
        return true;
    }

    // Case 2: Expression with fractional exponents (≥5 nodes)
    // e.g. (1-x^(1/3)+x^(2/3))/(1+x) from cube-root rationalization
    if factor_nodes >= 5 && has_fractional_exponents(ctx, factor) {
        return true;
    }

    // Case 3: Multi-variable fraction (≥3 vars, ≥10 nodes)
    // e.g. (-b+√(b²-4ac))/(2a) — distributing creates 5+ copies of this monster
    if factor_vars.len() >= 3 && factor_nodes >= 10 {
        return true;
    }

    false
}

/// Check if an expression tree contains any fractional exponents.
/// e.g. x^(1/3), x^(2/3), x^(1/2) — but NOT x^2 or x^(-1).
fn has_fractional_exponents(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Pow(base, exp) => {
                // Check if exponent is a non-integer rational
                if let Expr::Number(n) = ctx.get(*exp) {
                    if !n.is_integer() {
                        return true;
                    }
                }
                // Also check if exponent is Div(a,b) form (e.g. 1/3 as AST)
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
            _ => {} // Leaves: Number, Variable, Constant
        }
    }
    false
}

// DistributeRule: Runs in CORE, TRANSFORM, RATIONALIZE but NOT in POST
// This prevents Factor↔Distribute infinite loops (FactorCommonIntegerFromAdd runs in POST)
define_rule!(
    DistributeRule,
    "Distributive Property",
    None,
    // NO POST: evita ciclo con FactorCommonIntegerFromAdd (ver test_factor_distribute_no_loop)
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr, parent_ctx| {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't distribute when goal is Collected or Factored
        // This prevents undoing the effect of collect() or factor() commands
        match parent_ctx.goal() {
            NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
            _ => {}
        }

        // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
        // This protects patterns like abs((x-2)(x+2)) from expanding
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Block distribution when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin(t)*cos(t)*(cos²-sin²) as a single product
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }
        // Use zero-clone destructuring pattern
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        // GUARD: Skip distribution when a factor is 1.
        // 1*(a+b) -> 1*a + 1*b is a visual no-op (MulOne is applied in rendering),
        // and produces confusing "Before/After identical" steps.
        if crate::helpers::is_one(ctx, l) || crate::helpers::is_one(ctx, r) {
            return None;
        }

        // a * (b + c) -> a*b + a*c
        if let Some((b, c)) = crate::helpers::as_add(ctx, r) {
            // PERFORMANCE: Don't distribute expensive factors (complex irrationals,
            // fractional exponents, multi-variable fractions) across polynomials.
            if is_expensive_factor(ctx, l, r) {
                return None;
            }

            // Distribute if 'l' is a Number, Function, Add/Sub, Pow, Mul, or Div.
            // We exclude Var to keep x(x+1) factored, but allow x^2(x+1) to expand.
            // Exception: always allow if the additive side is variable-free (pure constants/surds)
            // so that x*(√3-2) -> √3·x - 2·x for like-term collection.
            let l_expr = ctx.get(l);
            let additive_is_constant = cas_ast::collect_variables(ctx, r).is_empty();
            let should_distribute = additive_is_constant
                || matches!(l_expr, Expr::Number(_))
                || matches!(l_expr, Expr::Function(_, _))
                || matches!(l_expr, Expr::Add(_, _))
                || matches!(l_expr, Expr::Sub(_, _))
                || matches!(l_expr, Expr::Pow(_, _))
                || matches!(l_expr, Expr::Mul(_, _))
                || matches!(l_expr, Expr::Div(_, _))
                || (matches!(l_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            // If we have (A+B)(A-B), do NOT distribute.
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection (secondary defense).
            // Primary defense is the pre-order conjugate pair contraction in
            // transform_binary. This guards against cases where the parent
            // references are still in the same form.
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, r, *pl) || is_conjugate(ctx, r, *pr) {
                        return None;
                    }
                }
            }

            // CRITICAL: Don't expand binomial*binomial products like (a-b)*(a-c)
            // This preserves factored form for opposite denominator detection
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like 1/2*(√2-1) instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(l) {
                if !n.is_integer() && is_binomial(ctx, r) {
                    return None;
                }
            }

            let ab = smart_mul(ctx, l, b);
            let ac = smart_mul(ctx, l, c);
            let new_expr = ctx.add(Expr::Add(ab, ac));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // a * (b - c) -> a*b - a*c
        if let Some((b, c)) = crate::helpers::as_sub(ctx, r) {
            // PERFORMANCE: Same expensive-factor guard as Add branch
            if is_expensive_factor(ctx, l, r) {
                return None;
            }

            let l_expr = ctx.get(l);
            let additive_is_constant = cas_ast::collect_variables(ctx, r).is_empty();
            let should_distribute = additive_is_constant
                || matches!(l_expr, Expr::Number(_))
                || matches!(l_expr, Expr::Function(_, _))
                || matches!(l_expr, Expr::Add(_, _))
                || matches!(l_expr, Expr::Sub(_, _))
                || matches!(l_expr, Expr::Pow(_, _))
                || matches!(l_expr, Expr::Mul(_, _))
                || matches!(l_expr, Expr::Div(_, _))
                || (matches!(l_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, r, *pl) || is_conjugate(ctx, r, *pr) {
                        return None;
                    }
                }
            }

            // Don't expand binomial*binomial products
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            if let Expr::Number(n) = ctx.get(l) {
                if !n.is_integer() && is_binomial(ctx, r) {
                    return None;
                }
            }

            let ab = smart_mul(ctx, l, b);
            let ac = smart_mul(ctx, l, c);
            let new_expr = ctx.add(Expr::Sub(ab, ac));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // (b + c) * a -> b*a + c*a
        if let Some((b, c)) = crate::helpers::as_add(ctx, l) {
            // PERFORMANCE: Same expensive-factor guard (mirror of a*(b+c))
            if is_expensive_factor(ctx, r, l) {
                return None;
            }

            // Same logic for 'r', with variable-free bypass for constant sums
            let r_expr = ctx.get(r);
            let additive_is_constant = cas_ast::collect_variables(ctx, l).is_empty();
            let should_distribute = additive_is_constant
                || matches!(r_expr, Expr::Number(_))
                || matches!(r_expr, Expr::Function(_, _))
                || matches!(r_expr, Expr::Add(_, _))
                || matches!(r_expr, Expr::Sub(_, _))
                || matches!(r_expr, Expr::Pow(_, _))
                || matches!(r_expr, Expr::Mul(_, _))
                || matches!(r_expr, Expr::Div(_, _))
                || (matches!(r_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection (mirror of RHS case above)
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, l, *pl) || is_conjugate(ctx, l, *pr) {
                        return None;
                    }
                }
            }

            // CRITICAL: Don't expand binomial*binomial products (Policy A+)
            // This preserves factored form like (a+b)*(c+d)
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like (√2-1)/2 instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(r) {
                if !n.is_integer() && is_binomial(ctx, l) {
                    return None;
                }
            }

            let ba = smart_mul(ctx, b, r);
            let ca = smart_mul(ctx, c, r);
            let new_expr = ctx.add(Expr::Add(ba, ca));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // (b - c) * a -> b*a - c*a
        if let Some((b, c)) = crate::helpers::as_sub(ctx, l) {
            // PERFORMANCE: Same expensive-factor guard (mirror of a*(b-c))
            if is_expensive_factor(ctx, r, l) {
                return None;
            }

            let r_expr = ctx.get(r);
            let additive_is_constant = cas_ast::collect_variables(ctx, l).is_empty();
            let should_distribute = additive_is_constant
                || matches!(r_expr, Expr::Number(_))
                || matches!(r_expr, Expr::Function(_, _))
                || matches!(r_expr, Expr::Add(_, _))
                || matches!(r_expr, Expr::Sub(_, _))
                || matches!(r_expr, Expr::Pow(_, _))
                || matches!(r_expr, Expr::Mul(_, _))
                || matches!(r_expr, Expr::Div(_, _))
                || (matches!(r_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, l, *pl) || is_conjugate(ctx, l, *pr) {
                        return None;
                    }
                }
            }

            // Don't expand binomial*binomial products
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            if let Expr::Number(n) = ctx.get(r) {
                if !n.is_integer() && is_binomial(ctx, l) {
                    return None;
                }
            }

            let ba = smart_mul(ctx, b, r);
            let ca = smart_mul(ctx, c, r);
            let new_expr = ctx.add(Expr::Sub(ba, ca));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        // Using AddView for shape-independent n-ary handling
        if let Some((numer, denom)) = crate::helpers::as_div(ctx, expr) {
            // Helper to check if division simplifies (shares factors) and return factor size
            let get_simplification_reduction = |ctx: &Context, num: ExprId, den: ExprId| -> usize {
                if num == den {
                    return cas_ast::count_nodes(ctx, num);
                }

                // Structural factor check
                let get_factors = |e: ExprId| -> Vec<ExprId> {
                    let mut factors = Vec::new();
                    let mut stack = vec![e];
                    while let Some(curr) = stack.pop() {
                        if let Expr::Mul(a, b) = ctx.get(curr) {
                            stack.push(*a);
                            stack.push(*b);
                        } else {
                            factors.push(curr);
                        }
                    }
                    factors
                };

                let num_factors = get_factors(num);
                let den_factors = get_factors(den);

                for df in den_factors {
                    // Check for structural equality using compare_expr
                    let found = num_factors
                        .iter()
                        .any(|nf| compare_expr(ctx, *nf, df) == Ordering::Equal);

                    if found {
                        let factor_size = cas_ast::count_nodes(ctx, df);
                        // Factor removed from num and den -> 2 * size
                        let mut reduction = factor_size * 2;
                        // If factor is entire denominator, Div is removed -> +1
                        if df == den {
                            reduction += 1;
                        }
                        return reduction;
                    }

                    // Check for numeric GCD
                    if let Expr::Number(n_den) = ctx.get(df) {
                        let found_numeric = num_factors.iter().any(|nf| {
                            if let Expr::Number(n_num) = ctx.get(*nf) {
                                if n_num.is_integer() && n_den.is_integer() {
                                    let num_int = n_num.to_integer();
                                    let den_int = n_den.to_integer();
                                    if !num_int.is_zero() && !den_int.is_zero() {
                                        let gcd = num_int.gcd(&den_int);
                                        return gcd > One::one();
                                    }
                                }
                            }
                            false
                        });
                        if found_numeric {
                            return 1; // Conservative estimate for number simplification
                        }
                    }
                }

                // Fallback to Polynomial GCD
                let vars = cas_ast::collect_variables(ctx, num);
                if vars.is_empty() {
                    return 0;
                }

                for var in vars {
                    if let (Ok(p_num), Ok(p_den)) = (
                        Polynomial::from_expr(ctx, num, &var),
                        Polynomial::from_expr(ctx, den, &var),
                    ) {
                        if p_den.is_zero() {
                            continue;
                        }
                        let gcd = p_num.gcd(&p_den);
                        // println!("DistributeRule Poly GCD check: num={:?} den={:?} var={} gcd={:?}", ctx.get(num), ctx.get(den), var, gcd);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            // Estimate complexity of GCD
                            // If GCD cancels denominator (degree match), reduction is high
                            if gcd.degree() == p_den.degree() {
                                // Assume denominator is removed (size(den) + 1)
                                return cas_ast::count_nodes(ctx, den) + 1;
                            }
                            // Otherwise, just return 1
                            return 1;
                        }
                    }
                }
                0
            };

            // N-ARY: Use AddView for shape-independent handling of sums
            // This correctly handles ((a+b)+c), (a+(b+c)), and balanced trees
            let num_view = AddView::from_expr(ctx, numer);

            // Check if it's actually a sum (more than 1 term)
            if num_view.terms.len() > 1 {
                // Calculate total reduction potential
                let mut total_reduction: usize = 0;
                let mut any_simplifies = false;

                for &(term, _sign) in &num_view.terms {
                    let red = get_simplification_reduction(ctx, term, denom);
                    if red > 0 {
                        any_simplifies = true;
                        total_reduction += red;
                    }
                }

                // Only distribute if at least one term simplifies
                if any_simplifies {
                    // Build new terms: each term divided by denominator
                    let new_terms: Vec<ExprId> = num_view
                        .terms
                        .iter()
                        .map(|&(term, sign)| {
                            let div_term = ctx.add(Expr::Div(term, denom));
                            match sign {
                                Sign::Pos => div_term,
                                Sign::Neg => ctx.add(Expr::Neg(div_term)),
                            }
                        })
                        .collect();

                    // Rebuild as balanced sum
                    let new_expr = build_balanced_add(ctx, &new_terms);

                    // Check complexity to prevent cycles with AddFractionsRule
                    let old_complexity = cas_ast::count_nodes(ctx, expr);
                    let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                    // Allow if predicted complexity (after simplification) is not worse
                    if new_complexity <= old_complexity + total_reduction {
                        return Some(
                            Rewrite::new(new_expr)
                                .desc("Distribute division (simplifying)")
                                .local(expr, new_expr),
                        );
                    }
                }
            }
        }
        None
    }
);

// AnnihilationRule: Detects and cancels terms like x - x or __hold(sum) - sum
// Domain Mode Policy: Like AddInverseRule, we must respect domain_mode
// because if `x` can be undefined (e.g., a/(a-1) when a=1), then x - x
// is undefined, not 0.
// - Strict: only if no term contains potentially-undefined subexpressions
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AnnihilationRule, "Annihilation", |ctx, expr, parent_ctx| {
    // Helper: check if expression contains any Div with non-literal denominator
    // Delegates to canonical implementation that handles Hold/Matrix
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        crate::collect::has_undefined_risk(ctx, expr)
    }

    // Only process Add/Sub expressions
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    // Flatten all terms
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    flatten_additive_terms(ctx, expr, false, &mut terms);

    if terms.len() < 2 {
        return None;
    }

    // CASE 1: Look for simple pairs that cancel (term and its negation)
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            let (term_i, neg_i) = &terms[i];
            let (term_j, neg_j) = &terms[j];

            // Only if opposite signs
            if neg_i == neg_j {
                continue;
            }

            // Unwrap __hold for comparison
            let unwrapped_i = unwrap_hold(ctx, *term_i);
            let unwrapped_j = unwrap_hold(ctx, *term_j);

            // Check structural or polynomial equality
            if poly_equal(ctx, unwrapped_i, unwrapped_j) {
                // These terms cancel. If they're the only 2 terms, result is 0
                if terms.len() == 2 {
                    // DOMAIN MODE GATE: Check for undefined risk
                    let domain_mode = parent_ctx.domain_mode();
                    let either_has_risk =
                        has_undefined_risk(ctx, *term_i) || has_undefined_risk(ctx, *term_j);

                    if domain_mode == crate::DomainMode::Strict && either_has_risk {
                        return None;
                    }

                    // Note: domain assumption would be emitted here if Assume mode and either_has_risk
                    // but assumption_events are not emitted for this case yet

                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("x - x = 0"));
                }
            }
        }
    }

    // CASE 2: Handle __hold(A+B+...) with distributed -(A) -(B) -(...)
    // Find __hold terms and check if remaining negated terms sum to their content
    for (idx, (term, is_neg)) in terms.iter().enumerate() {
        if *is_neg {
            continue; // Only check positive __hold terms
        }

        // Check if this is a __hold using canonical helper
        if cas_ast::hold::is_hold(ctx, *term) {
            // Unwrap the held content
            let held_content = cas_ast::hold::unwrap_hold(ctx, *term);

            // Flatten the held content to get its terms
            let mut held_terms: Vec<(ExprId, bool)> = Vec::new();
            flatten_additive_terms(ctx, held_content, false, &mut held_terms);

            // Get all other terms (excluding this __hold)
            let other_terms: Vec<(ExprId, bool)> = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, t)| *t)
                .collect();

            // Check if held_terms and other_terms cancel out
            // They cancel if for each held term there's an opposite signed other term
            if held_terms.len() == other_terms.len() {
                let mut all_cancel = true;
                let mut used = vec![false; other_terms.len()];

                for (held_term, held_neg) in &held_terms {
                    let mut found = false;

                    for (j, (other_term, other_neg)) in other_terms.iter().enumerate() {
                        if used[j] {
                            continue;
                        }

                        // Check if terms cancel (one positive, one negative equivalently)
                        // Case 1: Same term with opposite flags
                        if *other_neg != *held_neg {
                            // Use poly_equal for more robust comparison
                            // This handles cases where expressions are semantically equal
                            // but structurally different (e.g., Mul(15,x) vs Mul(x,15))
                            if poly_equal(ctx, *held_term, *other_term) {
                                used[j] = true;
                                found = true;
                                break;
                            }
                        }

                        // Case 2: Number with same flag but opposite value (e.g., 1 vs -1)
                        if *other_neg == *held_neg {
                            if let (Expr::Number(n1), Expr::Number(n2)) =
                                (ctx.get(*held_term), ctx.get(*other_term))
                            {
                                if n1 == &-n2.clone() {
                                    used[j] = true;
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !found {
                        all_cancel = false;
                        break;
                    }
                }

                if all_cancel && used.iter().all(|&u| u) {
                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("__hold(sum) - sum = 0"));
                }
            }
        }
    }

    None
});

// CombineLikeTermsRule: Collects like terms in Add/Mul expressions
// Now uses collect_with_semantics for domain_mode awareness:
// - Strict: refuses to cancel terms with undefined risk (e.g., x/(x+1) - x/(x+1))
// - Assume: cancels with domain_assumption warning
// - Generic: cancels unconditionally
define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr, parent_ctx| {
        // Only try to collect if it's an Add or Mul
        let is_add_or_mul = matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Mul(_, _));

        if is_add_or_mul {
            // CRITICAL: Do NOT apply to non-commutative expressions (e.g., matrices)
            if !ctx.is_mul_commutative(expr) {
                return None;
            }

            // Use semantics-aware collect that respects domain_mode
            let result = crate::collect::collect_with_semantics(ctx, expr, parent_ctx)?;

            // Check if structurally different to avoid infinite loops with ID regeneration
            if crate::ordering::compare_expr(ctx, result.new_expr, expr) == Ordering::Equal {
                return None;
            }

            // V2.14.26: Skip trivial changes that only normalize -1 coefficients
            // without actually combining or cancelling any terms.
            // This avoids noisy steps like "-1·x → -x" that don't add didactic value.
            if result.cancelled.is_empty() && result.combined.is_empty() {
                return None;
            }

            // V2.9.18: Restore granular focus using CollectResult's cancelled/combined groups
            // This provides specific focus like "5 - 5 → 0" for didactic clarity
            // Timeline highlighting uses step.path separately for broader context
            let (before_local, after_local, description) = select_best_focus(ctx, &result);

            let mut rewrite = Rewrite::new(result.new_expr).desc(description);
            if let (Some(before), Some(after)) = (before_local, after_local) {
                rewrite = rewrite.local(before, after);
            }
            return Some(rewrite);
        }
        None
    }
);

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    // V2.15.8: ExpandSmallBinomialPowRule - controlled by autoexpand_binomials flag
    // Enable via REPL: set autoexpand_binomials on
    simplifier.add_rule(Box::new(ExpandSmallBinomialPowRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
    simplifier.add_rule(Box::new(PolynomialIdentityZeroRule));
    // V2.15.8: HeuristicPolyNormalizeAddRule - poly-normalize Add/Sub in Heuristic mode
    // V2.15.9: HeuristicExtractCommonFactorAddRule - extract common factors first (priority 110)
    simplifier.add_rule(Box::new(HeuristicExtractCommonFactorAddRule));
    // V2.16: ExtractCommonMulFactorRule - extract common multiplicative factors from n-ary sums
    // Fixes cross-product NF divergence in metamorphic Mul tests (priority 108)
    simplifier.add_rule(Box::new(ExtractCommonMulFactorRule));
    simplifier.add_rule(Box::new(HeuristicPolyNormalizeAddRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

    #[test]
    fn test_distribute() {
        let mut ctx = Context::new();
        let rule = DistributeRule;
        // x^2 * (x + 3) - use x^2 (not an integer) so guard doesn't block
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let add = ctx.add(Expr::Add(x, three));
        let expr = ctx.add(Expr::Mul(x_sq, add));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be (x^2 * x) + (x^2 * 3) before further simplification
        // Note: x^2*x -> x^3 happens in a later pass, not in DistributeRule
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^2 * x + x^2 * 3" // Canonical: polynomial order (x terms before constants)
        );
    }

    #[test]
    fn test_annihilation() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));
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
            "0"
        );
    }

    #[test]
    fn test_combine_like_terms() {
        let mut ctx = Context::new();
        let rule = CombineLikeTermsRule;

        // 2x + 3x -> 5x
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let term1 = ctx.add(Expr::Mul(two, x));
        let term2 = ctx.add(Expr::Mul(three, x));
        let expr = ctx.add(Expr::Add(term1, term2));

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
            "5 * x"
        );

        // x + 2x -> 3x
        let term1 = x;
        let term2 = ctx.add(Expr::Mul(two, x));
        let expr2 = ctx.add(Expr::Add(term1, term2));
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
            "3 * x"
        );

        // ln(x) + ln(x) -> 2 * ln(x)
        let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);
        let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "2 * ln(x)"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule() {
        // Test: (a+b)^2 - (a^2 + 2ab + b^2) = 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2 + 2ab + b^2
        let a_sq = ctx.add(Expr::Pow(a, two));
        let b_sq = ctx.add(Expr::Pow(b, two));
        let ab = ctx.add(Expr::Mul(a, b));
        let two_ab = ctx.add(Expr::Mul(two, ab));
        let sum1 = ctx.add(Expr::Add(a_sq, two_ab));
        let rhs = ctx.add(Expr::Add(sum1, b_sq));

        // (a+b)^2 - (a^2 + 2ab + b^2)
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, rhs));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should simplify to 0
        assert!(rewrite.is_some(), "Polynomial identity should be detected");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.unwrap().new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule_non_identity() {
        // Test: (a+b)^2 - a^2 ≠ 0 (not an identity)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2
        let a_sq = ctx.add(Expr::Pow(a, two));

        // (a+b)^2 - a^2
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, a_sq));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should NOT return a rewrite (not an identity to 0)
        assert!(rewrite.is_none(), "Non-identity should not trigger rule");
    }
}
