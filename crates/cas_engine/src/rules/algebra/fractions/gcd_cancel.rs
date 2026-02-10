//! GCD-based cancellation rules and didactic factorization helpers.
//!
//! This module contains the heavy cancellation rules that use structural
//! comparison, polynomial GCD, and factorization to simplify fractions.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::polynomial::Polynomial;
use crate::rule::{ChainedRewrite, Rewrite};
use crate::rules::algebra::helpers::{
    collect_denominators, count_nodes_of_type, distribute, gcd_rational,
};
use crate::target_kind::TargetKind;
use cas_ast::{DisplayExpr, Expr, ExprId};
use num_traits::{One, Zero};

// Import helpers from sibling core_rules module
use super::core_rules::{poly_relation, try_multivar_gcd, SignRelation};
use crate::multipoly::GcdLayer;

// ========== Micro-API for safe Mul construction ==========
// Use this instead of ctx.add(Expr::Mul(...)) in this file.

// =============================================================================
// STEP 1.5: Cancel same-base power fractions P^m/P^n → P^(m-n) (shallow, PRE-ORDER)
// =============================================================================

// V2.14.35: Ultra-light rule for Pow(base,m)/Pow(base,n) → base^(m-n)
// Uses shallow ExprId comparison to avoid recursion/stack depth issues.
// This handles cases like ((x+y)^10)/((x+y)^9) that would otherwise overflow stack.
define_rule!(
    CancelPowersDivisionRule,
    "Cancel Same-Base Powers",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(Pow(base_num, exp_num), Pow(base_den, exp_den))
        let (num, den) = crate::helpers::as_div(ctx, expr)?;
        let (base_num, exp_num) = crate::helpers::as_pow(ctx, num)?;
        let (base_den, exp_den) = crate::helpers::as_pow(ctx, den)?;

        // STRUCTURAL COMPARISON: Use compare_expr to check if bases are structurally equal
        // This handles cases where (x+y) is parsed separately in num and den
        if crate::ordering::compare_expr(ctx, base_num, base_den) != std::cmp::Ordering::Equal {
            return None;
        }

        // Get exponents as integers
        let m = crate::helpers::as_i64(ctx, exp_num)?;
        let n = crate::helpers::as_i64(ctx, exp_den)?;

        // Skip if both are zero (undefined)
        if m == 0 && n == 0 {
            return None;
        }

        // DOMAIN GATE: need base ≠ 0 (derived from original denominator P^n ≠ 0)
        let domain_mode = parent_ctx.domain_mode();
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            domain_mode,
            parent_ctx.value_domain(),
            &Predicate::NonZero(base_num),
            "Cancel Same-Base Powers",
        );

        if !decision.allow {
            return None;
        }

        // Compute effective exponent difference: P^m / P^n = P^(m-n)
        let diff = m - n;

        // Build result based on diff
        let (result, desc) = if diff == 0 {
            // P^n / P^n → 1
            (ctx.num(1), format!("Cancel: P^{}/P^{} → 1", m, n))
        } else if diff == 1 {
            // P^(n+1) / P^n → P
            (base_num, format!("Cancel: P^{}/P^{} → P", m, n))
        } else if diff == -1 {
            // P^n / P^(n+1) → 1/P
            let one = ctx.num(1);
            let result = ctx.add(Expr::Div(one, base_num));
            (result, format!("Cancel: P^{}/P^{} → 1/P", m, n))
        } else if diff > 0 {
            // P^m / P^n → P^(m-n) where m > n
            let new_exp = ctx.num(diff);
            let result = ctx.add(Expr::Pow(base_num, new_exp));
            (result, format!("Cancel: P^{}/P^{} → P^{}", m, n, diff))
        } else {
            // P^m / P^n → 1/P^(n-m) where m < n
            let pos_diff = -diff;
            let new_exp = ctx.num(pos_diff);
            let pow_result = ctx.add(Expr::Pow(base_num, new_exp));
            let one = ctx.num(1);
            let result = ctx.add(Expr::Div(one, pow_result));
            (result, format!("Cancel: P^{}/P^{} → 1/P^{}", m, n, pos_diff))
        };

        Some(Rewrite::new(result)
            .desc(desc)
            .local(expr, result)
            .requires(ImplicitCondition::NonZero(base_num))
            .assume_all(decision.assumption_events(ctx, base_num)))
    }
);

// =============================================================================
// STEP 2: Cancel identical numerator/denominator (P/P → 1)
// =============================================================================

// Cancels P/P → 1 when numerator equals denominator structurally.
// This is Step 2 after didactic expansion rules (e.g., (a+b)² → a² + 2ab + b²).
define_rule!(
    CancelIdenticalFractionRule,
    "Cancel Identical Numerator/Denominator",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(num, den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Check if num == den structurally
        if crate::ordering::compare_expr(ctx, num, den) != std::cmp::Ordering::Equal {
            return None;
        }

        // DOMAIN GATE: In Strict mode, only cancel if den is provably non-zero
        let domain_mode = parent_ctx.domain_mode();
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            domain_mode,
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Cancel Identical Numerator/Denominator",
        );

        if !decision.allow {
            // Strict mode + Unknown proof: don't simplify (e.g., x/x stays)
            return None;
        }

        // Match! P/P → 1
        let one = ctx.num(1);

        Some(Rewrite::new(one)
            .desc("Cancel: P/P → 1")
            .local(expr, one)
            .requires(ImplicitCondition::NonZero(den))
            .assume_all(decision.assumption_events(ctx, den)))
    }
);

// Rule to cancel P^n / P → P^(n-1) (didactic step 2 for perfect squares and similar)
// Handles patterns like (x-y)²/(x-y) → x-y
define_rule!(
    CancelPowerFractionRule,
    "Cancel Power Fraction",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(Pow(base, exp), den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;
        let (base, exp) = crate::helpers::as_pow(ctx, num)?;

        // Check if base == den OR base == -den using poly_relation
        let relation = poly_relation(ctx, base, den)?;

        // Get exponent as integer
        let exp_val = crate::helpers::as_i64(ctx, exp)?;
        if exp_val < 1 {
            return None; // Only handle exp >= 1
        }

        // DOMAIN GATE
        let domain_mode = parent_ctx.domain_mode();
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            domain_mode,
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Cancel Power Fraction",
        );

        if !decision.allow {
            return None;
        }

        // Build base result: P^(n-1) or 1 if n=1
        let base_result = if exp_val == 1 {
            ctx.num(1)
        } else if exp_val == 2 {
            base
        } else {
            let new_exp = ctx.num(exp_val - 1);
            ctx.add(Expr::Pow(base, new_exp))
        };

        // Apply sign based on relation
        let (result, desc) = match relation {
            SignRelation::Same => (
                base_result,
                "Cancel: P^n/P → P^(n-1)"
            ),
            SignRelation::Negated => {
                // P^n / (-P) = -P^(n-1)
                let negated = ctx.add(Expr::Neg(base_result));
                (negated, "Cancel: P^n/(-P) → -P^(n-1)")
            }
        };

        Some(Rewrite::new(result)
            .desc(desc)
            .local(expr, result)
            .requires(ImplicitCondition::NonZero(den))
            .assume_all(decision.assumption_events(ctx, den)))
    }
);

// Didactic factorization helpers (extracted to gcd_cancel_didactic.rs)
use super::gcd_cancel_didactic::{
    try_difference_of_squares_in_num, try_expand_binomial_square_in_den_for_cancel,
    try_perfect_square_minus_in_num, try_power_quotient_preserve_form,
    try_sum_diff_of_cubes_in_num,
};

define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;
        use cas_ast::views::RationalFnView;

        // Capture domain mode for cancellation decisions
        let domain_mode = parent_ctx.domain_mode();

        // Use RationalFnView to detect any fraction form while preserving structure
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

        // EARLY RETURN: Check for didactic perfect-square cancellation
        // (a^2 + 2ab + b^2) / (a+b)^2 → 1 with visible expansion step
        if let Some(rewrite) = try_expand_binomial_square_in_den_for_cancel(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for difference of squares factorization
        // (a² - b²) / (a+b) → (a-b)(a+b) / (a+b) with visible factorization step
        if let Some(rewrite) = try_difference_of_squares_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for perfect square minus recognition
        // (a² - 2ab + b²) / (a-b) → (a-b)²/(a-b) = a-b
        if let Some(rewrite) = try_perfect_square_minus_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for sum/difference of cubes factorization
        // (a³ - b³) / (a-b) → (a-b)(a²+ab+b²) / (a-b) = a²+ab+b²
        if let Some(rewrite) = try_sum_diff_of_cubes_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Power quotient preserving factored form
        // V2.14.45: (x-1)⁴/(x-1)² → (x-1)² instead of x² - 2x + 1
        if let Some(rewrite) = try_power_quotient_preserve_form(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // NOTE: PR-2 shallow GCD integration deferred.
        // The gcd_shallow_for_fraction function exists in poly_gcd.rs but calling it
        // here adds stack depth that causes overflow on complex expressions.
        // Future work: investigate stack-safe approach for power cancellation.

        // 0. Try multivariate GCD (Layer 1: monomial + content)
        let vars = cas_ast::collect_variables(ctx, expr);
        if vars.len() > 1 {
            if let Some((new_num, new_den, gcd_expr, layer)) = try_multivar_gcd(ctx, num, den) {
                // DOMAIN GATE: Check if we can cancel by this GCD
                // In Strict mode, only allow if GCD is provably non-zero
                let decision = crate::domain_oracle::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(gcd_expr),
                    "Simplify Nested Fraction",
                );
                if !decision.allow {
                    // Strict mode + Unknown proof: don't simplify (e.g., x/x stays)
                    return None;
                }

                // Short-circuit: if new_num is 0, result is just 0 - avoid confusing 0*GCD/GCD pattern
                // Let DivZeroRule or subsequent simplification handle it cleanly
                let num_is_zero = matches!(ctx.get(new_num), Expr::Number(n) if n.is_zero());
                if num_is_zero {
                    // Just return 0 directly with a cleaner description
                    let zero = ctx.num(0);
                    use crate::implicit_domain::ImplicitCondition;
                    return Some(
                        Rewrite::new(zero)
                            .desc("Numerator simplifies to 0")
                            .local(num, zero)
                            .requires(ImplicitCondition::NonZero(den))
                    );
                }

                // Build factored form for display
                let factored_num = mul2_raw(ctx, new_num, gcd_expr);
                let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        gcd_expr
                    } else {
                        mul2_raw(ctx, new_den, gcd_expr)
                    }
                } else {
                    mul2_raw(ctx, new_den, gcd_expr)
                };
                let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

                // Layer tag for verbose description (omit for clean didactic display)
                let _layer_tag = match layer {
                    GcdLayer::Layer1MonomialContent => "Layer 1: monomial+content",
                    GcdLayer::Layer2HeuristicSeeds => "Layer 2: heuristic seeds",
                    GcdLayer::Layer25TensorGrid => "Layer 2.5: tensor grid",
                };

                // Compute final result
                let result = if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        new_num // Denominator simplified to 1
                    } else {
                        ctx.add(Expr::Div(new_num, new_den))
                    }
                } else {
                    ctx.add(Expr::Div(new_num, new_den))
                };

                // Normalize expressions to ensure Rule: and After: display consistently
                use crate::canonical_forms::normalize_core;
                let factored_form_norm = normalize_core(ctx, factored_form);
                let result_norm = normalize_core(ctx, result);

                // === ChainedRewrite Pattern: Factor -> Cancel ===
                // Step 1 (main): Factor - show the factored form
                // Use requires (not assume) to avoid duplicate Requires/Assumed display
                use crate::implicit_domain::ImplicitCondition;
                let gcd_display = format!("{}", DisplayExpr { context: ctx, id: gcd_expr });
                let factor_rw = Rewrite::new(factored_form_norm)
                    .desc_lazy(|| format!("Factor by GCD: {}", gcd_display))
                    .local(expr, factored_form_norm)
                    .requires(ImplicitCondition::NonZero(den));

                // Step 2 (chained): Cancel - reduce to final result
                let cancel = ChainedRewrite::new(result_norm)
                    .desc("Cancel common factor")
                    .local(factored_form_norm, result_norm);

                return Some(factor_rw.chain(cancel));
            }
        }


        // 1. Univariate path: require single variable
        if vars.len() != 1 {
            return None;
        }
        let var = vars.iter().next()?;

        // 2. Convert to Polynomials
        let p_num = Polynomial::from_expr(ctx, num, var).ok()?;
        let p_den = Polynomial::from_expr(ctx, den, var).ok()?;

        if p_den.is_zero() {
            return None;
        }

        // 3. Compute Polynomial GCD (monic)
        let poly_gcd = p_num.gcd(&p_den);

        // 4. Compute Numeric Content GCD
        // Polynomial GCD is monic, so it misses numeric factors like 27x^3 / 9 -> gcd=9
        let content_num = p_num.content();
        let content_den = p_den.content();

        // Helper to compute GCD of two rationals (assuming integers for now)
        let numeric_gcd = gcd_rational(content_num, content_den);

        // 5. Combine
        // full_gcd = poly_gcd * numeric_gcd
        let scalar = Polynomial::new(vec![numeric_gcd.clone()], var.to_string());
        let full_gcd = poly_gcd.mul(&scalar);

        // 6. Check if GCD is non-trivial
        // If degree is 0 and constant is 1, it's trivial.
        if full_gcd.degree() == 0 && full_gcd.leading_coeff().is_one() {
            return None;
        }

        // 7. Divide by GCD (full_gcd is non-zero since we checked it's non-trivial above)
        let (new_num_poly, rem_num) = match p_num.div_rem(&full_gcd) {
            Ok(result) => result,
            Err(_) => return None,
        };
        let (new_den_poly, rem_den) = match p_den.div_rem(&full_gcd) {
            Ok(result) => result,
            Err(_) => return None,
        };

        if !rem_num.is_zero() || !rem_den.is_zero() {
            return None;
        }

        let new_num = new_num_poly.to_expr(ctx);
        let new_den = new_den_poly.to_expr(ctx);
        let gcd_expr = full_gcd.to_expr(ctx);

        // DOMAIN GATE: Check if we can cancel by this GCD
        // In Strict mode, only allow if GCD is provably non-zero
        let decision = crate::domain_oracle::oracle_allows_with_hint(
            ctx,
            domain_mode,
            parent_ctx.value_domain(),
            &Predicate::NonZero(gcd_expr),
            "Simplify Nested Fraction",
        );
        if !decision.allow {
            // STRICT PARTIAL CANCEL: Try to cancel only numeric content
            // The numeric_gcd is always provably nonzero (it's a rational ≠ 0)
            if !numeric_gcd.is_one() && !numeric_gcd.is_zero() {
                // Divide both polys by numeric content only (safe in Strict)
                let new_num_partial = p_num.div_scalar(&numeric_gcd);
                let new_den_partial = p_den.div_scalar(&numeric_gcd);

                let new_num_expr = new_num_partial.to_expr(ctx);
                let new_den_expr = new_den_partial.to_expr(ctx);
                let result = ctx.add(Expr::Div(new_num_expr, new_den_expr));

                return Some(Rewrite::new(result)
                    .desc_lazy(|| format!("Reduced numeric content by gcd {} (strict-safe)", numeric_gcd))
                    .local(expr, result));
            }
            // No numeric content to cancel, don't simplify
            return None;
        }

        // Short-circuit: if new_num is 0, result is just 0 - avoid confusing 0*GCD/GCD pattern
        let num_is_zero = matches!(ctx.get(new_num), Expr::Number(n) if n.is_zero());
        if num_is_zero {
            let zero = ctx.num(0);
            use crate::implicit_domain::ImplicitCondition;
            return Some(
                Rewrite::new(zero)
                    .desc("Numerator simplifies to 0")
                    .local(num, zero)
                    .requires(ImplicitCondition::NonZero(den))
            );
        }

        // Build factored form for "Rule:" display: (new_num * gcd) / (new_den * gcd)
        // This shows the factorization step more clearly
        let factored_num = mul2_raw(ctx, new_num, gcd_expr);
        let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                gcd_expr // denominator is just the GCD
            } else {
                mul2_raw(ctx, new_den, gcd_expr)
            }
        } else {
            mul2_raw(ctx, new_den, gcd_expr)
        };
        let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

        // Compute final result
        let result = if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                new_num // Denominator simplified to 1
            } else {
                ctx.add(Expr::Div(new_num, new_den))
            }
        } else {
            ctx.add(Expr::Div(new_num, new_den))
        };

        // Normalize expressions to ensure Rule: and After: display consistently
        use crate::canonical_forms::normalize_core;
        let factored_form_norm = normalize_core(ctx, factored_form);
        let result_norm = normalize_core(ctx, result);

        // === ChainedRewrite Pattern: Factor -> Cancel ===
        // Step 1 (main): Factor - show the factored form
        // Use requires (not assume) to avoid duplicate Requires/Assumed display
        use crate::implicit_domain::ImplicitCondition;
        let gcd_display = format!("{}", DisplayExpr { context: ctx, id: gcd_expr });
        let factor_rw = Rewrite::new(factored_form_norm)
            .desc_lazy(|| format!("Factor by GCD: {}", gcd_display))
            .local(expr, factored_form_norm)
            .requires(ImplicitCondition::NonZero(den));

        // Step 2 (chained): Cancel - reduce to final result
        let cancel = ChainedRewrite::new(result_norm)
            .desc("Cancel common factor")
            .local(factored_form_norm, result_norm);

        return Some(factor_rw.chain(cancel));
    }
);

define_rule!(
    NestedFractionRule,
    "Simplify Complex Fraction",
    |ctx, expr| {
        use cas_ast::views::RationalFnView;

        // Use RationalFnView to detect any fraction form
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

        let num_denoms = collect_denominators(ctx, num);
        let den_denoms = collect_denominators(ctx, den);

        if num_denoms.is_empty() && den_denoms.is_empty() {
            return None;
        }

        // Collect all unique denominators
        let mut all_denoms = Vec::new();
        all_denoms.extend(num_denoms);
        all_denoms.extend(den_denoms);

        if all_denoms.is_empty() {
            return None;
        }

        // Construct the common multiplier (product of all unique denominators)
        // Ideally LCM, but product is safer for now.
        // We need to deduplicate.
        let mut unique_denoms: Vec<ExprId> = Vec::new();
        for d in all_denoms {
            if !unique_denoms.contains(&d) {
                unique_denoms.push(d);
            }
        }

        if unique_denoms.is_empty() {
            return None;
        }

        let (&first, rest) = unique_denoms.split_first()?;
        let multiplier = rest
            .iter()
            .copied()
            .fold(first, |acc, d| mul2_raw(ctx, acc, d));

        // Multiply num and den by multiplier
        let new_num = distribute(ctx, num, multiplier);
        let new_den = distribute(ctx, den, multiplier);

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        if new_expr == expr {
            return None;
        }

        // Complexity Check: Ensure we actually reduced the number of divisions or total nodes
        // Counting Div nodes is a good heuristic for "nested fraction simplified"
        let count_divs = |id| count_nodes_of_type(ctx, id, TargetKind::Div);
        let old_divs = count_divs(expr);
        let new_divs = count_divs(new_expr);

        if new_divs >= old_divs {
            return None;
        }

        return Some(
            Rewrite::new(new_expr)
                .desc("Simplify nested fraction")
                .local(expr, new_expr),
        );
    }
);

define_rule!(
    SimplifyMulDivRule,
    "Simplify Multiplication with Division",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use zero-clone destructuring
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        // Use FractionParts to detect any fraction-like structure
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        // If neither side has denominators, nothing to do
        if !fp_l.is_fraction() && !fp_r.is_fraction() {
            return None;
        }

        // Check for simple cancellation: (a/b) * b -> a
        // Only for simple cases to avoid over-simplification
        if fp_l.is_fraction() && fp_l.den.len() == 1 && fp_l.den[0].exp == 1 {
            let den_base = fp_l.den[0].base;
            // Check if r equals the denominator
            if crate::ordering::compare_expr(ctx, den_base, r) == std::cmp::Ordering::Equal {
                // Cancel: (a/b) * b -> a
                let result = if fp_l.num.is_empty() {
                    ctx.num(fp_l.sign as i64)
                } else {
                    let num_prod = FractionParts::build_product_static(ctx, &fp_l.num);
                    if fp_l.sign < 0 {
                        ctx.add(Expr::Neg(num_prod))
                    } else {
                        num_prod
                    }
                };
                return Some(Rewrite::new(result).desc("Cancel division: (a/b)*b -> a"));
            }
        }

        // Check for simple cancellation: a * (b/a) -> b
        if fp_r.is_fraction() && fp_r.den.len() == 1 && fp_r.den[0].exp == 1 {
            let den_base = fp_r.den[0].base;
            if crate::ordering::compare_expr(ctx, den_base, l) == std::cmp::Ordering::Equal {
                let result = if fp_r.num.is_empty() {
                    ctx.num(fp_r.sign as i64)
                } else {
                    let num_prod = FractionParts::build_product_static(ctx, &fp_r.num);
                    if fp_r.sign < 0 {
                        ctx.add(Expr::Neg(num_prod))
                    } else {
                        num_prod
                    }
                };
                return Some(Rewrite::new(result).desc("Cancel division: a*(b/a) -> b"));
            }
        }

        // Avoid combining if either side is just a constant (prefer k * (a/b) for CombineLikeTerms)
        if matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_))
            || matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_))
        {
            return None;
        }

        // Combine into single fraction: (n1/d1) * (n2/d2) -> (n1*n2)/(d1*d2)
        // Only do this if at least one side is an actual fraction
        if fp_l.is_fraction() || fp_r.is_fraction() {
            // Build combined numerator: products of all num factors
            let mut combined_num = Vec::new();
            combined_num.extend(fp_l.num.iter().cloned());
            combined_num.extend(fp_r.num.iter().cloned());

            // Build combined denominator
            let mut combined_den = Vec::new();
            combined_den.extend(fp_l.den.iter().cloned());
            combined_den.extend(fp_r.den.iter().cloned());

            let combined_sign = (fp_l.sign as i16 * fp_r.sign as i16) as i8;

            let result_fp = FractionParts {
                sign: combined_sign,
                num: combined_num,
                den: combined_den,
            };

            // Build as division for didactic output
            let new_expr = result_fp.build_as_div(ctx);

            // Avoid no-op rewrites
            if new_expr == expr {
                return None;
            }

            return Some(Rewrite::new(new_expr).desc("Combine fractions in multiplication"));
        }
        None
    }
);
