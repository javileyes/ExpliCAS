use crate::build::mul2_raw;
use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};

// Use parent module's helpers
use super::core_rules::{build_mul_from_factors_a1, collect_mul_factors_int_pow};

// =============================================================================
// DivAddCommonFactorFromDenRule: Factor out MULTI-FACTOR from Add numerator
// when those factors appear in the denominator, enabling cancellation.
// =============================================================================
//
// V2: Multi-factor version - extracts the MAXIMUM common factor:
// - Intersects factor multisets across ALL Add terms
// - Caps each factor exponent by what's in the denominator
// - Only fires if the common factor is non-trivial
//
// Pattern: Div(Add(f*g*a, f*g*b), f*g*c) → Div(f*g*Add(a, b), f*g*c) → Add(a, b)/c
// =============================================================================
define_rule!(
    DivAddCommonFactorFromDenRule,
    "Factor Common Factors from Add in Div",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        use std::collections::HashMap;
        // Helper to collect Add terms (flattened)
        fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
            match ctx.get(expr) {
                Expr::Add(l, r) => {
                    collect_add_terms(ctx, *l, terms);
                    collect_add_terms(ctx, *r, terms);
                }
                _ => terms.push(expr),
            }
        }

        // Only match Div(num, den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Numerator must be Add
        if !matches!(ctx.get(num), Expr::Add(_, _)) {
            return None;
        }

        let mut add_terms: Vec<ExprId> = Vec::new();
        collect_add_terms(ctx, num, &mut add_terms);

        if add_terms.len() < 2 {
            return None;
        }

        // Collect factors from denominator (key -> (ExprId, exp))
        // Helper to build factor map from (ExprId, exp) list
        // Uses string repr as key for structural equality
        fn factors_to_map(ctx: &Context, factors: &[(ExprId, i64)]) -> HashMap<String, (ExprId, i64)> {
            let mut map = HashMap::new();
            for &(base, exp) in factors {
                // Skip numeric factors (we're only interested in symbolic)
                if matches!(ctx.get(base), Expr::Number(_)) {
                    continue;
                }
                let key = format!("{}", cas_ast::DisplayExpr { context: ctx, id: base });
                let entry = map.entry(key).or_insert((base, 0));
                entry.1 += exp;
            }
            map
        }

        let den_factors_raw = collect_mul_factors_int_pow(ctx, den);
        let den_map = factors_to_map(ctx, &den_factors_raw);

        // If denominator has no symbolic factors, nothing to do
        if den_map.is_empty() {
            return None;
        }

        // Compute intersection of factor exponents across all Add terms
        // Start with factors from first term
        let first_factors = collect_mul_factors_int_pow(ctx, add_terms[0]);
        let mut common_map = factors_to_map(ctx, &first_factors);

        // Intersect with each subsequent term
        for term_id in add_terms.iter().skip(1) {
            let term_factors = collect_mul_factors_int_pow(ctx, *term_id);
            let term_map = factors_to_map(ctx, &term_factors);

            // Keep only keys present in both, with min exponent
            common_map.retain(|key, (_, exp)| {
                if let Some((_, term_exp)) = term_map.get(key) {
                    *exp = (*exp).min(*term_exp);
                    *exp >= 1 // Only keep if exponent is positive
                } else {
                    false // Factor not present in this term
                }
            });

            // Early exit if nothing left
            if common_map.is_empty() {
                return None;
            }
        }

        // Cap by denominator: only extract what the denominator has
        for (key, (_, exp)) in common_map.iter_mut() {
            if let Some((_, den_exp)) = den_map.get(key) {
                *exp = (*exp).min(*den_exp);
            } else {
                *exp = 0; // Don't extract factors not in denominator
            }
        }

        // Remove factors with 0 exponent
        common_map.retain(|_, (_, exp)| *exp >= 1);

        // If nothing to extract, bail
        if common_map.is_empty() {
            return None;
        }

        // Build the common factor list
        let common_factors: Vec<(ExprId, i64)> = common_map.values().cloned().collect();

        // Now extract common_factors from each Add term
        let mut new_terms: Vec<ExprId> = Vec::new();

        for term_id in &add_terms {
            // Check if term is Neg(inner) - need to preserve sign
            let (actual_term, is_neg) = match ctx.get(*term_id) {
                Expr::Neg(inner) => (*inner, true),
                _ => (*term_id, false),
            };

            let term_factors = collect_mul_factors_int_pow(ctx, actual_term);

            // Build quotient factors: original - common
            let mut quotient_factors: Vec<(ExprId, i64)> = Vec::new();

            for (base, exp) in term_factors {
                if matches!(ctx.get(base), Expr::Number(_)) {
                    // Keep numeric factors as-is
                    quotient_factors.push((base, exp));
                    continue;
                }

                let key = format!("{}", cas_ast::DisplayExpr { context: ctx, id: base });
                let common_exp = common_map.get(&key).map(|(_, e)| *e).unwrap_or(0);
                let new_exp = exp - common_exp;
                if new_exp > 0 {
                    quotient_factors.push((base, new_exp));
                }
            }

            let quotient = build_mul_from_factors_a1(ctx, &quotient_factors);

            // Preserve Neg wrapper if original term was negative
            let final_quotient = if is_neg {
                ctx.add(Expr::Neg(quotient))
            } else {
                quotient
            };
            new_terms.push(final_quotient);
        }

        // Build new Add from quotient terms
        let new_add = if new_terms.len() == 1 {
            new_terms[0]
        } else {
            let mut result = new_terms[0];
            for term in new_terms.iter().skip(1) {
                result = ctx.add(Expr::Add(result, *term));
            }
            result
        };

        // Build common factor product
        let common_product = build_mul_from_factors_a1(ctx, &common_factors);

        // New numerator: common_product * new_add
        let new_num = mul2_raw(ctx, common_product, new_add);

        // Build result: Div(new_num, den)
        let result = ctx.add(Expr::Div(new_num, den));

        // Only return if we actually changed something
        if result != expr {
            return Some(Rewrite::new(result)
                .desc("Factor common factors from Add in Div"));
        }

        None
    }
);

// =============================================================================
// DivDenFactorOutRule: Factor out common symbolic factors from Add denominator
// =============================================================================
//
// Pattern: Div(num, f*a + f*b + f*c) → Div(num, f*(a+b+c))
//
// This enables CancelCommonFactorsRule to then cancel f from num/den.
// Example: (x+2)³·eᵘ / (eᵘ·x³ + 6eᵘ·x² + 12eᵘ·x + 8eᵘ)
//       → (x+2)³·eᵘ / (eᵘ·(x³ + 6x² + 12x + 8))
//       → (x+2)³ / (x³ + 6x² + 12x + 8)   [via CancelCommonFactorsRule]
//       → 1                                  [via GCD cancel]
// =============================================================================
define_rule!(
    DivDenFactorOutRule,
    "Factor Out Common From Denominator",
    |ctx, expr| {
        use std::collections::HashMap;

        // Only match Div(num, den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Denominator must be Add
        if !matches!(ctx.get(den), Expr::Add(_, _)) {
            return None;
        }

        // Flatten denominator Add terms
        let den_terms = crate::nary::add_leaves(ctx, den);
        if den_terms.len() < 2 {
            return None;
        }

        // Collect factors from each denominator term, using string repr as key
        fn factors_to_map(
            ctx: &Context,
            factors: &[(ExprId, i64)],
        ) -> HashMap<String, (ExprId, i64)> {
            let mut map = HashMap::new();
            for &(base, exp) in factors {
                // Skip numeric factors (we're only interested in symbolic)
                if matches!(ctx.get(base), Expr::Number(_)) {
                    continue;
                }
                let key = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: base
                    }
                );
                let entry = map.entry(key).or_insert((base, 0));
                entry.1 += exp;
            }
            map
        }

        // Compute intersection of symbolic factors across all denominator Add terms
        let first_factors = collect_mul_factors_int_pow(ctx, den_terms[0]);
        let mut common_map = factors_to_map(ctx, &first_factors);

        for &term_id in den_terms.iter().skip(1) {
            let term_factors = collect_mul_factors_int_pow(ctx, term_id);
            let term_map = factors_to_map(ctx, &term_factors);

            // Keep only keys present in both, with min exponent
            common_map.retain(|key, (_, exp)| {
                if let Some((_, term_exp)) = term_map.get(key) {
                    *exp = (*exp).min(*term_exp);
                    *exp >= 1
                } else {
                    false
                }
            });

            if common_map.is_empty() {
                return None;
            }
        }

        if common_map.is_empty() {
            return None;
        }

        // STRICT GUARD: Only extract factors that have an EXACT structural match
        // in the numerator. This prevents the factoring-distribution infinite loop:
        // factor-out → distribute-back → factor-out → ...
        // We use compare_expr (AST equality) instead of string matching.
        let num_factors_raw = collect_mul_factors_int_pow(ctx, num);
        common_map.retain(|_key, (den_base, _exp)| {
            num_factors_raw.iter().any(|(num_base, _)| {
                crate::ordering::compare_expr(ctx, *den_base, *num_base)
                    == std::cmp::Ordering::Equal
            })
        });
        if common_map.is_empty() {
            return None;
        }

        // Build the common factor list
        let common_factors: Vec<(ExprId, i64)> = common_map.values().cloned().collect();

        // Extract common factors from each denominator term to form quotient terms
        let mut new_den_terms: Vec<ExprId> = Vec::new();

        for &term_id in &den_terms {
            // Handle Neg(inner)
            let (actual_term, is_neg) = match ctx.get(term_id) {
                Expr::Neg(inner) => (*inner, true),
                _ => (term_id, false),
            };

            let term_factors = collect_mul_factors_int_pow(ctx, actual_term);

            // Build quotient factors: original - common
            let mut quotient_factors: Vec<(ExprId, i64)> = Vec::new();

            for (base, exp) in term_factors {
                if matches!(ctx.get(base), Expr::Number(_)) {
                    quotient_factors.push((base, exp));
                    continue;
                }
                let key = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: base
                    }
                );
                let common_exp = common_map.get(&key).map(|(_, e)| *e).unwrap_or(0);
                let new_exp = exp - common_exp;
                if new_exp > 0 {
                    quotient_factors.push((base, new_exp));
                }
            }

            let quotient = build_mul_from_factors_a1(ctx, &quotient_factors);

            let final_quotient = if is_neg {
                ctx.add(Expr::Neg(quotient))
            } else {
                quotient
            };
            new_den_terms.push(final_quotient);
        }

        // Build new denominator: common_product * Add(quotient terms)
        let common_product = build_mul_from_factors_a1(ctx, &common_factors);

        let new_add = if new_den_terms.len() == 1 {
            new_den_terms[0]
        } else {
            let mut result = new_den_terms[0];
            for &term in new_den_terms.iter().skip(1) {
                result = ctx.add(Expr::Add(result, term));
            }
            result
        };

        let new_den = mul2_raw(ctx, common_product, new_add);
        let result = ctx.add(Expr::Div(num, new_den));

        // Only return if something actually changed
        if result != expr {
            return Some(
                Rewrite::new(result).desc("Factor common symbolic factors from denominator"),
            );
        }

        None
    }
);

// =============================================================================
// DivExpandNumForCancelRule: Distribute Mul in Div numerator to enable factor
// cancellation when both sides share a common symbolic factor.
// =============================================================================
//
// Pattern: Div(Mul(X, Add(f*a, f*b)), Add(f*c, f*d))
//       → Div(Add(X*f*a, X*f*b), Add(f*c, f*d))
//
// This allows DivAddSymmetricFactorRule to then cancel f from both sides.
// Only fires when the expansion would actually expose common factors.
// =============================================================================
define_rule!(
    DivExpandNumForCancelRule,
    "Expand Numerator Product for Div Cancellation",
    |ctx, expr| {
        use std::collections::HashMap;

        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Denominator must be an Add
        if !matches!(ctx.get(den), Expr::Add(_, _)) {
            return None;
        }
        // Numerator must NOT already be an Add (symmetric rule handles that)
        if matches!(ctx.get(num), Expr::Add(_, _)) {
            return None;
        }

        // Flatten the numerator's multiplicative factors
        let num_mul_factors = {
            let mut factors = Vec::new();
            let mut stack = vec![num];
            while let Some(curr) = stack.pop() {
                if let Expr::Mul(l, r) = ctx.get(curr) {
                    stack.push(*r);
                    stack.push(*l);
                } else {
                    factors.push(curr);
                }
            }
            factors
        };
        if num_mul_factors.len() < 2 {
            return None;
        }

        // Helper: collect symbolic factors as string keyed map
        fn sym_factors(ctx: &Context, factors: &[(ExprId, i64)]) -> HashMap<String, i64> {
            let mut map = HashMap::new();
            for &(base, exp) in factors {
                if matches!(ctx.get(base), Expr::Number(_)) {
                    continue;
                }
                let key = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: base
                    }
                );
                *map.entry(key).or_insert(0) += exp;
            }
            map
        }

        // Pre-compute common symbolic factors across denominator Add terms
        let den_terms = crate::nary::add_leaves(ctx, den);
        let first_den_factors = collect_mul_factors_int_pow(ctx, den_terms[0]);
        let mut den_common = sym_factors(ctx, &first_den_factors);
        for &term_id in den_terms.iter().skip(1) {
            let term_factors = collect_mul_factors_int_pow(ctx, term_id);
            let term_map = sym_factors(ctx, &term_factors);
            den_common.retain(|key, exp| {
                if let Some(&term_exp) = term_map.get(key) {
                    *exp = (*exp).min(term_exp);
                    *exp >= 1
                } else {
                    false
                }
            });
            if den_common.is_empty() {
                return None;
            }
        }

        // Try each Add factor in the Mul tree; pick the first one whose
        // expansion creates terms sharing common factors with the denominator.
        for (add_idx, &candidate) in num_mul_factors.iter().enumerate() {
            if !matches!(ctx.get(candidate), Expr::Add(_, _)) {
                continue;
            }

            // Build "outer" from all non-candidate factors
            let outer_factors: Vec<ExprId> = num_mul_factors
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != add_idx)
                .map(|(_, &f)| f)
                .collect();
            if outer_factors.is_empty() {
                continue;
            }

            let outer = {
                let mut out = outer_factors[0];
                for &f in outer_factors.iter().skip(1) {
                    out = mul2_raw(ctx, out, f);
                }
                out
            };

            let add_terms = crate::nary::add_leaves(ctx, candidate);
            if add_terms.len() < 2 {
                continue;
            }

            // Build virtually expanded terms: outer * each_add_term
            let expanded_terms: Vec<ExprId> =
                add_terms.iter().map(|&t| mul2_raw(ctx, outer, t)).collect();

            // Compute common symbolic factors across expanded terms
            let first_factors = collect_mul_factors_int_pow(ctx, expanded_terms[0]);
            let mut num_common = sym_factors(ctx, &first_factors);
            let mut bail = false;
            for &term_id in expanded_terms.iter().skip(1) {
                let term_factors = collect_mul_factors_int_pow(ctx, term_id);
                let term_map = sym_factors(ctx, &term_factors);
                num_common.retain(|key, exp| {
                    if let Some(&te) = term_map.get(key) {
                        *exp = (*exp).min(te);
                        *exp >= 1
                    } else {
                        false
                    }
                });
                if num_common.is_empty() {
                    bail = true;
                    break;
                }
            }
            if bail {
                continue;
            }

            // Check for shared factors between expanded num and den
            if !num_common.keys().any(|k| den_common.contains_key(k)) {
                continue;
            }

            // Build the expanded numerator: Add(outer*term₁, outer*term₂, ...)
            let new_num = {
                let mut result = expanded_terms[0];
                for &term in expanded_terms.iter().skip(1) {
                    result = ctx.add(Expr::Add(result, term));
                }
                result
            };

            let result = ctx.add(Expr::Div(new_num, den));
            if result != expr {
                return Some(
                    Rewrite::new(result)
                        .desc("Expand numerator product to enable factor cancellation"),
                );
            }
        }

        None
    }
);

// =============================================================================
// DivAddSymmetricFactorRule: Extract common factor from BOTH Add num AND Add den
// =============================================================================
//
// Pattern: Div(Add(f*a, f*b), Add(f*c, f*d)) → Div(Add(a, b), Add(c, d))
//
// This complements DivAddCommonFactorFromDenRule by handling cases where
// both numerator AND denominator are Adds that share a common factor.
// The factor cancels completely, simplifying the fraction.
// =============================================================================
define_rule!(
    DivAddSymmetricFactorRule,
    "Cancel Common Factor from Add/Add Fraction",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        use std::collections::HashMap;

        // Helper to collect Add terms (flattened)
        fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
            match ctx.get(expr) {
                Expr::Add(l, r) => {
                    collect_add_terms(ctx, *l, terms);
                    collect_add_terms(ctx, *r, terms);
                }
                _ => terms.push(expr),
            }
        }

        // Helper to build factor map from (ExprId, exp) list
        fn factors_to_map(ctx: &Context, factors: &[(ExprId, i64)]) -> HashMap<String, (ExprId, i64)> {
            let mut map = HashMap::new();
            for &(base, exp) in factors {
                if matches!(ctx.get(base), Expr::Number(_)) {
                    continue;
                }
                let key = format!("{}", cas_ast::DisplayExpr { context: ctx, id: base });
                let entry = map.entry(key).or_insert((base, 0));
                entry.1 += exp;
            }
            map
        }

        // Helper to compute common factors across all Add terms
        fn compute_common_factors(ctx: &Context, terms: &[ExprId]) -> HashMap<String, (ExprId, i64)> {
            if terms.is_empty() {
                return HashMap::new();
            }

            let first_factors = collect_mul_factors_int_pow(ctx, terms[0]);
            let mut common_map = factors_to_map(ctx, &first_factors);

            for term_id in terms.iter().skip(1) {
                let term_factors = collect_mul_factors_int_pow(ctx, *term_id);
                let term_map = factors_to_map(ctx, &term_factors);

                common_map.retain(|key, (_, exp)| {
                    if let Some((_, term_exp)) = term_map.get(key) {
                        *exp = (*exp).min(*term_exp);
                        *exp >= 1
                    } else {
                        false
                    }
                });

                if common_map.is_empty() {
                    return HashMap::new();
                }
            }

            common_map
        }

        // Helper to divide each term by common factors
        fn divide_terms_by_common(
            ctx: &mut Context,
            terms: &[ExprId],
            common_map: &HashMap<String, (ExprId, i64)>
        ) -> Vec<ExprId> {
            let mut new_terms = Vec::new();

            for term_id in terms {
                // Detect and preserve Neg wrapper (collect_mul_factors_int_pow strips it)
                let (actual_term, is_negated) = match ctx.get(*term_id) {
                    Expr::Neg(inner) => (*inner, true),
                    _ => (*term_id, false),
                };

                let term_factors = collect_mul_factors_int_pow(ctx, actual_term);
                let mut quotient_factors: Vec<(ExprId, i64)> = Vec::new();

                for (base, exp) in term_factors {
                    if matches!(ctx.get(base), Expr::Number(_)) {
                        quotient_factors.push((base, exp));
                        continue;
                    }

                    let key = format!("{}", cas_ast::DisplayExpr { context: ctx, id: base });
                    let common_exp = common_map.get(&key).map(|(_, e)| *e).unwrap_or(0);
                    let new_exp = exp - common_exp;
                    if new_exp > 0 {
                        quotient_factors.push((base, new_exp));
                    }
                }

                let mut quotient = build_mul_from_factors_a1(ctx, &quotient_factors);
                // Re-apply Neg if the original term was negated
                if is_negated {
                    quotient = ctx.add(Expr::Neg(quotient));
                }
                new_terms.push(quotient);
            }

            new_terms
        }

        // Only match Div(num, den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // BOTH must be Add
        if !matches!(ctx.get(num), Expr::Add(_, _)) {
            return None;
        }
        if !matches!(ctx.get(den), Expr::Add(_, _)) {
            return None;
        }

        // Collect terms from both
        let mut num_terms: Vec<ExprId> = Vec::new();
        let mut den_terms: Vec<ExprId> = Vec::new();
        collect_add_terms(ctx, num, &mut num_terms);
        collect_add_terms(ctx, den, &mut den_terms);

        if num_terms.len() < 2 || den_terms.len() < 2 {
            return None;
        }

        // Compute common factors for numerator
        let num_common = compute_common_factors(ctx, &num_terms);
        if num_common.is_empty() {
            return None;
        }

        // Compute common factors for denominator
        let den_common = compute_common_factors(ctx, &den_terms);
        if den_common.is_empty() {
            return None;
        }

        // Intersect: only keep factors common to BOTH num and den
        let mut shared_common: HashMap<String, (ExprId, i64)> = HashMap::new();
        for (key, (base, num_exp)) in &num_common {
            if let Some((_, den_exp)) = den_common.get(key) {
                let min_exp = (*num_exp).min(*den_exp);
                if min_exp >= 1 {
                    shared_common.insert(key.clone(), (*base, min_exp));
                }
            }
        }

        if shared_common.is_empty() {
            return None;
        }

        // Divide both num and den terms by shared_common
        let new_num_terms = divide_terms_by_common(ctx, &num_terms, &shared_common);
        let new_den_terms = divide_terms_by_common(ctx, &den_terms, &shared_common);

        // Build new Add for numerator
        let new_num = if new_num_terms.len() == 1 {
            new_num_terms[0]
        } else {
            let mut result = new_num_terms[0];
            for term in new_num_terms.iter().skip(1) {
                result = ctx.add(Expr::Add(result, *term));
            }
            result
        };

        // Build new Add for denominator
        let new_den = if new_den_terms.len() == 1 {
            new_den_terms[0]
        } else {
            let mut result = new_den_terms[0];
            for term in new_den_terms.iter().skip(1) {
                result = ctx.add(Expr::Add(result, *term));
            }
            result
        };

        // Build result: Div(new_num, new_den)
        let result = ctx.add(Expr::Div(new_num, new_den));

        // Only return if we actually changed something
        if result != expr {
            return Some(Rewrite::new(result)
                .desc("Cancel common factor from Add/Add fraction"));
        }

        None
    }
);

// Atomized rule for quotient of powers: a^n / a^m = a^(n-m)
// This is separated from CancelCommonFactorsRule for pedagogical clarity
define_rule!(
    QuotientOfPowersRule,
    "Quotient of Powers",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;
        use cas_ast::views::FractionParts;

        // Capture domain mode for cancellation decisions
        let domain_mode = parent_ctx.domain_mode();

        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);
        // Extract Pow fields from num and den via ref-and-copy
        let num_pow = match ctx.get(num) {
            Expr::Pow(b, e) => Some((*b, *e)),
            _ => None,
        };
        let den_pow = match ctx.get(den) {
            Expr::Pow(b, e) => Some((*b, *e)),
            _ => None,
        };

        // Case 1: a^n / a^m where both are Pow
        if let (Some((b_n, e_n)), Some((b_d, e_d))) = (num_pow, den_pow) {
            // Check same base
            if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal {
                // Check if exponents are numeric (so we can subtract)
                if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d)) {
                    // Only handle fractional exponents here - integer case is in CancelCommonFactors
                    if n.is_integer() && m.is_integer() {
                        return None;
                    }

                    let diff = n - m;
                    if diff.is_zero() {
                        // a^n / a^n = 1
                        // DOMAIN GATE: check if base is provably non-zero
                        let decision = crate::domain_oracle::oracle_allows_with_hint(
                            ctx,
                            domain_mode,
                            parent_ctx.value_domain(),
                            &Predicate::NonZero(b_n),
                            "Quotient of Powers",
                        );
                        if !decision.allow {
                            return None; // In Strict mode, don't cancel unknown factors
                        }
                        return Some(Rewrite::new(ctx.num(1)).desc("a^n / a^n = 1"));
                    } else if diff.is_one() {
                        // Result is just the base
                        return Some(Rewrite::new(b_n).desc("a^n / a^m = a^(n-m)"));
                    } else {
                        // Guard: Don't produce negative fractional exponents (anti-pattern for rationalization)
                        // E.g., sqrt(x)/x should NOT become x^(-1/2) as it undoes rationalization
                        if diff < num_rational::BigRational::zero() && !diff.is_integer() {
                            return None;
                        }
                        let new_exp = ctx.add(Expr::Number(diff));
                        let new_expr = ctx.add(Expr::Pow(b_n, new_exp));
                        return Some(Rewrite::new(new_expr).desc("a^n / a^m = a^(n-m)"));
                    }
                }
            }
        }

        // Case 2: a^n / a (denominator has implicit exponent 1)
        if let Some((b_n, e_n)) = num_pow {
            if crate::ordering::compare_expr(ctx, b_n, den) == std::cmp::Ordering::Equal {
                if let Expr::Number(n) = ctx.get(e_n) {
                    if !n.is_integer() {
                        let new_exp_val = n - num_rational::BigRational::one();
                        // Guard: Don't produce negative fractional exponents
                        if new_exp_val < num_rational::BigRational::zero() {
                            return None;
                        }
                        if new_exp_val.is_one() {
                            return Some(Rewrite::new(b_n).desc("a^n / a = a^(n-1)"));
                        } else {
                            let new_exp = ctx.add(Expr::Number(new_exp_val));
                            let new_expr = ctx.add(Expr::Pow(b_n, new_exp));
                            return Some(Rewrite::new(new_expr).desc("a^n / a = a^(n-1)"));
                        }
                    }
                }
            }
        }

        // Case 3: a / a^m (numerator has implicit exponent 1)
        if let Some((b_d, e_d)) = den_pow {
            if crate::ordering::compare_expr(ctx, num, b_d) == std::cmp::Ordering::Equal {
                if let Expr::Number(m) = ctx.get(e_d) {
                    if !m.is_integer() {
                        let new_exp_val = num_rational::BigRational::one() - m;
                        // Guard: Don't produce negative fractional exponents
                        // This would undo rationalization: sqrt(x)/x should stay as-is, NOT become x^(-1/2)
                        if new_exp_val < num_rational::BigRational::zero() {
                            return None;
                        }
                        let new_exp = ctx.add(Expr::Number(new_exp_val));
                        let new_expr = ctx.add(Expr::Pow(num, new_exp));
                        return Some(Rewrite::new(new_expr).desc("a / a^m = a^(1-m)"));
                    }
                }
            }
        }

        None
    }
);

define_rule!(
    PullConstantFromFractionRule,
    "Pull Constant From Fraction",
    |ctx, expr| {
        // NOTE: Keep simple Div detection to avoid infinite loop with Combine Like Terms
        // when detecting Neg(Div(...)) as a fraction
        let (n, d) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // Extract Mul or Neg shape from numerator via ref-and-copy
        enum NumShape {
            Mul(ExprId, ExprId),
            Neg(ExprId),
            Other,
        }
        let num_shape = match ctx.get(n) {
            Expr::Mul(l, r) => NumShape::Mul(*l, *r),
            Expr::Neg(inner) => NumShape::Neg(*inner),
            _ => NumShape::Other,
        };
        if let NumShape::Mul(l, r) = num_shape {
            // Check if l or r is a number/constant
            let l_is_const = matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_));
            let r_is_const = matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_));

            if l_is_const {
                // (c * x) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(r, d));
                let new_expr = mul2_raw(ctx, l, div);
                return Some(Rewrite::new(new_expr).desc("Pull constant from numerator"));
            } else if r_is_const {
                // (x * c) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(l, d));
                let new_expr = mul2_raw(ctx, r, div);
                return Some(Rewrite::new(new_expr).desc("Pull constant from numerator"));
            }
        }
        // Also handle Neg: (-x) / y -> -1 * (x / y)
        if let NumShape::Neg(inner) = num_shape {
            let minus_one = ctx.num(-1);
            let div = ctx.add(Expr::Div(inner, d));
            let new_expr = mul2_raw(ctx, minus_one, div);
            return Some(Rewrite::new(new_expr).desc("Pull negation from numerator"));
        }
        None
    }
);

define_rule!(
    FactorBasedLCDRule,
    "Factor-Based LCD",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Normalize a binomial to canonical form: (a-b) where a < b alphabetically
        // Returns (canonical_expr, sign_flip) where sign_flip is true if we negated
        let normalize_binomial = |ctx: &mut Context, e: ExprId| -> (ExprId, bool) {
            let add_parts = match ctx.get(e) {
                Expr::Add(l, r) => Some((*l, *r)),
                _ => None,
            };
            let sub_parts = match ctx.get(e) {
                Expr::Sub(l, r) => Some((*l, *r)),
                _ => None,
            };

            if let Some((l, r)) = add_parts {
                let neg_inner = match ctx.get(r) {
                    Expr::Neg(inner) => Some(*inner),
                    _ => None,
                };
                if let Some(inner) = neg_inner {
                    // Form: l + (-inner) = l - inner
                    if compare_expr(ctx, l, inner) == Ordering::Less {
                        (e, false) // Already canonical
                    } else {
                        // Create: -(inner - l) = (l - inner) negated
                        let neg_l = ctx.add(Expr::Neg(l));
                        let canonical = ctx.add(Expr::Add(inner, neg_l));
                        (canonical, true)
                    }
                } else {
                    (e, false) // Not a subtraction pattern
                }
            } else if let Some((l, r)) = sub_parts {
                if compare_expr(ctx, l, r) == Ordering::Less {
                    (e, false)
                } else {
                    let canonical = ctx.add(Expr::Sub(r, l));
                    (canonical, true)
                }
            } else {
                (e, false)
            }
        };

        // Extract factors from a product expression
        let get_factors = |ctx: &Context, e: ExprId| -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                match ctx.get(curr) {
                    Expr::Mul(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    _ => factors.push(curr),
                }
            }
            factors
        };

        // Check if expression is a binomial (Add with Neg or Sub)
        let is_binomial = |ctx: &Context, e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Add(_, r) => matches!(ctx.get(*r), Expr::Neg(_)),
                Expr::Sub(_, _) => true,
                _ => false,
            }
        };

        // Check if two expressions are equal (by compare_expr)
        let expr_eq = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
            compare_expr(ctx, a, b) == Ordering::Equal
        };

        // ===== Main Logic =====

        // Collect all terms from the Add tree
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(curr) = stack.pop() {
            match ctx.get(curr) {
                Expr::Add(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                _ => terms.push(curr),
            }
        }

        // Need at least 3 fractions - AddFractionsRule handles 2-fraction cases
        if terms.len() < 3 {
            return None;
        }

        // Extract (numerator, denominator) from each fraction
        let mut fractions: Vec<(ExprId, ExprId)> = Vec::new();
        for term in &terms {
            match ctx.get(*term) {
                Expr::Div(num, den) => fractions.push((*num, *den)),
                _ => return None, // Not all terms are fractions
            }
        }

        // For each denominator, extract and normalize binomial factors
        // Store: Vec<(canonical_factor, sign_flip)> for each fraction
        let mut all_factor_sets: Vec<Vec<(ExprId, bool)>> = Vec::new();

        for (_, den) in &fractions {
            let raw_factors = get_factors(ctx, *den);

            // All factors must be binomials for this rule to apply
            let mut normalized = Vec::new();
            for f in raw_factors {
                if !is_binomial(ctx, f) {
                    return None;
                }
                let (canonical, flipped) = normalize_binomial(ctx, f);
                normalized.push((canonical, flipped));
            }

            if normalized.is_empty() {
                return None;
            }

            all_factor_sets.push(normalized);
        }

        // Collect all unique canonical factors (the LCD factors)
        let mut unique_factors: Vec<ExprId> = Vec::new();
        for factor_set in &all_factor_sets {
            for (canonical, _) in factor_set {
                let exists = unique_factors.iter().any(|u| expr_eq(ctx, *u, *canonical));
                if !exists {
                    unique_factors.push(*canonical);
                }
            }
        }

        // Skip if all fractions already have the same denominator
        let all_same = all_factor_sets.iter().all(|fs| {
            fs.len() == unique_factors.len()
                && unique_factors
                    .iter()
                    .all(|uf| fs.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf)))
        });
        if all_same && fractions.len() == terms.len() {
            // If fractions share all factors (same LCD), AddFractionsRule handles it
            return None;
        }

        // Build LCD as product of all unique factors
        let lcd = if unique_factors.len() == 1 {
            unique_factors[0]
        } else {
            let (&first, rest) = unique_factors.split_first()?;
            rest.iter()
                .copied()
                .fold(first, |acc, f| mul2_raw(ctx, acc, f))
        };

        // For each fraction, compute numerator contribution
        let mut numerator_terms: Vec<ExprId> = Vec::new();

        for (i, (num, _den)) in fractions.iter().enumerate() {
            let factor_set = &all_factor_sets[i];

            // Compute overall sign from normalization
            let sign_flips: usize = factor_set.iter().filter(|(_, f)| *f).count();
            let is_negative = sign_flips % 2 == 1;

            // Find missing factors (in unique but not in this denominator)
            let mut missing: Vec<ExprId> = Vec::new();
            for uf in &unique_factors {
                let present = factor_set.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf));
                if !present {
                    missing.push(*uf);
                }
            }

            // Multiply numerator by all missing factors
            let mut contribution = *num;
            for mf in missing {
                contribution = mul2_raw(ctx, contribution, mf);
            }

            // Apply sign
            if is_negative {
                contribution = ctx.add(Expr::Neg(contribution));
            }

            numerator_terms.push(contribution);
        }

        // Sum all numerator contributions
        let total_num = if numerator_terms.len() == 1 {
            numerator_terms[0]
        } else {
            let (&first, rest) = numerator_terms.split_first()?;
            rest.iter()
                .copied()
                .fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
        };

        // Create the combined fraction
        let new_expr = ctx.add(Expr::Div(total_num, lcd));

        Some(Rewrite::new(new_expr).desc("Combine fractions with factor-based LCD"))
    }
);

// =============================================================================
// DivExpandToCancelRule: Cancel fractions by polynomial comparison
// =============================================================================
//
// When structural factor-based cancellation fails (CancelCommonFactorsRule,
// DivAddSymmetricFactorRule, etc.), this rule converts both numerator and
// denominator to canonical polynomial form and checks equality.
//
// This catches cases like:
//   (a² + b²)(x² + y²) / (a²x² + a²y² + b²x² + b²y²)  → 1
//   (1 + x) / ((1 + x^(1/3))(1 - x^(1/3) + x^(2/3)))   → 1
//
// Strategy:
//   1. Try poly_eq (MultiPoly) — handles pure polynomial expressions (no context mutation)
//   2. Fall back to expand() on a CLONED context — handles fractional exponents
//
// Safety:
//   - Budget-guarded: only fires when total node count is small enough
//   - Only fires when the expression contains expandable products (Mul+Add)
//   - poly_eq is read-only; expand fallback uses a cloned context
// =============================================================================
define_rule!(
    DivExpandToCancelRule,
    "Expand to Cancel Fraction",
    |ctx, expr| {
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Guard: skip trivial cases — both sides must be non-leaf
        let num_is_leaf = matches!(
            ctx.get(num),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        );
        let den_is_leaf = matches!(
            ctx.get(den),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        );
        if num_is_leaf && den_is_leaf {
            return None;
        }

        // Strategy 0: Opaque function substitution for polynomial cancellation.
        // If num/den contain shared function calls (arctan(u), sin(u), etc.),
        // substitute them with fresh variables, simplify the resulting polynomial
        // fraction, then substitute back. This handles cases like:
        //   (arctan(u)^2 - 1)/(arctan(u) - 1) → arctan(u)+1
        //   (sin(u)^2 - 4)/(sin(u) + 2) → sin(u) - 2
        // NOTE: Must run BEFORE contains_expandable guard which would reject
        // these cases since they have no expandable products.
        {
            use std::cell::Cell;
            thread_local! {
                static OPAQUE_SUB_DEPTH: Cell<u32> = const { Cell::new(0) };
            }

            let depth = OPAQUE_SUB_DEPTH.with(|c| c.get());
            if depth == 0 {
                fn collect_func_calls(
                    ctx: &cas_ast::Context,
                    expr: ExprId,
                    out: &mut Vec<ExprId>,
                    depth: usize,
                ) {
                    if depth > 4 {
                        return;
                    }
                    match ctx.get(expr) {
                        Expr::Function(_, _) => {
                            out.push(expr);
                        }
                        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                            collect_func_calls(ctx, *l, out, depth + 1);
                            collect_func_calls(ctx, *r, out, depth + 1);
                        }
                        Expr::Pow(base, exp) => {
                            collect_func_calls(ctx, *base, out, depth + 1);
                            collect_func_calls(ctx, *exp, out, depth + 1);
                        }
                        Expr::Neg(inner) => {
                            collect_func_calls(ctx, *inner, out, depth + 1);
                        }
                        _ => {}
                    }
                }

                let mut num_calls = Vec::new();
                let mut den_calls = Vec::new();
                collect_func_calls(ctx, num, &mut num_calls, 0);
                collect_func_calls(ctx, den, &mut den_calls, 0);

                let mut shared: Vec<(ExprId, ExprId)> = Vec::new();
                let mut used_den = vec![false; den_calls.len()];
                for &nc in &num_calls {
                    for (j, &dc) in den_calls.iter().enumerate() {
                        if !used_den[j]
                            && crate::ordering::compare_expr(ctx, nc, dc)
                                == std::cmp::Ordering::Equal
                        {
                            shared.push((nc, dc));
                            used_den[j] = true;
                            break;
                        }
                    }
                }

                if !shared.is_empty() && shared.len() <= 3 {
                    let mut sub_num = num;
                    let mut sub_den = den;
                    let mut temp_vars = Vec::new();

                    for (i, (num_call, den_call)) in shared.iter().enumerate() {
                        let temp_name = format!("__opq{}", i);
                        let temp_var = ctx.var(&temp_name);
                        let opts = crate::substitute::SubstituteOptions {
                            power_aware: true,
                            ..Default::default()
                        };
                        sub_num = crate::substitute::substitute_power_aware(
                            ctx, sub_num, *num_call, temp_var, opts,
                        );
                        sub_den = crate::substitute::substitute_power_aware(
                            ctx, sub_den, *den_call, temp_var, opts,
                        );
                        temp_vars.push((*num_call, temp_var));
                    }

                    // Guard: if both sides are now bare variables/leaves, this is a
                    // trivial f(x)/f(x) case that should be handled by normal
                    // cancellation rules (which respect Strict domain mode).
                    let sub_num_is_leaf = matches!(
                        ctx.get(sub_num),
                        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_)
                    );
                    let sub_den_is_leaf = matches!(
                        ctx.get(sub_den),
                        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_)
                    );
                    if sub_num_is_leaf && sub_den_is_leaf {
                        // Skip: let normal cancellation handle it
                    } else {
                        let sub_frac = ctx.add(Expr::Div(sub_num, sub_den));

                        OPAQUE_SUB_DEPTH.with(|c| c.set(depth + 1));
                        let mut simplifier = crate::Simplifier::with_default_rules();
                        simplifier.context = ctx.clone();
                        let (simplified, _) = simplifier.simplify(sub_frac);
                        OPAQUE_SUB_DEPTH.with(|c| c.set(depth));

                        let is_still_div =
                            matches!(simplifier.context.get(simplified), Expr::Div(_, _));
                        if !is_still_div {
                            let mut final_result = simplified;
                            for (call_id, temp_var) in &temp_vars {
                                let opts = crate::substitute::SubstituteOptions::default();
                                final_result = crate::substitute::substitute_power_aware(
                                    &mut simplifier.context,
                                    final_result,
                                    *temp_var,
                                    *call_id,
                                    opts,
                                );
                            }
                            *ctx = simplifier.context;
                            return Some(
                                Rewrite::new(final_result)
                                    .desc("Polynomial division with opaque substitution"),
                            );
                        }
                    } // else
                }
            }
        }

        // Guard: at least one side must contain an expandable product
        // (Mul where at least one child is Add/Sub, or Pow(Add, n))
        if !contains_expandable(ctx, num) && !contains_expandable(ctx, den) {
            return None;
        }

        // Budget guard: don't attempt expensive expansions on large expressions
        let num_nodes = crate::helpers::node_count_tree(ctx, num);
        let den_nodes = crate::helpers::node_count_tree(ctx, den);
        if num_nodes + den_nodes > 150 {
            return None;
        }

        // Strategy 1: Try polynomial comparison (read-only, no context mutation)
        // multipoly_from_expr internally expands Mul(Add,Add) in polynomial space
        {
            use crate::multipoly::{multipoly_from_expr, PolyBudget};
            let budget = PolyBudget {
                max_terms: 200,
                max_total_degree: 12,
                max_pow_exp: 6,
            };

            if let (Ok(p_num), Ok(p_den)) = (
                multipoly_from_expr(ctx, num, &budget),
                multipoly_from_expr(ctx, den, &budget),
            ) {
                if p_num == p_den {
                    return Some(
                        Rewrite::new(ctx.num(1)).desc("Expanded numerator equals denominator"),
                    );
                }
                // Polynomials differ — no cancellation possible
                return None;
            }
        }

        // Strategy 2: Expressions contain non-polynomial parts (fractional exponents, trig, etc.)
        // expand() distributes products but does NOT combine terms like x^(1/3)*x^(2/3) → x.
        // So we expand both sides on a CLONED context, then simplify each expanded side
        // INDIVIDUALLY (not as a quotient Div — that would trigger this rule recursively).
        // After simplification, we compare the results via poly_eq.
        //
        // GUARD: Use a thread-local counter to prevent nested invocations. The inner
        // Simplifier would trigger this rule again on any Div subexpressions, potentially
        // creating O(n²) nested chains that cause timeouts.
        {
            use std::cell::Cell;
            thread_local! {
                static EXPAND_CANCEL_DEPTH: Cell<u32> = const { Cell::new(0) };
            }
            let depth = EXPAND_CANCEL_DEPTH.with(|c| c.get());
            if depth > 0 {
                return None; // Already inside an inner Simplifier — skip Strategy 2
            }

            let mut ctx_clone = ctx.clone();
            let expanded_num = crate::expand::expand(&mut ctx_clone, num);
            let expanded_den = crate::expand::expand(&mut ctx_clone, den);

            // Check if expansion changed anything — if not, skip
            let num_changed = crate::ordering::compare_expr(&ctx_clone, num, expanded_num)
                != std::cmp::Ordering::Equal;
            let den_changed = crate::ordering::compare_expr(&ctx_clone, den, expanded_den)
                != std::cmp::Ordering::Equal;
            if !num_changed && !den_changed {
                return None;
            }

            // Simplify each expanded side individually with recursion guard active.
            EXPAND_CANCEL_DEPTH.with(|c| c.set(depth + 1));
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = ctx_clone;
            let (simplified_num, _) = simplifier.simplify(expanded_num);
            let (simplified_den, _) = simplifier.simplify(expanded_den);
            EXPAND_CANCEL_DEPTH.with(|c| c.set(depth)); // restore

            // Compare via poly_eq (order-independent polynomial comparison)
            use crate::multipoly::{multipoly_from_expr, PolyBudget};
            let budget = PolyBudget {
                max_terms: 200,
                max_total_degree: 12,
                max_pow_exp: 6,
            };
            if let (Ok(p_num), Ok(p_den)) = (
                multipoly_from_expr(&simplifier.context, simplified_num, &budget),
                multipoly_from_expr(&simplifier.context, simplified_den, &budget),
            ) {
                if p_num == p_den {
                    return Some(
                        Rewrite::new(ctx.num(1)).desc("Expanded numerator equals denominator"),
                    );
                }
            }

            // Structural comparison fallback
            if crate::ordering::compare_expr(&simplifier.context, simplified_num, simplified_den)
                == std::cmp::Ordering::Equal
            {
                return Some(
                    Rewrite::new(ctx.num(1)).desc("Expanded numerator equals denominator"),
                );
            }
        }

        None
    }
);

/// Check whether an expression tree contains an expandable product:
/// - Mul(_, Add/Sub) or Mul(Add/Sub, _)
/// - Pow(Add/Sub, integer ≥ 2)
///   Searches up to 3 levels deep to keep it cheap.
fn contains_expandable(ctx: &Context, expr: ExprId) -> bool {
    contains_expandable_depth(ctx, expr, 0)
}

fn contains_expandable_depth(ctx: &Context, expr: ExprId, depth: usize) -> bool {
    if depth > 3 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            let l_is_sum = matches!(ctx.get(*l), Expr::Add(_, _) | Expr::Sub(_, _));
            let r_is_sum = matches!(ctx.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));
            if l_is_sum || r_is_sum {
                return true;
            }
            // Recurse into children
            contains_expandable_depth(ctx, *l, depth + 1)
                || contains_expandable_depth(ctx, *r, depth + 1)
        }
        Expr::Pow(b, e) => {
            if matches!(ctx.get(*b), Expr::Add(_, _) | Expr::Sub(_, _)) {
                if let Expr::Number(n) = ctx.get(*e) {
                    if n.is_integer() && *n >= num_rational::BigRational::from_integer(2.into()) {
                        return true;
                    }
                }
            }
            contains_expandable_depth(ctx, *b, depth + 1)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            contains_expandable_depth(ctx, *l, depth + 1)
                || contains_expandable_depth(ctx, *r, depth + 1)
        }
        Expr::Div(l, r) => {
            contains_expandable_depth(ctx, *l, depth + 1)
                || contains_expandable_depth(ctx, *r, depth + 1)
        }
        Expr::Neg(e) => contains_expandable_depth(ctx, *e, depth + 1),
        _ => false,
    }
}
