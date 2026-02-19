//! Polynomial identity detection, heuristic normalization, and small binomial expansion rules.
//!
//! These rules are extracted from `expansion.rs` to keep the module focused on
//! core binomial/multinomial expansion logic.

use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::multinomial_expand::{try_expand_multinomial_direct, MultinomialExpandBudget};
use cas_math::multipoly::{MultiPoly, PolyBudget};
use num_traits::Signed;

// =============================================================================
// PolynomialIdentityZeroRule
// =============================================================================

/// PolynomialIdentityZeroRule: Always-on polynomial identity detector
/// Converts expressions to MultiPoly form and checks if result is 0.
/// Priority 90 (lower than AutoExpandSubCancelRule at 95 to avoid duplicate work)
pub struct PolynomialIdentityZeroRule;

impl PolynomialIdentityZeroRule {
    /// Budget limits for polynomial conversion
    /// V2.15.8: Increased max_pow_exp to 6 for binomial identities like (x+1)^5 - expansion = 0
    fn poly_budget() -> PolyBudget {
        PolyBudget {
            max_terms: 50,       // Max monomials in result
            max_total_degree: 6, // Max total degree (covers up to n=6)
            max_pow_exp: 6,      // Max exponent in Pow nodes
        }
    }

    /// Quick check: does expression look polynomial-like and worth checking?
    /// Avoids expensive conversion for obviously non-polynomial expressions.
    fn is_polynomial_candidate(ctx: &Context, expr: ExprId) -> bool {
        Self::is_polynomial_candidate_inner(ctx, expr, 0)
    }

    fn is_polynomial_candidate_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 30 {
            return false; // Too deep
        }

        match ctx.get(expr) {
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
            // Opaque function calls are valid polynomial "leaves" (treated as
            // atomic variables after substitution)
            Expr::Function(_, _) => true,
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::is_polynomial_candidate_inner(ctx, *l, depth + 1)
                    && Self::is_polynomial_candidate_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) => Self::is_polynomial_candidate_inner(ctx, *inner, depth + 1),
            Expr::Pow(base, exp) => {
                // Only integer exponents, and check base
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && !n.is_negative() {
                        use num_traits::ToPrimitive;
                        if let Some(e) = n.to_integer().to_u32() {
                            if e <= 6 {
                                // V2.15.8: Extended budget for binomial identities
                                return Self::is_polynomial_candidate_inner(ctx, *base, depth + 1);
                            }
                        }
                    }
                }
                // e^(k·u) is an opaque polynomial leaf (treated as atom after
                // exponential substitution in try_opaque_zero)
                if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) {
                    return true;
                }
                false
            }
            _ => false, // Division, etc. are not polynomial
        }
    }

    /// Convert expression to MultiPoly (reusing AutoExpandSubCancelRule's method)
    fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        let poly =
            cas_math::poly_convert::try_multipoly_from_expr_with_var_limit(ctx, id, budget, 4)?;
        *vars = poly.vars.clone();
        Some(poly)
    }

    /// Collect all function calls in an expression (e.g., sin(u), arctan(u))
    fn collect_func_calls(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>, depth: usize) {
        if depth > 4 {
            return;
        }
        match ctx.get(expr) {
            Expr::Function(_, _) => {
                out.push(expr);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::collect_func_calls(ctx, *l, out, depth + 1);
                Self::collect_func_calls(ctx, *r, out, depth + 1);
            }
            Expr::Pow(base, exp) => {
                Self::collect_func_calls(ctx, *base, out, depth + 1);
                Self::collect_func_calls(ctx, *exp, out, depth + 1);
            }
            Expr::Neg(inner) => {
                Self::collect_func_calls(ctx, *inner, out, depth + 1);
            }
            _ => {}
        }
    }

    /// Deduplicate function calls by structural comparison
    fn dedup_func_calls(ctx: &Context, calls: &[ExprId]) -> Vec<ExprId> {
        let mut unique = Vec::new();
        for &call in calls {
            let already = unique
                .iter()
                .any(|&u| crate::ordering::compare_expr(ctx, call, u) == std::cmp::Ordering::Equal);
            if !already {
                unique.push(call);
            }
        }
        unique
    }

    // ── Exponential atom detection ─────────────────────────────────────

    /// Collect all `e^(arg)` exponent arguments from an expression.
    /// For `e^(3u) + e^u + 1`, collects `[3u, u]`.
    fn collect_exp_exponents(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>, depth: usize) {
        if depth > 30 {
            return;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) {
                    out.push(*exp);
                } else {
                    // Check base and integer exponent children
                    Self::collect_exp_exponents(ctx, *base, out, depth + 1);
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::collect_exp_exponents(ctx, *l, out, depth + 1);
                Self::collect_exp_exponents(ctx, *r, out, depth + 1);
            }
            Expr::Neg(inner) => {
                Self::collect_exp_exponents(ctx, *inner, out, depth + 1);
            }
            _ => {}
        }
    }

    /// Try to split `expr` into `(k, rest)` where `expr = k * rest` and k is
    /// a positive integer. Returns `(1, expr)` if no integer factor found.
    fn extract_integer_factor(ctx: &Context, expr: ExprId) -> (u32, ExprId) {
        match ctx.get(expr) {
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = ctx.get(*l) {
                    if n.is_integer() && n.is_positive() {
                        use num_traits::ToPrimitive;
                        if let Some(k) = n.to_integer().to_u32() {
                            if k <= 6 {
                                return (k, *r);
                            }
                        }
                    }
                }
                if let Expr::Number(n) = ctx.get(*r) {
                    if n.is_integer() && n.is_positive() {
                        use num_traits::ToPrimitive;
                        if let Some(k) = n.to_integer().to_u32() {
                            if k <= 6 {
                                return (k, *l);
                            }
                        }
                    }
                }
                (1, expr)
            }
            _ => (1, expr),
        }
    }

    /// Given a list of exponent arguments (from `e^arg` atoms), find a common
    /// base such that every exponent is `k * base` for some positive integer k.
    /// Returns the base ExprId if found.
    fn find_exp_base(ctx: &Context, exponents: &[ExprId]) -> Option<ExprId> {
        if exponents.is_empty() {
            return None;
        }

        // Extract (k, rest) for each exponent
        let factored: Vec<(u32, ExprId)> = exponents
            .iter()
            .map(|&e| Self::extract_integer_factor(ctx, e))
            .collect();

        // All 'rest' parts must be structurally equal
        let base = factored[0].1;
        for &(_, rest) in &factored[1..] {
            if crate::ordering::compare_expr(ctx, base, rest) != std::cmp::Ordering::Equal {
                return None;
            }
        }
        Some(base)
    }

    /// Substitute all `e^(k*base)` nodes with `var^k` in the expression tree.
    fn substitute_exp_atoms(
        ctx: &mut Context,
        expr: ExprId,
        exp_base: ExprId,
        replacement_var: ExprId,
        depth: usize,
    ) -> ExprId {
        if depth > 30 {
            return expr;
        }
        match ctx.get(expr).clone() {
            Expr::Pow(base, exp)
                if matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E)) =>
            {
                let (k, rest) = Self::extract_integer_factor(ctx, exp);
                if crate::ordering::compare_expr(ctx, rest, exp_base) == std::cmp::Ordering::Equal {
                    if k == 1 {
                        return replacement_var;
                    }
                    let exp_k = ctx.num(k as i64);
                    return ctx.add(Expr::Pow(replacement_var, exp_k));
                }
                // Not matching our pattern, return as-is
                expr
            }
            Expr::Add(l, r) => {
                let new_l =
                    Self::substitute_exp_atoms(ctx, l, exp_base, replacement_var, depth + 1);
                let new_r =
                    Self::substitute_exp_atoms(ctx, r, exp_base, replacement_var, depth + 1);
                if new_l == l && new_r == r {
                    expr
                } else {
                    ctx.add(Expr::Add(new_l, new_r))
                }
            }
            Expr::Sub(l, r) => {
                let new_l =
                    Self::substitute_exp_atoms(ctx, l, exp_base, replacement_var, depth + 1);
                let new_r =
                    Self::substitute_exp_atoms(ctx, r, exp_base, replacement_var, depth + 1);
                if new_l == l && new_r == r {
                    expr
                } else {
                    ctx.add(Expr::Sub(new_l, new_r))
                }
            }
            Expr::Mul(l, r) => {
                let new_l =
                    Self::substitute_exp_atoms(ctx, l, exp_base, replacement_var, depth + 1);
                let new_r =
                    Self::substitute_exp_atoms(ctx, r, exp_base, replacement_var, depth + 1);
                if new_l == l && new_r == r {
                    expr
                } else {
                    ctx.add(Expr::Mul(new_l, new_r))
                }
            }
            Expr::Neg(inner) => {
                let new_inner =
                    Self::substitute_exp_atoms(ctx, inner, exp_base, replacement_var, depth + 1);
                if new_inner == inner {
                    expr
                } else {
                    ctx.add(Expr::Neg(new_inner))
                }
            }
            Expr::Pow(base, exp) => {
                // Non-exponential power: recurse into base (exp is integer)
                let new_base =
                    Self::substitute_exp_atoms(ctx, base, exp_base, replacement_var, depth + 1);
                if new_base == base {
                    expr
                } else {
                    ctx.add(Expr::Pow(new_base, exp))
                }
            }
            _ => expr,
        }
    }

    /// Try to prove the expression is zero by substituting opaque function calls
    /// with temporary variables. Returns `Some(PolynomialProofData)` with the
    /// substitution mapping and LHS/RHS normal forms if the identity is confirmed.
    ///
    /// Also handles exponential polynomials: expressions of the form
    /// `e^(k·u)` are treated as `t^k` where `t = e^u`.
    ///
    /// Display expressions are built in the main `ctx` using human-readable
    /// variable names (`t₀`, `t₁`, …) so the didactic renderer can show
    /// the actual expanded polynomial forms.
    fn try_opaque_zero(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<cas_math::multipoly_display::PolynomialProofData> {
        // ── Phase 1: collect opaque atoms ──────────────────────────────
        let mut calls = Vec::new();
        Self::collect_func_calls(ctx, expr, &mut calls, 0);
        let unique_calls = Self::dedup_func_calls(ctx, &calls);

        // Also check for exponential atoms: e^(k·base)
        let mut exp_exponents = Vec::new();
        Self::collect_exp_exponents(ctx, expr, &mut exp_exponents, 0);
        let exp_base = if !exp_exponents.is_empty() {
            Self::find_exp_base(ctx, &exp_exponents)
        } else {
            None
        };

        let total_atoms = unique_calls.len() + if exp_base.is_some() { 1 } else { 0 };
        if total_atoms == 0 || total_atoms > 4 {
            return None;
        }

        // ── Phase 2: build display name generator ──────────────────────
        const SUBSCRIPTS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
        let display_name = |i: usize| -> String {
            if total_atoms == 1 {
                "t".to_string()
            } else if i < 10 {
                format!("t{}", SUBSCRIPTS[i])
            } else {
                format!("t{}", i)
            }
        };

        // ── Phase 3: substitute in tmp context ─────────────────────────
        let mut tmp_ctx = ctx.clone();
        let mut sub_expr = expr;
        let mut substitutions: Vec<(String, ExprId)> = Vec::new();
        let mut atom_idx = 0;

        // 3a: Substitute exponential atoms first (e^(k·base) → t^k)
        if let Some(base) = exp_base {
            let temp_name = format!("__opq{}", atom_idx);
            let temp_var = tmp_ctx.var(&temp_name);
            sub_expr = Self::substitute_exp_atoms(&mut tmp_ctx, sub_expr, base, temp_var, 0);
            // Build the display e^base expression for the proof data
            let e_const = ctx.add(Expr::Constant(cas_ast::Constant::E));
            let exp_display = ctx.add(Expr::Pow(e_const, base));
            substitutions.push((display_name(atom_idx), exp_display));
            atom_idx += 1;
        }

        // 3b: Substitute function calls (sin(u) → t, etc.)
        for &call_id in &unique_calls {
            let temp_name = format!("__opq{}", atom_idx);
            let temp_var = tmp_ctx.var(&temp_name);
            let opts = crate::substitute::SubstituteOptions {
                power_aware: true,
                ..Default::default()
            };
            sub_expr = crate::substitute::substitute_power_aware(
                &mut tmp_ctx,
                sub_expr,
                call_id,
                temp_var,
                opts,
            );
            substitutions.push((display_name(atom_idx), call_id));
            atom_idx += 1;
        }

        // ── Phase 4: convert to multipoly and check if zero ────────────
        let budget = Self::poly_budget();
        let mut vars = Vec::new();
        let poly = Self::expr_to_multipoly(&tmp_ctx, sub_expr, &mut vars, &budget)?;
        if !poly.is_zero() {
            return None;
        }

        // ── Phase 5: build display expression in main context ──────────
        let mut display_expr = expr;
        let mut disp_idx = 0;

        // 5a: Substitute exponential atoms for display
        if let Some(base) = exp_base {
            let disp_var = ctx.var(&display_name(disp_idx));
            display_expr = Self::substitute_exp_atoms(ctx, display_expr, base, disp_var, 0);
            disp_idx += 1;
        }

        // 5b: Substitute function calls for display
        for &call_id in &unique_calls {
            let disp_var = ctx.var(&display_name(disp_idx));
            let opts = crate::substitute::SubstituteOptions {
                power_aware: true,
                ..Default::default()
            };
            display_expr = crate::substitute::substitute_power_aware(
                ctx,
                display_expr,
                call_id,
                disp_var,
                opts,
            );
            disp_idx += 1;
        }

        let display_vars: Vec<String> = vars
            .iter()
            .map(|v| {
                if let Some(idx_str) = v.strip_prefix("__opq") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        return display_name(idx);
                    }
                }
                v.clone()
            })
            .collect();

        // Compute the expanded form (each product expanded, but not cancelled)
        let expanded_form_expr =
            cas_math::multipoly_display::expand_additive_terms(ctx, display_expr, &display_vars);

        let mut proof = cas_math::multipoly_display::PolynomialProofData {
            monomials: 0,
            degree: 0,
            vars: display_vars,
            normal_form_expr: Some(display_expr),
            expanded_form_expr,
            lhs_stats: None,
            rhs_stats: None,
            opaque_substitutions: Vec::new(),
        };

        proof.opaque_substitutions = substitutions;
        Some(proof)
    }
}

impl crate::rule::Rule for PolynomialIdentityZeroRule {
    fn name(&self) -> &str {
        "Polynomial Identity"
    }

    fn priority(&self) -> i32 {
        90 // Lower than AutoExpandSubCancelRule (95) to avoid duplicate work
    }

    fn allowed_phases(&self) -> PhaseMask {
        // Also run in POST so that polynomial identities exposed AFTER Rationalize
        // (e.g., (x+1)^6 - expanded_form left behind when 1/√5 - √5/5 → 0)
        // get cancelled. TRANSFORM alone can't see these because the non-polynomial
        // terms haven't been removed yet at that point.
        PhaseMask::TRANSFORM | PhaseMask::POST
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Must be Add or Sub to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        // Quick node count check (avoid expensive conversion for huge expressions)
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 100 {
            return None; // Too big, skip
        }

        // Quick polynomial-like check
        if !Self::is_polynomial_candidate(ctx, expr) {
            return None;
        }

        // Try to convert to MultiPoly
        let budget = Self::poly_budget();
        let mut vars = Vec::new();
        let poly_opt = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget);

        // If direct multipoly conversion failed, try opaque substitution fallback
        let poly = match poly_opt {
            Some(p) => p,
            None => {
                // Expression contains function calls that multipoly can't handle.
                // Try substituting opaque calls with temp vars and check if zero.
                if let Some(proof_data) = Self::try_opaque_zero(ctx, expr) {
                    let zero = ctx.num(0);
                    return Some(
                        Rewrite::new(zero)
                            .desc("Polynomial identity (opaque substitution): cancel to 0")
                            .poly_proof(proof_data),
                    );
                }
                return None;
            }
        };

        // Check variable count
        if vars.len() > 4 {
            return None; // Too many variables
        }

        // If the result is zero, we have a polynomial identity!
        if poly.is_zero() {
            let zero = ctx.num(0);

            // Split terms into positive and negative to show LHS/RHS normal forms
            // For an expression like A + B - C - D, we show:
            //   LHS (positive): A + B expanded
            //   RHS (negative): C + D expanded
            let (positive_terms, negative_terms) = {
                let mut pos = Vec::new();
                let mut neg = Vec::new();

                // Collect all additive terms
                fn collect_terms(
                    ctx: &Context,
                    e: ExprId,
                    pos: &mut Vec<ExprId>,
                    neg: &mut Vec<ExprId>,
                ) {
                    match ctx.get(e) {
                        Expr::Add(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            collect_terms(ctx, *b, pos, neg);
                        }
                        Expr::Sub(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            // b is subtracted, so it goes to negative
                            neg.push(*b);
                        }
                        Expr::Neg(inner) => {
                            neg.push(*inner);
                        }
                        _ => {
                            pos.push(e);
                        }
                    }
                }
                collect_terms(ctx, expr, &mut pos, &mut neg);
                (pos, neg)
            };

            // Build proof data with LHS/RHS if we have both positive and negative terms
            let proof_data = if !positive_terms.is_empty() && !negative_terms.is_empty() {
                // Build polys for positive sum (LHS) and negative sum (RHS)
                let mut lhs_poly = cas_math::multipoly::MultiPoly::zero(vars.clone());
                let mut rhs_poly = cas_math::multipoly::MultiPoly::zero(vars.clone());

                // Sum positive terms - use the same vars we already collected
                for &term in &positive_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        // If same vars, can add directly
                        if term_poly.vars == lhs_poly.vars {
                            if let Ok(sum) = lhs_poly.add(&term_poly) {
                                lhs_poly = sum;
                            }
                        }
                    }
                }

                // Sum negative terms (these are the RHS that was subtracted)
                for &term in &negative_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        if term_poly.vars == rhs_poly.vars {
                            if let Ok(sum) = rhs_poly.add(&term_poly) {
                                rhs_poly = sum;
                            }
                        }
                    }
                }

                cas_math::multipoly_display::PolynomialProofData::from_identity(
                    ctx,
                    &lhs_poly,
                    &rhs_poly,
                    vars.clone(),
                )
            } else {
                // No clear LHS/RHS split
                cas_math::multipoly_display::PolynomialProofData {
                    monomials: 0,
                    degree: 0,
                    vars: vars.clone(),
                    normal_form_expr: Some(zero),
                    expanded_form_expr: None,
                    lhs_stats: None,
                    rhs_stats: None,
                    opaque_substitutions: Vec::new(),
                }
            };

            return Some(
                Rewrite::new(zero)
                    .desc("Polynomial identity: normalize and cancel to 0")
                    .poly_proof(proof_data),
            );
        }

        None
    }
}

// =============================================================================
// HeuristicPolyNormalizeAddRule
// =============================================================================

/// HeuristicPolyNormalizeAddRule: Poly-normalize sums with binomial powers
/// Priority 42 (after ExpandSmallBinomialPowRule at 40, before others)
pub struct HeuristicPolyNormalizeAddRule;

impl HeuristicPolyNormalizeAddRule {
    /// Check if expression contains Pow(Add, n) with 2 ≤ n ≤ 6
    fn contains_pow_add(ctx: &Context, expr: ExprId) -> bool {
        Self::contains_pow_add_inner(ctx, expr, 0)
    }

    fn contains_pow_add_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 20 {
            return false;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                // Check if this is Pow(Add, n) with 2 ≤ n ≤ 6 AND base is polynomial-like
                if matches!(ctx.get(*base), Expr::Add(_, _)) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && !n.is_negative() {
                            use num_traits::ToPrimitive;
                            if let Some(e) = n.to_integer().to_u32() {
                                // Must be polynomial-like base (no functions like sqrt, sin)
                                if (2..=6).contains(&e)
                                    && crate::auto_expand_scan::looks_polynomial_like(ctx, *base)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                Self::contains_pow_add_inner(ctx, *base, depth + 1)
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::contains_pow_add_inner(ctx, *l, depth + 1)
                    || Self::contains_pow_add_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) | Expr::Hold(inner) => {
                Self::contains_pow_add_inner(ctx, *inner, depth + 1)
            }
            Expr::Div(l, r) => {
                Self::contains_pow_add_inner(ctx, *l, depth + 1)
                    || Self::contains_pow_add_inner(ctx, *r, depth + 1)
            }
            Expr::Function(_, args) => args
                .iter()
                .any(|a| Self::contains_pow_add_inner(ctx, *a, depth + 1)),
            Expr::Matrix { data, .. } => data
                .iter()
                .any(|e| Self::contains_pow_add_inner(ctx, *e, depth + 1)),
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        }
    }
}

impl crate::rule::Rule for HeuristicPolyNormalizeAddRule {
    fn name(&self) -> &str {
        "Heuristic Poly Normalize"
    }

    fn priority(&self) -> i32 {
        100 // Very high priority - must process Add BEFORE children Pow are expanded
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger when heuristic_poly is On
        use crate::options::HeuristicPoly;
        if parent_ctx.heuristic_poly() != HeuristicPoly::On {
            return None;
        }

        // Skip in Solve mode
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Must be Add or Sub
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Must contain at least one Pow(Add, n) with 2 ≤ n ≤ 6
        // We process the ORIGINAL Add before children are expanded by ExpandSmallBinomialPowRule
        if !Self::contains_pow_add(ctx, expr) {
            return None;
        }

        // Quick size check
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 80 {
            return None;
        }

        // Try to convert to MultiPoly (this expands and combines terms)
        let budget = PolyBudget {
            max_terms: 40,
            max_total_degree: 6,
            max_pow_exp: 6,
        };

        let poly =
            cas_math::poly_convert::try_multipoly_from_expr_with_var_limit(ctx, expr, &budget, 4)?;

        // Check if result is reasonable
        if poly.terms.len() > 30 || poly.vars.len() > 3 {
            return None;
        }

        // If polynomial is zero, let PolynomialIdentityZeroRule handle it
        if poly.is_zero() {
            return None;
        }

        // Convert back to expression using multipoly_to_expr (produces flattened Add)
        let new_expr = cas_math::multipoly::multipoly_to_expr(&poly, ctx);

        // Don't rewrite to same expression
        if new_expr == expr {
            return None;
        }

        Some(
            Rewrite::new(new_expr)
                .desc("Expand and combine polynomial terms (heuristic)")
                .local(expr, new_expr),
        )
    }
}

// =============================================================================
// ExpandSmallBinomialPowRule
// =============================================================================
/// ExpandSmallBinomialPowRule: Always-on expansion for small binomial/trinomial powers
/// Priority 40 (before AutoExpandPowSumRule at 50, after most algebraic rules)
pub struct ExpandSmallBinomialPowRule;

impl crate::rule::Rule for ExpandSmallBinomialPowRule {
    fn name(&self) -> &str {
        "Expand Small Power"
    }

    fn priority(&self) -> i32 {
        40 // Before AutoExpandPowSumRule (50), after basic algebra
    }

    fn allowed_phases(&self) -> PhaseMask {
        // Only in TRANSFORM phase to avoid interfering with early simplification
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // V2.15.9: Check autoexpand_binomials mode (Off/On)
        use crate::options::AutoExpandBinomials;
        match parent_ctx.autoexpand_binomials() {
            AutoExpandBinomials::Off => return None, // Never expand standalone
            AutoExpandBinomials::On => {
                // Always expand (subject to budget checks below)
            }
        }

        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Skip if already in auto-expand context (let AutoExpandPowSumRule handle)
        if parent_ctx.in_auto_expand_context()
            && parent_ctx.autoexpand_binomials() != AutoExpandBinomials::On
        {
            return None;
        }

        // Pattern: Pow(base, exp)
        let (base, exp) = crate::helpers::as_pow(ctx, expr)?;

        // Very restrictive budget for automatic expansion in generic mode
        // - max_exp: 6 (binomial (x+1)^6 = 7 terms, trinomial (a+b+c)^4 = 15 terms)
        // - max_base_terms: 3 (binomial or trinomial only)
        // - max_vars: 2 (keeps output manageable)
        // - max_output_terms: 20 (strict limit to prevent bloat)
        let budget = MultinomialExpandBudget {
            max_exp: 6,
            max_base_terms: 3,
            max_vars: 2,
            max_output_terms: 20,
        };

        // try_expand_multinomial_direct already:
        // 1. Checks exponent is small positive integer
        // 2. Extracts linear terms (fails if base has functions/div)
        // 3. Estimates output terms and checks budget
        // 4. Wraps result in __hold() for anti-cycle protection
        let expanded = try_expand_multinomial_direct(ctx, base, exp, &budget)?;

        Some(
            Rewrite::new(expanded)
                .desc("Expand binomial/trinomial power")
                .local(expr, expanded),
        )
    }
}
