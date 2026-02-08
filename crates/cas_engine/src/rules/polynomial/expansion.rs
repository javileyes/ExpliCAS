//! Polynomial expansion rules: binomial expansion, auto-expand, identity detection,
//! and heuristic polynomial normalization.

use crate::build::mul2_raw;
use crate::multipoly::{MultiPoly, PolyBudget};
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive};

use super::count_additive_terms;

/// Binomial coefficient C(n, k)
pub(crate) fn binomial_coeff(n: u32, k: u32) -> u32 {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

// =============================================================================
// BinomialExpansionRule
// =============================================================================

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub struct BinomialExpansionRule;

impl crate::rule::Rule for BinomialExpansionRule {
    fn name(&self) -> &str {
        "Binomial Expansion"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Skip if expression is in canonical (elegant) form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Don't expand if this expression is protected as a sqrt-square base
        // This is set by pre-scan when this expr is inside sqrt(u²) or sqrt(u*u)
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sqrt_square_protected(expr) {
                // Protected from expansion - let sqrt(u²) → |u| shortcut fire instead
                return None;
            }
        }

        // (a + b)^n - ONLY true binomials (exactly 2 terms)
        // Extract Pow fields via ref-and-copy
        let pow_fields = match ctx.get(expr) {
            Expr::Pow(b, e) => Some((*b, *e)),
            _ => None,
        };
        if let Some((base, exp)) = pow_fields {
            // CRITICAL GUARD: Only expand if base has exactly 2 terms
            // This prevents multinomial expansion like (1 + x1 + x2 + ... + x7)^7
            // which would produce thousands of terms
            let term_count = count_additive_terms(ctx, base);
            if term_count != 2 {
                return None; // Not a binomial, skip expansion
            }

            let (a, b) = match ctx.get(base) {
                Expr::Add(a, b) => (*a, *b),
                Expr::Sub(a, b) => {
                    let b = *b;
                    let a = *a;
                    let neg_b = ctx.add(Expr::Neg(b));
                    (a, neg_b)
                }
                _ => return None,
            };

            // CLONE_OK: Exponent inspection for Neg/Number patterns
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() && !n.is_negative() {
                    if let Some(n_val) = n.to_integer().to_u32() {
                        // Only expand binomials in explicit expand mode
                        // In Standard mode, preserve structure like (x+1)^3
                        // This prevents unwanted expansion when doing poly_gcd(a*g, b*g) - g
                        if !parent_ctx.is_expand_mode() {
                            return None;
                        }

                        // Limit expansion to reasonable exponents even in expand mode
                        if (2..=20).contains(&n_val) {
                            // Expand: sum(k=0 to n) (n choose k) * a^(n-k) * b^k
                            let mut terms = Vec::new();
                            for k in 0..=n_val {
                                let coeff = binomial_coeff(n_val, k);
                                let exp_a = n_val - k;
                                let exp_b = k;

                                let term_a = if exp_a == 0 {
                                    ctx.num(1)
                                } else if exp_a == 1 {
                                    a
                                } else {
                                    let e = ctx.num(exp_a as i64);
                                    ctx.add(Expr::Pow(a, e))
                                };
                                let term_b = if exp_b == 0 {
                                    ctx.num(1)
                                } else if exp_b == 1 {
                                    b
                                } else {
                                    let e = ctx.num(exp_b as i64);
                                    ctx.add(Expr::Pow(b, e))
                                };

                                let mut term = mul2_raw(ctx, term_a, term_b);
                                if coeff > 1 {
                                    let c = ctx.num(coeff as i64);
                                    term = mul2_raw(ctx, c, term);
                                }
                                terms.push(term);
                            }

                            // Sum up terms
                            let mut expanded = terms[0];
                            for &term in terms.iter().skip(1) {
                                expanded = ctx.add(Expr::Add(expanded, term));
                            }

                            return Some(
                                Rewrite::new(expanded)
                                    .desc(format!("Expand binomial power ^{}", n_val)),
                            );
                        }
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }
}

// =============================================================================
// AutoExpandPowSumRule
// =============================================================================

/// Rule for auto-expanding cheap power-of-sum expressions within budget limits.
/// Unlike BinomialExpansionRule, this is opt-in and checks budget constraints.
///
/// Only triggers when `parent_ctx.is_auto_expand()` is true.
/// Respects budget limits: max_pow_exp, max_base_terms, max_generated_terms, max_vars.
pub struct AutoExpandPowSumRule;

impl AutoExpandPowSumRule {
    /// Count additive terms in an expression
    fn count_add_terms(ctx: &Context, expr: ExprId) -> u32 {
        match ctx.get(expr) {
            Expr::Add(l, r) => Self::count_add_terms(ctx, *l) + Self::count_add_terms(ctx, *r),
            _ => 1,
        }
    }

    /// Count unique variables in an expression
    fn count_variables(
        ctx: &Context,
        expr: ExprId,
        visited: &mut std::collections::HashSet<String>,
    ) {
        match ctx.get(expr) {
            Expr::Variable(sym_id) => {
                let name = ctx.sym_name(*sym_id).to_string();
                visited.insert(name);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                Self::count_variables(ctx, *l, visited);
                Self::count_variables(ctx, *r, visited);
            }
            Expr::Pow(b, e) => {
                Self::count_variables(ctx, *b, visited);
                Self::count_variables(ctx, *e, visited);
            }
            Expr::Neg(e) | Expr::Hold(e) => {
                Self::count_variables(ctx, *e, visited);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    Self::count_variables(ctx, *arg, visited);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    Self::count_variables(ctx, *elem, visited);
                }
            }
            // Leaves
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    /// Estimate number of terms generated by multinomial expansion: C(n+k-1, k-1)
    /// For binomial (k=2): C(n+1, 1) = n+1
    fn estimate_terms(k: u32, n: u32) -> u32 {
        // Multinomial: number of terms = C(n+k-1, k-1)
        // For binomial: C(n+1, 1) = n+1
        // For trinomial: C(n+2, 2) = (n+1)(n+2)/2
        // etc.
        if k <= 1 {
            return 1;
        }
        // Compute C(n+k-1, k-1) = C(n+k-1, n)
        let top = n + k - 1;
        let bottom = k - 1;
        binomial_coeff(top, bottom)
    }
}

impl crate::rule::Rule for AutoExpandPowSumRule {
    fn name(&self) -> &str {
        "Auto Expand Power Sum"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Expand if: global auto-expand mode OR inside a marked cancellation context
        // (e.g., difference quotient like ((x+h)^n - x^n)/h)
        let in_expand_context = parent_ctx.in_auto_expand_context();
        if !(parent_ctx.is_auto_expand() || in_expand_context) {
            return None;
        }

        // Get budget - use default if in context but no explicit budget set
        let default_budget = crate::phase::ExpandBudget::default();
        let budget = parent_ctx.auto_expand_budget().unwrap_or(&default_budget);

        // Skip if expression is in canonical form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // Pattern: Pow(Add(...), n) - use zero-clone destructuring
        let (base, exp) = crate::helpers::as_pow(ctx, expr)?;

        // Check exponent is a small positive integer
        let n_val = {
            let exp_expr = ctx.get(exp);
            match exp_expr {
                Expr::Number(n) if n.is_integer() && !n.is_negative() => n.to_integer().to_u32()?,
                _ => return None,
            }
        };

        // Budget check 1: max_pow_exp
        if n_val > budget.max_pow_exp {
            return None;
        }
        // At least square to be useful
        if n_val < 2 {
            return None;
        }

        // Check base is an Add and extract terms
        let (a, b) = match crate::helpers::as_add(ctx, base) {
            Some((a, b)) => (a, b),
            None => return None,
        };

        // Budget check 2: max_base_terms
        let num_terms = Self::count_add_terms(ctx, base);
        if num_terms > budget.max_base_terms {
            return None;
        }

        // Budget check 3: max_generated_terms
        let estimated_result_terms = Self::estimate_terms(num_terms, n_val);
        if estimated_result_terms > budget.max_generated_terms {
            return None;
        }

        // Budget check 4: max_vars
        let mut vars = std::collections::HashSet::new();
        Self::count_variables(ctx, base, &mut vars);
        if vars.len() as u32 > budget.max_vars {
            return None;
        }

        // All budget checks passed!
        // For binomials (2 terms), use binomial expansion
        if num_terms == 2 {
            // Use a and b extracted above
            let mut terms = Vec::new();
            for k in 0..=n_val {
                let coeff = binomial_coeff(n_val, k);
                let exp_a = n_val - k;
                let exp_b = k;

                let term_a = if exp_a == 0 {
                    ctx.num(1)
                } else if exp_a == 1 {
                    a
                } else {
                    let exp_a_id = ctx.num(exp_a as i64);
                    ctx.add(Expr::Pow(a, exp_a_id))
                };

                let term_b = if exp_b == 0 {
                    ctx.num(1)
                } else if exp_b == 1 {
                    b
                } else {
                    let exp_b_id = ctx.num(exp_b as i64);
                    ctx.add(Expr::Pow(b, exp_b_id))
                };

                let mut term = mul2_raw(ctx, term_a, term_b);
                if coeff > 1 {
                    let c = ctx.num(coeff as i64);
                    term = mul2_raw(ctx, c, term);
                }
                terms.push(term);
            }

            // Sum up terms
            let mut expanded = terms[0];
            for &term in terms.iter().skip(1) {
                expanded = ctx.add(Expr::Add(expanded, term));
            }

            return Some(Rewrite::new(expanded).desc(format!("Auto-expand (a+b)^{}", n_val)));
        }

        // For trinomials and higher, use general multinomial expansion
        // (more complex, skip for now - only binomials are auto-expanded)
        // Users can use explicit expand() for higher-order polynomials

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Include RATIONALIZE so auto-expand can clean up after rationalization
        // e.g., 1/(1+√2+√3) → ... → (1+√2)² - 3 → needs auto-expand to become 2√2
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        // Auto-expand steps are didactically important: users should see the expansion
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// AutoExpandSubCancelRule
// =============================================================================

/// AutoExpandSubCancelRule: Zero-shortcut for Sub(Pow(Add..), polynomial)
/// Priority 95 (higher than AutoExpandPowSumRule at 50)
pub struct AutoExpandSubCancelRule;

impl AutoExpandSubCancelRule {
    /// Convert expression to MultiPoly (returns None if not polynomial-representable)
    pub(crate) fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        Self::expr_to_multipoly_inner(ctx, id, vars, budget, 0)
    }

    fn expr_to_multipoly_inner(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
        depth: usize,
    ) -> Option<MultiPoly> {
        // Depth limit to prevent stack overflow
        if depth > 50 {
            return None;
        }

        match ctx.get(id) {
            Expr::Number(n) => {
                // Constant polynomial
                Some(MultiPoly::from_const(n.clone()))
            }
            Expr::Variable(sym_id) => {
                // Variable: ensure it's in our vars list
                let name = ctx.sym_name(*sym_id).to_string();
                if !vars.contains(&name) {
                    if vars.len() >= 4 {
                        return None; // Too many variables
                    }
                    vars.push(name.clone());
                }
                // Create polynomial for this variable
                let idx = vars.iter().position(|v| v == &name)?;
                let mut mono = vec![0u32; vars.len()];
                mono[idx] = 1;
                let terms = vec![(BigRational::one(), mono)];
                Some(MultiPoly {
                    vars: vars.clone(),
                    terms,
                })
            }
            Expr::Add(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                // Align variables
                let (p, q) = Self::align_vars(p, q, vars);
                p.add(&q).ok()
            }
            Expr::Sub(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                let (p, q) = Self::align_vars(p, q, vars);
                p.sub(&q).ok()
            }
            Expr::Mul(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                let (p, q) = Self::align_vars(p, q, vars);
                p.mul(&q, budget).ok()
            }
            Expr::Neg(inner) => {
                let p = Self::expr_to_multipoly_inner(ctx, *inner, vars, budget, depth + 1)?;
                Some(p.neg())
            }
            Expr::Pow(base, exp) => {
                // Only handle integer exponents >= 0
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && !n.is_negative() {
                        let exp_val = n.to_integer().to_u32()?;
                        if exp_val > budget.max_total_degree {
                            return None;
                        }
                        // Recursively convert base - this may grow vars
                        let base_poly =
                            Self::expr_to_multipoly_inner(ctx, *base, vars, budget, depth + 1)?;

                        // Handle exp == 0 case
                        if exp_val == 0 {
                            return Some(MultiPoly::one(vars.clone()));
                        }

                        // Compute base^exp via repeated multiplication
                        // Start with base aligned to current vars
                        let base_aligned = Self::align_to_vars(&base_poly, vars);
                        let mut result = base_aligned.clone();
                        for _ in 1..exp_val {
                            result = result.mul(&base_aligned, budget).ok()?;
                            if result.num_terms() > budget.max_terms {
                                return None;
                            }
                        }
                        return Some(result);
                    }
                }
                None
            }
            _ => None, // Not polynomial
        }
    }

    /// Align two polynomials to have the same variable set
    fn align_vars(p: MultiPoly, q: MultiPoly, target_vars: &[String]) -> (MultiPoly, MultiPoly) {
        (
            Self::align_to_vars(&p, target_vars),
            Self::align_to_vars(&q, target_vars),
        )
    }

    /// Align a polynomial to the target variable set
    fn align_to_vars(p: &MultiPoly, target_vars: &[String]) -> MultiPoly {
        if p.vars == target_vars {
            return p.clone();
        }
        // Reindex monomials to target_vars
        let mut new_terms = Vec::new();
        for (coeff, mono) in &p.terms {
            let mut new_mono = vec![0u32; target_vars.len()];
            for (i, var) in p.vars.iter().enumerate() {
                if let Some(target_idx) = target_vars.iter().position(|v| v == var) {
                    new_mono[target_idx] = mono[i];
                }
            }
            new_terms.push((coeff.clone(), new_mono));
        }
        MultiPoly {
            vars: target_vars.to_vec(),
            terms: new_terms,
        }
    }
}

impl crate::rule::Rule for AutoExpandSubCancelRule {
    fn name(&self) -> &str {
        "AutoExpandSubCancelRule"
    }

    fn priority(&self) -> i32 {
        95
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger if in auto-expand context (marked by scanner)
        // Use _for_expr to also check if current node is marked (not just ancestors)
        if !parent_ctx.in_auto_expand_context_for_expr(expr) {
            return None;
        }

        // Must be Sub or Add to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        // Budget for polynomial conversion
        let budget = PolyBudget {
            max_terms: 100,
            max_total_degree: 8,
            max_pow_exp: 4, // Small limit for cancellation checks
        };

        // Convert entire expression to MultiPoly
        // For Sub(a,b) this computes a-b
        // For Add(a, Neg(b), Neg(c), ...) this computes a + (-b) + (-c) + ...
        // If the result is 0, we have cancellation
        let mut vars = Vec::new();
        let poly = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // If the result is zero, we have proved cancellation!
        if poly.is_zero() {
            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc("Polynomial equality: expressions cancel to 0"));
        }

        None
    }
}

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
            Expr::Number(_) | Expr::Variable(_) => true,
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
                false
            }
            _ => false, // Functions, Division, etc. are not polynomial
        }
    }

    /// Convert expression to MultiPoly (reusing AutoExpandSubCancelRule's method)
    fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        AutoExpandSubCancelRule::expr_to_multipoly(ctx, id, vars, budget)
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
        PhaseMask::TRANSFORM
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
        let poly = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

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
                let mut lhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());
                let mut rhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());

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

                crate::multipoly_display::PolynomialProofData::from_identity(
                    ctx,
                    &lhs_poly,
                    &rhs_poly,
                    vars.clone(),
                )
            } else {
                // No clear LHS/RHS split
                crate::multipoly_display::PolynomialProofData {
                    monomials: 0,
                    degree: 0,
                    vars: vars.clone(),
                    normal_form_expr: Some(zero),
                    lhs_stats: None,
                    rhs_stats: None,
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

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add", "Sub"])
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

        let mut vars = Vec::new();
        let poly = AutoExpandSubCancelRule::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // Check if result is reasonable
        if poly.terms.len() > 30 || vars.len() > 3 {
            return None;
        }

        // If polynomial is zero, let PolynomialIdentityZeroRule handle it
        if poly.is_zero() {
            return None;
        }

        // Convert back to expression using multipoly_to_expr (produces flattened Add)
        let new_expr = crate::multipoly::multipoly_to_expr(&poly, ctx);

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

use crate::multinomial_expand::{try_expand_multinomial_direct, MultinomialExpandBudget};

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

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
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
