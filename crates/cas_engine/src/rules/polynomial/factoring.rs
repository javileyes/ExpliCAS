//! Polynomial factoring rules: common factor extraction from sums.

use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;

/// Parsed representation of a term in an Add/Sub expression
/// Represents: sign * coeff * base^exp
#[derive(Debug, Clone)]
struct ParsedTerm {
    sign: i8,     // +1 or -1
    coeff: i64,   // Integer coefficient (1 if implicit)
    base: ExprId, // The base expression (e.g., x+1)
    exp: u32,     // The exponent (>= 1)
}

/// HeuristicExtractCommonFactorAddRule: Extract common base factors from sums
pub struct HeuristicExtractCommonFactorAddRule;

impl HeuristicExtractCommonFactorAddRule {
    /// Parse a term into (sign, coeff, base, exp) form
    /// Returns None if term doesn't match expected pattern
    fn parse_term(ctx: &Context, term: ExprId, positive: bool) -> Option<ParsedTerm> {
        let sign: i8 = if positive { 1 } else { -1 };

        // Try to match: coeff * Pow(base, exp) or Pow(base, exp) or coeff * base
        match ctx.get(term) {
            // Direct power: base^exp
            Expr::Pow(base, exp_id) => {
                let exp = Self::extract_int_exp(ctx, *exp_id)?;
                if exp >= 1 {
                    Some(ParsedTerm {
                        sign,
                        coeff: 1,
                        base: *base,
                        exp,
                    })
                } else {
                    None
                }
            }
            // Multiplication: coeff * something
            Expr::Mul(l, r) => {
                // Try: coeff * base^exp
                if let Some(c) = Self::extract_int_coeff(ctx, *l) {
                    if let Expr::Pow(base, exp_id) = ctx.get(*r) {
                        let exp = Self::extract_int_exp(ctx, *exp_id)?;
                        if exp >= 1 {
                            return Some(ParsedTerm {
                                sign,
                                coeff: c,
                                base: *base,
                                exp,
                            });
                        }
                    }
                    // Try: coeff * base (implicit exp=1)
                    // Only if base is Add (polynomial-like)
                    if matches!(ctx.get(*r), Expr::Add(_, _)) {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *r,
                            exp: 1,
                        });
                    }
                }
                // Try: base^exp * coeff (reversed order)
                if let Some(c) = Self::extract_int_coeff(ctx, *r) {
                    if let Expr::Pow(base, exp_id) = ctx.get(*l) {
                        let exp = Self::extract_int_exp(ctx, *exp_id)?;
                        if exp >= 1 {
                            return Some(ParsedTerm {
                                sign,
                                coeff: c,
                                base: *base,
                                exp,
                            });
                        }
                    }
                    // Try: base * coeff (implicit exp=1)
                    if matches!(ctx.get(*l), Expr::Add(_, _)) {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *l,
                            exp: 1,
                        });
                    }
                }
                None
            }
            // Negation: -something
            Expr::Neg(inner) => Self::parse_term(ctx, *inner, !positive),
            _ => None,
        }
    }

    /// Extract integer exponent from an expression
    fn extract_int_exp(ctx: &Context, exp_id: ExprId) -> Option<u32> {
        if let Expr::Number(n) = ctx.get(exp_id) {
            if n.is_integer() && !n.is_negative() {
                use num_traits::ToPrimitive;
                return n.to_integer().to_u32();
            }
        }
        None
    }

    /// Extract integer coefficient from an expression
    fn extract_int_coeff(ctx: &Context, expr: ExprId) -> Option<i64> {
        if let Expr::Number(n) = ctx.get(expr) {
            if n.is_integer() {
                use num_traits::ToPrimitive;
                return n.to_integer().to_i64();
            }
        }
        None
    }

    /// Check structural equality of two expression bases
    fn bases_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
        if a == b {
            return true;
        }
        Self::exprs_equal_recursive(ctx, a, b)
    }

    /// Recursive structural equality check for expressions
    fn exprs_equal_recursive(ctx: &Context, a: ExprId, b: ExprId) -> bool {
        if a == b {
            return true;
        }
        match (ctx.get(a), ctx.get(b)) {
            (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
            (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
            (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
            (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Neg(e1), Expr::Neg(e2)) => Self::exprs_equal_recursive(ctx, *e1, *e2),
            (Expr::Function(n1, args1), Expr::Function(n2, args2)) => {
                n1 == n2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| Self::exprs_equal_recursive(ctx, *a1, *a2))
            }
            _ => false,
        }
    }

    /// Build the quotient term: term / base^g_exp
    /// term = sign * coeff * base^exp
    /// result = sign * coeff * base^(exp - g_exp)
    fn build_quotient_term(ctx: &mut Context, term: &ParsedTerm, g_exp: u32) -> ExprId {
        let remaining_exp = term.exp.saturating_sub(g_exp);

        // Build: coeff * base^remaining_exp (or just coeff or just base^rem)
        let coeff_part = if term.coeff == 1 {
            None
        } else {
            Some(ctx.num(term.coeff.abs()))
        };

        let power_part = if remaining_exp == 0 {
            None
        } else if remaining_exp == 1 {
            Some(term.base)
        } else {
            let exp_id = ctx.num(remaining_exp as i64);
            Some(ctx.add(Expr::Pow(term.base, exp_id)))
        };

        // Combine coeff and power parts
        let unsigned_result = match (coeff_part, power_part) {
            (None, None) => ctx.num(1),
            (Some(c), None) => c,
            (None, Some(p)) => p,
            (Some(c), Some(p)) => ctx.add(Expr::Mul(c, p)),
        };

        // Apply sign
        if term.sign < 0 || term.coeff < 0 {
            // XOR of signs: if exactly one is negative, result is negative
            let total_negative = (term.sign < 0) ^ (term.coeff < 0);
            if total_negative {
                ctx.add(Expr::Neg(unsigned_result))
            } else {
                unsigned_result
            }
        } else {
            unsigned_result
        }
    }

    /// Simplify an Add expression by combining numeric constants
    /// e.g., (x + 1) + 4 → x + 5, or 1 + 4 → 5
    fn simplify_add_constants(ctx: &mut Context, expr: ExprId) -> ExprId {
        // Collect all additive terms and sum numeric ones
        let mut numeric_sum: i64 = 0;
        let mut non_numeric: Vec<ExprId> = Vec::new();

        Self::collect_add_terms_for_const_fold(ctx, expr, true, &mut numeric_sum, &mut non_numeric);

        // Rebuild expression
        if non_numeric.is_empty() {
            // All numeric
            ctx.num(numeric_sum)
        } else {
            // Start with first non-numeric term
            let mut result = non_numeric[0];
            for term in &non_numeric[1..] {
                result = ctx.add(Expr::Add(result, *term));
            }
            // Add numeric sum if non-zero
            if numeric_sum != 0 {
                let num_expr = ctx.num(numeric_sum);
                result = ctx.add(Expr::Add(result, num_expr));
            }
            result
        }
    }

    fn collect_add_terms_for_const_fold(
        ctx: &Context,
        expr: ExprId,
        positive: bool,
        numeric_sum: &mut i64,
        non_numeric: &mut Vec<ExprId>,
    ) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                Self::collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
                Self::collect_add_terms_for_const_fold(ctx, *r, positive, numeric_sum, non_numeric);
            }
            Expr::Sub(l, r) => {
                Self::collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
                Self::collect_add_terms_for_const_fold(
                    ctx,
                    *r,
                    !positive,
                    numeric_sum,
                    non_numeric,
                );
            }
            Expr::Neg(inner) => {
                Self::collect_add_terms_for_const_fold(
                    ctx,
                    *inner,
                    !positive,
                    numeric_sum,
                    non_numeric,
                );
            }
            Expr::Number(n) => {
                if n.is_integer() {
                    use num_traits::ToPrimitive;
                    if let Some(v) = n.to_integer().to_i64() {
                        if positive {
                            *numeric_sum += v;
                        } else {
                            *numeric_sum -= v;
                        }
                        return;
                    }
                }
                // Non-integer or overflow: treat as non-numeric
                if positive {
                    non_numeric.push(expr);
                } else {
                    // Would need to wrap in Neg but we're avoiding that
                    non_numeric.push(expr);
                }
            }
            _ => {
                non_numeric.push(expr);
            }
        }
    }
}

impl crate::rule::Rule for HeuristicExtractCommonFactorAddRule {
    fn name(&self) -> &str {
        "Heuristic Extract Common Factor"
    }

    fn priority(&self) -> i32 {
        110 // Higher than HeuristicPolyNormalizeAddRule (100) to try factorization first
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::TRANSFORM
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

        // === SAFE MODE: Only handle Add(A, B) or Sub(A, B) with exactly 2 terms ===
        let (term1_id, term2_id, term2_positive) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r, true),
            Expr::Sub(l, r) => (*l, *r, false),
            _ => return None,
        };

        // Parse both terms
        let term1 = Self::parse_term(ctx, term1_id, true)?;
        let term2 = Self::parse_term(ctx, term2_id, term2_positive)?;

        // Both bases must be equal (structural comparison)
        if !Self::bases_equal(ctx, term1.base, term2.base) {
            return None;
        }

        // Base must be compound (Add) to be interesting for polynomial factorization
        if !matches!(ctx.get(term1.base), Expr::Add(_, _)) {
            return None;
        }

        // GCD exponent = min(exp1, exp2)
        let g_exp = term1.exp.min(term2.exp);
        if g_exp == 0 {
            return None;
        }

        // Build quotient terms
        let q1 = Self::build_quotient_term(ctx, &term1, g_exp);
        let q2 = Self::build_quotient_term(ctx, &term2, g_exp);

        // Build inner sum: q1 + q2
        let inner_sum_raw = ctx.add(Expr::Add(q1, q2));

        // Simplify constants in inner_sum (e.g., x + 1 + 4 → x + 5)
        let inner_sum = Self::simplify_add_constants(ctx, inner_sum_raw);

        // Build factor: base^g_exp
        let factor = if g_exp == 1 {
            term1.base
        } else {
            let exp_id = ctx.num(g_exp as i64);
            ctx.add(Expr::Pow(term1.base, exp_id))
        };

        // Build result: factor * inner_sum
        // Wrap in __hold to prevent DistributeRule from expanding it back
        let product = ctx.add(Expr::Mul(factor, inner_sum));
        let new_expr = cas_ast::hold::wrap_hold(ctx, product);

        // Complexity check: result should be simpler
        let old_nodes = cas_ast::count_nodes(ctx, expr);
        let new_nodes = cas_ast::count_nodes(ctx, new_expr);
        if new_nodes > old_nodes + 5 {
            return None; // Don't make things worse
        }

        Some(
            Rewrite::new(new_expr)
                .desc("Extract common polynomial factor")
                .local(expr, new_expr),
        )
    }
}
