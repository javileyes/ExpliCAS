//! Poly Lowering Pass
//!
//! Pre-simplification pass that collapses operations between `poly_ref` nodes.
//! Runs BEFORE the main simplifier pipeline.
//!
//! # Purpose
//!
//! When the user writes `expand(A^7) + expand(B^7)`, the eager eval produces:
//!
//! ```text
//! Add(poly_ref(1), poly_ref(2))
//! ```
//!
//! This pass detects such patterns and combines them internally:
//!
//! ```text
//! poly_ref(3)  // where store[3] = store[1] + store[2]
//! ```
//!
//! The simplifier then sees only a single `poly_ref` atom, avoiding O(n²) traversal.
//!
//! # Supported Patterns
//!
//! - `Add(poly_ref, poly_ref)` → combined poly_ref
//! - `Sub(poly_ref, poly_ref)` → combined poly_ref
//! - `Mul(poly_ref, poly_ref)` → combined poly_ref
//! - `Neg(poly_ref)` → negated poly_ref
//! - `Add(poly_ref, poly_like_expr)` → if expr can convert to poly, combine
//!
//! # Auto-Promotion
//!
//! When one operand is `poly_result` and the other is a simple polynomial expression,
//! the simple expression is automatically converted to `poly_result` for combination.

use crate::poly_modp_conv::{
    expr_to_poly_modp_with_store as expr_to_poly_modp, PolyModpBudget, VarTable,
};
use crate::poly_store::{
    thread_local_add, thread_local_insert, thread_local_meta, thread_local_mul, thread_local_neg,
    thread_local_pow, thread_local_sub, PolyId, PolyMeta,
};
use crate::Step;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Maximum node count for promotion (guards against huge expressions)
const PROMOTE_MAX_NODES: usize = 200;

/// Maximum terms after conversion for promotion
const PROMOTE_MAX_TERMS: usize = 10_000;

// =============================================================================
// Auto-promotion: convert simple poly-like expressions to poly_result
// =============================================================================

/// Try to promote a simple polynomial expression to a poly_result.
///
/// Only attempts promotion if:
/// 1. `base_id` is a valid poly_result (provides modulus and var_table context)
/// 2. `expr` is small enough (node count <= PROMOTE_MAX_NODES)
/// 3. `expr` can be converted to MultiPolyModP
/// 4. Resulting poly has <= PROMOTE_MAX_TERMS terms
///
/// Returns the new PolyId on success, None on failure.
fn try_promote_expr_to_poly(ctx: &Context, expr: ExprId, base_id: PolyId) -> Option<PolyId> {
    // Guard: check node count
    let (node_count, _) = cas_ast::traversal::count_nodes_and_max_depth(ctx, expr);
    if node_count > PROMOTE_MAX_NODES {
        tracing::debug!(
            poly_lowering = "skip_promote",
            reason = "node_count too large",
            node_count = node_count,
            max = PROMOTE_MAX_NODES,
            "Skipping promotion: expression too large"
        );
        return None;
    }

    // Get base metadata (for modulus)
    let base_meta = thread_local_meta(base_id)?;

    // Budget for conversion
    let budget = PolyModpBudget {
        max_terms: PROMOTE_MAX_TERMS,
        ..Default::default()
    };

    // Convert expression to polynomial
    let mut vars = VarTable::new();
    let poly = match expr_to_poly_modp(ctx, expr, base_meta.modulus, &budget, &mut vars) {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                poly_lowering = "skip_promote",
                reason = "conversion_failed",
                error = ?e,
                "Skipping promotion: cannot convert expression"
            );
            return None;
        }
    };

    // Check term count
    if poly.terms.len() > PROMOTE_MAX_TERMS {
        tracing::debug!(
            poly_lowering = "skip_promote",
            reason = "too_many_terms",
            terms = poly.terms.len(),
            max = PROMOTE_MAX_TERMS,
            "Skipping promotion: resulting poly too large"
        );
        return None;
    }

    // Unify variable tables
    let base_var_table = VarTable::from_names(&base_meta.var_names);
    let (unified, _remap_base, remap_new) = base_var_table.unify(&vars)?;

    // Remap the new polynomial to unified variable order
    let remapped_poly = poly.remap(&remap_new, unified.len());

    // Compute max total degree from terms
    let max_deg = remapped_poly
        .terms
        .iter()
        .map(|(mono, _)| mono.total_degree())
        .max()
        .unwrap_or(0);

    // Create metadata and insert
    let new_meta = PolyMeta {
        modulus: base_meta.modulus,
        n_terms: remapped_poly.terms.len(),
        n_vars: unified.len(),
        max_total_degree: max_deg,
        var_names: unified.names().to_vec(),
    };

    let new_id = thread_local_insert(new_meta, remapped_poly);

    tracing::debug!(
        poly_lowering = "promoted",
        new_id = new_id,
        terms = poly.terms.len(),
        "Successfully promoted expression to poly_result"
    );

    Some(new_id)
}

/// Result of poly lowering pass
pub struct PolyLowerResult {
    /// Transformed expression
    pub expr: ExprId,
    /// Steps generated during lowering
    pub steps: Vec<Step>,
    #[allow(dead_code)] // Constructed but not yet consumed; kept for future pipeline diagnostics
    pub combined_any: bool,
}

/// Run the poly lowering pass on an expression.
///
/// This should be called AFTER eager_eval_expand_calls and BEFORE the simplifier.
/// Uses the thread-local PolyStore for polynomial operations.
pub fn poly_lower_pass(ctx: &mut Context, expr: ExprId, collect_steps: bool) -> PolyLowerResult {
    let mut steps = Vec::new();
    let mut combined_any = false;
    let result = lower_recursive(ctx, expr, &mut steps, &mut combined_any, collect_steps);
    PolyLowerResult {
        expr: result,
        steps,
        combined_any,
    }
}

fn lower_recursive(
    ctx: &mut Context,
    expr: ExprId,
    steps: &mut Vec<Step>,
    combined_any: &mut bool,
    collect_steps: bool,
) -> ExprId {
    match ctx.get(expr).clone() {
        // Check if this is already a poly_result - pass through
        Expr::Function(ref name, ref _args) if ctx.is_builtin(*name, BuiltinFn::PolyResult) => expr,

        // Add: try to combine poly_results or promote simple expressions
        Expr::Add(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any, collect_steps);
            let nr = lower_recursive(ctx, r, steps, combined_any, collect_steps);

            let id_l = extract_poly_result_id(ctx, nl);
            let id_r = extract_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_add(id_l, id_r) {
                        *combined_any = true;
                        let result = make_poly_result(ctx, new_id);

                        if collect_steps {
                            steps.push(Step::new(
                                "Poly lowering: combined poly_result + poly_result",
                                "Polynomial Combination",
                                expr,
                                result,
                                Vec::new(),
                                Some(ctx),
                            ));
                        }
                        return result;
                    }
                }
                // Left is poly_result, try to promote right
                (Some(id_l), None) => {
                    if let Some(id_r_promoted) = try_promote_expr_to_poly(ctx, nr, id_l) {
                        if let Some(new_id) = thread_local_add(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            if collect_steps {
                                steps.push(Step::new(
                                    "Poly lowering: promoted and combined expressions",
                                    "Polynomial Combination",
                                    expr,
                                    result,
                                    Vec::new(),
                                    Some(ctx),
                                ));
                            }
                            return result;
                        }
                    }
                }
                // Right is poly_result, try to promote left
                (None, Some(id_r)) => {
                    if let Some(id_l_promoted) = try_promote_expr_to_poly(ctx, nl, id_r) {
                        if let Some(new_id) = thread_local_add(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            steps.push(Step::new(
                                "Poly lowering: promoted and combined expressions",
                                "Polynomial Combination",
                                expr,
                                result,
                                Vec::new(),
                                Some(ctx),
                            ));
                            return result;
                        }
                    }
                }
                // Neither is poly_result - no action
                (None, None) => {}
            }

            // No combination possible
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }

        // Sub: try to combine poly_results or promote simple expressions
        Expr::Sub(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any, collect_steps);
            let nr = lower_recursive(ctx, r, steps, combined_any, collect_steps);

            let id_l = extract_poly_result_id(ctx, nl);
            let id_r = extract_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_sub(id_l, id_r) {
                        *combined_any = true;
                        let result = make_poly_result(ctx, new_id);

                        if collect_steps {
                            steps.push(Step::new(
                                "Poly lowering: combined poly_result - poly_result",
                                "Polynomial Combination",
                                expr,
                                result,
                                Vec::new(),
                                Some(ctx),
                            ));
                        }
                        return result;
                    }
                }
                // Left is poly_result, try to promote right
                (Some(id_l), None) => {
                    if let Some(id_r_promoted) = try_promote_expr_to_poly(ctx, nr, id_l) {
                        if let Some(new_id) = thread_local_sub(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            if collect_steps {
                                steps.push(Step::new(
                                    "Poly lowering: promoted and combined expressions",
                                    "Polynomial Combination",
                                    expr,
                                    result,
                                    Vec::new(),
                                    Some(ctx),
                                ));
                            }
                            return result;
                        }
                    }
                }
                // Right is poly_result, try to promote left
                (None, Some(id_r)) => {
                    if let Some(id_l_promoted) = try_promote_expr_to_poly(ctx, nl, id_r) {
                        if let Some(new_id) = thread_local_sub(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            if collect_steps {
                                steps.push(Step::new(
                                    "Poly lowering: promoted and combined expressions",
                                    "Polynomial Combination",
                                    expr,
                                    result,
                                    Vec::new(),
                                    Some(ctx),
                                ));
                            }
                            return result;
                        }
                    }
                }
                // Neither is poly_result - no action
                (None, None) => {}
            }

            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }

        // Mul: try to combine poly_results or promote simple expressions
        Expr::Mul(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any, collect_steps);
            let nr = lower_recursive(ctx, r, steps, combined_any, collect_steps);

            let id_l = extract_poly_result_id(ctx, nl);
            let id_r = extract_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_mul(id_l, id_r) {
                        *combined_any = true;
                        let result = make_poly_result(ctx, new_id);

                        if collect_steps {
                            steps.push(Step::new(
                                "Poly lowering: combined poly_result * poly_result",
                                "Polynomial Combination",
                                expr,
                                result,
                                Vec::new(),
                                Some(ctx),
                            ));
                        }
                        return result;
                    }
                }
                // Left is poly_result, try to promote right
                (Some(id_l), None) => {
                    if let Some(id_r_promoted) = try_promote_expr_to_poly(ctx, nr, id_l) {
                        if let Some(new_id) = thread_local_mul(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            if collect_steps {
                                steps.push(Step::new(
                                    "Poly lowering: promoted and combined expressions",
                                    "Polynomial Combination",
                                    expr,
                                    result,
                                    Vec::new(),
                                    Some(ctx),
                                ));
                            }
                            return result;
                        }
                    }
                }
                // Right is poly_result, try to promote left
                (None, Some(id_r)) => {
                    if let Some(id_l_promoted) = try_promote_expr_to_poly(ctx, nl, id_r) {
                        if let Some(new_id) = thread_local_mul(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = make_poly_result(ctx, new_id);

                            if collect_steps {
                                steps.push(Step::new(
                                    "Poly lowering: promoted and combined expressions",
                                    "Polynomial Combination",
                                    expr,
                                    result,
                                    Vec::new(),
                                    Some(ctx),
                                ));
                            }
                            return result;
                        }
                    }
                }
                // Neither is poly_result - no action
                (None, None) => {}
            }

            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }

        // Neg: negate poly_result
        Expr::Neg(inner) => {
            let ni = lower_recursive(ctx, inner, steps, combined_any, collect_steps);

            if let Some(id) = extract_poly_result_id(ctx, ni) {
                if let Some(new_id) = thread_local_neg(id) {
                    *combined_any = true;
                    return make_poly_result(ctx, new_id);
                }
            }

            if ni != inner {
                ctx.add(Expr::Neg(ni))
            } else {
                expr
            }
        }

        // Pow: poly_result^n
        Expr::Pow(base, exp) => {
            let nb = lower_recursive(ctx, base, steps, combined_any, collect_steps);
            let ne = lower_recursive(ctx, exp, steps, combined_any, collect_steps);

            if let Some(id) = extract_poly_result_id(ctx, nb) {
                if let Some(n) = extract_int(ctx, ne) {
                    if n >= 0 {
                        if let Some(new_id) = thread_local_pow(id, n as u32) {
                            *combined_any = true;
                            return make_poly_result(ctx, new_id);
                        }
                    }
                }
            }

            if nb != base || ne != exp {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }

        // Div: no combination (poly division is complex)
        Expr::Div(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any, collect_steps);
            let nr = lower_recursive(ctx, r, steps, combined_any, collect_steps);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }

        // Functions (other than poly_result): recurse into args
        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| lower_recursive(ctx, a, steps, combined_any, collect_steps))
                .collect();
            if new_args.iter().zip(args.iter()).any(|(n, o)| n != o) {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }

        // Leaves - no recursion
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => expr,

        // Matrix - recurse into elements
        Expr::Matrix { rows, cols, data } => {
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|&e| lower_recursive(ctx, e, steps, combined_any, collect_steps))
                .collect();
            if new_data.iter().zip(data.iter()).any(|(n, o)| n != o) {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }

        // Hold blocks simplification - but recurse into inner for any poly_refs
        Expr::Hold(inner) => {
            let ni = lower_recursive(ctx, inner, steps, combined_any, collect_steps);
            if ni != inner {
                ctx.add(Expr::Hold(ni))
            } else {
                expr
            }
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Extract PolyId from poly_result(id) expression
fn extract_poly_result_id(ctx: &Context, expr: ExprId) -> Option<PolyId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::PolyResult) && args.len() == 1 {
            if let Expr::Number(n) = ctx.get(args[0]) {
                return n.to_integer().try_into().ok();
            }
        }
    }
    None
}

/// Create poly_result(id) expression
fn make_poly_result(ctx: &mut Context, id: PolyId) -> ExprId {
    crate::poly_result::wrap_poly_result(ctx, id)
}

/// Extract integer from expression
fn extract_int(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().try_into().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for poly_lowering
}
