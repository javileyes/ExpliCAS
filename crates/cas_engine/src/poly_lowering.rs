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

use crate::Step;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::poly_store::{
    thread_local_add, thread_local_mul, thread_local_neg, thread_local_pow,
    thread_local_promote_expr_with_base, thread_local_sub,
};

/// Maximum node count for promotion (guards against huge expressions)
const PROMOTE_MAX_NODES: usize = 200;

/// Maximum terms after conversion for promotion
const PROMOTE_MAX_TERMS: usize = 10_000;

// =============================================================================
// Auto-promotion: convert simple poly-like expressions to poly_result
// =============================================================================

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

            let id_l = cas_math::poly_result::parse_poly_result_id(ctx, nl);
            let id_r = cas_math::poly_result::parse_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_add(id_l, id_r) {
                        *combined_any = true;
                        let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_r_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nr,
                        id_l,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_add(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_l_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nl,
                        id_r,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_add(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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

            let id_l = cas_math::poly_result::parse_poly_result_id(ctx, nl);
            let id_r = cas_math::poly_result::parse_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_sub(id_l, id_r) {
                        *combined_any = true;
                        let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_r_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nr,
                        id_l,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_sub(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_l_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nl,
                        id_r,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_sub(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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

            let id_l = cas_math::poly_result::parse_poly_result_id(ctx, nl);
            let id_r = cas_math::poly_result::parse_poly_result_id(ctx, nr);

            match (id_l, id_r) {
                // Both are poly_result
                (Some(id_l), Some(id_r)) => {
                    if let Some(new_id) = thread_local_mul(id_l, id_r) {
                        *combined_any = true;
                        let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_r_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nr,
                        id_l,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_mul(id_l, id_r_promoted) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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
                    if let Some(id_l_promoted) = thread_local_promote_expr_with_base(
                        ctx,
                        nl,
                        id_r,
                        PROMOTE_MAX_NODES,
                        PROMOTE_MAX_TERMS,
                    ) {
                        if let Some(new_id) = thread_local_mul(id_l_promoted, id_r) {
                            *combined_any = true;
                            let result = cas_math::poly_result::wrap_poly_result(ctx, new_id);

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

            if let Some(id) = cas_math::poly_result::parse_poly_result_id(ctx, ni) {
                if let Some(new_id) = thread_local_neg(id) {
                    *combined_any = true;
                    return cas_math::poly_result::wrap_poly_result(ctx, new_id);
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

            if let Some(id) = cas_math::poly_result::parse_poly_result_id(ctx, nb) {
                if let Some(n) = extract_int(ctx, ne) {
                    if n >= 0 {
                        if let Some(new_id) = thread_local_pow(id, n as u32) {
                            *combined_any = true;
                            return cas_math::poly_result::wrap_poly_result(ctx, new_id);
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
