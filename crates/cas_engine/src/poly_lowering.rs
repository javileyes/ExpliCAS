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

use crate::poly_store::{
    thread_local_add, thread_local_mul, thread_local_neg, thread_local_pow, thread_local_sub,
    PolyId,
};
use crate::Step;
use cas_ast::{Context, Expr, ExprId};

/// Result of poly lowering pass
pub struct PolyLowerResult {
    /// Transformed expression
    pub expr: ExprId,
    /// Steps generated during lowering
    pub steps: Vec<Step>,
    /// Whether any poly_refs were combined
    pub combined_any: bool,
}

/// Run the poly lowering pass on an expression.
///
/// This should be called AFTER eager_eval_expand_calls and BEFORE the simplifier.
/// Uses the thread-local PolyStore for polynomial operations.
pub fn poly_lower_pass(ctx: &mut Context, expr: ExprId) -> PolyLowerResult {
    let mut steps = Vec::new();
    let mut combined_any = false;
    let result = lower_recursive(ctx, expr, &mut steps, &mut combined_any);
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
) -> ExprId {
    match ctx.get(expr).clone() {
        // Check if this is already a poly_result - pass through
        Expr::Function(ref name, ref _args) if name == "poly_result" => expr,

        // Add: try to combine poly_results
        Expr::Add(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any);
            let nr = lower_recursive(ctx, r, steps, combined_any);

            // Try to combine if both are poly_results
            if let (Some(id_l), Some(id_r)) = (
                extract_poly_result_id(ctx, nl),
                extract_poly_result_id(ctx, nr),
            ) {
                if let Some(new_id) = thread_local_add(id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_result(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_result + poly_result",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                } else {
                    // Combination failed - likely due to incompatible var_names
                    // Log warning (will be visible with RUST_LOG=warn)
                    tracing::warn!(
                        poly_lowering = "skipped",
                        reason = "incompatible variable tables",
                        left_id = id_l,
                        right_id = id_r,
                        "poly_lowering: Add skipped (Phase 3 will unify variable tables)"
                    );
                }
            }

            // No combination possible
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }

        // Sub: try to combine poly_results
        Expr::Sub(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any);
            let nr = lower_recursive(ctx, r, steps, combined_any);

            if let (Some(id_l), Some(id_r)) = (
                extract_poly_result_id(ctx, nl),
                extract_poly_result_id(ctx, nr),
            ) {
                if let Some(new_id) = thread_local_sub(id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_result(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_result - poly_result",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                } else {
                    tracing::warn!(
                        poly_lowering = "skipped",
                        reason = "incompatible variable tables",
                        left_id = id_l,
                        right_id = id_r,
                        "poly_lowering: Sub skipped (Phase 3 will unify variable tables)"
                    );
                }
            }

            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }

        // Mul: try to combine poly_results
        Expr::Mul(l, r) => {
            let nl = lower_recursive(ctx, l, steps, combined_any);
            let nr = lower_recursive(ctx, r, steps, combined_any);

            if let (Some(id_l), Some(id_r)) = (
                extract_poly_result_id(ctx, nl),
                extract_poly_result_id(ctx, nr),
            ) {
                if let Some(new_id) = thread_local_mul(id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_result(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_result * poly_result",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                } else {
                    tracing::warn!(
                        poly_lowering = "skipped",
                        reason = "incompatible variable tables",
                        left_id = id_l,
                        right_id = id_r,
                        "poly_lowering: Mul skipped (Phase 3 will unify variable tables)"
                    );
                }
            }

            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }

        // Neg: negate poly_result
        Expr::Neg(inner) => {
            let ni = lower_recursive(ctx, inner, steps, combined_any);

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
            let nb = lower_recursive(ctx, base, steps, combined_any);
            let ne = lower_recursive(ctx, exp, steps, combined_any);

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
            let nl = lower_recursive(ctx, l, steps, combined_any);
            let nr = lower_recursive(ctx, r, steps, combined_any);
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
                .map(|&a| lower_recursive(ctx, a, steps, combined_any))
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
                .map(|&e| lower_recursive(ctx, e, steps, combined_any))
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
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Extract PolyId from poly_result(id) expression
fn extract_poly_result_id(ctx: &Context, expr: ExprId) -> Option<PolyId> {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "poly_result" && args.len() == 1 {
            if let Expr::Number(n) = ctx.get(args[0]) {
                return n.to_integer().try_into().ok();
            }
        }
    }
    None
}

/// Create poly_result(id) expression
fn make_poly_result(ctx: &mut Context, id: PolyId) -> ExprId {
    let id_expr = ctx.num(id as i64);
    ctx.add(Expr::Function("poly_result".to_string(), vec![id_expr]))
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
