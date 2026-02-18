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
use cas_math::poly_lowering_ops::{
    try_combine_binary_poly_with_promotion, try_negate_poly_ref, try_pow_poly_ref, PolyBinaryOp,
    PolyCombineKind,
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

            if let Some(result) =
                try_lower_binary_poly(ctx, expr, nl, nr, PolyBinaryOp::Add, steps, collect_steps)
            {
                *combined_any = true;
                return result;
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

            if let Some(result) =
                try_lower_binary_poly(ctx, expr, nl, nr, PolyBinaryOp::Sub, steps, collect_steps)
            {
                *combined_any = true;
                return result;
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

            if let Some(result) =
                try_lower_binary_poly(ctx, expr, nl, nr, PolyBinaryOp::Mul, steps, collect_steps)
            {
                *combined_any = true;
                return result;
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

            if let Some(result) = try_negate_poly_ref(ctx, ni) {
                *combined_any = true;
                return result;
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

            if let Some(n) = extract_int(ctx, ne) {
                if let Ok(exp_u32) = u32::try_from(n) {
                    if let Some(result) = try_pow_poly_ref(ctx, nb, exp_u32) {
                        *combined_any = true;
                        return result;
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

fn try_lower_binary_poly(
    ctx: &mut Context,
    expr: ExprId,
    left: ExprId,
    right: ExprId,
    op: PolyBinaryOp,
    steps: &mut Vec<Step>,
    collect_steps: bool,
) -> Option<ExprId> {
    let combined = try_combine_binary_poly_with_promotion(
        ctx,
        left,
        right,
        op,
        PROMOTE_MAX_NODES,
        PROMOTE_MAX_TERMS,
    )?;
    if collect_steps {
        let message = match combined.kind {
            PolyCombineKind::Direct => match op {
                PolyBinaryOp::Add => "Poly lowering: combined poly_result + poly_result",
                PolyBinaryOp::Sub => "Poly lowering: combined poly_result - poly_result",
                PolyBinaryOp::Mul => "Poly lowering: combined poly_result * poly_result",
            },
            PolyCombineKind::Promoted => "Poly lowering: promoted and combined expressions",
        };
        steps.push(Step::new(
            message,
            "Polynomial Combination",
            expr,
            combined.expr,
            Vec::new(),
            Some(ctx),
        ));
    }
    Some(combined.expr)
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
