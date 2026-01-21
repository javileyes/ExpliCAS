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

use crate::poly_store::{PolyId, PolyMeta, PolyStore};
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
pub fn poly_lower_pass(ctx: &mut Context, store: &mut PolyStore, expr: ExprId) -> PolyLowerResult {
    let mut steps = Vec::new();
    let mut combined_any = false;
    let result = lower_recursive(ctx, store, expr, &mut steps, &mut combined_any);
    PolyLowerResult {
        expr: result,
        steps,
        combined_any,
    }
}

fn lower_recursive(
    ctx: &mut Context,
    store: &mut PolyStore,
    expr: ExprId,
    steps: &mut Vec<Step>,
    combined_any: &mut bool,
) -> ExprId {
    match ctx.get(expr).clone() {
        // Check if this is already a poly_ref - pass through
        Expr::Function(ref name, ref args) if name == "poly_ref" => {
            return expr;
        }

        // Add: try to combine poly_refs
        Expr::Add(l, r) => {
            let nl = lower_recursive(ctx, store, l, steps, combined_any);
            let nr = lower_recursive(ctx, store, r, steps, combined_any);

            // Try to combine if both are poly_refs
            if let (Some(id_l), Some(id_r)) =
                (extract_poly_ref_id(ctx, nl), extract_poly_ref_id(ctx, nr))
            {
                if let Some(new_id) = combine_add(store, id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_ref(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_ref + poly_ref",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                }
            }

            // No combination possible
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }

        // Sub: try to combine poly_refs
        Expr::Sub(l, r) => {
            let nl = lower_recursive(ctx, store, l, steps, combined_any);
            let nr = lower_recursive(ctx, store, r, steps, combined_any);

            if let (Some(id_l), Some(id_r)) =
                (extract_poly_ref_id(ctx, nl), extract_poly_ref_id(ctx, nr))
            {
                if let Some(new_id) = combine_sub(store, id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_ref(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_ref - poly_ref",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                }
            }

            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }

        // Mul: try to combine poly_refs
        Expr::Mul(l, r) => {
            let nl = lower_recursive(ctx, store, l, steps, combined_any);
            let nr = lower_recursive(ctx, store, r, steps, combined_any);

            if let (Some(id_l), Some(id_r)) =
                (extract_poly_ref_id(ctx, nl), extract_poly_ref_id(ctx, nr))
            {
                if let Some(new_id) = combine_mul(store, id_l, id_r) {
                    *combined_any = true;
                    let result = make_poly_ref(ctx, new_id);

                    steps.push(Step::new(
                        "Poly lowering: combined poly_ref * poly_ref",
                        "Polynomial Combination",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));

                    return result;
                }
            }

            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }

        // Neg: negate poly_ref
        Expr::Neg(inner) => {
            let ni = lower_recursive(ctx, store, inner, steps, combined_any);

            if let Some(id) = extract_poly_ref_id(ctx, ni) {
                if let Some(new_id) = combine_neg(store, id) {
                    *combined_any = true;
                    return make_poly_ref(ctx, new_id);
                }
            }

            if ni != inner {
                ctx.add(Expr::Neg(ni))
            } else {
                expr
            }
        }

        // Pow: poly_ref^n
        Expr::Pow(base, exp) => {
            let nb = lower_recursive(ctx, store, base, steps, combined_any);
            let ne = lower_recursive(ctx, store, exp, steps, combined_any);

            if let Some(id) = extract_poly_ref_id(ctx, nb) {
                if let Some(n) = extract_int(ctx, ne) {
                    if n >= 0 {
                        if let Some(new_id) = combine_pow(store, id, n as u32) {
                            *combined_any = true;
                            return make_poly_ref(ctx, new_id);
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
            let nl = lower_recursive(ctx, store, l, steps, combined_any);
            let nr = lower_recursive(ctx, store, r, steps, combined_any);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }

        // Functions (other than poly_ref): recurse into args
        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| lower_recursive(ctx, store, a, steps, combined_any))
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
                .map(|&e| lower_recursive(ctx, store, e, steps, combined_any))
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

/// Extract PolyId from poly_ref(id) expression
fn extract_poly_ref_id(ctx: &Context, expr: ExprId) -> Option<PolyId> {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "poly_ref" && args.len() == 1 {
            if let Expr::Number(n) = ctx.get(args[0]) {
                return n.to_integer().try_into().ok();
            }
        }
    }
    None
}

/// Create poly_ref(id) expression
fn make_poly_ref(ctx: &mut Context, id: PolyId) -> ExprId {
    let id_expr = ctx.num(id as i64);
    ctx.add(Expr::Function("poly_ref".to_string(), vec![id_expr]))
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

// =============================================================================
// Store combination operations (stubs - need implementation)
// =============================================================================

/// Add two polynomials in the store, return new ID
fn combine_add(store: &mut PolyStore, a: PolyId, b: PolyId) -> Option<PolyId> {
    let (meta_a, poly_a) = store.get(a)?;
    let (meta_b, poly_b) = store.get(b)?;

    // Ensure same modulus
    if meta_a.modulus != meta_b.modulus {
        return None;
    }

    // TODO: Unify VarTables and remap if needed
    // For now, only combine if same variable order
    if meta_a.var_names != meta_b.var_names {
        return None;
    }

    // Add polynomials
    let result_poly = poly_a.add(poly_b);

    let meta = PolyMeta {
        modulus: meta_a.modulus,
        n_terms: result_poly.num_terms(),
        n_vars: meta_a.n_vars.max(meta_b.n_vars),
        max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
        var_names: meta_a.var_names.clone(),
    };

    Some(store.insert(meta, result_poly))
}

/// Subtract two polynomials in the store, return new ID
fn combine_sub(store: &mut PolyStore, a: PolyId, b: PolyId) -> Option<PolyId> {
    let (meta_a, poly_a) = store.get(a)?;
    let (meta_b, poly_b) = store.get(b)?;

    if meta_a.modulus != meta_b.modulus {
        return None;
    }
    if meta_a.var_names != meta_b.var_names {
        return None;
    }

    let result_poly = poly_a.sub(poly_b);

    let meta = PolyMeta {
        modulus: meta_a.modulus,
        n_terms: result_poly.num_terms(),
        n_vars: meta_a.n_vars.max(meta_b.n_vars),
        max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
        var_names: meta_a.var_names.clone(),
    };

    Some(store.insert(meta, result_poly))
}

/// Multiply two polynomials in the store, return new ID
fn combine_mul(store: &mut PolyStore, a: PolyId, b: PolyId) -> Option<PolyId> {
    let (meta_a, poly_a) = store.get(a)?;
    let (meta_b, poly_b) = store.get(b)?;

    if meta_a.modulus != meta_b.modulus {
        return None;
    }
    if meta_a.var_names != meta_b.var_names {
        return None;
    }

    let result_poly = poly_a.mul(poly_b);

    let meta = PolyMeta {
        modulus: meta_a.modulus,
        n_terms: result_poly.num_terms(),
        n_vars: meta_a.n_vars.max(meta_b.n_vars),
        max_total_degree: meta_a.max_total_degree + meta_b.max_total_degree,
        var_names: meta_a.var_names.clone(),
    };

    Some(store.insert(meta, result_poly))
}

/// Negate polynomial in the store, return new ID
fn combine_neg(store: &mut PolyStore, a: PolyId) -> Option<PolyId> {
    let (meta_a, poly_a) = store.get(a)?;

    let result_poly = poly_a.neg();

    let meta = PolyMeta {
        modulus: meta_a.modulus,
        n_terms: result_poly.num_terms(),
        n_vars: meta_a.n_vars,
        max_total_degree: meta_a.max_total_degree,
        var_names: meta_a.var_names.clone(),
    };

    Some(store.insert(meta, result_poly))
}

/// Raise polynomial to power in the store, return new ID
fn combine_pow(store: &mut PolyStore, a: PolyId, n: u32) -> Option<PolyId> {
    let (meta_a, poly_a) = store.get(a)?;

    let result_poly = poly_a.pow(n);

    let meta = PolyMeta {
        modulus: meta_a.modulus,
        n_terms: result_poly.num_terms(),
        n_vars: meta_a.n_vars,
        max_total_degree: meta_a.max_total_degree * n,
        var_names: meta_a.var_names.clone(),
    };

    Some(store.insert(meta, result_poly))
}

#[cfg(test)]
mod tests {
    // TODO: Add tests when integrated with PolyStore
}
