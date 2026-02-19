use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::Expr;
use cas_math::expr_rewrite::distribute;
use cas_math::poly_store::expand_expr_modp_materialized_hold;

/// Maximum terms to materialize in expand().
/// Above this, expand() is left unevaluated with a warning.
/// Use poly_mul_modp() for large polynomial operations.
pub const EXPAND_MAX_MATERIALIZE_TERMS: u64 = 200_000;

/// Threshold for using fast mod-p expansion instead of symbolic.
/// Above this many terms, use materialized mod-p expansion which is much faster.
pub const EXPAND_MODP_THRESHOLD: u64 = 1_000;

// ExpandRule: only runs in Transform phase
define_rule!(
    ExpandRule,
    "Expand Polynomial",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Expand)) && args.len() == 1
            {
                let arg = args[0];

                // Estimate output terms
                let est = crate::expand::estimate_expand_terms(ctx, arg);

                // Guard: abort if too large
                if let Some(est) = est {
                    if est > EXPAND_MAX_MATERIALIZE_TERMS {
                        tracing::warn!(
                            estimated_terms = est,
                            limit = EXPAND_MAX_MATERIALIZE_TERMS,
                            "expand() aborted: estimated {} terms exceeds limit {}. \
                             Use poly_mul_modp() for large polynomial operations.",
                            est,
                            EXPAND_MAX_MATERIALIZE_TERMS
                        );
                        // Return None → leaves expand(...) unevaluated
                        return None;
                    }
                }

                // Strategy: use mod-p fast path for large polynomials (> 1000 terms)
                if est.unwrap_or(0) > EXPAND_MODP_THRESHOLD {
                    if let Some(result) = expand_expr_modp_materialized_hold(ctx, arg) {
                        let new_expr = crate::strip_all_holds(ctx, result);
                        return Some(Rewrite::new(new_expr).desc("expand() [mod-p fast path]"));
                    }
                    // Fall through to slow path if mod-p fails
                }

                let expanded = crate::expand::expand(ctx, arg);
                // Strip all nested __hold wrappers so user sees clean result
                let new_expr = crate::strip_all_holds(ctx, expanded);
                if new_expr != expr {
                    return Some(Rewrite::new(new_expr).desc("expand()"));
                } else {
                    return Some(Rewrite::new(arg).desc("expand(atom)"));
                }
            }
        }
        None
    }
);

// ConservativeExpandRule: only runs in Transform phase
define_rule!(
    ConservativeExpandRule,
    "Conservative Expand",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Expand)) && args.len() == 1
            {
                let arg = args[0];
                let expanded = crate::expand::expand(ctx, arg);
                // Strip all nested __hold wrappers so user sees clean result
                let new_expr = crate::strip_all_holds(ctx, expanded);
                if new_expr != expr {
                    return Some(Rewrite::new(new_expr).desc("expand()"));
                } else {
                    return Some(Rewrite::new(arg).desc("expand(atom)"));
                }
            }
        }

        // Implicit expansion (e.g. (x+1)^2)
        // Only expand if complexity does not increase
        let expanded_raw = crate::expand::expand(ctx, expr);
        // Strip all nested __hold wrappers
        let new_expr = crate::strip_all_holds(ctx, expanded_raw);
        if new_expr != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, new_expr);

            if new_count <= old_count {
                if crate::ordering::compare_expr(ctx, new_expr, expr) == std::cmp::Ordering::Equal {
                    return None;
                }
                return Some(Rewrite::new(new_expr).desc("Conservative Expansion"));
            }
        }
        None
    }
);

// DistributeRule: only runs in Transform phase
define_rule!(
    DistributeRule,
    "Distributive Property (Simple)",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l_id = *l;
            let r_id = *r;

            if matches!(ctx.get(r_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
                let new_expr = distribute(ctx, r_id, l_id);
                if new_expr != expr {
                    return Some(Rewrite::new(new_expr).desc("Distribute (RHS)"));
                }
            }
            if matches!(ctx.get(l_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
                let new_expr = distribute(ctx, l_id, r_id);
                if new_expr != expr {
                    return Some(Rewrite::new(new_expr).desc("Distribute (LHS)"));
                }
            }
        }
        None
    }
);
