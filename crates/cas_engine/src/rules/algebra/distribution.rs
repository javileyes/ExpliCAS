use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::Expr;

use super::helpers::*;

// ExpandRule: only runs in Transform phase
define_rule!(
    ExpandRule,
    "Expand Polynomial",
    None,
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                let expanded = crate::expand::expand(ctx, arg);
                // Strip all nested __hold wrappers so user sees clean result
                let new_expr = crate::strip_all_holds(ctx, expanded);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                } else {
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                let expanded = crate::expand::expand(ctx, arg);
                // Strip all nested __hold wrappers so user sees clean result
                let new_expr = crate::strip_all_holds(ctx, expanded);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                } else {
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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
                return Some(Rewrite {
                    new_expr,
                    description: "Conservative Expansion".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
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
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute (RHS)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }
            if matches!(ctx.get(l_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
                let new_expr = distribute(ctx, l_id, r_id);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute (LHS)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
                }
            }
        }
        None
    }
);
