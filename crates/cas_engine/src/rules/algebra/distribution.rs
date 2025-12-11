use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::expression::count_nodes;
use cas_ast::Expr;

use super::helpers::*;

define_rule!(ExpandRule, "Expand Polynomial", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "expand" && args.len() == 1 {
            let arg = args[0];
            let new_expr = crate::expand::expand(ctx, arg);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "expand()".to_string(),
                    before_local: None,
                    after_local: None,
                });
            } else {
                // If expand didn't change anything, maybe we should just unwrap?
                // "expand(x)" -> "x"
                return Some(Rewrite {
                    new_expr: arg,
                    description: "expand(atom)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(
    ConservativeExpandRule,
    "Conservative Expand",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            // If explicit expand() call, always expand
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                let new_expr = crate::expand::expand(ctx, arg);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                } else {
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }

        // Implicit expansion (e.g. (x+1)^2)
        // Only expand if complexity does not increase
        let new_expr = crate::expand::expand(ctx, expr);
        if new_expr != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, new_expr);

            if new_count <= old_count {
                // Check for structural equality to avoid loops with ID regeneration
                if crate::ordering::compare_expr(ctx, new_expr, expr) == std::cmp::Ordering::Equal {
                    return None;
                }
                return Some(Rewrite {
                    new_expr,
                    description: "Conservative Expansion".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        None
    }
);

define_rule!(DistributeRule, "Distributive Property", |ctx, expr| {
    // Skip canonical (elegant) forms - even in aggressive mode
    // e.g., (x+y)*(x-y) should stay factored, not be distributed
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let l_id = *l;
        let r_id = *r;

        // Try to distribute l into r if r is an Add/Sub
        if matches!(ctx.get(r_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
            let new_expr = distribute(ctx, r_id, l_id);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute (RHS)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        // Try to distribute r into l if l is an Add/Sub
        if matches!(ctx.get(l_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
            let new_expr = distribute(ctx, l_id, r_id);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute (LHS)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});
