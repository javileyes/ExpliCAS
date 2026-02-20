//! Rule wrapper for exact polynomial GCD over ℚ[x1,...,xn].

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_formatter::DisplayExpr;
use cas_math::gcd_exact::{gcd_exact, GcdExactBudget};

// Rule for poly_gcd_exact(a, b) function.
// Computes algebraic GCD of two polynomial expressions over ℚ.
define_rule!(
    PolyGcdExactRule,
    "Polynomial GCD Exact",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            (*fn_id, args.clone())
        } else {
            return None;
        };
        {
            let name = ctx.sym_name(fn_id);
            // Match poly_gcd_exact, pgcdx with 2 arguments
            let is_gcd_exact = name == "poly_gcd_exact" || name == "pgcdx";

            if is_gcd_exact && args.len() == 2 {
                let a = args[0];
                let b = args[1];

                let result = gcd_exact(ctx, a, b, &GcdExactBudget::default());

                return Some(Rewrite::simple(
                    result.gcd,
                    format!(
                        "poly_gcd_exact({}, {}) [{:?}]",
                        DisplayExpr {
                            context: ctx,
                            id: a
                        },
                        DisplayExpr {
                            context: ctx,
                            id: b
                        },
                        result.layer_used
                    ),
                ));
            }
        }

        None
    }
);
