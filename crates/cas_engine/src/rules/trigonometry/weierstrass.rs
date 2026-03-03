//! Weierstrass Substitution Rule
//!
//! Implements the half-angle tangent substitution (Weierstrass substitution)
//! which converts trigonometric expressions into rational polynomial expressions.
//!
//! The substitution: t = tan(x/2)
//!
//! Transformations:
//! - sin(x) → 2t/(1+t²)
//! - cos(x) → (1-t²)/(1+t²)
//! - tan(x) → 2t/(1-t²)

use crate::define_rule;
use cas_formatter::DisplayExpr;
use cas_math::trig_weierstrass_support::{
    try_rewrite_reverse_weierstrass_sin_expr, try_rewrite_weierstrass_substitution_function_expr,
    WeierstrassSubstitutionKind,
};

define_rule!(
    WeierstrassSubstitutionRule,
    "Weierstrass Substitution",
    |ctx, expr| {
        let rewrite = try_rewrite_weierstrass_substitution_function_expr(ctx, expr)?;
        let arg = rewrite.arg;

        let desc = match rewrite.kind {
            WeierstrassSubstitutionKind::Sin => format!(
                "Weierstrass: sin({}) = 2t/(1+t²) where t = tan({}/2)",
                DisplayExpr {
                    context: ctx,
                    id: arg
                },
                DisplayExpr {
                    context: ctx,
                    id: arg
                }
            ),
            WeierstrassSubstitutionKind::Cos => format!(
                "Weierstrass: cos({}) = (1-t²)/(1+t²) where t = tan({}/2)",
                DisplayExpr {
                    context: ctx,
                    id: arg
                },
                DisplayExpr {
                    context: ctx,
                    id: arg
                }
            ),
            WeierstrassSubstitutionKind::Tan => format!(
                "Weierstrass: tan({}) = 2t/(1-t²) where t = tan({}/2)",
                DisplayExpr {
                    context: ctx,
                    id: arg
                },
                DisplayExpr {
                    context: ctx,
                    id: arg
                }
            ),
        };

        Some(crate::rule::Rewrite::new(rewrite.rewritten).desc(desc))
    }
);

// Reverse Weierstrass: Convert 2t/(1+t²) back to sin(x) when t = tan(x/2)
define_rule!(
    ReverseWeierstrassRule,
    "Reverse Weierstrass",
    |ctx, expr| {
        let rewrite = try_rewrite_reverse_weierstrass_sin_expr(ctx, expr)?;
        Some(crate::rule::Rewrite::new(rewrite.rewritten).desc_lazy(|| {
            format!(
                "Reverse Weierstrass: 2t/(1+t²) = sin({})",
                DisplayExpr {
                    context: ctx,
                    id: rewrite.arg
                }
            )
        }))
    }
);
