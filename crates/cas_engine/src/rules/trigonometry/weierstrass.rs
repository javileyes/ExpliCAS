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
#[cfg(test)]
use cas_math::trig_weierstrass_support::{build_weierstrass_sin, extract_tan_half_angle_like};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::SimpleRule;
    use cas_ast::Context;
    use cas_formatter::render_expr;
    use cas_parser::parse;

    #[test]
    fn test_weierstrass_sin() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let t = build_weierstrass_sin(&mut ctx, x);
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: t
            }
        );
        assert!(result.contains("2") && result.contains("x"));
    }

    #[test]
    fn test_extract_tan_half_angle_like() {
        let mut ctx = Context::new();
        let expr = parse("sin(x/2) / cos(x/2)", &mut ctx).unwrap();
        let result = extract_tan_half_angle_like(&ctx, expr);
        assert!(result.is_some());
    }

    #[test]
    fn test_reverse_weierstrass_accepts_commuted_mul() {
        let mut ctx = Context::new();
        let expr = parse("(tan(x/2)*2)/(1+tan(x/2)^2)", &mut ctx).unwrap();
        let expected = parse("sin(x)", &mut ctx).unwrap();

        let rule = ReverseWeierstrassRule;
        let rewrite = rule
            .apply_simple(&mut ctx, expr)
            .expect("reverse weierstrass should match");

        assert_eq!(
            render_expr(&ctx, rewrite.new_expr),
            render_expr(&ctx, expected)
        );
    }
}
