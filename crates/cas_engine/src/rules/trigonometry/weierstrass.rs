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
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_weierstrass_support::extract_tan_half_angle_like;
use num_traits::One;

/// Build the Weierstrass expression for sin(x): 2t/(1+t²)
fn weierstrass_sin(ctx: &mut cas_ast::Context, t: ExprId) -> ExprId {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = smart_mul(ctx, two, t);
    let denominator = ctx.add(Expr::Add(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

/// Build the Weierstrass expression for cos(x): (1-t²)/(1+t²)
fn weierstrass_cos(ctx: &mut cas_ast::Context, t: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = ctx.add(Expr::Sub(one, t_squared));
    let denominator = ctx.add(Expr::Add(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

/// Build the Weierstrass expression for tan(x): 2t/(1-t²)
fn weierstrass_tan(ctx: &mut cas_ast::Context, t: ExprId) -> ExprId {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let t_squared = ctx.add(Expr::Pow(t, two));
    let numerator = smart_mul(ctx, two, t);
    let denominator = ctx.add(Expr::Sub(one, t_squared));
    ctx.add(Expr::Div(numerator, denominator))
}

define_rule!(
    WeierstrassSubstitutionRule,
    "Weierstrass Substitution",
    |ctx, expr| {
        // Extract data first to avoid borrow conflicts
        let (fn_id, arg) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let builtin = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };
        let name = builtin.name();

        // Build t = tan(x/2) as sin(x/2)/cos(x/2)
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let half_arg = smart_mul(ctx, half, arg);
        let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_arg]);
        let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_arg]);
        let t = ctx.add(Expr::Div(sin_half, cos_half)); // t = tan(x/2)

        let (new_expr, desc) = match name {
            "sin" => {
                let result = weierstrass_sin(ctx, t);
                let desc = format!(
                    "Weierstrass: sin({}) = 2t/(1+t²) where t = tan({}/2)",
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    },
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    }
                );
                (result, desc)
            }
            "cos" => {
                let result = weierstrass_cos(ctx, t);
                let desc = format!(
                    "Weierstrass: cos({}) = (1-t²)/(1+t²) where t = tan({}/2)",
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    },
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    }
                );
                (result, desc)
            }
            "tan" => {
                let result = weierstrass_tan(ctx, t);
                let desc = format!(
                    "Weierstrass: tan({}) = 2t/(1-t²) where t = tan({}/2)",
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    },
                    DisplayExpr {
                        context: ctx,
                        id: arg
                    }
                );
                (result, desc)
            }
            _ => return None,
        };

        Some(crate::rule::Rewrite::new(new_expr).desc(desc))
    }
);

// Reverse Weierstrass: Convert 2t/(1+t²) back to sin(x) when t = tan(x/2)
define_rule!(
    ReverseWeierstrassRule,
    "Reverse Weierstrass",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Pattern: 2*tan(x/2) / (1 + tan(x/2)²) → sin(x)
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Check numerator: 2*t
        let t_id = match ctx.get(num) {
            Expr::Mul(a, b) => {
                if let Expr::Number(n) = ctx.get(*a) {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        *b
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Check if t is tan(x/2)
        let original_angle = extract_tan_half_angle_like(ctx, t_id)?;

        // Check denominator: 1 + t²
        let (one_id, t_sq_id) = match ctx.get(den) {
            Expr::Add(a, b) => (*a, *b),
            _ => return None,
        };

        // Verify one_id is 1
        if let Expr::Number(n) = ctx.get(one_id) {
            if !n.is_one() {
                return None;
            }
        } else {
            return None;
        }

        // Verify t_sq_id is t²
        if let Expr::Pow(base, exp) = ctx.get(t_sq_id) {
            if *base != t_id {
                return None;
            }
            if let Expr::Number(e) = ctx.get(*exp) {
                if *e != num_rational::BigRational::from_integer(2.into()) {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            return None;
        }

        // Match! This is sin(x)
        let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![original_angle]);
        Some(crate::rule::Rewrite::new(sin_x).desc_lazy(|| {
            format!(
                "Reverse Weierstrass: 2t/(1+t²) = sin({})",
                DisplayExpr {
                    context: ctx,
                    id: original_angle
                }
            )
        }))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_weierstrass_sin() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let t = weierstrass_sin(&mut ctx, x);
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
}
