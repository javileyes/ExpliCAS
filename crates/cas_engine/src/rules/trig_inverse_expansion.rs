use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::Ratio;

// ========== Helper Functions ==========

/// Build sqrt(expr) = expr^(1/2)
fn build_sqrt(ctx: &mut Context, expr: ExprId) -> ExprId {
    let half = ctx.add(Expr::Number(Ratio::new(BigInt::from(1), BigInt::from(2))));
    ctx.add(Expr::Pow(expr, half))
}

/// Build 1 - x²
fn build_one_minus_x_sq(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Sub(one, x_sq))
}

/// Build 1 + x²
fn build_one_plus_x_sq(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Add(one, x_sq))
}

/// Build x² - 1
fn build_x_sq_minus_one(ctx: &mut Context, x: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x, two));
    ctx.add(Expr::Sub(x_sq, one))
}

// ========== Priority 1: Core Expansion Rules ==========

/// sin(arctan(x)) → x/sqrt(1+x²)
define_rule!(
    SinArctanExpansionRule,
    "sin(arctan(x)) → x/√(1+x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "sin" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arctan" || inner_name == "atan") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // x / sqrt(1 + x²)
                        let denom = build_one_plus_x_sq(ctx, x);
                        let sqrt_denom = build_sqrt(ctx, denom);
                        let result = ctx.add(Expr::Div(x, sqrt_denom));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "sin(arctan(x)) → x/√(1+x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// cos(arctan(x)) → 1/sqrt(1+x²)
define_rule!(
    CosArctanExpansionRule,
    "cos(arctan(x)) → 1/√(1+x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "cos" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arctan" || inner_name == "atan") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // 1 / sqrt(1 + x²)
                        let one = ctx.num(1);
                        let denom = build_one_plus_x_sq(ctx, x);
                        let sqrt_denom = build_sqrt(ctx, denom);
                        let result = ctx.add(Expr::Div(one, sqrt_denom));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "cos(arctan(x)) → 1/√(1+x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// tan(arcsin(x)) → x/sqrt(1-x²)
define_rule!(
    TanArcsinExpansionRule,
    "tan(arcsin(x)) → x/√(1-x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "tan" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsin" || inner_name == "asin") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // x / sqrt(1 - x²)
                        let denom = build_one_minus_x_sq(ctx, x);
                        let sqrt_denom = build_sqrt(ctx, denom);
                        let result = ctx.add(Expr::Div(x, sqrt_denom));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "tan(arcsin(x)) → x/√(1-x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// cot(arcsin(x)) → sqrt(1-x²)/x
define_rule!(
    CotArcsinExpansionRule,
    "cot(arcsin(x)) → √(1-x²)/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "cot" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsin" || inner_name == "asin") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(1 - x²) / x
                        let numer = build_one_minus_x_sq(ctx, x);
                        let sqrt_numer = build_sqrt(ctx, numer);
                        let result = ctx.add(Expr::Div(sqrt_numer, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "cot(arcsin(x)) → √(1-x²)/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

// ========== Priority 2: Secondary Expansion Rules ==========

/// sin(arcsec(x)) → sqrt(x²-1)/x
define_rule!(
    SinArcsecExpansionRule,
    "sin(arcsec(x)) → √(x²-1)/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "sin" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsec" || inner_name == "asec") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(x² - 1) / x
                        let numer = build_x_sq_minus_one(ctx, x);
                        let sqrt_numer = build_sqrt(ctx, numer);
                        let result = ctx.add(Expr::Div(sqrt_numer, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "sin(arcsec(x)) → √(x²-1)/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// cos(arcsec(x)) → 1/x
define_rule!(
    CosArcsecExpansionRule,
    "cos(arcsec(x)) → 1/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "cos" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsec" || inner_name == "asec") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // 1 / x
                        let one = ctx.num(1);
                        let result = ctx.add(Expr::Div(one, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "cos(arcsec(x)) → 1/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// tan(arccos(x)) → sqrt(1-x²)/x
define_rule!(
    TanArccosExpansionRule,
    "tan(arccos(x)) → √(1-x²)/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "tan" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arccos" || inner_name == "acos") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(1 - x²) / x
                        let numer = build_one_minus_x_sq(ctx, x);
                        let sqrt_numer = build_sqrt(ctx, numer);
                        let result = ctx.add(Expr::Div(sqrt_numer, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "tan(arccos(x)) → √(1-x²)/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// cot(arccos(x)) → x/sqrt(1-x²)
define_rule!(
    CotArccosExpansionRule,
    "cot(arccos(x)) → x/√(1-x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "cot" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arccos" || inner_name == "acos") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // x / sqrt(1 - x²)
                        let denom = build_one_minus_x_sq(ctx, x);
                        let sqrt_denom = build_sqrt(ctx, denom);
                        let result = ctx.add(Expr::Div(x, sqrt_denom));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "cot(arccos(x)) → x/√(1-x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

// ========== Priority 3: Reciprocal Expansion Rules ==========

/// sec(arctan(x)) → sqrt(1+x²)
define_rule!(
    SecArctanExpansionRule,
    "sec(arctan(x)) → √(1+x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "sec" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arctan" || inner_name == "atan") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(1 + x²)
                        let expr_inside = build_one_plus_x_sq(ctx, x);
                        let result = build_sqrt(ctx, expr_inside);
                        return Some(Rewrite {
                            new_expr: result,
                            description: "sec(arctan(x)) → √(1+x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// csc(arctan(x)) → sqrt(1+x²)/x
define_rule!(
    CscArctanExpansionRule,
    "csc(arctan(x)) → √(1+x²)/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "csc" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arctan" || inner_name == "atan") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // sqrt(1 + x²) / x
                        let numer = build_one_plus_x_sq(ctx, x);
                        let sqrt_numer = build_sqrt(ctx, numer);
                        let result = ctx.add(Expr::Div(sqrt_numer, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "csc(arctan(x)) → √(1+x²)/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// sec(arcsin(x)) → 1/sqrt(1-x²)
define_rule!(
    SecArcsinExpansionRule,
    "sec(arcsin(x)) → 1/√(1-x²)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "sec" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsin" || inner_name == "asin") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // 1 / sqrt(1 - x²)
                        let one = ctx.num(1);
                        let denom = build_one_minus_x_sq(ctx, x);
                        let sqrt_denom = build_sqrt(ctx, denom);
                        let result = ctx.add(Expr::Div(one, sqrt_denom));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "sec(arcsin(x)) → 1/√(1-x²)".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

/// csc(arcsin(x)) → 1/x
define_rule!(
    CscArcsinExpansionRule,
    "csc(arcsin(x)) → 1/x",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_name == "csc" && outer_args.len() == 1 {
                let inner = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                    if (inner_name == "arcsin" || inner_name == "asin") && inner_args.len() == 1 {
                        let x = inner_args[0];
                        // 1 / x
                        let one = ctx.num(1);
                        let result = ctx.add(Expr::Div(one, x));
                        return Some(Rewrite {
                            new_expr: result,
                            description: "csc(arcsin(x)) → 1/x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

// ========== Registration ==========

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    // Priority 1: Most common expansions
    simplifier.add_rule(Box::new(SinArctanExpansionRule));
    simplifier.add_rule(Box::new(CosArctanExpansionRule));
    simplifier.add_rule(Box::new(TanArcsinExpansionRule));
    simplifier.add_rule(Box::new(CotArcsinExpansionRule));

    // Priority 2: Secondary expansions
    simplifier.add_rule(Box::new(SinArcsecExpansionRule));
    simplifier.add_rule(Box::new(CosArcsecExpansionRule));
    simplifier.add_rule(Box::new(TanArccosExpansionRule));
    simplifier.add_rule(Box::new(CotArccosExpansionRule));

    // Priority 3: Reciprocal expansions
    simplifier.add_rule(Box::new(SecArctanExpansionRule));
    simplifier.add_rule(Box::new(CscArctanExpansionRule));
    simplifier.add_rule(Box::new(SecArcsinExpansionRule));
    simplifier.add_rule(Box::new(CscArcsinExpansionRule));
}
