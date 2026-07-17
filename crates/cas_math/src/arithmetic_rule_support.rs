//! Planning helpers for arithmetic rewrite rules.

use crate::expr_destructure::{as_add, as_div, as_mul, as_sub};
use crate::expr_predicates::is_minus_one_expr;
use crate::expr_rewrite::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct ArithmeticRewritePlan {
    pub rewritten: ExprId,
    pub description: String,
}

/// `x + 0 = x` and `0 + x = x` (the zero may be a `decimal(0)` display
/// wrapper — the identity result stays EXACT, sticky deliberately dropped:
/// the numeric operand vanishes and display is unchanged).
pub fn try_rewrite_add_zero_expr(ctx: &Context, expr: ExprId) -> Option<ArithmeticRewritePlan> {
    let (lhs, rhs) = as_add(ctx, expr)?;

    if let Some((n, _)) = as_fold_number(ctx, rhs) {
        if n.is_zero() {
            return Some(ArithmeticRewritePlan {
                rewritten: lhs,
                description: "x + 0 = x".to_string(),
            });
        }
    }
    if let Some((n, _)) = as_fold_number(ctx, lhs) {
        if n.is_zero() {
            return Some(ArithmeticRewritePlan {
                rewritten: rhs,
                description: "0 + x = x".to_string(),
            });
        }
    }
    None
}

/// `x * 1 = x` and `1 * x = x` (the one may be a `decimal(1)` display
/// wrapper — same exact-identity rationale as `try_rewrite_add_zero_expr`).
pub fn try_rewrite_mul_one_expr(ctx: &Context, expr: ExprId) -> Option<ArithmeticRewritePlan> {
    let (lhs, rhs) = as_mul(ctx, expr)?;

    if let Some((n, _)) = as_fold_number(ctx, rhs) {
        if n.is_one() {
            return Some(ArithmeticRewritePlan {
                rewritten: lhs,
                description: "x * 1 = x".to_string(),
            });
        }
    }
    if let Some((n, _)) = as_fold_number(ctx, lhs) {
        if n.is_one() {
            return Some(ArithmeticRewritePlan {
                rewritten: rhs,
                description: "1 * x = x".to_string(),
            });
        }
    }
    None
}

/// A numeric leaf for constant folding: a bare `Number` or a
/// `decimal(Number)` display wrapper (the `approx()` presentation node).
/// `sticky` records the wrapper so fold RESULTS keep the decimal display —
/// the "display-preference contagion" of the numeric-display layer. The
/// payload is exact either way; this never introduces f64 semantics.
fn as_fold_number(ctx: &Context, id: ExprId) -> Option<(BigRational, bool)> {
    match ctx.get(id) {
        Expr::Number(n) => Some((n.clone(), false)),
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.sym_name(*fn_id) == "decimal" => {
            match ctx.get(args[0]) {
                Expr::Number(n) => Some((n.clone(), true)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Rebuild a folded numeric result, re-wrapping in `decimal` iff any
/// operand carried the display preference.
fn wrap_fold_number(ctx: &mut Context, value: BigRational, sticky: bool) -> ExprId {
    let number = ctx.add(Expr::Number(value));
    if !sticky {
        return number;
    }
    let decimal_sym = ctx.intern_symbol("decimal");
    ctx.add(Expr::Function(decimal_sym, vec![number]))
}

/// Fold-description rendering: decimal when the operand is display-decimal
/// (didactic text must never leak `428571428571/1000000000000`).
fn fold_desc_number(value: &BigRational, sticky: bool) -> String {
    if sticky {
        crate::decimal_display::decimal_display_string(value)
    } else {
        format!("{value}")
    }
}

/// Try to combine numeric constants inside Add/Sub/Mul/Div expressions.
pub fn try_rewrite_combine_constants_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    if let Some((lhs, rhs)) = as_add(ctx, expr) {
        if let (Some((n1, s1)), Some((n2, s2))) =
            (as_fold_number(ctx, lhs), as_fold_number(ctx, rhs))
        {
            let sticky = s1 || s2;
            let sum = &n1 + &n2;
            let description = if n2 < BigRational::from_integer(0.into()) {
                let abs_n2 = -&n2;
                format!(
                    "{} - {} = {}",
                    fold_desc_number(&n1, s1),
                    fold_desc_number(&abs_n2, s2),
                    fold_desc_number(&sum, sticky)
                )
            } else {
                format!(
                    "{} + {} = {}",
                    fold_desc_number(&n1, s1),
                    fold_desc_number(&n2, s2),
                    fold_desc_number(&sum, sticky)
                )
            };
            return Some(ArithmeticRewritePlan {
                rewritten: wrap_fold_number(ctx, sum, sticky),
                description,
            });
        }

        if let Some((n1, s1)) = as_fold_number(ctx, lhs) {
            if let Some((rl, rr)) = as_add(ctx, rhs) {
                if let Some((n2, s2)) = as_fold_number(ctx, rl) {
                    let sticky = s1 || s2;
                    let sum = &n1 + &n2;
                    let description = format!(
                        "Combine nested constants: {} + {}",
                        fold_desc_number(&n1, s1),
                        fold_desc_number(&n2, s2)
                    );
                    let sum_expr = wrap_fold_number(ctx, sum, sticky);
                    return Some(ArithmeticRewritePlan {
                        rewritten: ctx.add(Expr::Add(sum_expr, rr)),
                        description,
                    });
                }
            }
        }
    }

    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        if let (Some((n1, s1)), Some((n2, s2))) =
            (as_fold_number(ctx, lhs), as_fold_number(ctx, rhs))
        {
            let sticky = s1 || s2;
            let prod = &n1 * &n2;
            let description = format!(
                "{} * {} = {}",
                fold_desc_number(&n1, s1),
                fold_desc_number(&n2, s2),
                fold_desc_number(&prod, sticky)
            );
            return Some(ArithmeticRewritePlan {
                rewritten: wrap_fold_number(ctx, prod, sticky),
                description,
            });
        }

        let mut factors = Vec::new();
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(id) {
                stack.push(*r);
                stack.push(*l);
            } else {
                factors.push(id);
            }
        }

        let mut numeric_factors: Vec<(BigRational, bool)> = Vec::new();
        let mut non_numeric: Vec<ExprId> = Vec::new();

        for &factor in &factors {
            if let Some(pair) = as_fold_number(ctx, factor) {
                numeric_factors.push(pair);
            } else {
                non_numeric.push(factor);
            }
        }

        if numeric_factors.len() >= 2 {
            let sticky = numeric_factors.iter().any(|(_, s)| *s);
            let product = numeric_factors
                .iter()
                .fold(BigRational::from_integer(1.into()), |acc, (n, _)| acc * n);

            if product.is_zero() {
                // Annihilator stays EXACT (sticky deliberately dropped): the
                // numeric operand vanishes from the result and exact-0 keeps
                // every downstream zero-test sound. Display is identical.
                return Some(ArithmeticRewritePlan {
                    rewritten: ctx.num(0),
                    description: "0 * x = 0".to_string(),
                });
            }

            let rewritten = if product.is_one() && !non_numeric.is_empty() {
                // Identity vanishes: exact result, sticky dropped (same
                // rationale as the annihilator).
                let mut result = non_numeric[0];
                for &factor in &non_numeric[1..] {
                    result = smart_mul(ctx, result, factor);
                }
                result
            } else if non_numeric.is_empty() {
                wrap_fold_number(ctx, product.clone(), sticky)
            } else {
                let prod_expr = wrap_fold_number(ctx, product.clone(), sticky);
                let mut result = prod_expr;
                for &factor in &non_numeric {
                    result = smart_mul(ctx, result, factor);
                }
                result
            };

            let nums: Vec<String> = numeric_factors
                .iter()
                .map(|(n, s)| fold_desc_number(n, *s))
                .collect();
            return Some(ArithmeticRewritePlan {
                rewritten,
                description: format!(
                    "Combine nested constants: {} = {}",
                    nums.join(" * "),
                    fold_desc_number(&product, sticky)
                ),
            });
        }

        if let Some((n1, s1)) = as_fold_number(ctx, lhs) {
            if let Some((rl, rr)) = as_mul(ctx, rhs) {
                if let Some((n2, s2)) = as_fold_number(ctx, rl) {
                    let sticky = s1 || s2;
                    let prod = &n1 * &n2;
                    let description = format!(
                        "Combine nested constants: {} * {}",
                        fold_desc_number(&n1, s1),
                        fold_desc_number(&n2, s2)
                    );
                    let prod_expr = wrap_fold_number(ctx, prod, sticky);
                    return Some(ArithmeticRewritePlan {
                        rewritten: smart_mul(ctx, prod_expr, rr),
                        description,
                    });
                }
            }

            if let Some((num, den)) = as_div(ctx, rhs) {
                if let Some((n2, s2)) = as_fold_number(ctx, den) {
                    if !n2.is_zero() {
                        let sticky = s1 || s2;
                        let ratio = &n1 / &n2;
                        let description = format!(
                            "{} * (x / {}) -> ({} / {}) * x",
                            fold_desc_number(&n1, s1),
                            fold_desc_number(&n2, s2),
                            fold_desc_number(&n1, s1),
                            fold_desc_number(&n2, s2)
                        );
                        let ratio_expr = wrap_fold_number(ctx, ratio, sticky);
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, num),
                            description,
                        });
                    }
                }
            }
        }
    }

    if let Some((lhs, rhs)) = as_sub(ctx, expr) {
        if let (Some((n1, s1)), Some((n2, s2))) =
            (as_fold_number(ctx, lhs), as_fold_number(ctx, rhs))
        {
            let sticky = s1 || s2;
            let diff = &n1 - &n2;
            let description = format!(
                "{} - {} = {}",
                fold_desc_number(&n1, s1),
                fold_desc_number(&n2, s2),
                fold_desc_number(&diff, sticky)
            );
            return Some(ArithmeticRewritePlan {
                rewritten: wrap_fold_number(ctx, diff, sticky),
                description,
            });
        }
    }

    if let Some((lhs, rhs)) = as_div(ctx, expr) {
        if let (Some((n1, s1)), Some((n2, s2))) =
            (as_fold_number(ctx, lhs), as_fold_number(ctx, rhs))
        {
            let sticky = s1 || s2;
            if !n2.is_zero() {
                let quot = &n1 / &n2;
                let description = format!(
                    "{} / {} = {}",
                    fold_desc_number(&n1, s1),
                    fold_desc_number(&n2, s2),
                    fold_desc_number(&quot, sticky)
                );
                return Some(ArithmeticRewritePlan {
                    rewritten: wrap_fold_number(ctx, quot, sticky),
                    description,
                });
            }
            // A rounded-to-zero payload IS zero by the WYSIWYG contract:
            // division by decimal(0) is division by zero.
            return Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Constant(cas_ast::Constant::Undefined)),
                description: "Division by zero".to_string(),
            });
        }

        if let Some((d, sd)) = as_fold_number(ctx, rhs) {
            if !d.is_zero() {
                if let Some((ml, mr)) = as_mul(ctx, lhs) {
                    if let Some((c, sc)) = as_fold_number(ctx, ml) {
                        let sticky = sc || sd;
                        let ratio = &c / &d;
                        let description = format!(
                            "({} * x) / {} -> ({} / {}) * x",
                            fold_desc_number(&c, sc),
                            fold_desc_number(&d, sd),
                            fold_desc_number(&c, sc),
                            fold_desc_number(&d, sd)
                        );
                        let ratio_expr = wrap_fold_number(ctx, ratio, sticky);
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, mr),
                            description,
                        });
                    }

                    if let Some((c, sc)) = as_fold_number(ctx, mr) {
                        let sticky = sc || sd;
                        let ratio = &c / &d;
                        let description = format!(
                            "(x * {}) / {} -> ({} / {}) * x",
                            fold_desc_number(&c, sc),
                            fold_desc_number(&d, sd),
                            fold_desc_number(&c, sc),
                            fold_desc_number(&d, sd)
                        );
                        let ratio_expr = wrap_fold_number(ctx, ratio, sticky);
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, ml),
                            description,
                        });
                    }
                }
            }
        }
    }

    None
}

/// Try to simplify numeric sums in exponents:
/// `x^(1/2 + 1/3) -> x^(5/6)`.
pub fn try_rewrite_simplify_numeric_exponents_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;

    let mut addends: Vec<BigRational> = Vec::new();
    let mut stack = vec![exp];
    let mut all_numeric = true;

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Number(n) => addends.push(n.clone()),
            Expr::Div(num, den) => {
                if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                    if !d.is_zero() {
                        addends.push(n / d);
                    } else {
                        all_numeric = false;
                    }
                } else {
                    all_numeric = false;
                }
            }
            _ => {
                all_numeric = false;
            }
        }
    }

    if !(all_numeric && addends.len() >= 2) {
        return None;
    }

    let sum: BigRational = addends.iter().sum();
    let new_exp = ctx.add(Expr::Number(sum.clone()));
    let rewritten = ctx.add(Expr::Pow(base, new_exp));

    let addend_strs: Vec<String> = addends
        .iter()
        .map(|r| {
            if r.is_integer() {
                format!("{}", r.numer())
            } else {
                format!("({}/{})", r.numer(), r.denom())
            }
        })
        .collect();
    let sum_str = if sum.is_integer() {
        format!("{}", sum.numer())
    } else {
        format!("{}/{}", sum.numer(), sum.denom())
    };

    Some(ArithmeticRewritePlan {
        rewritten,
        description: format!("{} = {}", addend_strs.join(" + "), sum_str),
    })
}

/// Normalize negation placement inside products.
pub fn try_rewrite_normalize_mul_neg_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let l = *l;
    let r = *r;

    let one = ctx.num(1);
    let l_neg = if let Expr::Neg(inner) = ctx.get(l) {
        Some(*inner)
    } else if is_minus_one_expr(ctx, l) {
        Some(one)
    } else {
        None
    };
    let r_neg = if let Expr::Neg(inner) = ctx.get(r) {
        Some(*inner)
    } else if is_minus_one_expr(ctx, r) {
        Some(one)
    } else {
        None
    };

    match (l_neg, r_neg) {
        (Some(a), Some(b)) => {
            let new_mul = smart_mul(ctx, a, b);
            Some(ArithmeticRewritePlan {
                rewritten: new_mul,
                description: "(-a) * (-b) = a * b".to_string(),
            })
        }
        (Some(a), None) => {
            let new_mul = smart_mul(ctx, a, r);
            Some(ArithmeticRewritePlan {
                rewritten: if new_mul == one {
                    ctx.num(-1)
                } else {
                    ctx.add(Expr::Neg(new_mul))
                },
                description: "(-a) * b = -(a * b)".to_string(),
            })
        }
        (None, Some(b)) => {
            let new_mul = smart_mul(ctx, l, b);
            Some(ArithmeticRewritePlan {
                rewritten: if new_mul == one {
                    ctx.num(-1)
                } else {
                    ctx.add(Expr::Neg(new_mul))
                },
                description: "a * (-b) = -(a * b)".to_string(),
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_add_zero_expr, try_rewrite_combine_constants_expr, try_rewrite_mul_one_expr,
        try_rewrite_normalize_mul_neg_expr, try_rewrite_simplify_numeric_exponents_expr,
    };
    use cas_ast::{Context, Expr, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use num_traits::Zero as _;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn combine_constants_add() {
        let mut ctx = Context::new();
        let expr = parse("2 + 3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "5");
    }

    #[test]
    fn add_zero_identity() {
        let mut ctx = Context::new();
        let expr = parse("x + 0", &mut ctx).expect("parse");
        let rewrite = try_rewrite_add_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "x");
    }

    #[test]
    fn mul_one_identity() {
        let mut ctx = Context::new();
        let expr = parse("1 * y", &mut ctx).expect("parse");
        let rewrite = try_rewrite_mul_one_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "y");
    }

    #[test]
    fn combine_constants_mul_chain() {
        let mut ctx = Context::new();
        let expr = parse("(2 * x) * 3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "6 * x");
    }

    #[test]
    fn simplify_numeric_exponents_sum() {
        let mut ctx = Context::new();
        let expr = parse("x^((1/2)+(1/3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_simplify_numeric_exponents_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "x^(5/6)");
    }

    #[test]
    fn normalize_mul_neg_right() {
        let mut ctx = Context::new();
        let expr = parse("a * (-b)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_normalize_mul_neg_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "-(a * b)");
    }

    fn decimal_of(ctx: &mut Context, n: i64, d: i64) -> ExprId {
        let number = ctx.add(Expr::Number(num_rational::BigRational::new(
            n.into(),
            d.into(),
        )));
        let sym = ctx.intern_symbol("decimal");
        ctx.add(Expr::Function(sym, vec![number]))
    }

    fn payload_of(ctx: &Context, id: ExprId) -> Option<(num_rational::BigRational, bool)> {
        super::as_fold_number(ctx, id)
    }

    #[test]
    fn combine_constants_folds_through_decimal_and_stays_sticky() {
        // decimal(3/7) - decimal-free exact rational: folds EXACTLY, result
        // keeps the decimal display preference (sticky contagion).
        let mut ctx = Context::new();
        let dec = decimal_of(&mut ctx, 3, 7);
        let exact = ctx.add(Expr::Number(num_rational::BigRational::new(
            3.into(),
            7.into(),
        )));
        let expr = ctx.add(Expr::Sub(dec, exact));
        let plan = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("folds");
        let (value, sticky) = payload_of(&ctx, plan.rewritten).expect("numeric result");
        assert!(value.is_zero());
        assert!(sticky, "result must keep the decimal display preference");

        // Sticky descriptions render decimally, never as huge fractions.
        let mut ctx = Context::new();
        let dec = decimal_of(&mut ctx, 428571428571, 1_000_000_000_000);
        let third = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            3.into(),
        )));
        let expr = ctx.add(Expr::Add(dec, third));
        let plan = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("folds");
        assert!(
            plan.description.contains("0.428571428571"),
            "desc must render decimal operand decimally: {}",
            plan.description
        );
        assert!(
            !plan.description.contains("1000000000000"),
            "desc must not leak the raw fraction: {}",
            plan.description
        );
    }

    #[test]
    fn decimal_zero_and_one_identities_stay_exact() {
        // Annihilator and identities drop stickiness DELIBERATELY: the
        // numeric operand vanishes and exact 0/x keeps zero-tests sound.
        let mut ctx = Context::new();
        let zero = decimal_of(&mut ctx, 0, 1);
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Add(x, zero));
        let plan = try_rewrite_add_zero_expr(&ctx, expr).expect("identity");
        assert_eq!(plan.rewritten, x);

        let one = decimal_of(&mut ctx, 1, 1);
        let expr = ctx.add(Expr::Mul(one, x));
        let plan = try_rewrite_mul_one_expr(&ctx, expr).expect("identity");
        assert_eq!(plan.rewritten, x);

        // Division by decimal(0) is division by zero (WYSIWYG contract).
        let zero2 = decimal_of(&mut ctx, 0, 1);
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(two, zero2));
        let plan = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("folds");
        assert!(matches!(
            ctx.get(plan.rewritten),
            Expr::Constant(cas_ast::Constant::Undefined)
        ));
    }
}
