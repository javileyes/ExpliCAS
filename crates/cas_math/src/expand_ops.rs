//! Pure expansion operations shared across runtime crates.
//!
//! These helpers intentionally avoid engine-specific step recording and budget
//! accounting. Runtime crates can wrap these operations with their own
//! instrumentation.

use crate::build::mul2_raw;
use crate::combinatorics::binomial_coeff;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive};

/// Expand an expression recursively.
///
/// This is the pure symbolic expansion implementation:
/// - expands children first (bottom-up),
/// - then applies distribution/binomial/multinomial rules.
pub fn expand(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            ctx.add(Expr::Add(el, er))
        }
        Expr::Sub(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            ctx.add(Expr::Sub(el, er))
        }
        Expr::Mul(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            expand_mul(ctx, el, er)
        }
        Expr::Div(l, r) => {
            let el = expand(ctx, l);
            let er = expand(ctx, r);
            expand_div(ctx, el, er)
        }
        Expr::Pow(b, e) => {
            let budget = crate::multinomial_expand::MultinomialExpandBudget::default();
            if let Some(result) =
                crate::multinomial_expand::try_expand_multinomial_direct(ctx, b, e, &budget)
            {
                return result;
            }

            let eb = expand(ctx, b);
            let ee = expand(ctx, e);
            expand_pow(ctx, eb, ee)
        }
        Expr::Neg(e) => {
            let ee = expand(ctx, e);
            ctx.add(Expr::Neg(ee))
        }
        Expr::Function(fn_id, args) => {
            let name = ctx.sym_name(fn_id);
            if name == "expand" && args.len() == 1 {
                return expand(ctx, args[0]);
            }
            let new_args: Vec<ExprId> = args.iter().map(|a| expand(ctx, *a)).collect();
            ctx.add(Expr::Function(fn_id, new_args))
        }
        Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Hold(_) => expr,
    }
}

/// Expand multiplication by distributing over sums/differences.
pub fn expand_mul(ctx: &mut Context, l: ExprId, r: ExprId) -> ExprId {
    if let Some(res) = distribute_single(ctx, l, r) {
        return res;
    }
    if let Some(res) = distribute_single(ctx, r, l) {
        return res;
    }
    mul2_raw(ctx, l, r)
}

fn distribute_single(ctx: &mut Context, multiplier: ExprId, target: ExprId) -> Option<ExprId> {
    let target_data = ctx.get(target).clone();
    match target_data {
        Expr::Add(a, b) => {
            let ma = expand_mul(ctx, multiplier, a);
            let mb = expand_mul(ctx, multiplier, b);
            Some(ctx.add(Expr::Add(ma, mb)))
        }
        Expr::Sub(a, b) => {
            let ma = expand_mul(ctx, multiplier, a);
            let mb = expand_mul(ctx, multiplier, b);
            Some(ctx.add(Expr::Sub(ma, mb)))
        }
        _ => None,
    }
}

/// Expand division by distributing numerator sums/differences.
pub fn expand_div(ctx: &mut Context, num: ExprId, den: ExprId) -> ExprId {
    let num_data = ctx.get(num).clone();
    match num_data {
        Expr::Add(a, b) => {
            let da = expand_div(ctx, a, den);
            let db = expand_div(ctx, b, den);
            ctx.add(Expr::Add(da, db))
        }
        Expr::Sub(a, b) => {
            let da = expand_div(ctx, a, den);
            let db = expand_div(ctx, b, den);
            ctx.add(Expr::Sub(da, db))
        }
        _ => ctx.add(Expr::Div(num, den)),
    }
}

/// Expand powers where pattern-specific rewrites apply.
///
/// Notes:
/// - `(a*b)^n -> a^n * b^n` for structural expansion.
/// - `(a+b)^n` is expanded for small integer `n` in `[2, 10]`.
/// - `(a-b)^n` is rewritten through `a + (-b)`.
/// - `(-a)^n` parity handling is preserved.
pub fn expand_pow(ctx: &mut Context, base: ExprId, exp: ExprId) -> ExprId {
    let base_data = ctx.get(base).clone();

    if let Expr::Mul(a, b) = base_data {
        let ea = expand_pow(ctx, a, exp);
        let eb = expand_pow(ctx, b, exp);
        return mul2_raw(ctx, ea, eb);
    }

    if let Expr::Add(a, b) = base_data {
        let exp_data = ctx.get(exp).clone();
        if let Expr::Number(n) = exp_data {
            if n.is_integer() && !n.is_negative() {
                if let Some(n_val) = n.to_integer().to_u32() {
                    if (2..=10).contains(&n_val) {
                        let mut terms = Vec::new();
                        for k in 0..=n_val {
                            let coeff = binomial_coeff(n_val, k);
                            let exp_a = n_val - k;
                            let exp_b = k;

                            let term_a = if exp_a == 0 {
                                ctx.num(1)
                            } else if exp_a == 1 {
                                a
                            } else {
                                let e = ctx.num(exp_a as i64);
                                ctx.add(Expr::Pow(a, e))
                            };
                            let term_b = if exp_b == 0 {
                                ctx.num(1)
                            } else if exp_b == 1 {
                                b
                            } else {
                                let e = ctx.num(exp_b as i64);
                                ctx.add(Expr::Pow(b, e))
                            };

                            let mut term = mul2_raw(ctx, term_a, term_b);
                            if coeff > 1 {
                                let c = ctx.num(coeff as i64);
                                term = mul2_raw(ctx, c, term);
                            }
                            terms.push(term);
                        }

                        let mut expanded = terms[0];
                        for &term in terms.iter().skip(1) {
                            expanded = ctx.add(Expr::Add(expanded, term));
                        }
                        return expand(ctx, expanded);
                    }
                }
            }
        }
    }

    if let Expr::Sub(a, b) = base_data {
        let neg_b = ctx.add(Expr::Neg(b));
        let sum = ctx.add(Expr::Add(a, neg_b));
        return expand_pow(ctx, sum, exp);
    }

    if let Expr::Neg(a) = base_data {
        let exp_data = ctx.get(exp).clone();
        if let Expr::Number(n) = exp_data {
            if n.is_integer() {
                if n.to_integer().is_even() {
                    return expand_pow(ctx, a, exp);
                }
                let p = expand_pow(ctx, a, exp);
                return ctx.add(Expr::Neg(p));
            }
        }
    }

    ctx.add(Expr::Pow(base, exp))
}

#[cfg(test)]
mod tests {
    use super::{expand, expand_div, expand_mul, expand_pow};
    use cas_ast::{Context, Expr, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn expand_mul_distributes_over_addition() {
        let mut ctx = Context::new();
        let l = parse("a", &mut ctx).expect("parse left");
        let r = parse("b + c", &mut ctx).expect("parse right");
        let out = expand_mul(&mut ctx, l, r);
        assert!(
            matches!(ctx.get(out), Expr::Add(_, _)),
            "expected expanded add, got {}",
            render(&ctx, out)
        );
    }

    #[test]
    fn expand_div_distributes_numerator_sum() {
        let mut ctx = Context::new();
        let num = parse("a + b", &mut ctx).expect("parse num");
        let den = parse("c", &mut ctx).expect("parse den");
        let out = expand_div(&mut ctx, num, den);
        assert!(
            matches!(ctx.get(out), Expr::Add(_, _)),
            "expected expanded add, got {}",
            render(&ctx, out)
        );
    }

    #[test]
    fn expand_pow_expands_small_binomial() {
        let mut ctx = Context::new();
        let base = parse("x + 1", &mut ctx).expect("parse base");
        let exp = parse("3", &mut ctx).expect("parse exp");
        let out = expand_pow(&mut ctx, base, exp);
        assert!(
            matches!(ctx.get(out), Expr::Add(_, _)),
            "expected expanded polynomial sum, got {}",
            render(&ctx, out)
        );
    }

    #[test]
    fn expand_unwraps_expand_function_calls() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse expr");
        let out = expand(&mut ctx, expr);
        if let Expr::Function(fn_id, _) = ctx.get(out) {
            assert_ne!(ctx.sym_name(*fn_id), "expand");
        }
        assert_ne!(out, expr, "expand(...) call should be rewritten");
    }
}
