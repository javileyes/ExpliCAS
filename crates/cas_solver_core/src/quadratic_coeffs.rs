use crate::isolation_utils::contains_var;
use cas_ast::{Context, Expr, ExprId};
use cas_math::build::mul2_raw;

/// Extract `(a, b, c)` from an expression in one variable interpreted as:
/// `a*x^2 + b*x + c`.
///
/// Returns `None` if a term is non-polynomial or has unsupported degree.
pub fn extract_quadratic_coefficients(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, ExprId)> {
    let zero = ctx.num(0);
    let mut a = zero;
    let mut b = zero;
    let mut c = zero;

    let mut stack = vec![(expr, true)];

    while let Some((curr, pos)) = stack.pop() {
        let curr_data = ctx.get(curr).clone();
        match curr_data {
            Expr::Add(l, r) => {
                stack.push((r, pos));
                stack.push((l, pos));
            }
            Expr::Sub(l, r) => {
                stack.push((r, !pos));
                stack.push((l, pos));
            }
            _ => {
                let (coeff, degree) = analyze_term(ctx, curr, var)?;
                let term_val = if pos {
                    coeff
                } else {
                    ctx.add(Expr::Neg(coeff))
                };

                match degree {
                    2 => a = ctx.add(Expr::Add(a, term_val)),
                    1 => b = ctx.add(Expr::Add(b, term_val)),
                    0 => c = ctx.add(Expr::Add(c, term_val)),
                    _ => return None,
                }
            }
        }
    }

    Some((a, b, c))
}

fn analyze_term(ctx: &mut Context, term: ExprId, var: &str) -> Option<(ExprId, i32)> {
    if !contains_var(ctx, term, var) {
        return Some((term, 0));
    }

    let term_data = ctx.get(term).clone();

    match term_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => Some((ctx.num(1), 1)),
        Expr::Pow(base, exp) => {
            if let Expr::Variable(sym_id) = ctx.get(base) {
                if ctx.sym_name(*sym_id) == var && !contains_var(ctx, exp, var) {
                    let degree = if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer() {
                            Some(n.to_integer())
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(d) = degree {
                        return Some((ctx.num(1), d.try_into().ok()?));
                    }
                }
            }
            None
        }
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            if l_has && r_has {
                let (c1, d1) = analyze_term(ctx, l, var)?;
                let (c2, d2) = analyze_term(ctx, r, var)?;
                let new_coeff = mul2_raw(ctx, c1, c2);
                Some((new_coeff, d1 + d2))
            } else if l_has {
                let (c, d) = analyze_term(ctx, l, var)?;
                let new_coeff = mul2_raw(ctx, c, r);
                Some((new_coeff, d))
            } else if r_has {
                let (c, d) = analyze_term(ctx, r, var)?;
                let new_coeff = mul2_raw(ctx, l, c);
                Some((new_coeff, d))
            } else {
                Some((term, 0))
            }
        }
        Expr::Div(l, r) => {
            if contains_var(ctx, r, var) {
                return None;
            }
            let (c, d) = analyze_term(ctx, l, var)?;
            let new_coeff = ctx.add(Expr::Div(c, r));
            Some((new_coeff, d))
        }
        Expr::Neg(inner) => {
            let (c, d) = analyze_term(ctx, inner, var)?;
            let new_coeff = ctx.add(Expr::Neg(c));
            Some((new_coeff, d))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_basic_quadratic_coefficients() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let x2 = ctx.add(Expr::Pow(x, two));
        let two_x2 = ctx.add(Expr::Mul(two, x2));
        let three_x = ctx.add(Expr::Mul(three, x));
        let poly = ctx.add(Expr::Add(two_x2, three_x));
        let poly = ctx.add(Expr::Add(poly, four));
        let (a, b, c) = extract_quadratic_coefficients(&mut ctx, poly, "x")
            .expect("must extract quadratic coefficients");
        assert!(!contains_var(&ctx, a, "x"));
        assert!(!contains_var(&ctx, b, "x"));
        assert!(!contains_var(&ctx, c, "x"));
    }

    #[test]
    fn rejects_variable_in_denominator() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Div(one, x));
        assert!(extract_quadratic_coefficients(&mut ctx, expr, "x").is_none());
    }
}
