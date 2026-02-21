use crate::isolation_utils::contains_var;
use cas_ast::{Context, Expr, ExprId};

/// Classification of a term with respect to one variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermClass {
    /// Term does not contain the variable.
    Const(ExprId),
    /// Term is linear in the variable: `coef * var`.
    /// `None` means implicit coefficient `1`.
    Linear(Option<ExprId>),
    /// Term contains the variable non-linearly.
    NonLinear,
}

/// Classify a term as `Const`, `Linear(coef)`, or `NonLinear`.
pub fn split_linear_term(ctx: &mut Context, term: ExprId, var: &str) -> TermClass {
    if !contains_var(ctx, term, var) {
        return TermClass::Const(term);
    }

    match ctx.get(term).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => TermClass::Linear(None),
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            match (l_has, r_has) {
                (true, false) => {
                    if matches!(ctx.get(l), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
                    {
                        TermClass::Linear(Some(r))
                    } else {
                        match split_linear_term(ctx, l, var) {
                            TermClass::Linear(inner_coef) => match inner_coef {
                                Some(k) => {
                                    let combined = ctx.add(Expr::Mul(r, k));
                                    TermClass::Linear(Some(combined))
                                }
                                None => TermClass::Linear(Some(r)),
                            },
                            _ => TermClass::NonLinear,
                        }
                    }
                }
                (false, true) => {
                    if matches!(ctx.get(r), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
                    {
                        TermClass::Linear(Some(l))
                    } else {
                        match split_linear_term(ctx, r, var) {
                            TermClass::Linear(inner_coef) => match inner_coef {
                                Some(k) => {
                                    let combined = ctx.add(Expr::Mul(l, k));
                                    TermClass::Linear(Some(combined))
                                }
                                None => TermClass::Linear(Some(l)),
                            },
                            _ => TermClass::NonLinear,
                        }
                    }
                }
                (true, true) => TermClass::NonLinear,
                (false, false) => TermClass::Const(term),
            }
        }
        Expr::Neg(inner) => match split_linear_term(ctx, inner, var) {
            TermClass::Const(_) => TermClass::Const(term),
            TermClass::Linear(_) => TermClass::Linear(Some(term)),
            TermClass::NonLinear => TermClass::NonLinear,
        },
        Expr::Pow(base, exp) => {
            if contains_var(ctx, exp, var) {
                TermClass::NonLinear
            } else if contains_var(ctx, base, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == num_rational::BigRational::from_integer(1.into()) {
                        return split_linear_term(ctx, base, var);
                    }
                }
                TermClass::NonLinear
            } else {
                TermClass::Const(term)
            }
        }
        Expr::Div(num, denom) => {
            if contains_var(ctx, denom, var) {
                TermClass::NonLinear
            } else {
                match split_linear_term(ctx, num, var) {
                    TermClass::Linear(_) => TermClass::NonLinear,
                    TermClass::Const(_) => TermClass::Const(term),
                    TermClass::NonLinear => TermClass::NonLinear,
                }
            }
        }
        Expr::Function(_, args) => {
            if args.iter().any(|a| contains_var(ctx, *a, var)) {
                TermClass::NonLinear
            } else {
                TermClass::Const(term)
            }
        }
        _ => TermClass::NonLinear,
    }
}

/// Build a sum expression from a list of parts.
pub fn build_sum(ctx: &mut Context, parts: &[ExprId]) -> ExprId {
    if parts.is_empty() {
        return ctx.num(0);
    }
    let mut result = parts[0];
    for &part in &parts[1..] {
        result = ctx.add(Expr::Add(result, part));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_linear_term_const() {
        let mut ctx = Context::new();
        let a = ctx.var("A");
        assert!(matches!(
            split_linear_term(&mut ctx, a, "P"),
            TermClass::Const(_)
        ));
    }

    #[test]
    fn split_linear_term_var() {
        let mut ctx = Context::new();
        let p = ctx.var("P");
        assert!(matches!(
            split_linear_term(&mut ctx, p, "P"),
            TermClass::Linear(_)
        ));
    }

    #[test]
    fn build_sum_empty_is_zero() {
        let mut ctx = Context::new();
        let s = build_sum(&mut ctx, &[]);
        assert!(
            matches!(ctx.get(s), Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()))
        );
    }
}
