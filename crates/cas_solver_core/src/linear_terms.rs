use crate::isolation_utils::contains_var;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};

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

/// Signed term decomposition used by linear-collect strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectDecomposition {
    pub coeff_parts: Vec<ExprId>,
    pub const_parts: Vec<ExprId>,
}

fn apply_sign(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

/// Decompose `expr = sum(terms)` into linear coefficient parts and constant parts.
///
/// Returns `None` when no linear terms are found or when any term is non-linear
/// in `var`, so linear-collect strategy should not apply.
pub fn decompose_linear_collect_terms(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<LinearCollectDecomposition> {
    let terms = add_terms_signed(ctx, expr);
    let mut coeff_parts: Vec<ExprId> = Vec::new();
    let mut const_parts: Vec<ExprId> = Vec::new();

    for (term, sign) in terms {
        match split_linear_term(ctx, term, var) {
            TermClass::Const(_) => const_parts.push(apply_sign(ctx, term, sign)),
            TermClass::Linear(c) => {
                let coef = c.unwrap_or_else(|| ctx.num(1));
                coeff_parts.push(apply_sign(ctx, coef, sign));
            }
            TermClass::NonLinear => return None,
        }
    }

    if coeff_parts.is_empty() {
        return None;
    }

    Some(LinearCollectDecomposition {
        coeff_parts,
        const_parts,
    })
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

    #[test]
    fn decompose_linear_collect_terms_splits_coeffs_and_constants() {
        let mut ctx = Context::new();
        let p = ctx.var("P");
        let a = ctx.var("A");
        let r = ctx.var("r");
        let t = ctx.var("t");
        let pr = ctx.add(Expr::Mul(p, r));
        let prt = ctx.add(Expr::Mul(pr, t));
        let sum = ctx.add(Expr::Add(p, prt));
        let expr = ctx.add(Expr::Sub(sum, a));

        let dec = decompose_linear_collect_terms(&mut ctx, expr, "P")
            .expect("must decompose linear terms");
        assert_eq!(dec.coeff_parts.len(), 2);
        assert_eq!(dec.const_parts.len(), 1);
    }

    #[test]
    fn decompose_linear_collect_terms_rejects_nonlinear() {
        let mut ctx = Context::new();
        let p = ctx.var("P");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(p, two));

        assert!(decompose_linear_collect_terms(&mut ctx, expr, "P").is_none());
    }
}
