//! Support for `collect(expr, var)` style polynomial term grouping.

use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::{is_one_expr, is_zero_expr};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::{flatten_add_sub_chain, flatten_mul_chain};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectByVarRewrite {
    pub target_expr: ExprId,
    pub var_name: String,
    pub rewritten: ExprId,
}

impl CollectByVarRewrite {
    /// Canonical didactic description used by the engine rule wrapper.
    pub fn desc(&self) -> String {
        format!("collect({}, {})", self.target_expr, self.var_name)
    }
}

/// Try rewriting a `collect(target, var)` call into grouped polynomial form.
pub fn try_rewrite_collect_by_var_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CollectByVarRewrite> {
    if !ctx.is_call_named(expr, "collect") {
        return None;
    }

    let (target_expr, var_name) = match ctx.get(expr) {
        Expr::Function(_, args) if args.len() == 2 => {
            let target_expr = args[0];
            let var_expr = args[1];
            let var_name = if let Expr::Variable(sym_id) = ctx.get(var_expr) {
                ctx.sym_name(*sym_id).to_string()
            } else {
                return None;
            };
            (target_expr, var_name)
        }
        _ => return None,
    };

    let terms = flatten_add_sub_chain(ctx, target_expr);

    // Map: polynomial degree -> list of coefficient terms.
    let mut groups: HashMap<i64, Vec<ExprId>> = HashMap::new();
    for term in terms {
        let (coeff, degree) = extract_coeff_degree_for_var(ctx, term, &var_name);
        groups.entry(degree).or_default().push(coeff);
    }

    let mut new_terms = Vec::new();
    let mut degrees: Vec<i64> = groups.keys().copied().collect();
    degrees.sort_by(|a, b| b.cmp(a));

    for deg in degrees {
        let coeffs = match groups.get(&deg) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };

        let combined_coeff = if coeffs.len() == 1 {
            coeffs[0]
        } else {
            let mut sum = coeffs[0];
            for c in coeffs.iter().skip(1) {
                sum = ctx.add(Expr::Add(sum, *c));
            }
            sum
        };

        let term = if deg == 0 {
            combined_coeff
        } else {
            let var_part = if deg == 1 {
                ctx.var(&var_name)
            } else {
                let degree_expr = ctx.num(deg);
                let var_expr = ctx.var(&var_name);
                ctx.add(Expr::Pow(var_expr, degree_expr))
            };

            if is_one_expr(ctx, combined_coeff) {
                var_part
            } else if is_zero_expr(ctx, combined_coeff) {
                continue;
            } else {
                smart_mul(ctx, combined_coeff, var_part)
            }
        };
        new_terms.push(term);
    }

    let rewritten = if new_terms.is_empty() {
        ctx.num(0)
    } else {
        let mut result = new_terms[0];
        for term in new_terms.into_iter().skip(1) {
            result = ctx.add(Expr::Add(result, term));
        }
        result
    };

    Some(CollectByVarRewrite {
        target_expr,
        var_name,
        rewritten,
    })
}

/// Returns `(coefficient, degree)` for one term with respect to `var`.
fn extract_coeff_degree_for_var(ctx: &mut Context, term: ExprId, var: &str) -> (ExprId, i64) {
    let factors = flatten_mul_chain(ctx, term);

    let mut degree: i64 = 0;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        match ctx.get(factor) {
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => {
                degree += 1;
            }
            Expr::Pow(base, exp) => {
                if let Expr::Variable(sym_id) = ctx.get(*base) {
                    if ctx.sym_name(*sym_id) == var {
                        if let Expr::Number(n) = ctx.get(*exp) {
                            if n.is_integer() {
                                degree += n.to_integer().try_into().unwrap_or(0);
                                continue;
                            }
                        }
                    }
                }
                coeff_factors.push(factor);
            }
            _ => coeff_factors.push(factor),
        }
    }

    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut c = coeff_factors[0];
        for factor in coeff_factors.into_iter().skip(1) {
            c = smart_mul(ctx, c, factor);
        }
        c
    };

    (coeff, degree)
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_collect_by_var_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn rewrites_collect_basic() {
        let mut ctx = Context::new();
        let expr = parse("collect(a*x + b*x, x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_collect_by_var_expr(&mut ctx, expr).expect("rewrite");
        let text = rendered(&ctx, rewrite.rewritten);
        assert!(text.contains("x"));
        assert!(text.contains("a + b") || text.contains("b + a"));
    }

    #[test]
    fn rewrites_collect_with_constants() {
        let mut ctx = Context::new();
        let expr = parse("collect(a*x + 2*x + 5, x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_collect_by_var_expr(&mut ctx, expr).expect("rewrite");
        let text = rendered(&ctx, rewrite.rewritten);
        assert!(text.contains("a + 2") || text.contains("2 + a"));
        assert!(text.contains("5"));
    }

    #[test]
    fn rewrites_collect_powers() {
        let mut ctx = Context::new();
        let expr = parse("collect(3*x^2 + y*x^2 + x, x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_collect_by_var_expr(&mut ctx, expr).expect("rewrite");
        let text = rendered(&ctx, rewrite.rewritten);
        assert!(text.contains("3 + y") || text.contains("y + 3"));
        assert!(text.contains("x^2"));
    }
}
