//! Calculus rules: differentiation, integration, summation, and products.
//!
//! This module is split into submodules:
//! - `differentiation`: symbolic derivative computation
//! - `integration`: symbolic integral computation + helpers
//! - `summation`: finite sum/product evaluation (SumRule, ProductRule)

mod differentiation;
mod integration;
mod summation;

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;

use differentiation::differentiate;
use integration::integrate;

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let name = ctx.sym_name(*fn_id);
        if name == "integrate" {
            if args.len() == 2 {
                let integrand = args[0];
                let var_expr = args[1];
                if let Expr::Variable(var_sym) = ctx.get(var_expr) {
                    let var_name = ctx.sym_name(*var_sym).to_string();
                    if let Some(result) = integrate(ctx, integrand, &var_name) {
                        return Some(Rewrite::new(result).desc_lazy(|| {
                            format!(
                                "integrate({}, {})",
                                cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: integrand
                                },
                                var_name
                            )
                        }));
                    }
                }
            } else if args.len() == 1 {
                // Default to 'x' if not specified? Or fail?
                // Let's assume 'x' for convenience if only 1 arg.
                let integrand = args[0];
                if let Some(result) = integrate(ctx, integrand, "x") {
                    return Some(Rewrite::new(result).desc_lazy(|| {
                        format!(
                            "integrate({}, x)",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: integrand
                            }
                        )
                    }));
                }
            }
        }
    }
    None
});

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let name = ctx.sym_name(*fn_id);
        if name == "diff" && args.len() == 2 {
            let target = args[0];
            let var_expr = args[1];
            if let Expr::Variable(var_sym) = ctx.get(var_expr) {
                let var_name = ctx.sym_name(*var_sym).to_string();
                if let Some(result) = differentiate(ctx, target, &var_name) {
                    return Some(Rewrite::new(result).desc_lazy(|| {
                        format!(
                            "diff({}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: target
                            },
                            var_name
                        )
                    }));
                }
            }
        }
    }
    None
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}
