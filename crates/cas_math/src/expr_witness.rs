//! Witness search helpers for structural domain constraints.

use crate::expr_extract::{extract_sqrt_argument_view, extract_unary_log_argument_view};
use crate::expr_predicates::is_even_root_exponent;
use cas_ast::{Context, Expr, ExprId};

/// Kind of structural witness to search for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessKind {
    /// `sqrt(t)` or `t^(p/q)` with even `q` as witness for `t >= 0`.
    Sqrt,
    /// unary `ln(t)` / `log(t)` as witness for `t > 0`.
    Log,
    /// `Div(_, t)` as witness for `t != 0`.
    Division,
}

/// Structural expression equality.
pub fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    cas_ast::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

/// Check if a witness survives in `output`.
pub fn witness_survives(ctx: &Context, target: ExprId, output: ExprId, kind: WitnessKind) -> bool {
    let mut stack = vec![output];
    while let Some(expr) = stack.pop() {
        if node_matches_witness(ctx, target, expr, kind, &mut stack) {
            return true;
        }
    }
    false
}

/// Check witness survival in full tree context with one node replaced.
pub fn witness_survives_in_context(
    ctx: &Context,
    target: ExprId,
    root: ExprId,
    replaced_node: ExprId,
    replacement: Option<ExprId>,
    kind: WitnessKind,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        if expr == replaced_node {
            if let Some(repl) = replacement {
                if witness_survives(ctx, target, repl, kind) {
                    return true;
                }
            }
            continue;
        }
        if node_matches_witness(ctx, target, expr, kind, &mut stack) {
            return true;
        }
    }
    false
}

fn node_matches_witness(
    ctx: &Context,
    target: ExprId,
    expr: ExprId,
    kind: WitnessKind,
    stack: &mut Vec<ExprId>,
) -> bool {
    match ctx.get(expr) {
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return false;
            };
            if kind == WitnessKind::Sqrt && exprs_equal(ctx, arg, target) {
                return true;
            }
            stack.push(arg);
            false
        }
        Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                return false;
            };
            if kind == WitnessKind::Log && exprs_equal(ctx, arg, target) {
                return true;
            }
            stack.push(arg);
            false
        }
        Expr::Pow(base, exp) => {
            if kind == WitnessKind::Sqrt {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if is_even_root_exponent(n) && exprs_equal(ctx, *base, target) {
                        return true;
                    }
                }
            }
            stack.push(*base);
            stack.push(*exp);
            false
        }
        Expr::Div(num, den) => {
            if kind == WitnessKind::Division && exprs_equal(ctx, *den, target) {
                return true;
            }
            stack.push(*num);
            stack.push(*den);
            false
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            stack.push(*l);
            stack.push(*r);
            false
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            stack.push(*inner);
            false
        }
        Expr::Function(_, args) => {
            stack.extend(args.iter().copied());
            false
        }
        Expr::Matrix { data, .. } => {
            stack.extend(data.iter().copied());
            false
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn witness_survives_for_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x) + y", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(witness_survives(&ctx, x, expr, WitnessKind::Sqrt));
    }

    #[test]
    fn witness_does_not_survive_when_absent() {
        let mut ctx = Context::new();
        let expr = parse("x + y", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(!witness_survives(&ctx, x, expr, WitnessKind::Sqrt));
    }

    #[test]
    fn witness_survives_in_context_with_replacement() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let sqrt_x = parse("sqrt(x)", &mut ctx).expect("parse sqrt");
        let root = parse("sqrt(x) + z", &mut ctx).expect("parse root");
        let replacement = parse("sqrt(x)", &mut ctx).expect("parse replacement");
        assert!(witness_survives_in_context(
            &ctx,
            x,
            root,
            sqrt_x,
            Some(replacement),
            WitnessKind::Sqrt
        ));
    }
}
