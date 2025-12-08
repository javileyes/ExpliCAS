use crate::{Constant, Context, DisplayExpr, Expr, ExprId};
use std::cmp::Ordering;

thread_local! {
    static DEBUG_COMPARE: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

pub fn enable_compare_debug() {
    DEBUG_COMPARE.with(|d| d.set(true));
}

pub fn disable_compare_debug() {
    DEBUG_COMPARE.with(|d| d.set(false));
}

pub fn compare_expr(context: &Context, a: ExprId, b: ExprId) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }

    let expr_a = context.get(a);
    let expr_b = context.get(b);

    let debug = DEBUG_COMPARE.with(|d| d.get());
    if debug {
        let display_a = DisplayExpr { context, id: a };
        let display_b = DisplayExpr { context, id: b };
        eprintln!(
            "compare_expr({}, {}) [ExprId({:?}, {:?})]",
            display_a, display_b, a.0, b.0
        );
    }

    use Expr::*;

    // 1. Hierarchy Check
    let rank_a = get_rank(expr_a);
    let rank_b = get_rank(expr_b);
    if rank_a != rank_b {
        let result = rank_a.cmp(&rank_b);
        if debug {
            eprintln!(
                "  → rank comparison: {} (rank {}) vs {} (rank {}) = {:?}",
                DisplayExpr { context, id: a },
                rank_a,
                DisplayExpr { context, id: b },
                rank_b,
                result
            );
        }
        return result;
    }

    // 2. Same Type Comparison
    let result = match (expr_a, expr_b) {
        (Number(n1), Number(n2)) => n1.cmp(n2),
        (Constant(c1), Constant(c2)) => compare_constant(c1, c2),
        (Variable(v1), Variable(v2)) => v1.cmp(v2),
        (Function(n1, args1), Function(n2, args2)) => match n1.cmp(n2) {
            Ordering::Equal => compare_args(context, args1, args2),
            ord => ord,
        },
        (Pow(b1, e1), Pow(b2, e2)) => match compare_expr(context, *b1, *b2) {
            Ordering::Equal => compare_expr(context, *e1, *e2),
            ord => ord,
        },
        (Neg(e1), Neg(e2)) => compare_expr(context, *e1, *e2),
        (Add(l1, r1), Add(l2, r2)) => compare_binary(context, *l1, *r1, *l2, *r2),
        (Sub(l1, r1), Sub(l2, r2)) => compare_binary(context, *l1, *r1, *l2, *r2),
        (Mul(l1, r1), Mul(l2, r2)) => compare_binary(context, *l1, *r1, *l2, *r2),
        (Div(l1, r1), Div(l2, r2)) => compare_binary(context, *l1, *r1, *l2, *r2),
        _ => Ordering::Equal, // Should be unreachable if ranks are correct
    };

    if debug {
        eprintln!("  → result: {:?}", result);
    }

    result
}

fn get_rank(expr: &Expr) -> u8 {
    use Expr::*;
    match expr {
        Number(_) => 0,
        Constant(_) => 1,
        Variable(_) => 2,
        Function(_, _) => 3,
        Neg(_) => 4,
        Pow(_, _) => 5,
        Mul(_, _) => 6,
        Div(_, _) => 7,
        Add(_, _) => 8,
        Sub(_, _) => 9,
        Matrix { .. } => 10,
    }
}

fn compare_constant(c1: &Constant, c2: &Constant) -> Ordering {
    use Constant::*;
    // Arbitrary order for constants
    let r1 = match c1 {
        Pi => 0,
        E => 1,
        Infinity => 2,
        Undefined => 3,
    };
    let r2 = match c2 {
        Pi => 0,
        E => 1,
        Infinity => 2,
        Undefined => 3,
    };
    r1.cmp(&r2)
}

fn compare_args(context: &Context, args1: &[ExprId], args2: &[ExprId]) -> Ordering {
    for (a1, a2) in args1.iter().zip(args2.iter()) {
        match compare_expr(context, *a1, *a2) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    args1.len().cmp(&args2.len())
}

fn compare_binary(context: &Context, l1: ExprId, r1: ExprId, l2: ExprId, r2: ExprId) -> Ordering {
    match compare_expr(context, l1, l2) {
        Ordering::Equal => compare_expr(context, r1, r2),
        ord => ord,
    }
}
