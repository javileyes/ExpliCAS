use cas_ast::{Expr, Constant};
use std::cmp::Ordering;
use std::rc::Rc;

pub fn compare_expr(a: &Expr, b: &Expr) -> Ordering {
    use Expr::*;

    // 1. Hierarchy Check
    let rank_a = get_rank(a);
    let rank_b = get_rank(b);
    if rank_a != rank_b {
        return rank_a.cmp(&rank_b);
    }

    // 2. Same Type Comparison
    match (a, b) {
        (Number(n1), Number(n2)) => n1.cmp(n2),
        (Constant(c1), Constant(c2)) => compare_constant(c1, c2),
        (Variable(v1), Variable(v2)) => v1.cmp(v2),
        (Function(n1, args1), Function(n2, args2)) => {
            match n1.cmp(n2) {
                Ordering::Equal => compare_args(args1, args2),
                ord => ord,
            }
        },
        (Pow(b1, e1), Pow(b2, e2)) => {
            match compare_expr(b1, b2) {
                Ordering::Equal => compare_expr(e1, e2),
                ord => ord,
            }
        },
        (Neg(e1), Neg(e2)) => compare_expr(e1, e2),
        (Add(l1, r1), Add(l2, r2)) => compare_binary(l1, r1, l2, r2),
        (Sub(l1, r1), Sub(l2, r2)) => compare_binary(l1, r1, l2, r2),
        (Mul(l1, r1), Mul(l2, r2)) => compare_binary(l1, r1, l2, r2),
        (Div(l1, r1), Div(l2, r2)) => compare_binary(l1, r1, l2, r2),
        _ => Ordering::Equal, // Should be unreachable if ranks are correct
    }
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
    }
}

fn compare_constant(c1: &Constant, c2: &Constant) -> Ordering {
    use Constant::*;
    // Arbitrary order for constants
    let r1 = match c1 { Pi => 0, E => 1, Infinity => 2, Undefined => 3 };
    let r2 = match c2 { Pi => 0, E => 1, Infinity => 2, Undefined => 3 };
    r1.cmp(&r2)
}

fn compare_args(args1: &[Rc<Expr>], args2: &[Rc<Expr>]) -> Ordering {
    for (a1, a2) in args1.iter().zip(args2.iter()) {
        match compare_expr(a1, a2) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    args1.len().cmp(&args2.len())
}

fn compare_binary(l1: &Rc<Expr>, r1: &Rc<Expr>, l2: &Rc<Expr>, r2: &Rc<Expr>) -> Ordering {
    match compare_expr(l1, l2) {
        Ordering::Equal => compare_expr(r1, r2),
        ord => ord,
    }
}
