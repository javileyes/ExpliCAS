use cas_ast::{BoundType, Constant, Context, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_rational::BigRational;
use std::cmp::Ordering;

// Helper to create -infinity
pub fn neg_inf(ctx: &mut Context) -> ExprId {
    let inf = ctx.add(Expr::Constant(Constant::Infinity));
    ctx.add(Expr::Neg(inf))
}

// Helper to create +infinity
pub fn pos_inf(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Infinity))
}

pub fn is_infinity(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(Constant::Infinity))
}

pub fn is_neg_infinity(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(inner) => is_infinity(ctx, *inner),
        _ => false,
    }
}

pub fn get_number(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => get_number(ctx, *inner).map(|n| -n),
        _ => None,
    }
}

pub fn compare_values(ctx: &Context, a: ExprId, b: ExprId) -> Ordering {
    // Handle Infinity
    let a_inf = is_infinity(ctx, a);
    let b_inf = is_infinity(ctx, b);
    let a_neg_inf = is_neg_infinity(ctx, a);
    let b_neg_inf = is_neg_infinity(ctx, b);

    if a_neg_inf {
        if b_neg_inf {
            return Ordering::Equal;
        }
        return Ordering::Less;
    }
    if b_neg_inf {
        return Ordering::Greater;
    }

    if a_inf {
        if b_inf {
            return Ordering::Equal;
        }
        return Ordering::Greater;
    }
    if b_inf {
        return Ordering::Less;
    }

    // Handle Numbers
    if let (Some(n1), Some(n2)) = (get_number(ctx, a), get_number(ctx, b)) {
        return n1.cmp(&n2);
    }

    // Fallback: Use structural comparison if we can't compare values
    cas_ast::ordering::compare_expr(ctx, a, b)
}

/// Build the solution set obtained after isolating a variable on the LHS:
/// `var <op> rhs`.
pub fn isolated_var_solution(ctx: &mut Context, rhs: ExprId, op: RelOp) -> SolutionSet {
    match op {
        RelOp::Eq => SolutionSet::Discrete(vec![rhs]),
        RelOp::Neq => {
            let i1 = Interval {
                min: neg_inf(ctx),
                min_type: BoundType::Open,
                max: rhs,
                max_type: BoundType::Open,
            };
            let i2 = Interval {
                min: rhs,
                min_type: BoundType::Open,
                max: pos_inf(ctx),
                max_type: BoundType::Open,
            };
            SolutionSet::Union(vec![i1, i2])
        }
        RelOp::Lt => SolutionSet::Continuous(Interval {
            min: neg_inf(ctx),
            min_type: BoundType::Open,
            max: rhs,
            max_type: BoundType::Open,
        }),
        RelOp::Gt => SolutionSet::Continuous(Interval {
            min: rhs,
            min_type: BoundType::Open,
            max: pos_inf(ctx),
            max_type: BoundType::Open,
        }),
        RelOp::Leq => SolutionSet::Continuous(Interval {
            min: neg_inf(ctx),
            min_type: BoundType::Open,
            max: rhs,
            max_type: BoundType::Closed,
        }),
        RelOp::Geq => SolutionSet::Continuous(Interval {
            min: rhs,
            min_type: BoundType::Closed,
            max: pos_inf(ctx),
            max_type: BoundType::Open,
        }),
    }
}

pub fn intersect_intervals(ctx: &Context, i1: &Interval, i2: &Interval) -> SolutionSet {
    // Intersection of [a, b] and [c, d] is [max(a,c), min(b,d)]

    // Compare mins
    let (min, min_type) = match compare_values(ctx, i1.min, i2.min) {
        Ordering::Less => (i2.min, i2.min_type.clone()), // i1.min < i2.min -> take i2
        Ordering::Greater => (i1.min, i1.min_type.clone()), // i1.min > i2.min -> take i1
        Ordering::Equal => {
            let type_ = if i1.min_type == BoundType::Open || i2.min_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.min, type_)
        }
    };

    // Compare maxs
    let (max, max_type) = match compare_values(ctx, i1.max, i2.max) {
        Ordering::Less => (i1.max, i1.max_type.clone()), // i1.max < i2.max -> take i1
        Ordering::Greater => (i2.max, i2.max_type.clone()), // i1.max > i2.max -> take i2
        Ordering::Equal => {
            let type_ = if i1.max_type == BoundType::Open || i2.max_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.max, type_)
        }
    };

    // Check if valid interval (min < max)
    match compare_values(ctx, min, max) {
        Ordering::Less => SolutionSet::Continuous(Interval {
            min,
            min_type,
            max,
            max_type,
        }),
        Ordering::Equal => {
            if min_type == BoundType::Closed && max_type == BoundType::Closed {
                SolutionSet::Discrete(vec![min])
            } else {
                SolutionSet::Empty
            }
        }
        Ordering::Greater => SolutionSet::Empty,
    }
}

pub fn union_solution_sets(ctx: &Context, s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    let intervals = match (s1, s2) {
        (SolutionSet::Empty, s) | (s, SolutionSet::Empty) => return s,
        (SolutionSet::AllReals, _) | (_, SolutionSet::AllReals) => return SolutionSet::AllReals,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => vec![i1, i2],
        (SolutionSet::Continuous(i), SolutionSet::Union(mut u))
        | (SolutionSet::Union(mut u), SolutionSet::Continuous(i)) => {
            u.push(i);
            u
        }
        (SolutionSet::Union(mut u1), SolutionSet::Union(u2)) => {
            u1.extend(u2);
            u1
        }
        (SolutionSet::Discrete(mut d1), SolutionSet::Discrete(d2)) => {
            d1.extend(d2);
            return SolutionSet::Discrete(d1);
        }
        (s1, _) => return s1,
    };

    let merged = merge_intervals(ctx, intervals);
    if merged.is_empty() {
        SolutionSet::Empty
    } else if merged.len() == 1 {
        let i = &merged[0];
        if is_neg_infinity(ctx, i.min) && is_infinity(ctx, i.max) {
            SolutionSet::AllReals
        } else {
            SolutionSet::Continuous(i.clone())
        }
    } else {
        SolutionSet::Union(merged)
    }
}

fn merge_intervals(ctx: &Context, mut intervals: Vec<Interval>) -> Vec<Interval> {
    if intervals.is_empty() {
        return vec![];
    }
    intervals.sort_by(|a, b| compare_values(ctx, a.min, b.min));

    let mut merged = Vec::new();
    let mut current = intervals[0].clone();

    for next in intervals.into_iter().skip(1) {
        let cmp_max_min = compare_values(ctx, current.max, next.min);

        let should_merge = match cmp_max_min {
            Ordering::Greater => true,
            Ordering::Equal => {
                current.max_type == BoundType::Closed || next.min_type == BoundType::Closed
            }
            Ordering::Less => false,
        };

        if should_merge {
            let cmp_maxs = compare_values(ctx, current.max, next.max);
            if cmp_maxs == Ordering::Less {
                current.max = next.max;
                current.max_type = next.max_type;
            } else if cmp_maxs == Ordering::Equal && next.max_type == BoundType::Closed {
                current.max_type = BoundType::Closed;
            }
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
}

pub fn intersect_solution_sets(ctx: &Context, s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    match (s1, s2) {
        (SolutionSet::Empty, _) => SolutionSet::Empty,
        (_, SolutionSet::Empty) => SolutionSet::Empty,
        (SolutionSet::AllReals, s) => s,
        (s, SolutionSet::AllReals) => s,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => {
            intersect_intervals(ctx, &i1, &i2)
        }
        (SolutionSet::Continuous(i), SolutionSet::Union(u)) => {
            // Intersect i with each interval in u
            let mut new_u = Vec::new();
            for interval in u {
                let res = intersect_intervals(ctx, &i, &interval);
                match res {
                    SolutionSet::Continuous(new_i) => new_u.push(new_i),
                    SolutionSet::Discrete(_d) => {
                        // Complex. Let's assume for now we get Continuous intervals.
                    }
                    _ => {}
                }
            }
            if new_u.is_empty() {
                SolutionSet::Empty
            } else if new_u.len() == 1 {
                SolutionSet::Continuous(new_u[0].clone())
            } else {
                SolutionSet::Union(new_u)
            }
        }
        (SolutionSet::Union(u), SolutionSet::Continuous(i)) => {
            intersect_solution_sets(ctx, SolutionSet::Continuous(i), SolutionSet::Union(u))
        }
        (SolutionSet::Union(u1), SolutionSet::Union(u2)) => {
            // Distributive property: (A U B) n (C U D) = (A n C) U (A n D) U (B n C) U (B n D)
            let mut new_u = Vec::new();
            for i1 in &u1 {
                for i2 in &u2 {
                    let res = intersect_intervals(ctx, i1, i2);
                    if let SolutionSet::Continuous(new_i) = res {
                        new_u.push(new_i)
                    }
                }
            }
            if new_u.is_empty() {
                SolutionSet::Empty
            } else if new_u.len() == 1 {
                SolutionSet::Continuous(new_u[0].clone())
            } else {
                SolutionSet::Union(new_u)
            }
        }
        _ => SolutionSet::Empty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    fn make_interval(
        ctx: &mut Context,
        min: i64,
        min_type: BoundType,
        max: i64,
        max_type: BoundType,
    ) -> Interval {
        Interval {
            min: ctx.num(min),
            min_type,
            max: ctx.num(max),
            max_type,
        }
    }

    #[test]
    fn test_merge_intervals_overlap() {
        let mut ctx = Context::new();
        let i1 = make_interval(&mut ctx, 0, BoundType::Closed, 2, BoundType::Closed);
        let i2 = make_interval(&mut ctx, 1, BoundType::Closed, 3, BoundType::Closed);

        let s1 = SolutionSet::Continuous(i1);
        let s2 = SolutionSet::Continuous(i2);

        let union = union_solution_sets(&ctx, s1, s2);

        if let SolutionSet::Continuous(i) = union {
            assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
            assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 3.into());
        } else {
            panic!("Expected Continuous set, got {:?}", union);
        }
    }

    #[test]
    fn test_merge_intervals_touching() {
        let mut ctx = Context::new();
        let i1 = make_interval(&mut ctx, 0, BoundType::Closed, 1, BoundType::Closed);
        let i2 = make_interval(&mut ctx, 1, BoundType::Closed, 2, BoundType::Closed);

        let s1 = SolutionSet::Continuous(i1);
        let s2 = SolutionSet::Continuous(i2);

        let union = union_solution_sets(&ctx, s1, s2);

        if let SolutionSet::Continuous(i) = union {
            assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
            assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 2.into());
        } else {
            panic!("Expected Continuous set, got {:?}", union);
        }
    }

    #[test]
    fn test_merge_intervals_touching_open_closed() {
        let mut ctx = Context::new();
        let i1 = make_interval(&mut ctx, 0, BoundType::Closed, 1, BoundType::Open); // [0, 1)
        let i2 = make_interval(&mut ctx, 1, BoundType::Closed, 2, BoundType::Closed); // [1, 2]

        let s1 = SolutionSet::Continuous(i1);
        let s2 = SolutionSet::Continuous(i2);

        let union = union_solution_sets(&ctx, s1, s2);

        if let SolutionSet::Continuous(i) = union {
            assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
            assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 2.into());
        } else {
            panic!("Expected Continuous set, got {:?}", union);
        }
    }

    #[test]
    fn test_merge_intervals_disjoint() {
        let mut ctx = Context::new();
        let i1 = make_interval(&mut ctx, 0, BoundType::Closed, 1, BoundType::Closed);
        let i2 = make_interval(&mut ctx, 2, BoundType::Closed, 3, BoundType::Closed);

        let s1 = SolutionSet::Continuous(i1);
        let s2 = SolutionSet::Continuous(i2);

        let union = union_solution_sets(&ctx, s1, s2);

        if let SolutionSet::Union(intervals) = union {
            assert_eq!(intervals.len(), 2);
        } else {
            panic!("Expected Union set, got {:?}", union);
        }
    }

    #[test]
    fn test_isolated_var_solution_eq() {
        let mut ctx = Context::new();
        let rhs = ctx.num(7);
        let set = isolated_var_solution(&mut ctx, rhs, RelOp::Eq);
        assert!(matches!(set, SolutionSet::Discrete(v) if v == vec![rhs]));
    }

    #[test]
    fn test_isolated_var_solution_neq() {
        let mut ctx = Context::new();
        let rhs = ctx.num(5);
        let set = isolated_var_solution(&mut ctx, rhs, RelOp::Neq);
        match set {
            SolutionSet::Union(intervals) => {
                assert_eq!(intervals.len(), 2);
                assert!(is_neg_infinity(&ctx, intervals[0].min));
                assert!(is_infinity(&ctx, intervals[1].max));
            }
            other => panic!("Expected Union, got {:?}", other),
        }
    }
}
