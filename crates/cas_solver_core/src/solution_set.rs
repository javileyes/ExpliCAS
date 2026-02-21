use cas_ast::{BoundType, Constant, Context, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_rational::BigRational;
use num_traits::Zero;
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

/// Sort and deduplicate expression ids using canonical structural ordering.
pub fn sort_and_dedup_exprs(ctx: &Context, exprs: &mut Vec<ExprId>) {
    exprs.sort_by(|a, b| cas_ast::ordering::compare_expr(ctx, *a, *b));
    exprs.dedup_by(|a, b| cas_ast::ordering::compare_expr(ctx, *a, *b) == Ordering::Equal);
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

/// Open positive domain: `(0, +inf)`.
pub fn open_positive_domain(ctx: &mut Context) -> SolutionSet {
    SolutionSet::Continuous(Interval {
        min: ctx.num(0),
        min_type: BoundType::Open,
        max: pos_inf(ctx),
        max_type: BoundType::Open,
    })
}

/// Open negative domain: `(-inf, 0)`.
pub fn open_negative_domain(ctx: &mut Context) -> SolutionSet {
    SolutionSet::Continuous(Interval {
        min: neg_inf(ctx),
        min_type: BoundType::Open,
        max: ctx.num(0),
        max_type: BoundType::Open,
    })
}

fn interval(min: ExprId, min_type: BoundType, max: ExprId, max_type: BoundType) -> Interval {
    Interval {
        min,
        min_type,
        max,
        max_type,
    }
}

fn open_interval(min: ExprId, max: ExprId) -> SolutionSet {
    SolutionSet::Continuous(interval(min, BoundType::Open, max, BoundType::Open))
}

fn closed_interval(min: ExprId, max: ExprId) -> SolutionSet {
    SolutionSet::Continuous(interval(min, BoundType::Closed, max, BoundType::Closed))
}

fn except_point(ctx: &mut Context, point: ExprId) -> SolutionSet {
    SolutionSet::Union(vec![
        interval(neg_inf(ctx), BoundType::Open, point, BoundType::Open),
        interval(point, BoundType::Open, pos_inf(ctx), BoundType::Open),
    ])
}

fn outside_roots(
    ctx: &mut Context,
    r1: ExprId,
    r2: ExprId,
    left_root_type: BoundType,
    right_root_type: BoundType,
) -> SolutionSet {
    SolutionSet::Union(vec![
        interval(neg_inf(ctx), BoundType::Open, r1, left_root_type),
        interval(r2, right_root_type, pos_inf(ctx), BoundType::Open),
    ])
}

/// Build solution sets for numeric quadratic relations `a*x^2 + b*x + c <op> 0`.
///
/// Assumes `r1 <= r2` when `delta > 0`. For `delta == 0`, `r1` is the repeated root.
pub fn quadratic_numeric_solution(
    ctx: &mut Context,
    op: RelOp,
    delta: &BigRational,
    opens_up: bool,
    r1: ExprId,
    r2: ExprId,
) -> SolutionSet {
    if delta > &BigRational::zero() {
        match op {
            RelOp::Eq => SolutionSet::Discrete(vec![r1, r2]),
            RelOp::Neq => SolutionSet::Union(vec![
                interval(neg_inf(ctx), BoundType::Open, r1, BoundType::Open),
                interval(r1, BoundType::Open, r2, BoundType::Open),
                interval(r2, BoundType::Open, pos_inf(ctx), BoundType::Open),
            ]),
            RelOp::Lt => {
                if opens_up {
                    open_interval(r1, r2)
                } else {
                    outside_roots(ctx, r1, r2, BoundType::Open, BoundType::Open)
                }
            }
            RelOp::Leq => {
                if opens_up {
                    closed_interval(r1, r2)
                } else {
                    outside_roots(ctx, r1, r2, BoundType::Closed, BoundType::Closed)
                }
            }
            RelOp::Gt => {
                if opens_up {
                    outside_roots(ctx, r1, r2, BoundType::Open, BoundType::Open)
                } else {
                    open_interval(r1, r2)
                }
            }
            RelOp::Geq => {
                if opens_up {
                    outside_roots(ctx, r1, r2, BoundType::Closed, BoundType::Closed)
                } else {
                    closed_interval(r1, r2)
                }
            }
        }
    } else if delta.is_zero() {
        match op {
            RelOp::Eq => SolutionSet::Discrete(vec![r1]),
            RelOp::Neq => except_point(ctx, r1),
            RelOp::Lt => {
                if opens_up {
                    SolutionSet::Empty
                } else {
                    except_point(ctx, r1)
                }
            }
            RelOp::Leq => {
                if opens_up {
                    SolutionSet::Discrete(vec![r1])
                } else {
                    SolutionSet::AllReals
                }
            }
            RelOp::Gt => {
                if opens_up {
                    except_point(ctx, r1)
                } else {
                    SolutionSet::Empty
                }
            }
            RelOp::Geq => {
                if opens_up {
                    SolutionSet::AllReals
                } else {
                    SolutionSet::Discrete(vec![r1])
                }
            }
        }
    } else {
        match op {
            RelOp::Eq => SolutionSet::Empty,
            RelOp::Neq => SolutionSet::AllReals,
            RelOp::Lt | RelOp::Leq => {
                if opens_up {
                    SolutionSet::Empty
                } else {
                    SolutionSet::AllReals
                }
            }
            RelOp::Gt | RelOp::Geq => {
                if opens_up {
                    SolutionSet::AllReals
                } else {
                    SolutionSet::Empty
                }
            }
        }
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

    #[test]
    fn test_open_positive_domain() {
        let mut ctx = Context::new();
        let set = open_positive_domain(&mut ctx);
        match set {
            SolutionSet::Continuous(i) => {
                assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
                assert!(is_infinity(&ctx, i.max));
                assert_eq!(i.min_type, BoundType::Open);
                assert_eq!(i.max_type, BoundType::Open);
            }
            other => panic!("Expected continuous interval, got {:?}", other),
        }
    }

    #[test]
    fn test_open_negative_domain() {
        let mut ctx = Context::new();
        let set = open_negative_domain(&mut ctx);
        match set {
            SolutionSet::Continuous(i) => {
                assert!(is_neg_infinity(&ctx, i.min));
                assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 0.into());
                assert_eq!(i.min_type, BoundType::Open);
                assert_eq!(i.max_type, BoundType::Open);
            }
            other => panic!("Expected continuous interval, got {:?}", other),
        }
    }

    #[test]
    fn test_sort_and_dedup_exprs() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let mut roots = vec![two, one, two];
        sort_and_dedup_exprs(&ctx, &mut roots);
        assert_eq!(roots, vec![one, two]);
    }

    #[test]
    fn test_quadratic_numeric_solution_delta_positive_eq() {
        let mut ctx = Context::new();
        let r1 = ctx.num(1);
        let r2 = ctx.num(3);
        let delta = BigRational::from_integer(4.into());
        let set = quadratic_numeric_solution(&mut ctx, RelOp::Eq, &delta, true, r1, r2);
        assert!(matches!(set, SolutionSet::Discrete(v) if v == vec![r1, r2]));
    }

    #[test]
    fn test_quadratic_numeric_solution_delta_positive_lt_opens_up() {
        let mut ctx = Context::new();
        let r1 = ctx.num(1);
        let r2 = ctx.num(3);
        let delta = BigRational::from_integer(4.into());
        let set = quadratic_numeric_solution(&mut ctx, RelOp::Lt, &delta, true, r1, r2);
        match set {
            SolutionSet::Continuous(i) => {
                assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 1.into());
                assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 3.into());
                assert_eq!(i.min_type, BoundType::Open);
                assert_eq!(i.max_type, BoundType::Open);
            }
            other => panic!("Expected continuous interval, got {:?}", other),
        }
    }

    #[test]
    fn test_quadratic_numeric_solution_delta_zero_geq_opens_down() {
        let mut ctx = Context::new();
        let r = ctx.num(2);
        let delta = BigRational::zero();
        let set = quadratic_numeric_solution(&mut ctx, RelOp::Geq, &delta, false, r, r);
        assert!(matches!(set, SolutionSet::Discrete(v) if v == vec![r]));
    }

    #[test]
    fn test_quadratic_numeric_solution_delta_negative_gt_opens_down() {
        let mut ctx = Context::new();
        let r = ctx.num(0);
        let delta = -BigRational::from_integer(1.into());
        let set = quadratic_numeric_solution(&mut ctx, RelOp::Gt, &delta, false, r, r);
        assert!(matches!(set, SolutionSet::Empty));
    }
}
