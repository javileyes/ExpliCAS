use cas_ast::{Expr, SolutionSet, Interval, BoundType, Constant};
use std::rc::Rc;
use std::cmp::Ordering;
use num_rational::BigRational;

// Helper to create -infinity
pub fn neg_inf() -> Rc<Expr> {
    Rc::new(Expr::Neg(Rc::new(Expr::Constant(Constant::Infinity))))
}

// Helper to create +infinity
pub fn pos_inf() -> Rc<Expr> {
    Rc::new(Expr::Constant(Constant::Infinity))
}

pub fn is_infinity(expr: &Expr) -> bool {
    matches!(expr, Expr::Constant(Constant::Infinity))
}

pub fn is_neg_infinity(expr: &Expr) -> bool {
    match expr {
        Expr::Neg(inner) => is_infinity(inner),
        _ => false,
    }
}

pub fn get_number(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => get_number(inner).map(|n| -n),
        _ => None,
    }
}

pub fn compare_values(a: &Expr, b: &Expr) -> Ordering {
    // Handle Infinity
    let a_inf = is_infinity(a);
    let b_inf = is_infinity(b);
    let a_neg_inf = is_neg_infinity(a);
    let b_neg_inf = is_neg_infinity(b);
    
    if a_neg_inf {
        if b_neg_inf { return Ordering::Equal; }
        return Ordering::Less;
    }
    if b_neg_inf { return Ordering::Greater; }
    
    if a_inf {
        if b_inf { return Ordering::Equal; }
        return Ordering::Greater;
    }
    if b_inf { return Ordering::Less; }
    
    // Handle Numbers
    if let (Some(n1), Some(n2)) = (get_number(a), get_number(b)) {
        return n1.cmp(&n2);
    }
    
    // Fallback: Use structural comparison if we can't compare values
    crate::ordering::compare_expr(a, b)
}

pub fn intersect_intervals(i1: &Interval, i2: &Interval) -> SolutionSet {
    // Intersection of [a, b] and [c, d] is [max(a,c), min(b,d)]
    
    // Compare mins
    let (min, min_type) = match compare_values(&i1.min, &i2.min) {
        Ordering::Less => (i2.min.clone(), i2.min_type.clone()), // i1.min < i2.min -> take i2
        Ordering::Greater => (i1.min.clone(), i1.min_type.clone()), // i1.min > i2.min -> take i1
        Ordering::Equal => {
            let type_ = if i1.min_type == BoundType::Open || i2.min_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.min.clone(), type_)
        }
    };

    // Compare maxs
    let (max, max_type) = match compare_values(&i1.max, &i2.max) {
        Ordering::Less => (i1.max.clone(), i1.max_type.clone()), // i1.max < i2.max -> take i1
        Ordering::Greater => (i2.max.clone(), i2.max_type.clone()), // i1.max > i2.max -> take i2
        Ordering::Equal => {
            let type_ = if i1.max_type == BoundType::Open || i2.max_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.max.clone(), type_)
        }
    };
    
    // Check if valid interval (min < max)
    match compare_values(&min, &max) {
        Ordering::Less => SolutionSet::Continuous(Interval { min, min_type, max, max_type }),
        Ordering::Equal => {
            if min_type == BoundType::Closed && max_type == BoundType::Closed {
                SolutionSet::Discrete(vec![min])
            } else {
                SolutionSet::Empty
            }
        },
        Ordering::Greater => SolutionSet::Empty,
    }
}

pub fn union_solution_sets(s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    match (s1, s2) {
        (SolutionSet::Empty, s) | (s, SolutionSet::Empty) => s,
        (SolutionSet::AllReals, _) | (_, SolutionSet::AllReals) => SolutionSet::AllReals,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => {
            // TODO: Merge if overlapping or touching
            let mut intervals = vec![i1, i2];
            intervals.sort_by(|a, b| compare_values(&a.min, &b.min));
            SolutionSet::Union(intervals)
        },
        (SolutionSet::Continuous(i), SolutionSet::Union(mut u)) | (SolutionSet::Union(mut u), SolutionSet::Continuous(i)) => {
            u.push(i);
             u.sort_by(|a, b| compare_values(&a.min, &b.min));
            SolutionSet::Union(u)
        },
        (SolutionSet::Union(mut u1), SolutionSet::Union(u2)) => {
            u1.extend(u2);
             u1.sort_by(|a, b| compare_values(&a.min, &b.min));
            SolutionSet::Union(u1)
        },

        (SolutionSet::Discrete(mut d1), SolutionSet::Discrete(d2)) => {
            d1.extend(d2);
            SolutionSet::Discrete(d1)
        },
        (s1, _) => s1, 
    }
}

pub fn intersect_solution_sets(s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    match (s1, s2) {
        (SolutionSet::Empty, _) => SolutionSet::Empty,
        (_, SolutionSet::Empty) => SolutionSet::Empty,
        (SolutionSet::AllReals, s) => s,
        (s, SolutionSet::AllReals) => s,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => {
            intersect_intervals(&i1, &i2)
        },
        (SolutionSet::Continuous(i), SolutionSet::Union(u)) => {
            // Intersect i with each interval in u
            let mut new_u = Vec::new();
            for interval in u {
                let res = intersect_intervals(&i, &interval);
                match res {
                    SolutionSet::Continuous(new_i) => new_u.push(new_i),
                    SolutionSet::Discrete(_d) => {
                        // Complex. Let's assume for now we get Continuous intervals.
                    },
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
        },
        (SolutionSet::Union(u), SolutionSet::Continuous(i)) => {
            intersect_solution_sets(SolutionSet::Continuous(i), SolutionSet::Union(u))
        },
        (SolutionSet::Union(u1), SolutionSet::Union(u2)) => {
            // Distributive property: (A U B) n (C U D) = (A n C) U (A n D) U (B n C) U (B n D)
            let mut new_u = Vec::new();
            for i1 in &u1 {
                for i2 in &u2 {
                    let res = intersect_intervals(i1, i2);
                    match res {
                        SolutionSet::Continuous(new_i) => new_u.push(new_i),
                        _ => {}
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
        },
        _ => SolutionSet::Empty,
    }
}
