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

    // Exact comparison of quadratic-surd values (e.g. the irrational roots of a quadratic) by the
    // sign of `a − b`. The structural fallback below does NOT reflect numeric value, so irrational
    // interval endpoints were emitted UNORDERED — `solve(x^2-3<0)` produced `(√3, -√3)`, a reversed
    // (empty) interval, and `solve(x^2-3>0)` produced an all-of-ℝ union. Rational endpoints already
    // ordered correctly via the Number branch above; this closes the surd case.
    if let Some(ord) = compare_quadratic_surds(ctx, a, b) {
        return ord;
    }

    // Fallback: Use structural comparison if we can't compare values
    cas_ast::ordering::compare_expr(ctx, a, b)
}

/// Decompose `expr` into the quadratic-surd normal form `A + B·√n` (`n ≥ 0` rational), also
/// recognising the golden ratio `φ = ½ + ½·√5`. `None` for anything outside a single real
/// quadratic surd (delegates to [`cas_math::root_forms::as_linear_surd`]).
fn as_surd_value(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational, BigRational)> {
    if matches!(ctx.get(expr), Expr::Constant(Constant::Phi)) {
        let half = BigRational::new(1.into(), 2.into());
        return Some((half.clone(), half, BigRational::from_integer(5.into())));
    }
    cas_math::root_forms::as_linear_surd(ctx, expr)
}

/// Exact sign of the quadratic surd `p + q·√n` (`n ≥ 0`).
fn sign_of_linear_surd(p: &BigRational, q: &BigRational, n: &BigRational) -> Ordering {
    let zero = BigRational::zero();
    if q.is_zero() || n.is_zero() {
        return p.cmp(&zero);
    }
    // n > 0, q ≠ 0.
    if p.is_zero() {
        return q.cmp(&zero); // sign(q), since √n > 0
    }
    let sp = p.cmp(&zero);
    let sq = q.cmp(&zero);
    if sp == sq {
        return sp; // same sign ⇒ that sign
    }
    // Opposite signs: sign(p + q·√n) = sign(q)·sign(q²·n − p²).
    let inner = (q.clone() * q.clone() * n.clone()).cmp(&(p.clone() * p.clone()));
    if sq == Ordering::Less {
        inner.reverse()
    } else {
        inner
    }
}

/// Exact sign of `p + q·√m + s·√n` (`m, n ≥ 0` rational, all coefficients rational), allowing
/// DISTINCT radicands `m ≠ n`. Two nested squarings, each comparing a rational against a single
/// quadratic surd via [`sign_of_linear_surd`] — fully exact (no f64).
fn sign_of_sum_two_surds(
    p: &BigRational,
    q: &BigRational,
    m: &BigRational,
    s: &BigRational,
    n: &BigRational,
) -> Ordering {
    let zero = BigRational::zero();
    // sign(X) where X = q·√m + s·√n: each term's sign, breaking a sign conflict by comparing the
    // squared magnitudes `q²m` vs `s²n` (both ≥ 0, so the comparison is on rationals).
    let q_term_zero = q.is_zero() || m.is_zero();
    let s_term_zero = s.is_zero() || n.is_zero();
    let sq = if q_term_zero {
        Ordering::Equal
    } else {
        q.cmp(&zero)
    };
    let ss = if s_term_zero {
        Ordering::Equal
    } else {
        s.cmp(&zero)
    };
    let q_sq_m = q * q * m;
    let s_sq_n = s * s * n;
    let sign_x = match (sq, ss) {
        (Ordering::Equal, _) => ss, // q·√m term is zero ⇒ sign(s·√n)
        (_, Ordering::Equal) => sq, // s·√n term is zero ⇒ sign(q·√m)
        _ if sq == ss => sq,        // same sign ⇒ that sign
        // Opposite signs ⇒ the larger magnitude wins: |q√m| vs |s√n| ⟺ q²m vs s²n.
        _ => match q_sq_m.cmp(&s_sq_n) {
            Ordering::Greater => sq,
            Ordering::Less => ss,
            Ordering::Equal => Ordering::Equal,
        },
    };

    let sign_p = p.cmp(&zero);
    if sign_p == Ordering::Equal {
        return sign_x;
    }
    if sign_x == Ordering::Equal {
        return sign_p;
    }
    if sign_p == sign_x {
        return sign_p;
    }
    // p and X have OPPOSITE signs ⇒ sign(p + X) is the sign of the larger magnitude. Compare
    // `p²` vs `X² = (q²m + s²n) + 2qs·√(mn)` exactly: sign(p² − X²) decides which wins.
    let p_sq = p * p;
    let rational_part = &p_sq - &q_sq_m - &s_sq_n;
    let two = BigRational::new(2.into(), 1.into());
    let surd_coeff = -(two * q * s);
    let mn = m * n;
    match sign_of_linear_surd(&rational_part, &surd_coeff, &mn) {
        Ordering::Greater => sign_p, // |p| > |X|
        Ordering::Less => sign_x,    // |X| > |p|
        Ordering::Equal => Ordering::Equal,
    }
}

/// Compare two quadratic-surd values `a`, `b` by VALUE via the exact sign of `a − b`. `None` only
/// when either is not a single real quadratic surd. DISTINCT radicands with both surd parts
/// non-zero (e.g. domain bound `√6` against constraint `√2 − 1`) are ordered exactly via
/// [`sign_of_sum_two_surds`].
fn compare_quadratic_surds(ctx: &Context, a: ExprId, b: ExprId) -> Option<Ordering> {
    let (aa, ab, an) = as_surd_value(ctx, a)?;
    let (ba, bb, bn) = as_surd_value(ctx, b)?;
    // a − b = (aa − ba) + ab·√an − bb·√bn. Combine the surd parts when they share a radicand (or one
    // side has no surd part); otherwise take the exact two-surd sign.
    let p = aa - ba;
    let (q, n) = if ab.is_zero() {
        (-bb, bn)
    } else if bb.is_zero() {
        (ab, an)
    } else if an == bn {
        (ab - bb, an)
    } else {
        let neg_bb = -bb;
        return Some(sign_of_sum_two_surds(&p, &ab, &an, &neg_bb, &bn));
    };
    Some(sign_of_linear_surd(&p, &q, &n))
}

/// Order two expression ids so the first is `<=` the second under `compare_values`.
pub fn order_pair_by_value(ctx: &Context, a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    if compare_values(ctx, a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    }
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

/// Finalize a 2-branch sign split by intersecting each branch with its domain
/// condition and then unioning both branch results.
pub fn finalize_sign_split_solution_set(
    ctx: &Context,
    positive_branch: SolutionSet,
    positive_domain: SolutionSet,
    negative_branch: SolutionSet,
    negative_domain: SolutionSet,
) -> SolutionSet {
    let final_pos = intersect_solution_sets(ctx, positive_branch, positive_domain);
    let final_neg = intersect_solution_sets(ctx, negative_branch, negative_domain);
    union_solution_sets(ctx, final_pos, final_neg)
}

/// Finalize product-zero inequality split from four solved branch equations:
/// `(A_case1 ∩ B_case1) ∪ (A_case2 ∩ B_case2)`.
pub fn finalize_product_zero_inequality_solution_set(
    ctx: &Context,
    case1_left: SolutionSet,
    case1_right: SolutionSet,
    case2_left: SolutionSet,
    case2_right: SolutionSet,
) -> SolutionSet {
    let case_set1 = intersect_solution_sets(ctx, case1_left, case1_right);
    let case_set2 = intersect_solution_sets(ctx, case2_left, case2_right);
    union_solution_sets(ctx, case_set1, case_set2)
}

/// Finalize isolated-denominator sign split by applying `x > 0` and `x < 0`
/// open-domain guards to positive/negative branches.
pub fn finalize_isolated_denominator_sign_split_solution_set(
    ctx: &mut Context,
    positive_branch: SolutionSet,
    negative_branch: SolutionSet,
) -> SolutionSet {
    let domain_pos = open_positive_domain(ctx);
    let domain_neg = open_negative_domain(ctx);
    finalize_sign_split_solution_set(
        ctx,
        positive_branch,
        domain_pos,
        negative_branch,
        domain_neg,
    )
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
        // A discrete set unioned with intervals: each isolated point becomes a degenerate closed
        // interval `[p, p]`, then the shared `merge_intervals` tail keeps it as an isolated point
        // unless it abuts an open endpoint (`{0} ∪ (0, 5) = [0, 5)`). Without these arms the old
        // catch-all `(s1, _)` silently DROPPED the interval side, losing a real solution (e.g. the
        // negative ray of a touch-point inequality `x + 1/x ≤ 2`, whose solution is `(−∞, 0) ∪ {1}`).
        // A purely-point result therefore renders as a degenerate interval; callers that want the
        // `{p}` form re-collapse the final result via `collapse_degenerate_intervals`.
        (SolutionSet::Discrete(d), SolutionSet::Continuous(i))
        | (SolutionSet::Continuous(i), SolutionSet::Discrete(d)) => {
            let mut ivs: Vec<Interval> = d
                .into_iter()
                .map(|p| interval(p, BoundType::Closed, p, BoundType::Closed))
                .collect();
            ivs.push(i);
            ivs
        }
        (SolutionSet::Discrete(d), SolutionSet::Union(mut u))
        | (SolutionSet::Union(mut u), SolutionSet::Discrete(d)) => {
            for p in d {
                u.push(interval(p, BoundType::Closed, p, BoundType::Closed));
            }
            u
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

    #[test]
    fn compare_values_orders_quadratic_surds_by_value() {
        use cas_parser::parse;
        // -√3 < √3, √2 < 2, (1-√5)/2 < φ, and φ > 1 — by VALUE, not structure. Before the surd
        // branch these fell to structural comparison and irrational interval endpoints reversed.
        let cases: [(&str, &str, Ordering); 6] = [
            ("sqrt(3)", "-sqrt(3)", Ordering::Greater),
            ("-sqrt(3)", "sqrt(3)", Ordering::Less),
            ("sqrt(2)", "2", Ordering::Less),
            ("phi", "1/2*(1 - sqrt(5))", Ordering::Greater),
            ("1/2*(1 - sqrt(5))", "phi", Ordering::Less),
            ("phi", "1", Ordering::Greater),
        ];
        for (a_src, b_src, want) in cases {
            let mut ctx = Context::new();
            let a = parse(a_src, &mut ctx).expect("parse a");
            let b = parse(b_src, &mut ctx).expect("parse b");
            assert_eq!(
                compare_values(&ctx, a, b),
                want,
                "compare_values({a_src}, {b_src})"
            );
        }
    }

    #[test]
    fn compare_values_orders_distinct_radicand_surds() {
        use cas_parser::parse;
        // Two surds with DIFFERENT radicands and both surd parts non-zero — the case the
        // structural fallback used to mis-order (silently dropping radical-inequality
        // constraints). `√6 ≈ 2.449`, `√2 − 1 ≈ 0.414`, `√7 ≈ 2.646`, `(√13 − 1)/2 ≈ 1.303`.
        let cases: [(&str, &str, Ordering); 6] = [
            ("sqrt(6)", "sqrt(2) - 1", Ordering::Greater),
            ("sqrt(2) - 1", "sqrt(6)", Ordering::Less),
            ("sqrt(7)", "1/2*(sqrt(13) - 1)", Ordering::Greater),
            ("1/2*(sqrt(13) - 1)", "sqrt(7)", Ordering::Less),
            // Equal value via distinct radicands: √8 = 2·√2.
            ("sqrt(8)", "2*sqrt(2)", Ordering::Equal),
            ("sqrt(3) + sqrt(2)", "sqrt(2) + sqrt(3)", Ordering::Equal),
        ];
        for (a_src, b_src, want) in cases {
            let mut ctx = Context::new();
            let a = parse(a_src, &mut ctx).expect("parse a");
            let b = parse(b_src, &mut ctx).expect("parse b");
            assert_eq!(
                compare_values(&ctx, a, b),
                want,
                "compare_values({a_src}, {b_src})"
            );
        }
    }

    #[test]
    fn sign_of_sum_two_surds_matches_float_over_grid() {
        // Exhaustive deterministic grid: sign(p + q·√m + s·√n) for distinct radicands must
        // agree with the f64 evaluation wherever the latter is decisively non-zero. This pins
        // the exact nested-squaring arithmetic against an independent (floating) oracle.
        let r = |num: i64, den: i64| BigRational::new(num.into(), den.into());
        let coeffs = [-5i64, -3, -2, -1, 1, 2, 3, 4];
        let rads = [0i64, 2, 3, 5, 6, 7, 8, 11, 13];
        let mut checked = 0u64;
        for &pn in &coeffs {
            for &pd in &[1i64, 2, 3] {
                for &q in &coeffs {
                    for &s in &coeffs {
                        for &m in &rads {
                            for &n in &rads {
                                let (pp, qq, mm, ss, nn) =
                                    (r(pn, pd), r(q, 1), r(m, 1), r(s, 1), r(n, 1));
                                let got = sign_of_sum_two_surds(&pp, &qq, &mm, &ss, &nn);
                                let val = (pn as f64) / (pd as f64)
                                    + (q as f64) * (m as f64).sqrt()
                                    + (s as f64) * (n as f64).sqrt();
                                let want = if val > 1e-9 {
                                    Ordering::Greater
                                } else if val < -1e-9 {
                                    Ordering::Less
                                } else {
                                    continue; // too close to 0 for f64 to adjudicate
                                };
                                assert_eq!(
                                    got, want,
                                    "sign({pn}/{pd} + {q}·√{m} + {s}·√{n}) = {val}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 5000, "expected a broad grid, checked {checked}");
    }

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
    fn test_union_discrete_with_continuous_keeps_both() {
        let mut ctx = Context::new();
        // {5} ∪ (0, 2): both kept (the old catch-all dropped the interval, losing a real solution).
        let point = SolutionSet::Discrete(vec![ctx.num(5)]);
        let iv = SolutionSet::Continuous(make_interval(
            &mut ctx,
            0,
            BoundType::Open,
            2,
            BoundType::Open,
        ));
        match union_solution_sets(&ctx, point, iv) {
            SolutionSet::Union(ivs) => {
                assert_eq!(ivs.len(), 2, "expected interval + isolated point")
            }
            other => panic!("expected Union, got {other:?}"),
        }
        // Symmetric order, and a touching point closes the open endpoint: (0, 5) ∪ {0} = [0, 5).
        let iv2 = SolutionSet::Continuous(make_interval(
            &mut ctx,
            0,
            BoundType::Open,
            5,
            BoundType::Open,
        ));
        let point0 = SolutionSet::Discrete(vec![ctx.num(0)]);
        match union_solution_sets(&ctx, iv2, point0) {
            SolutionSet::Continuous(i) => {
                assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
                assert_eq!(i.min_type, BoundType::Closed); // closed by the absorbed point
                assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 5.into());
            }
            other => panic!("expected merged [0, 5), got {other:?}"),
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
    fn test_finalize_sign_split_solution_set_applies_domains() {
        let mut ctx = Context::new();
        let positive_branch = SolutionSet::Continuous(Interval {
            min: ctx.num(1),
            min_type: BoundType::Open,
            max: ctx.num(10),
            max_type: BoundType::Open,
        });
        let positive_domain = open_positive_domain(&mut ctx);
        let negative_branch = SolutionSet::Continuous(Interval {
            min: ctx.num(-10),
            min_type: BoundType::Open,
            max: ctx.num(-1),
            max_type: BoundType::Open,
        });
        let negative_domain = open_negative_domain(&mut ctx);

        let out = finalize_sign_split_solution_set(
            &ctx,
            positive_branch,
            positive_domain,
            negative_branch,
            negative_domain,
        );
        assert!(matches!(out, SolutionSet::Union(intervals) if intervals.len() == 2));
    }

    #[test]
    fn test_finalize_product_zero_inequality_solution_set_combines_cases() {
        let mut ctx = Context::new();
        let case1_left = SolutionSet::Continuous(Interval {
            min: ctx.num(0),
            min_type: BoundType::Open,
            max: pos_inf(&mut ctx),
            max_type: BoundType::Open,
        });
        let case1_right = SolutionSet::Continuous(Interval {
            min: neg_inf(&mut ctx),
            min_type: BoundType::Open,
            max: ctx.num(5),
            max_type: BoundType::Open,
        });
        let case2_left = SolutionSet::Continuous(Interval {
            min: ctx.num(3),
            min_type: BoundType::Open,
            max: pos_inf(&mut ctx),
            max_type: BoundType::Open,
        });
        let case2_right = SolutionSet::Continuous(Interval {
            min: neg_inf(&mut ctx),
            min_type: BoundType::Open,
            max: ctx.num(10),
            max_type: BoundType::Open,
        });

        let out = finalize_product_zero_inequality_solution_set(
            &ctx,
            case1_left,
            case1_right,
            case2_left,
            case2_right,
        );
        match out {
            SolutionSet::Continuous(i) => {
                assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
                assert_eq!(get_number(&ctx, i.max).unwrap().to_integer(), 10.into());
            }
            other => panic!("Expected merged interval, got {:?}", other),
        }
    }

    #[test]
    fn test_finalize_isolated_denominator_sign_split_solution_set_applies_open_domains() {
        let mut ctx = Context::new();
        let out = finalize_isolated_denominator_sign_split_solution_set(
            &mut ctx,
            SolutionSet::AllReals,
            SolutionSet::Empty,
        );
        match out {
            SolutionSet::Continuous(i) => {
                assert_eq!(get_number(&ctx, i.min).unwrap().to_integer(), 0.into());
                assert!(is_infinity(&ctx, i.max));
                assert_eq!(i.min_type, BoundType::Open);
                assert_eq!(i.max_type, BoundType::Open);
            }
            other => panic!("Expected open positive interval, got {:?}", other),
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
    fn test_order_pair_by_value_swaps() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let two = ctx.num(2);
        let (left, right) = order_pair_by_value(&ctx, five, two);
        assert_eq!(get_number(&ctx, left).unwrap().to_integer(), 2.into());
        assert_eq!(get_number(&ctx, right).unwrap().to_integer(), 5.into());
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
