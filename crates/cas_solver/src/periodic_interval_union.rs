//! Solver-layer window algebra for `SolutionSet::PeriodicIntervalUnion`
//! (design: `docs/DESIGN_PERIODIC_INTERVAL_UNION.md` §6).
//!
//! The core `union/intersect_solution_sets` cannot combine these sets: window
//! arithmetic needs `endpoint ± period` expressions (a `&mut Context` + a
//! simplifier) and a FALLIBLE value order. This module owns the circular
//! (mod-period) algebra, following the `union_periodic_families_over_common_period`
//! architectural precedent. Every operation that cannot ORDER an endpoint
//! exactly returns `None` — callers must then decline the whole relation
//! honestly rather than combine blindly (the core-union-Periodic-drop lesson).
//!
//! Circle algebra (panel blocker, design §6): windows may legitimately
//! straddle any fixed fundamental domain (`sin u < 1/2` emits
//! `(5π/6, 13π/6)`), so a naive linear `intersect` on raw windows DROPS the
//! wrapped mass — `(0, π) ∩ (5π/6, 13π/6)` as flat intervals loses the
//! `(2kπ, π/6+2kπ)` component of `1/sin(x) > 2`. The algebra therefore:
//! anchor → translate by exact `k·T` → split at the seam → linear pass →
//! re-glue across the seam → re-establish the span invariant.

use cas_ast::{BoundType, Expr, ExprId, Interval, SolutionSet};
use cas_engine::Simplifier;
use cas_solver_core::solution_set::try_compare_values;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

/// Simplified `a - b`.
fn diff(simplifier: &mut Simplifier, a: ExprId, b: ExprId) -> ExprId {
    let e = simplifier.context.add(Expr::Sub(a, b));
    simplifier.simplify(e).0
}

/// Simplified `a + k·period` (`k` a small integer).
fn shift_by_periods(simplifier: &mut Simplifier, a: ExprId, period: ExprId, k: i64) -> ExprId {
    if k == 0 {
        return a;
    }
    let k_expr = simplifier.context.num(k);
    let k_t = simplifier.context.add(Expr::Mul(k_expr, period));
    let e = simplifier.context.add(Expr::Add(a, k_t));
    simplifier.simplify(e).0
}

/// Exact value order of two endpoints, or `None` when no oracle decides.
/// Chain: simplified difference as a plain number → `const_value_bounds`
/// separation on the difference (covers the π-lattice: `5π/6 − π/6` →
/// `2π/3`) → the core fallible comparator on the raw pair.
fn try_order(simplifier: &mut Simplifier, a: ExprId, b: ExprId) -> Option<Ordering> {
    if a == b {
        return Some(Ordering::Equal);
    }
    let d = diff(simplifier, a, b);
    let ctx = &simplifier.context;
    if let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, d) {
        return Some(if n.is_zero() {
            Ordering::Equal
        } else if n.is_positive() {
            Ordering::Greater
        } else {
            Ordering::Less
        });
    }
    if let Some((lo, hi)) = cas_math::const_sign::const_value_bounds(ctx, d) {
        if hi < BigRational::zero() {
            return Some(Ordering::Less);
        }
        if lo > BigRational::zero() {
            return Some(Ordering::Greater);
        }
    }
    try_compare_values(ctx, a, b)
}

/// Exact rational value of `a / b` (via the simplified quotient), or `None`.
/// Decides how many periods separate two endpoints on the π-lattice
/// (`(5π/6 − π/6) / 2π → 1/3`) and for plain rational lattices alike.
fn try_exact_ratio(simplifier: &mut Simplifier, a: ExprId, b: ExprId) -> Option<BigRational> {
    let q = simplifier.context.add(Expr::Div(a, b));
    let q = simplifier.simplify(q).0;
    // The simplifier may leave a rational as a nested `5/6/2` rather than a
    // single Number node; fold it with the exact rational evaluator.
    cas_math::numeric_eval::as_rational_const(&simplifier.context, q)
}

/// A window with resolved endpoint order (`min < max` established by the
/// caller through `try_order`).
#[derive(Clone, Debug)]
struct Win {
    min: ExprId,
    min_type: BoundType,
    max: ExprId,
    max_type: BoundType,
}

impl Win {
    fn from_interval(iv: &Interval) -> Self {
        Win {
            min: iv.min,
            min_type: iv.min_type.clone(),
            max: iv.max,
            max_type: iv.max_type.clone(),
        }
    }
    fn to_interval(&self) -> Interval {
        Interval {
            min: self.min,
            min_type: self.min_type.clone(),
            max: self.max,
            max_type: self.max_type.clone(),
        }
    }
}

/// Translate `w` by the exact number of periods that lands its LEFT endpoint
/// in `[anchor, anchor + T)`, then split it at the seam `anchor + T` if it
/// crosses. Returns 1 or 2 pieces, all inside `[anchor, anchor + T]`.
/// `None` when the required shift is not exactly decidable.
fn normalize_window(
    simplifier: &mut Simplifier,
    w: &Win,
    anchor: ExprId,
    period: ExprId,
) -> Option<Vec<Win>> {
    // k = floor((w.min - anchor) / T): exact rational ratio required.
    let d = diff(simplifier, w.min, anchor);
    let ratio = try_exact_ratio(simplifier, d, period)?;
    let k = ratio.floor().to_integer();
    let k: i64 = i64::try_from(&k).ok()?;
    let min = shift_by_periods(simplifier, w.min, period, -k);
    let max = shift_by_periods(simplifier, w.max, period, -k);

    // Seam check: does the (shifted) window cross anchor + T?
    let seam = shift_by_periods(simplifier, anchor, period, 1);
    match try_order(simplifier, max, seam)? {
        Ordering::Less | Ordering::Equal => Some(vec![Win {
            min,
            min_type: w.min_type.clone(),
            max,
            max_type: w.max_type.clone(),
        }]),
        Ordering::Greater => {
            // Split: (min, seam) ∪ [anchor, max − T). The seam VALUE (≡ anchor
            // mod T) is interior to the window, so the anchor-side piece is
            // CLOSED at anchor.
            let wrapped_max = shift_by_periods(simplifier, max, period, -1);
            Some(vec![
                Win {
                    min,
                    min_type: w.min_type.clone(),
                    max: seam,
                    max_type: BoundType::Open,
                },
                Win {
                    min: anchor,
                    min_type: BoundType::Closed,
                    max: wrapped_max,
                    max_type: w.max_type.clone(),
                },
            ])
        }
    }
}

/// Sort windows by `min` with the fallible order (insertion sort; tiny N).
fn try_sort(simplifier: &mut Simplifier, wins: &mut [Win]) -> Option<()> {
    for i in 1..wins.len() {
        let mut j = i;
        while j > 0 {
            match try_order(simplifier, wins[j - 1].min, wins[j].min)? {
                Ordering::Greater => {
                    wins.swap(j - 1, j);
                    j -= 1;
                }
                _ => break,
            }
        }
    }
    Some(())
}

/// Linear union of sorted, normalized windows: merge overlapping or touching
/// pieces (touching counts as joined when at least one side is Closed).
fn linear_union(simplifier: &mut Simplifier, wins: Vec<Win>) -> Option<Vec<Win>> {
    let mut out: Vec<Win> = Vec::with_capacity(wins.len());
    for w in wins {
        let Some(last) = out.last_mut() else {
            out.push(w);
            continue;
        };
        // Equal mins with mixed closedness: the Closed one upgrades the
        // merged window ((0,π) ∪ [0,π/6] = [0,π)); sorted order guarantees
        // w.min >= last.min so only the Equal case can occur here.
        if w.min_type == BoundType::Closed
            && try_order(simplifier, w.min, last.min)? == Ordering::Equal
        {
            last.min_type = BoundType::Closed;
        }
        // Compare w.min against last.max.
        match try_order(simplifier, w.min, last.max)? {
            Ordering::Greater => out.push(w),
            Ordering::Equal => {
                if w.min_type == BoundType::Closed || last.max_type == BoundType::Closed {
                    // Touching with the point covered: extend.
                    match try_order(simplifier, w.max, last.max)? {
                        Ordering::Greater => {
                            last.max = w.max;
                            last.max_type = w.max_type.clone();
                        }
                        Ordering::Equal => {
                            if w.max_type == BoundType::Closed {
                                last.max_type = BoundType::Closed;
                            }
                        }
                        Ordering::Less => {}
                    }
                } else {
                    // Both open at the touch point: genuinely disjoint.
                    out.push(w);
                }
            }
            Ordering::Less => {
                // Overlap: extend the last window if w reaches further.
                match try_order(simplifier, w.max, last.max)? {
                    Ordering::Greater => {
                        last.max = w.max;
                        last.max_type = w.max_type.clone();
                    }
                    Ordering::Equal => {
                        if w.max_type == BoundType::Closed {
                            last.max_type = BoundType::Closed;
                        }
                    }
                    Ordering::Less => {}
                }
            }
        }
    }
    Some(out)
}

/// Linear intersection of two lists of normalized windows (tiny N × M).
fn linear_intersection(simplifier: &mut Simplifier, a: &[Win], b: &[Win]) -> Option<Vec<Win>> {
    let mut out = Vec::new();
    for wa in a {
        for wb in b {
            // lo = max(wa.min, wb.min); hi = min(wa.max, wb.max).
            let (lo, lo_type) = match try_order(simplifier, wa.min, wb.min)? {
                Ordering::Greater => (wa.min, wa.min_type.clone()),
                Ordering::Less => (wb.min, wb.min_type.clone()),
                Ordering::Equal => (
                    wa.min,
                    if wa.min_type == BoundType::Open || wb.min_type == BoundType::Open {
                        BoundType::Open
                    } else {
                        BoundType::Closed
                    },
                ),
            };
            let (hi, hi_type) = match try_order(simplifier, wa.max, wb.max)? {
                Ordering::Less => (wa.max, wa.max_type.clone()),
                Ordering::Greater => (wb.max, wb.max_type.clone()),
                Ordering::Equal => (
                    wa.max,
                    if wa.max_type == BoundType::Open || wb.max_type == BoundType::Open {
                        BoundType::Open
                    } else {
                        BoundType::Closed
                    },
                ),
            };
            match try_order(simplifier, lo, hi)? {
                Ordering::Less => out.push(Win {
                    min: lo,
                    min_type: lo_type,
                    max: hi,
                    max_type: hi_type,
                }),
                Ordering::Equal => {
                    // Degenerate single point: only when both sides closed.
                    // Points are NOT windows (invariant); handled by the
                    // caller as a Periodic point — for now: skip (design P1:
                    // producers never build such operands; conservative).
                    if lo_type == BoundType::Closed && hi_type == BoundType::Closed {
                        return None;
                    }
                }
                Ordering::Greater => {}
            }
        }
    }
    Some(out)
}

/// Re-glue the first and last result windows across the seam when the seam
/// point is covered: `last.max == anchor + T` touching `first.min == anchor`
/// (they are `T`-translates of one another). The glued window wraps:
/// `(last.min, first.max + T)`.
fn reglue_at_seam(
    simplifier: &mut Simplifier,
    mut wins: Vec<Win>,
    anchor: ExprId,
    period: ExprId,
) -> Option<Vec<Win>> {
    if wins.len() < 2 {
        return Some(wins);
    }
    let seam = shift_by_periods(simplifier, anchor, period, 1);
    let first = wins.first().unwrap().clone();
    let last = wins.last().unwrap().clone();
    let first_at_anchor = try_order(simplifier, first.min, anchor)? == Ordering::Equal;
    let last_at_seam = try_order(simplifier, last.max, seam)? == Ordering::Equal;
    if first_at_anchor
        && last_at_seam
        && (first.min_type == BoundType::Closed || last.max_type == BoundType::Closed)
    {
        let new_max = shift_by_periods(simplifier, first.max, period, 1);
        wins.pop();
        wins.remove(0);
        wins.push(Win {
            min: last.min,
            min_type: last.min_type.clone(),
            max: new_max,
            max_type: first.max_type.clone(),
        });
        try_sort(simplifier, &mut wins)?;
    }
    Some(wins)
}

/// Package normalized windows as a `SolutionSet`, collapsing the full-cover
/// cases: no windows → `Empty`; a single window of length == period →
/// `AllReals` when any endpoint is closed (the seam point is covered),
/// punctured-line `PeriodicIntervalUnion` when both are open.
fn package(simplifier: &mut Simplifier, wins: Vec<Win>, period: ExprId) -> Option<SolutionSet> {
    if wins.is_empty() {
        return Some(SolutionSet::Empty);
    }
    if wins.len() == 1 {
        let w = &wins[0];
        let len = diff(simplifier, w.max, w.min);
        if let Some(ratio) = try_exact_ratio(simplifier, len, period) {
            if ratio == BigRational::from_integer(1.into())
                && (w.min_type == BoundType::Closed || w.max_type == BoundType::Closed)
            {
                return Some(SolutionSet::AllReals);
            }
            // ratio == 1 with both ends open: the punctured line — keep as PIU.
        }
    }
    Some(SolutionSet::PeriodicIntervalUnion {
        windows: wins.iter().map(Win::to_interval).collect(),
        period,
    })
}

/// Periods must be STRUCTURALLY equal (the `Periodic` core precedent) or
/// exactly value-equal.
fn periods_match(simplifier: &mut Simplifier, p1: ExprId, p2: ExprId) -> bool {
    p1 == p2
        || cas_ast::ordering::compare_expr(&simplifier.context, p1, p2) == Ordering::Equal
        || try_order(simplifier, p1, p2) == Some(Ordering::Equal)
}

/// Union of two same-period `PeriodicIntervalUnion` window lists.
/// `None` ⇒ not exactly combinable — the caller must decline honestly.
pub fn union_periodic_interval_unions_over_common_period(
    simplifier: &mut Simplifier,
    windows1: &[Interval],
    period1: ExprId,
    windows2: &[Interval],
    period2: ExprId,
) -> Option<SolutionSet> {
    if !periods_match(simplifier, period1, period2) {
        return None;
    }
    let anchor = windows1.first()?.min;
    let mut all: Vec<Win> = Vec::new();
    for iv in windows1.iter().chain(windows2.iter()) {
        all.extend(normalize_window(
            simplifier,
            &Win::from_interval(iv),
            anchor,
            period1,
        )?);
    }
    try_sort(simplifier, &mut all)?;
    let merged = linear_union(simplifier, all)?;
    let reglued = reglue_at_seam(simplifier, merged, anchor, period1)?;
    package(simplifier, reglued, period1)
}

/// Intersection of two same-period `PeriodicIntervalUnion` window lists.
/// `None` ⇒ not exactly combinable — the caller must decline honestly.
pub fn intersect_periodic_interval_unions_over_common_period(
    simplifier: &mut Simplifier,
    windows1: &[Interval],
    period1: ExprId,
    windows2: &[Interval],
    period2: ExprId,
) -> Option<SolutionSet> {
    if !periods_match(simplifier, period1, period2) {
        return None;
    }
    let anchor = windows1.first()?.min;
    let mut a: Vec<Win> = Vec::new();
    for iv in windows1 {
        a.extend(normalize_window(
            simplifier,
            &Win::from_interval(iv),
            anchor,
            period1,
        )?);
    }
    let mut b: Vec<Win> = Vec::new();
    for iv in windows2 {
        b.extend(normalize_window(
            simplifier,
            &Win::from_interval(iv),
            anchor,
            period1,
        )?);
    }
    try_sort(simplifier, &mut a)?;
    try_sort(simplifier, &mut b)?;
    let clipped = linear_intersection(simplifier, &a, &b)?;
    let merged = linear_union(simplifier, clipped)?;
    let reglued = reglue_at_seam(simplifier, merged, anchor, period1)?;
    package(simplifier, reglued, period1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pi_frac(simplifier: &mut Simplifier, num: i64, den: i64) -> ExprId {
        let n = simplifier.context.num(num);
        let d = simplifier.context.num(den);
        let pi = simplifier
            .context
            .add(Expr::Constant(cas_ast::Constant::Pi));
        let frac = simplifier.context.add(Expr::Div(n, d));
        let e = simplifier.context.add(Expr::Mul(frac, pi));
        simplifier.simplify(e).0
    }

    fn open_window(simplifier: &mut Simplifier, lo: (i64, i64), hi: (i64, i64)) -> Interval {
        Interval {
            min: pi_frac(simplifier, lo.0, lo.1),
            min_type: BoundType::Open,
            max: pi_frac(simplifier, hi.0, hi.1),
            max_type: BoundType::Open,
        }
    }

    fn fmt_set(simplifier: &Simplifier, s: &SolutionSet) -> String {
        match s {
            SolutionSet::PeriodicIntervalUnion { windows, period } => {
                cas_formatter::display_periodic_interval_union(
                    &simplifier.context,
                    windows,
                    *period,
                )
            }
            SolutionSet::AllReals => "AllReals".to_string(),
            SolutionSet::Empty => "Empty".to_string(),
            other => format!("{other:?}"),
        }
    }

    /// Design §6 mandatory unit test — witness #8: `sin(x) > 0 ∩ sin(x) < 1/2`
    /// must keep BOTH components, `(2kπ, π/6+2kπ) ∪ (5π/6+2kπ, π+2kπ)`. A
    /// flat (non-circular) intersection loses the wrapped `(0, π/6)` piece.
    #[test]
    fn witness_8_wrapped_intersection_keeps_both_components() {
        let mut simplifier = Simplifier::with_default_rules();
        let two_pi = pi_frac(&mut simplifier, 2, 1);
        // sin u > 0  → (0, π)
        let w_pos = open_window(&mut simplifier, (0, 1), (1, 1));
        // sin u < 1/2 → (5π/6, 13π/6): straddles the 2π seam by design.
        let w_lt = open_window(&mut simplifier, (5, 6), (13, 6));
        let out = intersect_periodic_interval_unions_over_common_period(
            &mut simplifier,
            &[w_pos],
            two_pi,
            &[w_lt],
            two_pi,
        )
        .expect("π-lattice endpoints must be combinable");
        let SolutionSet::PeriodicIntervalUnion { windows, .. } = &out else {
            panic!(
                "expected a periodic interval union, got {}",
                fmt_set(&simplifier, &out)
            );
        };
        assert_eq!(
            windows.len(),
            2,
            "must keep BOTH components, got {}",
            fmt_set(&simplifier, &out)
        );
        // Raw-built nodes render `2 * pi` where the pipeline shows `2·pi`;
        // normalize the separator before asserting the window structure.
        let rendered = fmt_set(&simplifier, &out).replace(" * ", "·");
        assert!(
            rendered.contains("(k·2·pi, 1/6·pi + k·2·pi)")
                && rendered.contains("(5/6·pi + k·2·pi, pi + k·2·pi)"),
            "windows wrong: {rendered}"
        );
    }

    /// Union re-glues across the seam: `(0, π)` ∪ `(5π/6, 13π/6]` covers the
    /// seam point 2π (closed) so the result is the single wrapped window
    /// `(5π/6, π + 2π)`… clipped by overlap it is `(0·…` — concretely
    /// `(0,π) ∪ (5π/6,13π/6]` = `(0, 13π/6]` whose span 13π/6 exceeds… the
    /// normalized circular union is `(5π/6 − 2π…` — assert the semantic:
    /// single window, length 13π/6−…  We assert the rendered set directly.
    #[test]
    fn union_merges_overlap_across_seam() {
        let mut simplifier = Simplifier::with_default_rules();
        let two_pi = pi_frac(&mut simplifier, 2, 1);
        let w1 = open_window(&mut simplifier, (0, 1), (1, 1)); // (0, π)
        let mut w2 = open_window(&mut simplifier, (5, 6), (13, 6)); // (5π/6, 13π/6]
        w2.max_type = BoundType::Closed;
        let out = union_periodic_interval_unions_over_common_period(
            &mut simplifier,
            &[w1],
            two_pi,
            &[w2],
            two_pi,
        )
        .expect("combinable");
        // (0,π) ∪ (5π/6, 2π) ∪ [0, π/6] (split+glue) = (0, π) ∪ (5π/6, 2π) ∪ [2π, 13π/6]
        // mod 2π ⇒ [0, π) ∪ (5π/6, 2π) with 0 covered ⇒ glue ⇒ (5π/6, π + 2π)?? —
        // semantics: covered points are [0, π/6] ∪ (0, π) ∪ (5π/6, 2π) = (5π/6, 2π) ∪ [0, π)
        // with 0 INCLUDED and 2π excluded… as a wrapped window: (5π/6, π + 2π) is WRONG
        // (π…2π−ε not all covered). Correct glued window: (5π/6, 2π)∪[0,π) = (5π/6, π+2π)
        // ONLY if coverage is contiguous through the seam: (5π/6,2π) ∪ [2π,2π+π) ≡ (5π/6, 3π).
        // Span 3π−5π/6 = 13π/6 > 2π ⇒ NOT a valid window; but Σ len = coverage
        // (5π/6→3π) minus… total covered length = 2π−5π/6 + π = 13π/6 > 2π ⇒ the
        // TRUE union covers everything except… (5π/6,3π) mod 2π covers ℝ? A window of
        // length > period covers all residues ⇒ AllReals.
        assert!(
            matches!(out, SolutionSet::AllReals),
            "coverage length 13π/6 > 2π must collapse to AllReals, got {}",
            fmt_set(&simplifier, &out)
        );
    }

    /// The punctured line stays a PIU: `(0, 2π)` ∪ `(π, 3π)` — two open
    /// windows overlapping through the seam cover everything except… they
    /// jointly cover ALL of ℝ (every residue is interior to one of them):
    /// AllReals. Contrast: `(0,2π)` alone is ℝ∖{2kπ} and must stay PIU.
    #[test]
    fn punctured_line_survives_packaging() {
        let mut simplifier = Simplifier::with_default_rules();
        let two_pi = pi_frac(&mut simplifier, 2, 1);
        let w = open_window(&mut simplifier, (0, 1), (2, 1)); // (0, 2π): ℝ∖{2kπ}
        let out = union_periodic_interval_unions_over_common_period(
            &mut simplifier,
            std::slice::from_ref(&w),
            two_pi,
            std::slice::from_ref(&w),
            two_pi,
        )
        .expect("combinable");
        let SolutionSet::PeriodicIntervalUnion { windows, .. } = &out else {
            panic!(
                "punctured line must stay PIU, got {}",
                fmt_set(&simplifier, &out)
            );
        };
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].min_type, BoundType::Open);
        assert_eq!(windows[0].max_type, BoundType::Open);
    }

    /// Unorderable endpoints (no arcsin oracle yet) must return None — the
    /// conservative deferral, never a blind combination.
    #[test]
    fn unorderable_symbolic_endpoints_defer() {
        let mut simplifier = Simplifier::with_default_rules();
        let two_pi = pi_frac(&mut simplifier, 2, 1);
        let third = {
            let one = simplifier.context.num(1);
            let three = simplifier.context.num(3);
            simplifier.context.add(Expr::Div(one, three))
        };
        let arcsin_third = simplifier.context.call("arcsin", vec![third]);
        let pi_v = pi_frac(&mut simplifier, 1, 1);
        let w_sym = Interval {
            min: arcsin_third,
            min_type: BoundType::Open,
            max: pi_v,
            max_type: BoundType::Open,
        };
        let w_exact = open_window(&mut simplifier, (1, 6), (5, 6));
        let out = intersect_periodic_interval_unions_over_common_period(
            &mut simplifier,
            &[w_sym],
            two_pi,
            &[w_exact],
            two_pi,
        );
        assert!(
            out.is_none(),
            "arcsin(1/3) endpoints are not orderable yet — must defer"
        );
    }
}
