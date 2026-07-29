use crate::expression::ExprId;
use num_rational::BigRational;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundType {
    Open,   // ( ... )
    Closed, // [ ... ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub min: ExprId,
    pub min_type: BoundType,
    pub max: ExprId,
    pub max_type: BoundType,
}

impl Interval {
    /// Reconstruye el intervalo aplicando `f` a sus dos extremos (los tipos de
    /// borde no dependen de la expresión). Ver `SolutionSet::map_exprs`.
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> Interval {
        Interval {
            min: f(self.min),
            min_type: self.min_type.clone(),
            max: f(self.max),
            max_type: self.max_type.clone(),
        }
    }

    pub fn closed(a: ExprId, b: ExprId) -> Self {
        Interval {
            min: a,
            min_type: BoundType::Closed,
            max: b,
            max_type: BoundType::Closed,
        }
    }

    pub fn open(a: ExprId, b: ExprId) -> Self {
        Interval {
            min: a,
            min_type: BoundType::Open,
            max: b,
            max_type: BoundType::Open,
        }
    }
}

// =============================================================================
// V2.0 Phase 1: Conditional Solutions Infrastructure
// =============================================================================

/// A condition predicate for guarded solutions.
///
/// These represent conditions that must hold for a solution to be valid.
/// Used in `SolutionSet::Conditional` to express piecewise solutions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConditionPredicate {
    /// Expression must be non-zero
    NonZero(ExprId),
    /// Expression must be positive (> 0)
    Positive(ExprId),
    /// Expression must be non-negative (≥ 0)
    NonNegative(ExprId),
    /// Expression must be at least a rational lower bound.
    LowerBound { expr: ExprId, lower: BigRational },
    /// Expression must be defined
    Defined(ExprId),
    /// Argument must be in principal range of inverse trig function
    InvTrigPrincipalRange { func: &'static str, arg: ExprId },
    /// Result uses the principal branch of a multivalued complex function
    /// (Arg ∈ (-π, π]); `func` names it ("ln", "arg", "pow").
    PrincipalBranch { func: &'static str, arg: ExprId },
    /// V2.0 Phase 2B: Expression equals zero
    EqZero(ExprId),
    /// V2.0 Phase 2B: Expression equals one
    EqOne(ExprId),
}

impl ConditionPredicate {
    /// Get a human-readable description of this condition
    pub fn display(&self) -> String {
        match self {
            Self::NonZero(_) => "≠ 0".to_string(),
            Self::Positive(_) => "> 0".to_string(),
            Self::NonNegative(_) => "≥ 0".to_string(),
            Self::LowerBound { lower, .. } => format!("≥ {}", lower),
            Self::Defined(_) => "is defined".to_string(),
            Self::InvTrigPrincipalRange { func, .. } => {
                format!("in principal range of {}", func)
            }
            Self::PrincipalBranch { func, .. } => {
                format!("via principal branch of {}", func)
            }
            Self::EqZero(_) => "= 0".to_string(),
            Self::EqOne(_) => "= 1".to_string(),
        }
    }

    /// Get the expression ID this condition refers to
    pub fn expr_id(&self) -> ExprId {
        match self {
            Self::NonZero(e)
            | Self::Positive(e)
            | Self::NonNegative(e)
            | Self::LowerBound { expr: e, .. }
            | Self::Defined(e)
            | Self::InvTrigPrincipalRange { arg: e, .. }
            | Self::PrincipalBranch { arg: e, .. }
            | Self::EqZero(e)
            | Self::EqOne(e) => *e,
        }
    }
}

/// A set of conditions (conjunction).
///
/// Represents "condition1 AND condition2 AND ...".
/// Predicates are sorted and deduplicated for stable comparison.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConditionSet {
    predicates: Vec<ConditionPredicate>,
}

impl ConditionSet {
    /// Create an empty condition set (always true)
    pub fn empty() -> Self {
        Self {
            predicates: Vec::new(),
        }
    }

    /// Create a condition set with a single predicate
    pub fn single(pred: ConditionPredicate) -> Self {
        Self {
            predicates: vec![pred],
        }
    }

    /// Create from multiple predicates (deduplicates)
    pub fn from_predicates(preds: Vec<ConditionPredicate>) -> Self {
        let mut predicates = preds;
        // Sort by debug string for stable ordering
        predicates.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        predicates.dedup();
        Self { predicates }
    }

    /// Check if this is an empty (always true) condition
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Get the predicates
    pub fn predicates(&self) -> &[ConditionPredicate] {
        &self.predicates
    }

    /// Add a predicate to this set
    pub fn push(&mut self, pred: ConditionPredicate) {
        if !self.predicates.contains(&pred) {
            self.predicates.push(pred);
            self.predicates
                .sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        }
    }

    /// V2.0 Phase 2D: Check if this is an "otherwise" (empty) condition
    pub fn is_otherwise(&self) -> bool {
        self.predicates.is_empty()
    }

    /// V2.0 Phase 2D: Combine two condition sets (conjunction/AND)
    /// Used for flattening nested Conditionals
    pub fn and(&self, other: &ConditionSet) -> ConditionSet {
        let mut combined = self.predicates.clone();
        for pred in &other.predicates {
            if !combined.contains(pred) {
                combined.push(pred.clone());
            }
        }
        // Sort for stable ordering
        combined.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        ConditionSet {
            predicates: combined,
        }
    }

    /// V2.0 Phase 3A: Simplify redundant predicates
    ///
    /// Rules:
    /// - Positive(x) implies NonZero(x) → remove NonZero
    /// - EqOne(x) implies Positive(x) → remove Positive
    /// - EqOne(x) implies NonNegative(x) → remove NonNegative
    /// - NonNegative(x) implies Defined(x) (for sqrt) → remove Defined
    /// - Duplicate predicates removed (already handled by dedup)
    pub fn simplify(&self) -> ConditionSet {
        let mut result: Vec<ConditionPredicate> = Vec::new();

        for pred in &self.predicates {
            let dominated = self.predicates.iter().any(|other| {
                if pred == other {
                    return false; // Don't compare with self
                }
                // Check if 'other' implies 'pred' (making pred redundant)
                match (other, pred) {
                    // Positive(x) implies NonZero(x)
                    (ConditionPredicate::Positive(e1), ConditionPredicate::NonZero(e2))
                        if e1 == e2 =>
                    {
                        true
                    }
                    // Positive(x) implies NonNegative(x)
                    (ConditionPredicate::Positive(e1), ConditionPredicate::NonNegative(e2))
                        if e1 == e2 =>
                    {
                        true
                    }
                    // EqOne(x) implies NonZero(x)
                    (ConditionPredicate::EqOne(e1), ConditionPredicate::NonZero(e2))
                        if e1 == e2 =>
                    {
                        true
                    }
                    // EqOne(x) implies Positive(x)
                    (ConditionPredicate::EqOne(e1), ConditionPredicate::Positive(e2))
                        if e1 == e2 =>
                    {
                        true
                    }
                    // EqOne(x) implies NonNegative(x)
                    (ConditionPredicate::EqOne(e1), ConditionPredicate::NonNegative(e2))
                        if e1 == e2 =>
                    {
                        true
                    }
                    _ => false,
                }
            });

            if !dominated {
                result.push(pred.clone());
            }
        }

        ConditionSet::from_predicates(result)
    }

    /// V2.0 Phase 3B: Check if this condition set is a contradiction (impossible)
    ///
    /// Returns true if predicates are mutually exclusive.
    pub fn is_contradiction(&self) -> bool {
        for (i, p1) in self.predicates.iter().enumerate() {
            for p2 in self.predicates.iter().skip(i + 1) {
                if Self::predicates_contradict(p1, p2) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if two predicates contradict each other
    fn predicates_contradict(p1: &ConditionPredicate, p2: &ConditionPredicate) -> bool {
        match (p1, p2) {
            // EqZero(x) contradicts NonZero(x)
            (ConditionPredicate::EqZero(e1), ConditionPredicate::NonZero(e2))
            | (ConditionPredicate::NonZero(e1), ConditionPredicate::EqZero(e2))
                if e1 == e2 =>
            {
                true
            }
            // EqZero(x) contradicts Positive(x)
            (ConditionPredicate::EqZero(e1), ConditionPredicate::Positive(e2))
            | (ConditionPredicate::Positive(e1), ConditionPredicate::EqZero(e2))
                if e1 == e2 =>
            {
                true
            }
            // EqZero(x) contradicts EqOne(x)
            (ConditionPredicate::EqZero(e1), ConditionPredicate::EqOne(e2))
            | (ConditionPredicate::EqOne(e1), ConditionPredicate::EqZero(e2))
                if e1 == e2 =>
            {
                true
            }
            _ => false,
        }
    }
}

/// V2.0: Complete result of a solve operation.
///
/// Contains both the solutions found and any unsolved residual.
/// This structure allows branches in conditional solutions to be partially resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    /// The solutions found (may be Conditional for piecewise solutions)
    pub solutions: SolutionSet,
    /// Unsolved portion, if any (a solve(...) expression)
    pub residual: Option<ExprId>,
}

impl SolveResult {
    /// Create a result with solutions only (fully resolved)
    pub fn solved(solutions: SolutionSet) -> Self {
        Self {
            solutions,
            residual: None,
        }
    }

    /// Create a result with only residual (nothing solved)
    pub fn unsolved(residual: ExprId) -> Self {
        Self {
            solutions: SolutionSet::Empty,
            residual: Some(residual),
        }
    }

    /// Create a result with both solutions and residual (partially solved)
    pub fn partial(solutions: SolutionSet, residual: ExprId) -> Self {
        Self {
            solutions,
            residual: Some(residual),
        }
    }

    /// Check if this result has any actual solutions
    pub fn has_solutions(&self) -> bool {
        !matches!(self.solutions, SolutionSet::Empty)
    }

    /// V2.0 Phase 2D: Flatten nested Conditional solutions
    pub fn flatten(self) -> Self {
        Self {
            solutions: self.solutions.flatten(),
            residual: self.residual,
        }
    }
}

/// A single case in a conditional solution set.
///
/// Represents "if `when` holds, then `then` is the result".
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    /// Conditions that must hold for this case
    pub when: ConditionSet,
    /// Complete solve result when conditions hold (may include residual)
    pub then: Box<SolveResult>,
}

impl Case {
    /// Create a new case with fully resolved solutions
    pub fn new(when: ConditionSet, solutions: SolutionSet) -> Self {
        Self {
            when,
            then: Box::new(SolveResult::solved(solutions)),
        }
    }

    /// Create a new case with a complete SolveResult
    pub fn with_result(when: ConditionSet, result: SolveResult) -> Self {
        Self {
            when,
            then: Box::new(result),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolutionSet {
    Discrete(Vec<ExprId>),
    Continuous(Interval),
    Union(Vec<Interval>),
    Empty,
    AllReals,
    /// A residual expression when the equation cannot be solved in the current domain.
    /// The ExprId represents a `solve(eq, var)` expression that was not fully resolved.
    /// Used in wildcard mode when complex logarithm is needed but we're in RealOnly.
    Residual(ExprId),
    /// V2.0: Conditional/piecewise solutions.
    /// Each case represents "if conditions hold, then these solutions".
    Conditional(Vec<Case>),
    /// One or more periodic families of solutions `baseᵢ + k·period` over every integer `k`
    /// (`sin(x)=0 → {k·π}`; `cos(x)=0 → {π/2 + k·π}`; `tan(x)=c → {arctan(c) + k·π}`;
    /// `sin(x)=1/2 → {π/6 + 2kπ, 5π/6 + 2kπ}` — TWO bases). `bases` are the representative roots in
    /// one period; `period` is the shared (positive) fundamental period. The infinite family that
    /// `Discrete` cannot represent and that the principal-root-only trig equation path dropped.
    Periodic {
        bases: Vec<ExprId>,
        period: ExprId,
    },
    /// A periodic union of INTERVALS `∪_{k∈ℤ} ∪_i (windows[i] + k·period)` —
    /// the solution shape of interior trig inequalities (`sin(x) > 1/2 →
    /// (π/6 + 2kπ, 5π/6 + 2kπ)`), which `Periodic` (points only) cannot
    /// represent. Membership: `x ∈ S ⟺ ∃k∈ℤ: x − k·period ∈ windows[i]`
    /// for some `i`.
    ///
    /// Invariants (see `docs/DESIGN_PERIODIC_INTERVAL_UNION.md` §2):
    /// - windows sorted by `min`, non-degenerate (`min < max` by VALUE),
    ///   finite endpoints, pairwise disjoint and non-adjacent MODULO the
    ///   period;
    /// - SPAN invariant: `max(last) − min(first) ≤ period` — all windows fit
    ///   in ONE length-period translate (windows may straddle any fixed
    ///   fundamental domain, e.g. `(−π/3, π/3)`, but never overlap their own
    ///   `+k·period` translates);
    /// - `Σ len ≤ period`, with equality exactly for the single all-open
    ///   window of length == period (the punctured line, `cos(x) < 1 →
    ///   (2kπ, 2π + 2kπ) = ℝ ∖ {2kπ}`). Length == period with any closed
    ///   endpoint denotes ℝ and must be emitted as `AllReals`, never as this
    ///   variant.
    PeriodicIntervalUnion {
        windows: Vec<Interval>,
        period: ExprId,
    },
}

impl Case {
    /// Ver `SolutionSet::map_exprs`: mapea tanto las condiciones del caso como su
    /// resultado (que puede anidar otro conjunto y su residual).
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> Case {
        Case {
            when: self.when.map_exprs(f),
            then: Box::new(self.then.map_exprs(f)),
        }
    }
}

impl SolveResult {
    /// Ver `SolutionSet::map_exprs`.
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> SolveResult {
        SolveResult {
            solutions: self.solutions.map_exprs(f),
            residual: self.residual.map(&mut *f),
        }
    }
}

impl ConditionSet {
    /// Ver `SolutionSet::map_exprs`.
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> ConditionSet {
        ConditionSet::from_predicates(self.predicates.iter().map(|p| p.map_exprs(f)).collect())
    }
}

impl ConditionPredicate {
    /// Ver `SolutionSet::map_exprs`.
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> ConditionPredicate {
        match self {
            ConditionPredicate::NonZero(e) => ConditionPredicate::NonZero(f(*e)),
            ConditionPredicate::Positive(e) => ConditionPredicate::Positive(f(*e)),
            ConditionPredicate::NonNegative(e) => ConditionPredicate::NonNegative(f(*e)),
            ConditionPredicate::LowerBound { expr, lower } => ConditionPredicate::LowerBound {
                expr: f(*expr),
                lower: lower.clone(),
            },
            ConditionPredicate::Defined(e) => ConditionPredicate::Defined(f(*e)),
            ConditionPredicate::InvTrigPrincipalRange { func, arg } => {
                ConditionPredicate::InvTrigPrincipalRange { func, arg: f(*arg) }
            }
            ConditionPredicate::PrincipalBranch { func, arg } => {
                ConditionPredicate::PrincipalBranch { func, arg: f(*arg) }
            }
            ConditionPredicate::EqZero(e) => ConditionPredicate::EqZero(f(*e)),
            ConditionPredicate::EqOne(e) => ConditionPredicate::EqOne(f(*e)),
        }
    }
}

impl SolutionSet {
    /// Reconstruye el conjunto aplicando `f` a CADA expresión que contiene
    /// (soluciones, extremos de intervalo, periodos, residuales y los argumentos
    /// de las condiciones de cada caso).
    ///
    /// Existe para las transformaciones de PRESENTACIÓN sobre un contexto de
    /// scratch — reescribir potencias fraccionarias a raíces, por ejemplo — donde
    /// hay que tocar todos los `ExprId` sin re-derivar la estructura. Vive junto al
    /// tipo, y no en el renderizador, precisamente para que el compilador obligue a
    /// cubrir cualquier variante que se añada: una que se olvide aquí se imprimiría
    /// con una notación distinta a la de sus hermanas, que es el defecto que este
    /// mapeo existe para evitar.
    pub fn map_exprs(&self, f: &mut impl FnMut(ExprId) -> ExprId) -> SolutionSet {
        match self {
            SolutionSet::Discrete(exprs) => {
                SolutionSet::Discrete(exprs.iter().map(|e| f(*e)).collect())
            }
            SolutionSet::Continuous(interval) => SolutionSet::Continuous(interval.map_exprs(f)),
            SolutionSet::Union(intervals) => {
                SolutionSet::Union(intervals.iter().map(|i| i.map_exprs(f)).collect())
            }
            SolutionSet::Empty => SolutionSet::Empty,
            SolutionSet::AllReals => SolutionSet::AllReals,
            SolutionSet::Residual(expr) => SolutionSet::Residual(f(*expr)),
            SolutionSet::Conditional(cases) => {
                SolutionSet::Conditional(cases.iter().map(|c| c.map_exprs(f)).collect())
            }
            SolutionSet::Periodic { bases, period } => SolutionSet::Periodic {
                bases: bases.iter().map(|e| f(*e)).collect(),
                period: f(*period),
            },
            SolutionSet::PeriodicIntervalUnion { windows, period } => {
                SolutionSet::PeriodicIntervalUnion {
                    windows: windows.iter().map(|w| w.map_exprs(f)).collect(),
                    period: f(*period),
                }
            }
        }
    }

    /// V2.0 Phase 2D: Flatten nested Conditional solutions
    ///
    /// Transforms nested structures like:
    ///   if G1 -> (if G2 -> S else R) else T
    /// into flat structure:
    ///   if (G1 ∧ G2) -> S
    ///   if (G1 ∧ G3) -> R
    ///   if otherwise -> T
    pub fn flatten(self) -> SolutionSet {
        match self {
            SolutionSet::Conditional(cases) => {
                let mut out: Vec<Case> = Vec::new();

                for case in cases {
                    // First, flatten the inner result
                    let flat_result = case.then.flatten();

                    // Check if the inner solutions are themselves Conditional
                    if let SolutionSet::Conditional(inner_cases) = flat_result.solutions {
                        // Expand: combine outer guard with each inner guard
                        for inner_case in inner_cases {
                            let combined_guard = case.when.and(&inner_case.when);
                            out.push(Case {
                                when: combined_guard,
                                then: inner_case.then,
                            });
                        }
                    } else {
                        // Not a nested Conditional, keep as-is (with flattened inner result)
                        out.push(Case {
                            when: case.when,
                            then: Box::new(flat_result),
                        });
                    }
                }

                // Sort: put "otherwise" cases last for better UX
                out.sort_by(
                    |a, b| match (a.when.is_otherwise(), b.when.is_otherwise()) {
                        (true, false) => std::cmp::Ordering::Greater,
                        (false, true) => std::cmp::Ordering::Less,
                        _ => std::cmp::Ordering::Equal,
                    },
                );

                SolutionSet::Conditional(out)
            }
            // Non-Conditional variants pass through unchanged
            other => other,
        }
    }

    /// V2.0 Phase 3: Full simplification pipeline
    ///
    /// Combines flatten() + simplify_cases()
    pub fn simplify(self) -> SolutionSet {
        self.flatten().simplify_cases()
    }

    /// V2.0 Phase 3B/C: Simplify cases in a Conditional
    ///
    /// - 3B: Filters out cases with contradictory guards
    /// - 3C: Simplifies each guard, removes duplicates
    fn simplify_cases(self) -> SolutionSet {
        match self {
            SolutionSet::Conditional(cases) => {
                let mut out: Vec<Case> = Vec::new();

                for case in cases {
                    // Simplify the guard (Phase 3A)
                    let simplified_when = case.when.simplify();

                    // Check for contradictions (Phase 3B)
                    if simplified_when.is_contradiction() {
                        continue; // Skip impossible cases
                    }

                    // Recursively simplify the inner result
                    let simplified_then = Box::new(SolveResult {
                        solutions: case.then.solutions.simplify_cases(),
                        residual: case.then.residual,
                    });

                    // Check if we already have a case with identical result (Phase 3C)
                    let duplicate = out.iter().any(|existing| {
                        existing.when == simplified_when && existing.then == simplified_then
                    });

                    if !duplicate {
                        out.push(Case {
                            when: simplified_when,
                            then: simplified_then,
                        });
                    }
                }

                // Sort: put "otherwise" cases last
                out.sort_by(
                    |a, b| match (a.when.is_otherwise(), b.when.is_otherwise()) {
                        (true, false) => std::cmp::Ordering::Greater,
                        (false, true) => std::cmp::Ordering::Less,
                        _ => std::cmp::Ordering::Equal,
                    },
                );

                // If only one case with otherwise, unwrap to plain solution
                if out.len() == 1 && out[0].when.is_otherwise() {
                    return out.remove(0).then.solutions;
                }

                SolutionSet::Conditional(out)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod map_exprs_tests {
    use super::*;
    use crate::expression::Context;

    /// `map_exprs` debe alcanzar TODA expresión del conjunto. El test recorre las
    /// nueve variantes con un contador: si una variante nueva se olvida de mapear
    /// algún hijo, ese hijo se imprimiría con otra notación que sus hermanos — el
    /// defecto exacto que este mapeo existe para evitar.
    fn count_visits(set: &SolutionSet) -> usize {
        let mut seen = 0usize;
        set.map_exprs(&mut |id| {
            seen += 1;
            id
        });
        seen
    }

    #[test]
    fn map_exprs_reaches_every_expression_of_every_variant() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);
        let interval = Interval::closed(a, b);

        assert_eq!(count_visits(&SolutionSet::Discrete(vec![a, b, c])), 3);
        assert_eq!(count_visits(&SolutionSet::Continuous(interval.clone())), 2);
        assert_eq!(
            count_visits(&SolutionSet::Union(vec![
                interval.clone(),
                interval.clone()
            ])),
            4
        );
        assert_eq!(count_visits(&SolutionSet::Empty), 0);
        assert_eq!(count_visits(&SolutionSet::AllReals), 0);
        assert_eq!(count_visits(&SolutionSet::Residual(a)), 1);
        assert_eq!(
            count_visits(&SolutionSet::Periodic {
                bases: vec![a, b],
                period: c,
            }),
            3
        );
        assert_eq!(
            count_visits(&SolutionSet::PeriodicIntervalUnion {
                windows: vec![interval.clone()],
                period: c,
            }),
            3
        );

        // Conditional anida condiciones + un SolveResult con su propio residual.
        let case = Case {
            when: ConditionSet::from_predicates(vec![
                ConditionPredicate::Positive(a),
                ConditionPredicate::NonZero(b),
            ]),
            then: Box::new(SolveResult {
                solutions: SolutionSet::Discrete(vec![c]),
                residual: Some(a),
            }),
        };
        assert_eq!(count_visits(&SolutionSet::Conditional(vec![case])), 4);
    }

    #[test]
    fn map_exprs_rebuilds_with_the_mapped_ids() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let set = SolutionSet::Discrete(vec![one]);
        let mapped = set.map_exprs(&mut |_| two);
        assert_eq!(mapped, SolutionSet::Discrete(vec![two]));
        // Y la identidad no reconstruye a otra cosa.
        assert_eq!(set.map_exprs(&mut |id| id), set);
    }
}
