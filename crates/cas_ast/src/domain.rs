use crate::expression::ExprId;

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
    /// Expression must be defined
    Defined(ExprId),
    /// Argument must be in principal range of inverse trig function
    InvTrigPrincipalRange { func: &'static str, arg: ExprId },
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
            Self::Defined(_) => "is defined".to_string(),
            Self::InvTrigPrincipalRange { func, .. } => {
                format!("in principal range of {}", func)
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
            | Self::Defined(e)
            | Self::InvTrigPrincipalRange { arg: e, .. }
            | Self::EqZero(e)
            | Self::EqOne(e) => *e,
        }
    }

    /// V2.0 Phase 2C: Get a pretty-printed condition with expression context
    /// Renders like "a = 1", "y > 0", etc.
    pub fn display_with_context(&self, ctx: &crate::Context) -> String {
        let expr_str = crate::DisplayExpr {
            context: ctx,
            id: self.expr_id(),
        }
        .to_string();

        match self {
            Self::NonZero(_) => format!("{} ≠ 0", expr_str),
            Self::Positive(_) => format!("{} > 0", expr_str),
            Self::NonNegative(_) => format!("{} ≥ 0", expr_str),
            Self::Defined(_) => format!("defined({})", expr_str),
            Self::InvTrigPrincipalRange { func, .. } => {
                format!("{} in principal range of {}", expr_str, func)
            }
            Self::EqZero(_) => format!("{} = 0", expr_str),
            Self::EqOne(_) => format!("{} = 1", expr_str),
        }
    }

    /// V2.0 Phase 2C: LaTeX display for timeline rendering
    /// Uses LaTeX math symbols for proper rendering
    pub fn latex_display_with_context(&self, ctx: &crate::Context) -> String {
        let expr_latex = crate::LaTeXExpr {
            context: ctx,
            id: self.expr_id(),
        }
        .to_latex();

        match self {
            Self::NonZero(_) => format!("{} \\neq 0", expr_latex),
            Self::Positive(_) => format!("{} > 0", expr_latex),
            Self::NonNegative(_) => format!("{} \\geq 0", expr_latex),
            Self::Defined(_) => format!("\\text{{defined}}({})", expr_latex),
            Self::InvTrigPrincipalRange { func, .. } => {
                format!("{} \\in \\text{{principal range of }}{}", expr_latex, func)
            }
            Self::EqZero(_) => format!("{} = 0", expr_latex),
            Self::EqOne(_) => format!("{} = 1", expr_latex),
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

    /// V2.0 Phase 2C: Display the condition set with context
    /// Returns "otherwise" for empty set, or comma-separated conditions
    pub fn display_with_context(&self, ctx: &crate::Context) -> String {
        if self.predicates.is_empty() {
            "otherwise".to_string()
        } else {
            self.predicates
                .iter()
                .map(|p| p.display_with_context(ctx))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }

    /// V2.0 Phase 2C: LaTeX display for timeline rendering
    pub fn latex_display_with_context(&self, ctx: &crate::Context) -> String {
        if self.predicates.is_empty() {
            "\\text{otherwise}".to_string()
        } else {
            self.predicates
                .iter()
                .map(|p| p.latex_display_with_context(ctx))
                .collect::<Vec<_>>()
                .join(" \\land ")
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
}

impl SolutionSet {
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
}
