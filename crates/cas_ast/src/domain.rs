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
        }
    }

    /// Get the expression ID this condition refers to
    pub fn expr_id(&self) -> ExprId {
        match self {
            Self::NonZero(e)
            | Self::Positive(e)
            | Self::NonNegative(e)
            | Self::Defined(e)
            | Self::InvTrigPrincipalRange { arg: e, .. } => *e,
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
}

/// A single case in a conditional solution set.
///
/// Represents "if `when` holds, then solutions are `then`".
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    /// Conditions that must hold for this case
    pub when: ConditionSet,
    /// Solutions when conditions hold
    pub then: Box<SolutionSet>,
}

impl Case {
    /// Create a new case
    pub fn new(when: ConditionSet, then: SolutionSet) -> Self {
        Self {
            when,
            then: Box::new(then),
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
