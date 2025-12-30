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
}
