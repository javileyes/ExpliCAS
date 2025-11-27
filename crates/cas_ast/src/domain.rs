use crate::expression::{Expr, Constant};
use std::rc::Rc;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundType {
    Open,   // ( ... )
    Closed, // [ ... ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub min: Rc<Expr>,
    pub min_type: BoundType,
    pub max: Rc<Expr>,
    pub max_type: BoundType,
}

impl Interval {
    pub fn closed(a: Rc<Expr>, b: Rc<Expr>) -> Self {
        Interval {
            min: a, min_type: BoundType::Closed,
            max: b, max_type: BoundType::Closed,
        }
    }

    pub fn open(a: Rc<Expr>, b: Rc<Expr>) -> Self {
        Interval {
            min: a, min_type: BoundType::Open,
            max: b, max_type: BoundType::Open,
        }
    }

    pub fn all_reals() -> Self {
        let inf = Rc::new(Expr::Constant(Constant::Infinity));
        let neg_inf = Rc::new(Expr::Neg(inf.clone()));
        
        Interval {
            min: neg_inf, min_type: BoundType::Open,
            max: inf,     max_type: BoundType::Open,
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let open_bracket = match self.min_type {
            BoundType::Open => "(",
            BoundType::Closed => "[",
        };
        let close_bracket = match self.max_type {
            BoundType::Open => ")",
            BoundType::Closed => "]",
        };
        write!(f, "{}{}, {}{}", open_bracket, self.min, self.max, close_bracket)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolutionSet {
    Discrete(Vec<Rc<Expr>>), 
    Continuous(Interval),
    Union(Vec<Interval>),
    Empty,
    AllReals,
}

impl fmt::Display for SolutionSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SolutionSet::Discrete(exprs) => {
                let s: Vec<String> = exprs.iter().map(|e| format!("{}", e)).collect();
                write!(f, "{{{}}}", s.join(", "))
            },
            SolutionSet::Continuous(interval) => write!(f, "{}", interval),
            SolutionSet::Union(intervals) => {
                let s: Vec<String> = intervals.iter().map(|i| format!("{}", i)).collect();
                write!(f, "{}", s.join(" U "))
            },
            SolutionSet::Empty => write!(f, "Empty Set"),
            SolutionSet::AllReals => write!(f, "All Real Numbers"),
        }
    }
}
