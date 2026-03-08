//! Shared step-trace primitive types.

/// Path from root to a transformed node.
#[derive(Debug, Clone, PartialEq)]
pub enum PathStep {
    /// Binary op left / Div numerator.
    Left,
    /// Binary op right / Div denominator.
    Right,
    /// Function argument index.
    Arg(usize),
    /// Power base.
    Base,
    /// Power exponent.
    Exponent,
    /// Negation inner / other unary.
    Inner,
}

impl PathStep {
    /// Convert to child index for ExprPath.
    pub fn to_child_index(&self) -> u8 {
        match self {
            PathStep::Left => 0,
            PathStep::Right => 1,
            PathStep::Base => 0,
            PathStep::Exponent => 1,
            PathStep::Inner => 0,
            PathStep::Arg(i) => *i as u8,
        }
    }
}

/// Convert a Vec<PathStep> to ExprPath.
pub fn pathsteps_to_expr_path(steps: &[PathStep]) -> cas_ast::ExprPath {
    steps.iter().map(|s| s.to_child_index()).collect()
}

/// Importance level for step filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportanceLevel {
    Trivial = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

/// Category of step for grouping and filtering by type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepCategory {
    #[default]
    General,
    Canonicalize,
    Simplify,
    Expand,
    Factor,
    Rationalize,
    ConstEval,
    ConstFold,
    Domain,
    Solve,
    Substitute,
    Limits,
}

/// Educational sub-step explaining rule application.
#[derive(Debug, Clone)]
pub struct SubStep {
    /// Title of this sub-step (e.g., "Pattern Recognition").
    pub title: String,
    /// Explanation lines (bullet points).
    pub lines: Vec<String>,
    /// Importance for verbosity filtering.
    pub importance: ImportanceLevel,
}

impl SubStep {
    /// Create a new substep with the given title and lines.
    pub fn new(title: impl Into<String>, lines: Vec<String>) -> Self {
        Self {
            title: title.into(),
            lines,
            importance: ImportanceLevel::Low,
        }
    }

    /// Create a substep with custom importance.
    pub fn with_importance(
        title: impl Into<String>,
        lines: Vec<String>,
        importance: ImportanceLevel,
    ) -> Self {
        Self {
            title: title.into(),
            lines,
            importance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{pathsteps_to_expr_path, PathStep};

    #[test]
    fn pathstep_to_expr_path_mapping() {
        let path = vec![PathStep::Left, PathStep::Arg(2), PathStep::Inner];
        let expr_path = pathsteps_to_expr_path(&path);
        assert_eq!(expr_path.as_slice(), &[0, 2, 0]);
    }
}
