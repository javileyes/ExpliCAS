use cas_ast::ExprId;

#[derive(Debug, Clone, PartialEq)]
pub enum PathStep {
    Left,       // Binary op left / Div numerator
    Right,      // Binary op right / Div denominator
    Arg(usize), // Function argument index
    Base,       // Power base
    Exponent,   // Power exponent
    Inner,      // Negation inner / other unary
}

/// Importance level for step filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportanceLevel {
    Trivial = 0, // x + 0 → x, x * 1 → x
    Low = 1,     // Combine constants, simple evaluations
    Medium = 2,  // Most algebraic transformations
    High = 3,    // Factor, expand, integrate, complex transforms
}

#[derive(Debug, Clone)]
pub struct Step {
    pub description: String,
    pub rule_name: String,
    /// The local expression before transformation (the subexpression that was rewritten)
    pub before: ExprId,
    /// The local expression after transformation
    pub after: ExprId,
    /// Path from root to the transformed node (kept for debugging/reference)
    pub path: Vec<PathStep>,
    /// String representation of after (for display)
    pub after_str: Option<String>,
    /// Complete root expression BEFORE this step's transformation
    pub global_before: Option<ExprId>,
    /// Complete root expression AFTER this step's transformation
    pub global_after: Option<ExprId>,
}

impl Step {
    pub fn new(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: Vec<PathStep>,
        context: Option<&cas_ast::Context>,
    ) -> Self {
        let after_str = context.map(|ctx| {
            format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: ctx,
                    id: after
                }
            )
        });
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
            path,
            after_str,
            global_before: None,
            global_after: None,
        }
    }

    /// Create a step with complete global snapshots before and after transformation
    pub fn with_snapshots(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: Vec<PathStep>,
        context: Option<&cas_ast::Context>,
        global_before: ExprId,
        global_after: ExprId,
    ) -> Self {
        let mut step = Self::new(description, rule_name, before, after, path, context);
        step.global_before = Some(global_before);
        step.global_after = Some(global_after);
        step
    }

    /// Classify the importance/significance of this step
    pub fn importance(&self) -> ImportanceLevel {
        // Trivial steps - identity operations
        if self.rule_name.contains("Add Zero")
            || self.rule_name.contains("Mul By One")
            || self.rule_name.contains("Sub Zero")
            || self.rule_name.contains("Identity Property")
        {
            return ImportanceLevel::Trivial;
        }

        // Low importance - simple evaluations and canonicalizations
        if self.rule_name.contains("Combine Constants")
            || self.rule_name.contains("Evaluate")
            || self.rule_name.contains("Canonicalize")
            || self.rule_name.contains("Sort")
            || self.rule_name.contains("Identity Power")
        {
            return ImportanceLevel::Low;
        }

        // High importance - major transformations
        if self.rule_name.contains("Factor")
            || self.rule_name.contains("Expand")
            || self.rule_name.contains("Integrate")
            || self.rule_name.contains("Differentiate")
            || self.rule_name.contains("Simplify Fraction")
            || self.rule_name.contains("Collect")
        {
            return ImportanceLevel::High;
        }

        // Default: Medium importance
        ImportanceLevel::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_classification() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");

        // Trivial
        let step = Step::new("x + 0 = x", "Add Zero", x, x, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Trivial);

        // Low
        let step = Step::new("2 + 3 = 5", "Combine Constants", x, x, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Low);

        // High
        let step = Step::new("Factor polynomial", "Factor", x, x, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::High);

        // Medium (default)
        let step = Step::new("Some transform", "Unknown Rule", x, x, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Medium);
    }
}
