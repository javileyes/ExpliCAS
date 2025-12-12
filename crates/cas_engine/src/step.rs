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
    /// Optional: The specific pattern matched (for n-ary rules like atan(x)+atan(1/x))
    /// Use this for "Rule: X -> Y" display if set, otherwise use before/after
    pub before_local: Option<ExprId>,
    /// Optional: The specific result of the pattern (for n-ary rules)
    pub after_local: Option<ExprId>,
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
            before_local: None,
            after_local: None,
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
    /// This is the SINGLE SOURCE OF TRUTH for step filtering in both CLI and timeline
    pub fn importance(&self) -> ImportanceLevel {
        // No-op steps are always trivial (before == after means no visible change)
        if self.before == self.after {
            return ImportanceLevel::Trivial;
        }

        // Trivial steps - identity operations that don't teach anything
        if self.rule_name.contains("Add Zero")
            || self.rule_name.contains("Mul By One")
            || self.rule_name.contains("Sub Zero")
            || self.rule_name.contains("Identity Property")
        {
            return ImportanceLevel::Trivial;
        }

        // EXCEPTION: Evaluate Numeric Power is didactically important
        // e.g., sqrt(12) → 2*√3 shows root simplification which students should see
        if self.rule_name == "Evaluate Numeric Power" {
            return ImportanceLevel::Medium;
        }

        // Low importance - internal reorganizations, not pedagogically valuable
        if self.rule_name.contains("Combine Constants")
            || self.rule_name.contains("Evaluate")
            || self.rule_name.contains("Canonicalize")
            || self.rule_name.contains("Sort")
            || self.rule_name.contains("Identity Power")
            || self.rule_name.contains("Collect")  // Moved from High - internal grouping
            || self.rule_name.starts_with("Identity")
        // All identity rules
        {
            return ImportanceLevel::Low;
        }

        // High importance - major transformations students should see
        if self.rule_name.contains("Factor")
            || self.rule_name.contains("Expand")
            || self.rule_name.contains("Integrate")
            || self.rule_name.contains("Differentiate")
            || self.rule_name.contains("Simplify Fraction")
        {
            return ImportanceLevel::High;
        }

        // Default: Medium importance - most algebraic steps
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
        let y = ctx.var("y");

        // Trivial (identity rule name takes precedence even with different before/after)
        let step = Step::new("x + 0 = x", "Add Zero", x, y, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Trivial);

        // Low (needs different before/after to avoid no-op shortcut)
        let step = Step::new("2 + 3 = 5", "Combine Constants", x, y, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Low);

        // High
        let step = Step::new("Factor polynomial", "Factor", x, y, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::High);

        // Medium (default)
        let step = Step::new("Some transform", "Unknown Rule", x, y, vec![], Some(&ctx));
        assert_eq!(step.importance(), ImportanceLevel::Medium);
    }
}
