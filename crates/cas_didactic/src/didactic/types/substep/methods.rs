use super::SubStep;

impl SubStep {
    /// Create a plain-text sub-step (no LaTeX).
    /// Text will be wrapped in `\text{}` by the JSON layer.
    pub fn new(
        description: impl Into<String>,
        before_expr: impl Into<String>,
        after_expr: impl Into<String>,
    ) -> Self {
        Self {
            description: description.into(),
            before_expr: before_expr.into(),
            after_expr: after_expr.into(),
            before_latex: None,
            after_latex: None,
        }
    }

    /// Set the LaTeX for `before_expr`.
    pub fn with_before_latex(mut self, latex: impl Into<String>) -> Self {
        self.before_latex = Some(latex.into());
        self
    }

    /// Set the LaTeX for `after_expr`.
    pub fn with_after_latex(mut self, latex: impl Into<String>) -> Self {
        self.after_latex = Some(latex.into());
        self
    }
}
