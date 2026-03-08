mod methods;

/// A synthetic sub-step that explains a hidden operation
#[derive(Debug, Clone)]
pub struct SubStep {
    /// Human-readable description of the operation
    pub description: String,
    /// Expression before the operation (plain text for CLI display)
    pub before_expr: String,
    /// Expression after the operation (plain text for CLI display)
    pub after_expr: String,
    /// Optional LaTeX for `before_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `before_expr`.
    pub before_latex: Option<String>,
    /// Optional LaTeX for `after_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `after_expr`.
    pub after_latex: Option<String>,
}
