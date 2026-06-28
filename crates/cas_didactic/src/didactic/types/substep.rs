#[path = "substep/methods.rs"]
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
    /// Optional i18n key for the title. When set, the wire layer renders the title from the locale
    /// table for the requested `Language` (filling `desc_args` into `{0}`, `{1}`, …); when `None`,
    /// `description` (built in Spanish) is used verbatim. This localizes a sub-step title without
    /// threading the language into every generator.
    pub desc_key: Option<&'static str>,
    /// Positional arguments (already-rendered, language-neutral, e.g. math) for `desc_key`.
    pub desc_args: Vec<String>,
}
