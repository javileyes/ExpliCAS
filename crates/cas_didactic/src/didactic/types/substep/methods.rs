use super::SubStep;
use cas_solver_core::eval_option_axes::Language;

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
            desc_key: None,
            desc_args: Vec::new(),
        }
    }

    /// Create a sub-step whose title is LOCALIZED through the i18n `key` (see
    /// `crate::didactic::locale`). `args` are positional, already-rendered values (math, numbers)
    /// substituted into the template's `{0}`, `{1}`, …. The Spanish rendering is materialized into
    /// `description` so existing Spanish-only readers keep working; the wire layer re-renders the
    /// title in the requested language from `desc_key` + `desc_args`.
    pub fn keyed(
        key: &'static str,
        args: Vec<String>,
        before_expr: impl Into<String>,
        after_expr: impl Into<String>,
    ) -> Self {
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let description = crate::didactic::locale::translate(key, &arg_refs, Language::Es);
        Self {
            description,
            before_expr: before_expr.into(),
            after_expr: after_expr.into(),
            before_latex: None,
            after_latex: None,
            desc_key: Some(key),
            desc_args: args,
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
