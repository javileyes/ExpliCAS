use cas_ast::ExprId;

pub type SubstituteEvalOutput = SubstituteSimplifyEvalOutput;

/// Render policy for substitute command step output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteRenderMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Parse/eval errors for `subst <expr>, <target>, <replacement>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteParseError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}

/// Evaluated payload for REPL-style `subst` followed by simplify.
#[derive(Debug, Clone)]
pub struct SubstituteSimplifyEvalOutput {
    pub simplified_expr: ExprId,
    pub strategy: crate::SubstituteStrategy,
    pub steps: Vec<crate::Step>,
}
