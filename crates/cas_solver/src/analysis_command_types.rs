use cas_ast::ExprId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualizeEvalError {
    Parse(String),
}

#[derive(Debug, Clone)]
pub struct ExplainGcdEvalOutput {
    pub steps: Vec<String>,
    pub value: Option<ExprId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainCommandEvalError {
    Parse(String),
    ExpectedFunctionCall,
    UnsupportedFunction(String),
    InvalidArity {
        function: String,
        expected: usize,
        found: usize,
    },
}

/// Output payload for `visualize` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualizeCommandOutput {
    pub file_name: String,
    pub dot_source: String,
    pub hint_lines: Vec<String>,
}
