use thiserror::Error;

#[derive(Error, Debug)]
pub enum CasError {
    #[error("Variable '{0}' not found")]
    VariableNotFound(String),
    #[error("Cannot isolate '{0}': {1}")]
    IsolationError(String, String),
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Solver error: {0}")]
    SolverError(String),
    #[error("Parse error: {0}")]
    ParseError(String), // Wrap ParseError if needed, or just string for now
    #[error("Polynomial error: {0}")]
    PolynomialError(String),
}
