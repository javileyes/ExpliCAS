use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Parse error: {0}")]
    NomError(String),
    #[error("Unconsumed input: {0}")]
    UnconsumedInput(String),
    #[error("Unknown error")]
    Unknown,
}
