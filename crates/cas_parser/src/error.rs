//! Parse error types with source span support.

use cas_ast::Span;
use thiserror::Error;

/// Parse error with optional source location.
///
/// # Stability Contract
///
/// - `kind()` and `span()` methods are stable API
/// - Error messages may change between versions
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    /// Syntax error at a specific location
    #[error("Parse error at {}: {message}", span.map(|s| format!("{}", s)).unwrap_or_else(|| "unknown".into()))]
    Syntax { message: String, span: Option<Span> },

    /// Unconsumed input after parsing
    #[error("Unconsumed input: {remaining}")]
    UnconsumedInput {
        remaining: String,
        span: Option<Span>,
    },

    /// Unknown/internal parse error
    #[error("Unknown parse error")]
    Unknown,
}

impl ParseError {
    /// Create a syntax error without span (legacy compatibility).
    pub fn syntax(message: impl Into<String>) -> Self {
        ParseError::Syntax {
            message: message.into(),
            span: None,
        }
    }

    /// Create a syntax error with span.
    pub fn syntax_at(message: impl Into<String>, span: Span) -> Self {
        ParseError::Syntax {
            message: message.into(),
            span: Some(span),
        }
    }

    /// Create an unconsumed input error.
    pub fn unconsumed(remaining: impl Into<String>) -> Self {
        ParseError::UnconsumedInput {
            remaining: remaining.into(),
            span: None,
        }
    }

    /// Get the source span if available.
    pub fn span(&self) -> Option<Span> {
        match self {
            ParseError::Syntax { span, .. } => *span,
            ParseError::UnconsumedInput { span, .. } => *span,
            ParseError::Unknown => None,
        }
    }

    /// Get the error message without location info.
    pub fn message(&self) -> &str {
        match self {
            ParseError::Syntax { message, .. } => message,
            ParseError::UnconsumedInput { remaining, .. } => remaining,
            ParseError::Unknown => "Unknown parse error",
        }
    }
}

// Legacy conversion from old variant names (for migration)
impl ParseError {
    /// Convert from legacy NomError format.
    pub fn from_nom_error(msg: String) -> Self {
        ParseError::Syntax {
            message: msg,
            span: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_error_without_span() {
        let err = ParseError::syntax("unexpected token");
        assert_eq!(err.message(), "unexpected token");
        assert!(err.span().is_none());
    }

    #[test]
    fn test_syntax_error_with_span() {
        let err = ParseError::syntax_at("unexpected token", Span::new(5, 10));
        assert_eq!(err.message(), "unexpected token");
        assert_eq!(err.span(), Some(Span::new(5, 10)));
    }

    #[test]
    fn test_error_display() {
        let err = ParseError::syntax_at("bad token", Span::new(3, 7));
        let msg = format!("{}", err);
        assert!(msg.contains("3..7"), "Should contain span: {}", msg);
        assert!(msg.contains("bad token"), "Should contain message: {}", msg);
    }
}
