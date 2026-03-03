/// Errors returned when applying a `let` assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentError {
    EmptyName,
    InvalidNameStart,
    ReservedName(String),
    Parse(String),
}

/// Parsed form of `let` command assignment input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedLetAssignment<'a> {
    pub name: &'a str,
    pub expr: &'a str,
    pub lazy: bool,
}

/// Error while parsing `let` command assignment syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LetAssignmentParseError {
    MissingAssignmentOperator,
}
