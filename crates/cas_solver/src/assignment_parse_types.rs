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
