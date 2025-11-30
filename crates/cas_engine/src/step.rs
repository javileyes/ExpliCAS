use cas_ast::ExprId;

#[derive(Debug, Clone, PartialEq)]
pub enum PathStep {
    Left,       // Binary op left / Div numerator
    Right,      // Binary op right / Div denominator
    Arg(usize), // Function argument index
    Base,       // Power base
    Exponent,   // Power exponent
    Inner,      // Negation inner / other unary
}

#[derive(Debug, Clone)]
pub struct Step {
    pub description: String,
    pub rule_name: String,
    pub before: ExprId,
    pub after: ExprId,
    pub path: Vec<PathStep>,
}

impl Step {
    pub fn new(description: &str, rule_name: &str, before: ExprId, after: ExprId, path: Vec<PathStep>) -> Self {
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
            path,
        }
    }
}
