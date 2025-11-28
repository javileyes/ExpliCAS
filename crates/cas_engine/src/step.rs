use cas_ast::ExprId;

#[derive(Debug, Clone)]
pub struct Step {
    pub description: String,
    pub rule_name: String,
    pub before: ExprId,
    pub after: ExprId,
}

impl Step {
    pub fn new(description: &str, rule_name: &str, before: ExprId, after: ExprId) -> Self {
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
        }
    }
}
