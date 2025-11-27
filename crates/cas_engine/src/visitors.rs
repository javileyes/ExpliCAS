use cas_ast::{Expr, Visitor};
use std::collections::HashSet;

pub struct VariableCollector {
    pub vars: HashSet<String>,
}

impl VariableCollector {
    pub fn new() -> Self {
        Self {
            vars: HashSet::new(),
        }
    }
}

impl Visitor for VariableCollector {
    fn visit_variable(&mut self, name: &str) {
        self.vars.insert(name.to_string());
    }
}
