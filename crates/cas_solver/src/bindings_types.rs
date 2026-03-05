/// Lightweight binding view for presentation layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingOverviewEntry {
    pub name: String,
    pub expr: cas_ast::ExprId,
}

/// Result of applying a `clear`-style binding command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearBindingsResult {
    All {
        cleared_count: usize,
    },
    Selected {
        cleared_count: usize,
        missing_names: Vec<String>,
    },
}
