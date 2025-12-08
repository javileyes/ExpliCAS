// Re-export ordering functionality from cas_ast
// This module used to contain compare_expr, but it has been moved to cas_ast
// to enable Context::add() to use it for automatic canonicalization.

pub use cas_ast::ordering::*;
