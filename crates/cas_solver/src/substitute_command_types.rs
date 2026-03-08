use cas_ast::ExprId;
pub use cas_solver_core::substitute_command_types::{SubstituteParseError, SubstituteRenderMode};

pub type SubstituteEvalOutput = SubstituteSimplifyEvalOutput;

/// Evaluated payload for REPL-style `subst` followed by simplify.
#[derive(Debug, Clone)]
pub struct SubstituteSimplifyEvalOutput {
    pub simplified_expr: ExprId,
    pub strategy: crate::SubstituteStrategy,
    pub steps: Vec<crate::Step>,
}
