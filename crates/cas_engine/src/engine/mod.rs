pub mod equivalence;

mod hold;
mod orchestration;
mod simplifier;
mod transform;

pub use cas_ast::hold::strip_all_holds;
pub use cas_ast::substitute_expr_by_id;
pub use cas_math::{eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions};
pub use cas_solver_core::equivalence::EquivalenceResult;
pub use orchestration::LoopConfig;
pub use simplifier::Simplifier;
