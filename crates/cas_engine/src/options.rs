//! Evaluation options facade.
//!
//! Canonical models now live in `cas_solver_core`.

pub use cas_solver_core::eval_option_axes::{
    AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, HeuristicPoly, StepsMode,
};
pub use cas_solver_core::eval_options::EvalOptions;
