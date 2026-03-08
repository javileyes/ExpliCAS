//! Semantic configuration for evaluation and simplification.
//!
//! This module defines the 4 orthogonal semantic axes that control
//! how ExpliCAS evaluates and simplifies expressions.
//!
//! See `docs/SEMANTICS_POLICY.md` for the full specification.

pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::branch_policy::BranchPolicy;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::normal_form_goal::NormalFormGoal;
pub use cas_solver_core::value_domain::ValueDomain;
