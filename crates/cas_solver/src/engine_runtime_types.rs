//! Local aliases for engine runtime façade types.
//!
//! These aliases keep the public surface solver-owned while migration
//! progressively decouples internals from `cas_engine`.

pub type AutoExpandBinomials = cas_solver_core::eval_option_axes::AutoExpandBinomials;
pub type BranchMode = cas_solver_core::eval_option_axes::BranchMode;
pub type CasError = cas_solver_core::error_model::CasError;
pub type ComplexMode = cas_solver_core::eval_option_axes::ComplexMode;
pub type ContextMode = cas_solver_core::eval_option_axes::ContextMode;
pub type Engine = crate::engine_bridge::Engine;
pub type EvalAction = cas_solver_core::eval_models::EvalAction;
pub type EvalOptions = cas_solver_core::eval_options::EvalOptions;
pub type EvalOutput = cas_solver_core::eval_output_model::EvalOutput;
pub type EvalRequest = cas_solver_core::eval_models::EvalRequest;
pub type EvalResult = cas_solver_core::eval_models::EvalResult;
pub type HeuristicPoly = cas_solver_core::eval_option_axes::HeuristicPoly;
pub type RuleProfiler = crate::engine_bridge::RuleProfiler;
pub type SharedSemanticConfig = cas_solver_core::simplify_options::SharedSemanticConfig;
pub type Simplifier = crate::engine_bridge::Simplifier;
pub type SimplifyOptions = cas_solver_core::simplify_options::SimplifyOptions;
pub type StepsMode = cas_solver_core::eval_option_axes::StepsMode;
