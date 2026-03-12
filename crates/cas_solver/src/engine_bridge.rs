//! Single compatibility bridge to `cas_engine`.
//!
//! Keeping all direct engine coupling here makes remaining migration work
//! explicit and easy to replace incrementally.

use cas_engine as engine;

pub use engine::Simplifier;
pub use engine::{Engine, Orchestrator, ParentContext, Rewrite, Rule, RuleProfiler, SimpleRule};
