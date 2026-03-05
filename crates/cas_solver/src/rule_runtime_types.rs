//! Local aliases for rule/orchestration runtime types.
//!
//! These are still engine-backed but exported from solver-owned modules so
//! compatibility does not depend on importing `engine_exports` directly.

pub use crate::engine_bridge::{
    LogExpansionRule, Orchestrator, ParentContext, Rewrite, Rule, SimpleRule,
};
