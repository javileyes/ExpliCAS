//! Solver facade crate.
//!
//! During migration this crate re-exports the solver API from `cas_engine`.

pub use cas_engine::domain::take_blocked_hints;
pub use cas_engine::implicit_domain::normalize_and_dedupe_conditions;
pub use cas_engine::parent_context::ParentContext;
pub use cas_engine::rule::Rule;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::solve_safety::*;
pub use cas_engine::solver::*;
pub use cas_engine::{BlockedHint, DomainMode};
