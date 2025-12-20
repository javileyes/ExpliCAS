pub mod auto_expand_scan;
pub mod canonical_forms;
pub mod collect;
pub mod cycle_detector;
pub mod didactic;
pub mod display_context;
pub mod engine;
pub mod env;
pub mod eval;
pub mod expand;
pub mod factor;
pub mod gcd_zippel_modp;
pub mod matrix;
pub mod modp;
pub mod mono;
pub mod multipoly;
pub mod multipoly_modp;
pub mod options;
pub mod orchestrator;
pub mod ordering;
pub mod parent_context;
pub mod pattern_detection;
pub mod pattern_marks;
pub mod pattern_scanner;
pub mod phase;
pub mod polynomial;
pub mod profile_cache;
pub mod profiler;
pub mod rationalize;
pub mod rationalize_policy;
pub mod rule;
pub mod rules;
pub mod semantic_equality;
pub mod session;
pub mod session_state;
pub mod solver;
pub mod step;
pub mod step_optimization;
pub mod strategies;
pub mod telescoping;
pub mod timeline;
pub mod timeline_templates;
pub mod unipoly_modp;
pub mod visualizer;

pub mod build;
pub mod error;
pub mod helpers;
pub mod visitors;
#[macro_use]
pub mod macros;

pub use engine::Simplifier;
pub use error::CasError;
pub use eval::*;
pub use phase::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseStats, PipelineStats, SimplifyOptions,
    SimplifyPhase,
};
pub use rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome, RationalizeReason};
pub use rule::Rule;
pub use session::{resolve_session_refs, Entry, EntryId, EntryKind, ResolveError, SessionStore};
pub use session_state::SessionState;
pub use step::Step;
pub use visitors::{DepthVisitor, VariableCollector};
