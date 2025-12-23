// Clippy allows for patterns that are difficult to refactor safely
// TODO: Progressively fix these and remove allows
#![allow(clippy::needless_range_loop)] // Many loops index multiple arrays
#![allow(clippy::too_many_arguments)] // Complex math algorithms need many params
#![allow(clippy::field_reassign_with_default)] // Used in test setup patterns
#![allow(clippy::arc_with_non_send_sync)] // Internal threading model is safe
#![allow(clippy::match_like_matches_macro)] // Some matches are clearer expanded
#![allow(clippy::nonminimal_bool)] // Mathematical boolean logic can be explicit
#![allow(clippy::never_loop)] // Some early-return patterns are intentional
#![allow(clippy::should_implement_trait)] // Custom from_str methods are intentional
#![allow(dead_code)] // Some fields reserved for future use

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
pub mod multinomial_expand;
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
pub mod poly_modp_conv;
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
pub mod nary;
pub mod visitors;
#[macro_use]
pub mod macros;

pub use engine::{strip_all_holds, Simplifier};
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
