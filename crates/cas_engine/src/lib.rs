pub mod canonical_forms;
pub mod canonicalization;
pub mod collect;
pub mod engine;
pub mod expand;
pub mod factor;
pub mod matrix;
pub mod orchestrator;
pub mod ordering;
pub mod parent_context;
pub mod pattern_detection;
pub mod pattern_marks;
pub mod pattern_scanner;
pub mod polynomial;
pub mod profiler;
pub mod rule;
pub mod rules;
pub mod semantic_equality;
pub mod solver;
pub mod step;
pub mod step_optimization;
pub mod strategies;
pub mod timeline;
pub mod visualizer;

pub mod error;
pub mod helpers;
pub mod visitors;
#[macro_use]
pub mod macros;

pub use engine::Simplifier;
pub use error::CasError;
pub use rule::Rule;
pub use step::Step;
pub use visitors::{DepthVisitor, VariableCollector};
