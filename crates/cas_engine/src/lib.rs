pub mod rule;
pub mod step;
pub mod engine;
pub mod rules;
pub mod polynomial;
pub mod solver;
pub mod ordering;
pub mod collect;
pub mod expand;
pub mod factor;
pub mod orchestrator;
pub mod step_optimization;
pub mod strategies;
pub mod profiler;
pub mod visualizer;

pub mod visitors;
pub mod error;
pub mod helpers;
#[macro_use]
pub mod macros;

pub use visitors::{VariableCollector, DepthVisitor};
pub use error::CasError;
pub use engine::Simplifier;
pub use rule::Rule;
pub use step::Step;
