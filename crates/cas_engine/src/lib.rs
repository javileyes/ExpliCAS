pub mod rule;
pub mod step;
pub mod engine;
pub mod rules;
pub mod polynomial;
pub mod solver;
pub mod ordering;

pub mod visitors;
pub mod error;
#[macro_use]
pub mod macros;

pub use visitors::{VariableCollector, DepthVisitor};
pub use error::CasError;
pub use engine::Simplifier;
pub use rule::Rule;
pub use step::Step;
