pub mod rule;
pub mod step;
pub mod engine;
pub mod rules; // Export rules module
pub mod polynomial;
pub mod solver;

pub use engine::Simplifier;
pub use rule::Rule;
pub use step::Step;
