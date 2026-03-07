mod names;
mod parse;

/// Category of health test case
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Transform,
    Expansion,
    Fractions,
    Rationalization,
    Mixed,
    Baseline,
    Roots,
    Powers,
    Stress,
    Policy,
}
