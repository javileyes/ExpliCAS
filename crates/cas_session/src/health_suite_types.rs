use cas_solver::SimplifyPhase;
use std::str::FromStr;

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

impl Category {
    /// All available categories
    pub fn all() -> &'static [Category] {
        &[
            Category::Transform,
            Category::Expansion,
            Category::Fractions,
            Category::Rationalization,
            Category::Mixed,
            Category::Baseline,
            Category::Roots,
            Category::Powers,
            Category::Stress,
            Category::Policy,
        ]
    }

    /// Short name for display
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Transform => "transform",
            Category::Expansion => "expansion",
            Category::Fractions => "fractions",
            Category::Rationalization => "rationalization",
            Category::Mixed => "mixed",
            Category::Baseline => "baseline",
            Category::Roots => "roots",
            Category::Powers => "powers",
            Category::Stress => "stress",
            Category::Policy => "policy",
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for Category {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "transform" | "trans" | "t" => Ok(Category::Transform),
            "expansion" | "expand" | "exp" | "e" => Ok(Category::Expansion),
            "fractions" | "frac" | "f" => Ok(Category::Fractions),
            "rationalization" | "rational" | "rat" | "r" => Ok(Category::Rationalization),
            "mixed" | "mix" | "m" => Ok(Category::Mixed),
            "baseline" | "base" | "b" => Ok(Category::Baseline),
            "roots" | "root" => Ok(Category::Roots),
            "powers" | "pow" | "p" => Ok(Category::Powers),
            "stress" | "s" => Ok(Category::Stress),
            "policy" | "pol" => Ok(Category::Policy),
            "all" | "*" => Err("Use None for all categories".to_string()),
            _ => Err(format!("Unknown category: '{}'. Valid: transform, expansion, fractions, rationalization, mixed, baseline, roots, powers, stress", s)),
        }
    }
}

/// Health limits for a test case
#[derive(Debug, Clone)]
pub struct HealthLimits {
    /// Maximum total rewrites across all phases
    pub max_total_rewrites: usize,
    /// Maximum positive node growth
    pub max_growth: i64,
    /// Maximum rewrites in Transform phase
    pub max_transform_rewrites: usize,
    /// Whether cycles should cause failure (default: true)
    pub forbid_cycles: bool,
}

impl Default for HealthLimits {
    fn default() -> Self {
        Self {
            max_total_rewrites: 100,
            max_growth: 200,
            max_transform_rewrites: 50,
            forbid_cycles: true,
        }
    }
}

/// A single health test case
#[derive(Debug, Clone)]
pub struct HealthCase {
    /// Human-readable name
    pub name: &'static str,
    /// Category
    pub category: Category,
    /// Input expression
    pub expr: &'static str,
    /// Health limits
    pub limits: HealthLimits,
}

/// Result of running a health case
#[derive(Debug)]
pub struct HealthCaseResult {
    pub case: HealthCase,
    pub passed: bool,
    pub total_rewrites: usize,
    /// Per-phase rewrites
    pub core_rewrites: usize,
    pub transform_rewrites: usize,
    pub rationalize_rewrites: usize,
    pub post_rewrites: usize,
    /// Growth metrics
    pub growth: i64, // total_positive_growth
    pub shrink: i64, // total_negative_growth (absolute value)
    pub cycle_detected: Option<(SimplifyPhase, usize)>, // (phase, period)
    pub top_rules: Vec<(String, usize)>,
    pub failure_reason: Option<String>,
    /// Warning: cycle detected but not failing (forbid_cycles=false) or near limit
    pub warning: Option<String>,
}
