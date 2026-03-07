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
            _ => Err(format!(
                "Unknown category: '{}'. Valid: transform, expansion, fractions, rationalization, mixed, baseline, roots, powers, stress",
                s
            )),
        }
    }
}
