impl super::Category {
    /// All available categories
    pub fn all() -> &'static [super::Category] {
        &[
            super::Category::Transform,
            super::Category::Expansion,
            super::Category::Fractions,
            super::Category::Rationalization,
            super::Category::Mixed,
            super::Category::Baseline,
            super::Category::Roots,
            super::Category::Powers,
            super::Category::Stress,
            super::Category::Policy,
        ]
    }

    /// Short name for display
    pub fn as_str(&self) -> &'static str {
        match self {
            super::Category::Transform => "transform",
            super::Category::Expansion => "expansion",
            super::Category::Fractions => "fractions",
            super::Category::Rationalization => "rationalization",
            super::Category::Mixed => "mixed",
            super::Category::Baseline => "baseline",
            super::Category::Roots => "roots",
            super::Category::Powers => "powers",
            super::Category::Stress => "stress",
            super::Category::Policy => "policy",
        }
    }
}

impl std::fmt::Display for super::Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
