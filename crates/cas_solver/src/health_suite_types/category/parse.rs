use std::str::FromStr;

impl FromStr for super::Category {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "transform" | "trans" | "t" => Ok(super::Category::Transform),
            "expansion" | "expand" | "exp" | "e" => Ok(super::Category::Expansion),
            "fractions" | "frac" | "f" => Ok(super::Category::Fractions),
            "rationalization" | "rational" | "rat" | "r" => Ok(super::Category::Rationalization),
            "mixed" | "mix" | "m" => Ok(super::Category::Mixed),
            "baseline" | "base" | "b" => Ok(super::Category::Baseline),
            "roots" | "root" => Ok(super::Category::Roots),
            "powers" | "pow" | "p" => Ok(super::Category::Powers),
            "stress" | "s" => Ok(super::Category::Stress),
            "policy" | "pol" => Ok(super::Category::Policy),
            "all" | "*" => Err("Use None for all categories".to_string()),
            _ => Err(format!(
                "Unknown category: '{}'. Valid: transform, expansion, fractions, rationalization, mixed, baseline, roots, powers, stress",
                s
            )),
        }
    }
}
