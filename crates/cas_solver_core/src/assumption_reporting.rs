//! Visibility policy for reporting domain assumptions.
//!
//! This enum is shared by engine/solver frontends and intentionally lives in
//! `cas_solver_core` to avoid coupling reporting policy to one runtime crate.

/// Controls how assumptions are reported to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssumptionReporting {
    /// No assumptions shown (hard off - not in JSON)
    #[default]
    Off,
    /// Deduped summary list at end
    Summary,
    /// Include step locations and trace info
    Trace,
}

impl AssumptionReporting {
    /// Parse from string (for REPL commands).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "off" => Some(Self::Off),
            "summary" => Some(Self::Summary),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }
}

impl std::fmt::Display for AssumptionReporting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Summary => write!(f, "summary"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AssumptionReporting;

    #[test]
    fn parse_is_case_insensitive() {
        assert_eq!(
            AssumptionReporting::parse("off"),
            Some(AssumptionReporting::Off)
        );
        assert_eq!(
            AssumptionReporting::parse("SUMMARY"),
            Some(AssumptionReporting::Summary)
        );
        assert_eq!(
            AssumptionReporting::parse("Trace"),
            Some(AssumptionReporting::Trace)
        );
        assert_eq!(AssumptionReporting::parse("invalid"), None);
    }

    #[test]
    fn display_round_trip() {
        assert_eq!(AssumptionReporting::Off.to_string(), "off");
        assert_eq!(AssumptionReporting::Summary.to_string(), "summary");
        assert_eq!(AssumptionReporting::Trace.to_string(), "trace");
    }
}
