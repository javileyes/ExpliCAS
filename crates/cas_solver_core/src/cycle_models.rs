//! Shared cycle-detection event models.

/// Information about a detected cycle.
#[derive(Debug, Clone)]
pub struct CycleInfo {
    /// Which phase detected the cycle.
    pub phase: crate::simplify_phase::SimplifyPhase,
    /// Cycle period (1=self-loop, 2=A<->B, ...).
    pub period: usize,
    /// At which rewrite step the cycle was detected.
    pub at_step: usize,
}

/// Level at which the cycle was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycleLevel {
    /// Detected within node-local rule application.
    IntraNode,
    /// Detected between full-tree iterations.
    InterIteration,
}

impl std::fmt::Display for CycleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CycleLevel::IntraNode => write!(f, "intra-node"),
            CycleLevel::InterIteration => write!(f, "inter-iteration"),
        }
    }
}

/// Record of a detected cycle during simplification.
#[derive(Debug, Clone)]
pub struct CycleEvent {
    /// Which pipeline phase detected the cycle.
    pub phase: crate::simplify_phase::SimplifyPhase,
    /// Cycle period (1=self-loop, 2=ping-pong, ...).
    pub period: usize,
    /// Detection level (intra-node vs inter-iteration).
    pub level: CycleLevel,
    /// Name of the rule that triggered the cycle.
    pub rule_name: String,
    /// Structural fingerprint of the expression at detection point.
    pub expr_fingerprint: u64,
    /// Human-readable expression (typically truncated).
    pub expr_display: String,
    /// Rewrite step count at detection time.
    pub rewrite_step: usize,
}

impl std::fmt::Display for CycleEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} cycle(period={}) at step {} by '{}': {}",
            self.phase,
            self.level,
            self.period,
            self.rewrite_step,
            self.rule_name,
            self.expr_display
        )
    }
}
