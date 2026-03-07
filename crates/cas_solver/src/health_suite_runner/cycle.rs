use crate::{PipelineStats, SimplifyPhase};

pub(super) fn detect_cycle(stats: &PipelineStats) -> Option<(SimplifyPhase, usize)> {
    if let Some(ref c) = stats.core.cycle {
        return Some((SimplifyPhase::Core, c.period));
    }
    if let Some(ref c) = stats.transform.cycle {
        return Some((SimplifyPhase::Transform, c.period));
    }
    if let Some(ref c) = stats.rationalize.cycle {
        return Some((SimplifyPhase::Rationalize, c.period));
    }
    if let Some(ref c) = stats.post_cleanup.cycle {
        return Some((SimplifyPhase::PostCleanup, c.period));
    }
    None
}
