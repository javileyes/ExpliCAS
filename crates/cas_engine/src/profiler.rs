use crate::phase::SimplifyPhase;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lightweight profiler for tracking rule application statistics per phase
#[derive(Default)]
pub struct RuleProfiler {
    /// Stats per phase: [Core, Transform, Rationalize, PostCleanup]
    per_phase: [HashMap<String, RuleStats>; 4],
    enabled: bool,
    /// Track health metrics (rejected counts, node delta) - only when enabled
    health_enabled: bool,
}

#[derive(Default)]
pub struct RuleStats {
    /// Number of times the rule was successfully applied
    pub applied: AtomicUsize,
    /// Times rejected due to semantic equality check
    pub rejected_semantic: AtomicUsize,
    /// Times rejected due to phase restrictions
    pub rejected_phase: AtomicUsize,
    /// Times rejected due to being disabled
    pub rejected_disabled: AtomicUsize,
    /// Total node delta (positive = growth, negative = reduction)
    pub total_delta_nodes: std::sync::atomic::AtomicI64,
}

/// Convert SimplifyPhase to array index
#[inline]
fn phase_idx(phase: SimplifyPhase) -> usize {
    match phase {
        SimplifyPhase::Core => 0,
        SimplifyPhase::Transform => 1,
        SimplifyPhase::Rationalize => 2,
        SimplifyPhase::PostCleanup => 3,
    }
}

impl RuleProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            per_phase: Default::default(),
            enabled,
            health_enabled: false,
        }
    }

    /// Record a successful rule application (for a specific phase)
    pub fn record(&mut self, phase: SimplifyPhase, rule_name: &str) {
        if !self.enabled {
            return;
        }

        let stats = self.per_phase[phase_idx(phase)]
            .entry(rule_name.to_string())
            .or_default();
        stats.applied.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful rule application with node delta (for health metrics)
    pub fn record_with_delta(&mut self, phase: SimplifyPhase, rule_name: &str, delta_nodes: i64) {
        if !self.enabled {
            return;
        }

        let stats = self.per_phase[phase_idx(phase)]
            .entry(rule_name.to_string())
            .or_default();
        stats.applied.fetch_add(1, Ordering::Relaxed);

        if self.health_enabled {
            stats
                .total_delta_nodes
                .fetch_add(delta_nodes, Ordering::Relaxed);
        }
    }

    /// Record a rejection due to semantic equality
    pub fn record_rejected_semantic(&mut self, phase: SimplifyPhase, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.per_phase[phase_idx(phase)]
            .entry(rule_name.to_string())
            .or_default();
        stats.rejected_semantic.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rejection due to phase restrictions
    pub fn record_rejected_phase(&mut self, phase: SimplifyPhase, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.per_phase[phase_idx(phase)]
            .entry(rule_name.to_string())
            .or_default();
        stats.rejected_phase.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rejection due to rule being disabled
    pub fn record_rejected_disabled(&mut self, phase: SimplifyPhase, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.per_phase[phase_idx(phase)]
            .entry(rule_name.to_string())
            .or_default();
        stats.rejected_disabled.fetch_add(1, Ordering::Relaxed);
    }

    /// Aggregate stats across all phases (for backward compatibility)
    fn aggregate_stats(&self) -> HashMap<String, RuleStats> {
        let mut aggregated: HashMap<String, RuleStats> = HashMap::new();

        for phase_stats in &self.per_phase {
            for (name, stats) in phase_stats {
                let agg = aggregated.entry(name.clone()).or_default();
                agg.applied
                    .fetch_add(stats.applied.load(Ordering::Relaxed), Ordering::Relaxed);
                agg.rejected_semantic.fetch_add(
                    stats.rejected_semantic.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                agg.rejected_phase.fetch_add(
                    stats.rejected_phase.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                agg.rejected_disabled.fetch_add(
                    stats.rejected_disabled.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                agg.total_delta_nodes.fetch_add(
                    stats.total_delta_nodes.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
            }
        }

        aggregated
    }

    /// Generate profiling report (aggregated across all phases)
    pub fn report(&self) -> String {
        if !self.enabled {
            return "Profiling not enabled. Use 'profile enable' to activate.".to_string();
        }

        let aggregated = self.aggregate_stats();

        let mut entries: Vec<_> = aggregated
            .iter()
            .map(|(name, stats)| (name.as_str(), stats.applied.load(Ordering::Relaxed)))
            .filter(|(_, count)| *count > 0)
            .collect();

        if entries.is_empty() {
            return "No rules have been applied yet.".to_string();
        }

        // Sort by hit count (descending)
        entries.sort_by_key(|(_, count)| Reverse(*count));

        let mut report = String::from("Rule Profiling Report\n");
        report.push_str("─────────────────────────────────────────────\n");
        report.push_str(&format!("{:40} {:>6}\n", "Rule", "Hits"));
        report.push_str("─────────────────────────────────────────────\n");

        for (name, count) in entries {
            report.push_str(&format!("{:40} {:>6}\n", truncate(name, 40), count));
        }

        let total_hits: usize = aggregated
            .values()
            .map(|s| s.applied.load(Ordering::Relaxed))
            .sum();

        report.push_str("─────────────────────────────────────────────\n");
        report.push_str(&format!("{:40} {:>6}\n", "TOTAL", total_hits));

        report
    }

    /// Generate health report (aggregated or per-phase)
    ///
    /// - `phase = None`: aggregate across all phases
    /// - `phase = Some(p)`: only stats for that phase
    pub fn health_report_for_phase(&self, phase: Option<SimplifyPhase>) -> String {
        if !self.enabled {
            return "Profiling not enabled.".to_string();
        }

        let stats_map: HashMap<String, RuleStats> = match phase {
            None => self.aggregate_stats(),
            Some(p) => {
                // Clone stats for the specific phase
                let mut result = HashMap::new();
                for (name, stats) in &self.per_phase[phase_idx(p)] {
                    let new_stats = RuleStats::default();
                    new_stats
                        .applied
                        .store(stats.applied.load(Ordering::Relaxed), Ordering::Relaxed);
                    new_stats.rejected_semantic.store(
                        stats.rejected_semantic.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    new_stats.rejected_phase.store(
                        stats.rejected_phase.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    new_stats.rejected_disabled.store(
                        stats.rejected_disabled.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    new_stats.total_delta_nodes.store(
                        stats.total_delta_nodes.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    result.insert(name.clone(), new_stats);
                }
                result
            }
        };

        let phase_label = match phase {
            None => "Rule Health Report".to_string(),
            Some(p) => format!("Rule Health Report ({:?})", p),
        };

        let mut report = format!("{}\n", phase_label);
        report.push_str("────────────────────────────────────────────────────────────────\n");

        // Top applied rules
        let mut applied: Vec<_> = stats_map
            .iter()
            .map(|(n, s)| (n.as_str(), s.applied.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        applied.sort_by_key(|(_, c)| Reverse(*c));

        report.push_str("Top Applied Rules:\n");
        if applied.is_empty() {
            report.push_str("  (none)\n");
        } else {
            for (name, count) in applied.iter().take(5) {
                report.push_str(&format!("  {:40} {:>4}\n", truncate(name, 40), count));
            }
        }

        // Top semantic rejections
        let mut rejected_sem: Vec<_> = stats_map
            .iter()
            .map(|(n, s)| (n.as_str(), s.rejected_semantic.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        rejected_sem.sort_by_key(|(_, c)| Reverse(*c));

        if !rejected_sem.is_empty() {
            report.push_str("\nTop Semantic Rejections:\n");
            for (name, count) in rejected_sem.iter().take(5) {
                report.push_str(&format!("  {:40} {:>4}\n", truncate(name, 40), count));
            }
        }

        // Top growth rules (positive delta)
        let mut growth: Vec<_> = stats_map
            .iter()
            .map(|(n, s)| (n.as_str(), s.total_delta_nodes.load(Ordering::Relaxed)))
            .filter(|(_, d)| *d > 0)
            .collect();
        growth.sort_by_key(|(_, d)| Reverse(*d));

        if !growth.is_empty() {
            report.push_str("\nTop Growth Rules (node increase):\n");
            for (name, delta) in growth.iter().take(5) {
                report.push_str(&format!(
                    "  {:40} +{:>4} nodes\n",
                    truncate(name, 40),
                    delta
                ));
            }
        }

        report.push_str("────────────────────────────────────────────────────────────────\n");
        report
    }

    /// Generate health report (aggregated) - backward compatible
    pub fn health_report(&self) -> String {
        self.health_report_for_phase(None)
    }

    /// Get top N applied rules for a specific phase (for cycle culprit hints)
    pub fn top_applied_for_phase(&self, phase: SimplifyPhase, n: usize) -> Vec<(String, usize)> {
        let mut applied: Vec<_> = self.per_phase[phase_idx(phase)]
            .iter()
            .map(|(name, stats)| (name.clone(), stats.applied.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        applied.sort_by_key(|(_, c)| Reverse(*c));
        applied.truncate(n);
        applied
    }

    /// Get total positive growth (node increase) across all rules
    pub fn total_positive_growth(&self) -> i64 {
        self.aggregate_stats()
            .values()
            .map(|s| s.total_delta_nodes.load(Ordering::Relaxed))
            .filter(|d| *d > 0)
            .sum()
    }

    /// Get total negative growth (node reduction) across all rules
    pub fn total_negative_growth(&self) -> i64 {
        self.aggregate_stats()
            .values()
            .map(|s| s.total_delta_nodes.load(Ordering::Relaxed))
            .filter(|d| *d < 0)
            .sum()
    }

    /// Get total applied rules count
    pub fn total_applied(&self) -> usize {
        self.aggregate_stats()
            .values()
            .map(|s| s.applied.load(Ordering::Relaxed))
            .sum()
    }

    /// Get total semantic rejections count
    pub fn total_rejected_semantic(&self) -> usize {
        self.aggregate_stats()
            .values()
            .map(|s| s.rejected_semantic.load(Ordering::Relaxed))
            .sum()
    }

    /// Clear all statistics
    pub fn clear(&mut self) {
        for phase_stats in &mut self.per_phase {
            for stats in phase_stats.values() {
                stats.applied.store(0, Ordering::Relaxed);
                stats.rejected_semantic.store(0, Ordering::Relaxed);
                stats.rejected_phase.store(0, Ordering::Relaxed);
                stats.rejected_disabled.store(0, Ordering::Relaxed);
                stats.total_delta_nodes.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Clear health metrics for a new run (keeps profiler enabled state)
    pub fn clear_run(&mut self) {
        for phase_stats in &mut self.per_phase {
            for stats in phase_stats.values() {
                stats.applied.store(0, Ordering::Relaxed);
                stats.rejected_semantic.store(0, Ordering::Relaxed);
                stats.rejected_phase.store(0, Ordering::Relaxed);
                stats.rejected_disabled.store(0, Ordering::Relaxed);
                stats.total_delta_nodes.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Enable health metrics (node counting, rejection tracking)
    pub fn enable_health(&mut self) {
        self.enabled = true;
        self.health_enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
        self.health_enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if health metrics are enabled
    pub fn is_health_enabled(&self) -> bool {
        self.health_enabled
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_disabled_by_default() {
        let profiler = RuleProfiler::default();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profiler_record_and_report() {
        let mut profiler = RuleProfiler::new(true);

        profiler.record(SimplifyPhase::Core, "AddZeroRule");
        profiler.record(SimplifyPhase::Core, "AddZeroRule");
        profiler.record(SimplifyPhase::Transform, "MulOneRule");

        let report = profiler.report();
        assert!(report.contains("AddZeroRule"));
        assert!(report.contains("2"));
        assert!(report.contains("MulOneRule"));
        assert!(report.contains("1"));
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = RuleProfiler::new(true);

        profiler.record(SimplifyPhase::Core, "TestRule");
        assert!(profiler.report().contains("TestRule"));

        profiler.clear();
        assert!(profiler.report().contains("No rules"));
    }

    #[test]
    fn test_per_phase_isolation() {
        let mut profiler = RuleProfiler::new(true);

        profiler.record(SimplifyPhase::Core, "CoreRule");
        profiler.record(SimplifyPhase::Core, "CoreRule");
        profiler.record(SimplifyPhase::Transform, "TransformRule");

        // Check Core phase
        let top_core = profiler.top_applied_for_phase(SimplifyPhase::Core, 5);
        assert_eq!(top_core.len(), 1);
        assert_eq!(top_core[0].0, "CoreRule");
        assert_eq!(top_core[0].1, 2);

        // Check Transform phase
        let top_transform = profiler.top_applied_for_phase(SimplifyPhase::Transform, 5);
        assert_eq!(top_transform.len(), 1);
        assert_eq!(top_transform[0].0, "TransformRule");
        assert_eq!(top_transform[0].1, 1);

        // Check Rationalize phase (should be empty)
        let top_rat = profiler.top_applied_for_phase(SimplifyPhase::Rationalize, 5);
        assert!(top_rat.is_empty());
    }

    #[test]
    fn test_health_report_per_phase() {
        let mut profiler = RuleProfiler::new(true);
        profiler.enable_health();

        profiler.record_with_delta(SimplifyPhase::Core, "CoreRule", 5);
        profiler.record_with_delta(SimplifyPhase::Transform, "Distribute", 10);

        // Core report should only show CoreRule
        let core_report = profiler.health_report_for_phase(Some(SimplifyPhase::Core));
        assert!(core_report.contains("CoreRule"));
        assert!(!core_report.contains("Distribute"));

        // Transform report should only show Distribute
        let transform_report = profiler.health_report_for_phase(Some(SimplifyPhase::Transform));
        assert!(transform_report.contains("Distribute"));
        assert!(!transform_report.contains("CoreRule"));

        // Aggregate should show both
        let agg_report = profiler.health_report();
        assert!(agg_report.contains("CoreRule"));
        assert!(agg_report.contains("Distribute"));
    }
}
