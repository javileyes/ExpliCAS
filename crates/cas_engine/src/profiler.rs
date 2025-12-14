use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lightweight profiler for tracking rule application statistics
#[derive(Default)]
pub struct RuleProfiler {
    stats: HashMap<String, RuleStats>,
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

impl RuleProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            stats: HashMap::new(),
            enabled,
            health_enabled: false,
        }
    }

    /// Record a successful rule application
    pub fn record(&mut self, rule_name: &str) {
        if !self.enabled {
            return;
        }

        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.applied.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful rule application with node delta (for health metrics)
    pub fn record_with_delta(&mut self, rule_name: &str, delta_nodes: i64) {
        if !self.enabled {
            return;
        }

        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.applied.fetch_add(1, Ordering::Relaxed);

        if self.health_enabled {
            stats
                .total_delta_nodes
                .fetch_add(delta_nodes, Ordering::Relaxed);
        }
    }

    /// Record a rejection due to semantic equality
    pub fn record_rejected_semantic(&mut self, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.rejected_semantic.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rejection due to phase restrictions
    pub fn record_rejected_phase(&mut self, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.rejected_phase.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rejection due to rule being disabled
    pub fn record_rejected_disabled(&mut self, rule_name: &str) {
        if !self.enabled || !self.health_enabled {
            return;
        }
        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.rejected_disabled.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate profiling report
    pub fn report(&self) -> String {
        if !self.enabled {
            return "Profiling not enabled. Use 'profile enable' to activate.".to_string();
        }

        let mut entries: Vec<_> = self
            .stats
            .iter()
            .map(|(name, stats)| (name, stats.applied.load(Ordering::Relaxed)))
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

        let total_hits: usize = self
            .stats
            .values()
            .map(|s| s.applied.load(Ordering::Relaxed))
            .sum();

        report.push_str("─────────────────────────────────────────────\n");
        report.push_str(&format!("{:40} {:>6}\n", "TOTAL", total_hits));

        report
    }

    /// Generate health report (includes rejections and node delta)
    pub fn health_report(&self) -> String {
        if !self.enabled {
            return "Profiling not enabled.".to_string();
        }

        let mut report = String::from("Rule Health Report\n");
        report.push_str("────────────────────────────────────────────────────────────────\n");

        // Top applied rules
        let mut applied: Vec<_> = self
            .stats
            .iter()
            .map(|(n, s)| (n.as_str(), s.applied.load(Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        applied.sort_by_key(|(_, c)| Reverse(*c));

        report.push_str("Top Applied Rules:\n");
        for (name, count) in applied.iter().take(5) {
            report.push_str(&format!("  {:40} {:>4}\n", truncate(name, 40), count));
        }

        // Top semantic rejections
        let mut rejected_sem: Vec<_> = self
            .stats
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
        let mut growth: Vec<_> = self
            .stats
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

    /// Get total positive growth (node increase) across all rules
    pub fn total_positive_growth(&self) -> i64 {
        self.stats
            .values()
            .map(|s| s.total_delta_nodes.load(Ordering::Relaxed))
            .filter(|d| *d > 0)
            .sum()
    }

    /// Get total negative growth (node reduction) across all rules
    pub fn total_negative_growth(&self) -> i64 {
        self.stats
            .values()
            .map(|s| s.total_delta_nodes.load(Ordering::Relaxed))
            .filter(|d| *d < 0)
            .sum()
    }

    /// Get total applied rules count
    pub fn total_applied(&self) -> usize {
        self.stats
            .values()
            .map(|s| s.applied.load(Ordering::Relaxed))
            .sum()
    }

    /// Get total semantic rejections count
    pub fn total_rejected_semantic(&self) -> usize {
        self.stats
            .values()
            .map(|s| s.rejected_semantic.load(Ordering::Relaxed))
            .sum()
    }

    /// Clear all statistics
    pub fn clear(&mut self) {
        for stats in self.stats.values() {
            stats.applied.store(0, Ordering::Relaxed);
            stats.rejected_semantic.store(0, Ordering::Relaxed);
            stats.rejected_phase.store(0, Ordering::Relaxed);
            stats.rejected_disabled.store(0, Ordering::Relaxed);
            stats.total_delta_nodes.store(0, Ordering::Relaxed);
        }
    }

    /// Clear health metrics for a new run (keeps profiler enabled state)
    pub fn clear_run(&mut self) {
        for stats in self.stats.values() {
            stats.applied.store(0, Ordering::Relaxed);
            stats.rejected_semantic.store(0, Ordering::Relaxed);
            stats.rejected_phase.store(0, Ordering::Relaxed);
            stats.rejected_disabled.store(0, Ordering::Relaxed);
            stats.total_delta_nodes.store(0, Ordering::Relaxed);
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

        profiler.record("AddZeroRule");
        profiler.record("AddZeroRule");
        profiler.record("MulOneRule");

        let report = profiler.report();
        assert!(report.contains("AddZeroRule"));
        assert!(report.contains("2"));
        assert!(report.contains("MulOneRule"));
        assert!(report.contains("1"));
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = RuleProfiler::new(true);

        profiler.record("TestRule");
        assert!(profiler.report().contains("TestRule"));

        profiler.clear();
        assert!(profiler.report().contains("No rules"));
    }
}
