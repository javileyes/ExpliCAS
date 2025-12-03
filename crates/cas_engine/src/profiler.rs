use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cmp::Reverse;

/// Lightweight profiler for tracking rule application statistics
#[derive(Default)]
pub struct RuleProfiler {
    stats: HashMap<String, RuleStats>,
    enabled: bool,
}

#[derive(Default)]
pub struct RuleStats {
    pub hit_count: AtomicUsize,
}

impl RuleProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            stats: HashMap::new(),
            enabled,
        }
    }
    
    /// Record a rule application (thread-safe, minimal overhead)
    pub fn record(&mut self, rule_name: &str) {
        if !self.enabled {
            return;
        }
        
        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.hit_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Generate profiling report
    pub fn report(&self) -> String {
        if !self.enabled {
            return "Profiling not enabled. Use 'profile enable' to activate.".to_string();
        }
        
        let mut entries: Vec<_> = self.stats.iter()
            .map(|(name, stats)| (name, stats.hit_count.load(Ordering::Relaxed)))
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
            report.push_str(&format!(
                "{:40} {:>6}\n",
                truncate(name, 40),
                count
            ));
        }
        
        let total_hits: usize = self.stats.values()
            .map(|s| s.hit_count.load(Ordering::Relaxed))
            .sum();
        
        report.push_str("─────────────────────────────────────────────\n");
        report.push_str(&format!("{:40} {:>6}\n", "TOTAL", total_hits));
        
        report
    }
    
    /// Clear all statistics
    pub fn clear(&mut self) {
        for stats in self.stats.values() {
            stats.hit_count.store(0, Ordering::Relaxed);
        }
    }
    
    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len-3])
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
