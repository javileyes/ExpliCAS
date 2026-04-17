use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Duration;

static CAS_PROFILE_ORCHESTRATOR_SHORTCUTS_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("CAS_PROFILE_ORCHESTRATOR_SHORTCUTS").is_ok());

#[derive(Debug, Clone, Default)]
struct ShortcutStats {
    attempts: usize,
    hits: usize,
    total_duration: Duration,
    samples: Vec<String>,
}

#[derive(Default)]
struct ShortcutProfiler {
    stats: HashMap<&'static str, ShortcutStats>,
}

impl ShortcutProfiler {
    fn record_attempt(&mut self, name: &'static str, matched: bool, elapsed: Duration) {
        let stats = self.stats.entry(name).or_default();
        stats.attempts += 1;
        if matched {
            stats.hits += 1;
        }
        stats.total_duration += elapsed;
    }

    fn record_sample(&mut self, name: &'static str, sample: String) {
        let stats = self.stats.entry(name).or_default();
        if stats.samples.len() >= 4 || stats.samples.iter().any(|existing| existing == &sample) {
            return;
        }
        stats.samples.push(sample);
    }

    fn clear(&mut self) {
        self.stats.clear();
    }

    fn report(&self) -> String {
        if self.stats.is_empty() {
            return "No orchestrator sections have been profiled yet.".to_string();
        }

        let mut rows: Vec<_> = self.stats.iter().collect();
        rows.sort_by_key(|(_, stats)| Reverse(stats.total_duration));

        let mut report = String::from("Orchestrator Profiling Report\n");
        report.push_str(
            "──────────────────────────────────────────────────────────────────────────────────────────────\n",
        );
        report.push_str(&format!(
            "{:48} {:>8} {:>8} {:>8} {:>12} {:>12}\n",
            "Section", "Attempts", "Hits", "Misses", "Total ms", "Avg us"
        ));
        report.push_str(
            "──────────────────────────────────────────────────────────────────────────────────────────────\n",
        );

        for (name, stats) in rows {
            let misses = stats.attempts.saturating_sub(stats.hits);
            let total_ms = stats.total_duration.as_secs_f64() * 1000.0;
            let avg_us = if stats.attempts == 0 {
                0.0
            } else {
                stats.total_duration.as_secs_f64() * 1_000_000.0 / stats.attempts as f64
            };
            report.push_str(&format!(
                "{:48} {:>8} {:>8} {:>8} {:>12.3} {:>12.2}\n",
                truncate(name, 48),
                stats.attempts,
                stats.hits,
                misses,
                total_ms,
                avg_us
            ));
        }

        let total_attempts: usize = self.stats.values().map(|stats| stats.attempts).sum();
        let total_hits: usize = self.stats.values().map(|stats| stats.hits).sum();
        let total_duration: Duration = self.stats.values().map(|stats| stats.total_duration).sum();

        report.push_str(
            "──────────────────────────────────────────────────────────────────────────────────────────────\n",
        );
        report.push_str(&format!(
            "{:48} {:>8} {:>8} {:>8} {:>12.3} {:>12.2}\n",
            "TOTAL",
            total_attempts,
            total_hits,
            total_attempts.saturating_sub(total_hits),
            total_duration.as_secs_f64() * 1000.0,
            if total_attempts == 0 {
                0.0
            } else {
                total_duration.as_secs_f64() * 1_000_000.0 / total_attempts as f64
            }
        ));

        let mut sample_rows: Vec<_> = self
            .stats
            .iter()
            .filter(|(_, stats)| !stats.samples.is_empty())
            .collect();
        sample_rows.sort_by_key(|(_, stats)| Reverse(stats.total_duration));
        if !sample_rows.is_empty() {
            report.push_str(
                "──────────────────────────────────────────────────────────────────────────────────────────────\n",
            );
            report.push_str("Sample expressions\n");
            report.push_str(
                "──────────────────────────────────────────────────────────────────────────────────────────────\n",
            );
            for (name, stats) in sample_rows {
                report.push_str(&format!("{}\n", name));
                for sample in &stats.samples {
                    report.push_str(&format!("  - {}\n", truncate(sample, 120)));
                }
            }
        }
        report
    }
}

thread_local! {
    static ORCHESTRATOR_SHORTCUT_PROFILER: RefCell<ShortcutProfiler> =
        RefCell::new(ShortcutProfiler::default());
}

pub fn orchestrator_shortcut_profiling_enabled() -> bool {
    *CAS_PROFILE_ORCHESTRATOR_SHORTCUTS_ENABLED
}

pub fn record_orchestrator_shortcut_attempt(name: &'static str, matched: bool, elapsed: Duration) {
    if !orchestrator_shortcut_profiling_enabled() {
        return;
    }
    ORCHESTRATOR_SHORTCUT_PROFILER.with(|profiler| {
        profiler.borrow_mut().record_attempt(name, matched, elapsed);
    });
}

pub fn clear_orchestrator_shortcut_profile() {
    ORCHESTRATOR_SHORTCUT_PROFILER.with(|profiler| {
        profiler.borrow_mut().clear();
    });
}

pub fn record_orchestrator_shortcut_sample(name: &'static str, sample: String) {
    if !orchestrator_shortcut_profiling_enabled() {
        return;
    }
    ORCHESTRATOR_SHORTCUT_PROFILER.with(|profiler| {
        profiler.borrow_mut().record_sample(name, sample);
    });
}

pub fn orchestrator_shortcut_profile_report() -> String {
    if !orchestrator_shortcut_profiling_enabled() {
        return "Orchestrator profiling not enabled. Set CAS_PROFILE_ORCHESTRATOR_SHORTCUTS=1."
            .to_string();
    }
    ORCHESTRATOR_SHORTCUT_PROFILER.with(|profiler| profiler.borrow().report())
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }
    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        if idx + 1 >= max_len {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::{ShortcutProfiler, ShortcutStats};
    use std::time::Duration;

    #[test]
    fn report_aggregates_attempts_hits_and_duration() {
        let mut profiler = ShortcutProfiler::default();
        profiler.record_attempt("root.mul.alpha", true, Duration::from_micros(50));
        profiler.record_attempt("root.mul.alpha", false, Duration::from_micros(70));
        profiler.record_attempt("root.mul.beta", true, Duration::from_micros(20));

        let report = profiler.report();
        assert!(report.contains("root.mul.alpha"));
        assert!(report.contains("root.mul.beta"));
        assert!(report.contains("TOTAL"));
        assert!(report.contains("Attempts"));
        assert!(report.contains("Misses"));
    }

    #[test]
    fn clear_drops_all_stats() {
        let mut profiler = ShortcutProfiler::default();
        profiler.record_attempt("root.mul.alpha", true, Duration::from_micros(10));
        assert!(!profiler.stats.is_empty());

        profiler.clear();
        assert!(profiler.stats.is_empty());
        assert_eq!(
            profiler.report(),
            "No orchestrator sections have been profiled yet."
        );
    }

    #[test]
    fn shortcut_stats_default_is_zeroed() {
        let stats = ShortcutStats::default();
        assert_eq!(stats.attempts, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.total_duration, Duration::ZERO);
    }

    #[test]
    fn report_includes_sample_expressions() {
        let mut profiler = ShortcutProfiler::default();
        profiler.record_attempt("root.mul.alpha", true, Duration::from_micros(10));
        profiler.record_sample("root.mul.alpha", "sin(pi/6) + 1".to_string());

        let report = profiler.report();
        assert!(report.contains("Sample expressions"));
        assert!(report.contains("sin(pi/6) + 1"));
    }
}
