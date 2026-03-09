#![allow(dead_code)]

use criterion::{measurement::WallTime, BenchmarkGroup};
use std::time::Duration;

const FAST_BENCH_FLAG: &str = "CAS_BENCH_FAST";

pub fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn fast_mode_enabled() -> bool {
    env_flag_enabled(FAST_BENCH_FLAG)
}

/// Configure a standard benchmark group.
///
/// Default mode keeps Criterion defaults intact. Fast mode reduces warmup and
/// measurement time and lowers the sample count for quicker local iteration.
pub fn configure_standard_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    if fast_mode_enabled() {
        group.sample_size(25);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
    }
}

/// Configure an intentionally slow benchmark group.
///
/// These benches already use lower sample sizes but much longer measurement
/// windows. Fast mode keeps the lower sample size and shortens timings.
pub fn configure_slow_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    if fast_mode_enabled() {
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
    } else {
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(30));
    }
}
