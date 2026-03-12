#![allow(dead_code)]

use criterion::{measurement::WallTime, BenchmarkGroup};
use std::time::Duration;

const FAST_BENCH_FLAG: &str = "CAS_BENCH_FAST";

fn fast_mode_enabled() -> bool {
    std::env::var(FAST_BENCH_FLAG)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn configure_standard_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    if fast_mode_enabled() {
        group.sample_size(25);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
    }
}
