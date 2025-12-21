//! Contract tests for poly_gcd_modp presets
//!
//! These tests verify that preset budget values don't change silently.

use cas_engine::gcd_zippel_modp::{ZippelBudget, ZippelPreset};

#[test]
fn test_preset_mmgcd_contract() {
    // This test ensures the mm_gcd preset doesn't change silently.
    // If you intentionally change the preset, update this test.
    let budget = ZippelBudget::for_preset(ZippelPreset::MmGcd);

    assert_eq!(
        budget.max_points_per_var, 8,
        "mm_gcd preset: max_points changed"
    );
    assert_eq!(budget.max_retries, 8, "mm_gcd preset: max_retries changed");
    assert_eq!(
        budget.verify_trials, 3,
        "mm_gcd preset: verify_trials changed"
    );
    assert_eq!(
        budget.forced_main_var, None,
        "mm_gcd preset: forced_main_var should be None"
    );
}

#[test]
fn test_preset_safe_contract() {
    let budget = ZippelBudget::for_preset(ZippelPreset::Safe);

    assert_eq!(
        budget.max_points_per_var, 16,
        "safe preset: max_points changed"
    );
    assert_eq!(budget.max_retries, 32, "safe preset: max_retries changed");
    assert_eq!(
        budget.verify_trials, 6,
        "safe preset: verify_trials changed"
    );
    assert_eq!(
        budget.forced_main_var, None,
        "safe preset: forced_main_var should be None"
    );
}

#[test]
fn test_preset_aggressive_contract() {
    let budget = ZippelBudget::for_preset(ZippelPreset::Aggressive);

    assert_eq!(
        budget.max_points_per_var, 10,
        "aggressive preset: max_points changed"
    );
    assert_eq!(
        budget.max_retries, 16,
        "aggressive preset: max_retries changed"
    );
    assert_eq!(
        budget.verify_trials, 4,
        "aggressive preset: verify_trials changed"
    );
    assert_eq!(
        budget.forced_main_var, None,
        "aggressive preset: forced_main_var should be None"
    );
}

#[test]
fn test_preset_parsing() {
    // Case-insensitive parsing
    assert_eq!(ZippelPreset::from_str("mm_gcd"), Some(ZippelPreset::MmGcd));
    assert_eq!(ZippelPreset::from_str("mmgcd"), Some(ZippelPreset::MmGcd));
    assert_eq!(ZippelPreset::from_str("mm"), Some(ZippelPreset::MmGcd));
    assert_eq!(ZippelPreset::from_str("MM_GCD"), Some(ZippelPreset::MmGcd));

    assert_eq!(ZippelPreset::from_str("safe"), Some(ZippelPreset::Safe));
    assert_eq!(ZippelPreset::from_str("SAFE"), Some(ZippelPreset::Safe));

    assert_eq!(
        ZippelPreset::from_str("aggressive"),
        Some(ZippelPreset::Aggressive)
    );
    assert_eq!(
        ZippelPreset::from_str("fast"),
        Some(ZippelPreset::Aggressive)
    );

    assert_eq!(ZippelPreset::from_str("unknown"), None);
    assert_eq!(ZippelPreset::from_str(""), None);
}

#[test]
fn test_with_main_var_builder() {
    let budget = ZippelBudget::for_preset(ZippelPreset::MmGcd).with_main_var(Some(6));

    assert_eq!(budget.forced_main_var, Some(6));
    assert_eq!(budget.max_points_per_var, 8); // Other fields unchanged
}

#[test]
fn test_default_is_safe() {
    let default = ZippelBudget::default();
    let safe = ZippelBudget::for_preset(ZippelPreset::Safe);

    assert_eq!(default.max_points_per_var, safe.max_points_per_var);
    assert_eq!(default.max_retries, safe.max_retries);
    assert_eq!(default.verify_trials, safe.verify_trials);
}
