mod expand;
mod factor_division;
mod fractions;
mod integrate;
mod logs;
mod match_support;
mod power_merge;
mod radicals;
mod rationalize;
mod strategy;
mod target_classifier;
mod target_form;
mod trig;

pub(crate) use expand::{try_rewrite_expanded_target_aware, ExpandRewriteKind};
pub(crate) use factor_division::{
    detect_factor_out_with_division_target, extract_factored_inner_target,
};
pub(crate) use fractions::{
    looks_like_fraction_expanded_target, looks_like_mixed_fraction_target,
    looks_like_telescoping_fraction_target, try_build_combined_fraction_from_fold_add,
    try_rewrite_exact_fraction_cancel_target_aware, try_rewrite_fraction_combination_target_aware,
    try_rewrite_fraction_expansion_target_aware,
};
pub(crate) use integrate::try_rewrite_integrate_prep_target_aware;
pub(crate) use logs::{
    try_rewrite_log_contraction_target_aware, try_rewrite_log_expansion_target_aware,
};
pub(crate) use match_support::{presentational_target_match, strong_target_match};
pub(crate) use power_merge::try_rewrite_power_merge_target_aware;
pub(crate) use radicals::try_rewrite_odd_half_power_target_aware;
pub(crate) use rationalize::looks_rationalizable_source;
pub(crate) use strategy::{ordered_strategies_for_target, DeriveStrategy};
pub(crate) use target_classifier::{classify_target_profile, DeriveTargetProfile};
pub(crate) use target_form::DeriveTargetForm;
pub(crate) use trig::{
    try_rewrite_pythagorean_factor_form_target_aware, try_rewrite_trig_contraction_target_aware,
    try_rewrite_trig_expansion, try_rewrite_trig_identity_to_one_target_aware,
};
