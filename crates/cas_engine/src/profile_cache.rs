//! Rule Profile Caching
//!
//! Caches rule profiles to avoid rebuilding rules on each evaluation.
//! Profiles are keyed by (BranchMode, ContextMode, ComplexMode) combination.

use crate::options::{BranchMode, ComplexMode, ContextMode, EvalOptions};
use crate::rule::Rule;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Key for profile cache - combines all options that affect rule selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProfileKey {
    pub branch_mode: BranchMode,
    pub context_mode: ContextMode,
    pub complex_mode: ComplexMode,
}

impl ProfileKey {
    pub fn from_options(opts: &EvalOptions) -> Self {
        Self {
            branch_mode: opts.branch_mode,
            context_mode: opts.shared.context_mode,
            complex_mode: opts.complex_mode,
        }
    }
}

/// A cached rule profile containing pre-built rules.
/// Shared via `Arc` between Simplifiers with the same options.
#[derive(Debug)]
#[allow(clippy::arc_with_non_send_sync)] // Rules are intentionally not Send+Sync for flexibility
pub struct RuleProfile {
    /// Rules indexed by target type
    pub rules: HashMap<crate::target_kind::TargetKind, Vec<Arc<dyn Rule>>>,
    /// Global rules (apply to all expression types)
    pub global_rules: Vec<Arc<dyn Rule>>,
    /// Disabled rule names for this profile
    pub disabled_rules: HashSet<String>,
    /// The key that identifies this profile
    pub key: ProfileKey,
}

/// Cache for rule profiles, reusing rules across Simplifier instances.
#[derive(Default, Debug)]
pub struct ProfileCache {
    profiles: HashMap<ProfileKey, Arc<RuleProfile>>,
}

impl ProfileCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Get or build a profile for the given options.
    #[allow(clippy::arc_with_non_send_sync)] // Rules use Arc for shared ownership, not thread safety
    pub fn get_or_build(&mut self, opts: &EvalOptions) -> Arc<RuleProfile> {
        let key = ProfileKey::from_options(opts);

        if let Some(profile) = self.profiles.get(&key) {
            return Arc::clone(profile);
        }

        // Build new profile
        let profile = Arc::new(build_profile(opts));
        self.profiles.insert(key, Arc::clone(&profile));
        profile
    }

    /// Number of cached profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
}

/// Build a RuleProfile for the given options.
/// This mirrors the logic in Simplifier::with_profile but captures the result.
fn build_profile(opts: &EvalOptions) -> RuleProfile {
    use crate::Simplifier;

    // Create a temporary simplifier to build the rules
    let simplifier = Simplifier::with_profile(opts);

    // Extract the rules (this is safe because we're building, not sharing yet)
    let key = ProfileKey::from_options(opts);

    RuleProfile {
        rules: simplifier.get_rules_clone(),
        global_rules: simplifier.get_global_rules_clone(),
        disabled_rules: simplifier.get_disabled_rules_clone(),
        key,
    }
}

impl std::fmt::Debug for dyn Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rule({})", self.name())
    }
}
