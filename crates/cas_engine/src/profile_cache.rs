//! Rule Profile Caching
//!
//! Caches rule profiles to avoid rebuilding rules on each evaluation.
//! Profiles are keyed by (BranchMode, ContextMode, ComplexMode) combination.

use crate::options::{BranchMode, ComplexMode, ContextMode, EvalOptions};
use crate::rule::Rule;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(crate) const PHASE_SLOT_COUNT: usize = 4;
type RuleBuckets = HashMap<crate::target_kind::TargetKind, Vec<Arc<dyn Rule>>>;
type PhaseRuleBuckets = [RuleBuckets; PHASE_SLOT_COUNT];

#[inline]
pub(crate) fn phase_index(phase: crate::phase::SimplifyPhase) -> usize {
    match phase {
        crate::phase::SimplifyPhase::Core => 0,
        crate::phase::SimplifyPhase::Transform => 1,
        crate::phase::SimplifyPhase::Rationalize => 2,
        crate::phase::SimplifyPhase::PostCleanup => 3,
    }
}

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
    pub rules: RuleBuckets,
    /// Rules pre-filtered by simplification phase for hot-loop dispatch.
    pub phase_rules: PhaseRuleBuckets,
    /// Global rules (apply to all expression types)
    pub global_rules: Vec<Arc<dyn Rule>>,
    /// Global rules pre-filtered by simplification phase.
    pub phase_global_rules: [Vec<Arc<dyn Rule>>; PHASE_SLOT_COUNT],
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
        use std::collections::hash_map::Entry;

        let key = ProfileKey::from_options(opts);
        match self.profiles.entry(key) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let profile = Arc::new(build_profile(opts));
                Arc::clone(entry.insert(profile))
            }
        }
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

#[allow(clippy::arc_with_non_send_sync)] // Rules are intentionally not Send+Sync for flexibility
pub fn default_rule_profile() -> Arc<RuleProfile> {
    thread_local! {
        static DEFAULT_RULE_PROFILE: RefCell<Option<Arc<RuleProfile>>> = const { RefCell::new(None) };
    }

    DEFAULT_RULE_PROFILE.with(|slot| {
        let mut slot = slot.borrow_mut();
        Arc::clone(slot.get_or_insert_with(|| Arc::new(build_profile(&EvalOptions::default()))))
    })
}

#[allow(clippy::arc_with_non_send_sync)] // Rules are intentionally not Send+Sync for flexibility
pub fn rule_profile_from_simplifier(
    simplifier: &crate::Simplifier,
    opts: &EvalOptions,
) -> Arc<RuleProfile> {
    let key = ProfileKey::from_options(opts);
    let rules = simplifier.get_rules_clone();
    let global_rules = simplifier.get_global_rules_clone();
    let disabled_rules = simplifier.get_disabled_rules_clone();

    Arc::new(RuleProfile {
        phase_rules: build_phase_rules(&rules, &disabled_rules),
        rules,
        phase_global_rules: build_phase_global_rules(&global_rules, &disabled_rules),
        global_rules,
        disabled_rules,
        key,
    })
}

/// Build a RuleProfile for the given options.
/// This mirrors the logic in Simplifier::with_profile but captures the result.
fn build_profile(opts: &EvalOptions) -> RuleProfile {
    use crate::Simplifier;

    // Create a temporary simplifier to build the rules
    let key = ProfileKey::from_options(opts);
    let simplifier = Simplifier::with_profile(opts);
    let rules = simplifier.get_rules_clone();
    let global_rules = simplifier.get_global_rules_clone();

    let disabled_rules = simplifier.get_disabled_rules_clone();

    RuleProfile {
        phase_rules: build_phase_rules(&rules, &disabled_rules),
        rules,
        phase_global_rules: build_phase_global_rules(&global_rules, &disabled_rules),
        global_rules,
        disabled_rules,
        key,
    }
}

fn build_phase_rules(rules: &RuleBuckets, disabled_rules: &HashSet<String>) -> PhaseRuleBuckets {
    std::array::from_fn(|slot| {
        let phase = crate::phase::SimplifyPhase::all()[slot];
        let phase_mask = phase.mask();
        let mut filtered = RuleBuckets::with_capacity(rules.len());

        for (&target_kind, bucket) in rules {
            let phase_bucket: Vec<_> = bucket
                .iter()
                .filter(|rule| {
                    !disabled_rules.contains(rule.name())
                        && rule.allowed_phases().contains(phase_mask)
                })
                .cloned()
                .collect();
            if !phase_bucket.is_empty() {
                filtered.insert(target_kind, phase_bucket);
            }
        }

        filtered
    })
}

fn build_phase_global_rules(
    global_rules: &[Arc<dyn Rule>],
    disabled_rules: &HashSet<String>,
) -> [Vec<Arc<dyn Rule>>; PHASE_SLOT_COUNT] {
    std::array::from_fn(|slot| {
        let phase = crate::phase::SimplifyPhase::all()[slot];
        let phase_mask = phase.mask();
        global_rules
            .iter()
            .filter(|rule| {
                !disabled_rules.contains(rule.name()) && rule.allowed_phases().contains(phase_mask)
            })
            .cloned()
            .collect()
    })
}

impl std::fmt::Debug for dyn Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rule({})", self.name())
    }
}
