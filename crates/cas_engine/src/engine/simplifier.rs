//! Simplifier struct definition, construction, and configuration.
//!
//! The `Simplifier` is the main entry point for the CAS simplification engine.
//! It holds the expression context, registered rules, and configuration state.

use crate::options::StepsMode;
use crate::profiler::RuleProfiler;
use crate::rule::Rule;
use crate::target_kind::TargetKind;
use cas_ast::{Context, ExprId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing::debug;

pub struct Simplifier {
    pub context: Context,
    pub(super) rules: HashMap<TargetKind, Vec<Arc<dyn Rule>>>,
    pub(super) global_rules: Vec<Arc<dyn Rule>>,
    /// Steps collection mode (On/Off/Compact)
    pub steps_mode: StepsMode,
    pub allow_numerical_verification: bool,
    pub debug_mode: bool,
    pub(super) disabled_rules: HashSet<String>,
    pub enable_polynomial_strategy: bool,
    pub profiler: RuleProfiler,
    /// Domain warnings from last simplify() call (side-channel for Off mode)
    pub(super) last_domain_warnings: Vec<(String, String)>,
    /// Blocked hints from last simplify() call (pedagogical hints for blocked Analytic conditions)
    pub(super) last_blocked_hints: Vec<crate::domain::BlockedHint>,
    /// Sticky root expression: when set, this is used instead of recalculating per-phase
    /// This preserves inherited requires across all phases (e.g., x≥0 from sqrt(x))
    pub(super) sticky_root_expr: Option<ExprId>,
    /// Sticky implicit domain: when set, this is propagated to all phases
    /// Computed from the original input, survives even after witnesses are consumed
    pub(super) sticky_implicit_domain: Option<crate::implicit_domain::ImplicitDomain>,
}

impl Default for Simplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplifier {
    /// Create a new Simplifier without rules registered.
    ///
    /// Usually you want `Simplifier::with_default_rules()` instead.
    pub fn new() -> Self {
        Self {
            context: Context::new(),
            rules: HashMap::new(),
            global_rules: Vec::new(),
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false), // Disabled by default
            last_domain_warnings: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
        }
    }

    /// Create a Simplifier with all default rules registered.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::Simplifier;
    /// use cas_parser::parse;
    ///
    /// let mut s = Simplifier::with_default_rules();
    /// let expr = parse("2 + 2", &mut s.context).unwrap();
    /// let (result, _steps) = s.simplify(expr);
    ///
    /// // 2+2 is simplified to 4
    /// use cas_ast::Expr;
    /// if let Expr::Number(n) = s.context.get(result) {
    ///     assert_eq!(n.to_string(), "4");
    /// }
    /// ```
    pub fn with_default_rules() -> Self {
        let mut s = Self::new();
        s.register_default_rules();
        s
    }

    /// Create a Simplifier with a pre-populated Context (for session restoration).
    /// Registers default rules on the provided context.
    pub fn with_context(context: Context) -> Self {
        let mut s = Self {
            context,
            rules: HashMap::new(),
            global_rules: Vec::new(),
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false),
            last_domain_warnings: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
        };
        s.register_default_rules();
        s
    }

    /// Create a simplifier based on evaluation options.
    /// This is the main entry point for context-aware simplification.
    ///
    /// NOTE: PrincipalBranchInverseTrigRule is now self-gated by inv_trig_policy().
    /// It is always registered but only applies when inv_trig == PrincipalValue.
    /// The branch_mode check below is for backward compatibility until fully migrated.
    pub fn with_profile(opts: &crate::options::EvalOptions) -> Self {
        use crate::options::ContextMode;

        let mut s = Self::with_default_rules();

        // Apply context mode rules (placeholder for future rule bundles)
        match opts.shared.context_mode {
            ContextMode::IntegratePrep => {
                crate::rules::integration::register_integration_prep(&mut s);
                // Disable angle expansion rules that destroy telescoping patterns
                // These rules transform cos(2x), cos(4x) before telescoping can match
                s.disabled_rules.insert("Double Angle Identity".to_string());
                s.disabled_rules.insert("Triple Angle Identity".to_string());
                s.disabled_rules
                    .insert("Recursive Trig Expansion".to_string());
            }
            ContextMode::Solve => {
                // Disable rules that introduce abs() which can cause issues with solver strategies
                // SimplifySqrtSquareRule: sqrt(x^2) -> |x|
                // SimplifySqrtOddPowerRule: x^(3/2) -> |x|·sqrt(x)
                s.disabled_rules
                    .insert("Simplify Square Root of Square".to_string());
                s.disabled_rules
                    .insert("Simplify Odd Half-Integer Power".to_string());
            }
            ContextMode::Auto | ContextMode::Standard => {
                // Standard rules only (already registered)
            }
        }

        s
    }

    /// Create a simplifier from a cached profile.
    /// This avoids rebuilding rules and is the preferred way when using ProfileCache.
    pub fn from_profile(profile: std::sync::Arc<crate::profile_cache::RuleProfile>) -> Self {
        Self {
            context: Context::new(),
            rules: profile.rules.clone(),
            global_rules: profile.global_rules.clone(),
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: profile.disabled_rules.clone(),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false),
            last_domain_warnings: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
        }
    }

    /// Backward-compatible getter: returns true if steps_mode is not Off
    #[inline]
    pub fn collect_steps(&self) -> bool {
        self.steps_mode != StepsMode::Off
    }

    /// Backward-compatible setter: sets steps_mode to On or Off
    pub fn set_collect_steps(&mut self, collect: bool) {
        self.steps_mode = if collect {
            StepsMode::On
        } else {
            StepsMode::Off
        };
    }

    /// Get the current steps collection mode
    #[inline]
    pub fn get_steps_mode(&self) -> StepsMode {
        self.steps_mode
    }

    /// Set the steps collection mode directly
    pub fn set_steps_mode(&mut self, mode: StepsMode) {
        self.steps_mode = mode;
    }

    /// Take and clear domain warnings from the last simplify() call.
    /// This is the side-channel to get warnings even in Off mode (when steps is empty).
    /// Warnings are deduplicated by (rule_name, message), preserving first-occurrence order.
    pub fn take_domain_warnings(&mut self) -> Vec<(String, String)> {
        let mut warnings = std::mem::take(&mut self.last_domain_warnings);
        // Dedup preserving first occurrence order
        let mut seen = std::collections::HashSet::new();
        warnings.retain(|w| seen.insert((w.0.clone(), w.1.clone())));
        warnings
    }

    /// Take and clear blocked hints from the last simplify() call.
    /// These are pedagogical hints when Generic mode blocks Analytic conditions.
    /// Hints are deduplicated by (rule, assumption_key), preserving first-occurrence order.
    pub fn take_blocked_hints(&mut self) -> Vec<crate::domain::BlockedHint> {
        let mut hints = std::mem::take(&mut self.last_blocked_hints);
        // Dedup by (rule, key) preserving first occurrence order
        let mut seen = std::collections::HashSet::new();
        hints.retain(|h| seen.insert((h.rule.clone(), h.key.clone())));
        hints
    }

    /// Extend blocked hints from an external source (used for context transfer).
    /// Hints will be deduplicated when take_blocked_hints is called.
    pub fn extend_blocked_hints(&mut self, hints: Vec<crate::domain::BlockedHint>) {
        self.last_blocked_hints.extend(hints);
    }

    /// Set sticky implicit domain from the original input expression.
    /// This domain will be used for all phases instead of recalculating per-phase.
    /// Call this at the start of a simplification pipeline to preserve inherited requires.
    pub fn set_sticky_implicit_domain(
        &mut self,
        root: ExprId,
        value_domain: crate::semantics::ValueDomain,
    ) {
        use crate::implicit_domain::infer_implicit_domain;
        self.sticky_root_expr = Some(root);
        self.sticky_implicit_domain =
            Some(infer_implicit_domain(&self.context, root, value_domain));
    }

    /// Clear sticky implicit domain (call after pipeline completes).
    pub fn clear_sticky_implicit_domain(&mut self) {
        self.sticky_root_expr = None;
        self.sticky_implicit_domain = None;
    }

    /// Get the sticky implicit domain, if set.
    pub fn sticky_implicit_domain(&self) -> Option<&crate::implicit_domain::ImplicitDomain> {
        self.sticky_implicit_domain.as_ref()
    }

    /// Get the sticky root expression, if set.
    pub fn sticky_root_expr(&self) -> Option<ExprId> {
        self.sticky_root_expr
    }

    pub fn enable_debug(&mut self) {
        self.debug_mode = true;
    }

    pub fn disable_debug(&mut self) {
        self.debug_mode = false;
    }

    pub fn debug(&self, msg: &str) {
        // Use tracing for structured logging.
        // We still check debug_mode to allow per-instance toggling if needed,
        // but ideally this should be controlled by RUST_LOG.
        // For now, let's log if EITHER debug_mode is on OR tracing is enabled at debug level.
        // Actually, let's just delegate to tracing. The subscriber will filter.
        debug!("{}", msg);
    }

    pub fn disable_rule(&mut self, rule_name: &str) {
        self.disabled_rules.insert(rule_name.to_string());
    }

    pub fn enable_rule(&mut self, rule_name: &str) {
        self.disabled_rules.remove(rule_name);
    }

    pub fn register_default_rules(&mut self) {
        use crate::rules::*;

        arithmetic::register(self);
        infinity::register(self); // Infinity arithmetic (∞ absorption, indeterminates)
        canonicalization::register(self);
        constants::register(self); // Algebraic constants (phi)
        exponents::register(self);
        logarithms::register(self);

        // CRITICAL ORDER: Compositions must resolve BEFORE conversions and expansions
        // Otherwise tan(arctan(x)) would become sin(arctan(x))/cos(arctan(x))
        trigonometry::register(self); // Base trig functions
        inverse_trig::register(self); // Compositions like tan(arctan(x)) → x

        // Weierstrass / inverse-trig bridge identities
        // sin(2·atan(t)) → 2t/(1+t²), cos(2·atan(t)) → (1−t²)/(1+t²), etc.
        trig_inverse_compositions::register(self);

        // Expand trig(inverse_trig) to algebraic forms AFTER compositions
        trig_inverse_expansion::register(self);

        hyperbolic::register(self); // Hyperbolic functions
        reciprocal_trig::register(self); // Reciprocal trig identities

        // Sophisticated context-aware canonicalization
        // Only converts in beneficial patterns (Pythagorean, mixed fractions)
        // Preserves compositions like tan(arctan(x))
        trig_canonicalization::register(self);

        // CRITICAL: matrix_ops MUST come before polynomial and grouping
        // so that MatrixAddRule and MatrixSubRule can handle matrix addition/subtraction
        // before CombineLikeTermsRule tries to collect them
        matrix_ops::register(self);
        polynomial::register(self);
        algebra::register(self);
        calculus::register(self);
        functions::register(self);
        grouping::register(self);
        number_theory::register(self);

        // Complex number rules (i^n → {1, i, -1, -i})
        // Registered unconditionally - only fires when i^n patterns exist
        complex::register(self);

        // P0: Validate no duplicate rule names (debug only)
        #[cfg(debug_assertions)]
        self.assert_unique_rule_names();
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        let rule_rc: Arc<dyn Rule> = rule.into();

        if let Some(targets) = rule_rc.target_types() {
            for target in targets.iter() {
                let vec = self.rules.entry(target).or_default();
                // Insert maintaining priority order (higher first)
                // For equal priority, preserve insertion order (append after same-priority rules)
                let priority = rule_rc.priority();
                let pos = vec
                    .iter()
                    .position(|r| r.priority() < priority)
                    .unwrap_or(vec.len());
                vec.insert(pos, rule_rc.clone());
            }
        } else {
            // Insert into global_rules maintaining priority order
            let priority = rule_rc.priority();
            let pos = self
                .global_rules
                .iter()
                .position(|r| r.priority() < priority)
                .unwrap_or(self.global_rules.len());
            self.global_rules.insert(pos, rule_rc);
        }
    }

    pub fn get_all_rule_names(&self) -> Vec<String> {
        let mut names = HashSet::new();

        for rule in &self.global_rules {
            names.insert(rule.name().to_string());
        }

        for rules in self.rules.values() {
            for rule in rules {
                names.insert(rule.name().to_string());
            }
        }

        let mut sorted_names: Vec<String> = names.into_iter().collect();
        sorted_names.sort();
        sorted_names
    }

    /// Panics if there are duplicate rule names (debug builds only).
    /// This prevents accidental rule name collisions which can cause
    /// confusing precedence behavior.
    /// Note: Rules with multiple target_types appear in multiple buckets
    /// but are the same rule instance, so we deduplicate by Arc pointer first.
    #[cfg(debug_assertions)]
    #[allow(clippy::panic)] // Intentional: debug-only invariant enforcement
    pub fn assert_unique_rule_names(&self) {
        use std::collections::HashSet;

        // Collect all rule names (deduplicating by name as we go)
        // A rule appearing in multiple buckets with the same name is OK
        // (e.g., LogContractionRule targets both Add and Sub)
        let mut seen_names: HashSet<&str> = HashSet::new();

        for rule in &self.global_rules {
            let name = rule.name();
            if !seen_names.insert(name) {
                // Same name seen again - but might be same rule in different bucket
                // Only panic if it's truly a different rule (checked below)
            }
        }

        // For typed rules, same rule may appear in multiple buckets
        // We need to check if duplicate names come from the SAME Arc or different ones
        let mut name_to_first_arc: std::collections::HashMap<&str, *const ()> =
            std::collections::HashMap::new();

        for rule in &self.global_rules {
            let name = rule.name();
            let ptr = Arc::as_ptr(rule).cast::<()>();
            if let Some(&existing_ptr) = name_to_first_arc.get(name) {
                if !std::ptr::eq(ptr, existing_ptr) {
                    panic!(
                        "Duplicate rule name detected: '{}'. Each rule must have a unique name.",
                        name
                    );
                }
            } else {
                name_to_first_arc.insert(name, ptr);
            }
        }

        for rules in self.rules.values() {
            for rule in rules {
                let name = rule.name();
                let ptr = Arc::as_ptr(rule).cast::<()>();
                if let Some(&existing_ptr) = name_to_first_arc.get(name) {
                    if !std::ptr::eq(ptr, existing_ptr) {
                        panic!(
                            "Duplicate rule name detected: '{}'. Each rule must have a unique name.",
                            name
                        );
                    }
                } else {
                    name_to_first_arc.insert(name, ptr);
                }
            }
        }
    }

    /// Get a clone of the rules map (for profile caching).
    pub fn get_rules_clone(&self) -> HashMap<TargetKind, Vec<Arc<dyn Rule>>> {
        self.rules.clone()
    }

    /// Get a clone of the global rules (for profile caching).
    pub fn get_global_rules_clone(&self) -> Vec<Arc<dyn Rule>> {
        self.global_rules.clone()
    }

    /// Get a clone of the disabled rules set (for profile caching).
    pub fn get_disabled_rules_clone(&self) -> HashSet<String> {
        self.disabled_rules.clone()
    }
}
