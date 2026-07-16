//! Simplifier struct definition, construction, and configuration.
//!
//! The `Simplifier` is the main entry point for the CAS simplification engine.
//! It holds the expression context, registered rules, and configuration state.

use crate::options::StepsMode;
use crate::profiler::RuleProfiler;
use crate::rule::Rule;
use crate::target_kind::TargetKind;
use crate::Step;
use cas_ast::{Context, ExprId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing::debug;

/// The default rule set: target-bucketed rules plus the untargeted global rules. Built once
/// per thread and cloned per simplifier (see [`Simplifier::register_default_rules`]).
type DefaultRuleRegistry = (HashMap<TargetKind, Vec<Arc<dyn Rule>>>, Vec<Arc<dyn Rule>>);

pub struct Simplifier {
    pub context: Context,
    pub(super) rules: HashMap<TargetKind, Vec<Arc<dyn Rule>>>,
    pub(super) global_rules: Vec<Arc<dyn Rule>>,
    pub(super) use_profile_fast_path: bool,
    /// Steps collection mode (On/Off/Compact)
    pub steps_mode: StepsMode,
    pub allow_numerical_verification: bool,
    pub debug_mode: bool,
    pub(super) disabled_rules: HashSet<String>,
    pub(super) profile: Option<std::sync::Arc<crate::profile_cache::RuleProfile>>,
    pub enable_polynomial_strategy: bool,
    pub profiler: RuleProfiler,
    /// Domain warnings from last simplify() call (side-channel for Off mode)
    pub(super) last_domain_warnings: Vec<(String, String)>,
    /// Required conditions introduced by rewrites in the last simplify() call.
    pub(super) last_required_conditions: Vec<crate::ImplicitCondition>,
    /// Blocked hints from last simplify() call (pedagogical hints for blocked Analytic conditions)
    pub(super) last_blocked_hints: Vec<crate::BlockedHint>,
    /// Sticky root expression: when set, this is used instead of recalculating per-phase
    /// This preserves inherited requires across all phases (e.g., x≥0 from sqrt(x))
    pub(super) sticky_root_expr: Option<ExprId>,
    /// Sticky implicit domain: when set, this is propagated to all phases
    /// Computed from the original input, survives even after witnesses are consumed
    pub(super) sticky_implicit_domain: Option<crate::ImplicitDomain>,
    /// Optional observer that receives rewrite events during simplification.
    pub(super) step_listener: Option<Box<dyn crate::StepListener>>,
    /// P16: memo for repeated plain `simplify()` calls inside a solve scope.
    /// The solver's handler chain re-simplifies the same interned expression
    /// 4-8x per solve (measured up to 85% redundant calls). Keyed by
    /// (input, sticky root) — sticky state changes simplify semantics — and
    /// replaying the last_* side channels so a hit is observably identical
    /// to a fresh call. Active only while `solve_memo_depth > 0`.
    pub(super) solve_memo: HashMap<(ExprId, Option<ExprId>), SolveMemoEntry>,
    pub(super) solve_memo_depth: u32,
    /// Sticky value domain consumed by plain `simplify()` (which otherwise runs
    /// with `SimplifyOptions::default()` = RealOnly). The solve backend sets it
    /// from its options on entry (save/restore) so solver-internal
    /// simplification folds complex forms (`√(-3) → i·√3`) under
    /// `--value-domain complex`. Default `RealOnly` keeps every other caller
    /// byte-identical.
    pub(super) sticky_value_domain: crate::semantics::ValueDomain,
}

/// Cached result of one plain `simplify()` call (P16 solve-scope memo).
/// Captures the return value AND the per-call side channels so replaying a
/// hit leaves the simplifier in the same observable state as a fresh call.
pub(super) struct SolveMemoEntry {
    pub(super) out: ExprId,
    pub(super) steps: Vec<Step>,
    pub(super) domain_warnings: Vec<(String, String)>,
    pub(super) required_conditions: Vec<crate::ImplicitCondition>,
    pub(super) blocked_hints: Vec<crate::BlockedHint>,
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
        crate::rules::arithmetic::clear_solve_prep_gate_memos();
        crate::polynomial_identity_support::clear_identity_zero_negative_memo();
        Self {
            context: Context::new(),
            rules: HashMap::new(),
            global_rules: Vec::new(),
            use_profile_fast_path: false,
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            profile: None,
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false), // Disabled by default
            last_domain_warnings: Vec::new(),
            last_required_conditions: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
            step_listener: None,
            solve_memo: HashMap::new(),
            solve_memo_depth: 0,
            sticky_value_domain: crate::semantics::ValueDomain::RealOnly,
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
            use_profile_fast_path: false,
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            profile: None,
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false),
            last_domain_warnings: Vec::new(),
            last_required_conditions: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
            step_listener: None,
            solve_memo: HashMap::new(),
            solve_memo_depth: 0,
            sticky_value_domain: crate::semantics::ValueDomain::RealOnly,
        };
        s.register_default_rules();
        s
    }

    /// Create a simplifier based on evaluation options.
    /// This is the main entry point for context-aware simplification.
    ///
    /// NOTE: PrincipalBranchInverseTrigRule is self-gated by inv_trig_policy().
    /// It is always registered but only applies when inv_trig == PrincipalValue.
    /// The legacy BranchMode axis no longer affects rule selection (the CLI
    /// --branch flag is a deprecated alias of --inv-trig).
    pub fn with_profile(opts: &crate::options::EvalOptions) -> Self {
        use crate::options::ContextMode;

        let mut s = Self::with_default_rules();
        s.steps_mode = opts.steps_mode;

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
                // Quintuple expansion mutilates cos(5x)/sin(5x) into power
                // polynomials before product-to-sum can match (and before
                // the linear-argument rule can give sin(5x)/5).
                s.disabled_rules
                    .insert("Quintuple Angle Identity".to_string());
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
        Self::from_profile_with_context(profile, Context::new())
    }

    /// Create a simplifier from a cached profile and an existing Context.
    /// This avoids building a throwaway empty Context before swapping in parsed input.
    pub fn from_profile_with_context(
        profile: std::sync::Arc<crate::profile_cache::RuleProfile>,
        context: Context,
    ) -> Self {
        Self {
            context,
            rules: HashMap::new(),
            global_rules: Vec::new(),
            use_profile_fast_path: true,
            steps_mode: StepsMode::On,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            profile: Some(profile),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false),
            last_domain_warnings: Vec::new(),
            last_required_conditions: Vec::new(),
            last_blocked_hints: Vec::new(),
            sticky_root_expr: None,
            sticky_implicit_domain: None,
            step_listener: None,
            solve_memo: HashMap::new(),
            solve_memo_depth: 0,
            sticky_value_domain: crate::semantics::ValueDomain::RealOnly,
        }
    }

    #[inline]
    pub(super) fn using_profile_fast_path(&self) -> bool {
        self.use_profile_fast_path && self.profile.is_some()
    }

    fn materialize_profile_rules(&mut self) {
        if !self.using_profile_fast_path() {
            return;
        }

        if let Some(profile) = self.profile.as_ref() {
            self.rules = profile.rules.clone();
            self.global_rules = profile.global_rules.clone();
            self.disabled_rules = profile.disabled_rules.clone();
        }
        self.use_profile_fast_path = false;
    }

    fn active_rules(&self) -> &HashMap<TargetKind, Vec<Arc<dyn Rule>>> {
        if self.using_profile_fast_path() {
            if let Some(profile) = self.profile.as_ref() {
                return &profile.rules;
            }
        }
        &self.rules
    }

    fn active_global_rules(&self) -> &[Arc<dyn Rule>] {
        if self.using_profile_fast_path() {
            if let Some(profile) = self.profile.as_ref() {
                return &profile.global_rules;
            }
        }
        &self.global_rules
    }

    pub(super) fn active_disabled_rules(&self) -> &HashSet<String> {
        if self.using_profile_fast_path() {
            if let Some(profile) = self.profile.as_ref() {
                return &profile.disabled_rules;
            }
        }
        &self.disabled_rules
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

    #[inline]
    pub(crate) fn has_step_listener(&self) -> bool {
        self.step_listener.is_some()
    }

    /// Set the steps collection mode directly
    pub fn set_steps_mode(&mut self, mode: StepsMode) {
        self.steps_mode = mode;
    }

    /// Install or clear an optional listener for engine rewrite events.
    pub fn set_step_listener(&mut self, listener: Option<Box<dyn crate::StepListener>>) {
        self.step_listener = listener;
    }

    /// Replace the current engine event listener, returning the previous one.
    pub fn replace_step_listener(
        &mut self,
        listener: Option<Box<dyn crate::StepListener>>,
    ) -> Option<Box<dyn crate::StepListener>> {
        std::mem::replace(&mut self.step_listener, listener)
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

    /// Take and clear rewrite-introduced required conditions from the last simplify() call.
    pub fn take_required_conditions(&mut self) -> Vec<crate::ImplicitCondition> {
        let mut conditions = std::mem::take(&mut self.last_required_conditions);
        let mut seen = std::collections::HashSet::new();
        conditions.retain(|condition| seen.insert(condition.clone()));
        conditions
    }

    /// Record required conditions discovered by orchestrator shortcuts.
    pub(crate) fn extend_required_conditions<I>(&mut self, conditions: I)
    where
        I: IntoIterator<Item = crate::ImplicitCondition>,
    {
        self.last_required_conditions.extend(conditions);
    }

    /// Take and clear blocked hints from the last simplify() call.
    /// These are pedagogical hints when Generic mode blocks Analytic conditions.
    /// Hints are deduplicated by (rule, assumption_key), preserving first-occurrence order.
    pub fn take_blocked_hints(&mut self) -> Vec<crate::BlockedHint> {
        let mut hints = std::mem::take(&mut self.last_blocked_hints);
        // Dedup by (rule, key) preserving first occurrence order
        let mut seen = std::collections::HashSet::new();
        hints.retain(|h| seen.insert((h.rule.clone(), h.key.clone())));
        hints
    }

    /// Extend blocked hints from an external source (used for context transfer).
    /// Hints will be deduplicated when take_blocked_hints is called.
    pub fn extend_blocked_hints(&mut self, hints: Vec<crate::BlockedHint>) {
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
        use crate::infer_implicit_domain;
        self.sticky_root_expr = Some(root);
        self.sticky_implicit_domain =
            Some(infer_implicit_domain(&self.context, root, value_domain));
        // Sticky state changes simplify semantics: drop any solve-scope memo.
        self.solve_memo.clear();
    }

    /// Clear sticky implicit domain (call after pipeline completes).
    pub fn clear_sticky_implicit_domain(&mut self) {
        self.sticky_root_expr = None;
        self.sticky_implicit_domain = None;
        self.solve_memo.clear();
    }

    /// Set the sticky value domain consumed by plain `simplify()` calls and
    /// return the previous value (save/restore discipline for scoped callers
    /// like the solve backend). Sticky state changes simplify semantics, so a
    /// domain change drops the solve-scope memo.
    pub fn set_sticky_value_domain(
        &mut self,
        value_domain: crate::semantics::ValueDomain,
    ) -> crate::semantics::ValueDomain {
        let previous = self.sticky_value_domain;
        if previous != value_domain {
            self.sticky_value_domain = value_domain;
            self.solve_memo.clear();
        }
        previous
    }

    /// P16: enter a solve-scoped simplify memo region (re-entrant).
    /// While active, repeated plain `simplify()` calls on the same interned
    /// expression replay the cached result and side channels.
    pub fn begin_solve_simplify_memo(&mut self) {
        self.solve_memo_depth += 1;
    }

    /// P16: leave the solve-scoped memo region; the cache drops when the
    /// outermost scope exits.
    pub fn end_solve_simplify_memo(&mut self) {
        self.solve_memo_depth = self.solve_memo_depth.saturating_sub(1);
        if self.solve_memo_depth == 0 {
            self.solve_memo.clear();
        }
    }

    /// Get the sticky implicit domain, if set.
    pub fn sticky_implicit_domain(&self) -> Option<&crate::ImplicitDomain> {
        self.sticky_implicit_domain.as_ref()
    }

    /// Get the sticky root expression, if set.
    pub fn sticky_root_expr(&self) -> Option<ExprId> {
        self.sticky_root_expr
    }

    pub fn enable_debug(&mut self) {
        self.debug_mode = true;
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
        self.materialize_profile_rules();
        self.disabled_rules.insert(rule_name.to_string());
        // Rule toggling changes simplify semantics: any solve-scoped memo
        // entries computed under the previous rule set are stale (P16).
        self.solve_memo.clear();
    }

    pub fn enable_rule(&mut self, rule_name: &str) {
        self.materialize_profile_rules();
        self.disabled_rules.remove(rule_name);
        self.solve_memo.clear();
    }

    /// Whether a rule is currently disabled by name.
    pub fn is_rule_disabled(&self, rule_name: &str) -> bool {
        self.disabled_rules.contains(rule_name)
    }

    /// Populate this simplifier with the default rule set.
    ///
    /// The default registry (~340 stateless `Arc<dyn Rule>` in priority order) is
    /// identical for every simplifier, so it is built ONCE per thread and cloned here
    /// (cloning the bucket `HashMap`/`Vec`s bumps `Arc` refcounts; it does NOT reconstruct
    /// the rule objects or re-run the priority-ordered inserts). This is the hot path for
    /// the ~57 `run_default_simplify` call sites that used to rebuild all ~340 rules per
    /// call (P15 of the saneamiento audit). `thread_local` (not a global `LazyLock`) because
    /// `dyn Rule` is not `Sync`; per-thread isolation also keeps rule objects unshared across
    /// threads.
    pub fn register_default_rules(&mut self) {
        thread_local! {
            static DEFAULT_RULES: DefaultRuleRegistry = {
                let mut builder = Simplifier::new();
                builder.register_default_rules_uncached();
                (builder.rules, builder.global_rules)
            };
        }
        let (rules, global_rules) = DEFAULT_RULES.with(|r| r.clone());
        self.rules = rules;
        self.global_rules = global_rules;
    }

    fn register_default_rules_uncached(&mut self) {
        use crate::rules::*;

        arithmetic::register(self);
        infinity::register(self); // Infinity arithmetic (∞ absorption, indeterminates)
                                  // ─── Canonicalization tier ───────────────────────────────────────
                                  // ORDER CONTRACT: canonicalization → rational_canonicalization
                                  //   1. canonicalization: Sub→Add(Neg), sort Add/Mul, roots→Pow, signs
                                  //   2. rational_canonicalization: Div(p,q)→Number(p/q), nested Pow fold
                                  //      (MUST come before any rule that compares numeric exponents/coefficients)
                                  //
                                  // ⚠ "Sub is NOT stable": CanonicalizeNegationRule converts Sub(a,b)
                                  //    to Add(a, Neg(b)) very early. Any future rule that depends on
                                  //    matching Sub nodes must EITHER:
                                  //      a) be registered BEFORE canonicalization, or
                                  //      b) also match Add(x, Neg(y)) patterns, or
                                  //      c) be an equation-level operation in the solver pipeline
                                  //         (see rules/cancel_common_terms.rs for this pattern).
                                  //
                                  // cancel_common_terms is NOT registered here — it provides equation-level
                                  // primitives called by the solver (see solve_core.rs), not simplifier rules.
        canonicalization::register(self);
        rational_canonicalization::register(self);
        constants::register(self); // Algebraic constants (phi)
        exponents::register(self);
        logarithms::register(self);

        // CRITICAL ORDER: Compositions must resolve BEFORE conversions and expansions
        // Otherwise tan(arctan(x)) would become sin(arctan(x))/cos(arctan(x))
        trigonometry::register(self); // Base trig functions
        inverse_trig::register(self); // Compositions like tan(arctan(x)) → x

        // Generalized n-angle inverse-trig compositions (Weierstrass / Chebyshev / recurrence)
        // Handles sin/cos/tan(n·arctan(t)) for n=1..10 via (1+it)^n recurrence.
        inv_trig_n_angle::register(self);

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
        self.materialize_profile_rules();
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

    /// Panics if there are duplicate rule names (debug builds only).
    /// This prevents accidental rule name collisions which can cause
    /// confusing precedence behavior.
    /// Note: Rules with multiple target_types appear in multiple buckets
    /// but are the same rule instance, so we deduplicate by Arc pointer first.
    #[cfg(debug_assertions)]
    #[allow(clippy::panic)] // Intentional: debug-only invariant enforcement
    pub fn assert_unique_rule_names(&self) {
        use std::collections::HashSet;
        let global_rules = self.active_global_rules();
        let rules = self.active_rules();

        // Collect all rule names (deduplicating by name as we go)
        // A rule appearing in multiple buckets with the same name is OK
        // (e.g., LogContractionRule targets both Add and Sub)
        let mut seen_names: HashSet<&str> = HashSet::new();

        for rule in global_rules {
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

        for rule in global_rules {
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

        for rules in rules.values() {
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
        self.active_rules().clone()
    }

    /// Get a clone of the global rules (for profile caching).
    pub fn get_global_rules_clone(&self) -> Vec<Arc<dyn Rule>> {
        self.active_global_rules().to_vec()
    }

    /// Get a clone of the disabled rules set (for profile caching).
    pub fn get_disabled_rules_clone(&self) -> HashSet<String> {
        self.active_disabled_rules().clone()
    }
}
