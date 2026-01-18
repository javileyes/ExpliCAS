use crate::canonical_forms::normalize_core;
use crate::options::StepsMode;
use crate::profiler::RuleProfiler;
use crate::rule::Rule;
use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing::debug;

// =============================================================================
// EquivalenceResult: Tri-state result for equiv command
// =============================================================================
//
// V2.14.45: Proper equivalence checking with domain awareness.
// Returns conditional true when rules with EquivalenceUnderIntroducedRequires
// are used or when domain conditions are introduced.
// =============================================================================

/// Result of equivalence checking between two expressions.
#[derive(Debug, Clone)]
pub enum EquivalenceResult {
    /// A ≡ B unconditionally (no domain assumptions needed)
    True,
    /// A ≡ B under specified conditions (domain restrictions)
    ConditionalTrue {
        /// Requires conditions introduced during simplification
        requires: Vec<String>,
    },
    /// A ≢ B (found counterexample or proved non-equivalent)
    False,
    /// Cannot determine (no proof either way)
    Unknown,
}

impl EquivalenceResult {
    /// Returns true if the result indicates equivalence (True or ConditionalTrue)
    pub fn is_equivalent(&self) -> bool {
        matches!(
            self,
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        )
    }
}

// =============================================================================
// HoldAll function semantics
// =============================================================================

/// Returns true if a function has HoldAll semantics, meaning its arguments
/// should NOT be simplified before the function rule is applied.
/// This is crucial for functions like poly_gcd that need to see the raw
/// multiplicative structure of their arguments.
/// Also includes '__hold' which is an internal invisible barrier.
fn is_hold_all_function(name: &str) -> bool {
    matches!(name, "poly_gcd" | "pgcd" | "__hold")
}

/// Unwrap top-level __hold() wrapper after simplification.
/// This is called at the end of eval/simplify so the user sees clean results
/// without the internal barrier visible.
fn unwrap_hold_top(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_hold(ctx, expr)
}

/// Re-export strip_all_holds from cas_ast for use by rules.
///
/// This is the CANONICAL implementation - see cas_ast::hold for the contract.
/// Do NOT duplicate this function elsewhere.
pub fn strip_all_holds(ctx: &mut Context, expr: ExprId) -> ExprId {
    cas_ast::hold::strip_all_holds(ctx, expr)
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
/// Returns new ExprId if substitution occurred, otherwise returns original root.
pub fn substitute_expr_by_id(
    context: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    if root == target {
        return replacement;
    }

    let expr = context.get(root).clone();
    match expr {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Add(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Sub(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Mul(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Div(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Pow(b, e) => {
            let new_b = substitute_expr_by_id(context, b, target, replacement);
            let new_e = substitute_expr_by_id(context, e, target, replacement);
            if new_b != b || new_e != e {
                context.add(Expr::Pow(new_b, new_e))
            } else {
                root
            }
        }
        Expr::Neg(inner) => {
            let new_inner = substitute_expr_by_id(context, inner, target, replacement);
            if new_inner != inner {
                context.add(Expr::Neg(new_inner))
            } else {
                root
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args.iter() {
                let new_arg = substitute_expr_by_id(context, *arg, target, replacement);
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                context.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::new();
            let mut changed = false;
            for elem in data.iter() {
                let new_elem = substitute_expr_by_id(context, *elem, target, replacement);
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                context.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                root
            }
        }
        _ => root,
    }
}

pub struct Simplifier {
    pub context: Context,
    rules: HashMap<String, Vec<Arc<dyn Rule>>>,
    global_rules: Vec<Arc<dyn Rule>>,
    /// Steps collection mode (On/Off/Compact)
    pub steps_mode: StepsMode,
    pub allow_numerical_verification: bool,
    pub debug_mode: bool,
    disabled_rules: HashSet<String>,
    pub enable_polynomial_strategy: bool,
    pub profiler: RuleProfiler,
    /// Domain warnings from last simplify() call (side-channel for Off mode)
    last_domain_warnings: Vec<(String, String)>,
    /// Blocked hints from last simplify() call (pedagogical hints for blocked Analytic conditions)
    last_blocked_hints: Vec<crate::domain::BlockedHint>,
    /// Sticky root expression: when set, this is used instead of recalculating per-phase
    /// This preserves inherited requires across all phases (e.g., x≥0 from sqrt(x))
    sticky_root_expr: Option<ExprId>,
    /// Sticky implicit domain: when set, this is propagated to all phases
    /// Computed from the original input, survives even after witnesses are consumed
    sticky_implicit_domain: Option<crate::implicit_domain::ImplicitDomain>,
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
        match opts.context_mode {
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
        exponents::register(self);
        logarithms::register(self);

        // CRITICAL ORDER: Compositions must resolve BEFORE conversions and expansions
        // Otherwise tan(arctan(x)) would become sin(arctan(x))/cos(arctan(x))
        trigonometry::register(self); // Base trig functions
        inverse_trig::register(self); // Compositions like tan(arctan(x)) → x

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
            for target in targets {
                let vec = self.rules.entry(target.to_string()).or_default();
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
    pub fn get_rules_clone(&self) -> HashMap<String, Vec<Arc<dyn Rule>>> {
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

    pub fn local_simplify(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
    ) -> (ExprId, Vec<Step>) {
        // Default to Core phase for local_simplify (safe, non-expanding)
        self.local_simplify_with_phase(expr_id, pattern_marks, crate::phase::SimplifyPhase::Core)
    }

    pub fn local_simplify_with_phase(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
    ) -> (ExprId, Vec<Step>) {
        // Create initial ParentContext with pattern marks
        let initial_parent_ctx =
            crate::parent_context::ParentContext::with_marks(pattern_marks.clone());

        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules: &self.rules,
            global_rules: &self.global_rules,
            disabled_rules: &self.disabled_rules,
            steps_mode: self.steps_mode,
            steps: Vec::new(),
            domain_warnings: Vec::new(),
            cache: HashMap::new(),
            current_path: Vec::new(),
            profiler: &mut self.profiler,
            pattern_marks: pattern_marks.clone(),
            initial_parent_ctx,
            root_expr: expr_id,
            current_phase: phase,
            cycle_detector: None,
            cycle_phase: None,
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            blocked_rules: std::collections::HashSet::new(),
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
            // Budget tracking (unified)
            rewrite_count: 0,
            nodes_snap: 0,
            budget_op: crate::budget::Operation::SimplifyCore,
            stop_reason: None,
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
        };

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
        // Copy domain_warnings to self (survives even in Off mode)
        self.last_domain_warnings = std::mem::take(&mut local_transformer.domain_warnings);
        drop(local_transformer);

        (new_expr, steps)
    }

    pub fn simplify(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::default())
    }

    /// Simplify with custom options controlling phases and policies.
    pub fn simplify_with_options(
        &mut self,
        expr_id: ExprId,
        options: crate::phase::SimplifyOptions,
    ) -> (ExprId, Vec<Step>) {
        let (result, steps, _stats) = self.simplify_with_stats(expr_id, options);
        // Unwrap any top-level hold() wrapper so user sees clean result
        let unwrapped = unwrap_hold_top(&self.context, result);
        (unwrapped, steps)
    }

    /// Simplify with options and return pipeline statistics for diagnostics.
    pub fn simplify_with_stats(
        &mut self,
        expr_id: ExprId,
        options: crate::phase::SimplifyOptions,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        // Clear blocked hints from previous simplifications
        crate::domain::clear_blocked_hints();

        let mut orchestrator = crate::orchestrator::Orchestrator::new();
        orchestrator.enable_polynomial_strategy = self.enable_polynomial_strategy;
        orchestrator.options = options;
        orchestrator.options.collect_steps = self.collect_steps();
        let result = orchestrator.simplify_pipeline(expr_id, self);

        // Collect blocked hints from thread-local to Simplifier field
        self.last_blocked_hints = crate::domain::take_blocked_hints();

        result
    }

    /// Expand without rationalization.
    /// After expansion, recursively strips all __hold() wrappers so user sees clean result.
    pub fn expand(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        let (result, steps) =
            self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::for_expand());
        // Strip ALL nested __hold wrappers (not just top-level)
        let clean_result = strip_all_holds(&mut self.context, result);
        (clean_result, steps)
    }

    /// Simplify for solver pre-pass: only safe rules (SolveSafety::Always).
    /// Blocks rules that could corrupt solution sets (those requiring assumptions).
    /// Steps are not collected (pre-pass is invisible to user).
    pub fn simplify_for_solve(&mut self, expr_id: ExprId) -> ExprId {
        let (result, _steps) =
            self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::for_solve_prepass());
        result
    }

    pub fn apply_rules_loop(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        // Default to Transform phase (allows all rules including distribution)
        self.apply_rules_loop_with_phase(
            expr_id,
            pattern_marks,
            crate::phase::SimplifyPhase::Transform,
        )
    }

    pub fn apply_rules_loop_with_phase(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        // Default: not in expand mode, no auto-expand, Generic domain mode, Strict inv_trig, RealOnly value_domain, Simplify goal
        self.apply_rules_loop_with_phase_and_mode(
            expr_id,
            pattern_marks,
            phase,
            false,
            false,
            crate::phase::ExpandBudget::default(),
            crate::domain::DomainMode::default(),
            crate::semantics::InverseTrigPolicy::default(),
            crate::semantics::ValueDomain::default(),
            crate::semantics::NormalFormGoal::default(),
            crate::solve_safety::SimplifyPurpose::default(),
            crate::options::ContextMode::default(),
            crate::options::AutoExpandBinomials::Off, // autoexpand_binomials: On by default
            crate::options::HeuristicPoly::On,       // heuristic_poly: On by default
        )
    }

    /// Apply rules loop with explicit expand_mode and auto_expand control.
    /// Returns PassStats for the caller to charge the Budget.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rules_loop_with_phase_and_mode(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
        expand_mode: bool,
        auto_expand: bool,
        expand_budget: crate::phase::ExpandBudget,
        domain_mode: crate::domain::DomainMode,
        inv_trig: crate::semantics::InverseTrigPolicy,
        value_domain: crate::semantics::ValueDomain,
        goal: crate::semantics::NormalFormGoal,
        simplify_purpose: crate::solve_safety::SimplifyPurpose,
        context_mode: crate::options::ContextMode,
        autoexpand_binomials: crate::options::AutoExpandBinomials, // V2.15.8
        heuristic_poly: crate::options::HeuristicPoly,             // V2.15.9
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        let rules = &self.rules;
        let global_rules = &self.global_rules;
        let steps_mode = self.steps_mode;

        // Create initial ParentContext with pattern marks, expand_mode, auto-expand, domain_mode, inv_trig, value_domain, goal, simplify_purpose, and context_mode
        // V2.15: Reset domain inference call counter for regression testing
        crate::implicit_domain::infer_domain_calls_reset();

        // V2.15.8: Use sticky implicit domain if set (from original input), otherwise calculate from current expr
        // This preserves inherited requires (e.g., x≥0 from sqrt) across all phases
        let initial_parent_ctx = if let Some(sticky_domain) = &self.sticky_implicit_domain {
            let sticky_root = self.sticky_root_expr.unwrap_or(expr_id);
            crate::parent_context::ParentContext::with_expand_mode(
                pattern_marks.clone(),
                expand_mode,
            )
            .with_auto_expand_flag(
                auto_expand,
                if auto_expand {
                    Some(expand_budget)
                } else {
                    None
                },
            )
            .with_domain_mode(domain_mode)
            .with_inv_trig(inv_trig)
            .with_value_domain(value_domain)
            .with_goal(goal)
            .with_simplify_purpose(simplify_purpose)
            .with_context_mode(context_mode)
            .with_root_expr_only(sticky_root)
            .with_implicit_domain(Some(sticky_domain.clone()))
            .with_autoexpand_binomials(autoexpand_binomials)
            .with_heuristic_poly(heuristic_poly)
        } else {
            crate::parent_context::ParentContext::with_expand_mode(
                pattern_marks.clone(),
                expand_mode,
            )
            .with_auto_expand_flag(
                auto_expand,
                if auto_expand {
                    Some(expand_budget)
                } else {
                    None
                },
            )
            .with_domain_mode(domain_mode)
            .with_inv_trig(inv_trig)
            .with_value_domain(value_domain)
            .with_goal(goal)
            .with_simplify_purpose(simplify_purpose)
            .with_context_mode(context_mode)
            .with_root_expr(&self.context, expr_id)
            .with_autoexpand_binomials(autoexpand_binomials)
            .with_heuristic_poly(heuristic_poly)
        };

        // Capture nodes_created BEFORE creating transformer (can't access while borrowed)
        let nodes_snap = self.context.stats().nodes_created;

        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules,
            global_rules,
            disabled_rules: &self.disabled_rules,
            steps_mode,
            steps: Vec::new(),
            domain_warnings: Vec::new(),
            cache: HashMap::new(),
            current_path: Vec::new(),
            profiler: &mut self.profiler,
            pattern_marks: pattern_marks.clone(),
            initial_parent_ctx,
            root_expr: expr_id,
            current_phase: phase,
            cycle_detector: None,
            cycle_phase: None,
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            blocked_rules: std::collections::HashSet::new(),
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
            // Budget tracking (unified)
            rewrite_count: 0,
            nodes_snap,
            budget_op: match phase {
                crate::phase::SimplifyPhase::Core | crate::phase::SimplifyPhase::PostCleanup => {
                    crate::budget::Operation::SimplifyCore
                }
                crate::phase::SimplifyPhase::Transform
                | crate::phase::SimplifyPhase::Rationalize => {
                    crate::budget::Operation::SimplifyTransform
                }
            },
            stop_reason: None,
            simplify_purpose,
        };

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract budget tracking stats BEFORE dropping transformer
        let rewrite_count = local_transformer.rewrite_count;
        let budget_op = local_transformer.budget_op;
        let nodes_snap = local_transformer.nodes_snap;
        let stop_reason = local_transformer.stop_reason.take();

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
        // Copy domain_warnings to self (survives even in Off mode)
        self.last_domain_warnings
            .append(&mut local_transformer.domain_warnings);
        drop(local_transformer);

        // Calculate nodes delta AFTER dropping transformer (now we can borrow self.context)
        let nodes_delta = self
            .context
            .stats()
            .nodes_created
            .saturating_sub(nodes_snap);

        // Build PassStats for caller to use with Budget.charge()
        let pass_stats = crate::budget::PassStats {
            op: budget_op,
            rewrite_count,
            nodes_delta,
            terms_materialized: 0, // Simplify doesn't expand terms
            poly_ops: 0,           // Simplify doesn't do poly ops
            stop_reason,
        };

        (new_expr, steps, pass_stats)
    }

    pub fn are_equivalent(&mut self, a: ExprId, b: ExprId) -> bool {
        let diff = self.context.add(Expr::Sub(a, b));
        let expand_str = "expand".to_string();
        let expanded_diff = self.context.add(Expr::Function(expand_str, vec![diff]));
        let (simplified_diff, _) = self.simplify(expanded_diff);

        let result_expr = {
            let expr = self.context.get(simplified_diff);
            if let Expr::Function(name, args) = expr {
                if name == "expand" && args.len() == 1 {
                    args[0]
                } else {
                    simplified_diff
                }
            } else {
                simplified_diff
            }
        };

        let expr = self.context.get(result_expr);
        match expr {
            Expr::Number(n) => n.is_zero(),
            _ => {
                if !self.allow_numerical_verification {
                    return false;
                }
                let vars = self.collect_variables(result_expr);
                let mut var_map = HashMap::new();
                for var in vars {
                    var_map.insert(var, 1.23456789);
                }

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    val.abs() < 1e-9
                } else {
                    false
                }
            }
        }
    }

    /// Normalize, deduplicate, and sort requires strings for stable output.
    ///
    /// Normalization rules:
    /// - `expr ≠ 0` where expr starts with `-` → canonicalize to positive form
    /// - Deduplicate by exact string match
    /// - Sort alphabetically for deterministic output
    fn normalize_requires(&self, requires: &mut Vec<String>) {
        use std::collections::HashSet;

        // Normalize each require string
        for req in requires.iter_mut() {
            // Handle "expr ≠ 0" → strip leading negative if present
            if let Some(expr_part) = req.strip_suffix(" ≠ 0") {
                let trimmed = expr_part.trim();
                // If expression starts with "-(" and ends with ")", remove the negative
                if let Some(inner) = trimmed.strip_prefix("-(") {
                    if let Some(inner) = inner.strip_suffix(")") {
                        *req = format!("{} ≠ 0", inner.trim());
                    }
                } else if let Some(inner) = trimmed.strip_prefix("-") {
                    // Simple negative like "-x" → "x"
                    if !inner.starts_with('(') && !inner.contains(' ') {
                        *req = format!("{} ≠ 0", inner.trim());
                    }
                }
            }
        }

        // Deduplicate
        let mut seen = HashSet::new();
        requires.retain(|r| seen.insert(r.clone()));

        // Sort for deterministic output
        requires.sort();
    }

    /// Extended equivalence check returning tri-state result with domain conditions.
    ///
    /// V2.14.45: This method uses the same simplifier pipeline as the REPL,
    /// and properly interprets SoundnessLabel and Requires from rules.
    ///
    /// Returns:
    /// - `True` if A-B simplifies to 0 with pure Equivalence rules
    /// - `ConditionalTrue` if A-B simplifies to 0 but rules with
    ///   EquivalenceUnderIntroducedRequires were used or domain conditions introduced
    /// - `Unknown` if cannot simplify to 0 but no counterexample found
    /// - `False` if numeric verification finds counterexample
    pub fn are_equivalent_extended(&mut self, a: ExprId, b: ExprId) -> EquivalenceResult {
        use crate::rule::SoundnessLabel;
        use crate::semantic_equality::SemanticEqualityChecker;

        // Enable step collection to track soundness labels
        let was_collecting = self.collect_steps();
        self.set_collect_steps(true);

        // =================================================================
        // OPTION 2: Normal forms comparison
        // Simplify A and B separately, then compare.
        // This catches cases like tan(x)*tan(pi/3-x)*tan(pi/3+x) ≡ tan(3x)
        // where both simplify to tan(3x) but diff doesn't cancel due to
        // expansion rules firing before cancellation.
        // =================================================================
        let (simplified_a, steps_a) = self.simplify(a);

        // Early check: compare simplified_a with b (before simplifying b)
        // This catches cases where A simplifies to exactly B
        // e.g., tan(x)*tan(pi/3-x)*tan(pi/3+x) → tan(3x) ≡ tan(3*x)
        let checker = SemanticEqualityChecker::new(&self.context);
        if checker.are_equal(simplified_a, b) {
            let has_conditional_rules = steps_a
                .iter()
                .any(|step| step.soundness != SoundnessLabel::Equivalence);
            let mut requires: Vec<String> = Vec::new();

            for step in &steps_a {
                for req in &step.required_conditions {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            self.set_collect_steps(was_collecting);
            self.normalize_requires(&mut requires);

            return if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            };
        }

        // Also check: b simplified vs a (before simplifying a further)
        let (simplified_b, steps_b) = self.simplify(b);

        // Check if simplified forms are semantically equal
        let checker = SemanticEqualityChecker::new(&self.context);
        if checker.are_equal(simplified_a, simplified_b) {
            // Merge steps for soundness analysis
            let mut all_steps = steps_a;
            all_steps.extend(steps_b);

            let mut has_conditional_rules = false;
            let mut requires: Vec<String> = Vec::new();

            for step in &all_steps {
                if step.soundness != SoundnessLabel::Equivalence {
                    has_conditional_rules = true;
                }
                for req in &step.required_conditions {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            // Check blocked hints
            for hint in &self.last_blocked_hints {
                let expr_display = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::assumptions::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            self.set_collect_steps(was_collecting);
            self.normalize_requires(&mut requires);

            return if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            };
        }

        // =================================================================
        // Fallback: Try A - B = 0
        // =================================================================
        let diff = self.context.add(Expr::Sub(a, b));
        let (simplified_diff, steps) = self.simplify(diff);

        self.set_collect_steps(was_collecting);

        let result_expr = simplified_diff;

        // Check if result is 0
        let is_zero = match self.context.get(result_expr) {
            Expr::Number(n) => n.is_zero(),
            _ => false,
        };

        if is_zero {
            // Success! Now determine if unconditional or conditional
            // Check for any soundness label worse than Equivalence
            let mut has_conditional_rules = false;
            let mut requires: Vec<String> = Vec::new();

            for step in &steps {
                // Check soundness label
                if step.soundness != SoundnessLabel::Equivalence {
                    has_conditional_rules = true;
                }

                // Collect required_conditions from steps
                for req in &step.required_conditions {
                    let condition_str = req.display(&self.context);
                    if !requires.contains(&condition_str) {
                        requires.push(condition_str);
                    }
                }
            }

            // Also check blocked hints (from Strict mode)
            for hint in &self.last_blocked_hints {
                // Build condition string from AssumptionKey
                let expr_display = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: &self.context,
                        id: hint.expr_id
                    }
                );
                let hint_str = match &hint.key {
                    crate::assumptions::AssumptionKey::NonZero { .. } => {
                        format!("{} ≠ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Positive { .. } => {
                        format!("{} > 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::NonNegative { .. } => {
                        format!("{} ≥ 0", expr_display)
                    }
                    _ => format!("{} ({})", expr_display, hint.rule),
                };
                if !requires.contains(&hint_str) {
                    requires.push(hint_str);
                }
            }

            self.normalize_requires(&mut requires);

            if has_conditional_rules || !requires.is_empty() {
                EquivalenceResult::ConditionalTrue { requires }
            } else {
                EquivalenceResult::True
            }
        } else {
            // Not zero symbolically - try numeric verification
            if self.allow_numerical_verification {
                let vars = self.collect_variables(result_expr);
                let mut var_map = HashMap::new();
                for var in &vars {
                    var_map.insert(var.clone(), 1.23456789);
                }

                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    if val.abs() < 1e-9 {
                        // Numeric evidence suggests equivalence but couldn't prove symbolically
                        return EquivalenceResult::Unknown;
                    } else {
                        // Found counterexample
                        return EquivalenceResult::False;
                    }
                }
            }
            // Can't determine
            EquivalenceResult::Unknown
        }
    }

    fn collect_variables(&self, expr_id: ExprId) -> HashSet<String> {
        use crate::visitors::VariableCollector;
        use cas_ast::Visitor;

        let mut collector = VariableCollector::new();
        collector.visit_expr(&self.context, expr_id);
        collector.vars
    }
}

/// Evaluate an expression numerically with f64 values.
/// Used for numeric property testing to verify rewrite correctness.
/// Has a depth limit of 200 to prevent stack overflow on deeply nested expressions.
pub fn eval_f64(ctx: &Context, expr: ExprId, var_map: &HashMap<String, f64>) -> Option<f64> {
    eval_f64_depth(ctx, expr, var_map, 200)
}

// =============================================================================
// Checked evaluator for robust numeric testing
// =============================================================================

/// Error types for checked numeric evaluation.
/// Provides detailed information about why evaluation failed.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalCheckedError {
    /// Denominator is too close to zero (likely a pole)
    NearPole {
        /// Operation that caused the pole (e.g., "Div", "Tan", "Sec")
        op: &'static str,
        /// Denominator value
        denom: f64,
        /// Threshold that was exceeded
        threshold: f64,
    },
    /// Division by exactly zero
    DivisionByZero { op: &'static str },
    /// Domain error (e.g., log of non-positive, sqrt of negative in RealOnly)
    Domain { function: String, arg: f64 },
    /// Result is not finite (NaN or Inf)
    NonFinite,
    /// Depth limit exceeded
    DepthExceeded,
    /// Variable not found in var_map
    UnboundVariable { name: String },
    /// Unsupported expression type
    Unsupported,
}

/// Options for checked evaluation.
#[derive(Debug, Clone)]
pub struct EvalCheckedOptions {
    /// Absolute epsilon for near-zero denominator detection in Div
    pub zero_abs_eps: f64,
    /// Relative epsilon for near-zero denominator detection in Div
    pub zero_rel_eps: f64,
    /// Epsilon for trigonometric pole detection (tan, sec, csc, cot)
    /// Should be larger than zero_abs_eps due to floating point errors near π/2
    pub trig_pole_eps: f64,
    /// Maximum recursion depth
    pub max_depth: usize,
}

impl Default for EvalCheckedOptions {
    fn default() -> Self {
        Self {
            zero_abs_eps: 1e-12,
            zero_rel_eps: 1e-12,
            trig_pole_eps: 1e-9, // Larger for trig due to FP errors near π/2
            max_depth: 200,
        }
    }
}

/// Evaluate an expression numerically with detailed error reporting.
/// Used for robust numeric testing where we need to distinguish between:
/// - Near-pole singularities (denominator close to zero)
/// - Domain errors (log of negative, etc.)
/// - Other evaluation failures
pub fn eval_f64_checked(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
) -> Result<f64, EvalCheckedError> {
    eval_f64_checked_depth(ctx, expr, var_map, opts, opts.max_depth)
}

/// Internal checked evaluator with depth tracking.
fn eval_f64_checked_depth(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
    depth: usize,
) -> Result<f64, EvalCheckedError> {
    if depth == 0 {
        return Err(EvalCheckedError::DepthExceeded);
    }

    let result = match ctx.get(expr) {
        Expr::Number(n) => n.to_f64().ok_or(EvalCheckedError::NonFinite)?,

        Expr::Variable(v) => *var_map
            .get(v)
            .ok_or_else(|| EvalCheckedError::UnboundVariable { name: v.clone() })?,

        Expr::Add(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                + eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Sub(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                - eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Mul(l, r) => {
            eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?
                * eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?
        }

        Expr::Div(l, r) => {
            // Evaluate denominator first to check for near-pole
            let b = eval_f64_checked_depth(ctx, *r, var_map, opts, depth - 1)?;

            if !b.is_finite() {
                return Err(EvalCheckedError::NonFinite);
            }

            if b == 0.0 {
                return Err(EvalCheckedError::DivisionByZero { op: "Div" });
            }

            // Evaluate numerator for relative threshold calculation
            let a = eval_f64_checked_depth(ctx, *l, var_map, opts, depth - 1)?;

            if !a.is_finite() {
                return Err(EvalCheckedError::NonFinite);
            }

            // Check for near-pole: |b| <= eps_abs + eps_rel * max(1, |a|)
            let scale = f64::max(1.0, a.abs());
            let threshold = opts.zero_abs_eps + opts.zero_rel_eps * scale;

            if b.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Div",
                    denom: b,
                    threshold,
                });
            }

            a / b
        }

        Expr::Pow(b, e) => {
            let base = eval_f64_checked_depth(ctx, *b, var_map, opts, depth - 1)?;
            let exp = eval_f64_checked_depth(ctx, *e, var_map, opts, depth - 1)?;

            // Domain check: base < 0 and non-integer exponent -> complex result
            if base < 0.0 && exp.fract() != 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "pow".to_string(),
                    arg: base,
                });
            }

            base.powf(exp)
        }

        Expr::Neg(e) => -eval_f64_checked_depth(ctx, *e, var_map, opts, depth - 1)?,

        Expr::Function(name, args) => eval_function_checked(ctx, name, args, var_map, opts, depth)?,

        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => std::f64::consts::PI,
            cas_ast::Constant::E => std::f64::consts::E,
            cas_ast::Constant::Phi => 1.618033988749895, // (1+√5)/2
            cas_ast::Constant::Infinity => return Err(EvalCheckedError::NonFinite),
            cas_ast::Constant::Undefined => return Err(EvalCheckedError::NonFinite),
            cas_ast::Constant::I => {
                return Err(EvalCheckedError::Domain {
                    function: "constant".to_string(),
                    arg: 0.0,
                })
            }
        },

        Expr::Matrix { .. } | Expr::SessionRef(_) => {
            return Err(EvalCheckedError::Unsupported);
        }
    };

    // Final check for non-finite result
    if !result.is_finite() {
        return Err(EvalCheckedError::NonFinite);
    }

    Ok(result)
}

/// Evaluate functions with domain checking.
#[inline(never)]
fn eval_function_checked(
    ctx: &Context,
    name: &str,
    args: &[ExprId],
    var_map: &HashMap<String, f64>,
    opts: &EvalCheckedOptions,
    depth: usize,
) -> Result<f64, EvalCheckedError> {
    // Evaluate all arguments
    let arg_vals: Result<Vec<f64>, _> = args
        .iter()
        .map(|a| eval_f64_checked_depth(ctx, *a, var_map, opts, depth - 1))
        .collect();
    let arg_vals = arg_vals?;

    match name {
        // Basic trig - check for tan/sec/csc poles via cos/sin near zero
        "sin" => Ok(arg_vals.first().copied().unwrap_or(0.0).sin()),
        "cos" => Ok(arg_vals.first().copied().unwrap_or(0.0).cos()),
        "tan" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let cos_x = x.cos();
            let threshold = opts.trig_pole_eps;
            if cos_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Tan",
                    denom: cos_x,
                    threshold,
                });
            }
            Ok(x.tan())
        }

        // Reciprocal trig with pole detection
        "sec" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let cos_x = x.cos();
            let threshold = opts.trig_pole_eps;
            if cos_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Sec",
                    denom: cos_x,
                    threshold,
                });
            }
            Ok(1.0 / cos_x)
        }
        "csc" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let sin_x = x.sin();
            let threshold = opts.trig_pole_eps;
            if sin_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Csc",
                    denom: sin_x,
                    threshold,
                });
            }
            Ok(1.0 / sin_x)
        }
        "cot" => {
            let x = arg_vals.first().copied().unwrap_or(0.0);
            let sin_x = x.sin();
            let threshold = opts.trig_pole_eps;
            if sin_x.abs() <= threshold {
                return Err(EvalCheckedError::NearPole {
                    op: "Cot",
                    denom: sin_x,
                    threshold,
                });
            }
            Ok(1.0 / x.tan())
        }

        // Inverse trig
        "asin" | "arcsin" => Ok(arg_vals.first().copied().unwrap_or(0.0).asin()),
        "acos" | "arccos" => Ok(arg_vals.first().copied().unwrap_or(0.0).acos()),
        "atan" | "arctan" => Ok(arg_vals.first().copied().unwrap_or(0.0).atan()),

        // Hyperbolic
        "sinh" => Ok(arg_vals.first().copied().unwrap_or(0.0).sinh()),
        "cosh" => Ok(arg_vals.first().copied().unwrap_or(0.0).cosh()),
        "tanh" => Ok(arg_vals.first().copied().unwrap_or(0.0).tanh()),
        "asinh" | "arcsinh" => Ok(arg_vals.first().copied().unwrap_or(0.0).asinh()),
        "acosh" | "arccosh" => Ok(arg_vals.first().copied().unwrap_or(0.0).acosh()),
        "atanh" | "arctanh" => Ok(arg_vals.first().copied().unwrap_or(0.0).atanh()),

        // Logarithm with domain checking
        "ln" => {
            let arg = arg_vals.first().copied().unwrap_or(0.0);
            if arg <= 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "ln".to_string(),
                    arg,
                });
            }
            Ok(arg.ln())
        }
        "log" => {
            if arg_vals.len() == 2 {
                let base = arg_vals[0];
                let arg = arg_vals[1];
                if base <= 0.0 || base == 1.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log_base".to_string(),
                        arg: base,
                    });
                }
                if arg <= 0.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log_arg".to_string(),
                        arg,
                    });
                }
                Ok(arg.ln() / base.ln())
            } else if arg_vals.len() == 1 {
                let arg = arg_vals[0];
                if arg <= 0.0 {
                    return Err(EvalCheckedError::Domain {
                        function: "log10".to_string(),
                        arg,
                    });
                }
                Ok(arg.log10())
            } else {
                Err(EvalCheckedError::Unsupported)
            }
        }

        // Exponential
        "exp" => Ok(arg_vals.first().copied().unwrap_or(0.0).exp()),

        // Square root with domain checking
        "sqrt" => {
            let arg = arg_vals.first().copied().unwrap_or(0.0);
            if arg < 0.0 {
                return Err(EvalCheckedError::Domain {
                    function: "sqrt".to_string(),
                    arg,
                });
            }
            Ok(arg.sqrt())
        }

        // Other functions
        "abs" => Ok(arg_vals.first().copied().unwrap_or(0.0).abs()),
        "floor" => Ok(arg_vals.first().copied().unwrap_or(0.0).floor()),
        "ceil" => Ok(arg_vals.first().copied().unwrap_or(0.0).ceil()),
        "round" => Ok(arg_vals.first().copied().unwrap_or(0.0).round()),
        "sign" | "sgn" => Ok(arg_vals.first().copied().unwrap_or(0.0).signum()),

        // __hold is transparent
        "__hold" => {
            if let Some(&arg_id) = args.first() {
                eval_f64_checked_depth(ctx, arg_id, var_map, opts, depth - 1)
            } else {
                Err(EvalCheckedError::Unsupported)
            }
        }

        _ => Err(EvalCheckedError::Unsupported),
    }
}

/// Internal eval_f64 with explicit depth limit.
fn eval_f64_depth(
    ctx: &Context,
    expr: ExprId,
    var_map: &HashMap<String, f64>,
    depth: usize,
) -> Option<f64> {
    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => n.to_f64(),
        Expr::Variable(v) => var_map.get(v).cloned(),
        Expr::Add(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                + eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Sub(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                - eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Mul(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                * eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Div(l, r) => Some(
            eval_f64_depth(ctx, *l, var_map, depth - 1)?
                / eval_f64_depth(ctx, *r, var_map, depth - 1)?,
        ),
        Expr::Pow(b, e) => Some(
            eval_f64_depth(ctx, *b, var_map, depth - 1)?.powf(eval_f64_depth(
                ctx,
                *e,
                var_map,
                depth - 1,
            )?),
        ),
        Expr::Neg(e) => Some(-eval_f64_depth(ctx, *e, var_map, depth - 1)?),
        Expr::Function(name, args) => {
            let arg_vals: Option<Vec<f64>> = args
                .iter()
                .map(|a| eval_f64_depth(ctx, *a, var_map, depth - 1))
                .collect();
            let arg_vals = arg_vals?;
            match name.as_str() {
                // Basic trig
                "sin" => Some(arg_vals.first()?.sin()),
                "cos" => Some(arg_vals.first()?.cos()),
                "tan" => Some(arg_vals.first()?.tan()),

                // Reciprocal trig
                "sec" => Some(1.0 / arg_vals.first()?.cos()),
                "csc" => Some(1.0 / arg_vals.first()?.sin()),
                "cot" => Some(1.0 / arg_vals.first()?.tan()),

                // Inverse trig
                "asin" | "arcsin" => Some(arg_vals.first()?.asin()),
                "acos" | "arccos" => Some(arg_vals.first()?.acos()),
                "atan" | "arctan" => Some(arg_vals.first()?.atan()),

                // Hyperbolic
                "sinh" => Some(arg_vals.first()?.sinh()),
                "cosh" => Some(arg_vals.first()?.cosh()),
                "tanh" => Some(arg_vals.first()?.tanh()),

                // Inverse hyperbolic
                "asinh" | "arcsinh" => Some(arg_vals.first()?.asinh()),
                "acosh" | "arccosh" => Some(arg_vals.first()?.acosh()),
                "atanh" | "arctanh" => Some(arg_vals.first()?.atanh()),

                // Exponential and logarithm
                "exp" => Some(arg_vals.first()?.exp()),
                "ln" => Some(arg_vals.first()?.ln()),
                // log(base, arg) -> ln(arg) / ln(base)
                "log" => {
                    if arg_vals.len() == 2 {
                        let base = arg_vals[0];
                        let arg = arg_vals[1];
                        Some(arg.ln() / base.ln())
                    } else if arg_vals.len() == 1 {
                        // log(x) = log base 10
                        Some(arg_vals[0].log10())
                    } else {
                        None
                    }
                }

                // Other
                "sqrt" => Some(arg_vals.first()?.sqrt()),
                "abs" => Some(arg_vals.first()?.abs()),
                "floor" => Some(arg_vals.first()?.floor()),
                "ceil" => Some(arg_vals.first()?.ceil()),
                "round" => Some(arg_vals.first()?.round()),
                "sign" | "sgn" => Some(arg_vals.first()?.signum()),

                // __hold is transparent for evaluation - just evaluate the held expression
                "__hold" => {
                    if args.len() == 1 {
                        eval_f64_depth(ctx, args[0], var_map, depth - 1)
                    } else {
                        None
                    }
                }

                _ => None,
            }
        }
        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Phi => Some(1.618033988749895), // (1+√5)/2
            cas_ast::Constant::Infinity => Some(f64::INFINITY),
            cas_ast::Constant::Undefined => Some(f64::NAN),
            cas_ast::Constant::I => None, // Imaginary unit cannot be evaluated to f64
        },
        Expr::Matrix { .. } => None, // Matrix evaluation not supported in f64
        Expr::SessionRef(_) => None, // SessionRef should be resolved before eval
    }
}

/// Maximum recursion depth for simplification to prevent stack overflow.
///
/// V2.15: Set to 50 after frame size optimizations (see `record_step()`).
/// Previously 30 was needed due to ~150KB frames; with `#[inline(never)]`
/// on step recording, frames are small enough for 50+ on 8MB stack.
///
/// For deeper expressions, use `recursion_guard::with_stack(16MB, || ...)`.
/// Note: All major arms extracted to helpers; depth=100 still overflows due to base frame size.
const MAX_SIMPLIFY_DEPTH: usize = 50;

/// Path to log expressions that exceed the depth limit for later investigation.
const DEPTH_OVERFLOW_LOG_PATH: &str = "/tmp/cas_depth_overflow_expressions.log";

/// Binary operation type for transform_binary helper
#[derive(Clone, Copy)]
#[allow(dead_code)] // Div is kept for consistency, may be used if Div early-detection is removed
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

struct LocalSimplificationTransformer<'a> {
    context: &'a mut Context,
    rules: &'a HashMap<String, Vec<Arc<dyn Rule>>>,
    global_rules: &'a Vec<Arc<dyn Rule>>,
    disabled_rules: &'a HashSet<String>,
    steps_mode: StepsMode,
    steps: Vec<Step>,
    /// Domain warnings collected regardless of steps_mode (for Off mode warning survival)
    domain_warnings: Vec<(String, String)>, // (message, rule_name)
    cache: HashMap<ExprId, ExprId>,
    current_path: Vec<crate::step::PathStep>,
    profiler: &'a mut RuleProfiler,
    #[allow(dead_code)]
    pattern_marks: crate::pattern_marks::PatternMarks, // For context-aware guards (used via initial_parent_ctx)
    initial_parent_ctx: crate::parent_context::ParentContext, // Carries marks to rules
    /// The current root expression being simplified, used to compute global_after for steps
    root_expr: ExprId,
    /// Current phase of the simplification pipeline (controls which rules can run)
    current_phase: crate::phase::SimplifyPhase,
    /// Cycle detector for ping-pong detection (always-on as of V2.14.30)
    cycle_detector: Option<crate::cycle_detector::CycleDetector>,
    /// Phase that the cycle detector was initialized for (reset when phase changes)
    cycle_phase: Option<crate::phase::SimplifyPhase>,
    /// Fingerprint memoization cache (cleared per phase)
    fp_memo: crate::cycle_detector::FingerprintMemo,
    /// Last detected cycle info (for PhaseStats)
    last_cycle: Option<crate::cycle_detector::CycleInfo>,
    /// Blocked (fingerprint, rule) pairs to prevent cycle re-entry
    blocked_rules: std::collections::HashSet<(u64, String)>,
    /// Current recursion depth for stack overflow prevention
    current_depth: usize,
    /// Flag to track if we already warned about depth overflow (to avoid spamming)
    depth_overflow_warned: bool,
    /// Stack of ancestor ExprIds for parent context propagation to rules
    ancestor_stack: Vec<ExprId>,
    // === Budget tracking (Phase 2 unified) ===
    /// Count of rewrites accepted in this pass (charged to Budget at end of pass)
    rewrite_count: u64,
    /// Snapshot of nodes_created at start of pass (for delta charging)
    nodes_snap: u64,
    /// Operation type for budget charging (SimplifyCore or SimplifyTransform)
    budget_op: crate::budget::Operation,
    /// Set when budget exceeded - contains the error details for the caller
    stop_reason: Option<crate::budget::BudgetExceeded>,
    /// Purpose of simplification: controls which rules are filtered by solve_safety()
    simplify_purpose: crate::solve_safety::SimplifyPurpose,
}

use cas_ast::visitor::Transformer;

// NOTE on ancestor_stack pattern:
//
// We cannot use RAII guards (like AncestorScope) for push/pop because:
// 1. AncestorScope would borrow &mut self.ancestor_stack
// 2. transform_expr_recursive needs &mut self
// 3. Rust doesn't allow split borrows of struct fields through methods
//
// REQUIRED PATTERN for all operator cases that recurse into children:
// ```
// self.ancestor_stack.push(id);  // Before transform_expr_recursive
// let result = self.transform_expr_recursive(child);
// self.ancestor_stack.pop();     // After transform_expr_recursive (balanced!)
// ```
//
// This is critical for context-aware rules (like AutoExpandPowSumRule) that check
// in_auto_expand_context() - they need to see their ancestor chain.
// Test: test_auto_expand_step_visible_in_sub_context

impl<'a> Transformer for LocalSimplificationTransformer<'a> {
    fn transform_expr(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        self.transform_expr_recursive(id)
    }
}

impl<'a> LocalSimplificationTransformer<'a> {
    fn indent(&self) -> String {
        "  ".repeat(self.current_path.len())
    }

    /// Reconstruct the global expression by substituting `replacement` at the given path
    fn reconstruct_at_path(&mut self, replacement: ExprId) -> ExprId {
        use crate::step::PathStep;

        fn reconstruct_recursive(
            context: &mut Context,
            root: ExprId,
            path: &[PathStep],
            replacement: ExprId,
        ) -> ExprId {
            if path.is_empty() {
                return replacement;
            }

            let current_step = &path[0];
            let remaining_path = &path[1..];
            let expr = context.get(root).clone();

            match (expr, current_step) {
                (Expr::Add(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Add(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Add(l, r), PathStep::Right) => {
                    // Follow AST literally - don't do magic Neg unwrapping.
                    // If we need to modify inside a Neg, the path should include PathStep::Inner.
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Add(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Sub(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Sub(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Mul(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Mul(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Div(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Div(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Base) => {
                    let new_b = reconstruct_recursive(context, b, remaining_path, replacement);
                    context.add_raw(Expr::Pow(new_b, e)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Exponent) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Pow(b, new_e)) // Use add_raw to preserve structure
                }
                (Expr::Neg(e), PathStep::Inner) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Neg(new_e)) // Use add_raw to preserve structure
                }
                (Expr::Function(name, args), PathStep::Arg(idx)) => {
                    let mut new_args = args;
                    if *idx < new_args.len() {
                        new_args[*idx] = reconstruct_recursive(
                            context,
                            new_args[*idx],
                            remaining_path,
                            replacement,
                        );
                        context.add_raw(Expr::Function(name, new_args)) // Use add_raw to preserve structure
                    } else {
                        root
                    }
                }
                _ => root, // Path mismatch
            }
        }

        let new_root = reconstruct_recursive(
            self.context,
            self.root_expr,
            &self.current_path,
            replacement,
        );
        self.root_expr = new_root; // Update root for next step
        new_root
    }

    /// Record a step without inflating the recursive frame.
    /// Using #[inline(never)] to ensure Step construction stays out of transform_expr_recursive.
    #[inline(never)]
    fn record_step(
        &mut self,
        name: &'static str,
        description: &'static str,
        before: ExprId,
        after: ExprId,
    ) {
        if self.steps_mode != StepsMode::Off {
            let step = crate::step::Step::new(
                name,
                description,
                before,
                after,
                self.current_path.clone(),
                Some(self.context),
            );
            self.steps.push(step);
        }
    }

    /// Transform binary expression (Add/Sub/Mul) by simplifying children.
    /// Extracted to reduce stack frame size in transform_expr_recursive.
    #[inline(never)]
    fn transform_binary(&mut self, id: ExprId, l: ExprId, r: ExprId, op: BinaryOp) -> ExprId {
        self.current_path.push(crate::step::PathStep::Left);
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        self.current_path.pop();

        self.current_path.push(crate::step::PathStep::Right);
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        self.current_path.pop();

        if new_l != l || new_r != r {
            let expr = match op {
                BinaryOp::Add => Expr::Add(new_l, new_r),
                BinaryOp::Sub => Expr::Sub(new_l, new_r),
                BinaryOp::Mul => Expr::Mul(new_l, new_r),
                BinaryOp::Div => Expr::Div(new_l, new_r),
            };
            self.context.add(expr)
        } else {
            id
        }
    }

    /// Transform Pow expression with early detection for sqrt-of-square patterns.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_pow(&mut self, id: ExprId, base: ExprId, exp: ExprId) -> ExprId {
        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if crate::helpers::is_half(self.context, exp) {
            // Try (something^2)^(1/2) -> |something|
            if let Some(result) = self.try_sqrt_of_square(id, base) {
                return result;
            }
            // Try (u * u)^(1/2) -> |u|
            if let Some(result) = self.try_sqrt_of_product(id, base) {
                return result;
            }
        }

        // Check if this Pow is canonical before recursing into children
        if crate::canonical_forms::is_canonical_form(self.context, id) {
            debug!(
                "Skipping simplification of canonical Pow: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        self.current_path.push(crate::step::PathStep::Base);
        self.ancestor_stack.push(id);
        let new_b = self.transform_expr_recursive(base);
        self.ancestor_stack.pop();
        self.current_path.pop();

        self.current_path.push(crate::step::PathStep::Exponent);
        self.ancestor_stack.push(id);
        let new_e = self.transform_expr_recursive(exp);
        self.ancestor_stack.pop();
        self.current_path.pop();

        if new_b != base || new_e != exp {
            self.context.add(Expr::Pow(new_b, new_e))
        } else {
            id
        }
    }

    /// Try to simplify (u^2)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_square(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Pow(inner_base, inner_exp) = self.context.get(base) {
            if let Expr::Number(n) = self.context.get(*inner_exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    let abs_expr = self
                        .context
                        .add(Expr::Function("abs".to_string(), vec![*inner_base]));
                    self.record_step(
                        "sqrt(u^2) = |u|",
                        "Simplify Square Root of Square",
                        id,
                        abs_expr,
                    );
                    return Some(self.transform_expr_recursive(abs_expr));
                }
            }
        }
        None
    }

    /// Try to simplify (u * u)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_product(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Mul(left, right) = self.context.get(base) {
            if crate::ordering::compare_expr(self.context, *left, *right)
                == std::cmp::Ordering::Equal
            {
                let abs_expr = self
                    .context
                    .add(Expr::Function("abs".to_string(), vec![*left]));
                self.record_step(
                    "sqrt(u * u) = |u|",
                    "Simplify Square Root of Product",
                    id,
                    abs_expr,
                );
                return Some(self.transform_expr_recursive(abs_expr));
            }
        }
        None
    }

    /// Transform Function expression by simplifying children.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_function(&mut self, id: ExprId, name: String, args: Vec<ExprId>) -> ExprId {
        // Check if this function is canonical before recursing into children
        if (name == "sqrt" || name == "abs")
            && crate::canonical_forms::is_canonical_form(self.context, id)
        {
            debug!(
                "Skipping simplification of canonical Function: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // HoldAll semantics: do NOT simplify arguments for these functions
        if is_hold_all_function(&name) {
            debug!(
                "HoldAll function, skipping child simplification: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let mut new_args = Vec::with_capacity(args.len());
        let mut changed = false;
        for (i, arg) in args.iter().enumerate() {
            self.current_path.push(crate::step::PathStep::Arg(i));
            self.ancestor_stack.push(id);
            let new_arg = self.transform_expr_recursive(*arg);
            self.ancestor_stack.pop();
            self.current_path.pop();

            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }

        if changed {
            self.context.add(Expr::Function(name, new_args))
        } else {
            id
        }
    }

    /// Transform Div expression with early detection for difference-of-squares pattern.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_div(&mut self, id: ExprId, l: ExprId, r: ExprId) -> ExprId {
        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
            self.context,
            id,
            l,
            r,
            self.steps_mode != StepsMode::Off,
            &mut self.steps,
            &self.current_path,
        ) {
            // Note: don't decrement depth here - transform_expr_recursive manages it
            return self.transform_expr_recursive(early_result);
        }

        // Simplify children
        self.current_path.push(crate::step::PathStep::Left);
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        self.current_path.pop();

        self.current_path.push(crate::step::PathStep::Right);
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        self.current_path.pop();

        if new_l != l || new_r != r {
            self.context.add(Expr::Div(new_l, new_r))
        } else {
            id
        }
    }

    fn transform_expr_recursive(&mut self, id: ExprId) -> ExprId {
        // Depth guard: prevent stack overflow by limiting recursion depth
        self.current_depth += 1;
        if self.current_depth > MAX_SIMPLIFY_DEPTH {
            if !self.depth_overflow_warned {
                self.depth_overflow_warned = true;

                // Log the expression to file for later investigation
                let display = cas_ast::DisplayExpr {
                    context: self.context,
                    id: self.root_expr,
                };
                let expr_str = display.to_string();
                let log_entry = format!(
                    "[{:?}] Depth overflow at phase {:?}, depth {}: {}\n",
                    std::time::SystemTime::now(),
                    self.current_phase,
                    self.current_depth,
                    expr_str
                );

                // Append to log file (ignore errors - this is best-effort)
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(DEPTH_OVERFLOW_LOG_PATH)
                {
                    let _ = file.write_all(log_entry.as_bytes());
                }

                // Emit warning via tracing
                tracing::warn!(
                    target: "simplify",
                    depth = self.current_depth,
                    phase = ?self.current_phase,
                    expr = %expr_str,
                    "depth_overflow - returning expression unsimplified"
                );
            }

            // Return expression as-is without further simplification
            self.current_depth -= 1;
            return id;
        }

        // Use tracing for debug logging
        let expr = self.context.get(id);
        debug!("{}[DEBUG] Visiting: {:?}", self.indent(), expr);

        // println!("Visiting: {:?} {:?}", id, self.context.get(id));
        // println!("Simplifying: {:?}", id);
        if let Some(&cached) = self.cache.get(&id) {
            self.current_depth -= 1;
            return cached;
        }

        // 1. Simplify children first (Bottom-Up)
        let expr = self.context.get(id).clone();

        let expr_with_simplified_children = match expr {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => id,
            Expr::Add(l, r) => self.transform_binary(id, l, r, BinaryOp::Add),
            Expr::Sub(l, r) => self.transform_binary(id, l, r, BinaryOp::Sub),
            Expr::Mul(l, r) => self.transform_binary(id, l, r, BinaryOp::Mul),
            Expr::Div(l, r) => self.transform_div(id, l, r),
            Expr::Pow(b, e) => self.transform_pow(id, b, e),
            Expr::Neg(e) => {
                self.current_path.push(crate::step::PathStep::Inner);
                self.ancestor_stack.push(id); // Track current node as parent for children
                let new_e = self.transform_expr_recursive(e);
                self.ancestor_stack.pop();
                self.current_path.pop();

                if new_e != e {
                    self.context.add(Expr::Neg(new_e))
                } else {
                    id
                }
            }
            Expr::Function(name, args) => self.transform_function(id, name, args),
            Expr::Matrix { rows, cols, data } => {
                // Recursively simplify matrix elements
                let mut new_data = Vec::new();
                let mut changed = false;
                for (i, elem) in data.iter().enumerate() {
                    self.current_path.push(crate::step::PathStep::Arg(i));
                    self.ancestor_stack.push(id); // Track current node as parent for children
                    let new_elem = self.transform_expr_recursive(*elem);
                    self.ancestor_stack.pop();
                    self.current_path.pop();
                    if new_elem != *elem {
                        changed = true;
                    }
                    new_data.push(new_elem);
                }
                if changed {
                    self.context.add(Expr::Matrix {
                        rows,
                        cols,
                        data: new_data,
                    })
                } else {
                    id
                }
            }
            // SessionRef is a leaf - return as-is (should be resolved before simplification)
            Expr::SessionRef(_) => id,
        };

        // 2. Apply rules
        let result = self.apply_rules(expr_with_simplified_children);
        self.cache.insert(id, result);
        result
    }

    fn apply_rules(&mut self, mut expr_id: ExprId) -> ExprId {
        // Note: This loop pattern with early returns is intentional for structured exit points
        #[allow(clippy::never_loop)]
        loop {
            let mut changed = false;
            let variant = get_variant_name(self.context.get(expr_id));
            // println!("apply_rules for {:?} variant: {}", expr_id, variant);
            // Try specific rules
            if let Some(specific_rules) = self.rules.get(variant) {
                for rule in specific_rules {
                    if self.disabled_rules.contains(rule.name()) {
                        self.profiler
                            .record_rejected_disabled(self.current_phase, rule.name());
                        continue;
                    }
                    // Phase ownership: only run rule if allowed in current phase
                    let phase_mask = self.current_phase.mask();
                    if !rule.allowed_phases().contains(phase_mask) {
                        self.profiler
                            .record_rejected_phase(self.current_phase, rule.name());
                        continue;
                    }
                    // SolveSafety filter: in SolvePrepass, only allow Always-safe rules
                    // In SolveTactic, use domain_mode to determine if conditional rules are allowed
                    match self.simplify_purpose {
                        crate::solve_safety::SimplifyPurpose::Eval => {
                            // Eval: all rules allowed (default behavior)
                        }
                        crate::solve_safety::SimplifyPurpose::SolvePrepass => {
                            // Pre-pass: only SolveSafety::Always rules
                            if !rule.solve_safety().safe_for_prepass() {
                                continue;
                            }
                        }
                        crate::solve_safety::SimplifyPurpose::SolveTactic => {
                            // Tactic: check against domain_mode
                            let domain_mode = self.initial_parent_ctx.domain_mode();
                            if !rule.solve_safety().safe_for_tactic(domain_mode) {
                                continue;
                            }
                        }
                    }
                    // Build ParentContext with ancestors from traversal stack + pattern marks + expand_mode + auto_expand
                    let parent_ctx = {
                        let mut ctx = crate::parent_context::ParentContext::root();
                        // Copy pattern marks from initial context
                        if let Some(marks) = self.initial_parent_ctx.pattern_marks() {
                            ctx = crate::parent_context::ParentContext::with_marks(marks.clone());
                        }
                        // CRITICAL: Copy expand_mode from initial context
                        // This enables BinomialExpansionRule when Simplifier::expand() is called
                        if self.initial_parent_ctx.is_expand_mode() {
                            ctx = ctx.with_expand_mode_flag(true);
                        }
                        // Copy auto_expand from initial context
                        // This enables AutoExpandPowSumRule when autoexpand is on
                        if self.initial_parent_ctx.is_auto_expand() {
                            ctx = ctx.with_auto_expand_flag(
                                true,
                                self.initial_parent_ctx.auto_expand_budget().cloned(),
                            );
                        }
                        // Copy domain_mode from initial context for factor cancellation
                        ctx = ctx.with_domain_mode(self.initial_parent_ctx.domain_mode());
                        // Copy inv_trig from initial context for inverse trig simplification
                        ctx = ctx.with_inv_trig(self.initial_parent_ctx.inv_trig_policy());
                        // Copy value_domain from initial context for log expansion rules
                        ctx = ctx.with_value_domain(self.initial_parent_ctx.value_domain());
                        // Copy goal from initial context for expand_log and collect gating
                        ctx = ctx.with_goal(self.initial_parent_ctx.goal());
                        // V2.14.21: Copy root_expr for lazy implicit_domain computation in rules
                        if let Some(root) = self.initial_parent_ctx.root_expr() {
                            ctx = ctx.with_root_expr_only(root);
                        }
                        // V2.15.8: Propagate implicit_domain for domain-aware simplifications
                        // (e.g., AbsNonNegativeSimplifyRule needs to see x≥0 from sqrt)
                        // Note: check_analytic_expansion is gated by rule.solve_safety() below
                        ctx = ctx.with_implicit_domain(
                            self.initial_parent_ctx.implicit_domain().cloned(),
                        );
                        // Build ancestor chain from stack (for Div tracking)
                        for &ancestor in &self.ancestor_stack {
                            ctx = ctx.extend_with_div_check(ancestor, self.context);
                        }
                        // V2.15.8: Copy autoexpand_binomials from initial context
                        ctx = ctx.with_autoexpand_binomials(
                            self.initial_parent_ctx.autoexpand_binomials(),
                        );
                        // V2.15.9: Copy heuristic_poly from initial context
                        ctx = ctx.with_heuristic_poly(self.initial_parent_ctx.heuristic_poly());
                        ctx
                    };

                    if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                        // Check semantic equality - skip if no real change
                        // EXCEPTION: Didactic rules should always generate steps
                        // even if result is semantically equivalent (e.g., sqrt(12) → 2*√3)
                        let is_didactic_rule = rule.name() == "Evaluate Numeric Power"
                            || rule.name() == "Sum Exponents";

                        if !is_didactic_rule {
                            use crate::semantic_equality::SemanticEqualityChecker;
                            let checker = SemanticEqualityChecker::new(self.context);
                            if checker.are_equal(expr_id, rewrite.new_expr) {
                                debug!(
                                    "{}[DEBUG] Rule '{}' produced semantically equal result, skipping",
                                    self.indent(),
                                    rule.name()
                                );
                                self.profiler
                                    .record_rejected_semantic(self.current_phase, rule.name());
                                continue;
                            }
                        }

                        // ANTI-WORSEN BUDGET: Reject rewrites that grow expression beyond threshold.
                        // This is a GLOBAL SAFETY NET against exponential explosion (e.g., sin(16*x) expansion).
                        // Budget policy: Block if BOTH:
                        // - Absolute growth > 30 nodes
                        // - Relative growth > 1.5x (50% larger)
                        // Exception: expand_mode bypasses this check (user explicitly requested expansion)
                        if !parent_ctx.is_expand_mode()
                            && crate::helpers::rewrite_worsens_too_much(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                                30,  // max_growth_abs
                                1.5, // max_growth_ratio
                            )
                        {
                            debug!(
                                "{}[DEBUG] Rule '{}' blocked by anti-worsen budget (expression grew too much)",
                                self.indent(),
                                rule.name()
                            );
                            continue;
                        }

                        // Domain Delta Airbag: Check if rewrite expands analytic domain
                        // This catches any rewrite that removes implicit constraints like x≥0 from sqrt(x)
                        // V2.15.8: Only run this check for rules that declare NeedsCondition(Analytic)
                        // This allows rules like AbsNonNegativeSimplifyRule to use implicit_domain
                        // without triggering the airbag that would block LogExpInverseRule
                        // Behavior by mode:
                        // - Strict/Generic: Block if expansion detected
                        // - Assume: Allow but register dropped predicates as assumptions
                        let vd = parent_ctx.value_domain();
                        let mode = parent_ctx.domain_mode();
                        let needs_analytic_check = matches!(
                            rule.solve_safety(),
                            crate::solve_safety::SolveSafety::NeedsCondition(
                                crate::assumptions::ConditionClass::Analytic
                            )
                        );
                        if parent_ctx.implicit_domain().is_some() && needs_analytic_check {
                            use crate::implicit_domain::{
                                check_analytic_expansion, AnalyticExpansionResult,
                                ImplicitCondition,
                            };

                            let expansion = check_analytic_expansion(
                                self.context,
                                self.root_expr,
                                expr_id,
                                rewrite.new_expr,
                                vd,
                            );

                            if let AnalyticExpansionResult::WouldExpand { dropped, sources } =
                                expansion
                            {
                                match mode {
                                    crate::domain::DomainMode::Strict => {
                                        // Strict: block - don't expand domain at all
                                        debug!(
                                            "{}[DEBUG] Rule '{}' would expand analytic domain ({}), blocked in Strict mode",
                                            self.indent(),
                                            rule.name(),
                                            sources.join(", ")
                                        );
                                        continue;
                                    }
                                    crate::domain::DomainMode::Generic => {
                                        // Generic: allow rewrite but emit as required_conditions (not assumptions!)
                                        // This communicates "the input already required these conditions implicitly"
                                        rewrite.required_conditions.extend(dropped.clone());
                                        debug!(
                                            "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Generic mode with required conditions: {}",
                                            self.indent(),
                                            rule.name(),
                                            sources.join(", ")
                                        );
                                    }
                                    crate::domain::DomainMode::Assume => {
                                        // In Assume mode: allow rewrite but register assumptions
                                        for cond in dropped {
                                            match cond {
                                                ImplicitCondition::NonNegative(t) => {
                                                    rewrite.assumption_events.push(
                                                        crate::assumptions::AssumptionEvent::nonnegative(self.context, t)
                                                    );
                                                }
                                                ImplicitCondition::Positive(t) => {
                                                    rewrite.assumption_events.push(
                                                        crate::assumptions::AssumptionEvent::positive(self.context, t)
                                                    );
                                                }
                                                ImplicitCondition::NonZero(_) => {} // Skip definability
                                            }
                                        }
                                        debug!(
                                            "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Assume mode with assumptions: {}",
                                            self.indent(),
                                            rule.name(),
                                            sources.join(", ")
                                        );
                                    }
                                }
                            }
                        }

                        // Record rule application with delta_nodes for health metrics
                        let delta = if self.profiler.is_health_enabled() {
                            let before = crate::helpers::count_all_nodes(self.context, expr_id);
                            let after =
                                crate::helpers::count_all_nodes(self.context, rewrite.new_expr);
                            after as i64 - before as i64
                        } else {
                            0
                        };
                        self.profiler
                            .record_with_delta(self.current_phase, rule.name(), delta);

                        // TRACE: Log applied rules for debugging cycles
                        if std::env::var("CAS_TRACE_RULES").is_ok() {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("/tmp/rule_trace.log")
                            {
                                let node_count_before =
                                    crate::helpers::node_count(self.context, expr_id);
                                let node_count_after =
                                    crate::helpers::node_count(self.context, rewrite.new_expr);
                                let _ = writeln!(
                                    f,
                                    "APPLIED depth={} rule={} nodes={}->{}",
                                    self.current_depth,
                                    rule.name(),
                                    node_count_before,
                                    node_count_after
                                );
                                let _ = f.flush();
                            }
                        }

                        // println!(
                        //     "Rule '{}' applied: {:?} -> {:?}",
                        //     rule.name(),
                        //     expr_id,
                        //     rewrite.new_expr
                        // );
                        debug!(
                            "{}[DEBUG] Rule '{}' applied: {:?} -> {:?}",
                            self.indent(),
                            rule.name(),
                            expr_id,
                            rewrite.new_expr
                        );
                        if self.steps_mode != StepsMode::Off {
                            // Extract rewrite fields to avoid borrow conflicts with self methods
                            let main_new_expr = rewrite.new_expr;
                            let main_description = rewrite.description.clone();
                            let main_before_local = rewrite.before_local;
                            let main_after_local = rewrite.after_local;
                            let main_assumptions = rewrite.assumption_events.clone();
                            let main_required = rewrite.required_conditions.clone();
                            let main_poly_proof = rewrite.poly_proof.clone();
                            let main_substeps = rewrite.substeps.clone();
                            let chained_rewrites = rewrite.chained.clone();

                            // Determine final result (last of chained, or main rewrite)
                            let final_result = chained_rewrites
                                .last()
                                .map(|c| c.after)
                                .unwrap_or(main_new_expr);

                            let global_before = self.root_expr;
                            let main_global_after = self.reconstruct_at_path(main_new_expr);

                            // Main step: before=expr_id, after=main_new_expr
                            let mut step = Step::with_snapshots(
                                &main_description,
                                rule.name(),
                                expr_id,
                                main_new_expr,
                                self.current_path.clone(),
                                Some(self.context),
                                global_before,
                                main_global_after,
                            );
                            step.before_local = main_before_local;
                            step.after_local = main_after_local;
                            step.assumption_events = main_assumptions;
                            step.required_conditions = main_required;
                            step.poly_proof = main_poly_proof;
                            step.substeps = main_substeps;
                            step.importance = rule.importance();
                            self.steps.push(step);

                            // V2.14.20: Trace coherence verification
                            // Verify that the step's global_after matches the updated root_expr
                            // This catches mismatches where subsequent steps might have stale global_before
                            debug_assert_eq!(
                                main_global_after, self.root_expr,
                                "[Trace Coherence] Step global_after doesn't match updated root_expr. \
                                 Rule: {}, This will cause trace mismatch for next step.",
                                rule.name()
                            );

                            // Process chained rewrites sequentially
                            // Invariant: each step's before equals previous step's after
                            let mut current = main_new_expr;
                            for chain_rw in chained_rewrites {
                                let chain_global_before = self.reconstruct_at_path(current);
                                let chain_global_after = self.reconstruct_at_path(chain_rw.after);

                                let mut chain_step = Step::with_snapshots(
                                    &chain_rw.description,
                                    rule.name(),
                                    current,
                                    chain_rw.after,
                                    self.current_path.clone(),
                                    Some(self.context),
                                    chain_global_before,
                                    chain_global_after,
                                );
                                chain_step.before_local = chain_rw.before_local;
                                chain_step.after_local = chain_rw.after_local;
                                chain_step.assumption_events = chain_rw.assumption_events;
                                chain_step.required_conditions = chain_rw.required_conditions;
                                chain_step.poly_proof = chain_rw.poly_proof;
                                chain_step.importance =
                                    chain_rw.importance.unwrap_or_else(|| rule.importance());
                                chain_step.is_chained = true; // V2.12.13: Gate didactic substeps
                                self.steps.push(chain_step);

                                current = chain_rw.after;
                            }

                            expr_id = final_result;
                        } else {
                            // Without steps, just use final result
                            expr_id = rewrite
                                .chained
                                .last()
                                .map(|c| c.after)
                                .unwrap_or(rewrite.new_expr);
                        }

                        // Budget tracking: count this rewrite (charged at end of pass)
                        self.rewrite_count += 1;

                        // Note: Rule application tracking for rationalization is now handled by phase, not flag
                        // Apply canonical normalization to prevent loops
                        expr_id = normalize_core(self.context, expr_id);

                        // V2.14.30: Always-On Cycle Detection with blocklist
                        // Reset detector if phase changed since last initialization
                        if self.cycle_phase != Some(self.current_phase) {
                            self.cycle_detector = Some(crate::cycle_detector::CycleDetector::new(
                                self.current_phase,
                            ));
                            self.cycle_phase = Some(self.current_phase);
                            self.fp_memo.clear();
                            // Note: blocked_rules persists across phases (conservative)
                        }

                        let h = crate::cycle_detector::expr_fingerprint(
                            self.context,
                            expr_id,
                            &mut self.fp_memo,
                        );
                        if let Some(info) = self.cycle_detector.as_mut().unwrap().observe(h) {
                            // Add to blocklist to prevent re-entry
                            let rule_name_static = rule.name();
                            if self.blocked_rules.insert((h, rule_name_static.to_string())) {
                                // First time seeing this (fingerprint, rule) - emit hint
                                // But don't emit for constants/numbers (they're not meaningful cycles)
                                let is_constant = matches!(
                                    self.context.get(expr_id),
                                    cas_ast::Expr::Number(_) | cas_ast::Expr::Constant(_)
                                );
                                if !is_constant {
                                    crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                                        key: crate::assumptions::AssumptionKey::Defined {
                                            expr_fingerprint: h,
                                        },
                                        expr_id,
                                        rule: rule_name_static.to_string(),
                                        suggestion: "cycle detected; consider disabling heuristic rules or tightening budget",
                                    });
                                }
                            }
                            self.last_cycle = Some(info);
                            // Treat as fixed-point: stop this phase early
                            self.current_depth -= 1;
                            return expr_id;
                        }

                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            // Try global rules
            for rule in self.global_rules {
                if self.disabled_rules.contains(rule.name()) {
                    continue;
                }
                // Phase ownership: only run rule if allowed in current phase
                let phase_mask = self.current_phase.mask();
                if !rule.allowed_phases().contains(phase_mask) {
                    self.profiler
                        .record_rejected_phase(self.current_phase, rule.name());
                    continue;
                }

                // Apply rule with parent_ctx containing ancestorsfrom traversal stack
                // V2.14.27: Update to use full parent_ctx with ancestors for Div tracking guards
                let parent_ctx = {
                    let mut ctx = crate::parent_context::ParentContext::root();
                    // Copy pattern marks from initial context
                    if let Some(marks) = self.initial_parent_ctx.pattern_marks() {
                        ctx = crate::parent_context::ParentContext::with_marks(marks.clone());
                    }
                    // Copy other settings from initial context
                    if self.initial_parent_ctx.is_expand_mode() {
                        ctx = ctx.with_expand_mode_flag(true);
                    }
                    if self.initial_parent_ctx.is_auto_expand() {
                        ctx = ctx.with_auto_expand_flag(
                            true,
                            self.initial_parent_ctx.auto_expand_budget().cloned(),
                        );
                    }
                    ctx = ctx.with_domain_mode(self.initial_parent_ctx.domain_mode());
                    ctx = ctx.with_inv_trig(self.initial_parent_ctx.inv_trig_policy());
                    ctx = ctx.with_value_domain(self.initial_parent_ctx.value_domain());
                    ctx = ctx.with_goal(self.initial_parent_ctx.goal());
                    if let Some(root) = self.initial_parent_ctx.root_expr() {
                        ctx = ctx.with_root_expr_only(root);
                    }
                    // V2.14.27: Propagate context_mode and simplify_purpose for Solve mode blocking
                    ctx = ctx.with_context_mode(self.initial_parent_ctx.context_mode());
                    ctx = ctx.with_simplify_purpose(self.initial_parent_ctx.simplify_purpose());
                    // V2.14.27: Propagate implicit_domain for domain-aware simplifications
                    // This is needed for sqrt(x)^2 → x in Generic mode
                    ctx = ctx
                        .with_implicit_domain(self.initial_parent_ctx.implicit_domain().cloned());
                    // Build ancestor chain from stack (for Div tracking)
                    for &ancestor in &self.ancestor_stack {
                        ctx = ctx.extend_with_div_check(ancestor, self.context);
                    }
                    // V2.15.8: Copy autoexpand_binomials from initial context
                    ctx = ctx
                        .with_autoexpand_binomials(self.initial_parent_ctx.autoexpand_binomials());
                    // V2.15.9: Copy heuristic_poly from initial context
                    ctx = ctx.with_heuristic_poly(self.initial_parent_ctx.heuristic_poly());
                    ctx
                };
                if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                    // Fast path: if rewrite produces identical ExprId, skip entirely
                    if rewrite.new_expr == expr_id {
                        continue;
                    }

                    // Semantic equality check to prevent infinite loops
                    // Skip rewrites that produce semantically equivalent results without improvement
                    let is_didactic_rule =
                        rule.name() == "Evaluate Numeric Power" || rule.name() == "Sum Exponents";

                    if !is_didactic_rule {
                        use crate::semantic_equality::SemanticEqualityChecker;
                        let checker = SemanticEqualityChecker::new(self.context);
                        if checker.are_equal(expr_id, rewrite.new_expr) {
                            // Provably equal - only accept if it improves normal form
                            if !crate::helpers::nf_score_after_is_better(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                            ) {
                                continue; // Skip - no improvement
                            }
                        }
                    }

                    // Domain Delta Airbag: Check if rewrite expands analytic domain
                    // Behavior by mode:
                    // - Strict/Generic: Block if expansion detected
                    // - Assume: Allow but register dropped predicates as assumptions
                    let vd = self.initial_parent_ctx.value_domain();
                    let mode = self.initial_parent_ctx.domain_mode();
                    if self.initial_parent_ctx.implicit_domain().is_some() {
                        use crate::implicit_domain::{
                            check_analytic_expansion, AnalyticExpansionResult, ImplicitCondition,
                        };

                        let expansion = check_analytic_expansion(
                            self.context,
                            self.root_expr,
                            expr_id,
                            rewrite.new_expr,
                            vd,
                        );

                        if let AnalyticExpansionResult::WouldExpand { dropped, sources } = expansion
                        {
                            match mode {
                                crate::domain::DomainMode::Strict => {
                                    // Strict: block - don't expand domain at all
                                    debug!(
                                        "{}[DEBUG] Global Rule '{}' would expand analytic domain ({}), blocked in Strict mode",
                                        self.indent(),
                                        rule.name(),
                                        sources.join(", ")
                                    );
                                    continue;
                                }
                                crate::domain::DomainMode::Generic => {
                                    // Generic: allow rewrite but emit as required_conditions (not assumptions!)
                                    // This communicates "the input already required these conditions implicitly"
                                    rewrite.required_conditions.extend(dropped.clone());
                                    debug!(
                                        "{}[DEBUG] Global Rule '{}' expands analytic domain, allowed in Generic mode with required conditions: {}",
                                        self.indent(),
                                        rule.name(),
                                        sources.join(", ")
                                    );
                                }
                                crate::domain::DomainMode::Assume => {
                                    // In Assume mode: allow rewrite but register assumptions
                                    for cond in dropped {
                                        match cond {
                                            ImplicitCondition::NonNegative(t) => {
                                                rewrite.assumption_events.push(
                                                    crate::assumptions::AssumptionEvent::nonnegative(self.context, t)
                                                );
                                            }
                                            ImplicitCondition::Positive(t) => {
                                                rewrite.assumption_events.push(
                                                    crate::assumptions::AssumptionEvent::positive(
                                                        self.context,
                                                        t,
                                                    ),
                                                );
                                            }
                                            ImplicitCondition::NonZero(_) => {} // Skip definability
                                        }
                                    }
                                    debug!(
                                        "{}[DEBUG] Global Rule '{}' expands analytic domain, allowed in Assume mode with assumptions: {}",
                                        self.indent(),
                                        rule.name(),
                                        sources.join(", ")
                                    );
                                }
                            }
                        }
                    }

                    // Record rule application for profiling
                    self.profiler.record(self.current_phase, rule.name());

                    debug!(
                        "{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}",
                        self.indent(),
                        rule.name(),
                        expr_id,
                        rewrite.new_expr
                    );
                    if self.steps_mode != StepsMode::Off {
                        // Extract rewrite fields to avoid borrow conflicts
                        let main_new_expr = rewrite.new_expr;
                        let main_description = rewrite.description.clone();
                        let main_before_local = rewrite.before_local;
                        let main_after_local = rewrite.after_local;
                        let main_assumptions = rewrite.assumption_events.clone();
                        let main_required = rewrite.required_conditions.clone();
                        let main_poly_proof = rewrite.poly_proof.clone();
                        let main_substeps = rewrite.substeps.clone();
                        let chained_rewrites = rewrite.chained.clone();

                        let final_result = chained_rewrites
                            .last()
                            .map(|c| c.after)
                            .unwrap_or(main_new_expr);

                        let global_before = self.root_expr;
                        let main_global_after = self.reconstruct_at_path(main_new_expr);

                        let mut step = Step::with_snapshots(
                            &main_description,
                            rule.name(),
                            expr_id,
                            main_new_expr,
                            self.current_path.clone(),
                            Some(self.context),
                            global_before,
                            main_global_after,
                        );
                        step.before_local = main_before_local;
                        step.after_local = main_after_local;
                        step.assumption_events = main_assumptions;
                        step.required_conditions = main_required;
                        step.poly_proof = main_poly_proof;
                        step.substeps = main_substeps;
                        step.importance = rule.importance();
                        self.steps.push(step);

                        // V2.14.20: Trace coherence verification (global rules)
                        debug_assert_eq!(
                            main_global_after, self.root_expr,
                            "[Trace Coherence] Global rule step global_after doesn't match updated root_expr. \
                             Rule: {}, This will cause trace mismatch for next step.",
                            rule.name()
                        );

                        // Process chained rewrites
                        let mut current = main_new_expr;
                        for chain_rw in chained_rewrites {
                            let chain_global_before = self.reconstruct_at_path(current);
                            let chain_global_after = self.reconstruct_at_path(chain_rw.after);

                            let mut chain_step = Step::with_snapshots(
                                &chain_rw.description,
                                rule.name(),
                                current,
                                chain_rw.after,
                                self.current_path.clone(),
                                Some(self.context),
                                chain_global_before,
                                chain_global_after,
                            );
                            chain_step.before_local = chain_rw.before_local;
                            chain_step.after_local = chain_rw.after_local;
                            chain_step.assumption_events = chain_rw.assumption_events;
                            chain_step.required_conditions = chain_rw.required_conditions;
                            chain_step.poly_proof = chain_rw.poly_proof;
                            chain_step.importance =
                                chain_rw.importance.unwrap_or_else(|| rule.importance());
                            chain_step.is_chained = true; // V2.12.13: Gate didactic substeps
                            self.steps.push(chain_step);

                            current = chain_rw.after;
                        }

                        expr_id = final_result;
                    } else {
                        expr_id = rewrite
                            .chained
                            .last()
                            .map(|c| c.after)
                            .unwrap_or(rewrite.new_expr);
                    }

                    // Budget tracking: count this rewrite (charged at end of pass)
                    self.rewrite_count += 1;

                    // Note: Rule application tracking for rationalization is now handled by phase, not flag
                    // Apply canonical normalization to prevent loops
                    expr_id = normalize_core(self.context, expr_id);
                    changed = true;
                    break;
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            self.current_depth -= 1;
            return expr_id;
        }
    }
}

fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Variable(_) => "Variable",
        Expr::Number(_) => "Number",
        Expr::Constant(_) => "Constant",
        Expr::Matrix { .. } => "Matrix",
        Expr::SessionRef(_) => "SessionRef",
    }
}
