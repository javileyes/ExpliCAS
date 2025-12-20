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
    match ctx.get(expr) {
        Expr::Function(name, args) if name == "__hold" && args.len() == 1 => args[0],
        _ => expr,
    }
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
}

impl Default for Simplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplifier {
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
        }
    }

    pub fn with_default_rules() -> Self {
        let mut s = Self::new();
        s.register_default_rules();
        s
    }

    /// Create simplifier with principal branch rules enabled.
    /// This includes `PrincipalBranchInverseTrigRule` which simplifies
    /// compositions like `arctan(tan(u)) → u` with domain warnings.
    pub fn with_principal_branch_rules() -> Self {
        let mut s = Self::with_default_rules();
        crate::rules::inverse_trig::register_principal_branch(&mut s);
        s
    }

    /// Create a simplifier based on evaluation options.
    /// This is the main entry point for context-aware simplification.
    pub fn with_profile(opts: &crate::options::EvalOptions) -> Self {
        use crate::options::{BranchMode, ContextMode};

        let mut s = Self::with_default_rules();

        // Apply branch mode rules
        if opts.branch_mode == BranchMode::PrincipalBranch {
            crate::rules::inverse_trig::register_principal_branch(&mut s);
        }

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
    #[cfg(debug_assertions)]
    pub fn assert_unique_rule_names(&self) {
        let mut seen = HashSet::new();
        for rule in &self.global_rules {
            let name = rule.name();
            if !seen.insert(name) {
                panic!(
                    "Duplicate rule name detected: '{}'. Each rule must have a unique name.",
                    name
                );
            }
        }
        for rules in self.rules.values() {
            for rule in rules {
                let name = rule.name();
                if !seen.insert(name) {
                    panic!(
                        "Duplicate rule name detected: '{}'. Each rule must have a unique name.",
                        name
                    );
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
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
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
        let mut orchestrator = crate::orchestrator::Orchestrator::new();
        orchestrator.enable_polynomial_strategy = self.enable_polynomial_strategy;
        orchestrator.options = options;
        orchestrator.options.collect_steps = self.collect_steps();
        orchestrator.simplify_pipeline(expr_id, self)
    }

    /// Expand without rationalization.
    pub fn expand(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::for_expand())
    }

    pub fn apply_rules_loop(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
    ) -> (ExprId, Vec<Step>) {
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
    ) -> (ExprId, Vec<Step>) {
        // Default: not in expand mode
        self.apply_rules_loop_with_phase_and_mode(expr_id, pattern_marks, phase, false)
    }

    /// Apply rules loop with explicit expand_mode control
    pub fn apply_rules_loop_with_phase_and_mode(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
        expand_mode: bool,
    ) -> (ExprId, Vec<Step>) {
        let rules = &self.rules;
        let global_rules = &self.global_rules;
        let steps_mode = self.steps_mode;

        // Create initial ParentContext with pattern marks and expand_mode
        let initial_parent_ctx = crate::parent_context::ParentContext::with_expand_mode(
            pattern_marks.clone(),
            expand_mode,
        );

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
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
        };

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
        // Copy domain_warnings to self (survives even in Off mode)
        self.last_domain_warnings
            .extend(local_transformer.domain_warnings.drain(..));
        drop(local_transformer);

        (new_expr, steps)
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

    fn collect_variables(&self, expr_id: ExprId) -> HashSet<String> {
        use crate::visitors::VariableCollector;
        use cas_ast::Visitor;

        let mut collector = VariableCollector::new();
        collector.visit_expr(&self.context, expr_id);
        collector.vars
    }
}

fn eval_f64(ctx: &Context, expr: ExprId, var_map: &HashMap<String, f64>) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(n) => n.to_f64(),
        Expr::Variable(v) => var_map.get(v).cloned(),
        Expr::Add(l, r) => Some(eval_f64(ctx, *l, var_map)? + eval_f64(ctx, *r, var_map)?),
        Expr::Sub(l, r) => Some(eval_f64(ctx, *l, var_map)? - eval_f64(ctx, *r, var_map)?),
        Expr::Mul(l, r) => Some(eval_f64(ctx, *l, var_map)? * eval_f64(ctx, *r, var_map)?),
        Expr::Div(l, r) => Some(eval_f64(ctx, *l, var_map)? / eval_f64(ctx, *r, var_map)?),
        Expr::Pow(b, e) => Some(eval_f64(ctx, *b, var_map)?.powf(eval_f64(ctx, *e, var_map)?)),
        Expr::Neg(e) => Some(-eval_f64(ctx, *e, var_map)?),
        Expr::Function(name, args) => {
            let arg_vals: Option<Vec<f64>> =
                args.iter().map(|a| eval_f64(ctx, *a, var_map)).collect();
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

                _ => None,
            }
        }
        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Infinity => Some(f64::INFINITY),
            cas_ast::Constant::Undefined => Some(f64::NAN),
            cas_ast::Constant::I => None, // Imaginary unit cannot be evaluated to f64
        },
        Expr::Matrix { .. } => None, // Matrix evaluation not supported in f64
        Expr::SessionRef(_) => None, // SessionRef should be resolved before eval
    }
}

/// Maximum recursion depth for simplification to prevent stack overflow.
/// This is intentionally high to avoid interfering with normal operation.
/// If exceeded, a warning is logged and the expression is returned unsimplified.
const MAX_SIMPLIFY_DEPTH: usize = 500;

/// Path to log expressions that exceed the depth limit for later investigation.
const DEPTH_OVERFLOW_LOG_PATH: &str = "/tmp/cas_depth_overflow_expressions.log";

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
    /// Cycle detector for ping-pong detection (health mode only)
    cycle_detector: Option<crate::cycle_detector::CycleDetector>,
    /// Fingerprint memoization cache (cleared per phase)
    fp_memo: crate::cycle_detector::FingerprintMemo,
    /// Last detected cycle info (for PhaseStats)
    last_cycle: Option<crate::cycle_detector::CycleInfo>,
    /// Current recursion depth for stack overflow prevention
    current_depth: usize,
    /// Flag to track if we already warned about depth overflow (to avoid spamming)
    depth_overflow_warned: bool,
    /// Stack of ancestor ExprIds for parent context propagation to rules
    ancestor_stack: Vec<ExprId>,
}

use cas_ast::visitor::Transformer;

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
            Expr::Add(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                self.ancestor_stack.push(id); // Track current node as parent for children
                let new_l = self.transform_expr_recursive(l);
                self.ancestor_stack.pop();
                self.current_path.pop();

                self.current_path.push(crate::step::PathStep::Right);
                self.ancestor_stack.push(id); // Track current node as parent for children
                let new_r = self.transform_expr_recursive(r);
                self.ancestor_stack.pop();
                self.current_path.pop();

                if new_l != l || new_r != r {
                    self.context.add(Expr::Add(new_l, new_r))
                } else {
                    id
                }
            }
            Expr::Sub(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();

                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();

                if new_l != l || new_r != r {
                    self.context.add(Expr::Sub(new_l, new_r))
                } else {
                    id
                }
            }
            Expr::Mul(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();

                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();

                if new_l != l || new_r != r {
                    self.context.add(Expr::Mul(new_l, new_r))
                } else {
                    id
                }
            }
            Expr::Div(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();

                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();

                if new_l != l || new_r != r {
                    self.context.add(Expr::Div(new_l, new_r))
                } else {
                    id
                }
            }
            Expr::Pow(b, e) => {
                // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
                // Must check BEFORE recursing into children to prevent binomial expansion
                if crate::helpers::is_half(self.context, e) {
                    // Outer is ^(1/2), check if base is (something)^2
                    if let Expr::Pow(inner_base, inner_exp) = self.context.get(b) {
                        if let Expr::Number(n) = self.context.get(*inner_exp) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer(2.into())
                            {
                                // Pattern matched: (inner_base^2)^(1/2) -> |inner_base|
                                let abs_expr = self
                                    .context
                                    .add(Expr::Function("abs".to_string(), vec![*inner_base]));

                                // Record the step
                                if self.steps_mode != StepsMode::Off {
                                    let step = crate::step::Step::new(
                                        "sqrt(u^2) = |u|",
                                        "Simplify Square Root of Square",
                                        id,
                                        abs_expr,
                                        self.current_path.clone(),
                                        Some(self.context),
                                    );
                                    self.steps.push(step);
                                }

                                // Continue simplifying the result
                                return self.transform_expr_recursive(abs_expr);
                            }
                        }
                    }
                    // Also check for (u * u)^(1/2) -> |u| directly
                    // This prevents binomial expansion from firing on the squared form
                    if let Expr::Mul(left, right) = self.context.get(b) {
                        if crate::ordering::compare_expr(self.context, *left, *right)
                            == std::cmp::Ordering::Equal
                        {
                            // Pattern matched: (u * u)^(1/2) -> |u|
                            let abs_expr = self
                                .context
                                .add(Expr::Function("abs".to_string(), vec![*left]));

                            // Record the step
                            if self.steps_mode != StepsMode::Off {
                                let step = crate::step::Step::new(
                                    "sqrt(u * u) = |u|",
                                    "Simplify Square Root of Product",
                                    id,
                                    abs_expr,
                                    self.current_path.clone(),
                                    Some(self.context),
                                );
                                self.steps.push(step);
                            }

                            // Continue simplifying the result
                            return self.transform_expr_recursive(abs_expr);
                        }
                    }
                }

                // Check if this Pow is canonical before recursing into children
                // If it's canonical (like ((x+1)*(x-1))^2), we should NOT simplify the base
                if crate::canonical_forms::is_canonical_form(self.context, id) {
                    debug!(
                        "Skipping simplification of canonical Pow: {:?}",
                        self.context.get(id)
                    );
                    id // Return as-is without recursing
                } else {
                    self.current_path.push(crate::step::PathStep::Base);
                    let new_b = self.transform_expr_recursive(b);
                    self.current_path.pop();

                    self.current_path.push(crate::step::PathStep::Exponent);
                    let new_e = self.transform_expr_recursive(e);
                    self.current_path.pop();

                    if new_b != b || new_e != e {
                        self.context.add(Expr::Pow(new_b, new_e))
                    } else {
                        id
                    }
                }
            }
            Expr::Neg(e) => {
                self.current_path.push(crate::step::PathStep::Inner);
                let new_e = self.transform_expr_recursive(e);
                self.current_path.pop();

                if new_e != e {
                    self.context.add(Expr::Neg(new_e))
                } else {
                    id
                }
            }
            Expr::Function(name, args) => {
                // Check if this function is canonical before recursing into children
                // For sqrt() and abs(), we might want to preserve certain forms like sqrt((x-1)^2)
                if (name == "sqrt" || name == "abs")
                    && crate::canonical_forms::is_canonical_form(self.context, id)
                {
                    debug!(
                        "Skipping simplification of canonical Function: {:?}",
                        self.context.get(id)
                    );
                    id // Return as-is without recursing into children
                } else if is_hold_all_function(&name) {
                    // HoldAll semantics: do NOT simplify arguments for these functions
                    // This allows poly_gcd(a*g, b*g) to see the raw structure
                    debug!(
                        "HoldAll function, skipping child simplification: {:?}",
                        self.context.get(id)
                    );
                    id // Return as-is, let the rule handle raw args
                } else {
                    let mut new_args = Vec::new();
                    let mut changed = false;
                    for (i, arg) in args.iter().enumerate() {
                        self.current_path.push(crate::step::PathStep::Arg(i));
                        let new_arg = self.transform_expr_recursive(*arg);
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
            }
            Expr::Matrix { rows, cols, data } => {
                // Recursively simplify matrix elements
                let mut new_data = Vec::new();
                let mut changed = false;
                for (i, elem) in data.iter().enumerate() {
                    self.current_path.push(crate::step::PathStep::Arg(i));
                    let new_elem = self.transform_expr_recursive(*elem);
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
                    // Build ParentContext with ancestors from traversal stack + pattern marks + expand_mode
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
                        // Build ancestor chain from stack
                        for &ancestor in &self.ancestor_stack {
                            ctx = ctx.extend(ancestor);
                        }
                        ctx
                    };
                    if let Some(rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
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
                        // Always accumulate domain_warnings (survives Off mode)
                        if let Some(assumption) = rewrite.domain_assumption {
                            self.domain_warnings
                                .push((rule.name().to_string(), assumption.to_string()));
                        }
                        if self.steps_mode != StepsMode::Off {
                            let global_before = self.root_expr;
                            let global_after = self.reconstruct_at_path(rewrite.new_expr);
                            let mut step = Step::with_snapshots(
                                &rewrite.description,
                                rule.name(),
                                expr_id,
                                rewrite.new_expr,
                                self.current_path.clone(),
                                Some(self.context),
                                global_before,
                                global_after,
                            );
                            // Propagate local before/after from Rewrite for accurate Rule display
                            step.before_local = rewrite.before_local;
                            step.after_local = rewrite.after_local;
                            step.domain_assumption = rewrite.domain_assumption;
                            self.steps.push(step);
                        }
                        expr_id = rewrite.new_expr;
                        // Note: Rule application tracking for rationalization is now handled by phase, not flag
                        // Apply canonical normalization to prevent loops
                        expr_id = normalize_core(self.context, expr_id);

                        // Cycle detection (health mode only) - detect ping-pong patterns
                        if self.profiler.is_health_enabled() {
                            // Initialize detector if needed
                            if self.cycle_detector.is_none() {
                                self.cycle_detector = Some(
                                    crate::cycle_detector::CycleDetector::new(self.current_phase),
                                );
                            }

                            let h = crate::cycle_detector::expr_fingerprint(
                                self.context,
                                expr_id,
                                &mut self.fp_memo,
                            );
                            if let Some(info) = self.cycle_detector.as_mut().unwrap().observe(h) {
                                self.last_cycle = Some(info);
                                // Treat as fixed-point: stop this phase early
                                self.current_depth -= 1;
                                return expr_id;
                            }
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

                // Apply rule with initial_parent_ctx which contains pattern_marks
                // CRITICAL: Must use initial_parent_ctx for pattern-aware guards (like AngleIdentityRule)
                if let Some(rewrite) = rule.apply(self.context, expr_id, &self.initial_parent_ctx) {
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

                    // Record rule application for profiling
                    self.profiler.record(self.current_phase, rule.name());

                    debug!(
                        "{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}",
                        self.indent(),
                        rule.name(),
                        expr_id,
                        rewrite.new_expr
                    );
                    // Always accumulate domain_warnings (survives Off mode)
                    if let Some(assumption) = rewrite.domain_assumption {
                        self.domain_warnings
                            .push((rule.name().to_string(), assumption.to_string()));
                    }
                    if self.steps_mode != StepsMode::Off {
                        let global_before = self.root_expr;
                        let global_after = self.reconstruct_at_path(rewrite.new_expr);
                        let mut step = Step::with_snapshots(
                            &rewrite.description,
                            rule.name(),
                            expr_id,
                            rewrite.new_expr,
                            self.current_path.clone(),
                            Some(self.context),
                            global_before,
                            global_after,
                        );
                        // Propagate local before/after from Rewrite for accurate Rule display
                        step.before_local = rewrite.before_local;
                        step.after_local = rewrite.after_local;
                        // Propagate domain assumption from Rewrite to Step
                        step.domain_assumption = rewrite.domain_assumption;
                        self.steps.push(step);
                    }

                    // Record domain assumption to profiler if present
                    if rewrite.domain_assumption.is_some() {
                        self.profiler
                            .record_domain_assumption(self.current_phase, rule.name());
                    }
                    expr_id = rewrite.new_expr;
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
