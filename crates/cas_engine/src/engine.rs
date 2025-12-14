use crate::canonical_forms::normalize_core;
use crate::profiler::RuleProfiler;
use crate::rule::Rule;
use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{ToPrimitive, Zero};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use tracing::debug;

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
    rules: HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: Vec<Rc<dyn Rule>>,
    pub collect_steps: bool,
    pub allow_numerical_verification: bool,
    pub debug_mode: bool,
    disabled_rules: HashSet<String>,
    pub enable_polynomial_strategy: bool,
    pub profiler: RuleProfiler,
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
            collect_steps: true,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false), // Disabled by default
        }
    }

    pub fn with_default_rules() -> Self {
        let mut s = Self::new();
        s.register_default_rules();
        s
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
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        let rule_rc: Rc<dyn Rule> = rule.into();

        if let Some(targets) = rule_rc.target_types() {
            for target in targets {
                self.rules
                    .entry(target.to_string())
                    .or_default()
                    .push(rule_rc.clone());
            }
        } else {
            self.global_rules.push(rule_rc);
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
            collect_steps: self.collect_steps,
            steps: Vec::new(),
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
        };

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
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
        (result, steps)
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
        orchestrator.options.collect_steps = self.collect_steps;
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
        let rules = &self.rules;
        let global_rules = &self.global_rules;
        let collect_steps = self.collect_steps;

        // Create initial ParentContext with pattern marks
        let initial_parent_ctx =
            crate::parent_context::ParentContext::with_marks(pattern_marks.clone());

        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules,
            global_rules,
            disabled_rules: &self.disabled_rules,
            collect_steps,
            steps: Vec::new(),
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
        };

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
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
        },
        Expr::Matrix { .. } => None, // Matrix evaluation not supported in f64
    }
}

struct LocalSimplificationTransformer<'a> {
    context: &'a mut Context,
    rules: &'a HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: &'a Vec<Rc<dyn Rule>>,
    disabled_rules: &'a HashSet<String>,
    collect_steps: bool,
    steps: Vec<Step>,
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
        // Use tracing for debug logging
        let expr = self.context.get(id);
        debug!("{}[DEBUG] Visiting: {:?}", self.indent(), expr);

        // println!("Visiting: {:?} {:?}", id, self.context.get(id));
        // println!("Simplifying: {:?}", id);
        if let Some(&cached) = self.cache.get(&id) {
            return cached;
        }

        // 1. Simplify children first (Bottom-Up)
        let expr = self.context.get(id).clone();

        let expr_with_simplified_children = match expr {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => id,
            Expr::Add(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();

                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
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
                        self.profiler.record_rejected_disabled(rule.name());
                        continue;
                    }
                    // Phase control: only allow distribution in Transform phase
                    if !self.current_phase.allows_distribution()
                        && rule.name().starts_with("Distributive Property")
                    {
                        self.profiler.record_rejected_phase(rule.name());
                        continue;
                    }
                    // CRITICAL: Use initial_parent_ctx which contains pattern_marks
                    if let Some(rewrite) =
                        rule.apply(self.context, expr_id, &self.initial_parent_ctx)
                    {
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
                                self.profiler.record_rejected_semantic(rule.name());
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
                        self.profiler.record_with_delta(rule.name(), delta);

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
                        if self.collect_steps {
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
                // Phase control: only allow distribution in Transform phase
                if !self.current_phase.allows_distribution()
                    && rule.name().starts_with("Distributive Property")
                {
                    continue;
                }

                // Apply rule with semantic checking
                if let Some(rewrite) = crate::semantic_equality::apply_rule_with_semantic_check(
                    self.context,
                    rule.as_ref(),
                    expr_id,
                ) {
                    // Record rule application for profiling
                    self.profiler.record(rule.name());

                    debug!(
                        "{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}",
                        self.indent(),
                        rule.name(),
                        expr_id,
                        rewrite.new_expr
                    );
                    if self.collect_steps {
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
                        self.steps.push(step);
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
    }
}
