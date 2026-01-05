use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_engine::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_engine::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule};
use cas_engine::Simplifier;

use cas_ast::{
    Context, DisplayExpr, DisplayExprStyled, Expr, ExprId, ParseStyleSignals, StylePreferences,
};
use cas_engine::rules::algebra::{ExpandRule, FactorRule, SimplifyFractionRule};
use cas_engine::rules::calculus::{DiffRule, IntegrateRule};
use cas_engine::rules::grouping::CollectRule;
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_engine::rules::trigonometry::{
    AngleIdentityRule, DoubleAngleRule, EvaluateTrigRule, PythagoreanIdentityRule, TanToSinCosRule,
};
use cas_engine::step::PathStep;
use rustyline::error::ReadlineError;

use crate::completer::CasHelper;
use crate::config::CasConfig;

/// Unwrap top-level __hold() wrapper (used for eager let bindings)
fn unwrap_hold_top(ctx: &Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "__hold" && args.len() == 1 {
            return args[0];
        }
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    None,
    Succinct, // Compact: same filtering as Normal but 1 line per step
    Normal,
    Verbose,
}

/// Clean verbose patterns from display strings for better didactic quality
/// Removes patterns like "1 * x" -> "x" and "x * 1" -> "x"
fn clean_display_string(s: &str) -> String {
    let mut result = s.to_string();
    let mut changed = true;

    // Iterate until no more changes (handles nested patterns)
    while changed {
        let before = result.clone();

        // Replace "1 * " at the very start
        if result.starts_with("1 * ") {
            result = result[4..].to_string();
        }

        // Replace " * 1" at the very end (but not " * 10" etc)
        if result.ends_with(" * 1") && !result.ends_with("0 * 1") {
            result = result[..result.len() - 4].to_string();
        }

        // Order matters: do more specific patterns first

        // "(1 * " at start of parenthesized expression
        result = result.replace("(1 * ", "(");

        // " * 1)" at end of parenthesized expression
        result = result.replace(" * 1)", ")");

        // "1 * 1" -> "1" (common in fraction combination)
        result = result.replace("1 * 1", "1");

        // " + 1 * " -> " + " (after addition)
        result = result.replace(" + 1 * ", " + ");

        // " - 1 * " -> " - " (after subtraction)
        result = result.replace(" - 1 * ", " - ");

        // "/ (1 * " -> "/ ("
        result = result.replace("/ (1 * ", "/ (");

        // " * 1 +" -> " +"
        result = result.replace(" * 1 +", " +");

        // " * 1 -" -> " -"
        result = result.replace(" * 1 -", " -");

        // " * 1 /" -> " /"
        result = result.replace(" * 1 /", " /");

        // " * 1 *" -> " *" (chained multiplications)
        result = result.replace(" * 1 *", " *");

        // Handle edge case: "1 * -" at start (like "1 * -1^2")
        if result.starts_with("1 * -") {
            result = result[4..].to_string();
        }

        // "1 * " folsuccincted by digit at start (like "1 * 2")
        if result.starts_with("1 * ") && result.len() > 4 {
            let next_char = result.chars().nth(4);
            if let Some(c) = next_char {
                if c.is_ascii_digit() || c == '-' || c == 'x' || c == 'y' || c == '(' {
                    result = result[4..].to_string();
                }
            }
        }

        // "/ (1 * x)" -> "/ x" for denominators
        result = result.replace("/ (1 * x)", "/ x");
        result = result.replace("/ (1 * y)", "/ y");

        // Handle "2 * -1 * x" -> "-2 * x" is too complex, instead handle "* -1 * " later
        result = result.replace(" * -1 * ", " * -");

        // "x^2 + 1 * " -> "x^2 + " when folsuccincted by digit
        // Need to handle "... + 1 * y" patterns more aggressively
        // Already have " + 1 * " but it expects space after

        changed = before != result;
    }

    // Remove __hold(...) wrapper - it's an internal invisible barrier
    while let Some(start) = result.find("__hold(") {
        // Find matching closing paren
        let content_start = start + 7; // Length of "__hold("
        let mut depth = 1;
        let mut end = content_start;
        for (i, c) in result[content_start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = content_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let inner = &result[content_start..end];
            result = format!("{}{}{}", &result[..start], inner, &result[end + 1..]);
        } else {
            break; // Malformed, stop
        }
    }

    result
}

/// Render an expression with scoped display transforms based on the rule name.
/// Used for per-step rendering where certain rules (e.g., "Quadratic Formula")
/// should display sqrt notation instead of ^(1/2).
fn render_with_rule_scope(ctx: &Context, id: ExprId, rule_name: &str) -> String {
    // Map rule names to scopes
    let scopes: Vec<cas_ast::display_transforms::ScopeTag> = match rule_name {
        "Quadratic Formula" => vec![cas_ast::display_transforms::ScopeTag::Rule(
            "QuadraticFormula",
        )],
        // Add more rule mappings as needed
        _ => vec![],
    };

    if scopes.is_empty() {
        // No transforms apply - use standard display
        DisplayExpr { context: ctx, id }.to_string()
    } else {
        // Use scoped renderer with transforms
        let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
        let renderer = cas_ast::display_transforms::ScopedRenderer::new(ctx, &scopes, &registry);
        renderer.render(id)
    }
}

pub struct Repl {
    /// The high-level Engine instance (wraps Simplifier)
    pub engine: cas_engine::Engine,
    verbosity: Verbosity,
    config: CasConfig,
    /// Options controlling the simplification pipeline (phases, budgets)
    simplify_options: cas_engine::SimplifyOptions,
    /// When true, show pipeline diagnostics after simplification
    explain_mode: bool,
    /// Last pipeline stats for diagnostics
    last_stats: Option<cas_engine::PipelineStats>,
    /// When true, always track health metrics (independent of explain)
    health_enabled: bool,
    /// Last health report string for `health` command
    last_health_report: Option<String>,
    /// Session state (store + env)
    state: cas_engine::SessionState,
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
/// This is more robust than path-based reconstruction because it finds by identity, not position.
#[allow(dead_code)]
fn substitute_expr_by_id(
    context: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    // If this node is the target, return replacement
    if root == target {
        return replacement;
    }
    match context.get(root).clone() {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l == l && new_r == r {
                root
            } else {
                context.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(base, exp) => {
            let new_base = substitute_expr_by_id(context, base, target, replacement);
            let new_exp = substitute_expr_by_id(context, exp, target, replacement);
            if new_base == base && new_exp == exp {
                root
            } else {
                context.add(Expr::Pow(new_base, new_exp))
            }
        }
        Expr::Neg(e) => {
            let new_e = substitute_expr_by_id(context, e, target, replacement);
            if new_e == e {
                root
            } else {
                context.add(Expr::Neg(new_e))
            }
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|&a| substitute_expr_by_id(context, a, target, replacement))
                .collect();
            if new_args == args {
                root
            } else {
                context.add(Expr::Function(name, new_args))
            }
        }
        _ => root,
    }
}

fn reconstruct_global_expr(
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
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Add(new_l, r))
        }
        // Special case: Sub(a,b) may have been canonicalized to Add(a, Neg(b))
        // When PathStep::Right expects to modify the original "b", we need to
        // traverse into the Neg wrapper and reconstruct there.
        (Expr::Add(l, r), PathStep::Right) => {
            // Check if right side is Neg - if so, this might be a canonicalized Sub
            if let Expr::Neg(inner) = context.get(r).clone() {
                // Traverse into the Neg and wrap result back in Neg
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(l, new_neg))
            } else {
                // Normal case - not a canonicalized Sub
                let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
                context.add(Expr::Add(l, new_r))
            }
        }
        (Expr::Sub(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Sub(new_l, r))
        }
        (Expr::Sub(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Sub(l, new_r))
        }
        (Expr::Mul(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Mul(new_l, r))
        }
        (Expr::Mul(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Mul(l, new_r))
        }
        (Expr::Div(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Div(new_l, r))
        }
        (Expr::Div(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Div(l, new_r))
        }
        (Expr::Pow(b, e), PathStep::Base) => {
            let new_b = reconstruct_global_expr(context, b, remaining_path, replacement);
            context.add(Expr::Pow(new_b, e))
        }
        (Expr::Pow(b, e), PathStep::Exponent) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Pow(b, new_e))
        }
        (Expr::Neg(e), PathStep::Inner) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Neg(new_e))
        }
        (Expr::Function(name, args), PathStep::Arg(idx)) => {
            let mut new_args = args.clone();
            if *idx < new_args.len() {
                new_args[*idx] =
                    reconstruct_global_expr(context, new_args[*idx], remaining_path, replacement);
                context.add(Expr::Function(name, new_args))
            } else {
                root // Should not happen if path is valid
            }
        }
        _ => root, // Path mismatch or invalid structure
    }
}

fn should_show_step(step: &cas_engine::step::Step, verbosity: Verbosity) -> bool {
    use cas_engine::step::ImportanceLevel;

    match verbosity {
        Verbosity::None => false,
        Verbosity::Verbose => true,
        // Succinct and Normal both show Medium+ importance, just different display
        Verbosity::Succinct | Verbosity::Normal => {
            if step.get_importance() < ImportanceLevel::Medium {
                return false;
            }

            // Additional check: global no-ops (expansion then contraction cycles)
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }

            true
        }
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

impl Repl {
    pub fn new() -> Self {
        let config = CasConfig::load();
        let mut simplifier = Simplifier::with_default_rules();

        // Always enabled core rules
        simplifier.add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        simplifier.add_rule(Box::new(EvaluateTrigRule));
        simplifier.add_rule(Box::new(PythagoreanIdentityRule));
        if config.trig_angle_sum {
            simplifier.add_rule(Box::new(AngleIdentityRule));
        }
        simplifier.add_rule(Box::new(TanToSinCosRule));
        if config.trig_double_angle {
            simplifier.add_rule(Box::new(DoubleAngleRule));
        }
        if config.canonicalize_trig_square {
            simplifier.add_rule(Box::new(
                cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        simplifier.add_rule(Box::new(EvaluateLogRule));
        simplifier.add_rule(Box::new(ExponentialLogRule));
        simplifier.add_rule(Box::new(SimplifyFractionRule));
        simplifier.add_rule(Box::new(ExpandRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));
        simplifier.add_rule(Box::new(FactorRule));
        simplifier.add_rule(Box::new(CollectRule));
        simplifier.add_rule(Box::new(EvaluatePowerRule));
        simplifier.add_rule(Box::new(EvaluatePowerRule));
        if config.log_split_exponents {
            simplifier.add_rule(Box::new(
                cas_engine::rules::logarithms::SplitLogExponentsRule,
            ));
        }

        // Advanced Algebra Rules (Critical for Solver)
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::NestedFractionRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::AddFractionsRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::SimplifyMulDivRule));
        if config.rationalize_denominator {
            simplifier.add_rule(Box::new(
                cas_engine::rules::algebra::RationalizeDenominatorRule,
            ));
        }
        simplifier.add_rule(Box::new(
            cas_engine::rules::algebra::CancelCommonFactorsRule,
        ));

        // Configurable rules
        if config.distribute {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeRule));
        }

        if config.expand_binomials {
            simplifier.add_rule(Box::new(
                cas_engine::rules::polynomial::BinomialExpansionRule,
            ));
        }

        if config.factor_difference_squares {
            simplifier.add_rule(Box::new(
                cas_engine::rules::algebra::FactorDifferenceSquaresRule,
            ));
        }

        if config.root_denesting {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::RootDenestingRule));
        }

        if config.auto_factor {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::AutomaticFactorRule));
        }

        simplifier.add_rule(Box::new(
            cas_engine::rules::trigonometry::AngleConsistencyRule,
        ));
        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(AnnihilationRule));
        simplifier.add_rule(Box::new(ProductPowerRule));
        simplifier.add_rule(Box::new(PowerPowerRule));
        simplifier.add_rule(Box::new(PowerProductRule));
        simplifier.add_rule(Box::new(PowerQuotientRule));
        simplifier.add_rule(Box::new(IdentityPowerRule));
        simplifier.add_rule(Box::new(
            cas_engine::rules::exponents::NegativeBasePowerRule,
        ));
        simplifier.add_rule(Box::new(AddZeroRule));
        simplifier.add_rule(Box::new(MulOneRule));
        simplifier.add_rule(Box::new(MulZeroRule));
        simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::DivZeroRule));
        simplifier.add_rule(Box::new(CombineConstantsRule));
        simplifier.add_rule(Box::new(IntegrateRule));
        simplifier.add_rule(Box::new(DiffRule));
        simplifier.add_rule(Box::new(NumberTheoryRule));

        let mut repl = Self {
            engine: cas_engine::Engine { simplifier },
            verbosity: Verbosity::Normal,
            config,
            simplify_options: cas_engine::SimplifyOptions::default(),
            explain_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            state: cas_engine::SessionState::new(),
        };
        repl.sync_config_to_simplifier();
        repl
    }

    /// Simplify expression using current pipeline options
    #[allow(dead_code)]
    fn do_simplify(&mut self, expr: cas_ast::ExprId) -> (cas_ast::ExprId, Vec<cas_engine::Step>) {
        // Use state.options.to_simplify_options() to get correct expand_policy, context_mode, etc.
        // (self.simplify_options is legacy and doesn't sync expand_policy)
        let mut opts = self.state.options.to_simplify_options();
        opts.collect_steps = self.engine.simplifier.collect_steps();

        // Note: Tool dispatcher for collect/expand_log is in Engine::eval (cas_engine/src/eval.rs)
        // This function is dead code but kept for internal use; no dispatcher needed here.

        // Enable health metrics and clear previous run if explain or health mode is on
        if self.explain_mode || self.health_enabled {
            self.engine.simplifier.profiler.enable_health();
            self.engine.simplifier.profiler.clear_run();
        }

        let (result, steps, stats) = self.engine.simplifier.simplify_with_stats(expr, opts);

        // Store health report for the `health` command
        // Always store if health_enabled; for explain-only use threshold
        if self.health_enabled || (self.explain_mode && stats.total_rewrites >= 5) {
            self.last_health_report = Some(self.engine.simplifier.profiler.health_report());
        }

        // Show explain output if enabled
        if self.explain_mode {
            self.print_pipeline_stats(&stats);

            // Policy A+ hint: when simplify makes minimal changes to a Mul expression
            if stats.total_rewrites <= 1
                && matches!(
                    self.engine.simplifier.context.get(result),
                    cas_ast::Expr::Mul(_, _)
                )
            {
                println!("Note: simplify preserves factored products. Use expand(...) to expand.");
            }

            // Show health report if significant activity (>= 5 rewrites)
            if stats.total_rewrites >= 5 {
                println!();
                if let Some(ref report) = self.last_health_report {
                    print!("{}", report);
                }
            }
        }

        self.last_stats = Some(stats.clone());

        // Print assumptions summary if reporting is enabled and there are assumptions
        if self.state.options.assumption_reporting != cas_engine::AssumptionReporting::Off
            && !stats.assumptions.is_empty()
        {
            // Build summary line
            let items: Vec<String> = stats
                .assumptions
                .iter()
                .map(|r| {
                    if r.count > 1 {
                        format!("{}({}) (×{})", r.kind, r.expr, r.count)
                    } else {
                        format!("{}({})", r.kind, r.expr)
                    }
                })
                .collect();
            println!("⚠ Assumptions: {}", items.join(", "));
        }

        (result, steps)
    }

    /// Print pipeline statistics for diagnostics
    #[allow(dead_code)]
    fn print_pipeline_stats(&self, stats: &cas_engine::PipelineStats) {
        println!();
        println!("──── Pipeline Diagnostics ────");
        println!(
            "  Core:       {} iters, {} rewrites",
            stats.core.iters_used, stats.core.rewrites_used
        );
        if let Some(ref cycle) = stats.core.cycle {
            println!(
                "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
                cycle.period, cycle.at_step
            );
            let top = self
                .engine
                .simplifier
                .profiler
                .top_applied_for_phase(cas_engine::SimplifyPhase::Core, 2);
            if !top.is_empty() {
                let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
                println!("              Likely contributors: {}", hints.join(", "));
            }
        }
        println!(
            "  Transform:  {} iters, {} rewrites, changed={}",
            stats.transform.iters_used, stats.transform.rewrites_used, stats.transform.changed
        );
        if let Some(ref cycle) = stats.transform.cycle {
            println!(
                "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
                cycle.period, cycle.at_step
            );
            let top = self
                .engine
                .simplifier
                .profiler
                .top_applied_for_phase(cas_engine::SimplifyPhase::Transform, 2);
            if !top.is_empty() {
                let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
                println!("              Likely contributors: {}", hints.join(", "));
            }
        }
        println!(
            "  Rationalize: {:?}",
            stats
                .rationalize_level
                .unwrap_or(cas_engine::AutoRationalizeLevel::Off)
        );

        if let Some(ref outcome) = stats.rationalize_outcome {
            match outcome {
                cas_engine::RationalizeOutcome::Applied => {
                    println!("              → Applied ✓");
                }
                cas_engine::RationalizeOutcome::NotApplied(reason) => {
                    println!("              → NotApplied: {:?}", reason);
                }
            }
        }
        if let Some(ref cycle) = stats.rationalize.cycle {
            println!(
                "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
                cycle.period, cycle.at_step
            );
            let top = self
                .engine
                .simplifier
                .profiler
                .top_applied_for_phase(cas_engine::SimplifyPhase::Rationalize, 2);
            if !top.is_empty() {
                let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
                println!("              Likely contributors: {}", hints.join(", "));
            }
        }

        println!(
            "  PostCleanup: {} iters, {} rewrites",
            stats.post_cleanup.iters_used, stats.post_cleanup.rewrites_used
        );
        if let Some(ref cycle) = stats.post_cleanup.cycle {
            println!(
                "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
                cycle.period, cycle.at_step
            );
            let top = self
                .engine
                .simplifier
                .profiler
                .top_applied_for_phase(cas_engine::SimplifyPhase::PostCleanup, 2);
            if !top.is_empty() {
                let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
                println!("              Likely contributors: {}", hints.join(", "));
            }
        }
        println!("  Total rewrites: {}", stats.total_rewrites);
        println!("───────────────────────────────");
    }

    fn sync_config_to_simplifier(&mut self) {
        let config = &self.config;

        // Helper to toggle rule
        let mut toggle = |name: &str, enabled: bool| {
            if enabled {
                self.engine.simplifier.enable_rule(name);
            } else {
                self.engine.simplifier.disable_rule(name);
            }
        };

        toggle("Distributive Property", config.distribute);
        toggle("Binomial Expansion", config.expand_binomials);
        toggle("Distribute Constant", config.distribute_constants);
        toggle(
            "Factor Difference of Squares",
            config.factor_difference_squares,
        );
        toggle("Root Denesting", config.root_denesting);
        toggle("Double Angle Identity", config.trig_double_angle);
        toggle("Angle Sum/Diff Identity", config.trig_angle_sum);
        toggle("Split Log Exponents", config.log_split_exponents);
        toggle("Rationalize Denominator", config.rationalize_denominator);
        toggle("Canonicalize Trig Square", config.canonicalize_trig_square);

        // Auto Factor Logic:
        // If auto_factor is on, we enable AutomaticFactorRule AND ConservativeExpandRule.
        // We DISABLE the aggressive ExpandRule to prevent loops.
        if config.auto_factor {
            self.engine
                .simplifier
                .enable_rule("Automatic Factorization");
            self.engine.simplifier.enable_rule("Conservative Expand");
            self.engine.simplifier.disable_rule("Expand Polynomial");
            self.engine.simplifier.disable_rule("Binomial Expansion");
        } else {
            self.engine
                .simplifier
                .disable_rule("Automatic Factorization");
            self.engine.simplifier.disable_rule("Conservative Expand");
            self.engine.simplifier.enable_rule("Expand Polynomial");
            // Re-enable Binomial Expansion if config says so
            if config.expand_binomials {
                self.engine.simplifier.enable_rule("Binomial Expansion");
            }
        }
    }

    /// Build the REPL prompt with mode indicators.
    /// Only shows indicators for non-default modes to keep prompt clean.
    fn build_prompt(&self) -> String {
        use cas_engine::options::{BranchMode, ComplexMode, ContextMode, StepsMode};

        let mut indicators = Vec::new();

        // Show steps mode if not On (default)
        match self.state.options.steps_mode {
            StepsMode::Off => indicators.push("[steps:off]"),
            StepsMode::Compact => indicators.push("[steps:compact]"),
            StepsMode::On => {} // Default, no indicator
        }

        // Show context mode if not Auto (default)
        match self.state.options.context_mode {
            ContextMode::IntegratePrep => indicators.push("[ctx:integrate]"),
            ContextMode::Solve => indicators.push("[ctx:solve]"),
            ContextMode::Standard => indicators.push("[ctx:standard]"),
            ContextMode::Auto => {} // Default, no indicator
        }

        // Show branch mode if not Strict (default)
        match self.state.options.branch_mode {
            BranchMode::PrincipalBranch => indicators.push("[branch:principal]"),
            BranchMode::Strict => {} // Default, no indicator
        }

        // Show complex mode if not Auto (default)
        match self.state.options.complex_mode {
            ComplexMode::On => indicators.push("[cx:on]"),
            ComplexMode::Off => indicators.push("[cx:off]"),
            ComplexMode::Auto => {} // Default, no indicator
        }

        // Show expand_policy if Auto (not default Off)
        use cas_engine::phase::ExpandPolicy;
        if self.state.options.expand_policy == ExpandPolicy::Auto {
            indicators.push("[autoexp:on]");
        }

        if indicators.is_empty() {
            "> ".to_string()
        } else {
            format!("{} > ", indicators.join(""))
        }
    }

    pub fn run(&mut self) -> rustyline::Result<()> {
        println!("Rust CAS Step-by-Step Demo");
        println!("Step-by-step output enabled (Normal).");
        println!("Enter an expression (e.g., '2 * 3 + 0'):");

        let helper = CasHelper::new();
        let config = rustyline::Config::builder()
            .max_history_size(100)?
            .completion_type(rustyline::CompletionType::List)
            .build();
        let mut rl =
            rustyline::Editor::<CasHelper, rustyline::history::DefaultHistory>::with_config(
                config,
            )?;
        rl.set_helper(Some(helper));

        // History file path: ~/.cas_history
        let history_path = dirs::home_dir()
            .map(|p| p.join(".cas_history"))
            .unwrap_or_else(|| std::path::PathBuf::from(".cas_history"));

        // Load history if file exists (errors are silently ignored)
        let _ = rl.load_history(&history_path);

        loop {
            let prompt = self.build_prompt();
            let readline = rl.readline(&prompt);
            match readline {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    rl.add_history_entry(line)?;

                    if line == "quit" || line == "exit" {
                        println!("Goodbye!");
                        break;
                    }

                    // Split by semicolon to allow multiple statements on one line
                    // e.g., "let a = 3*x; let b = 4*x; a + b"
                    for statement in line.split(';') {
                        let statement = statement.trim();
                        if statement.is_empty() {
                            continue;
                        }
                        self.handle_command(statement);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }

        // Save history on exit (errors are silently ignored)
        let _ = rl.save_history(&history_path);

        Ok(())
    }

    /// Converts function-style commands to command-style
    /// Examples:
    ///   simplify(...) -> simplify x^2 + 1
    ///   solve(...) -> solve x + 2 = 5, x
    fn preprocess_function_syntax(&self, line: &str) -> String {
        let line = line.trim();

        // Check for simplify(...)
        if line.starts_with("simplify(") && line.ends_with(")") {
            let content = &line["simplify(".len()..line.len() - 1];
            return format!("simplify {}", content);
        }

        // Check for solve(...)
        if line.starts_with("solve(") && line.ends_with(")") {
            let content = &line["solve(".len()..line.len() - 1];
            return format!("solve {}", content);
        }

        // Return unchanged
        line.to_string()
    }

    pub fn handle_command(&mut self, line: &str) {
        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);

        // Check for "help" command
        if line.starts_with("help") {
            self.handle_help(&line);
            return;
        }

        // ========== SESSION ENVIRONMENT COMMANDS ==========

        // "let <name> = <expr>" - assign variable
        if let Some(rest) = line.strip_prefix("let ") {
            self.handle_let_command(rest);
            return;
        }

        // "<name> := <expr>" - alternative assignment syntax
        if let Some(idx) = line.find(":=") {
            let name = line[..idx].trim();
            let expr_str = line[idx + 2..].trim();
            if !name.is_empty() && !expr_str.is_empty() {
                self.handle_assignment(name, expr_str, true); // := is lazy
                return;
            }
        }

        // "vars" - list all variables
        if line == "vars" {
            self.handle_vars_command();
            return;
        }

        // "clear" or "clear <names>" - clear variables
        if line == "clear" || line.starts_with("clear ") {
            self.handle_clear_command(&line);
            return;
        }

        // "reset" - reset entire session
        if line == "reset" {
            self.handle_reset_command();
            return;
        }

        // "reset full" - reset everything including profile cache
        if line == "reset full" {
            self.handle_reset_full_command();
            return;
        }

        // "cache clear" - clear only profile cache
        if line == "cache clear" || line == "cache" {
            self.handle_cache_command(&line);
            return;
        }

        // "semantics" - unified semantic settings (domain, value, branch, inv_trig, const_fold)
        if line == "semantics" || line.starts_with("semantics ") {
            self.handle_semantics(&line);
            return;
        }

        // "mode" - DEPRECATED: redirect to semantics
        if line == "mode" || line.starts_with("mode ") {
            println!("⚠️  The 'mode' command is deprecated.");
            println!("Use 'semantics set inv_trig strict|principal' instead.");
            println!("Run 'semantics' to see current settings.");
            return;
        }

        // "context" - show/switch context mode (auto, standard, solve, integrate)
        if line == "context" || line.starts_with("context ") {
            self.handle_context_command(&line);
            return;
        }

        // "complex" - DEPRECATED: redirect to semantics
        if line == "complex" || line.starts_with("complex ") {
            println!("⚠️  The 'complex' command is deprecated.");
            println!("Use 'semantics set value real|complex' instead.");
            println!("Run 'semantics' to see current settings.");
            return;
        }

        // "steps" - show/switch steps collection mode (on, off, compact)
        if line == "steps" || line.starts_with("steps ") {
            self.handle_steps_command(&line);
            return;
        }

        // "autoexpand" - show/switch auto-expand policy (on, off)
        if line == "autoexpand" || line.starts_with("autoexpand ") {
            self.handle_autoexpand_command(&line);
            return;
        }

        // "budget" - V2.0: Control Conditional branching budget for solve
        if line == "budget" || line.starts_with("budget ") {
            self.handle_budget_command(&line);
            return;
        }

        // "history" or "list" - show session history
        if line == "history" || line == "list" {
            self.handle_history_command();
            return;
        }

        // "show #id" - show a specific session entry
        if let Some(rest) = line.strip_prefix("show ") {
            self.handle_show_command(rest);
            return;
        }

        // "del #id [#id...]" - delete session entries
        if let Some(rest) = line.strip_prefix("del ") {
            self.handle_del_command(rest);
            return;
        }

        // ========== END SESSION ENVIRONMENT COMMANDS ==========

        // Check for "set" command (pipeline options)
        if line.starts_with("set ") {
            self.handle_set_command(&line);
            return;
        }

        // Check for "help" command (duplicate check in original code?)
        if line == "help" {
            self.print_general_help();
            return;
        }

        // Check for "equiv" command
        if line.starts_with("equiv ") {
            self.handle_equiv(&line);
            return;
        }

        // Check for "subst" command
        if line.starts_with("subst ") {
            self.handle_subst(&line);
            return;
        }

        // Check for "solve" command
        if line.starts_with("solve ") {
            self.handle_solve(&line);
            return;
        }

        // Check for "simplify" command
        if line.starts_with("simplify ") {
            self.handle_full_simplify(&line);
            return;
        }

        // Check for "config" command
        if line.starts_with("config ") {
            self.handle_config(&line);
            return;
        }

        // Check for "timeline" command
        if line.starts_with("timeline ") {
            self.handle_timeline(&line);
            return;
        }

        // Check for "visualize" command
        if line.starts_with("visualize ") {
            self.handle_visualize(&line);
            return;
        }

        // Check for "explain" command
        if line.starts_with("explain ") {
            self.handle_explain(&line);
            return;
        }

        // Check for "det" command
        if line.starts_with("det ") {
            self.handle_det(&line);
            return;
        }

        // Check for "transpose" command
        if line.starts_with("transpose ") {
            self.handle_transpose(&line);
            return;
        }

        // Check for "trace" command
        if line.starts_with("trace ") {
            self.handle_trace(&line);
            return;
        }

        // Check for "telescope" command - for proving telescoping identities
        if line.starts_with("telescope ") {
            self.handle_telescope(&line);
            return;
        }

        // Check for "weierstrass" command - Weierstrass substitution (t = tan(x/2))
        if line.starts_with("weierstrass ") {
            self.handle_weierstrass(&line);
            return;
        }

        // Check for "expand_log" command - explicit logarithm expansion
        // MUST come before "expand" check due to prefix matching
        if line.starts_with("expand_log ") || line == "expand_log" {
            self.handle_expand_log(&line);
            return;
        }

        // Check for "expand" command - aggressive expansion/distribution
        if line.starts_with("expand ") {
            self.handle_expand(&line);
            return;
        }

        // Check for "rationalize" command - rationalize denominators with surds
        if line.starts_with("rationalize ") {
            self.handle_rationalize(&line);
            return;
        }

        // Check for "limit" command - compute limits at infinity
        if line.starts_with("limit ") {
            self.handle_limit(&line);
            return;
        }

        // Check for "profile" commands
        if line.starts_with("profile") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "profile" - show report
                println!("{}", self.engine.simplifier.profiler.report());
            } else {
                match parts[1] {
                    "enable" => {
                        self.engine.simplifier.profiler.enable();
                        println!("Profiler enabled.");
                    }
                    "disable" => {
                        self.engine.simplifier.profiler.disable();
                        println!("Profiler disabled.");
                    }
                    "clear" => {
                        self.engine.simplifier.profiler.clear();
                        println!("Profiler statistics cleared.");
                    }
                    _ => println!("Usage: profile [enable|disable|clear]"),
                }
            }
            return;
        }

        // Check for "health" commands
        if line.starts_with("health") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "health" - show last report
                // First show any cycles detected
                if let Some(ref stats) = self.last_stats {
                    let cycles: Vec<_> = [
                        (&stats.core.cycle, "Core"),
                        (&stats.transform.cycle, "Transform"),
                        (&stats.rationalize.cycle, "Rationalize"),
                        (&stats.post_cleanup.cycle, "PostCleanup"),
                    ]
                    .iter()
                    .filter_map(|(c, name)| c.as_ref().map(|info| (*name, info)))
                    .collect();

                    for (phase_name, cycle) in &cycles {
                        println!(
                            "⚠ Cycle detected in {}: period={} at rewrite={} (stopped early)",
                            phase_name, cycle.period, cycle.at_step
                        );
                    }
                    if !cycles.is_empty() {
                        println!();
                    }
                }

                if let Some(ref report) = self.last_health_report {
                    println!("{}", report);
                } else {
                    println!("No health report available.");
                    println!("Run a simplification first (health is captured when explain mode or health mode is on).");
                    println!("Enable with: health on");
                }
            } else {
                match parts[1] {
                    "on" | "enable" => {
                        self.health_enabled = true;
                        println!("Health tracking ENABLED (metrics captured after each simplify)");
                    }
                    "off" | "disable" => {
                        self.health_enabled = false;
                        println!("Health tracking DISABLED");
                    }
                    "reset" | "clear" => {
                        self.engine.simplifier.profiler.clear_run();
                        self.last_health_report = None;
                        println!("Health statistics cleared.");
                    }
                    "status" => {
                        // Parse options: status [--list | --category <cat>]
                        let opts: Vec<&str> = parts.iter().skip(2).copied().collect();

                        if opts.contains(&"--list") || opts.contains(&"-l") {
                            // List available cases
                            println!("{}", crate::health_suite::list_cases());
                            return;
                        }

                        // Check for --category
                        let category_filter = if let Some(idx) =
                            opts.iter().position(|&x| x == "--category" || x == "-c")
                        {
                            if let Some(cat_str) = opts.get(idx + 1) {
                                if *cat_str == "all" {
                                    None
                                } else {
                                    match cat_str.parse::<crate::health_suite::Category>() {
                                        Ok(cat) => Some(cat),
                                        Err(e) => {
                                            println!("Error: {}", e);
                                            println!(
                                                "Available categories: {}",
                                                crate::health_suite::category_names().join(", ")
                                            );
                                            return;
                                        }
                                    }
                                }
                            } else {
                                println!("Error: --category requires an argument");
                                println!(
                                    "Available categories: {}",
                                    crate::health_suite::category_names().join(", ")
                                );
                                return;
                            }
                        } else {
                            None // Run all
                        };

                        // Run the health status suite
                        let cat_msg = category_filter.map_or("all".to_string(), |c| c.to_string());
                        println!("Running health status suite [category={}]...\n", cat_msg);

                        let results = crate::health_suite::run_suite_filtered(
                            &mut self.engine.simplifier,
                            category_filter,
                        );
                        let report =
                            crate::health_suite::format_report_filtered(&results, category_filter);
                        println!("{}", report);

                        let (_passed, failed) = crate::health_suite::count_results(&results);
                        if failed > 0 {
                            println!(
                                "\n⚠ {} tests failed. Check Transform rules for churn.",
                                failed
                            );
                        }
                    }
                    _ => {
                        println!("Usage: health [on|off|reset|status]");
                        println!("       health               Show last health report");
                        println!("       health on            Enable health tracking");
                        println!("       health off           Disable health tracking");
                        println!("       health reset         Clear health statistics");
                        println!("       health status        Run diagnostic test suite");
                        println!("       health status --list List available test cases");
                        println!("       health status --category <cat>  Run only category");
                        println!(
                            "                            Categories: {}",
                            crate::health_suite::category_names().join(", ")
                        );
                    }
                }
            }
            return;
        }

        self.handle_eval(&line);
    }

    fn handle_config(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Usage: config <list|enable|disable|save|restore> [rule]");
            return;
        }

        match parts[1] {
            "list" => {
                println!("Current Configuration:");
                println!("  distribute: {}", self.config.distribute);
                println!("  expand_binomials: {}", self.config.expand_binomials);
                println!(
                    "  distribute_constants: {}",
                    self.config.distribute_constants
                );
                println!(
                    "  factor_difference_squares: {}",
                    self.config.factor_difference_squares
                );
                println!("  root_denesting: {}", self.config.root_denesting);
                println!("  trig_double_angle: {}", self.config.trig_double_angle);
                println!("  trig_angle_sum: {}", self.config.trig_angle_sum);
                println!("  log_split_exponents: {}", self.config.log_split_exponents);
                println!(
                    "  rationalize_denominator: {}",
                    self.config.rationalize_denominator
                );
                println!(
                    "  canonicalize_trig_square: {}",
                    self.config.canonicalize_trig_square
                );
                println!("  auto_factor: {}", self.config.auto_factor);
            }
            "save" => match self.config.save() {
                Ok(_) => println!("Configuration saved to cas_config.toml"),
                Err(e) => println!("Error saving configuration: {}", e),
            },
            "restore" => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                println!("Configuration restored to defaults.");
            }
            "enable" | "disable" => {
                if parts.len() < 3 {
                    println!("Usage: config {} <rule>", parts[1]);
                    return;
                }
                let rule = parts[2];
                let enable = parts[1] == "enable";

                let mut changed = true;
                match rule {
                    "distribute" => self.config.distribute = enable,
                    "expand_binomials" => self.config.expand_binomials = enable,
                    "distribute_constants" => self.config.distribute_constants = enable,
                    "factor_difference_squares" => self.config.factor_difference_squares = enable,
                    "root_denesting" => self.config.root_denesting = enable,
                    "trig_double_angle" => self.config.trig_double_angle = enable,
                    "trig_angle_sum" => self.config.trig_angle_sum = enable,
                    "log_split_exponents" => self.config.log_split_exponents = enable,
                    "rationalize_denominator" => self.config.rationalize_denominator = enable,
                    "canonicalize_trig_square" => self.config.canonicalize_trig_square = enable,
                    "auto_factor" => self.config.auto_factor = enable,
                    _ => {
                        println!("Unknown rule: {}", rule);
                        changed = false;
                    }
                }

                if changed {
                    self.sync_config_to_simplifier();
                    println!("Rule '{}' set to {}.", rule, enable);
                }
            }
            _ => println!("Unknown config command: {}", parts[1]),
        }
    }

    /// Handle 'set' command for pipeline phase control
    fn handle_set_command(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            self.print_set_help();
            return;
        }

        match parts[1] {
            "transform" => match parts[2] {
                "on" | "true" | "1" => {
                    self.simplify_options.enable_transform = true;
                    println!("Transform phase ENABLED (distribution, expansion)");
                }
                "off" | "false" | "0" => {
                    self.simplify_options.enable_transform = false;
                    println!("Transform phase DISABLED (no distribution/expansion)");
                }
                _ => println!("Usage: set transform <on|off>"),
            },
            "rationalize" => match parts[2] {
                "on" | "true" | "auto" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level15;
                    println!("Rationalization ENABLED (Level 1.5)");
                }
                "off" | "false" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Off;
                    println!("Rationalization DISABLED");
                }
                "0" | "level0" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level0;
                    println!("Rationalization set to Level 0 (single sqrt)");
                }
                "1" | "level1" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level1;
                    println!("Rationalization set to Level 1 (binomial conjugate)");
                }
                "1.5" | "level15" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level15;
                    println!("Rationalization set to Level 1.5 (same-surd products)");
                }
                _ => println!("Usage: set rationalize <on|off|0|1|1.5>"),
            },
            "max-rewrites" => {
                if let Ok(n) = parts[2].parse::<usize>() {
                    self.simplify_options.budgets.max_total_rewrites = n;
                    println!("Max rewrites set to {}", n);
                } else {
                    println!("Usage: set max-rewrites <number>");
                }
            }
            "explain" => match parts[2] {
                "on" | "true" | "1" => {
                    self.explain_mode = true;
                    println!("Explain mode ENABLED (pipeline diagnostics after each simplify)");
                }
                "off" | "false" | "0" => {
                    self.explain_mode = false;
                    println!("Explain mode DISABLED");
                }
                _ => println!("Usage: set explain <on|off>"),
            },
            _ => self.print_set_help(),
        }
    }

    fn print_set_help(&self) {
        println!("Pipeline settings:");
        println!("  set transform <on|off>         Enable/disable distribution & expansion");
        println!("  set rationalize <on|off|0|1|1.5>  Set rationalization level");
        println!("  set max-rewrites <N>           Set max total rewrites (safety limit)");
        println!("  set explain <on|off>           Show pipeline diagnostics after simplify");
        println!();
        println!("Current settings:");
        println!(
            "  transform: {}",
            if self.simplify_options.enable_transform {
                "on"
            } else {
                "off"
            }
        );
        println!(
            "  rationalize: {:?}",
            self.simplify_options.rationalize.auto_level
        );
        println!(
            "  max-rewrites: {}",
            self.simplify_options.budgets.max_total_rewrites
        );
        println!(
            "  explain: {}",
            if self.explain_mode { "on" } else { "off" }
        );
    }

    fn handle_help(&self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            self.print_general_help();
            return;
        }

        match parts[1] {
            "simplify" => {
                println!("Command: simplify <expr>");
                println!(
                    "Description: Simplifies an expression using the full power of the engine."
                );
                println!(
                    "             This includes aggressive distribution and other rules that may"
                );
                println!("             undo factorizations, but guarantee maximum simplification.");
                println!("Example: simplify (x+1)*(x-1) -> x^2 - 1");
            }
            "diff" => {
                println!("Command: diff <expr> <var>");
                println!("Description: Computes the symbolic derivative of an expression with respect to a variable.");
                println!("             Supports basic arithmetic, power rule, chain rule, and common functions.");
                println!("Example: diff(sin(x^2), x) -> 2*x*cos(x^2)");
            }
            "sum" => {
                println!("Function: sum(expr, var, start, end)");
                println!("Description: Evaluates finite summations Σ(var=start to end) expr.");
                println!("             Supports numeric evaluation and telescoping detection.");
                println!("Features:");
                println!("  - Numeric: sum(k, k, 1, 10) -> 55");
                println!("  - Symbolic end: sum(1/(k*(k+1)), k, 1, n) -> 1 - 1/(n+1)");
                println!("  - Telescoping: Automatically detects 1/(k*(k+a)) patterns");
                println!("Examples:");
                println!("  sum(k^2, k, 1, 5)           -> 55");
                println!("  sum(1/(k*(k+1)), k, 1, n)   -> 1 - 1/(1+n)");
                println!("  sum(1/(k*(k+2)), k, 1, n)   -> 1/2 - 1/2/(1+n)");
            }
            "product" => {
                println!("Function: product(expr, var, start, end)");
                println!("Description: Evaluates finite products Π(var=start to end) expr.");
                println!("             Supports numeric evaluation and telescoping detection.");
                println!("Features:");
                println!("  - Numeric: product(k, k, 1, 5) -> 120 (5!)");
                println!("  - Symbolic end: product((k+1)/k, k, 1, n) -> n+1");
                println!("  - Telescoping: Detects (k+a)/(k+b) quotient patterns");
                println!("Examples:");
                println!("  product(k, k, 1, 5)         -> 120");
                println!("  product((k+1)/k, k, 1, n)   -> 1 + n");
                println!("  product((k+1)/k, k, 1, 10)  -> 11");
            }
            "gcd" => {
                println!("Function: gcd <a, b>");
                println!("Description: Computes the Greatest Common Divisor of two integers.");
                println!("Example: gcd(12, 18) -> 6");
            }
            "lcm" => {
                println!("Function: lcm <a, b>");
                println!("Description: Computes the Least Common Multiple of two integers.");
                println!("Example: lcm(4, 6) -> 12");
            }
            "mod" => {
                println!("Function: mod <a, n>");
                println!(
                    "Description: Computes the remainder of a divided by n (Euclidean modulo)."
                );
                println!("Example: mod(10, 3) -> 1");
            }
            "factors" | "prime_factors" => {
                println!("Function: factors <n>");
                println!("Description: Computes the prime factorization of an integer.");
                println!("Example: factors(12) -> 2^2 * 3");
            }
            "fact" | "factorial" => {
                println!("Function: fact <n> or <n>!");
                println!("Description: Computes the factorial of a non-negative integer.");
                println!("Example: fact(5) -> 120, 5! -> 120");
            }
            "choose" | "nCr" => {
                println!("Function: choose <n, k>");
                println!("Description: Computes the binomial coefficient nCk (combinations).");
                println!("Example: choose(5, 2) -> 10");
            }
            "perm" | "nPr" => {
                println!("Function: perm <n, k>");
                println!("Description: Computes the number of permutations nPk.");
                println!("Example: perm(5, 2) -> 20");
            }
            "config" => {
                println!("Command: config <subcommand> [args]");
                println!("Description: Manages CLI configuration.");
                println!("Subcommands:");
                println!("  list             Show current configuration");
                println!("  enable <rule>    Enable a simplification rule");
                println!("  disable <rule>   Disable a simplification rule");
                println!("  save             Save configuration to file");
                println!("  restore          Restore default configuration");
                println!("Rules:");
                println!("  distribute       Aggressive distribution (a*(b+c) -> a*b + a*c)");
                println!("  distribute_constants Safe distribution (-1*(x+y) -> -x-y)");
                println!("  expand_binomials Expand powers ((a+b)^2 -> a^2+2ab+b^2)");
                println!("  factor_difference_squares Factor difference of squares (a^2-b^2 -> (a-b)(a+b))");
            }
            "subst" => {
                println!("Command: subst <expr>, <target>, <replacement>");
                println!("Description: Substitutes a pattern with a replacement and simplifies.");
                println!();
                println!("Variable substitution:");
                println!("  subst x^2 + x, x, 3           → 12");
                println!();
                println!("Power-aware expression substitution:");
                println!("  subst x^4 + x^2 + 1, x^2, y   → y² + y + 1");
                println!("  subst x^3, x^2, y             → y·x (with remainder)");
                println!("  subst x^6, x^2, y             → y³");
            }
            "expand" => {
                println!("Command: expand <expr>");
                println!("Description: Expands polynomials and products.");
                println!("Example: expand(x+1)^2 -> x^2 + 2*x + 1");
            }
            "factor" => {
                println!("Command: factor <expr>");
                println!("Description: Factors polynomials.");
                println!("Example: factor(x^2 - 1) -> (x - 1) * (x + 1)");
            }
            "collect" => {
                println!("Command: collect <expr>, <var>");
                println!("Description: Groups terms by powers of a variable.");
                println!("Example: collect(a*x + b*x + c, x) -> (a + b) * x + c");
            }
            "equiv" => {
                println!("Command: equiv <expr1>, <expr2>");
                println!("Description: Checks if two expressions are mathematically equivalent.");
                println!("             Returns true if expr1 - expr2 simplifies to 0.");
            }
            "solve" => {
                println!("Command: solve <equation>, <var>");
                println!("Description: Solves an equation for a variable.");
                println!("Example: solve x + 2 = 5, x -> x = 3");
            }
            "steps" => {
                println!("Command: steps <level>");
                println!("Description: Controls the verbosity of simplification steps.");
                println!("Levels:");
                println!("  normal (or on)   Show clarifying steps (Global state). Default.");
                println!("  succinct              Compact: same steps as normal but 1 line each.");
                println!("  verbose          Show all steps (Local + Global details).");
                println!("  none (or off)    Disable step output.");
            }
            "profile" => {
                println!("Command: profile [subcommand]");
                println!("Description: Rule profiler for debugging and performance analysis.");
                println!("Subcommands:");
                println!("  (none)           Show profiling report");
                println!("  enable           Enable profiler");
                println!("  disable          Disable profiler");
                println!("  clear            Clear statistics");
                println!("Example: profile enable, then run expressions, then profile");
            }
            "visualize" => {
                println!("Command: visualize <expr>");
                println!("Description: Export expression tree to Graphviz DOT format.");
                println!("             Generates ast.dot file for rendering.");
                println!("Example: visualize (x+1)*(x-1)");
                println!();
                println!("To render the generated file, use Graphviz in your terminal:");
                println!("  $ dot -Tsvg ast.dot -o ast.svg");
                println!("  $ open ast.svg");
            }
            "timeline" => {
                println!("Command: timeline <expr>");
                println!("Description: Export simplification steps to interactive HTML.");
                println!("             Generates timeline.html with MathJax rendering.");
                println!("Example: timeline (x+1)^2");
                println!("         Open timeline.html in browser to view.");
            }
            "explain" => {
                println!("Command: explain <function>");
                println!(
                    "Description: Provides step-by-step educational explanations of mathematical"
                );
                println!("             operations. Shows the detailed algorithm steps in Spanish.");
                println!("Supported functions:");
                println!("  gcd(a, b)    Greatest Common Divisor using Euclidean algorithm");
                println!("               Works for both integers and polynomials.");
                println!("Examples:");
                println!("  explain gcd(48, 18)");
                println!("  explain gcd(2*x^2 + 7*x + 3, 2*x^2 + 5*x + 2)");
            }
            "det" => {
                println!("Command: det <matrix>");
                println!("Description: Compute the determinant of a square matrix.");
                println!("             Supports 1×1, 2×2, and 3×3 matrices.");
                println!("Examples:");
                println!("  det [[1, 2], [3, 4]]        → -2");
                println!("  det [[2]]                    → 2");
                println!("  det [[1, 2, 3], [4, 5, 6], [7, 8, 9]]");
            }
            "transpose" => {
                println!("Command: transpose <matrix>");
                println!("Description: Transpose a matrix (swap rows and columns).");
                println!("             Works with any rectangular matrix.");
                println!("Examples:");
                println!("  transpose [[1, 2, 3], [4, 5, 6]]");
                println!("    → [[1, 4], [2, 5], [3, 6]]");
                println!("  transpose [[1, 2], [3, 4]]");
                println!("    → [[1, 3], [2, 4]]");
            }
            "trace" => {
                println!("Command: trace <matrix>");
                println!("Description: Compute the trace of a square matrix.");
                println!("             The trace is the sum of diagonal elements.");
                println!("Examples:");
                println!("  trace [[1, 2], [3, 4]]      → 5");
                println!("  trace [[5, 0, 0], [0, 3, 0], [0, 0, 2]]  → 10");
            }
            "rationalize" => {
                println!("Command: rationalize <expr>");
                println!("Description: Rationalize denominators containing surds (square roots).");
                println!(
                    "             Eliminates irrational numbers from denominators by multiplying"
                );
                println!("             by the conjugate.");
                println!("Examples:");
                println!("  rationalize 1/(1 + sqrt(2))      → √2 - 1");
                println!("  rationalize 1/(3 - 2*sqrt(5))    → -(3 + 2*√5)/11");
                println!("  rationalize x/(sqrt(3) + 1)      → x*(√3 - 1)/2");
            }
            "status" | "health" => {
                println!("Command: health [on|off|reset|status]");
                println!("Description: Engine health monitoring and diagnostic test suite.");
                println!();
                println!("Subcommands:");
                println!("  health on                Enable profiler");
                println!("  health off               Disable profiler");
                println!("  health reset             Reset profiler stats");
                println!("  health status            Run diagnostic test suite");
                println!();
                println!("Test suite options:");
                println!("  health status --list             List all test cases");
                println!("  health status --category <cat>   Run specific category");
                println!("  health status -c <cat>           Shorthand for --category");
                println!();
                println!("Categories: transform, expansion, fractions, rationalization,");
                println!("            mixed, baseline, roots, powers, stress, all");
                println!();
                println!("Examples:");
                println!("  health status                Run all test categories");
                println!("  health status -c stress      Run only stress tests");
                println!("  health status --list         List available tests");
            }
            // Session environment commands
            "let" => {
                println!("Command: let <name> = <expr>");
                println!("Description: Assigns an expression to a variable name.");
                println!("             The variable can be used in subsequent expressions.");
                println!("             Substitution is transitive and cycle-safe.");
                println!();
                println!("Examples:");
                println!("  let a = 5");
                println!("  let b = a + 1        → b becomes 6");
                println!("  let f = x^2 + 1      → f stores symbolic expression");
            }
            "vars" => {
                println!("Command: vars");
                println!("Description: Lists all defined variables and their values.");
                println!();
                println!("Example output:");
                println!("  a = 5");
                println!("  b = 6");
                println!("  f = x^2 + 1");
            }
            "clear" => {
                println!("Command: clear [name ...]");
                println!("Description: Clears variable bindings from the environment.");
                println!("             Without arguments, clears ALL variables.");
                println!("             With arguments, clears only the specified variables.");
                println!();
                println!("Examples:");
                println!("  clear           → clears all variables");
                println!("  clear a b       → clears only a and b");
            }
            "reset" => {
                println!("Command: reset");
                println!("Description: Resets the entire session state.");
                println!("             Clears all variables AND session history (#ids).");
            }
            "history" | "list" => {
                println!("Command: history (or list)");
                println!("Description: Shows all stored session entries with their #ids.");
                println!("             Each expression you evaluate is stored with a unique ID.");
                println!();
                println!("Example output:");
                println!("  #1: x + 1");
                println!("  #2: 2*x - 3");
                println!("  #3: x + 1 = 5  [Eq]");
            }
            "show" => {
                println!("Command: show #<id>");
                println!("Description: Displays a specific session entry by its ID.");
                println!();
                println!("Example:");
                println!("  show #1         → shows the expression stored as #1");
            }
            "del" => {
                println!("Command: del #<id> [#<id> ...]");
                println!("Description: Deletes session entries by their IDs.");
                println!("             IDs are never reused after deletion.");
                println!();
                println!("Examples:");
                println!("  del #1          → deletes entry #1");
                println!("  del #2 #3 #5    → deletes entries #2, #3, and #5");
            }
            "poly_gcd" | "pgcd" => {
                println!("Command: poly_gcd(expr1, expr2)");
                println!("Alias: pgcd(expr1, expr2)");
                println!("Description: Computes the STRUCTURAL GCD of two polynomial expressions.");
                println!(
                    "             Finds common factors that appear explicitly as multiplicands."
                );
                println!("             Does NOT factor expressions to find hidden common factors.");
                println!();
                println!("Examples:");
                println!("  poly_gcd((x+1)*(y+2), (x+1)*(z+3)) → (x + 1)");
                println!("  poly_gcd((x+1)^3, (x+1)^2)         → (x + 1)²");
                println!("  poly_gcd(x*g, y*g)                 → g");
                println!();
                println!("See also: poly_gcd_exact for algebraic GCD");
            }
            "poly_gcd_exact" | "pgcdx" => {
                println!("Command: poly_gcd_exact(expr1, expr2)");
                println!("Alias: pgcdx(expr1, expr2)");
                println!(
                    "Description: Computes the ALGEBRAIC GCD of two polynomials over ℚ[x₁,...,xₙ]."
                );
                println!(
                    "             Interprets expressions as polynomials and finds the true GCD."
                );
                println!("             Uses Euclidean algorithm for univariate, interpolation for multivariate.");
                println!();
                println!("Examples:");
                println!("  poly_gcd_exact(x^2 - 1, x - 1)         → x - 1");
                println!("  poly_gcd_exact(x^2 - 1, x^2 - 2*x + 1) → x - 1");
                println!("  poly_gcd_exact(2*x + 2*y, 4*x + 4*y)   → x + y");
                println!("  poly_gcd_exact(6, 15)                  → 1 (constants over ℚ)");
                println!();
                println!("Result is normalized: primitive (GCD of coefficients = 1), positive leading coefficient.");
                println!();
                println!("See also: poly_gcd for structural (visible factor) GCD");
            }
            "limit" => {
                println!("Command: limit <expr> [, <var> [, <direction>]]");
                println!();
                println!("Description: Compute the limit of an expression as a variable approaches infinity.");
                println!("             Uses polynomial degree comparison for rational functions P(x)/Q(x).");
                println!();
                println!("Arguments:");
                println!("  <expr>       Expression to evaluate the limit of");
                println!("  <var>        Variable approaching the limit (default: x)");
                println!("  <direction>  Direction: infinity or -infinity (default: infinity)");
                println!();
                println!("Examples:");
                println!("  limit x^2                      → infinity");
                println!("  limit (x^2+1)/(2*x^2-3), x     → 1/2");
                println!("  limit x^3/x^2, x, -infinity    → -infinity");
                println!("  limit x^2/x^3                  → 0");
                println!();
                println!("Behavior:");
                println!("  - deg(P) < deg(Q): limit = 0");
                println!("  - deg(P) = deg(Q): limit = leading_coeff(P) / leading_coeff(Q)");
                println!(
                    "  - deg(P) > deg(Q): limit = ±∞ (sign depends on coefficients and approach)"
                );
                println!();
                println!("Residuals:");
                println!("  If the limit cannot be determined (e.g., sin(x)/x, non-polynomial expressions),");
                println!("  returns limit(...) as a symbolic residual with a warning.");
            }
            _ => {
                println!("Unknown command: {}", parts[1]);
                self.print_general_help();
            }
        }
    }

    fn print_general_help(&self) {
        println!("Rust CAS Commands:");
        println!();

        println!("Basic Operations:");
        println!("  <expr>                  Evaluate and simplify an expression");
        println!("  simplify <expr>         Aggressive simplification (full power)");
        println!("  expand <expr>           Expand polynomials");
        println!("  factor <expr>           Factor polynomials");
        println!("  collect <expr>, <var>   Group terms by variable");
        println!();

        println!("Polynomial GCD:");
        println!("  poly_gcd(a, b)          Structural GCD (visible factors)");
        println!("  poly_gcd_exact(a, b)    Algebraic GCD over ℚ[x₁,...,xₙ]");
        println!("  pgcd                    Alias for poly_gcd");
        println!("  pgcdx                   Alias for poly_gcd_exact");
        println!();

        println!("Equation Solving:");
        println!("  solve <eq>, <var>       Solve equation for variable");
        println!("  equiv <e1>, <e2>        Check if two expressions are equivalent");
        println!("  subst <expr>, <var>=<val> Substitute a variable and simplify");
        println!();

        println!("Calculus:");
        println!("  diff <expr>, <var>      Compute symbolic derivative");
        println!("  limit <expr>            Compute limit at ±∞ (CLI: expli limit)");
        println!("  sum(e, v, a, b)         Finite summation: Σ(v=a to b) e");
        println!("  product(e, v, a, b)     Finite product: Π(v=a to b) e");
        println!();

        println!("Number Theory:");
        println!("  gcd <a, b>              Greatest Common Divisor");
        println!("  lcm <a, b>              Least Common Multiple");
        println!("  mod <a, n>              Modular arithmetic");
        println!("  factors <n>             Prime factorization");
        println!("  fact <n>                Factorial (or n!)");
        println!("  choose <n, k>           Binomial coefficient (nCk)");
        println!("  perm <n, k>             Permutations (nPk)");
        println!();

        println!("Matrix Operations:");
        println!("  det <matrix>            Compute determinant (up to 3×3)");
        println!("  transpose <matrix>      Transpose a matrix");
        println!("  trace <matrix>          Compute trace (sum of diagonal)");
        println!();

        println!("Analysis & Verification:");
        println!("  explain <function>      Show step-by-step explanation");
        println!("  telescope <expr>        Prove telescoping identities (Dirichlet kernel)");
        println!("  steps <level>           Set step verbosity (normal, succinct, verbose, none)");
        println!();

        println!("Visualization & Output:");
        println!("  visualize <expr>        Export AST to Graphviz DOT (generates ast.dot)");
        println!("  timeline <expr>         Export steps to interactive HTML");
        println!();

        println!(
            "  set <option> <value>    Pipeline settings (transform, rationalize, max-rewrites)"
        );
        println!("  semantics [set|help]    Semantic settings (domain, value, inv_trig, branch)");
        println!("  context [mode]          Context mode (auto, standard, solve, integrate)");
        println!("  config <subcmd>         Manage configuration (list, enable, disable...)");
        println!("  profile [cmd]           Rule profiler (enable/disable/clear)");
        println!("  health [cmd]            Health tracking (on/off/reset/status)");
        println!("  help [cmd]              Show this help message or details for a command");
        println!("  quit / exit             Exit the REPL");
        println!();

        println!("Session Environment:");
        println!("  let <name> = <expr>     Assign a variable");
        println!("  <name> := <expr>        Alternative assignment syntax");
        println!("  vars                    List all defined variables");
        println!("  clear [name]            Clear one or all variables");
        println!("  reset                   Clear all session state (keeps cache)");
        println!("  reset full              Clear all session state AND profile cache");
        println!("  budget [N]              Set/show Conditional branching budget (0-3)");
        println!("  cache [status|clear]    View or clear profile cache");
        println!("  history / list          Show session history (#ids)");
        println!("  show #<id>              Display a session entry");
        println!("  del #<id> ...           Delete session entries");
        println!();

        println!("Type 'help <command>' for more details on a specific command.");
    }

    fn handle_equiv(&mut self, line: &str) {
        use cas_ast::Expr;
        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

        let rest = line[6..].trim();
        if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
            let expr1_str = expr1_str.trim();
            let expr2_str = expr2_str.trim();

            // Helper to parse string to ExprId
            // Note: We use a block to limit mutable borrow scope if needed, though sequential calls work for separate statements.
            // But we need to use 'self' to access context inside logic? No, passed as arg.
            fn parse_arg(s: &str, ctx: &mut cas_ast::Context) -> Result<cas_ast::ExprId, String> {
                if s.starts_with('#') && s[1..].chars().all(char::is_numeric) {
                    Ok(ctx.add(Expr::Variable(s.to_string())))
                } else {
                    match cas_parser::parse_statement(s, ctx) {
                        Ok(Statement::Equation(eq)) => {
                            Ok(ctx.add(Expr::Function("Equal".to_string(), vec![eq.lhs, eq.rhs])))
                        }
                        Ok(Statement::Expression(e)) => Ok(e),
                        Err(e) => Err(format!("{}", e)),
                    }
                }
            }

            let e1_res = parse_arg(expr1_str, &mut self.engine.simplifier.context);
            // Verify e1_res to avoid borrow issues? No, Result doesn't borrow.
            let e2_res = parse_arg(expr2_str, &mut self.engine.simplifier.context);

            match (e1_res, e2_res) {
                (Ok(e1), Ok(e2)) => {
                    let req = EvalRequest {
                        raw_input: expr1_str.to_string(),
                        parsed: e1,
                        kind: EntryKind::Expr(e1),
                        action: EvalAction::Equiv { other: e2 },
                        auto_store: false,
                    };

                    match self.engine.eval(&mut self.state, req) {
                        Ok(output) => match output.result {
                            EvalResult::Bool(b) => {
                                if b {
                                    println!("True")
                                } else {
                                    println!("False")
                                }
                            }
                            _ => println!("Unexpected result type"),
                        },
                        Err(e) => println!("Error: {}", e),
                    }
                }
                (Err(e), _) => println!("Error parsing first arg: {}", e),
                (_, Err(e)) => println!("Error parsing second arg: {}", e),
            }
        } else {
            println!("Usage: equiv <expr1>, <expr2>");
        }
    }

    fn handle_subst(&mut self, line: &str) {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let rest = line[6..].trim();

        // Split by commas (respecting parentheses)
        let parts: Vec<&str> = split_by_comma_ignoring_parens(rest);

        if parts.len() != 3 {
            println!("Usage: subst <expr>, <target>, <replacement>");
            println!();
            println!("Examples:");
            println!("  subst x^2 + x, x, 3              → 12");
            println!("  subst x^4 + x^2 + 1, x^2, y      → y² + y + 1");
            println!("  subst x^3, x^2, y                → y·x");
            return;
        }

        let expr_str = parts[0].trim();
        let target_str = parts[1].trim();
        let replacement_str = parts[2].trim();

        // Parse the main expression
        let expr = match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Error parsing expression: {}", e);
                return;
            }
        };

        // Parse target
        let target_expr = match cas_parser::parse(target_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Error parsing target: {}", e);
                return;
            }
        };

        // Parse replacement
        let replacement_expr =
            match cas_parser::parse(replacement_str, &mut self.engine.simplifier.context) {
                Ok(e) => e,
                Err(e) => {
                    println!("Error parsing replacement: {}", e);
                    return;
                }
            };

        // Detect if target is a simple variable or an expression
        let is_simple_var = target_str.chars().all(|c| c.is_alphanumeric() || c == '_');

        let subbed = if is_simple_var {
            // Variable substitution
            if self.verbosity != Verbosity::None {
                println!(
                    "Variable substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                );
            }
            let target_var = self.engine.simplifier.context.var(target_str);
            cas_engine::solver::strategies::substitute_expr(
                &mut self.engine.simplifier.context,
                expr,
                target_var,
                replacement_expr,
            )
        } else {
            // Expression substitution (power-aware)
            if self.verbosity != Verbosity::None {
                println!(
                    "Expression substitution: {} → {} in {}",
                    target_str, replacement_str, expr_str
                );
            }
            cas_engine::substitute::substitute_power_aware(
                &mut self.engine.simplifier.context,
                expr,
                target_expr,
                replacement_expr,
                cas_engine::substitute::SubstituteOptions::default(),
            )
        };

        let (result, steps) = self.engine.simplifier.simplify(subbed);
        if self.verbosity != Verbosity::None && !steps.is_empty() {
            if self.verbosity != Verbosity::Succinct {
                println!("Steps:");
            }
            for step in steps.iter() {
                if should_show_step(step, self.verbosity) {
                    if self.verbosity == Verbosity::Succinct {
                        println!(
                            "-> {}",
                            DisplayExpr {
                                context: &self.engine.simplifier.context,
                                id: step.global_after.unwrap_or(step.after)
                            }
                        );
                    } else {
                        println!("  {}  [{}]", step.description, step.rule_name);
                    }
                }
            }
        }
        println!(
            "Result: {}",
            DisplayExpr {
                context: &self.engine.simplifier.context,
                id: result
            }
        );
    }

    fn handle_timeline(&mut self, line: &str) {
        let rest = line[9..].trim();

        // Check if the user wants to use "solve" within timeline
        // e.g., "timeline solve x + 2 = 5, x"
        if let Some(solve_rest) = rest.strip_prefix("solve ") {
            self.handle_timeline_solve(solve_rest);
            return;
        }

        // Check if the user wants to use "simplify" within timeline
        // e.g., "timeline simplify(expr)" or "timeline simplify expr"
        let (expr_str, use_aggressive) = if let Some(inner) = rest
            .strip_prefix("simplify(")
            .and_then(|s| s.strip_suffix(')'))
        {
            // Extract expression from "simplify(expr)"
            (inner, true)
        } else if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
            // Extract expression from "simplify expr"
            (simplify_rest, true)
        } else {
            // No simplify prefix, treat entire rest as expression
            (rest, false)
        };

        // Choose simplifier based on whether aggressive mode is requested
        let (steps, expr_id, simplified) = if use_aggressive {
            // Create temporary simplifier with aggressive rules (like handle_full_simplify)
            let mut temp_simplifier = Simplifier::with_default_rules();
            temp_simplifier.set_collect_steps(true); // Always collect steps for timeline

            // Swap context to preserve variables
            std::mem::swap(
                &mut self.engine.simplifier.context,
                &mut temp_simplifier.context,
            );

            match cas_parser::parse(expr_str.trim(), &mut temp_simplifier.context) {
                Ok(expr) => {
                    let (simplified, steps) = temp_simplifier.simplify(expr);

                    // Swap context back
                    std::mem::swap(
                        &mut self.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );

                    (steps, expr, simplified)
                }
                Err(e) => {
                    // Swap context back even on error
                    std::mem::swap(
                        &mut self.engine.simplifier.context,
                        &mut temp_simplifier.context,
                    );
                    println!("Parse error: {}", e);
                    return;
                }
            }
        } else {
            // Use engine.eval like handle_eval does - this ensures the same pipeline
            // (Core → Transform → Rationalize → PostCleanup) is used
            use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
            use cas_engine::EntryKind;

            // Force collect_steps for timeline
            let was_collecting = self.engine.simplifier.collect_steps();
            self.engine.simplifier.set_collect_steps(true);

            match cas_parser::parse(expr_str.trim(), &mut self.engine.simplifier.context) {
                Ok(expr) => {
                    let req = EvalRequest {
                        raw_input: expr_str.to_string(),
                        parsed: expr,
                        kind: EntryKind::Expr(expr),
                        action: EvalAction::Simplify,
                        auto_store: false, // Don't store in session history for timeline
                    };

                    match self.engine.eval(&mut self.state, req) {
                        Ok(output) => {
                            let simplified = match output.result {
                                EvalResult::Expr(e) => e,
                                _ => expr, // Fallback
                            };
                            self.engine.simplifier.set_collect_steps(was_collecting);
                            (output.steps, expr, simplified)
                        }
                        Err(e) => {
                            self.engine.simplifier.set_collect_steps(was_collecting);
                            println!("Simplification error: {}", e);
                            return;
                        }
                    }
                }
                Err(e) => {
                    self.engine.simplifier.set_collect_steps(was_collecting);
                    println!("Parse error: {}", e);
                    return;
                }
            }
        };

        if steps.is_empty() {
            println!("No simplification steps to visualize.");
            return;
        }

        // Filter out non-productive steps (where global state doesn't change)
        // But pass ALL steps to timeline so it can correctly compute final result
        let _filtered_steps = cas_engine::strategies::filter_non_productive_steps(
            &mut self.engine.simplifier.context,
            expr_id,
            steps.clone(),
        );

        // Convert CLI verbosity to timeline verbosity
        // Use Normal level - shows important steps without low-level canonicalization
        let timeline_verbosity = cas_engine::timeline::VerbosityLevel::Normal;

        // Generate HTML timeline with ALL steps and the known simplified result
        let mut timeline = cas_engine::timeline::TimelineHtml::new_with_result(
            &mut self.engine.simplifier.context,
            &steps,
            expr_id,
            Some(simplified),
            timeline_verbosity,
        );
        let html = timeline.to_html();

        let filename = "timeline.html";
        match std::fs::write(filename, &html) {
            Ok(_) => {
                println!("Timeline exported to {}", filename);
                if use_aggressive {
                    println!("(Aggressive simplification mode)");
                }
                println!("Open in browser to view interactive visualization.");

                // Try to auto-open on macOS
                #[cfg(target_os = "macos")]
                {
                    let _ = std::process::Command::new("open").arg(filename).spawn();
                }
            }
            Err(e) => println!("Error writing file: {}", e),
        }
    }

    fn handle_visualize(&mut self, line: &str) {
        let rest = line
            .strip_prefix("visualize ")
            .or_else(|| line.strip_prefix("viz "))
            .unwrap_or(line)
            .trim();

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                let mut viz =
                    cas_engine::visualizer::AstVisualizer::new(&self.engine.simplifier.context);
                let dot = viz.to_dot(expr);

                // Save to file
                let filename = "ast.dot";
                match std::fs::write(filename, &dot) {
                    Ok(_) => {
                        println!("AST exported to {}", filename);
                        println!("Render with: dot -Tsvg {} -o ast.svg", filename);
                        println!("Or: dot -Tpng {} -o ast.png", filename);
                    }
                    Err(e) => println!("Error writing file: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    fn handle_explain(&mut self, line: &str) {
        let rest = line[8..].trim(); // Remove "explain "

        // Parse the expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Check if it's a function call
                let expr_data = self.engine.simplifier.context.get(expr).clone();
                if let Expr::Function(name, args) = expr_data {
                    match name.as_str() {
                        "gcd" => {
                            if args.len() == 2 {
                                // Call the explain_gcd function
                                let result = cas_engine::rules::number_theory::explain_gcd(
                                    &mut self.engine.simplifier.context,
                                    args[0],
                                    args[1],
                                );

                                println!("Parsed: {}", rest);
                                println!();
                                println!("Educational Steps:");
                                println!("{}", "─".repeat(60));

                                for step in &result.steps {
                                    println!("{}", step);
                                }

                                println!("{}", "─".repeat(60));
                                println!();

                                if let Some(result_expr) = result.value {
                                    println!(
                                        "Result: {}",
                                        DisplayExpr {
                                            context: &self.engine.simplifier.context,
                                            id: result_expr
                                        }
                                    );
                                } else {
                                    println!("Could not compute GCD");
                                }
                            } else {
                                println!("Usage: explain gcd(a, b)");
                            }
                        }
                        _ => {
                            println!("Explain mode not yet implemented for function '{}'", name);
                            println!("Currently supported: gcd");
                        }
                    }
                } else {
                    println!("Explain mode currently only supports function calls");
                    println!("Try: explain gcd(48, 18)");
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    // ========== SESSION ENVIRONMENT HANDLERS ==========

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    fn handle_let_command(&mut self, rest: &str) {
        // Detect := (lazy) before = (eager) - order matters!
        if let Some(idx) = rest.find(":=") {
            let name = rest[..idx].trim();
            let expr_str = rest[idx + 2..].trim();
            self.handle_assignment(name, expr_str, true); // lazy
        } else if let Some(eq_idx) = rest.find('=') {
            let name = rest[..eq_idx].trim();
            let expr_str = rest[eq_idx + 1..].trim();
            self.handle_assignment(name, expr_str, false); // eager
        } else {
            println!("Usage: let <name> = <expr>   (eager - evaluates)");
            println!("       let <name> := <expr>  (lazy - stores formula)");
            println!("Example: let a = expand((1+x)^3)");
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    fn handle_assignment(&mut self, name: &str, expr_str: &str, lazy: bool) {
        // Validate name
        if name.is_empty() {
            println!("Error: Variable name cannot be empty");
            return;
        }

        // Check if identifier is valid (alphanumeric + underscore, starts with letter/underscore)
        if !name.chars().next().unwrap().is_alphabetic() && !name.starts_with('_') {
            println!("Error: Variable name must start with a letter or underscore");
            return;
        }

        // Check reserved names
        if cas_engine::env::is_reserved(name) {
            println!(
                "Error: '{}' is a reserved name and cannot be assigned",
                name
            );
            return;
        }

        // Parse the expression
        match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(rhs_expr) => {
                // Temporarily remove this binding to prevent self-reference in substitute
                let old_binding = self.state.env.get(name);
                self.state.env.unset(name);

                // Substitute using current environment and session refs
                let rhs_substituted = match self
                    .state
                    .resolve_all(&mut self.engine.simplifier.context, rhs_expr)
                {
                    Ok(r) => r,
                    Err(_) => rhs_expr,
                };

                let result = if lazy {
                    // LAZY (:=): store the expression without evaluating
                    rhs_substituted
                } else {
                    // EAGER (=): simplify the expression, then unwrap __hold
                    let (simplified, _steps) = self.engine.simplifier.simplify(rhs_substituted);

                    // Unwrap top-level __hold to get the actual polynomial
                    unwrap_hold_top(&self.engine.simplifier.context, simplified)
                };

                // Store the binding
                self.state.env.set(name.to_string(), result);

                // Display confirmation (with mode indicator for lazy)
                let display = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: result,
                };
                if lazy {
                    println!("{} := {}", name, display);
                } else {
                    println!("{} = {}", name, display);
                }

                // Note: we don't restore old_binding - this is an assignment/update
                let _ = old_binding;
            }
            Err(e) => {
                println!("Parse error: {}", e);
            }
        }
    }

    /// Handle "vars" command - list all variable bindings
    fn handle_vars_command(&self) {
        let bindings = self.state.env.list();
        if bindings.is_empty() {
            println!("No variables defined.");
        } else {
            println!("Variables:");
            for (name, expr_id) in bindings {
                let display = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: expr_id,
                };
                println!("  {} = {}", name, display);
            }
        }
    }

    /// Handle "clear" or "clear <names>" command
    fn handle_clear_command(&mut self, line: &str) {
        if line == "clear" {
            // Clear all
            let count = self.state.env.len();
            self.state.env.clear_all();
            if count == 0 {
                println!("No variables to clear.");
            } else {
                println!("Cleared {} variable(s).", count);
            }
        } else {
            // Clear specific variables
            let names: Vec<&str> = line[6..].split_whitespace().collect();
            let mut cleared = 0;
            for name in names {
                if self.state.env.unset(name) {
                    cleared += 1;
                } else {
                    println!("Warning: '{}' was not defined", name);
                }
            }
            if cleared > 0 {
                println!("Cleared {} variable(s).", cleared);
            }
        }
    }

    /// Handle "reset" command - reset entire session
    fn handle_reset_command(&mut self) {
        // Clear session state (history + env)
        self.state.clear();

        // Reset simplifier with new context
        self.engine.simplifier = Simplifier::with_default_rules();

        // Re-register custom rules (same as in new())
        self.engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        self.engine.simplifier.add_rule(Box::new(EvaluateTrigRule));
        self.engine
            .simplifier
            .add_rule(Box::new(PythagoreanIdentityRule));
        if self.config.trig_angle_sum {
            self.engine.simplifier.add_rule(Box::new(AngleIdentityRule));
        }
        self.engine.simplifier.add_rule(Box::new(TanToSinCosRule));
        if self.config.trig_double_angle {
            self.engine.simplifier.add_rule(Box::new(DoubleAngleRule));
        }
        if self.config.canonicalize_trig_square {
            self.engine.simplifier.add_rule(Box::new(
                cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        self.engine.simplifier.add_rule(Box::new(EvaluateLogRule));
        self.engine
            .simplifier
            .add_rule(Box::new(ExponentialLogRule));
        self.engine
            .simplifier
            .add_rule(Box::new(SimplifyFractionRule));
        self.engine.simplifier.add_rule(Box::new(ExpandRule));
        self.engine
            .simplifier
            .add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));

        // Sync config
        self.sync_config_to_simplifier();

        // Reset options
        self.explain_mode = false;
        self.last_stats = None;
        self.health_enabled = false;
        self.last_health_report = None;

        println!("Session reset. Environment and context cleared.");
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    fn handle_reset_full_command(&mut self) {
        // First do normal reset
        self.handle_reset_command();

        // Also clear profile cache
        self.state.profile_cache.clear();

        println!("Profile cache cleared (will rebuild on next eval).");
    }

    /// Handle "cache" command - show status or clear cache
    fn handle_cache_command(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1).copied() {
            None | Some("status") => {
                // Show cache status
                let count = self.state.profile_cache.len();
                println!("Profile Cache: {} profiles cached", count);
                if count == 0 {
                    println!("  (empty - profiles will be built on first eval)");
                } else {
                    println!("  (profiles are reused across evaluations)");
                }
            }
            Some("clear") => {
                self.state.profile_cache.clear();
                println!("Profile cache cleared.");
            }
            Some(cmd) => {
                println!("Unknown cache command: {}", cmd);
                println!("Usage: cache [status|clear]");
            }
        }
    }

    /// Handle "semantics" command - unified control for semantic axes
    fn handle_semantics(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "semantics" - show current settings
                self.print_semantics();
            }
            Some(&"help") => {
                self.print_semantics_help();
            }
            Some(&"set") => {
                // Parse remaining args as axis=value pairs or axis value pairs
                self.parse_semantics_set(&args[2..]);
            }
            Some(&"domain") => {
                self.print_axis_status("domain");
            }
            Some(&"value") => {
                self.print_axis_status("value");
            }
            Some(&"branch") => {
                self.print_axis_status("branch");
            }
            Some(&"inv_trig") => {
                self.print_axis_status("inv_trig");
            }
            Some(&"const_fold") => {
                self.print_axis_status("const_fold");
            }
            Some(&"assumptions") => {
                self.print_axis_status("assumptions");
            }
            Some(&"assume_scope") => {
                self.print_axis_status("assume_scope");
            }
            Some(&"preset") => {
                self.handle_preset(&args[2..]);
            }
            Some(other) => {
                println!("Unknown semantics subcommand: '{}'", other);
                println!("Usage: semantics [set|preset|help|<axis>]");
                println!("  semantics            Show all settings");
                println!("  semantics <axis>     Show one axis (domain|value|branch|inv_trig|const_fold|assumptions|assume_scope)");
                println!("  semantics help       Show help");
                println!("  semantics set ...    Change settings");
                println!("  semantics preset     List/apply presets");
            }
        }
    }

    fn print_semantics(&self) {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        let domain = match self.simplify_options.domain {
            DomainMode::Strict => "strict",
            DomainMode::Assume => "assume",
            DomainMode::Generic => "generic",
        };

        let value = match self.simplify_options.value_domain {
            ValueDomain::RealOnly => "real",
            ValueDomain::ComplexEnabled => "complex",
        };

        let branch = match self.simplify_options.branch {
            BranchPolicy::Principal => "principal",
        };

        let inv_trig = match self.simplify_options.inv_trig {
            InverseTrigPolicy::Strict => "strict",
            InverseTrigPolicy::PrincipalValue => "principal",
        };

        println!("Semantics:");
        println!("  domain_mode: {}", domain);
        println!("  value_domain: {}", value);

        // Show branch with inactive note if value=real
        if self.simplify_options.value_domain == ValueDomain::RealOnly {
            println!("  branch: {} (inactive: value_domain=real)", branch);
        } else {
            println!("  branch: {}", branch);
        }

        println!("  inv_trig: {}", inv_trig);

        let const_fold = match self.state.options.const_fold {
            cas_engine::const_fold::ConstFoldMode::Off => "off",
            cas_engine::const_fold::ConstFoldMode::Safe => "safe",
        };
        println!("  const_fold: {}", const_fold);

        let assumptions = match self.state.options.assumption_reporting {
            cas_engine::AssumptionReporting::Off => "off",
            cas_engine::AssumptionReporting::Summary => "summary",
            cas_engine::AssumptionReporting::Trace => "trace",
        };
        println!("  assumptions: {}", assumptions);

        // Show assume_scope with inactive note if domain_mode != Assume
        let assume_scope = match self.simplify_options.assume_scope {
            cas_engine::AssumeScope::Real => "real",
            cas_engine::AssumeScope::Wildcard => "wildcard",
        };
        if self.simplify_options.domain != DomainMode::Assume {
            println!(
                "  assume_scope: {} (inactive: domain_mode != assume)",
                assume_scope
            );
        } else {
            println!("  assume_scope: {}", assume_scope);
        }

        // Show hints_enabled
        let hints = if self.state.options.hints_enabled {
            "on"
        } else {
            "off"
        };
        println!("  hints: {}", hints);
    }

    /// Print status for a single semantic axis with current value and available options
    fn print_axis_status(&self, axis: &str) {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        match axis {
            "domain" => {
                let current = match self.simplify_options.domain {
                    DomainMode::Strict => "strict",
                    DomainMode::Assume => "assume",
                    DomainMode::Generic => "generic",
                };
                println!("domain: {}", current);
                println!("  Values: strict | generic | assume");
                println!("  strict:  No domain assumptions (x/x stays x/x)");
                println!("  generic: Classic CAS 'almost everywhere' algebra");
                println!("  assume:  Use assumptions with warnings");
            }
            "value" => {
                let current = match self.simplify_options.value_domain {
                    ValueDomain::RealOnly => "real",
                    ValueDomain::ComplexEnabled => "complex",
                };
                println!("value: {}", current);
                println!("  Values: real | complex");
                println!("  real:    ℝ only (sqrt(-1) undefined)");
                println!("  complex: ℂ enabled (sqrt(-1) = i)");
            }
            "branch" => {
                let current = match self.simplify_options.branch {
                    BranchPolicy::Principal => "principal",
                };
                let inactive = self.simplify_options.value_domain == ValueDomain::RealOnly;
                if inactive {
                    println!("branch: {} (inactive: value=real)", current);
                } else {
                    println!("branch: {}", current);
                }
                println!("  Values: principal");
                println!("  principal: Use principal branch for multi-valued functions");
                if inactive {
                    println!("  Note: Only active when value=complex");
                }
            }
            "inv_trig" => {
                let current = match self.simplify_options.inv_trig {
                    InverseTrigPolicy::Strict => "strict",
                    InverseTrigPolicy::PrincipalValue => "principal",
                };
                println!("inv_trig: {}", current);
                println!("  Values: strict | principal");
                println!("  strict:    arctan(tan(x)) unchanged");
                println!("  principal: arctan(tan(x)) → x with warning");
            }
            "const_fold" => {
                let current = match self.state.options.const_fold {
                    cas_engine::const_fold::ConstFoldMode::Off => "off",
                    cas_engine::const_fold::ConstFoldMode::Safe => "safe",
                };
                println!("const_fold: {}", current);
                println!("  Values: off | safe");
                println!("  off:  No constant folding (defer semantic decisions)");
                println!("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)");
            }
            "assumptions" => {
                let current = match self.state.options.assumption_reporting {
                    cas_engine::AssumptionReporting::Off => "off",
                    cas_engine::AssumptionReporting::Summary => "summary",
                    cas_engine::AssumptionReporting::Trace => "trace",
                };
                println!("assumptions: {}", current);
                println!("  Values: off | summary | trace");
                println!("  off:     No assumption reporting");
                println!("  summary: Deduped summary line at end");
                println!("  trace:   Detailed trace (future)");
            }
            "assume_scope" => {
                let current = match self.simplify_options.assume_scope {
                    cas_engine::AssumeScope::Real => "real",
                    cas_engine::AssumeScope::Wildcard => "wildcard",
                };
                let inactive = self.simplify_options.domain != DomainMode::Assume;
                if inactive {
                    println!(
                        "assume_scope: {} (inactive: domain_mode != assume)",
                        current
                    );
                } else {
                    println!("assume_scope: {}", current);
                }
                println!("  Values: real | wildcard");
                println!("  real:     Assume for ℝ, error if ℂ needed");
                println!("  wildcard: Assume for ℝ, residual+warning if ℂ needed");
                if inactive {
                    println!("  Note: Only active when domain_mode=assume");
                }
            }
            _ => {
                println!("Unknown axis: {}", axis);
            }
        }
    }

    fn print_semantics_help(&self) {
        println!("Semantics: Control evaluation semantics");
        println!();
        println!("Usage:");
        println!("  semantics                    Show current settings");
        println!("  semantics set <axis> <val>   Set one axis");
        println!("  semantics set k=v k=v ...    Set multiple axes");
        println!();
        println!("Axes:");
        println!("  domain      strict | generic | assume");
        println!("              strict:  No domain assumptions (x/x stays x/x)");
        println!("              generic: Classic CAS 'almost everywhere' algebra");
        println!("              assume:  Use assumptions with warnings");
        println!();
        println!("  value       real | complex");
        println!("              real:    ℝ only (sqrt(-1) undefined)");
        println!("              complex: ℂ enabled (sqrt(-1) = i)");
        println!();
        println!("  branch      principal");
        println!("              (only active when value=complex)");
        println!();
        println!("  inv_trig    strict | principal");
        println!("              strict:    arctan(tan(x)) unchanged");
        println!("              principal: arctan(tan(x)) → x with warning");
        println!();
        println!("  const_fold  off | safe");
        println!("              off:  No constant folding");
        println!("              safe: Fold literals (2^3 → 8)");
        println!();
        println!("  assume_scope real | wildcard");
        println!("              real:     Assume for ℝ, error if ℂ needed");
        println!("              wildcard: Assume for ℝ, residual+warning if ℂ needed");
        println!("              (only active when domain_mode=assume)");
        println!();
        println!("Examples:");
        println!("  semantics set domain strict");
        println!("  semantics set value complex inv_trig principal");
        println!("  semantics set domain=strict value=complex");
        println!("  semantics set assume_scope wildcard");
        println!();
        println!("Presets:");
        println!("  semantics preset              List available presets");
        println!("  semantics preset <name>       Apply a preset");
        println!("  semantics preset help <name>  Show preset details");
    }

    /// Handle "semantics preset" subcommand
    fn handle_preset(&mut self, args: &[&str]) {
        use cas_engine::const_fold::ConstFoldMode;
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        // Preset definitions: (name, description, domain, value, branch, inv_trig, const_fold)
        struct Preset {
            name: &'static str,
            description: &'static str,
            domain: DomainMode,
            value: ValueDomain,
            branch: BranchPolicy,
            inv_trig: InverseTrigPolicy,
            const_fold: ConstFoldMode,
        }

        let presets = [
            Preset {
                name: "default",
                description: "Reset to engine defaults",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "strict",
                description: "Conservative real + strict domain",
                domain: DomainMode::Strict,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Off,
            },
            Preset {
                name: "complex",
                description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
                domain: DomainMode::Generic,
                value: ValueDomain::ComplexEnabled,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::Strict,
                const_fold: ConstFoldMode::Safe,
            },
            Preset {
                name: "school",
                description: "Real + principal inverse trig (arctan(tan(x)) → x)",
                domain: DomainMode::Generic,
                value: ValueDomain::RealOnly,
                branch: BranchPolicy::Principal,
                inv_trig: InverseTrigPolicy::PrincipalValue,
                const_fold: ConstFoldMode::Off,
            },
        ];

        match args.first() {
            None => {
                // List presets
                println!("Available presets:");
                for p in &presets {
                    println!("  {:10} {}", p.name, p.description);
                }
                println!();
                println!("Usage:");
                println!("  semantics preset <name>       Apply preset");
                println!("  semantics preset help <name>  Show preset axes");
            }
            Some(&"help") => {
                // Show preset details
                let name = args.get(1);
                if name.is_none() {
                    println!("Usage: semantics preset help <name>");
                    println!("Presets: default, strict, complex, school");
                    return;
                }
                let name = name.unwrap();
                if let Some(p) = presets.iter().find(|p| p.name == *name) {
                    let domain_str = match p.domain {
                        DomainMode::Strict => "strict",
                        DomainMode::Generic => "generic",
                        DomainMode::Assume => "assume",
                    };
                    let value_str = match p.value {
                        ValueDomain::RealOnly => "real",
                        ValueDomain::ComplexEnabled => "complex",
                    };
                    let inv_trig_str = match p.inv_trig {
                        InverseTrigPolicy::Strict => "strict",
                        InverseTrigPolicy::PrincipalValue => "principal",
                    };
                    let const_fold_str = match p.const_fold {
                        ConstFoldMode::Off => "off",
                        ConstFoldMode::Safe => "safe",
                    };
                    println!("{}:", p.name);
                    println!("  domain_mode  = {}", domain_str);
                    println!("  value_domain = {}", value_str);
                    println!("  branch       = principal");
                    println!("  inv_trig     = {}", inv_trig_str);
                    println!("  const_fold   = {}", const_fold_str);
                    println!();
                    println!("Purpose: {}", p.description);
                } else {
                    println!("Unknown preset: '{}'", name);
                    println!("Available: default, strict, complex, school");
                }
            }
            Some(name) => {
                // Apply preset
                if let Some(p) = presets.iter().find(|preset| preset.name == *name) {
                    // Capture old values for diff
                    let old_domain = self.simplify_options.domain;
                    let old_value = self.simplify_options.value_domain;
                    let old_branch = self.simplify_options.branch;
                    let old_inv_trig = self.simplify_options.inv_trig;
                    let old_const_fold = self.state.options.const_fold;

                    // Apply preset
                    self.simplify_options.domain = p.domain;
                    self.simplify_options.value_domain = p.value;
                    self.simplify_options.branch = p.branch;
                    self.simplify_options.inv_trig = p.inv_trig;
                    self.state.options.const_fold = p.const_fold;
                    // Sync to state.options (used by evaluation pipeline)
                    self.state.options.domain_mode = p.domain;
                    self.state.options.value_domain = p.value;
                    self.state.options.branch = p.branch;
                    self.state.options.inv_trig = p.inv_trig;

                    self.sync_config_to_simplifier();

                    println!("Applied preset: {}", p.name);
                    println!("Changes:");

                    // Print changes
                    let mut changes = 0;
                    if old_domain != p.domain {
                        let old_str = match old_domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        let new_str = match p.domain {
                            DomainMode::Strict => "strict",
                            DomainMode::Generic => "generic",
                            DomainMode::Assume => "assume",
                        };
                        println!("  domain_mode:  {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if old_value != p.value {
                        let old_str = match old_value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        let new_str = match p.value {
                            ValueDomain::RealOnly => "real",
                            ValueDomain::ComplexEnabled => "complex",
                        };
                        println!("  value_domain: {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if old_branch != p.branch {
                        println!("  branch:       principal → principal");
                        changes += 1;
                    }
                    if old_inv_trig != p.inv_trig {
                        let old_str = match old_inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        let new_str = match p.inv_trig {
                            InverseTrigPolicy::Strict => "strict",
                            InverseTrigPolicy::PrincipalValue => "principal",
                        };
                        println!("  inv_trig:     {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if old_const_fold != p.const_fold {
                        let old_str = match old_const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        let new_str = match p.const_fold {
                            ConstFoldMode::Off => "off",
                            ConstFoldMode::Safe => "safe",
                        };
                        println!("  const_fold:   {} → {}", old_str, new_str);
                        changes += 1;
                    }
                    if changes == 0 {
                        println!("  (no changes - already at this preset)");
                    }
                } else {
                    println!("Unknown preset: '{}'", name);
                    println!("Available: default, strict, complex, school");
                }
            }
        }
    }

    fn parse_semantics_set(&mut self, args: &[&str]) {
        if args.is_empty() {
            println!("Usage: semantics set <axis> <value>");
            println!("  or:  semantics set <axis>=<value> ...");
            return;
        }

        let mut i = 0;
        while i < args.len() {
            let arg = args[i];

            // Check for key=value format
            if let Some((key, value)) = arg.split_once('=') {
                if !self.set_semantic_axis(key, value) {
                    return;
                }
                i += 1;
            } else {
                // key value format
                if i + 1 >= args.len() {
                    println!("ERROR: Missing value for axis '{}'", arg);
                    return;
                }

                // Special case: "solve check on|off" is a 3-part axis
                if arg == "solve" && args.get(i + 1) == Some(&"check") && i + 2 < args.len() {
                    let on_off = args[i + 2];
                    match on_off {
                        "on" => {
                            self.state.options.check_solutions = true;
                            println!("Solve check: ON (solutions will be verified)");
                        }
                        "off" => {
                            self.state.options.check_solutions = false;
                            println!("Solve check: OFF");
                        }
                        _ => {
                            println!("ERROR: Invalid value '{}' for 'solve check'", on_off);
                            println!("Allowed: on, off");
                            return;
                        }
                    }
                    i += 3;
                    continue;
                }

                let value = args[i + 1];
                if !self.set_semantic_axis(arg, value) {
                    return;
                }
                i += 2;
            }
        }

        self.sync_config_to_simplifier();
        self.print_semantics();
    }

    fn set_semantic_axis(&mut self, axis: &str, value: &str) -> bool {
        use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
        use cas_engine::DomainMode;

        match axis {
            "domain" => match value {
                "strict" => {
                    self.simplify_options.domain = DomainMode::Strict;
                    self.state.options.domain_mode = DomainMode::Strict;
                }
                "generic" => {
                    self.simplify_options.domain = DomainMode::Generic;
                    self.state.options.domain_mode = DomainMode::Generic;
                }
                "assume" => {
                    self.simplify_options.domain = DomainMode::Assume;
                    self.state.options.domain_mode = DomainMode::Assume;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'domain'", value);
                    println!("Allowed: strict, generic, assume");
                    return false;
                }
            },
            "value" => match value {
                "real" => {
                    self.simplify_options.value_domain = ValueDomain::RealOnly;
                    self.state.options.value_domain = ValueDomain::RealOnly;
                }
                "complex" => {
                    self.simplify_options.value_domain = ValueDomain::ComplexEnabled;
                    self.state.options.value_domain = ValueDomain::ComplexEnabled;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'value'", value);
                    println!("Allowed: real, complex");
                    return false;
                }
            },
            "branch" => match value {
                "principal" => {
                    self.simplify_options.branch = BranchPolicy::Principal;
                    self.state.options.branch = BranchPolicy::Principal;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'branch'", value);
                    println!("Allowed: principal");
                    return false;
                }
            },
            "inv_trig" => match value {
                "strict" => {
                    self.simplify_options.inv_trig = InverseTrigPolicy::Strict;
                    self.state.options.inv_trig = InverseTrigPolicy::Strict;
                }
                "principal" => {
                    self.simplify_options.inv_trig = InverseTrigPolicy::PrincipalValue;
                    self.state.options.inv_trig = InverseTrigPolicy::PrincipalValue;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'inv_trig'", value);
                    println!("Allowed: strict, principal");
                    return false;
                }
            },
            "const_fold" => {
                use cas_engine::const_fold::ConstFoldMode;
                match value {
                    "off" => {
                        self.state.options.const_fold = ConstFoldMode::Off;
                    }
                    "safe" => {
                        self.state.options.const_fold = ConstFoldMode::Safe;
                    }
                    _ => {
                        println!("ERROR: Invalid value '{}' for axis 'const_fold'", value);
                        println!("Allowed: off, safe");
                        return false;
                    }
                }
            }
            "assumptions" => match value {
                "off" => {
                    self.state.options.assumption_reporting = cas_engine::AssumptionReporting::Off;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Off;
                }
                "summary" => {
                    self.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Summary;
                }
                "trace" => {
                    self.state.options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                    self.simplify_options.assumption_reporting =
                        cas_engine::AssumptionReporting::Trace;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'assumptions'", value);
                    println!("Allowed: off, summary, trace");
                    return false;
                }
            },
            "assume_scope" => match value {
                "real" => {
                    self.simplify_options.assume_scope = cas_engine::AssumeScope::Real;
                    self.state.options.assume_scope = cas_engine::AssumeScope::Real;
                }
                "wildcard" => {
                    self.simplify_options.assume_scope = cas_engine::AssumeScope::Wildcard;
                    self.state.options.assume_scope = cas_engine::AssumeScope::Wildcard;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'assume_scope'", value);
                    println!("Allowed: real, wildcard");
                    return false;
                }
            },
            "hints" => match value {
                "on" => {
                    self.state.options.hints_enabled = true;
                }
                "off" => {
                    self.state.options.hints_enabled = false;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'hints'", value);
                    println!("Allowed: on, off");
                    return false;
                }
            },
            "solve" => match value {
                "check" => {
                    // "solve check" is special: toggle without secondary value
                    println!("ERROR: Use 'semantics set solve check on' or 'semantics set solve check off'");
                    return false;
                }
                _ => {
                    println!("ERROR: Invalid value '{}' for axis 'solve'", value);
                    println!("Allowed: 'check on', 'check off'");
                    return false;
                }
            },
            _ => {
                println!("ERROR: Unknown axis '{}'", axis);
                println!("Valid axes: domain, value, branch, inv_trig, const_fold, assumptions, assume_scope, hints, solve");
                return false;
            }
        }
        true
    }

    /// Handle "context" command - show or switch context mode
    fn handle_context_command(&mut self, line: &str) {
        use cas_engine::options::ContextMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "context" - show current context
                let ctx_str = match self.state.options.context_mode {
                    ContextMode::Auto => "auto",
                    ContextMode::Standard => "standard",
                    ContextMode::Solve => "solve",
                    ContextMode::IntegratePrep => "integrate",
                };
                println!("Current context: {}", ctx_str);
                println!("  (use 'context auto|standard|solve|integrate' to change)");
            }
            Some(&"auto") => {
                self.state.options.context_mode = ContextMode::Auto;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: auto (infers from expression)");
            }
            Some(&"standard") => {
                self.state.options.context_mode = ContextMode::Standard;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: standard (safe simplification only)");
            }
            Some(&"solve") => {
                self.state.options.context_mode = ContextMode::Solve;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: solve (preserves solver-friendly forms)");
            }
            Some(&"integrate") => {
                self.state.options.context_mode = ContextMode::IntegratePrep;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Context: integrate-prep");
                println!("  ⚠️ Enables transforms for integration (telescoping, product→sum)");
            }
            Some(other) => {
                println!("Unknown context: '{}'", other);
                println!("Usage: context [auto | standard | solve | integrate]");
            }
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    fn handle_steps_command(&mut self, line: &str) {
        use cas_engine::options::StepsMode;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "steps" - show current mode
                let mode_str = match self.state.options.steps_mode {
                    StepsMode::On => "on",
                    StepsMode::Off => "off",
                    StepsMode::Compact => "compact",
                };
                let verbosity_str = match self.verbosity {
                    Verbosity::None => "none",
                    Verbosity::Succinct => "succinct",
                    Verbosity::Normal => "normal",
                    Verbosity::Verbose => "verbose",
                };
                println!("Steps collection: {}", mode_str);
                println!("Steps display: {}", verbosity_str);
                println!("  (use 'steps on|off|compact' for collection)");
                println!("  (use 'steps verbose|succinct|normal|none' for display)");
            }
            // Collection modes (StepsMode)
            Some(&"on") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                println!("Steps: on (full collection, normal display)");
            }
            Some(&"off") => {
                self.state.options.steps_mode = StepsMode::Off;
                self.engine.simplifier.set_steps_mode(StepsMode::Off);
                self.verbosity = Verbosity::None;
                println!("Steps: off");
                println!("  ⚡ Steps disabled (faster). Warnings still enabled.");
            }
            Some(&"compact") => {
                self.state.options.steps_mode = StepsMode::Compact;
                self.engine.simplifier.set_steps_mode(StepsMode::Compact);
                println!("Steps: compact (no before/after snapshots)");
            }
            // Display modes (Verbosity)
            Some(&"verbose") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Verbose;
                println!("Steps: verbose (all rules, full detail)");
            }
            Some(&"succinct") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Succinct;
                println!("Steps: succinct (compact 1-line per step)");
            }
            Some(&"normal") => {
                self.state.options.steps_mode = StepsMode::On;
                self.engine.simplifier.set_steps_mode(StepsMode::On);
                self.verbosity = Verbosity::Normal;
                println!("Steps: normal (default display)");
            }
            Some(&"none") => {
                self.verbosity = Verbosity::None;
                println!("Steps display: none (collection still active)");
                println!("  Use 'steps off' to also disable collection.");
            }
            Some(other) => {
                println!("Unknown steps mode: '{}'", other);
                println!("Usage: steps [on | off | compact | verbose | succinct | normal | none]");
                println!("  Collection modes:");
                println!("    on      - Full steps with snapshots (default)");
                println!("    off     - No steps (fastest, warnings preserved)");
                println!("    compact - Minimal steps (no snapshots)");
                println!("  Display modes:");
                println!("    verbose - Show all rules, full detail");
                println!("    succinct- Compact 1-line per step");
                println!("    normal  - Standard display (default)");
                println!("    none    - Hide steps output (collection still active)");
            }
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    fn handle_autoexpand_command(&mut self, line: &str) {
        use cas_engine::phase::ExpandPolicy;

        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "autoexpand" - show current mode
                let policy_str = match self.state.options.expand_policy {
                    ExpandPolicy::Off => "off",
                    ExpandPolicy::Auto => "on",
                };
                println!("Auto-expand: {}", policy_str);
                let budget = &self.state.options.expand_budget;
                println!(
                    "  Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}",
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                );
                println!("  (use 'autoexpand on|off' to change)");
            }
            Some(&"on") => {
                self.state.options.expand_policy = ExpandPolicy::Auto;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                let budget = &self.state.options.expand_budget;
                println!("Auto-expand: on");
                println!(
                    "  Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}",
                    budget.max_pow_exp,
                    budget.max_base_terms,
                    budget.max_generated_terms,
                    budget.max_vars
                );
                println!("  ⚠️ Expands small (sum)^n patterns automatically.");
            }
            Some(&"off") => {
                self.state.options.expand_policy = ExpandPolicy::Off;
                self.engine.simplifier = cas_engine::Simplifier::with_profile(&self.state.options);
                self.sync_config_to_simplifier();
                println!("Auto-expand: off");
                println!("  Polynomial expansions require explicit expand().");
            }
            Some(other) => {
                println!("Unknown autoexpand mode: '{}'", other);
                println!("Usage: autoexpand [on | off]");
                println!("  on  - Auto-expand cheap polynomial powers");
                println!("  off - Only expand when explicitly requested (default)");
            }
        }
    }

    /// V2.0: Handle "budget" command - control Conditional branching for solve
    fn handle_budget_command(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();

        match args.get(1) {
            None => {
                // Just "budget" - show current setting
                let budget = self.state.options.budget;
                println!("Solve budget: max_branches={}", budget.max_branches);
                println!("  Controls how many case splits the solver can create.");
                println!("  0: No splits (fallback to simple solutions)");
                println!("  1: Conservative (default)");
                println!("  2+: Allow case splits for symbolic bases (a^x=a, etc)");
                println!("  (use 'budget N' to change, e.g. 'budget 2')");
            }
            Some(n_str) => {
                if let Ok(n) = n_str.parse::<usize>() {
                    self.state.options.budget.max_branches = n;
                    println!("Solve budget: max_branches = {}", n);
                    if n == 0 {
                        println!("  ⚠️ No case splits allowed (fallback to simple solutions)");
                    } else if n == 1 {
                        println!("  Conservative mode (default)");
                    } else {
                        println!("  ✓ Case splits enabled for symbolic bases");
                        println!("  Try: solve a^x = a");
                    }
                } else {
                    println!("Invalid budget value: '{}' (expected a number)", n_str);
                    println!("Usage: budget N");
                    println!("  budget 0  - No case splits");
                    println!("  budget 1  - Conservative (default)");
                    println!("  budget 2  - Allow case splits for a^x=a patterns");
                }
            }
        }
    }

    /// Handle "history" or "list" command - show session history
    fn handle_history_command(&self) {
        let entries = self.state.store.list();
        if entries.is_empty() {
            println!("No entries in session history.");
            return;
        }

        println!("Session history ({} entries):", entries.len());
        for entry in entries {
            let type_indicator = match &entry.kind {
                cas_engine::EntryKind::Expr(_) => "Expr",
                cas_engine::EntryKind::Eq { .. } => "Eq  ",
            };
            // Show simplified form if possible
            let display = match &entry.kind {
                cas_engine::EntryKind::Expr(expr_id) => {
                    format!(
                        "{}",
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *expr_id
                        }
                    )
                }
                cas_engine::EntryKind::Eq { lhs, rhs } => {
                    format!(
                        "{} = {}",
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *lhs
                        },
                        cas_ast::DisplayExpr {
                            context: &self.engine.simplifier.context,
                            id: *rhs
                        }
                    )
                }
            };
            println!("  #{:<3} [{}] {}", entry.id, type_indicator, display);
        }
    }

    /// Handle "show #id" command - show details of a specific entry
    fn handle_show_command(&mut self, line: &str) {
        let input = line.trim().trim_start_matches('#');
        match input.parse::<u64>() {
            Ok(id) => {
                if let Some(entry) = self.state.store.get(id) {
                    println!("Entry #{}:", id);
                    println!("  Type:       {}", entry.type_str());
                    println!("  Raw:        {}", entry.raw_text);

                    match &entry.kind {
                        cas_engine::EntryKind::Expr(expr_id) => {
                            // Show parsed expression
                            println!(
                                "  Parsed:     {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *expr_id
                                }
                            );

                            // Show resolved (after #id and env substitution)
                            let resolved = match self
                                .state
                                .resolve_all(&mut self.engine.simplifier.context, *expr_id)
                            {
                                Ok(r) => r,
                                Err(_) => *expr_id,
                            };
                            if resolved != *expr_id {
                                println!(
                                    "  Resolved:   {}",
                                    DisplayExpr {
                                        context: &self.engine.simplifier.context,
                                        id: resolved
                                    }
                                );
                            }

                            // Show simplified
                            let (simplified, _) = self.engine.simplifier.simplify(resolved);
                            if simplified != resolved {
                                println!(
                                    "  Simplified: {}",
                                    DisplayExpr {
                                        context: &self.engine.simplifier.context,
                                        id: simplified
                                    }
                                );
                            }
                        }
                        cas_engine::EntryKind::Eq { lhs, rhs } => {
                            // Show LHS and RHS
                            println!(
                                "  LHS:        {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *lhs
                                }
                            );
                            println!(
                                "  RHS:        {}",
                                DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: *rhs
                                }
                            );

                            // Note about equation-as-expression
                            println!();
                            println!("  Note: When used as expression, this becomes (LHS - RHS).");
                        }
                    }
                } else {
                    // Check if this ID was ever assigned (it's above next_id means never existed)
                    // Entry not found — could be deleted or never existed
                    println!("Error: Entry #{} not found.", id);
                    println!("Hint: Use 'history' to see available entries.");
                }
            }
            Err(_) => {
                println!("Error: Invalid entry ID. Use 'show #N' or 'show N'.");
            }
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    fn handle_del_command(&mut self, line: &str) {
        let ids: Vec<u64> = line
            .split_whitespace()
            .filter_map(|s| s.trim_start_matches('#').parse::<u64>().ok())
            .collect();

        if ids.is_empty() {
            println!("Error: No valid IDs specified. Use 'del #1 #2' or 'del 1 2'.");
            return;
        }

        let before_len = self.state.store.len();
        self.state.store.remove(&ids);
        let removed = before_len - self.state.store.len();

        if removed > 0 {
            let id_str: Vec<String> = ids.iter().map(|id| format!("#{}", id)).collect();
            println!("Deleted {} entry/entries: {}", removed, id_str.join(", "));
        } else {
            println!("No entries found with the specified IDs.");
        }
    }

    // ========== END SESSION ENVIRONMENT HANDLERS ==========

    fn handle_det(&mut self, line: &str) {
        let rest = line[4..].trim(); // Remove "det "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in det() function call
                let det_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("det".to_string(), vec![expr]));

                // Simplify to compute determinant
                let (result, steps) = self.engine.simplifier.simplify(det_expr);

                println!("Parsed: det({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                        for event in &step.assumption_events {
                            println!("   ⚠ Domain: {}", event.message);
                        }
                    }
                }

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: result
                    }
                );
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    fn handle_transpose(&mut self, line: &str) {
        let rest = line[10..].trim(); // Remove "transpose "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in transpose() function call
                let transpose_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("transpose".to_string(), vec![expr]));

                // Simplify to compute transpose
                let (result, steps) = self.engine.simplifier.simplify(transpose_expr);

                println!("Parsed: transpose({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                    }
                }

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: result
                    }
                );
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    fn handle_trace(&mut self, line: &str) {
        let rest = line[6..].trim(); // Remove "trace "

        // Parse the matrix expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                // Wrap in trace() function call
                let trace_expr = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("trace".to_string(), vec![expr]));

                // Simplify to compute trace
                let (result, steps) = self.engine.simplifier.simplify(trace_expr);

                println!("Parsed: trace({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                    }
                }

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: result
                    }
                );
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    fn handle_telescope(&mut self, line: &str) {
        let rest = line[10..].trim(); // Remove "telescope "

        if rest.is_empty() {
            println!("Usage: telescope <expression>");
            println!("Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)");
            return;
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", rest);
                println!();

                // Apply telescoping strategy
                let result =
                    cas_engine::telescoping::telescope(&mut self.engine.simplifier.context, expr);

                // Print formatted output
                println!("{}", result.format(&self.engine.simplifier.context));
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses cas_engine::expand::expand() which distributes without educational guards
    fn handle_expand(&mut self, line: &str) {
        use cas_ast::DisplayExpr;

        let rest = line.strip_prefix("expand").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: expand <expr>");
            println!("Description: Aggressively expands and distributes polynomials.");
            println!("Example: expand 1/2 * (sqrt(2) - 1) → sqrt(2)/2 - 1/2");
            return;
        }

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: expr
                    }
                );

                // Use the expansion module directly (bypasses DistributeRule guards)
                let expanded =
                    cas_engine::expand::expand(&mut self.engine.simplifier.context, expr);

                // Simplify to clean up the result
                let (simplified, _steps) = self.engine.simplifier.simplify(expanded);

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: simplified
                    }
                );
            }
            Err(e) => println!("Parse error: {:?}", e),
        }
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    fn handle_expand_log(&mut self, line: &str) {
        use cas_ast::DisplayExpr;

        let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: expand_log <expr>");
            println!("Description: Expand logarithms using log properties.");
            println!("Transformations:");
            println!("  ln(x*y)   → ln(x) + ln(y)");
            println!("  ln(x/y)   → ln(x) - ln(y)");
            println!("  ln(x^n)   → n * ln(x)");
            println!("Example: expand_log ln(x^2 * y) → 2*ln(x) + ln(y)");
            return;
        }

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: expr
                    }
                );

                // Apply LogExpansionRule recursively to all subexpressions
                let expanded = self.expand_log_recursive(expr);

                // NOTE: We do NOT call simplify() here because LogContractionRule
                // (which is in default rules) would immediately undo the expansion.
                // The expanded form is the desired result.

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: expanded
                    }
                );
            }
            Err(e) => println!("Parse error: {:?}", e),
        }
    }

    /// Recursively apply LogExpansionRule to all subexpressions
    fn expand_log_recursive(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        use cas_ast::Expr;
        use cas_engine::parent_context::ParentContext;
        use cas_engine::rule::Rule;
        use cas_engine::rules::logarithms::LogExpansionRule;
        use cas_engine::DomainMode;

        // Create a parent context with Assume mode to allow expansion of symbolic variables
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Assume);

        let rule = LogExpansionRule;

        // Try to apply the rule at this node
        if let Some(rewrite) = rule.apply(&mut self.engine.simplifier.context, expr, &parent_ctx) {
            // Recursively expand the result
            return self.expand_log_recursive(rewrite.new_expr);
        }

        // If rule didn't apply, recurse into children
        let expr_data = self.engine.simplifier.context.get(expr).clone();
        match expr_data {
            Expr::Add(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.engine.simplifier.context.add(Expr::Add(new_l, new_r))
            }
            Expr::Sub(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.engine.simplifier.context.add(Expr::Sub(new_l, new_r))
            }
            Expr::Mul(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.engine.simplifier.context.add(Expr::Mul(new_l, new_r))
            }
            Expr::Div(l, r) => {
                let new_l = self.expand_log_recursive(l);
                let new_r = self.expand_log_recursive(r);
                self.engine.simplifier.context.add(Expr::Div(new_l, new_r))
            }
            Expr::Pow(b, e) => {
                let new_b = self.expand_log_recursive(b);
                let new_e = self.expand_log_recursive(e);
                self.engine.simplifier.context.add(Expr::Pow(new_b, new_e))
            }
            Expr::Neg(inner) => {
                let new_inner = self.expand_log_recursive(inner);
                self.engine.simplifier.context.add(Expr::Neg(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<_> = args.iter().map(|a| self.expand_log_recursive(*a)).collect();
                self.engine
                    .simplifier
                    .context
                    .add(Expr::Function(name, new_args))
            }
            // Atoms: Number, Variable, Constant, Matrix, SessionRef - return as-is
            _ => expr,
        }
    }

    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    fn handle_weierstrass(&mut self, line: &str) {
        let rest = line[12..].trim(); // Remove "weierstrass "

        if rest.is_empty() {
            println!("Usage: weierstrass <expression>");
            println!("Description: Apply Weierstrass substitution (t = tan(x/2))");
            println!("Transforms:");
            println!("  sin(x) → 2t/(1+t²)");
            println!("  cos(x) → (1-t²)/(1+t²)");
            println!("  tan(x) → 2t/(1-t²)");
            println!("Example: weierstrass sin(x) + cos(x)");
            return;
        }

        // Parse the expression
        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(expr) => {
                use cas_ast::DisplayExpr;
                println!("Parsed: {}", rest);
                println!();

                // Apply Weierstrass substitution recursively
                let result = self.apply_weierstrass_recursive(expr);

                // Display result
                let result_str = format!(
                    "{}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: result
                    }
                );
                println!("Weierstrass substitution (t = tan(x/2)):");
                println!("  {} → {}", rest, result_str);

                // Try to simplify the result
                println!();
                println!("Simplifying...");
                let (simplified, _steps) = self.engine.simplifier.simplify(result);
                let simplified_str = format!(
                    "{}",
                    DisplayExpr {
                        context: &self.engine.simplifier.context,
                        id: simplified
                    }
                );
                println!("Result: {}", simplified_str);
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }

    /// Apply Weierstrass substitution recursively to all trig functions
    fn apply_weierstrass_recursive(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        use cas_ast::Expr;

        match self.engine.simplifier.context.get(expr).clone() {
            Expr::Function(name, args)
                if matches!(name.as_str(), "sin" | "cos" | "tan") && args.len() == 1 =>
            {
                let arg = args[0];

                // Build t = tan(x/2) as sin(x/2)/cos(x/2)
                let two_num = self.engine.simplifier.context.num(2);
                let half_arg = self.engine.simplifier.context.add(Expr::Div(arg, two_num));
                let sin_half = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("sin".to_string(), vec![half_arg]));
                let cos_half = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Function("cos".to_string(), vec![half_arg]));
                let t = self
                    .engine
                    .simplifier
                    .context
                    .add(Expr::Div(sin_half, cos_half)); // t = tan(x/2)

                // Apply appropriate transformation
                match name.as_str() {
                    "sin" => {
                        // sin(x) → 2t/(1+t²)
                        let two = self.engine.simplifier.context.num(2);
                        let one = self.engine.simplifier.context.num(1);
                        let t_squared = self.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.engine.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Add(one, t_squared));
                        self.engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "cos" => {
                        // cos(x) → (1-t²)/(1+t²)
                        let one = self.engine.simplifier.context.num(1);
                        let two = self.engine.simplifier.context.num(2);
                        let t_squared = self.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Sub(one, t_squared));
                        let denominator = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Add(one, t_squared));
                        self.engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "tan" => {
                        // tan(x) → 2t/(1-t²)
                        let two = self.engine.simplifier.context.num(2);
                        let one = self.engine.simplifier.context.num(1);
                        let t_squared = self.engine.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.engine.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Sub(one, t_squared));
                        self.engine
                            .simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    _ => expr,
                }
            }
            Expr::Add(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.engine.simplifier.context.add(Expr::Add(new_l, new_r))
            }
            Expr::Sub(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.engine.simplifier.context.add(Expr::Sub(new_l, new_r))
            }
            Expr::Mul(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.engine.simplifier.context.add(Expr::Mul(new_l, new_r))
            }
            Expr::Div(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.engine.simplifier.context.add(Expr::Div(new_l, new_r))
            }
            Expr::Pow(base, exp) => {
                let new_base = self.apply_weierstrass_recursive(base);
                let new_exp = self.apply_weierstrass_recursive(exp);
                self.engine
                    .simplifier
                    .context
                    .add(Expr::Pow(new_base, new_exp))
            }
            Expr::Neg(e) => {
                let new_e = self.apply_weierstrass_recursive(e);
                self.engine.simplifier.context.add(Expr::Neg(new_e))
            }
            Expr::Function(name, args) => {
                // Recurse into function arguments
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_weierstrass_recursive(a))
                    .collect();
                self.engine
                    .simplifier
                    .context
                    .add(Expr::Function(name.clone(), new_args))
            }
            _ => expr, // Number, Variable, Constant, Matrix - leave as is
        }
    }

    fn handle_timeline_solve(&mut self, rest: &str) {
        // Parse equation and variable: "x + 2 = 5, x" or "x + 2 = 5 x"
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
        } else {
            // No comma. Try to see if it looks like "eq var"
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let v_trim = v.trim();
                if !v_trim.is_empty() && v_trim.chars().all(char::is_alphabetic) {
                    (e.trim(), v_trim)
                } else {
                    (rest, "x")
                }
            } else {
                (rest, "x")
            }
        };

        match cas_parser::parse_statement(eq_str, &mut self.engine.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Call solver with step collection enabled and semantic options
                self.engine.simplifier.set_collect_steps(true);
                let solver_opts = cas_engine::solver::SolverOptions {
                    value_domain: self.state.options.value_domain,
                    domain_mode: self.state.options.domain_mode,
                    assume_scope: self.state.options.assume_scope,
                    budget: self.state.options.budget,
                };

                match cas_engine::solver::solve_with_options(
                    &eq,
                    var,
                    &mut self.engine.simplifier,
                    solver_opts,
                ) {
                    Ok((solution_set, steps)) => {
                        if steps.is_empty() {
                            println!("No solving steps to visualize.");
                            println!(
                                "Result: {}",
                                display_solution_set(
                                    &self.engine.simplifier.context,
                                    &solution_set
                                )
                            );
                            return;
                        }

                        // Generate HTML timeline for solve steps
                        let mut timeline = cas_engine::timeline::SolveTimelineHtml::new(
                            &mut self.engine.simplifier.context,
                            &steps,
                            &eq,
                            &solution_set,
                            var,
                        );
                        let html = timeline.to_html();

                        let filename = "timeline.html";
                        match std::fs::write(filename, &html) {
                            Ok(_) => {
                                println!("Solve timeline exported to {}", filename);
                                println!(
                                    "Result: {}",
                                    display_solution_set(
                                        &self.engine.simplifier.context,
                                        &solution_set
                                    )
                                );
                                println!("Open in browser to view interactive visualization.");

                                // Try to auto-open on macOS
                                #[cfg(target_os = "macos")]
                                {
                                    let _ =
                                        std::process::Command::new("open").arg(filename).spawn();
                                }
                            }
                            Err(e) => println!("Error writing file: {}", e),
                        }
                    }
                    Err(e) => println!("Error solving: {}", e),
                }
            }
            Ok(cas_parser::Statement::Expression(_)) => {
                println!("Error: Expected an equation for solve timeline, got an expression.");
                println!("Usage: timeline solve <equation>, <variable>");
                println!("Example: timeline solve x + 2 = 5, x");
            }
            Err(e) => println!("Error parsing equation: {}", e),
        }
    }

    fn handle_solve(&mut self, line: &str) {
        use cas_ast::{DisplayExpr, Expr};
        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

        // solve [--check] <equation>, <var>
        let rest = line[6..].trim();

        // Parse --check flag (one-shot override)
        let (check_enabled, rest) = if rest.starts_with("--check") {
            let after_flag = rest[7..].trim_start();
            (true, after_flag)
        } else {
            // Use session toggle if no explicit flag
            (self.state.options.check_solutions, rest)
        };

        // Split by comma or space to get equation and var
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
        } else {
            // No comma. Try to see if it looks like "eq var"
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let e_trim = e.trim();
                let v_trim = v.trim();
                // Check if v is a variable name (alphabetic) AND
                // the remaining equation doesn't end with '=' (which would mean v is the RHS) AND
                // there are no operators after '=' (which would mean v is part of an expression)
                let has_operators_after_eq = if let Some(eq_pos) = e_trim.find('=') {
                    let after_eq = &e_trim[eq_pos + 1..];
                    after_eq.contains('+')
                        || after_eq.contains('-')
                        || after_eq.contains('*')
                        || after_eq.contains('/')
                        || after_eq.contains('^')
                } else {
                    false
                };
                if !v_trim.is_empty()
                    && v_trim.chars().all(char::is_alphabetic)
                    && !e_trim.ends_with('=')
                    && !has_operators_after_eq
                {
                    (e_trim, v_trim)
                } else {
                    (rest, "x")
                }
            } else {
                (rest, "x")
            }
        };

        // Parse equation part
        // Style signals handled during display logic mostly, removing invalid context access

        // Handle #id manually as Variable to let Engine resolve it, or parse string
        let parsed_expr_res =
            if eq_str.starts_with('#') && eq_str[1..].chars().all(char::is_numeric) {
                // Pass as Variable("#id") - the engine will now handle this resolution!
                Ok(Statement::Expression(
                    self.engine
                        .simplifier
                        .context
                        .add(Expr::Variable(eq_str.to_string())),
                ))
            } else {
                cas_parser::parse_statement(eq_str, &mut self.engine.simplifier.context)
            };

        match parsed_expr_res {
            Ok(stmt) => {
                // Store equation for potential verification
                let original_equation: Option<cas_ast::Equation> = match &stmt {
                    Statement::Equation(eq) => Some(eq.clone()),
                    Statement::Expression(_) => None,
                };

                let (kind, parsed_expr) = match stmt {
                    Statement::Equation(eq) => {
                        let eq_expr = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Function("Equal".to_string(), vec![eq.lhs, eq.rhs]));
                        (
                            EntryKind::Eq {
                                lhs: eq.lhs,
                                rhs: eq.rhs,
                            },
                            eq_expr,
                        )
                    }
                    Statement::Expression(e) => (EntryKind::Expr(e), e),
                };

                let req = EvalRequest {
                    raw_input: eq_str.to_string(),
                    parsed: parsed_expr,
                    kind,
                    action: EvalAction::Solve {
                        var: var.to_string(),
                    },
                    auto_store: true,
                };

                match self.engine.eval(&mut self.state, req) {
                    Ok(output) => {
                        // Show ID
                        if let Some(id) = output.stored_id {
                            print!("#{}: ", id);
                        }
                        println!("Solving for {}...", var);

                        for w in &output.domain_warnings {
                            println!("⚠ {} (from {})", w.message, w.rule_name);
                        }

                        // Show solver assumptions summary if any
                        if !output.solver_assumptions.is_empty() {
                            let items: Vec<String> = output
                                .solver_assumptions
                                .iter()
                                .map(|a| {
                                    if a.count > 1 {
                                        format!("{}({}) (×{})", a.kind, a.expr, a.count)
                                    } else {
                                        format!("{}({})", a.kind, a.expr)
                                    }
                                })
                                .collect();
                            println!("⚠ Assumptions: {}", items.join(", "));
                        }

                        // Show Solve Steps
                        if !output.solve_steps.is_empty() && self.verbosity != Verbosity::None {
                            if self.verbosity != Verbosity::Succinct {
                                println!("Steps:");
                            }
                            // Prepare scoped renderer if scopes are present
                            let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                            let has_scopes = !output.output_scopes.is_empty();
                            let renderer = if has_scopes {
                                Some(cas_ast::display_transforms::ScopedRenderer::new(
                                    &self.engine.simplifier.context,
                                    &output.output_scopes,
                                    &registry,
                                ))
                            } else {
                                None
                            };

                            for (i, step) in output.solve_steps.iter().enumerate() {
                                println!("{}. {}", i + 1, step.description);
                                // Display equation after step with scoped transforms
                                let ctx = &self.engine.simplifier.context;
                                let (lhs_str, rhs_str) = if let Some(ref r) = renderer {
                                    (
                                        r.render(step.equation_after.lhs),
                                        r.render(step.equation_after.rhs),
                                    )
                                } else {
                                    (
                                        DisplayExpr {
                                            context: ctx,
                                            id: step.equation_after.lhs,
                                        }
                                        .to_string(),
                                        DisplayExpr {
                                            context: ctx,
                                            id: step.equation_after.rhs,
                                        }
                                        .to_string(),
                                    )
                                };
                                println!(
                                    "   -> {} {} {}",
                                    lhs_str, step.equation_after.op, rhs_str
                                );
                            }
                        }

                        match output.result {
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set including Conditional
                                let ctx = &self.engine.simplifier.context;
                                println!("Result: {}", display_solution_set(ctx, solution_set));
                            }
                            EvalResult::Set(ref sols) => {
                                // Legacy: discrete solutions as Vec<ExprId>
                                let ctx = &self.engine.simplifier.context;
                                let sol_strs: Vec<String> = if !output.output_scopes.is_empty() {
                                    let registry = cas_ast::display_transforms::DisplayTransformRegistry::with_defaults();
                                    let renderer = cas_ast::display_transforms::ScopedRenderer::new(
                                        ctx,
                                        &output.output_scopes,
                                        &registry,
                                    );
                                    sols.iter().map(|id| renderer.render(*id)).collect()
                                } else {
                                    // Standard display without transforms
                                    sols.iter()
                                        .map(|id| {
                                            DisplayExpr {
                                                context: ctx,
                                                id: *id,
                                            }
                                            .to_string()
                                        })
                                        .collect()
                                };
                                if sol_strs.is_empty() {
                                    println!("Result: No solution");
                                } else {
                                    println!("Result: {{ {} }}", sol_strs.join(", "));
                                }
                            }
                            _ => println!("Result: {:?}", output.result),
                        }

                        // Issue #5: Solution verification (--check flag)
                        if check_enabled {
                            if let EvalResult::SolutionSet(ref solution_set) = output.result {
                                if let Some(ref eq) = original_equation {
                                    use cas_engine::solver::check::{
                                        verify_solution_set, VerifySummary,
                                    };

                                    let verify_result = verify_solution_set(
                                        &mut self.engine.simplifier,
                                        eq,
                                        var,
                                        solution_set,
                                    );

                                    // Display verification status
                                    match verify_result.summary {
                                        VerifySummary::AllVerified => {
                                            println!("✓ All solutions verified");
                                        }
                                        VerifySummary::PartiallyVerified => {
                                            println!("⚠ Some solutions verified");
                                            for (sol_id, status) in &verify_result.solutions {
                                                let sol_str = DisplayExpr {
                                                    context: &self.engine.simplifier.context,
                                                    id: *sol_id,
                                                }
                                                .to_string();
                                                match status {
                                                    cas_engine::solver::check::VerifyStatus::Verified => {
                                                        println!("  ✓ {} = {} verified", var, sol_str);
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::Unverifiable { reason, .. } => {
                                                        println!("  ⚠ {} = {}: {}", var, sol_str, reason);
                                                    }
                                                    cas_engine::solver::check::VerifyStatus::NotCheckable { reason } => {
                                                        println!("  ℹ {} = {}: {}", var, sol_str, reason);
                                                    }
                                                }
                                            }
                                        }
                                        VerifySummary::NoneVerified => {
                                            println!("⚠ No solutions could be verified");
                                        }
                                        VerifySummary::NotCheckable => {
                                            if let Some(desc) = verify_result.guard_description {
                                                println!("ℹ {}", desc);
                                            } else {
                                                println!("ℹ Solution type not checkable");
                                            }
                                        }
                                        VerifySummary::Empty => {
                                            // Empty set - nothing to verify
                                        }
                                    }
                                }
                            }
                        }

                        // V2.1 Issue #3: Explain mode - structured summary for solve
                        // Collect blocked hints for explain output
                        let hints = cas_engine::domain::take_blocked_hints();
                        let has_assumptions = !output.solver_assumptions.is_empty();
                        let has_blocked = !hints.is_empty();

                        if self.explain_mode && (has_assumptions || has_blocked) {
                            println!(); // Separator line
                            let ctx = &self.engine.simplifier.context;

                            // Block 1: Assumptions used
                            if has_assumptions {
                                println!("ℹ️ Assumptions used:");
                                // Dedup and stable order by (kind, expr)
                                let mut assumption_items: Vec<_> = output
                                    .solver_assumptions
                                    .iter()
                                    .map(|a| {
                                        let cond_str = match a.kind.as_str() {
                                            "Positive" => format!("{} > 0", a.expr),
                                            "NonZero" => format!("{} ≠ 0", a.expr),
                                            "NonNegative" => format!("{} ≥ 0", a.expr),
                                            _ => format!("{} ({})", a.expr, a.kind),
                                        };
                                        cond_str
                                    })
                                    .collect();
                                // Stable sort and dedup
                                assumption_items.sort();
                                assumption_items.dedup();
                                for cond in assumption_items {
                                    println!("  - {}", cond);
                                }
                            }

                            // Block 2: Blocked simplifications
                            if has_blocked {
                                println!("ℹ️ Blocked simplifications:");
                                // Helper to format condition with expression
                                let format_condition = |hint: &cas_engine::BlockedHint| -> String {
                                    let expr_str = cas_ast::DisplayExpr {
                                        context: ctx,
                                        id: hint.expr_id,
                                    }
                                    .to_string();
                                    match hint.key.kind() {
                                        "positive" => format!("{} > 0", expr_str),
                                        "nonzero" => format!("{} ≠ 0", expr_str),
                                        "nonnegative" => format!("{} ≥ 0", expr_str),
                                        _ => format!("{} ({})", expr_str, hint.key.kind()),
                                    }
                                };

                                // Dedup by (condition, rule)
                                let mut blocked_items: Vec<_> = hints
                                    .iter()
                                    .map(|h| (format_condition(h), h.rule.to_string()))
                                    .collect();
                                blocked_items.sort();
                                blocked_items.dedup();
                                for (cond, rule) in blocked_items {
                                    println!("  - requires {}  [{}]", cond, rule);
                                }

                                // Contextual suggestion
                                let suggestion = match self.state.options.domain_mode {
                                    cas_engine::DomainMode::Strict => {
                                        "tip: use `domain generic` or `domain assume` to allow"
                                    }
                                    cas_engine::DomainMode::Generic => {
                                        "tip: use `semantics set domain assume` to allow"
                                    }
                                    cas_engine::DomainMode::Assume => {
                                        "tip: assumptions already enabled"
                                    }
                                };
                                println!("  {}", suggestion);
                            }
                        } else if has_blocked && self.state.options.hints_enabled {
                            // Legacy: show blocked hints even without explain_mode if hints_enabled
                            let ctx = &self.engine.simplifier.context;
                            let format_condition = |hint: &cas_engine::BlockedHint| -> String {
                                let expr_str = cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: hint.expr_id,
                                }
                                .to_string();
                                match hint.key.kind() {
                                    "positive" => format!("{} > 0", expr_str),
                                    "nonzero" => format!("{} ≠ 0", expr_str),
                                    "nonnegative" => format!("{} ≥ 0", expr_str),
                                    _ => format!("{} ({})", expr_str, hint.key.kind()),
                                }
                            };

                            let suggestion = match self.state.options.domain_mode {
                                cas_engine::DomainMode::Strict => {
                                    "use `domain generic` or `domain assume` to allow"
                                }
                                cas_engine::DomainMode::Generic => {
                                    "use `semantics set domain assume` to allow"
                                }
                                cas_engine::DomainMode::Assume => "assumptions already enabled",
                            };

                            println!("\nℹ️ Blocked simplifications:");
                            for hint in &hints {
                                println!("  - requires {} [{}]", format_condition(hint), hint.rule);
                            }
                            println!("  tip: {}", suggestion);
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }
    fn show_simplification_steps(
        &mut self,
        expr: cas_ast::ExprId,
        steps: Vec<cas_engine::Step>,
        style_signals: cas_ast::root_style::ParseStyleSignals,
    ) {
        use cas_ast::root_style::StylePreferences;
        use cas_ast::DisplayExprStyled;

        if self.verbosity == Verbosity::None {
            return;
        }

        // Create global style preferences from input signals + AST
        let style_prefs = StylePreferences::from_expression_with_signals(
            &self.engine.simplifier.context,
            expr,
            Some(&style_signals),
        );

        if steps.is_empty() {
            // Even with no engine steps, show didactic sub-steps if there are fraction sums
            let standalone_substeps = cas_engine::didactic::get_standalone_substeps(
                &self.engine.simplifier.context,
                expr,
            );

            if !standalone_substeps.is_empty() && self.verbosity != Verbosity::Succinct {
                println!("Computation:");
                // Helper function for LaTeX to plain text
                fn latex_to_text(s: &str) -> String {
                    let mut result = s.to_string();
                    while let Some(start) = result.find("\\frac{") {
                        let end_start = start + 6;
                        if let Some(first_close) = result[end_start..].find('}') {
                            let numer_end = end_start + first_close;
                            let numer = &result[end_start..numer_end];
                            if result.len() > numer_end + 1
                                && result.chars().nth(numer_end + 1) == Some('{')
                            {
                                if let Some(second_close) = result[numer_end + 2..].find('}') {
                                    let denom_end = numer_end + 2 + second_close;
                                    let denom = &result[numer_end + 2..denom_end];
                                    let replacement = format!("({}/{})", numer, denom);
                                    result = format!(
                                        "{}{}{}",
                                        &result[..start],
                                        replacement,
                                        &result[denom_end + 1..]
                                    );
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                    result.replace("\\", "")
                }

                for sub in &standalone_substeps {
                    println!("   → {}", sub.description);
                    if !sub.before_latex.is_empty() {
                        println!(
                            "     {} → {}",
                            latex_to_text(&sub.before_latex),
                            latex_to_text(&sub.after_latex)
                        );
                    }
                }
            } else if self.verbosity != Verbosity::Succinct {
                println!("No simplification steps needed.");
            }
        } else {
            if self.verbosity != Verbosity::Succinct {
                println!("Steps:");
            }

            // Enrich steps ONCE before iterating
            let enriched_steps = cas_engine::didactic::enrich_steps(
                &self.engine.simplifier.context,
                expr,
                steps.clone(),
            );

            let mut current_root = expr;
            let mut step_count = 0;
            let mut sub_steps_shown = false; // Track to show sub-steps only on first visible step
            for (step_idx, step) in steps.iter().enumerate() {
                if should_show_step(step, self.verbosity) {
                    // Early check for display no-op: skip step entirely if before/after display identical
                    let before_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.engine.simplifier.context,
                            step.before,
                            &style_prefs
                        )
                    ));
                    let after_disp = clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(
                            &self.engine.simplifier.context,
                            step.after,
                            &style_prefs
                        )
                    ));
                    // Display no-op check removed/simplified for brevity, logic copied from prev if needed
                    // But let's assume helper needs to be robust. I'll rely on copied logic.
                    if before_disp == after_disp {
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        }
                        continue;
                    }

                    step_count += 1;

                    if self.verbosity == Verbosity::Succinct {
                        // Low mode: just global state
                        current_root = reconstruct_global_expr(
                            &mut self.engine.simplifier.context,
                            current_root,
                            &step.path,
                            step.after,
                        );
                        println!(
                            "-> {}",
                            DisplayExprStyled::new(
                                &self.engine.simplifier.context,
                                current_root,
                                &style_prefs
                            )
                        );
                    } else {
                        // Normal/Verbose
                        println!("{}. {}  [{}]", step_count, step.description, step.rule_name);

                        if self.verbosity == Verbosity::Verbose
                            || self.verbosity == Verbosity::Normal
                        {
                            // Show Before: global expression before this step (always)
                            if let Some(global_before) = step.global_before {
                                println!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.engine.simplifier.context,
                                            global_before,
                                            &style_prefs
                                        )
                                    ))
                                );
                            } else {
                                println!(
                                    "   Before: {}",
                                    clean_display_string(&format!(
                                        "{}",
                                        DisplayExprStyled::new(
                                            &self.engine.simplifier.context,
                                            current_root,
                                            &style_prefs
                                        )
                                    ))
                                );
                            }

                            // Didactic: Show sub-steps AFTER Before: line
                            if !sub_steps_shown {
                                if let Some(enriched_step) = enriched_steps.get(step_idx) {
                                    if !enriched_step.sub_steps.is_empty() {
                                        sub_steps_shown = true;
                                        // Helper function for LaTeX to plain text
                                        fn latex_to_text(s: &str) -> String {
                                            let mut result = s.to_string();
                                            while let Some(start) = result.find("\\frac{") {
                                                let end_start = start + 6;
                                                if let Some(first_close) =
                                                    result[end_start..].find('}')
                                                {
                                                    let numer_end = end_start + first_close;
                                                    let numer = &result[end_start..numer_end];
                                                    if result.len() > numer_end + 1
                                                        && result.chars().nth(numer_end + 1)
                                                            == Some('{')
                                                    {
                                                        if let Some(second_close) =
                                                            result[numer_end + 2..].find('}')
                                                        {
                                                            let denom_end =
                                                                numer_end + 2 + second_close;
                                                            let denom =
                                                                &result[numer_end + 2..denom_end];
                                                            let replacement =
                                                                format!("({}/{})", numer, denom);
                                                            result = format!(
                                                                "{}{}{}",
                                                                &result[..start],
                                                                replacement,
                                                                &result[denom_end + 1..]
                                                            );
                                                            continue;
                                                        }
                                                    }
                                                }
                                                break;
                                            }
                                            result.replace("\\", "")
                                        }

                                        // Show title for substeps section (detect type from description)
                                        let has_fraction_sum =
                                            enriched_step.sub_steps.iter().any(|s| {
                                                s.description.contains("common denominator")
                                                    || s.description.contains("Sum the fractions")
                                            });
                                        let has_factorization =
                                            enriched_step.sub_steps.iter().any(|s| {
                                                s.description.contains("Cancel common factor")
                                                    || s.description.contains("Factor")
                                            });

                                        if has_fraction_sum {
                                            println!("   [Suma de fracciones en exponentes]");
                                        } else if has_factorization {
                                            println!("   [Factorización de polinomios]");
                                        }

                                        for sub in &enriched_step.sub_steps {
                                            println!("      → {}", sub.description);
                                            if !sub.before_latex.is_empty() {
                                                println!(
                                                    "        {} → {}",
                                                    latex_to_text(&sub.before_latex),
                                                    latex_to_text(&sub.after_latex)
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            // Show Rule: local transformation
                            let (rule_before_id, rule_after_id) =
                                match (step.before_local, step.after_local) {
                                    (Some(bl), Some(al)) => (bl, al),
                                    _ => (step.before, step.after),
                                };

                            let before_disp = clean_display_string(&format!(
                                "{}",
                                DisplayExprStyled::new(
                                    &self.engine.simplifier.context,
                                    rule_before_id,
                                    &style_prefs
                                )
                            ));
                            // Use scoped renderer for after expression if rule has transforms
                            let after_disp = clean_display_string(&render_with_rule_scope(
                                &self.engine.simplifier.context,
                                rule_after_id,
                                &step.rule_name,
                            ));

                            if before_disp == after_disp {
                                if let Some(global_after) = step.global_after {
                                    current_root = global_after;
                                }
                                continue;
                            }

                            println!("   Rule: {} -> {}", before_disp, after_disp);
                        }

                        // Use precomputed global_after if available, fall back to reconstruction
                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        } else {
                            current_root = reconstruct_global_expr(
                                &mut self.engine.simplifier.context,
                                current_root,
                                &step.path,
                                step.after,
                            );
                        }

                        // Show After
                        if self.verbosity == Verbosity::Normal
                            || self.verbosity == Verbosity::Verbose
                        {
                            println!(
                                "   After: {}",
                                clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.engine.simplifier.context,
                                        current_root,
                                        &style_prefs
                                    )
                                ))
                            );

                            for event in &step.assumption_events {
                                println!("   ⚠ Domain: {}", event.message);
                            }
                        }
                    }
                } else if let Some(global_after) = step.global_after {
                    current_root = global_after;
                } else {
                    current_root = reconstruct_global_expr(
                        &mut self.engine.simplifier.context,
                        current_root,
                        &step.path,
                        step.after,
                    );
                }
            }
        }
    }

    fn handle_eval(&mut self, line: &str) {
        use cas_ast::root_style::ParseStyleSignals;

        use cas_engine::eval::{EvalAction, EvalRequest, EvalResult};
        use cas_engine::EntryKind;
        use cas_parser::Statement;

        let style_signals = ParseStyleSignals::from_input_string(line);
        let parser_result = cas_parser::parse_statement(line, &mut self.engine.simplifier.context);

        match parser_result {
            Ok(stmt) => {
                // Map to EvalRequest
                let (kind, parsed_expr) = match stmt {
                    Statement::Equation(eq) => {
                        let eq_expr = self
                            .engine
                            .simplifier
                            .context
                            .add(Expr::Function("Equal".to_string(), vec![eq.lhs, eq.rhs]));
                        (
                            EntryKind::Eq {
                                lhs: eq.lhs,
                                rhs: eq.rhs,
                            },
                            eq_expr,
                        )
                    }
                    Statement::Expression(e) => (EntryKind::Expr(e), e),
                };

                let req = EvalRequest {
                    raw_input: line.to_string(),
                    parsed: parsed_expr,
                    kind,
                    // Eval usually just Simplifies.
                    action: EvalAction::Simplify,
                    auto_store: true,
                };

                match self.engine.eval(&mut self.state, req) {
                    Ok(output) => {
                        // Display entry number with parsed expression
                        // NOTE: Removed duplicate print! that was causing "#1: #1:" display bug
                        // I'll skip it to reduce noise or rely on Result.
                        // Actually old logic printed: `#{id}  {expr}`.
                        if let Some(id) = output.stored_id {
                            println!(
                                "#{}: {}",
                                id,
                                cas_ast::DisplayExpr {
                                    context: &self.engine.simplifier.context,
                                    id: output.parsed
                                }
                            );
                        }

                        // Show warnings
                        for w in output.domain_warnings {
                            println!("⚠ {} (from {})", w.message, w.rule_name);
                        }

                        // Collect assumptions from steps for assumption reporting (before steps are consumed)
                        // Deduplicate by (condition_kind, expr_fingerprint) and group by rule
                        let show_assumptions = self.state.options.assumption_reporting
                            != cas_engine::AssumptionReporting::Off;
                        let assumed_conditions: Vec<(String, String)> = if show_assumptions {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let mut result = Vec::new();
                            for step in &output.steps {
                                for event in &step.assumption_events {
                                    // Dedupe by fingerprint
                                    let fp = match &event.key {
                                        cas_engine::assumptions::AssumptionKey::NonZero { expr_fingerprint } => *expr_fingerprint,
                                        cas_engine::assumptions::AssumptionKey::Positive { expr_fingerprint } => *expr_fingerprint + 1_000_000,
                                        cas_engine::assumptions::AssumptionKey::NonNegative { expr_fingerprint } => *expr_fingerprint + 2_000_000,
                                        cas_engine::assumptions::AssumptionKey::Defined { expr_fingerprint } => *expr_fingerprint + 3_000_000,
                                        cas_engine::assumptions::AssumptionKey::InvTrigPrincipalRange { arg_fingerprint, .. } => *arg_fingerprint + 4_000_000,
                                        cas_engine::assumptions::AssumptionKey::ComplexPrincipalBranch { arg_fingerprint, .. } => *arg_fingerprint + 5_000_000,
                                    };
                                    if seen.insert(fp) {
                                        // Format: "x ≠ 0" instead of "≠ 0 (NonZero)"
                                        let condition = match &event.key {
                                            cas_engine::assumptions::AssumptionKey::NonZero { .. } => {
                                                format!("{} ≠ 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::Positive { .. } => {
                                                format!("{} > 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::NonNegative { .. } => {
                                                format!("{} ≥ 0", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::Defined { .. } => {
                                                format!("{} is defined", event.expr_display)
                                            }
                                            cas_engine::assumptions::AssumptionKey::InvTrigPrincipalRange { func, .. } => {
                                                format!("{} in {} principal range", event.expr_display, func)
                                            }
                                            cas_engine::assumptions::AssumptionKey::ComplexPrincipalBranch { func, .. } => {
                                                format!("{}({}) principal branch", func, event.expr_display)
                                            }
                                        };
                                        let rule = step.rule_name.clone();
                                        result.push((condition, rule));
                                    }
                                }
                            }
                            result
                        } else {
                            Vec::new()
                        };

                        // Show steps using helper
                        // We use output.resolved (input to simplify) and output.steps
                        if !output.steps.is_empty() || self.verbosity != Verbosity::None {
                            // trigger logic if verbosity on
                            self.show_simplification_steps(
                                output.resolved,
                                output.steps,
                                style_signals.clone(),
                            );
                        }

                        // Show Final Result with style sniffing (root notation preservation)
                        let style_prefs = cas_ast::StylePreferences::from_expression_with_signals(
                            &self.engine.simplifier.context,
                            output.parsed,
                            Some(&style_signals),
                        );

                        match output.result {
                            EvalResult::Expr(res) => {
                                // Check if it is Equal function
                                let context = &self.engine.simplifier.context;
                                if let Expr::Function(name, args) = context.get(res) {
                                    if name == "Equal" && args.len() == 2 {
                                        println!(
                                            "Result: {} = {}",
                                            cas_ast::DisplayExprStyled::new(
                                                context,
                                                args[0],
                                                &style_prefs
                                            ),
                                            cas_ast::DisplayExprStyled::new(
                                                context,
                                                args[1],
                                                &style_prefs
                                            )
                                        );
                                        return;
                                    }
                                }

                                println!(
                                    "Result: {}",
                                    cas_ast::DisplayExprStyled::new(context, res, &style_prefs)
                                );
                            }
                            EvalResult::SolutionSet(ref solution_set) => {
                                // V2.0: Display full solution set
                                let ctx = &self.engine.simplifier.context;
                                println!("Result: {}", display_solution_set(ctx, solution_set));
                            }
                            EvalResult::Set(_sols) => {
                                println!("Result: Set(...)"); // Simplify result logic doesn't usually produce Set
                            }
                            EvalResult::Bool(b) => println!("Result: {}", b),
                            EvalResult::None => {}
                        }

                        // Display blocked hints (pedagogical warnings for Generic mode)
                        // Respects hints_enabled option (can be toggled with `semantics set hints off`)
                        let hints = self.engine.simplifier.take_blocked_hints();
                        if !hints.is_empty() && self.state.options.hints_enabled {
                            let ctx = &self.engine.simplifier.context;

                            // Helper to format condition with expression
                            let format_condition = |hint: &cas_engine::BlockedHint| -> String {
                                let expr_str = cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: hint.expr_id,
                                }
                                .to_string();
                                match hint.key.kind() {
                                    "positive" => format!("{} > 0", expr_str),
                                    "nonzero" => format!("{} ≠ 0", expr_str),
                                    "nonnegative" => format!("{} ≥ 0", expr_str),
                                    _ => format!("{} ({})", expr_str, hint.key.kind()),
                                }
                            };

                            // Group hints by rule name
                            let mut grouped: std::collections::HashMap<&str, Vec<String>> =
                                std::collections::HashMap::new();
                            for hint in &hints {
                                grouped
                                    .entry(hint.rule)
                                    .or_default()
                                    .push(format_condition(hint));
                            }

                            // Contextual suggestion based on current mode
                            let suggestion = match self.state.options.domain_mode {
                                cas_engine::DomainMode::Strict => {
                                    "use `domain generic` or `domain assume` to allow"
                                }
                                cas_engine::DomainMode::Generic => {
                                    "use `semantics set domain assume` to allow analytic assumptions"
                                }
                                cas_engine::DomainMode::Assume => {
                                    // Should not happen, but fallback
                                    "assumptions already enabled"
                                }
                            };

                            // Display grouped hints
                            if grouped.len() == 1 && hints.len() == 1 {
                                // Single hint: compact format
                                let hint = &hints[0];
                                println!(
                                    "ℹ️  Blocked: requires {} [{}]",
                                    format_condition(hint),
                                    hint.rule
                                );
                                println!("   {}", suggestion);
                            } else {
                                // Multiple hints or multiple rules: grouped format
                                println!("ℹ️  Some simplifications were blocked:");
                                for (rule, conditions) in &grouped {
                                    if conditions.len() == 1 {
                                        println!(" - Requires {}  [{}]", conditions[0], rule);
                                    } else {
                                        // Compact multiple conditions for same rule
                                        println!(
                                            " - Requires {}  [{}]",
                                            conditions.join(", "),
                                            rule
                                        );
                                    }
                                }
                                println!("   Tip: {}", suggestion);
                            }
                        }

                        // Show assumptions summary when assumption_reporting is enabled (after hints)
                        if show_assumptions && !assumed_conditions.is_empty() {
                            // Group conditions by rule
                            let mut by_rule: std::collections::HashMap<String, Vec<String>> =
                                std::collections::HashMap::new();
                            for (condition, rule) in &assumed_conditions {
                                by_rule
                                    .entry(rule.clone())
                                    .or_default()
                                    .push(condition.clone());
                            }

                            if assumed_conditions.len() == 1 {
                                let (cond, rule) = &assumed_conditions[0];
                                println!("ℹ️  Assumptions used (assumed): {} [{}]", cond, rule);
                            } else {
                                println!("ℹ️  Assumptions used (assumed):");
                                for (rule, conds) in &by_rule {
                                    println!("   - {} [{}]", conds.join(", "), rule);
                                }
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
    }
    fn handle_full_simplify(&mut self, line: &str) {
        // simplify <expr>
        // Uses a temporary simplifier with ALL default rules (including aggressive distribution)
        let expr_str = line[9..].trim();

        // We need to use the existing context to parse, but then we want to simplify using a different rule set.
        // The Simplifier struct owns the context.
        // Option 1: Create a new Simplifier, parse into it.
        // Option 2: Swap rules in current simplifier? (Hard)
        // Option 3: Create a new Simplifier, copy context? (Hard)

        // Easiest: Create new simplifier, parse string into it.
        // Note: Variables from previous history won't be available if we don't copy context.
        // But REPL history is just text in rustyline, not context state (unless we implement variable storage).
        // Current implementation: Context is reset per line? No, self.engine.simplifier.context persists.
        // If we want to support "x = 5; simplify x", we need to share context.

        // Better approach:
        // 1. Parse expression using current context.
        // 2. Create a temporary Simplifier that SHARES the context?
        //    Simplifier owns Context. We can't easily share.
        //    But we can temporarily TAKE the context, use it in a new Simplifier, and then put it back.

        let mut temp_simplifier = Simplifier::with_default_rules();
        // Swap context and profiler so temp_simplifier uses main profiler
        std::mem::swap(
            &mut self.engine.simplifier.context,
            &mut temp_simplifier.context,
        );
        std::mem::swap(
            &mut self.engine.simplifier.profiler,
            &mut temp_simplifier.profiler,
        );

        // Ensure we have the aggressive rules we want (DistributeRule is in default)
        // Also add DistributeConstantRule just in case (though DistributeRule covers it)

        // Set steps mode
        temp_simplifier.set_collect_steps(self.verbosity != Verbosity::None);

        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
                // Note: Tool dispatcher is handled in Engine::eval, not here
                // This code path is for timeline/specific commands, not regular expression evaluation

                // Resolve session variables (A, B, etc.) before simplifying
                let resolved_expr = match self.state.resolve_all(&mut temp_simplifier.context, expr)
                {
                    Ok(resolved) => resolved,
                    Err(e) => {
                        println!("Error resolving variables: {:?}", e);
                        // Swap context and profiler back before returning
                        std::mem::swap(
                            &mut self.engine.simplifier.context,
                            &mut temp_simplifier.context,
                        );
                        std::mem::swap(
                            &mut self.engine.simplifier.profiler,
                            &mut temp_simplifier.profiler,
                        );
                        return;
                    }
                };

                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                // Parse equation part
                // Style signals handled during display logic mostly, removing invalid context access
                let style_signals = ParseStyleSignals::from_input_string(expr_str);
                let style_prefs = StylePreferences::from_expression_with_signals(
                    &temp_simplifier.context,
                    resolved_expr,
                    Some(&style_signals),
                );

                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &temp_simplifier.context,
                        id: resolved_expr
                    }
                );
                // Use session options (expand_policy, context_mode, etc.) for simplification
                let mut opts = self.state.options.to_simplify_options();
                opts.collect_steps = self.verbosity != Verbosity::None;

                let (simplified, steps, _stats) =
                    temp_simplifier.simplify_with_stats(resolved_expr, opts);

                if self.verbosity != Verbosity::None {
                    if steps.is_empty() {
                        if self.verbosity != Verbosity::Succinct {
                            println!("No simplification steps needed.");
                        }
                    } else {
                        if self.verbosity != Verbosity::Succinct {
                            println!("Steps (Aggressive Mode):");
                        }
                        let mut current_root = expr;
                        let mut step_count = 0;
                        for step in steps.iter() {
                            if should_show_step(step, self.verbosity) {
                                step_count += 1;

                                if self.verbosity == Verbosity::Succinct {
                                    // Low mode: just global state
                                    current_root = reconstruct_global_expr(
                                        &mut temp_simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                    println!(
                                        "-> {}",
                                        DisplayExpr {
                                            context: &temp_simplifier.context,
                                            id: current_root
                                        }
                                    );
                                } else {
                                    // Normal/Verbose
                                    println!(
                                        "{}. {}  [{}]",
                                        step_count, step.description, step.rule_name
                                    );

                                    if self.verbosity == Verbosity::Verbose
                                        || self.verbosity == Verbosity::Normal
                                    {
                                        // Show Before: global expression before this step
                                        if let Some(global_before) = step.global_before {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprStyled::new(
                                                        &temp_simplifier.context,
                                                        global_before,
                                                        &style_prefs
                                                    )
                                                ))
                                            );
                                        } else {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprStyled::new(
                                                        &temp_simplifier.context,
                                                        current_root,
                                                        &style_prefs
                                                    )
                                                ))
                                            );
                                        }

                                        // Show Rule: local transformation
                                        // Use before_local/after_local if available (for n-ary rules),
                                        // otherwise fall back to before/after
                                        let (rule_before_id, rule_after_id) =
                                            match (step.before_local, step.after_local) {
                                                (Some(bl), Some(al)) => (bl, al),
                                                _ => (step.before, step.after),
                                            };

                                        let before_disp = clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &temp_simplifier.context,
                                                rule_before_id,
                                                &style_prefs
                                            )
                                        ));
                                        // Use scoped renderer for after expression if rule has transforms
                                        let after_disp =
                                            clean_display_string(&render_with_rule_scope(
                                                &temp_simplifier.context,
                                                rule_after_id,
                                                &step.rule_name,
                                            ));

                                        println!("   Rule: {} -> {}", before_disp, after_disp);
                                    }

                                    // Use precomputed global_after if available, fall back to reconstruction
                                    if let Some(global_after) = step.global_after {
                                        current_root = global_after;
                                    } else {
                                        current_root = reconstruct_global_expr(
                                            &mut temp_simplifier.context,
                                            current_root,
                                            &step.path,
                                            step.after,
                                        );
                                    }

                                    // Show After: global expression after this step
                                    println!(
                                        "   After: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &temp_simplifier.context,
                                                current_root,
                                                &style_prefs
                                            )
                                        ))
                                    );

                                    for event in &step.assumption_events {
                                        println!("   ⚠ Domain: {}", event.message);
                                    }
                                }
                            } else {
                                // Step not shown, but still update current_root for subsequent steps
                                if let Some(global_after) = step.global_after {
                                    current_root = global_after;
                                } else {
                                    current_root = reconstruct_global_expr(
                                        &mut temp_simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                }
                            }
                        }
                    }
                }
                // Use DisplayExprStyled with detected preferences for consistent output
                println!(
                    "Result: {}",
                    DisplayExprStyled::new(&temp_simplifier.context, simplified, &style_prefs)
                );
            }
            Err(e) => println!("Error: {}", e),
        }

        // Swap context and profiler back
        std::mem::swap(
            &mut self.engine.simplifier.context,
            &mut temp_simplifier.context,
        );
        std::mem::swap(
            &mut self.engine.simplifier.profiler,
            &mut temp_simplifier.profiler,
        );

        // Store health report for the `health` command (if health tracking is enabled)
        if self.health_enabled {
            self.last_health_report = Some(self.engine.simplifier.profiler.health_report());
        }
    }

    /// Handle the 'rationalize' command for rationalizing denominators
    /// Example: rationalize 1/(1 + sqrt(2) + sqrt(3))
    fn handle_rationalize(&mut self, line: &str) {
        use cas_engine::rationalize::{
            rationalize_denominator, RationalizeConfig, RationalizeResult,
        };

        let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: rationalize <expr>");
            println!("Example: rationalize 1/(1 + sqrt(2) + sqrt(3))");
            return;
        }

        match cas_parser::parse(rest, &mut self.engine.simplifier.context) {
            Ok(parsed_expr) => {
                // CANONICALIZE: Rebuild tree to trigger Add auto-flatten at all levels
                // Parser creates tree incrementally, so nested Adds may not be flattened
                // normalize_core forces reconstruction ensuring canonical form
                let expr = cas_engine::canonical_forms::normalize_core(
                    &mut self.engine.simplifier.context,
                    parsed_expr,
                );
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let user_style = cas_ast::detect_root_style(&self.engine.simplifier.context, expr);

                let disp = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: expr,
                };
                println!("Parsed: {}", disp);

                let config = RationalizeConfig::default();
                let result =
                    rationalize_denominator(&mut self.engine.simplifier.context, expr, &config);

                match result {
                    RationalizeResult::Success(rationalized) => {
                        // Simplify the result
                        let (simplified, _) = self.engine.simplifier.simplify(rationalized);

                        // Use StyledExpr with detected style for consistent output
                        let result_disp = cas_ast::StyledExpr::new(
                            &self.engine.simplifier.context,
                            simplified,
                            user_style,
                        );
                        println!("Rationalized: {}", result_disp);
                    }
                    RationalizeResult::NotApplicable => {
                        println!("Cannot rationalize: denominator is not a sum of surds");
                        println!("(Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)");
                    }
                    RationalizeResult::BudgetExceeded => {
                        println!("Rationalization aborted: expression became too complex");
                    }
                }
            }
            Err(e) => println!("Parse error: {:?}", e),
        }
    }

    /// Handle the limit command: compute limit at ±∞
    fn handle_limit(&mut self, line: &str) {
        use cas_engine::limits::{limit, Approach, LimitOptions};
        use cas_engine::Budget;

        let rest = line.strip_prefix("limit").unwrap_or(line).trim();
        if rest.is_empty() {
            println!("Usage: limit <expr> [, <var> [, <direction> [, safe]]]");
            println!("Examples:");
            println!("  limit x^2                      → infinity (default: x → +∞)");
            println!("  limit (x^2+1)/(2*x^2-3), x     → 1/2");
            println!("  limit x^3/x^2, x, -infinity    → -infinity");
            println!("  limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)");
            return;
        }

        // Parse: expr [, var [, direction [, mode]]]
        // Split by comma, respecting parentheses
        let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();

        let expr_str = parts.first().unwrap_or(&"");
        let var_str = parts.get(1).copied().unwrap_or("x");
        let dir_str = parts.get(2).copied().unwrap_or("infinity");
        let mode_str = parts.get(3).copied().unwrap_or("off");

        // Parse expression
        let expr = match cas_parser::parse(expr_str, &mut self.engine.simplifier.context) {
            Ok(e) => e,
            Err(e) => {
                println!("Parse error: {:?}", e);
                return;
            }
        };

        // Get variable
        let var = self.engine.simplifier.context.var(var_str);

        // Parse direction
        let approach = if dir_str.contains("-infinity") || dir_str.contains("-inf") {
            Approach::NegInfinity
        } else {
            Approach::PosInfinity
        };

        // Parse presimplify mode
        let presimplify = if mode_str.eq_ignore_ascii_case("safe") {
            cas_engine::limits::PreSimplifyMode::Safe
        } else {
            cas_engine::limits::PreSimplifyMode::Off
        };

        // Compute limit
        let mut budget = Budget::new();
        let opts = LimitOptions {
            presimplify,
            ..Default::default()
        };

        match limit(
            &mut self.engine.simplifier.context,
            expr,
            var,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => {
                let result_disp = cas_ast::DisplayExpr {
                    context: &self.engine.simplifier.context,
                    id: result.expr,
                };

                let dir_disp = match approach {
                    Approach::PosInfinity => "+∞",
                    Approach::NegInfinity => "-∞",
                };

                println!("lim_{{{}→{}}} = {}", var_str, dir_disp, result_disp);

                if let Some(warning) = result.warning {
                    println!("Warning: {}", warning);
                }
            }
            Err(e) => {
                println!("Error computing limit: {}", e);
            }
        }
    }
}

// Helper to split string by delimiter, ignoring delimiters inside parentheses
fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut balance = 0;
    let mut split_idx = None;

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            balance += 1;
        } else if c == '(' {
            balance -= 1;
        } else if c == delimiter && balance == 0 {
            split_idx = Some(i);
            break;
        }
    }

    if let Some(idx) = split_idx {
        Some((&s[..idx], &s[idx + 1..]))
    } else {
        None
    }
}

/// Split string by commas, respecting parentheses nesting.
/// Returns a Vec of the split parts.
fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => balance += 1,
            ')' | ']' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    // Add the last part
    if start < s.len() {
        parts.push(&s[start..]);
    }
    parts
}

fn display_solution_set(ctx: &cas_ast::Context, set: &cas_ast::SolutionSet) -> String {
    match set {
        cas_ast::SolutionSet::Empty => "Empty Set".to_string(),
        cas_ast::SolutionSet::AllReals => "All Real Numbers".to_string(),
        cas_ast::SolutionSet::Discrete(exprs) => {
            let s: Vec<String> = exprs
                .iter()
                .map(|e| {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: *e
                        }
                    )
                })
                .collect();
            format!("{{ {} }}", s.join(", "))
        }
        cas_ast::SolutionSet::Continuous(interval) => display_interval(ctx, interval),
        cas_ast::SolutionSet::Union(intervals) => {
            let s: Vec<String> = intervals.iter().map(|i| display_interval(ctx, i)).collect();
            s.join(" U ")
        }
        cas_ast::SolutionSet::Residual(expr) => {
            // Display residual expression (unsolved)
            format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
        cas_ast::SolutionSet::Conditional(cases) => {
            // V2.0 Phase 2C: Pretty-print conditional solutions
            // V2.1: Use "otherwise:" without "if" prefix for natural reading
            let case_strs: Vec<String> = cases
                .iter()
                .map(|case| {
                    let sol_str = display_solution_set(ctx, &case.then.solutions);
                    if case.when.is_otherwise() {
                        format!("  otherwise: {}", sol_str)
                    } else {
                        let cond_str = case.when.display_with_context(ctx);
                        format!("  if {}: {}", cond_str, sol_str)
                    }
                })
                .collect();
            format!("Conditional:\n{}", case_strs.join("\n"))
        }
    }
}

fn display_interval(ctx: &cas_ast::Context, interval: &cas_ast::Interval) -> String {
    let min_bracket = match interval.min_type {
        cas_ast::BoundType::Open => "(",
        cas_ast::BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        cas_ast::BoundType::Open => ")",
        cas_ast::BoundType::Closed => "]",
    };

    // Simple display without trying to simplify
    // The intervals should already have simplified bounds
    format!(
        "{}{}, {}{}",
        min_bracket,
        DisplayExpr {
            context: ctx,
            id: interval.min
        },
        DisplayExpr {
            context: ctx,
            id: interval.max
        },
        max_bracket
    )
}
