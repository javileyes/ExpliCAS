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
use rustyline::config::Configurer;

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

    result
}

pub struct Repl {
    simplifier: Simplifier,
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
    /// Session environment for variable bindings
    env: cas_engine::env::Environment,
    /// Session store for expression history with #id references
    session: cas_engine::SessionStore,
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
/// This is more robust than path-based reconstruction because it finds by identity, not position.
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
        // Matrix: substitute in data elements
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
        // Leaf nodes: no substitution possible
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
            if step.importance() < ImportanceLevel::Medium {
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

        if config.distribute_constants {}

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
            simplifier,
            verbosity: Verbosity::Normal,
            config,
            simplify_options: cas_engine::SimplifyOptions::default(),
            explain_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            env: cas_engine::env::Environment::default(),
            session: cas_engine::SessionStore::new(),
        };
        repl.sync_config_to_simplifier();
        repl
    }

    /// Simplify expression using current pipeline options
    fn do_simplify(&mut self, expr: cas_ast::ExprId) -> (cas_ast::ExprId, Vec<cas_engine::Step>) {
        let mut opts = self.simplify_options.clone();
        opts.collect_steps = self.simplifier.collect_steps;

        // Enable health metrics and clear previous run if explain or health mode is on
        if self.explain_mode || self.health_enabled {
            self.simplifier.profiler.enable_health();
            self.simplifier.profiler.clear_run();
        }

        let (result, steps, stats) = self.simplifier.simplify_with_stats(expr, opts);

        // Store health report for the `health` command
        // Always store if health_enabled; for explain-only use threshold
        if self.health_enabled || (self.explain_mode && stats.total_rewrites >= 5) {
            self.last_health_report = Some(self.simplifier.profiler.health_report());
        }

        // Show explain output if enabled
        if self.explain_mode {
            self.print_pipeline_stats(&stats);

            // Policy A+ hint: when simplify makes minimal changes to a Mul expression
            if stats.total_rewrites <= 1 {
                if matches!(
                    self.simplifier.context.get(result),
                    cas_ast::Expr::Mul(_, _)
                ) {
                    println!(
                        "Note: simplify preserves factored products. Use expand(...) to expand."
                    );
                }
            }

            // Show health report if significant activity (>= 5 rewrites)
            if stats.total_rewrites >= 5 {
                println!();
                if let Some(ref report) = self.last_health_report {
                    print!("{}", report);
                }
            }
        }

        self.last_stats = Some(stats);
        (result, steps)
    }

    /// Print pipeline statistics for diagnostics
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
                self.simplifier.enable_rule(name);
            } else {
                self.simplifier.disable_rule(name);
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
            self.simplifier.enable_rule("Automatic Factorization");
            self.simplifier.enable_rule("Conservative Expand");
            self.simplifier.disable_rule("Expand Polynomial");
            self.simplifier.disable_rule("Binomial Expansion");
        } else {
            self.simplifier.disable_rule("Automatic Factorization");
            self.simplifier.disable_rule("Conservative Expand");
            self.simplifier.enable_rule("Expand Polynomial");
            // Re-enable Binomial Expansion if config says so
            if config.expand_binomials {
                self.simplifier.enable_rule("Binomial Expansion");
            }
        }
    }

    pub fn run(&mut self) -> rustyline::Result<()> {
        println!("Rust CAS Step-by-Step Demo");
        println!("Step-by-step output enabled (Normal).");
        println!("Enter an expression (e.g., '2 * 3 + 0'):");

        let helper = CasHelper::new();
        let mut rl = rustyline::Editor::<CasHelper, rustyline::history::DefaultHistory>::new()?;
        rl.set_helper(Some(helper));
        rl.set_completion_type(rustyline::CompletionType::List);

        // Load history if file exists (optional, skipping for simplicity or can add later)

        loop {
            let readline = rl.readline("> ");
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

                    self.handle_command(line);
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
        if line.starts_with("let ") {
            self.handle_let_command(&line[4..]);
            return;
        }

        // "<name> := <expr>" - alternative assignment syntax
        if let Some(idx) = line.find(":=") {
            let name = line[..idx].trim();
            let expr_str = line[idx + 2..].trim();
            if !name.is_empty() && !expr_str.is_empty() {
                self.handle_assignment(name, expr_str);
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

        // "history" or "list" - show session history
        if line == "history" || line == "list" {
            self.handle_history_command();
            return;
        }

        // "show #id" - show a specific session entry
        if line.starts_with("show ") {
            self.handle_show_command(&line[5..]);
            return;
        }

        // "del #id [#id...]" - delete session entries
        if line.starts_with("del ") {
            self.handle_del_command(&line[4..]);
            return;
        }

        // ========== END SESSION ENVIRONMENT COMMANDS ==========

        // Check for "steps" command
        if line.starts_with("steps ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[1] {
                    "on" | "normal" => {
                        self.verbosity = Verbosity::Normal;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Normal).");
                    }
                    "off" | "none" => {
                        self.verbosity = Verbosity::None;
                        self.simplifier.collect_steps = false;
                        println!("Step-by-step output disabled.");
                    }
                    "verbose" => {
                        self.verbosity = Verbosity::Verbose;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Verbose).");
                    }
                    "succinct" => {
                        self.verbosity = Verbosity::Succinct;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Succinct - compact display).");
                    }
                    _ => println!("Usage: steps <on|off|normal|verbose|succinct|none>"),
                }
            } else {
                println!("Usage: steps <on|off|normal|verbose|succinct|none>");
            }
            return;
        }

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

        // Check for "profile" commands
        if line.starts_with("profile") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 1 {
                // Just "profile" - show report
                println!("{}", self.simplifier.profiler.report());
            } else {
                match parts[1] {
                    "enable" => {
                        self.simplifier.profiler.enable();
                        println!("Profiler enabled.");
                    }
                    "disable" => {
                        self.simplifier.profiler.disable();
                        println!("Profiler disabled.");
                    }
                    "clear" => {
                        self.simplifier.profiler.clear();
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
                        self.simplifier.profiler.clear_run();
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
                            &mut self.simplifier,
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
                println!("Command: subst <expr>, <var>=<val>");
                println!("Description: Substitutes a variable with a value (or another expression) and simplifies.");
                println!("Example: subst x^2 + x, x=3 -> 12");
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

        println!("Equation Solving:");
        println!("  solve <eq>, <var>       Solve equation for variable");
        println!("  equiv <e1>, <e2>        Check if two expressions are equivalent");
        println!("  subst <expr>, <var>=<val> Substitute a variable and simplify");
        println!();

        println!("Calculus:");
        println!("  diff <expr>, <var>      Compute symbolic derivative");
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

        println!("System:");
        println!(
            "  set <option> <value>    Pipeline settings (transform, rationalize, max-rewrites)"
        );
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
        println!("  reset                   Clear all session state");
        println!("  history / list          Show session history (#ids)");
        println!("  show #<id>              Display a session entry");
        println!("  del #<id> ...           Delete session entries");
        println!();

        println!("Type 'help <command>' for more details on a specific command.");
    }

    fn handle_equiv(&mut self, line: &str) {
        let rest = line[6..].trim();
        if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
            // We need to parse both, but parse takes &mut Context.
            // We can't borrow self.simplifier.context mutably twice.
            // So we parse one, then the other.

            let e1_res = cas_parser::parse(expr1_str.trim(), &mut self.simplifier.context);
            match e1_res {
                Ok(e1) => {
                    let e2_res = cas_parser::parse(expr2_str.trim(), &mut self.simplifier.context);
                    match e2_res {
                        Ok(e2) => {
                            // Resolve session references (e.g. #1) and variables for E1
                            let e1 = match cas_engine::resolve_session_refs(
                                &mut self.simplifier.context,
                                e1,
                                &self.session,
                            ) {
                                Ok(r) => cas_engine::env::substitute(
                                    &mut self.simplifier.context,
                                    &self.env,
                                    r,
                                ),
                                Err(_) => e1,
                            };

                            // Resolve session references (e.g. #2) and variables for E2
                            let e2 = match cas_engine::resolve_session_refs(
                                &mut self.simplifier.context,
                                e2,
                                &self.session,
                            ) {
                                Ok(r) => cas_engine::env::substitute(
                                    &mut self.simplifier.context,
                                    &self.env,
                                    r,
                                ),
                                Err(_) => e2,
                            };

                            let are_eq = self.simplifier.are_equivalent(e1, e2);
                            if are_eq {
                                println!("True");
                            } else {
                                println!("False");
                            }
                        }
                        Err(e) => println!("Error parsing second expression: {}", e),
                    }
                }
                Err(e) => println!("Error parsing first expression: {}", e),
            }
        } else {
            println!("Usage: equiv <expr1>, <expr2>");
        }
    }

    fn handle_subst(&mut self, line: &str) {
        // Format: subst <expr>, <var>=<val>
        // Example: subst x+1, x=2
        let rest = line[6..].trim();

        // Try splitting by comma first (preferred)
        let (expr_str, assign_str) = if let Some((e, a)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), a.trim())
        } else if let Some((e, a)) = rsplit_ignoring_parens(rest, ' ') {
            // Fallback to last space
            (e.trim(), a.trim())
        } else {
            println!("Usage: subst <expression>, <var>=<value>");
            return;
        };

        if let Some((var, val_str)) = assign_str.split_once('=') {
            let var = var.trim();
            let val_str = val_str.trim();

            // Parse expr first
            match cas_parser::parse(expr_str, &mut self.simplifier.context) {
                Ok(expr) => {
                    // Parse val
                    match cas_parser::parse(val_str, &mut self.simplifier.context) {
                        Ok(val_expr) => {
                            if self.verbosity != Verbosity::None {
                                println!("Substituting {} = {} into {}", var, val_str, expr_str);
                            }
                            // Substitute
                            let target_var = self.simplifier.context.var(var);
                            let subbed = cas_engine::solver::strategies::substitute_expr(
                                &mut self.simplifier.context,
                                expr,
                                target_var,
                                val_expr,
                            );

                            let (result, steps) = self.simplifier.simplify(subbed);
                            if self.verbosity != Verbosity::None {
                                if self.verbosity != Verbosity::Succinct {
                                    println!("Steps:");
                                }
                                let mut current_root = subbed;
                                let mut step_count = 0;
                                for step in steps.iter() {
                                    if should_show_step(step, self.verbosity) {
                                        step_count += 1;

                                        if self.verbosity == Verbosity::Succinct {
                                            // Low mode: just global state
                                            current_root = reconstruct_global_expr(
                                                &mut self.simplifier.context,
                                                current_root,
                                                &step.path,
                                                step.after,
                                            );
                                            println!(
                                                "-> {}",
                                                DisplayExpr {
                                                    context: &self.simplifier.context,
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
                                                            DisplayExpr {
                                                                context: &self.simplifier.context,
                                                                id: global_before,
                                                            }
                                                        ))
                                                    );
                                                } else {
                                                    println!(
                                                        "   Before: {}",
                                                        clean_display_string(&format!(
                                                            "{}",
                                                            DisplayExpr {
                                                                context: &self.simplifier.context,
                                                                id: current_root,
                                                            }
                                                        ))
                                                    );
                                                }

                                                // Show Rule: local transformation
                                                let after_disp = if let Some(s) = &step.after_str {
                                                    s.clone()
                                                } else {
                                                    format!(
                                                        "{}",
                                                        DisplayExpr {
                                                            context: &self.simplifier.context,
                                                            id: step.after
                                                        }
                                                    )
                                                };
                                                println!(
                                                    "   Rule: {} -> {}",
                                                    clean_display_string(&format!(
                                                        "{}",
                                                        DisplayExpr {
                                                            context: &self.simplifier.context,
                                                            id: step.before
                                                        }
                                                    )),
                                                    clean_display_string(&after_disp)
                                                );
                                            }

                                            // Use precomputed global_after if available, fall back to ExprId-based substitution
                                            if let Some(global_after) = step.global_after {
                                                current_root = global_after;
                                            } else {
                                                // Use identity-based substitution instead of path-based reconstruction
                                                current_root = substitute_expr_by_id(
                                                    &mut self.simplifier.context,
                                                    current_root,
                                                    step.before,
                                                    step.after,
                                                );
                                            }

                                            // Show After: global expression after this step
                                            if self.verbosity == Verbosity::Verbose
                                                || self.verbosity == Verbosity::Normal
                                            {
                                                println!(
                                                    "   After: {}",
                                                    clean_display_string(&format!(
                                                        "{}",
                                                        DisplayExpr {
                                                            context: &self.simplifier.context,
                                                            id: current_root,
                                                        }
                                                    ))
                                                );
                                            }
                                        }
                                    } else {
                                        if let Some(global_after) = step.global_after {
                                            current_root = global_after;
                                        } else {
                                            current_root = substitute_expr_by_id(
                                                &mut self.simplifier.context,
                                                current_root,
                                                step.before,
                                                step.after,
                                            );
                                        }
                                    }
                                }
                            }
                            println!(
                                "Result: {}",
                                DisplayExpr {
                                    context: &self.simplifier.context,
                                    id: result
                                }
                            );
                        }
                        Err(e) => println!("Error parsing value: {}", e),
                    }
                }
                Err(e) => println!("Error parsing expression: {}", e),
            }
            return;
        }
        println!("Usage: subst <expression>, <var>=<value>");
    }

    fn handle_timeline(&mut self, line: &str) {
        let rest = line[9..].trim();

        // Check if the user wants to use "solve" within timeline
        // e.g., "timeline solve x + 2 = 5, x"
        if rest.starts_with("solve ") {
            self.handle_timeline_solve(&rest[6..]);
            return;
        }

        // Check if the user wants to use "simplify" within timeline
        // e.g., "timeline simplify(expr)" or "timeline simplify expr"
        let (expr_str, use_aggressive) = if rest.starts_with("simplify(") && rest.ends_with(')') {
            // Extract expression from "simplify(expr)"
            (&rest[9..rest.len() - 1], true)
        } else if rest.starts_with("simplify ") {
            // Extract expression from "simplify expr"
            (&rest[9..], true)
        } else {
            // No simplify prefix, treat entire rest as expression
            (rest, false)
        };

        // Choose simplifier based on whether aggressive mode is requested
        let (steps, expr_id, simplified) = if use_aggressive {
            // Create temporary simplifier with aggressive rules (like handle_full_simplify)
            let mut temp_simplifier = Simplifier::with_default_rules();
            temp_simplifier.collect_steps = true; // Always collect steps for timeline

            // Swap context to preserve variables
            std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);

            match cas_parser::parse(expr_str.trim(), &mut temp_simplifier.context) {
                Ok(expr) => {
                    let (simplified, steps) = temp_simplifier.simplify(expr);

                    // Swap context back
                    std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);

                    (steps, expr, simplified)
                }
                Err(e) => {
                    // Swap context back even on error
                    std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
                    println!("Parse error: {}", e);
                    return;
                }
            }
        } else {
            // Use normal simplification
            match cas_parser::parse(expr_str.trim(), &mut self.simplifier.context) {
                Ok(expr) => {
                    let (simplified, steps) = self.do_simplify(expr);
                    (steps, expr, simplified)
                }
                Err(e) => {
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
            &mut self.simplifier.context,
            expr_id,
            steps.clone(),
        );

        // Convert CLI verbosity to timeline verbosity
        let timeline_verbosity = match self.verbosity {
            Verbosity::None | Verbosity::Succinct => cas_engine::timeline::VerbosityLevel::Low,
            Verbosity::Normal => cas_engine::timeline::VerbosityLevel::Normal,
            Verbosity::Verbose => cas_engine::timeline::VerbosityLevel::Verbose,
        };

        // Generate HTML timeline with ALL steps and the known simplified result
        let mut timeline = cas_engine::timeline::TimelineHtml::new_with_result(
            &mut self.simplifier.context,
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
        let rest = if line.starts_with("visualize ") {
            &line[10..]
        } else {
            &line[4..]
        }
        .trim();

        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                let mut viz = cas_engine::visualizer::AstVisualizer::new(&self.simplifier.context);
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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                // Check if it's a function call
                let expr_data = self.simplifier.context.get(expr).clone();
                if let Expr::Function(name, args) = expr_data {
                    match name.as_str() {
                        "gcd" => {
                            if args.len() == 2 {
                                // Call the explain_gcd function
                                let result = cas_engine::rules::number_theory::explain_gcd(
                                    &mut self.simplifier.context,
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
                                            context: &self.simplifier.context,
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

    /// Handle "let <name> = <expr>" command
    fn handle_let_command(&mut self, rest: &str) {
        // Parse: <name> = <expr>
        if let Some(eq_idx) = rest.find('=') {
            let name = rest[..eq_idx].trim();
            let expr_str = rest[eq_idx + 1..].trim();
            self.handle_assignment(name, expr_str);
        } else {
            println!("Usage: let <name> = <expr>");
            println!("Example: let a = 1 + sqrt(2)");
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    fn handle_assignment(&mut self, name: &str, expr_str: &str) {
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
        match cas_parser::parse(expr_str, &mut self.simplifier.context) {
            Ok(rhs_expr) => {
                // Temporarily remove this binding to prevent self-reference in substitute
                let old_binding = self.env.get(name).map(|id| id);
                self.env.unset(name);

                // Substitute using current environment (allows a = b+1 where b is defined)
                let rhs_substituted =
                    cas_engine::env::substitute(&mut self.simplifier.context, &self.env, rhs_expr);

                // Store the binding
                self.env.set(name.to_string(), rhs_substituted);

                // Display confirmation
                let display = cas_ast::DisplayExpr {
                    context: &self.simplifier.context,
                    id: rhs_substituted,
                };
                println!("{} = {}", name, display);

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
        let bindings = self.env.list();
        if bindings.is_empty() {
            println!("No variables defined.");
        } else {
            println!("Variables:");
            for (name, expr_id) in bindings {
                let display = cas_ast::DisplayExpr {
                    context: &self.simplifier.context,
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
            let count = self.env.len();
            self.env.clear_all();
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
                if self.env.unset(name) {
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
        // Clear environment
        self.env.clear_all();
        // Clear session history explicitly to avoid referencing invalid ExprIds from destroyed context
        // This fixes the OOB panic when calling 'list' after 'reset'
        self.session.clear();

        // Reset simplifier with new context
        self.simplifier = Simplifier::with_default_rules();

        // Re-register custom rules (same as in new())
        self.simplifier
            .add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        self.simplifier.add_rule(Box::new(EvaluateTrigRule));
        self.simplifier.add_rule(Box::new(PythagoreanIdentityRule));
        if self.config.trig_angle_sum {
            self.simplifier.add_rule(Box::new(AngleIdentityRule));
        }
        self.simplifier.add_rule(Box::new(TanToSinCosRule));
        if self.config.trig_double_angle {
            self.simplifier.add_rule(Box::new(DoubleAngleRule));
        }
        if self.config.canonicalize_trig_square {
            self.simplifier.add_rule(Box::new(
                cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule,
            ));
        }
        self.simplifier.add_rule(Box::new(EvaluateLogRule));
        self.simplifier.add_rule(Box::new(ExponentialLogRule));
        self.simplifier.add_rule(Box::new(SimplifyFractionRule));
        self.simplifier.add_rule(Box::new(ExpandRule));
        self.simplifier
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

    /// Handle "history" or "list" command - show session history
    fn handle_history_command(&self) {
        let entries = self.session.list();
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
                            context: &self.simplifier.context,
                            id: *expr_id
                        }
                    )
                }
                cas_engine::EntryKind::Eq { lhs, rhs } => {
                    format!(
                        "{} = {}",
                        cas_ast::DisplayExpr {
                            context: &self.simplifier.context,
                            id: *lhs
                        },
                        cas_ast::DisplayExpr {
                            context: &self.simplifier.context,
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
                if let Some(entry) = self.session.get(id) {
                    println!("Entry #{}:", id);
                    println!("  Type:       {}", entry.type_str());
                    println!("  Raw:        {}", entry.raw_text);

                    match &entry.kind {
                        cas_engine::EntryKind::Expr(expr_id) => {
                            // Show parsed expression
                            println!(
                                "  Parsed:     {}",
                                DisplayExpr {
                                    context: &self.simplifier.context,
                                    id: *expr_id
                                }
                            );

                            // Show resolved (after #id and env substitution)
                            let resolved = match cas_engine::resolve_session_refs(
                                &mut self.simplifier.context,
                                *expr_id,
                                &self.session,
                            ) {
                                Ok(r) => cas_engine::env::substitute(
                                    &mut self.simplifier.context,
                                    &self.env,
                                    r,
                                ),
                                Err(_) => *expr_id,
                            };
                            if resolved != *expr_id {
                                println!(
                                    "  Resolved:   {}",
                                    DisplayExpr {
                                        context: &self.simplifier.context,
                                        id: resolved
                                    }
                                );
                            }

                            // Show simplified
                            let (simplified, _) = self.simplifier.simplify(resolved);
                            if simplified != resolved {
                                println!(
                                    "  Simplified: {}",
                                    DisplayExpr {
                                        context: &self.simplifier.context,
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
                                    context: &self.simplifier.context,
                                    id: *lhs
                                }
                            );
                            println!(
                                "  RHS:        {}",
                                DisplayExpr {
                                    context: &self.simplifier.context,
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

        let before_len = self.session.len();
        self.session.remove(&ids);
        let removed = before_len - self.session.len();

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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                // Wrap in det() function call
                let det_expr = self
                    .simplifier
                    .context
                    .add(Expr::Function("det".to_string(), vec![expr]));

                // Simplify to compute determinant
                let (result, steps) = self.simplifier.simplify(det_expr);

                println!("Parsed: det({})", rest);

                // Print steps if verbosity is not None
                if self.verbosity != Verbosity::None && !steps.is_empty() {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                        if let Some(assumption) = &step.domain_assumption {
                            println!("   ⚠ Domain: {}", assumption);
                        }
                    }
                }

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.simplifier.context,
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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                // Wrap in transpose() function call
                let transpose_expr = self
                    .simplifier
                    .context
                    .add(Expr::Function("transpose".to_string(), vec![expr]));

                // Simplify to compute transpose
                let (result, steps) = self.simplifier.simplify(transpose_expr);

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
                        context: &self.simplifier.context,
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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                // Wrap in trace() function call
                let trace_expr = self
                    .simplifier
                    .context
                    .add(Expr::Function("trace".to_string(), vec![expr]));

                // Simplify to compute trace
                let (result, steps) = self.simplifier.simplify(trace_expr);

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
                        context: &self.simplifier.context,
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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", rest);
                println!();

                // Apply telescoping strategy
                let result = cas_engine::telescoping::telescope(&mut self.simplifier.context, expr);

                // Print formatted output
                println!("{}", result.format(&self.simplifier.context));
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

        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(expr) => {
                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.simplifier.context,
                        id: expr
                    }
                );

                // Use the expansion module directly (bypasses DistributeRule guards)
                let expanded = cas_engine::expand::expand(&mut self.simplifier.context, expr);

                // Simplify to clean up the result
                let (simplified, _steps) = self.simplifier.simplify(expanded);

                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &self.simplifier.context,
                        id: simplified
                    }
                );
            }
            Err(e) => println!("Parse error: {:?}", e),
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
        match cas_parser::parse(rest, &mut self.simplifier.context) {
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
                        context: &self.simplifier.context,
                        id: result
                    }
                );
                println!("Weierstrass substitution (t = tan(x/2)):");
                println!("  {} → {}", rest, result_str);

                // Try to simplify the result
                println!();
                println!("Simplifying...");
                let (simplified, _steps) = self.simplifier.simplify(result);
                let simplified_str = format!(
                    "{}",
                    DisplayExpr {
                        context: &self.simplifier.context,
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

        match self.simplifier.context.get(expr).clone() {
            Expr::Function(name, args)
                if matches!(name.as_str(), "sin" | "cos" | "tan") && args.len() == 1 =>
            {
                let arg = args[0];

                // Build t = tan(x/2) as sin(x/2)/cos(x/2)
                let two_num = self.simplifier.context.num(2);
                let half_arg = self.simplifier.context.add(Expr::Div(arg, two_num));
                let sin_half = self
                    .simplifier
                    .context
                    .add(Expr::Function("sin".to_string(), vec![half_arg]));
                let cos_half = self
                    .simplifier
                    .context
                    .add(Expr::Function("cos".to_string(), vec![half_arg]));
                let t = self.simplifier.context.add(Expr::Div(sin_half, cos_half)); // t = tan(x/2)

                // Apply appropriate transformation
                match name.as_str() {
                    "sin" => {
                        // sin(x) → 2t/(1+t²)
                        let two = self.simplifier.context.num(2);
                        let one = self.simplifier.context.num(1);
                        let t_squared = self.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self.simplifier.context.add(Expr::Add(one, t_squared));
                        self.simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "cos" => {
                        // cos(x) → (1-t²)/(1+t²)
                        let one = self.simplifier.context.num(1);
                        let two = self.simplifier.context.num(2);
                        let t_squared = self.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.simplifier.context.add(Expr::Sub(one, t_squared));
                        let denominator = self.simplifier.context.add(Expr::Add(one, t_squared));
                        self.simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    "tan" => {
                        // tan(x) → 2t/(1-t²)
                        let two = self.simplifier.context.num(2);
                        let one = self.simplifier.context.num(1);
                        let t_squared = self.simplifier.context.add(Expr::Pow(t, two));
                        let numerator = self.simplifier.context.add(Expr::Mul(two, t));
                        let denominator = self.simplifier.context.add(Expr::Sub(one, t_squared));
                        self.simplifier
                            .context
                            .add(Expr::Div(numerator, denominator))
                    }
                    _ => expr,
                }
            }
            Expr::Add(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.simplifier.context.add(Expr::Add(new_l, new_r))
            }
            Expr::Sub(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.simplifier.context.add(Expr::Sub(new_l, new_r))
            }
            Expr::Mul(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.simplifier.context.add(Expr::Mul(new_l, new_r))
            }
            Expr::Div(l, r) => {
                let new_l = self.apply_weierstrass_recursive(l);
                let new_r = self.apply_weierstrass_recursive(r);
                self.simplifier.context.add(Expr::Div(new_l, new_r))
            }
            Expr::Pow(base, exp) => {
                let new_base = self.apply_weierstrass_recursive(base);
                let new_exp = self.apply_weierstrass_recursive(exp);
                self.simplifier.context.add(Expr::Pow(new_base, new_exp))
            }
            Expr::Neg(e) => {
                let new_e = self.apply_weierstrass_recursive(e);
                self.simplifier.context.add(Expr::Neg(new_e))
            }
            Expr::Function(name, args) => {
                // Recurse into function arguments
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_weierstrass_recursive(a))
                    .collect();
                self.simplifier
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

        match cas_parser::parse_statement(eq_str, &mut self.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Call solver with step collection enabled
                self.simplifier.collect_steps = true;

                match cas_engine::solver::solve(&eq, var, &mut self.simplifier) {
                    Ok((solution_set, steps)) => {
                        if steps.is_empty() {
                            println!("No solving steps to visualize.");
                            println!(
                                "Result: {}",
                                display_solution_set(&self.simplifier.context, &solution_set)
                            );
                            return;
                        }

                        // Generate HTML timeline for solve steps
                        let mut timeline = cas_engine::timeline::SolveTimelineHtml::new(
                            &mut self.simplifier.context,
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
                                    display_solution_set(&self.simplifier.context, &solution_set)
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
        // solve <equation>, <var>
        let rest = line[6..].trim();

        // Split by comma or space to get equation and var
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
        } else {
            // No comma. Try to see if it looks like "eq var"
            // We only accept "eq var" if "eq" is a valid equation.
            // Otherwise, we assume the whole string is the equation (e.g. "ln(x) = a + b")
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                let v_trim = v.trim();
                // Check if v is a variable name (alphabetic)
                if !v_trim.is_empty() && v_trim.chars().all(char::is_alphabetic) {
                    (e.trim(), v_trim)
                } else {
                    (rest, "x")
                }
            } else {
                (rest, "x")
            }
        };

        // Check if eq_str is a session reference (e.g., "#1")
        let eq_str_trimmed = eq_str.trim().trim_start_matches('#');
        let session_eq: Option<cas_ast::Equation> = if eq_str.trim().starts_with('#') {
            if let Ok(id) = eq_str_trimmed.parse::<u64>() {
                if let Some(entry) = self.session.get(id) {
                    match &entry.kind {
                        cas_engine::EntryKind::Eq { lhs, rhs } => Some(cas_ast::Equation {
                            lhs: *lhs,
                            rhs: *rhs,
                            op: cas_ast::RelOp::Eq,
                        }),
                        cas_engine::EntryKind::Expr(_) => {
                            println!("Error: Entry #{} is an expression, not an equation.", id);
                            println!(
                                "Hint: Use 'solve <expr> = <value>, <var>' to solve an expression."
                            );
                            return;
                        }
                    }
                } else {
                    println!(
                        "Error: Entry #{} not found. Use 'history' to see available entries.",
                        id
                    );
                    return;
                }
            } else {
                None
            }
        } else {
            None
        };

        // Use session equation if found, otherwise parse the string
        let eq_result = if let Some(eq) = session_eq {
            Ok(eq)
        } else {
            match cas_parser::parse_statement(eq_str, &mut self.simplifier.context) {
                Ok(cas_parser::Statement::Equation(eq)) => Ok(eq),
                Ok(cas_parser::Statement::Expression(_)) => {
                    Err("Expected an equation, got an expression.".to_string())
                }
                Err(e) => Err(format!("{}", e)),
            }
        };

        match eq_result {
            Ok(eq) => {
                // Check if variable exists in equation
                // We should simplify the equation first to handle cases like "ln(x) + ln(x) = 2" -> "2*ln(x) = 2"
                let (sim_lhs, steps_lhs) = self.simplifier.simplify(eq.lhs);
                let (sim_rhs, steps_rhs) = self.simplifier.simplify(eq.rhs);

                if self.verbosity != Verbosity::None
                    && (!steps_lhs.is_empty() || !steps_rhs.is_empty())
                {
                    if self.verbosity != Verbosity::Succinct {
                        println!("Simplification Steps:");
                    }
                    for (i, step) in steps_lhs.iter().enumerate() {
                        if should_show_step(step, self.verbosity) {
                            if self.verbosity == Verbosity::Succinct {
                                // Low mode: just global state? No, for solve simplification we don't track global state easily here
                                // because steps_lhs are local to lhs.
                                // We can show the result of the step on LHS.
                                // But wait, solve simplification is just pre-simplification.
                                // Let's just show it if not Low.
                                // Or if Low, maybe we skip pre-simplification steps display?
                                // User said "Low mode only shows global changes".
                                // For solve, the "Global" is the equation.
                                // But here we simplify LHS and RHS separately.
                                // Let's skip detailed steps in Low mode for pre-simplification,
                                // and just show the simplified equation.
                            } else {
                                println!(
                                    "LHS {}. {}  [{}]",
                                    i + 1,
                                    step.description,
                                    step.rule_name
                                );
                                let after_disp = if let Some(s) = &step.after_str {
                                    s.clone()
                                } else {
                                    format!(
                                        "{}",
                                        DisplayExpr {
                                            context: &self.simplifier.context,
                                            id: step.after
                                        }
                                    )
                                };
                                println!(
                                    "   Rule: {} -> {}",
                                    DisplayExpr {
                                        context: &self.simplifier.context,
                                        id: step.before
                                    },
                                    after_disp
                                );
                            }
                        }
                    }
                    for (i, step) in steps_rhs.iter().enumerate() {
                        if should_show_step(step, self.verbosity) {
                            if self.verbosity != Verbosity::Succinct {
                                println!(
                                    "RHS {}. {}  [{}]",
                                    i + 1,
                                    step.description,
                                    step.rule_name
                                );
                                let after_disp = if let Some(s) = &step.after_str {
                                    s.clone()
                                } else {
                                    format!(
                                        "{}",
                                        DisplayExpr {
                                            context: &self.simplifier.context,
                                            id: step.after
                                        }
                                    )
                                };
                                println!(
                                    "   Rule: {} -> {}",
                                    DisplayExpr {
                                        context: &self.simplifier.context,
                                        id: step.before
                                    },
                                    after_disp
                                );
                            }
                        }
                    }
                    if self.verbosity != Verbosity::Succinct {
                        println!(
                            "Solving simplified equation: {} {} {}",
                            DisplayExpr {
                                context: &self.simplifier.context,
                                id: sim_lhs
                            },
                            eq.op,
                            DisplayExpr {
                                context: &self.simplifier.context,
                                id: sim_rhs
                            }
                        );
                    }
                }

                let simplified_eq = cas_ast::Equation {
                    lhs: sim_lhs,
                    rhs: sim_rhs,
                    op: eq.op.clone(),
                };

                let lhs_has = cas_engine::solver::contains_var(
                    &self.simplifier.context,
                    simplified_eq.lhs,
                    var,
                );
                let rhs_has = cas_engine::solver::contains_var(
                    &self.simplifier.context,
                    simplified_eq.rhs,
                    var,
                );

                if !lhs_has && !rhs_has {
                    // Constant equation (w.r.t var). Evaluate truthiness.
                    // Already simplified above
                    // We need to compare values.
                    // But sim_lhs and sim_rhs are ExprIds.
                    // We can use are_equivalent? No, that checks symbolic equivalence.
                    // We can use compare_values from solution_set if we expose it?
                    // Or just check if they are same ID?
                    // If simplified, they should be same ID if they are identical.
                    if sim_lhs == sim_rhs {
                        println!("True (Identity)");
                    } else {
                        println!("False (Contradiction)");
                        println!(
                            "{} != {}",
                            DisplayExpr {
                                context: &self.simplifier.context,
                                id: sim_lhs
                            },
                            DisplayExpr {
                                context: &self.simplifier.context,
                                id: sim_rhs
                            }
                        );
                    }
                } else {
                    // Pass the ORIGINAL equation to solve, so it can check for domain restrictions (singularities).
                    // If we pass simplified_eq, we lose information about e.g. (x-1) in denominator.
                    match cas_engine::solver::solve(&eq, var, &mut self.simplifier) {
                        Ok((solution_set, steps)) => {
                            if self.verbosity != Verbosity::None {
                                if self.verbosity != Verbosity::Succinct {
                                    println!("Steps:");
                                }
                                for (i, step) in steps.iter().enumerate() {
                                    // SolveStep is different from Step, so we can't use should_show_step directly.
                                    // For now, just show all steps if verbosity is not None/Low?
                                    // Or implement filtering for SolveStep too?
                                    // SolveStep has description but no rule_name in the same way?
                                    // Let's just show it.
                                    if true {
                                        // Simplify the equation for display
                                        let (sim_lhs, _) =
                                            self.simplifier.simplify(step.equation_after.lhs);
                                        let (sim_rhs, _) =
                                            self.simplifier.simplify(step.equation_after.rhs);

                                        if self.verbosity == Verbosity::Succinct {
                                            println!(
                                                "-> {} {} {}",
                                                DisplayExpr {
                                                    context: &self.simplifier.context,
                                                    id: sim_lhs
                                                },
                                                step.equation_after.op,
                                                DisplayExpr {
                                                    context: &self.simplifier.context,
                                                    id: sim_rhs
                                                }
                                            );
                                        } else {
                                            println!("{}. {}", i + 1, step.description);
                                            println!(
                                                "   -> {} {} {}",
                                                DisplayExpr {
                                                    context: &self.simplifier.context,
                                                    id: sim_lhs
                                                },
                                                step.equation_after.op,
                                                DisplayExpr {
                                                    context: &self.simplifier.context,
                                                    id: sim_rhs
                                                }
                                            );
                                        }
                                    }
                                }
                            }
                            // SolutionSet doesn't implement Display with Context.
                            // We need to manually display it.
                            println!(
                                "Result: {}",
                                display_solution_set(&self.simplifier.context, &solution_set)
                            );
                        }
                        Err(e) => println!("Error solving: {}", e),
                    }
                }
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    fn handle_eval(&mut self, line: &str) {
        // Sniff style preferences from input string BEFORE parsing
        let style_signals = ParseStyleSignals::from_input_string(line);

        // Use parse_statement to support both expressions and equations
        match cas_parser::parse_statement(line, &mut self.simplifier.context) {
            Ok(cas_parser::Statement::Expression(expr)) => {
                // Auto-store: save the parsed expression to session history
                let entry_id = self
                    .session
                    .push(cas_engine::EntryKind::Expr(expr), line.to_string());
                println!(
                    "#{}  {}",
                    entry_id,
                    DisplayExpr {
                        context: &self.simplifier.context,
                        id: expr
                    }
                );

                // Resolve session references (#id) before substitution
                let expr = match cas_engine::resolve_session_refs(
                    &mut self.simplifier.context,
                    expr,
                    &self.session,
                ) {
                    Ok(resolved) => resolved,
                    Err(e) => {
                        println!("Error resolving session reference: {}", e);
                        return;
                    }
                };

                // Substitute variables from environment
                let expr =
                    cas_engine::env::substitute(&mut self.simplifier.context, &self.env, expr);

                let (simplified, steps) = self.do_simplify(expr);

                // Create global style preferences from input signals + AST
                let style_prefs = StylePreferences::from_expression_with_signals(
                    &self.simplifier.context,
                    expr,
                    Some(&style_signals),
                );

                if self.verbosity != Verbosity::None {
                    if steps.is_empty() {
                        // Even with no engine steps, show didactic sub-steps if there are fraction sums
                        let standalone_substeps = cas_engine::didactic::get_standalone_substeps(
                            &self.simplifier.context,
                            expr,
                        );

                        if !standalone_substeps.is_empty() && self.verbosity != Verbosity::Succinct
                        {
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
                                            if let Some(second_close) =
                                                result[numer_end + 2..].find('}')
                                            {
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
                            &self.simplifier.context,
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
                                        &self.simplifier.context,
                                        step.before,
                                        &style_prefs
                                    )
                                ));
                                let after_disp = clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.simplifier.context,
                                        step.after,
                                        &style_prefs
                                    )
                                ));
                                if before_disp == after_disp {
                                    // Display no-op - still update state but skip step display
                                    if let Some(global_after) = step.global_after {
                                        current_root = global_after;
                                    }
                                    continue;
                                }

                                step_count += 1;

                                if self.verbosity == Verbosity::Succinct {
                                    // Low mode: just global state
                                    current_root = reconstruct_global_expr(
                                        &mut self.simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                    println!(
                                        "-> {}",
                                        DisplayExprStyled::new(
                                            &self.simplifier.context,
                                            current_root,
                                            &style_prefs
                                        )
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
                                        // Show Before: global expression before this step (always)
                                        if let Some(global_before) = step.global_before {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprStyled::new(
                                                        &self.simplifier.context,
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
                                                        &self.simplifier.context,
                                                        current_root,
                                                        &style_prefs
                                                    )
                                                ))
                                            );
                                        }

                                        // Didactic: Show sub-steps AFTER Before: line
                                        // Sub-steps explain hidden computations (e.g., fraction sums)
                                        // Only show on first visible step to avoid duplication
                                        if !sub_steps_shown {
                                            if let Some(enriched_step) =
                                                enriched_steps.get(step_idx)
                                            {
                                                if !enriched_step.sub_steps.is_empty() {
                                                    sub_steps_shown = true;
                                                    // Helper function for LaTeX to plain text
                                                    fn latex_to_text(s: &str) -> String {
                                                        let mut result = s.to_string();
                                                        while let Some(start) =
                                                            result.find("\\frac{")
                                                        {
                                                            let end_start = start + 6;
                                                            if let Some(first_close) =
                                                                result[end_start..].find('}')
                                                            {
                                                                let numer_end =
                                                                    end_start + first_close;
                                                                let numer =
                                                                    &result[end_start..numer_end];
                                                                if result.len() > numer_end + 1
                                                                    && result
                                                                        .chars()
                                                                        .nth(numer_end + 1)
                                                                        == Some('{')
                                                                {
                                                                    if let Some(second_close) =
                                                                        result[numer_end + 2..]
                                                                            .find('}')
                                                                    {
                                                                        let denom_end = numer_end
                                                                            + 2
                                                                            + second_close;
                                                                        let denom = &result
                                                                            [numer_end + 2
                                                                                ..denom_end];
                                                                        let replacement = format!(
                                                                            "({}/{})",
                                                                            numer, denom
                                                                        );
                                                                        result = format!(
                                                                            "{}{}{}",
                                                                            &result[..start],
                                                                            replacement,
                                                                            &result
                                                                                [denom_end + 1..]
                                                                        );
                                                                        continue;
                                                                    }
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        result.replace("\\", "")
                                                    }

                                                    // Show sub-steps (Before: already shown above for all steps)
                                                    // Show title for substeps section (detect type from description)
                                                    let has_fraction_sum =
                                                        enriched_step.sub_steps.iter().any(|s| {
                                                            s.description
                                                                .contains("common denominator")
                                                                || s.description
                                                                    .contains("Sum the fractions")
                                                        });
                                                    let has_factorization =
                                                        enriched_step.sub_steps.iter().any(|s| {
                                                            s.description
                                                                .contains("Cancel common factor")
                                                                || s.description.contains("Factor")
                                                        });

                                                    if has_fraction_sum {
                                                        println!(
                                                            "   [Suma de fracciones en exponentes]"
                                                        );
                                                    } else if has_factorization {
                                                        println!(
                                                            "   [Factorización de polinomios]"
                                                        );
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
                                                &self.simplifier.context,
                                                rule_before_id,
                                                &style_prefs
                                            )
                                        ));
                                        let after_disp = clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &self.simplifier.context,
                                                rule_after_id,
                                                &style_prefs
                                            )
                                        ));

                                        // Skip display-only no-op steps (e.g., -1*(1-√2) → -(1-√2) both display as -1-√2)
                                        if before_disp == after_disp {
                                            // Still update current_root to maintain state
                                            if let Some(global_after) = step.global_after {
                                                current_root = global_after;
                                            }
                                            continue;
                                        }

                                        // Always show Rule line - it now shows the accurate local transformation
                                        println!("   Rule: {} -> {}", before_disp, after_disp);
                                    }

                                    // Use precomputed global_after if available, fall back to reconstruction
                                    if let Some(global_after) = step.global_after {
                                        current_root = global_after;
                                    } else {
                                        current_root = reconstruct_global_expr(
                                            &mut self.simplifier.context,
                                            current_root,
                                            &step.path,
                                            step.after,
                                        );
                                    }

                                    // Show After: global expression after this step
                                    if self.verbosity == Verbosity::Normal
                                        || self.verbosity == Verbosity::Verbose
                                    {
                                        println!(
                                            "   After: {}",
                                            clean_display_string(&format!(
                                                "{}",
                                                DisplayExprStyled::new(
                                                    &self.simplifier.context,
                                                    current_root,
                                                    &style_prefs
                                                )
                                            ))
                                        );

                                        // Show domain assumption warning if present
                                        if let Some(assumption) = &step.domain_assumption {
                                            println!("   ⚠ Domain: {}", assumption);
                                        }
                                    }
                                }
                            } else {
                                if let Some(global_after) = step.global_after {
                                    current_root = global_after;
                                } else {
                                    current_root = reconstruct_global_expr(
                                        &mut self.simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                }
                            }
                        }
                    }
                }
                println!(
                    "Result: {}",
                    DisplayExprStyled::new(&self.simplifier.context, simplified, &style_prefs)
                );
            }
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Store equation in session history
                let entry_id = self.session.push(
                    cas_engine::EntryKind::Eq {
                        lhs: eq.lhs,
                        rhs: eq.rhs,
                    },
                    line.to_string(),
                );

                // Display the equation
                let lhs_disp = DisplayExpr {
                    context: &self.simplifier.context,
                    id: eq.lhs,
                };
                let rhs_disp = DisplayExpr {
                    context: &self.simplifier.context,
                    id: eq.rhs,
                };
                let op_str = match eq.op {
                    cas_ast::RelOp::Eq => "=",
                    cas_ast::RelOp::Neq => "≠",
                    cas_ast::RelOp::Lt => "<",
                    cas_ast::RelOp::Gt => ">",
                    cas_ast::RelOp::Leq => "≤",
                    cas_ast::RelOp::Geq => "≥",
                };
                println!("#{}  {} {} {}  [Eq]", entry_id, lhs_disp, op_str, rhs_disp);
            }
            Err(e) => println!("Error: {}", e),
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
        // Current implementation: Context is reset per line? No, self.simplifier.context persists.
        // If we want to support "x = 5; simplify x", we need to share context.

        // Better approach:
        // 1. Parse expression using current context.
        // 2. Create a temporary Simplifier that SHARES the context?
        //    Simplifier owns Context. We can't easily share.
        //    But we can temporarily TAKE the context, use it in a new Simplifier, and then put it back.

        let mut temp_simplifier = Simplifier::with_default_rules();
        // Swap context and profiler so temp_simplifier uses main profiler
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
        std::mem::swap(&mut self.simplifier.profiler, &mut temp_simplifier.profiler);

        // Ensure we have the aggressive rules we want (DistributeRule is in default)
        // Also add DistributeConstantRule just in case (though DistributeRule covers it)

        // Set steps mode
        temp_simplifier.collect_steps = self.verbosity != Verbosity::None;

        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let style_signals = ParseStyleSignals::from_input_string(expr_str);
                let style_prefs = StylePreferences::from_expression_with_signals(
                    &temp_simplifier.context,
                    expr,
                    Some(&style_signals),
                );

                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &temp_simplifier.context,
                        id: expr
                    }
                );
                let (simplified, steps) = temp_simplifier.simplify(expr);

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
                                        let after_disp = clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &temp_simplifier.context,
                                                rule_after_id,
                                                &style_prefs
                                            )
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

                                    // Show domain assumption warning if present
                                    if let Some(assumption) = &step.domain_assumption {
                                        println!("   ⚠ Domain: {}", assumption);
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
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
        std::mem::swap(&mut self.simplifier.profiler, &mut temp_simplifier.profiler);

        // Store health report for the `health` command (if health tracking is enabled)
        if self.health_enabled {
            self.last_health_report = Some(self.simplifier.profiler.health_report());
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

        match cas_parser::parse(rest, &mut self.simplifier.context) {
            Ok(parsed_expr) => {
                // CANONICALIZE: Rebuild tree to trigger Add auto-flatten at all levels
                // Parser creates tree incrementally, so nested Adds may not be flattened
                // normalize_core forces reconstruction ensuring canonical form
                let expr = cas_engine::canonical_forms::normalize_core(
                    &mut self.simplifier.context,
                    parsed_expr,
                );
                // STYLE SNIFFING: Detect user's preferred notation BEFORE processing
                let user_style = cas_ast::detect_root_style(&self.simplifier.context, expr);

                let disp = cas_ast::DisplayExpr {
                    context: &self.simplifier.context,
                    id: expr,
                };
                println!("Parsed: {}", disp);

                let config = RationalizeConfig::default();
                let result = rationalize_denominator(&mut self.simplifier.context, expr, &config);

                match result {
                    RationalizeResult::Success(rationalized) => {
                        // Simplify the result
                        let (simplified, _) = self.simplifier.simplify(rationalized);

                        // Use StyledExpr with detected style for consistent output
                        let result_disp = cas_ast::StyledExpr::new(
                            &self.simplifier.context,
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
