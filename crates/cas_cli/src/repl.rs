use cas_engine::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_engine::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_engine::rules::polynomial::{AnnihilationRule, CombineLikeTermsRule};
use cas_engine::Simplifier;

use cas_ast::{Context, DisplayExpr, DisplayExprWithHints, Expr, ExprId};
use cas_engine::display_context::build_display_context_with_result;
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
        };
        repl.sync_config_to_simplifier();
        repl
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
        println!("  config <subcmd>         Manage configuration (list, enable, disable...)");
        println!("  profile [cmd]           Rule profiler (enable/disable/clear)");
        println!("  help [cmd]              Show this help message or details for a command");
        println!("  quit / exit             Exit the REPL");
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
                    let (simplified, steps) = self.simplifier.simplify(expr);
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
        let filtered_steps = cas_engine::strategies::filter_non_productive_steps(
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

        match cas_parser::parse_statement(eq_str, &mut self.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
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
            Ok(cas_parser::Statement::Expression(_)) => {
                println!("Error: Expected an equation, got an expression.");
            }
            Err(e) => println!("Error parsing equation: {}", e),
        }
    }

    fn handle_eval(&mut self, line: &str) {
        match cas_parser::parse(line, &mut self.simplifier.context) {
            Ok(expr) => {
                println!(
                    "Parsed: {}",
                    DisplayExpr {
                        context: &self.simplifier.context,
                        id: expr
                    }
                );
                let (simplified, steps) = self.simplifier.simplify(expr);

                // Build display hints for preserving root notation (only when showing steps)
                let display_hints = if self.verbosity != Verbosity::None {
                    build_display_context_with_result(
                        &self.simplifier.context,
                        expr,
                        &steps,
                        Some(simplified),
                    )
                } else {
                    cas_ast::DisplayContext::new()
                };

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
                                    DisplayExprWithHints {
                                        context: &self.simplifier.context,
                                        id: step.before,
                                        hints: &display_hints
                                    }
                                ));
                                let after_disp = clean_display_string(&format!(
                                    "{}",
                                    DisplayExprWithHints {
                                        context: &self.simplifier.context,
                                        id: step.after,
                                        hints: &display_hints
                                    }
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
                                        DisplayExprWithHints {
                                            context: &self.simplifier.context,
                                            id: current_root,
                                            hints: &display_hints
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
                                        // Didactic: Show sub-steps BEFORE Local transformation
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
                                                    // Show title for substeps section
                                                    println!(
                                                        "   [Suma de fracciones en exponentes]"
                                                    );
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

                                        // Show Before: global expression before this step
                                        if let Some(global_before) = step.global_before {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprWithHints {
                                                        context: &self.simplifier.context,
                                                        id: global_before,
                                                        hints: &display_hints
                                                    }
                                                ))
                                            );
                                        } else {
                                            println!(
                                                "   Before: {}",
                                                clean_display_string(&format!(
                                                    "{}",
                                                    DisplayExprWithHints {
                                                        context: &self.simplifier.context,
                                                        id: current_root,
                                                        hints: &display_hints
                                                    }
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
                                            DisplayExprWithHints {
                                                context: &self.simplifier.context,
                                                id: rule_before_id,
                                                hints: &display_hints
                                            }
                                        ));
                                        let after_disp = clean_display_string(&format!(
                                            "{}",
                                            DisplayExprWithHints {
                                                context: &self.simplifier.context,
                                                id: rule_after_id,
                                                hints: &display_hints
                                            }
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
                                                DisplayExprWithHints {
                                                    context: &self.simplifier.context,
                                                    id: current_root,
                                                    hints: &display_hints
                                                }
                                            ))
                                        );
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
                    DisplayExprWithHints {
                        context: &self.simplifier.context,
                        id: simplified,
                        hints: &display_hints
                    }
                );
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
        // Swap context
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);

        // Ensure we have the aggressive rules we want (DistributeRule is in default)
        // Also add DistributeConstantRule just in case (though DistributeRule covers it)

        // Set steps mode
        temp_simplifier.collect_steps = self.verbosity != Verbosity::None;

        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
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
                                        let after_disp = if let Some(s) = &step.after_str {
                                            s.clone()
                                        } else {
                                            format!(
                                                "{}",
                                                DisplayExpr {
                                                    context: &temp_simplifier.context,
                                                    id: step.after
                                                }
                                            )
                                        };
                                        println!(
                                            "   Rule: {} -> {}",
                                            clean_display_string(&format!(
                                                "{}",
                                                DisplayExpr {
                                                    context: &temp_simplifier.context,
                                                    id: step.before
                                                }
                                            )),
                                            clean_display_string(&after_disp)
                                        );
                                    }

                                    current_root = reconstruct_global_expr(
                                        &mut temp_simplifier.context,
                                        current_root,
                                        &step.path,
                                        step.after,
                                    );
                                    println!(
                                        "   After: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExpr {
                                                context: &temp_simplifier.context,
                                                id: current_root
                                            }
                                        ))
                                    );
                                }
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
                println!(
                    "Result: {}",
                    DisplayExpr {
                        context: &temp_simplifier.context,
                        id: simplified
                    }
                );
            }
            Err(e) => println!("Error: {}", e),
        }

        // Swap context back
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
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
