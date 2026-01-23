use super::*;

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
            debug_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            state: cas_engine::SessionState::new(),
        };
        repl.sync_config_to_simplifier();
        repl
    }

    /// Print a ReplReply to stdout/stderr.
    /// This is the single point where ReplCore output becomes visible.
    pub fn print_reply(&self, reply: ReplReply) {
        for msg in reply {
            match msg {
                ReplMsg::Output(s) => println!("{s}"),
                ReplMsg::Info(s) => println!("{s}"),
                ReplMsg::Warn(s) => println!("⚠ {s}"),
                ReplMsg::Error(s) => eprintln!("✖ {s}"),
                ReplMsg::Steps(s) => println!("{s}"),
                ReplMsg::Debug(s) => println!("{s}"),
            }
        }
    }

    /// Simplify expression using current pipeline options
    #[allow(dead_code)]
    pub(crate) fn do_simplify(
        &mut self,
        expr: cas_ast::ExprId,
    ) -> (cas_ast::ExprId, Vec<cas_engine::Step>) {
        // Use state.options.to_simplify_options() to get correct expand_policy, context_mode, etc.
        // (self.simplify_options is legacy and doesn't sync expand_policy)
        let mut opts = self.state.options.to_simplify_options();
        opts.collect_steps = self.engine.simplifier.collect_steps();
        // V2.15.8: Copy autoexpand_binomials from simplify_options (set by 'set autoexpand_binomials on')
        opts.autoexpand_binomials = self.simplify_options.autoexpand_binomials;
        opts.heuristic_poly = self.simplify_options.heuristic_poly;

        // Note: Tool dispatcher for collect/expand_log is in Engine::eval (cas_engine/src/eval.rs)
        // This function is dead code but kept for internal use; no dispatcher needed here.

        // Enable health metrics and clear previous run if debug or health mode is on
        if self.debug_mode || self.health_enabled {
            self.engine.simplifier.profiler.enable_health();
            self.engine.simplifier.profiler.clear_run();
        }

        let (result, steps, stats) = self.engine.simplifier.simplify_with_stats(expr, opts);

        // Store health report for the `health` command
        // Always store if health_enabled; for debug-only use threshold
        if self.health_enabled || (self.debug_mode && stats.total_rewrites >= 5) {
            self.last_health_report = Some(self.engine.simplifier.profiler.health_report());
        }

        // Show debug output if enabled
        if self.debug_mode {
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
    pub(crate) fn print_pipeline_stats(&self, stats: &cas_engine::PipelineStats) {
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

    pub(crate) fn sync_config_to_simplifier(&mut self) {
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
    pub(crate) fn build_prompt(&self) -> String {
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
    pub(crate) fn preprocess_function_syntax(&self, line: &str) -> String {
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
}
