use super::*;

fn solver_rule_config_from_cli(config: &CasConfig) -> cas_solver::SimplifierRuleConfig {
    cas_solver::SimplifierRuleConfig {
        distribute: config.distribute,
        expand_binomials: config.expand_binomials,
        factor_difference_squares: config.factor_difference_squares,
        root_denesting: config.root_denesting,
        trig_double_angle: config.trig_double_angle,
        trig_angle_sum: config.trig_angle_sum,
        log_split_exponents: config.log_split_exponents,
        rationalize_denominator: config.rationalize_denominator,
        canonicalize_trig_square: config.canonicalize_trig_square,
        auto_factor: config.auto_factor,
    }
}

fn solver_toggle_config_from_cli(config: &CasConfig) -> cas_solver::SimplifierToggleConfig {
    cas_solver::SimplifierToggleConfig {
        distribute: config.distribute,
        expand_binomials: config.expand_binomials,
        distribute_constants: config.distribute_constants,
        factor_difference_squares: config.factor_difference_squares,
        root_denesting: config.root_denesting,
        trig_double_angle: config.trig_double_angle,
        trig_angle_sum: config.trig_angle_sum,
        log_split_exponents: config.log_split_exponents,
        rationalize_denominator: config.rationalize_denominator,
        canonicalize_trig_square: config.canonicalize_trig_square,
        auto_factor: config.auto_factor,
    }
}

impl Repl {
    pub(crate) fn config_as_solver_toggle(&self) -> cas_solver::SimplifierToggleConfig {
        solver_toggle_config_from_cli(&self.config)
    }

    pub(crate) fn set_config_from_solver_toggle(
        &mut self,
        toggles: cas_solver::SimplifierToggleConfig,
    ) {
        self.config.distribute = toggles.distribute;
        self.config.expand_binomials = toggles.expand_binomials;
        self.config.distribute_constants = toggles.distribute_constants;
        self.config.factor_difference_squares = toggles.factor_difference_squares;
        self.config.root_denesting = toggles.root_denesting;
        self.config.trig_double_angle = toggles.trig_double_angle;
        self.config.trig_angle_sum = toggles.trig_angle_sum;
        self.config.log_split_exponents = toggles.log_split_exponents;
        self.config.rationalize_denominator = toggles.rationalize_denominator;
        self.config.canonicalize_trig_square = toggles.canonicalize_trig_square;
        self.config.auto_factor = toggles.auto_factor;
    }

    pub fn new() -> Self {
        let config = CasConfig::load();
        let simplifier =
            cas_solver::build_simplifier_with_rule_config(solver_rule_config_from_cli(&config));

        let mut repl = Self {
            core: ReplCore::with_simplifier(simplifier),
            verbosity: Verbosity::Normal,
            config,
        };
        repl.sync_config_to_simplifier();
        repl
    }

    pub(crate) fn rebuild_engine_simplifier_from_config(&mut self) {
        self.core.engine.simplifier = cas_solver::build_simplifier_with_rule_config(
            solver_rule_config_from_cli(&self.config),
        );
    }

    /// Print a ReplReply to stdout/stderr.
    /// This is the single point where ReplCore output becomes visible.
    /// WriteFile actions are executed here (file I/O), with results printed.
    pub fn print_reply(&self, reply: ReplReply) {
        for msg in reply {
            match msg {
                ReplMsg::Output(s) => println!("{s}"),
                ReplMsg::Info(s) => println!("{s}"),
                ReplMsg::Warn(s) => println!("⚠ {s}"),
                ReplMsg::Error(s) => eprintln!("✖ {s}"),
                ReplMsg::Steps(s) => println!("{s}"),
                ReplMsg::Debug(s) => println!("{s}"),
                ReplMsg::WriteFile { path, contents } => {
                    // Execute file write (I/O happens in shell, not core)
                    // Create parent directories if needed
                    if let Some(parent) = path.parent() {
                        if !parent.exists() {
                            if let Err(e) = std::fs::create_dir_all(parent) {
                                eprintln!(
                                    "✖ Failed to create directory {}: {}",
                                    parent.display(),
                                    e
                                );
                                continue;
                            }
                        }
                    }
                    match std::fs::write(&path, &contents) {
                        Ok(()) => println!("Wrote: {}", path.display()),
                        Err(e) => eprintln!("✖ Failed to write {}: {}", path.display(), e),
                    }
                }
                ReplMsg::OpenFile { path } => {
                    #[cfg(target_os = "macos")]
                    {
                        match std::process::Command::new("open").arg(&path).spawn() {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("✖ Failed to open {}: {}", path.display(), e);
                            }
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        let _ = path;
                    }
                }
            }
        }
    }

    pub(crate) fn sync_config_to_simplifier(&mut self) {
        let toggles = self.config_as_solver_toggle();
        cas_solver::apply_simplifier_toggle_config(&mut self.core.engine.simplifier, toggles);
    }

    /// Build the REPL prompt with mode indicators.
    /// Only shows indicators for non-default modes to keep prompt clean.
    pub(crate) fn build_prompt(&self) -> String {
        cas_solver::build_prompt_from_eval_options(self.core.state.options())
    }

    /// Generate startup banner messages (no I/O here)
    pub(crate) fn startup_messages(&self) -> ReplReply {
        reply_output(
            "Rust CAS Step-by-Step Demo\n\
             Step-by-step output enabled (Normal).\n\
             Enter an expression (e.g., '2 * 3 + 0'):",
        )
    }

    /// Generate goodbye message (no I/O here)
    pub(crate) fn goodbye_message(&self) -> ReplReply {
        reply_output("Goodbye!")
    }

    pub fn run(&mut self) -> rustyline::Result<()> {
        // Print startup banner
        self.print_reply(self.startup_messages());

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
                        self.print_reply(self.goodbye_message());
                        break;
                    }

                    // Split using solver-owned parsing rules. This keeps command
                    // syntax details (like solve_system semicolons) out of CLI.
                    for statement in cas_solver::split_repl_statements(line) {
                        self.handle_command(statement);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    self.print_reply(reply_output("CTRL-C"));
                    break;
                }
                Err(ReadlineError::Eof) => {
                    self.print_reply(reply_output("CTRL-D"));
                    break;
                }
                Err(err) => {
                    self.print_reply(reply_output(format!("Error: {:?}", err)));
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
        cas_solver::preprocess_repl_function_syntax(line)
    }
}
