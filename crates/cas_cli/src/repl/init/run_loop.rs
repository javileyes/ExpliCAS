use super::*;

fn history_path() -> std::path::PathBuf {
    dirs::home_dir()
        .map(|path| path.join(".cas_history"))
        .unwrap_or_else(|| std::path::PathBuf::from(".cas_history"))
}

fn build_editor(
) -> rustyline::Result<rustyline::Editor<CasHelper, rustyline::history::DefaultHistory>> {
    let helper = CasHelper::new();
    let config = rustyline::Config::builder()
        .max_history_size(100)?
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut editor =
        rustyline::Editor::<CasHelper, rustyline::history::DefaultHistory>::with_config(config)?;
    editor.set_helper(Some(helper));
    Ok(editor)
}

impl Repl {
    pub fn run(&mut self) -> rustyline::Result<()> {
        self.print_reply(self.startup_messages());

        let mut rl = build_editor()?;
        let history_path = history_path();
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

                    for statement in cas_solver::session_api::repl::split_repl_statements(line) {
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
                Err(error) => {
                    self.print_reply(reply_output(format!("Error: {:?}", error)));
                    break;
                }
            }
        }

        let _ = rl.save_history(&history_path);
        Ok(())
    }
}
