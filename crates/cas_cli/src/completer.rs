use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper};


pub struct CasHelper {
    commands: Vec<String>,
    functions: Vec<String>,
}

impl CasHelper {
    pub fn new() -> Self {
        Self {
            commands: vec![
                "subst".to_string(),
                "expand".to_string(),
                "factor".to_string(),
                "collect".to_string(),
                "equiv".to_string(),
                "solve".to_string(),
                "steps on".to_string(),
                "steps off".to_string(),
                "help".to_string(),
                "quit".to_string(),
                "exit".to_string(),
            ],
            functions: vec![
                "sin".to_string(),
                "cos".to_string(),
                "tan".to_string(),
                "ln".to_string(),
                "sqrt".to_string(),
                "abs".to_string(),
            ],
        }
    }
}

impl Completer for CasHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let (start, word) = extract_word(line, pos);
        let mut matches = Vec::new();

        // Check commands
        for cmd in &self.commands {
            if cmd.starts_with(word) {
                matches.push(Pair {
                    display: cmd.clone(),
                    replacement: cmd.clone(),
                });
            }
        }

        // Check functions
        for func in &self.functions {
            if func.starts_with(word) {
                matches.push(Pair {
                    display: func.clone(),
                    replacement: func.clone(),
                });
            }
        }

        Ok((start, matches))
    }
}

impl Hinter for CasHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for CasHelper {}

impl Validator for CasHelper {}

impl Helper for CasHelper {}

fn extract_word(line: &str, pos: usize) -> (usize, &str) {
    let line = &line[..pos];
    if line.is_empty() {
        return (0, "");
    }

    let mut start = pos;
    for (i, c) in line.char_indices().rev() {
        if c.is_whitespace() || c == '(' || c == ',' || c == '+' || c == '-' || c == '*' || c == '/' || c == '=' {
            break;
        }
        start = i;
    }
    (start, &line[start..pos])
}
