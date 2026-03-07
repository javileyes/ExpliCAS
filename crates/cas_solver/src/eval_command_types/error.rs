/// Errors while evaluating REPL `eval` command.
#[derive(Debug, Clone)]
pub enum EvalCommandError {
    Parse(cas_parser::ParseError),
    Eval(String),
}
