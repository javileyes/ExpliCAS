mod repl;
mod completer;
mod config;

use repl::Repl;

fn main() -> rustyline::Result<()> {
    let mut repl = Repl::new();
    repl.run()
}
