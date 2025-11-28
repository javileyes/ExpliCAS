mod repl;
mod completer;

use repl::Repl;

fn main() -> rustyline::Result<()> {
    let mut repl = Repl::new();
    repl.run()
}
