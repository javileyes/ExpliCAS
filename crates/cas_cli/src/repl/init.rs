pub(super) use super::*;

mod build;
mod messages;
mod run_loop;

fn ensure_parent_dir(path: &std::path::Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    if parent.exists() {
        return Ok(());
    }

    std::fs::create_dir_all(parent).map_err(|error| {
        format!(
            "✖ Failed to create directory {}: {}",
            parent.display(),
            error
        )
    })
}

fn write_file_action(path: &std::path::Path, contents: &str) -> Result<(), String> {
    ensure_parent_dir(path)?;
    std::fs::write(path, contents)
        .map_err(|error| format!("✖ Failed to write {}: {}", path.display(), error))
}

impl Repl {
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
                    match write_file_action(&path, &contents) {
                        Ok(()) => println!("Wrote: {}", path.display()),
                        Err(message) => eprintln!("{message}"),
                    }
                }
                ReplMsg::OpenFile { path } => {
                    #[cfg(target_os = "macos")]
                    {
                        match std::process::Command::new("open").arg(&path).spawn() {
                            Ok(_) => {}
                            Err(error) => {
                                eprintln!("✖ Failed to open {}: {}", path.display(), error);
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
}
