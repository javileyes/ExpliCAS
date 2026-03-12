//! Rendered CLI command output before writing to stdout/stderr.

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr_lines: Vec<String>,
}

impl CommandOutput {
    pub fn from_stdout(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr_lines: Vec::new(),
        }
    }

    pub fn push_stderr_line(&mut self, line: impl Into<String>) {
        self.stderr_lines.push(line.into());
    }
}
