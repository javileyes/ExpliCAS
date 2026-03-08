/// Parsed `help` command input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpCommandInput<'a> {
    General,
    Topic(&'a str),
}

/// Parse `help` command line.
pub fn parse_help_command_input(line: &str) -> HelpCommandInput<'_> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    match parts.get(1) {
        Some(topic) => HelpCommandInput::Topic(topic),
        None => HelpCommandInput::General,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_help_command_input, HelpCommandInput};

    #[test]
    fn parse_help_command_input_without_topic_is_general() {
        assert_eq!(parse_help_command_input("help"), HelpCommandInput::General);
    }

    #[test]
    fn parse_help_command_input_with_topic_reads_topic() {
        assert_eq!(
            parse_help_command_input("help simplify"),
            HelpCommandInput::Topic("simplify")
        );
    }
}
