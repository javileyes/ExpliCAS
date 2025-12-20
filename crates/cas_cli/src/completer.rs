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
                "diff".to_string(),
                "gcd".to_string(),
                "lcm".to_string(),
                "mod".to_string(),
                "factors".to_string(),
                "fact".to_string(),
                "choose".to_string(),
                "perm".to_string(),
                "simplify".to_string(),
                "rationalize".to_string(),
                "config".to_string(),
                "steps".to_string(),
                "explain".to_string(),
                "det".to_string(),
                "transpose".to_string(),
                "trace".to_string(),
                "help".to_string(),
                "profile".to_string(),
                "health".to_string(),
                "set".to_string(),
                "visualize".to_string(),
                "timeline".to_string(),
                "telescope".to_string(),
                // Session environment commands
                "let".to_string(),
                "vars".to_string(),
                "clear".to_string(),
                "reset".to_string(),
                "mode".to_string(),
                "context".to_string(),
                "complex".to_string(),
                "autoexpand".to_string(),
                "history".to_string(),
                "list".to_string(),
                "show".to_string(),
                "del".to_string(),
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
                "gcd".to_string(),
                "lcm".to_string(),
                "mod".to_string(),
                "prime_factors".to_string(),
                "sum".to_string(),
                "product".to_string(),
                // Polynomial functions
                "poly_gcd".to_string(),
                "poly_gcd_exact".to_string(),
                "pgcd".to_string(),
                "pgcdx".to_string(),
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

        // Check for "config" context
        if line.starts_with("config ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            // Case 1: "config <TAB>" or "config li<TAB>"
            // If ends with space, we are starting a new word (subcommand). parts=["config"]
            // If not ends with space, we are completing subcommand. parts=["config", "li"]

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let subcommands = vec!["list", "enable", "disable", "save", "restore"];
                // If new word, word is empty. If completing, word is partial subcommand.
                for sub in subcommands {
                    if sub.starts_with(word) {
                        matches.push(Pair {
                            display: sub.to_string(),
                            replacement: sub.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }

            // Case 2: "config enable <TAB>" or "config enable dis<TAB>"
            // parts=["config", "enable"] (ends with space) -> show rules
            // parts=["config", "enable", "dis"] (!ends with space) -> show rules matching "dis"

            if parts.len() >= 2 && (parts[1] == "enable" || parts[1] == "disable") {
                if (parts.len() == 2 && ends_with_space) || (parts.len() == 3 && !ends_with_space) {
                    let rules = vec![
                        "distribute",
                        "distribute_constants",
                        "expand_binomials",
                        "factor_difference_squares",
                        "root_denesting",
                        "trig_double_angle",
                        "trig_angle_sum",
                        "log_split_exponents",
                        "rationalize_denominator",
                        "canonicalize_trig_square",
                        "auto_factor",
                    ];

                    for rule in rules {
                        if rule.starts_with(word) {
                            matches.push(Pair {
                                display: rule.to_string(),
                                replacement: rule.to_string(),
                            });
                        }
                    }
                    return Ok((start, matches));
                }
            }
        }

        // Check for "mode" context
        if line.starts_with("mode ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            // Case: "mode <TAB>" or "mode st<TAB>"
            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let modes = vec!["strict", "principal"];
                for m in modes {
                    if m.starts_with(word) {
                        matches.push(Pair {
                            display: m.to_string(),
                            replacement: m.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "context" context
        if line.starts_with("context ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let contexts = vec!["auto", "standard", "solve", "integrate"];
                for ctx in contexts {
                    if ctx.starts_with(word) {
                        matches.push(Pair {
                            display: ctx.to_string(),
                            replacement: ctx.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "complex" context
        if line.starts_with("complex ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let modes = vec!["auto", "on", "off"];
                for m in modes {
                    if m.starts_with(word) {
                        matches.push(Pair {
                            display: m.to_string(),
                            replacement: m.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "autoexpand" context
        if line.starts_with("autoexpand ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let modes = vec!["on", "off"];
                for m in modes {
                    if m.starts_with(word) {
                        matches.push(Pair {
                            display: m.to_string(),
                            replacement: m.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "steps" context
        if line.starts_with("steps ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                // Full list matching handler: on|off|normal|verbose|succinct|none
                let modes = vec!["on", "off", "normal", "verbose", "succinct", "none"];
                for m in modes {
                    if m.starts_with(word) {
                        matches.push(Pair {
                            display: m.to_string(),
                            replacement: m.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "profile" context
        if line.starts_with("profile ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let subcommands = vec!["enable", "disable", "clear"];
                for sub in subcommands {
                    if sub.starts_with(word) {
                        matches.push(Pair {
                            display: sub.to_string(),
                            replacement: sub.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }
        }

        // Check for "help" context
        if line.starts_with("help ") {
            // Suggest commands to get help on
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    matches.push(Pair {
                        display: cmd.clone(),
                        replacement: cmd.clone(),
                    });
                }
            }
            return Ok((start, matches));
        }

        // Check for "health" context
        if line.starts_with("health ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            // health <subcommand>
            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let subcommands = vec!["on", "off", "reset", "status"];
                for sub in subcommands {
                    if sub.starts_with(word) {
                        matches.push(Pair {
                            display: sub.to_string(),
                            replacement: sub.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }

            // health status <options>
            if parts.len() >= 2 && parts[1] == "status" {
                // health status <TAB> or health status --<TAB>
                if (parts.len() == 2 && ends_with_space) || (parts.len() == 3 && !ends_with_space) {
                    let options = vec!["--list", "--category", "-c"];
                    for opt in options {
                        if opt.starts_with(word) {
                            matches.push(Pair {
                                display: opt.to_string(),
                                replacement: opt.to_string(),
                            });
                        }
                    }
                    return Ok((start, matches));
                }

                // health status --category <TAB>
                if parts.len() >= 3 && (parts[2] == "--category" || parts[2] == "-c") {
                    if (parts.len() == 3 && ends_with_space)
                        || (parts.len() == 4 && !ends_with_space)
                    {
                        let categories = vec![
                            "transform",
                            "expansion",
                            "fractions",
                            "rationalization",
                            "mixed",
                            "baseline",
                            "roots",
                            "powers",
                            "stress",
                            "all",
                        ];
                        for cat in categories {
                            if cat.starts_with(word) {
                                matches.push(Pair {
                                    display: cat.to_string(),
                                    replacement: cat.to_string(),
                                });
                            }
                        }
                        return Ok((start, matches));
                    }
                }
            }
        }

        // Check for "set" context
        if line.starts_with("set ") {
            let parts: Vec<&str> = line[..pos].split_whitespace().collect();
            let ends_with_space = line[..pos].ends_with(' ');

            // First argument: option name
            if (parts.len() == 1 && ends_with_space) || (parts.len() == 2 && !ends_with_space) {
                let options = vec!["explain", "transform", "rationalize", "max-rewrites"];
                for opt in options {
                    if opt.starts_with(word) {
                        matches.push(Pair {
                            display: opt.to_string(),
                            replacement: opt.to_string(),
                        });
                    }
                }
                return Ok((start, matches));
            }

            // Second argument: value based on option
            if parts.len() >= 2 {
                if (parts.len() == 2 && ends_with_space) || (parts.len() == 3 && !ends_with_space) {
                    let values = match parts[1] {
                        "explain" | "transform" => vec!["on", "off"],
                        "rationalize" => vec!["off", "0", "1", "1.5"],
                        _ => vec![],
                    };
                    for val in values {
                        if val.starts_with(word) {
                            matches.push(Pair {
                                display: val.to_string(),
                                replacement: val.to_string(),
                            });
                        }
                    }
                    return Ok((start, matches));
                }
            }
        }

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
        if c.is_whitespace()
            || c == '('
            || c == ','
            || c == '+'
            || c == '-'
            || c == '*'
            || c == '/'
            || c == '='
        {
            break;
        }
        start = i;
    }
    (start, &line[start..pos])
}
