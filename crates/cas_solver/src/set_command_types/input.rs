#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Raw parsed `set` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetCommandInput<'a> {
    ShowAll,
    ShowOption(&'a str),
    SetOption { option: &'a str, value: &'a str },
}
