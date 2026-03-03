#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Snapshot of current REPL `set` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCommandState {
    pub transform: bool,
    pub rationalize: cas_solver::AutoRationalizeLevel,
    pub heuristic_poly: cas_solver::HeuristicPoly,
    pub autoexpand_binomials: cas_solver::AutoExpandBinomials,
    pub steps_mode: cas_solver::StepsMode,
    pub display_mode: SetDisplayMode,
    pub max_rewrites: usize,
    pub debug_mode: bool,
}

/// Raw parsed `set` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetCommandInput<'a> {
    ShowAll,
    ShowOption(&'a str),
    SetOption { option: &'a str, value: &'a str },
}

/// Normalized mutation plan for applying `set` changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetCommandPlan {
    pub set_transform: Option<bool>,
    pub set_rationalize: Option<cas_solver::AutoRationalizeLevel>,
    pub set_heuristic_poly: Option<cas_solver::HeuristicPoly>,
    pub set_autoexpand_binomials: Option<cas_solver::AutoExpandBinomials>,
    pub set_steps_mode: Option<cas_solver::StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
    pub set_max_rewrites: Option<usize>,
    pub set_debug_mode: Option<bool>,
    pub message: String,
}

impl SetCommandPlan {
    pub(crate) fn with_message(message: impl Into<String>) -> Self {
        Self {
            set_transform: None,
            set_rationalize: None,
            set_heuristic_poly: None,
            set_autoexpand_binomials: None,
            set_steps_mode: None,
            set_display_mode: None,
            set_max_rewrites: None,
            set_debug_mode: None,
            message: message.into(),
        }
    }
}

/// Side effects produced while applying a `SetCommandPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SetCommandApplyEffects {
    pub set_steps_mode: Option<cas_solver::StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
}

/// Evaluated `set` command result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetCommandResult {
    ShowHelp { message: String },
    ShowValue { message: String },
    Apply { plan: SetCommandPlan },
    Invalid { message: String },
}
