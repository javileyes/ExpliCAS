use cas_math::limit_types::PreSimplifyMode;

/// String-level approach spec (F7, Fase 3): the parse layer has no `Context`,
/// so a finite point travels as SOURCE TEXT and the evaluator parses it into
/// its fresh context — the same unification that lets the REPL `limit` command
/// accept finite points instead of erroring "not supported yet".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandApproachSpec<'a> {
    PosInfinity,
    NegInfinity,
    Finite(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: LimitCommandApproachSpec<'a>,
    pub presimplify: PreSimplifyMode,
}
