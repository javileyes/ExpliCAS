use cas_math::limit_types::{Approach, PreSimplifyMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: Approach,
    pub presimplify: PreSimplifyMode,
}
