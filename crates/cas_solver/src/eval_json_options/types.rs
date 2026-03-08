/// Stringly option axes accepted by eval-json command.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalJsonOptionAxes<'a> {
    pub context: &'a str,
    pub branch: &'a str,
    pub complex: &'a str,
    pub autoexpand: &'a str,
    pub steps: &'a str,
    pub domain: &'a str,
    pub value_domain: &'a str,
    pub inv_trig: &'a str,
    pub complex_branch: &'a str,
    pub assume_scope: &'a str,
}
