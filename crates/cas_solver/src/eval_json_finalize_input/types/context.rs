use cas_ast::Context;

pub(crate) struct EvalJsonFinalizeContext<'a> {
    pub(crate) result: &'a crate::EvalResult,
    pub(crate) ctx: &'a Context,
    pub(crate) max_chars: usize,
}
