use cas_ast::Context;

pub(crate) struct EvalOutputFinalizeContext<'a> {
    pub(crate) result: &'a crate::EvalResult,
    pub(crate) ctx: &'a Context,
    pub(crate) max_chars: usize,
}
