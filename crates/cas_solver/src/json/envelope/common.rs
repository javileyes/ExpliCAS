use cas_api_models::{EnvelopeEvalOptions, RequestInfo, RequestOptions};

pub fn display_expr(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> String {
    cas_formatter::DisplayExpr { context: ctx, id }.to_string()
}

pub fn build_request_info(expr: &str, opts: &EnvelopeEvalOptions) -> RequestInfo {
    RequestInfo::eval(
        expr,
        RequestOptions {
            domain_mode: opts.domain.clone(),
            value_domain: opts.value_domain.clone(),
            hints: true,
            explain: false,
        },
    )
}
