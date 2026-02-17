//! envelope-json subcommand handler.
//!
//! Evaluates a single expression and returns the stable OutputEnvelope V1 format
//! designed for Android/FFI integration.

use std::time::Instant;

use anyhow::Result;
use clap::Args;

use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::{Engine, EvalAction, EvalOutput, EvalRequest, EvalResult};

use crate::json_types::{
    AssumptionDto, ConditionDto, EngineInfo, ExprDto, OutputEnvelope, RequestInfo, RequestOptions,
    ResultDto, TransparencyDto,
};

/// Arguments for envelope-json subcommand
#[derive(Args, Debug)]
pub struct EnvelopeJsonArgs {
    /// Expression to evaluate
    pub expr: String,

    /// Domain mode: strict, generic, assume
    #[arg(long, default_value = "generic")]
    pub domain: String,

    /// Value domain: real, complex
    #[arg(long, default_value = "real")]
    pub value_domain: String,
}

/// Run the envelope-json command
pub fn run(args: EnvelopeJsonArgs) {
    match run_inner(&args) {
        Ok(output) => {
            print_pretty_json(&output);
        }
        Err(e) => {
            let error_envelope = OutputEnvelope {
                schema_version: 1,
                engine: EngineInfo::default(),
                request: RequestInfo {
                    kind: "eval".to_string(),
                    input: args.expr.clone(),
                    solve_var: None,
                    options: RequestOptions::default(),
                },
                result: ResultDto::Error {
                    message: e.to_string(),
                },
                transparency: TransparencyDto::default(),
                steps: vec![],
            };
            print_pretty_json(&error_envelope);
        }
    }
}

fn run_inner(args: &EnvelopeJsonArgs) -> Result<OutputEnvelope> {
    let _start = Instant::now();

    // Create engine and session state
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Configure options from args
    state.options.shared.semantics.domain_mode = match args.domain.as_str() {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    };

    state.options.shared.semantics.value_domain = match args.value_domain.as_str() {
        "complex" => cas_solver::ValueDomain::ComplexEnabled,
        _ => cas_solver::ValueDomain::RealOnly,
    };

    // Parse expression
    let parsed = parse(&args.expr, &mut engine.simplifier.context)
        .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;

    // Build eval request
    let req = EvalRequest {
        raw_input: args.expr.clone(),
        parsed,
        kind: cas_session::EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output = engine.eval(&mut state, req)?;

    // Extract result expression
    let result_expr = match &output.result {
        EvalResult::Expr(e) => *e,
        EvalResult::Set(v) if !v.is_empty() => v[0],
        EvalResult::Bool(b) => {
            return Ok(OutputEnvelope {
                schema_version: 1,
                engine: EngineInfo::default(),
                request: build_request_info(args),
                result: ResultDto::Eval {
                    value: ExprDto {
                        display: b.to_string(),
                        canonical: b.to_string(),
                    },
                },
                transparency: build_transparency(&output, &engine.simplifier.context),
                steps: vec![],
            });
        }
        _ => {
            return Err(anyhow::anyhow!("No result expression"));
        }
    };

    // Format result
    let display = DisplayExpr {
        context: &engine.simplifier.context,
        id: result_expr,
    }
    .to_string();

    Ok(OutputEnvelope {
        schema_version: 1,
        engine: EngineInfo::default(),
        request: build_request_info(args),
        result: ResultDto::Eval {
            value: ExprDto {
                display: display.clone(),
                canonical: display,
            },
        },
        transparency: build_transparency(&output, &engine.simplifier.context),
        steps: vec![], // TODO: populate if steps enabled
    })
}

fn build_request_info(args: &EnvelopeJsonArgs) -> RequestInfo {
    RequestInfo {
        kind: "eval".to_string(),
        input: args.expr.clone(),
        solve_var: None,
        options: RequestOptions {
            domain_mode: args.domain.clone(),
            value_domain: args.value_domain.clone(),
            hints: true,
            explain: false,
        },
    }
}

fn build_transparency(output: &EvalOutput, ctx: &cas_ast::Context) -> TransparencyDto {
    use cas_solver::ImplicitCondition;

    let required_conditions: Vec<ConditionDto> = output
        .required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
            ConditionDto {
                kind: kind.to_string(),
                display: cond.display(ctx),
                expr_display: expr_str.clone(),
                expr_canonical: expr_str,
            }
        })
        .collect();

    let assumptions_used: Vec<AssumptionDto> = output
        .domain_warnings
        .iter()
        .map(|w| AssumptionDto {
            kind: "NonZero".to_string(), // Default for domain warnings
            display: w.message.clone(),
            expr_canonical: String::new(),
            rule: w.rule_name.clone(),
        })
        .collect();

    TransparencyDto {
        required_conditions,
        assumptions_used,
        blocked_hints: vec![],
    }
}

fn print_pretty_json<T: serde::Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{}", s),
        Err(e) => {
            eprintln!("JSON serialization error: {}", e);
            match serde_json::to_string(value) {
                Ok(s) => println!("{}", s),
                Err(_) => println!(
                    "{{\"schema_version\":1,\"ok\":false,\"error\":\"JSON_SERIALIZATION_FAILED\"}}"
                ),
            }
        }
    }
}
