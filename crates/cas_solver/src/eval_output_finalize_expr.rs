use cas_ast::{Context, ExprId};
use cas_formatter::ParseStyleSignals;

use crate::eval_output_finalize::{build_eval_output, EvalOutputResultPayload, EvalOutputWire};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

fn build_expr_result_payload(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    approx_hint: bool,
    style_signals: &ParseStyleSignals,
) -> EvalOutputResultPayload {
    let (result_str, truncated, char_count) = crate::eval_output_stats::format_limited_output_expr(
        ctx,
        result_expr,
        max_chars,
        style_signals,
    );
    let stats = crate::eval_output_stats::expr_output_stats(ctx, result_expr);
    let hash = if truncated {
        Some(crate::eval_output_stats::expr_output_hash(ctx, result_expr))
    } else {
        None
    };
    let result_approx = if approx_hint {
        compute_result_approx_hint(ctx, result_expr, &result_str)
    } else {
        None
    };
    // «Calcular bignum» affordance: same opt-in flag, same size gates as
    // bignum itself — the button can never promise a materialization the
    // meta-function would refuse.
    let bignum_available = (approx_hint
        && cas_math::big_materialize::bignum_would_materialize(ctx, result_expr))
    .then_some(true);

    // LaTeX is for typesetting, and typesetting has a practical size ceiling:
    // MathJax locks the page up on six-figure token counts (a bignum() result
    // carries up to 2M digits). Past the cap the wire publishes plain text
    // only — same contract as a truncated result.
    const RESULT_LATEX_MAX_CHARS: usize = 50_000;
    let result_latex = if !truncated && char_count <= RESULT_LATEX_MAX_CHARS {
        Some(crate::eval_output_latex_style::render_expr_latex_for_eval(
            ctx,
            result_expr,
            style_signals,
            crate::eval_output_latex_style::EvalLatexRenderIntent::Result,
        ))
    } else {
        None
    };

    EvalOutputResultPayload {
        result: result_str,
        result_truncated: truncated,
        result_chars: char_count,
        result_latex,
        result_approx,
        bignum_available,
        stats,
        hash,
    }
}

/// The `≈` hint (`result_approx`): the numeric reading of an EXACT result,
/// rendered exactly as `approx()` would (12 significant digits, scientific
/// notation for large magnitudes). Presentation only — computed at the
/// output boundary, opt-in per request, and never consulted by any rule.
///
/// Declines (no hint) when the result already IS a numeric presentation
/// (contains a `decimal(...)` node), is not a closed numeric value (free
/// variables), or reads identically to the exact render (`1001 ≈ 1001` is
/// noise). Beyond f64's range the exact sci lane takes over, so a `bignum`
/// integer of thousands of digits still gets its compact `m×10^k` reading.
fn compute_result_approx_hint(
    ctx: &Context,
    result_expr: ExprId,
    result_str: &str,
) -> Option<String> {
    use cas_math::decimal_display::DECIMAL_DISPLAY_SIG_DIGITS;

    if cas_math::numeric_presentation::contains_decimal_node(ctx, result_expr) {
        return None;
    }

    // ASCII on the wire (`*10^`, negative exponents parenthesized): the hint
    // is a COPYABLE value and must re-parse; `×` does not. Frontends restyle
    // the glyph for display.
    //
    // Lane order mirrors approx(): the exact sci lane OWNS everything beyond
    // f64 (|exp10| > 308) — the f64 walker silently flattens that territory
    // (`1/5^123456789` → 0.0 by underflow, which would publish a lying
    // `≈ 0`). The f64 lane then serves its native range, NORMAL values only,
    // and the ungated sci retry picks up the ±308 borderline f64 overflows
    // on (`2^1024`).
    let sci_rendering = |sci: cas_math::sci_approx::SciApprox| {
        let mantissa = cas_formatter::decimal::format_rational_decimal(
            &sci.mantissa,
            DECIMAL_DISPLAY_SIG_DIGITS,
        );
        if sci.exp10.sign() == num_bigint::Sign::Minus {
            format!("{mantissa}*10^({})", sci.exp10)
        } else {
            format!("{mantissa}*10^{}", sci.exp10)
        }
    };
    let sci = cas_math::sci_approx::try_sci_approx_expr(ctx, result_expr);
    let rendered = match &sci {
        Some(value) if value.exp10.magnitude() > &num_bigint::BigUint::from(308u32) => {
            sci_rendering(value.clone())
        }
        _ => {
            let f64_rendering =
                cas_math::rootsum_numeric::numeric_eval_with_rootsum(ctx, result_expr)
                    .filter(|value| value.is_normal())
                    .and_then(cas_math::decimal_display::approx_display_rational)
                    .map(|rational| {
                        cas_formatter::decimal::format_rational_decimal(
                            &rational,
                            DECIMAL_DISPLAY_SIG_DIGITS,
                        )
                    });
            match f64_rendering {
                Some(rendering) => rendering,
                None => sci_rendering(sci?),
            }
        }
    };

    if rendered == result_str {
        return None;
    }
    Some(rendered)
}

pub(crate) fn finalize_expr_like_eval_output(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    let payload = build_expr_result_payload(
        ctx,
        result_expr,
        max_chars,
        shared.approx_hint,
        &shared.style_signals,
    );
    let steps_count = shared.primary_steps_count();

    build_eval_output(payload, steps_count, shared)
}
