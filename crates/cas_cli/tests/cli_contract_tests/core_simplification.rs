use super::*;

#[test]
fn test_eval_common_scale_zero_collapse_requires_exact_zero() {
    // A "collapse to 0" shortcut mistook `1/(x²−1) − 1/(x−1)` (= −x/(x²−1)) for a common-scale
    // cancellation and folded it to 0 — a wrong CONSTANT that then poisoned `solve` into a false
    // "All real numbers". A root shortcut may now only collapse to 0 when the expression exactly
    // vanishes at a generic rational point.
    for (input, expected) in [
        ("1/(x^2-1) - 1/(x-1)", "-x / (x^2 - 1)"),
        ("1/(x-1) - 1/(x^2-1)", "x / (x^2 - 1)"),
        ("1/(2^2-1) - 1/(2-1)", "-2/3"),
        ("solve(1/(x^2-1)=1/(x-1), x)", "{ 0 }"),
        ("solve(1/(x^2-1)-1/(x-1)=0, x)", "{ 0 }"),
        ("solve((x+2)/(x^2-1)=(x+2)/(x-1), x)", "{ -2, 0 }"),
        // Genuine zero differences must STILL collapse (the guard only vetoes non-zero witnesses).
        ("2*x/(x-1) - 2*x/(x-1)", "0"),
        ("(x-1)*(x+1) - (x^2-1)", "0"),
        ("csc(x)^2 - cot(x)^2", "1"),
        ("(a+b)^2 - a^2 - 2*a*b - b^2", "0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}
#[test]
fn test_eval_numeric_quotient_plain_matches_steps() {
    // A top-level pure-real `Number/Number` quotient must evaluate IDENTICALLY whether or not
    // steps are requested. A RealOnly "complex noop" root shortcut (`is_real_domain_complex_noop_root`)
    // accepted any bare `Number` as a Gaussian component, so a real `Number/Number` matched and was
    // returned UNEVALUATED in the plain (no-step-listener) path while `--steps` ran the full pipeline.
    // The result then depended on whether steps were asked for — a consistency/soundness defect. Worst
    // case: `1/0` reported `"1 / 0"` with `ok:true` (a division by zero accepted as a valid value)
    // in plain mode but `undefined` with `--steps`. The shortcut now requires an actual imaginary unit
    // `i`, so real quotients fold through the pipeline in both modes.
    for (input, expected) in [
        // Division by zero is undefined — never a valid finite value, in either mode.
        ("1/0", "undefined"),
        ("0/0", "undefined"),
        ("2/0", "undefined"),
        ("100/0", "undefined"),
        // Exact integer quotients fold.
        ("6/3", "2"),
        ("8/4", "2"),
        ("144/12", "12"),
        ("5/1", "5"),
        ("0/7", "0"),
        // Reducible/irreducible rationals fold to lowest terms.
        ("10/4", "5/2"),
        ("9/6", "3/2"),
        ("7/2", "7/2"),
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");

        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");

        assert_eq!(
            plain_wire["result"].as_str(),
            Some(expected),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}
#[test]
fn test_eval_multi_factor_cancellation_fully_reduces() {
    // `(2·x·y)/(5·x·y)` shares TWO common factors. The plain-mode one-factor shortcut cancelled only
    // `y`, returning the partially-reduced `2·x / (5·x)` and diverging from `--steps` (which cancels
    // all common factors to `2/5`). When a residual common factor remains the shortcut now declines,
    // so the full pipeline reduces it completely.
    for (input, expected) in [
        ("(2*x*y)/(5*x*y)", "2/5"),
        ("(x*y*z)/(u*y*z)", "x / u"),
        ("(6*x*y)/(4*x*y)", "3/2"),
        ("(a*b*c)/(d*b*c)", "a / d"),
        // Single common factor is unaffected (still cancels in the shortcut).
        ("(x*y)/(u*y)", "x / u"),
        ("(a*b)/b", "a"),
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");
        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");
        assert_eq!(
            plain_wire["result"].as_str(),
            Some(expected),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}
