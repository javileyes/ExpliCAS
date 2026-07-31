use super::*;

/// The visible rule name must describe the manoeuvre THIS step performed.
///
/// `Distribute Division Into Sum` covers two different rewrites and the static
/// table could only name one: on `taylor(sin(x), x, 0, 5)` step 3 factors the
/// common coefficient 6 out of `(6x^5 - 120x^3)/720` — it distributes nothing —
/// while the name read "Repartir el denominador entre los sumandos". The rule
/// already says which one it did in its own description.
#[test]
fn visible_rule_name_distinguishes_the_two_distribute_division_manoeuvres() {
    for (lang, expected, forbidden) in [
        (
            "es",
            "Sacar el factor común del numerador",
            "Repartir el denominador",
        ),
        (
            "en",
            "Factor the common coefficient out of the numerator",
            "Split the denominator",
        ),
    ] {
        let (wire, _) = cli_eval_json_with_stderr_args(
            "taylor(sin(x), x, 0, 5)",
            &["--steps", "on", "--lang", lang],
        );
        let rules: Vec<String> = wire["steps"]
            .as_array()
            .expect("steps with --steps on")
            .iter()
            .filter_map(|s| s["rule"].as_str().map(str::to_string))
            .collect();
        assert!(rules.iter().any(|r| r == expected), "[{lang}] {rules:?}");
        assert!(
            !rules.iter().any(|r| r.contains(forbidden)),
            "[{lang}] the step distributes nothing: {rules:?}"
        );
    }
}
