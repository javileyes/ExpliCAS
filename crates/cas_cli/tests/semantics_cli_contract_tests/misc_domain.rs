use super::*;

#[test]
fn domain_strict_with_semantics() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["domain_mode"], "strict");
    assert_eq!(wire["domain"]["mode"], "strict");
}
#[test]
fn assume_scope_default_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "real",
        "assume_scope default should be 'real'"
    );
}
#[test]
fn assume_scope_wildcard_flag_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--assume-scope",
        "wildcard",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "wildcard",
        "--assume-scope wildcard should be reflected in wire output"
    );
}
#[test]
fn assume_scope_flag_does_not_change_result() {
    // Infrastructure-only: changing assume_scope should NOT change result
    // (behavior changes come in PR-SCOPE-3)
    let (output1, _) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "generic"]);
    let (output2, _) = run_cli(&[
        "eval",
        "x/x",
        "--format",
        "json",
        "--domain",
        "generic",
        "--assume-scope",
        "wildcard",
    ]);

    let wire1 = parse_wire(&output1);
    let wire2 = parse_wire(&output2);

    assert_eq!(
        wire1["result"], wire2["result"],
        "assume_scope flag should not change result (infra only)"
    );
}
#[test]
fn equiv_honors_value_domain_axis() {
    // SOUNDNESS (auditoría 2026-07-30, ficha S5-002): `equiv((e^z)^w, e^(z*w))`
    // publicaba `true` bajo --value-domain complex — la identidad es real-only
    // (con rama principal, (e^(4i))^(1/2) = −e^(2i) ≠ e^(2i)) y el comparador
    // corría sin ejes: are_equivalent simplificaba en RealOnly y su veredicto
    // ganaba el `||`; el fallback expand() construía opciones frescas RealOnly.
    // Ahora el eje llega por el sticky value domain (armado desde las
    // semánticas de la petición) y el comparador declina en complejo.
    for expr in ["equiv((e^z)^w, e^(z*w))", "equiv((2^z)^w, 2^(z*w))"] {
        let (output, _code) = run_cli(&[
            "eval",
            expr,
            "--value-domain",
            "complex",
            "--complex",
            "on",
            "--format",
            "json",
            "--no-pretty",
        ]);
        let wire = parse_wire(&output);
        assert_ne!(
            wire["result"], "true",
            "{expr} must NOT confirm under complex: {output}"
        );
        // En real la identidad SÍ vale (e^z > 0) y debe seguir confirmando.
        let (real_output, _code) = run_cli(&["eval", expr, "--format", "json", "--no-pretty"]);
        let real_wire = parse_wire(&real_output);
        assert_eq!(
            real_wire["result"], "true",
            "{expr} must stay true in real mode: {real_output}"
        );
    }
}
