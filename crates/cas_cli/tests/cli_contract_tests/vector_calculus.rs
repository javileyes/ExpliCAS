use super::*;

#[test]
fn test_eval_potential_verb_f6() {
    // Fase 3 · F6: potencial escalar por caminos con VERIFICACIÓN exacta
    // (∂φ/∂xᵢ ≡ Fᵢ vía poly_eq) — la emisión la gatea la verificación, no la
    // construcción: un campo no conservativo jamás verifica y declina.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(eval_result("potential([2*x*y, x^2], [x,y])"), "y·x^2");
    assert_eq!(eval_result("potential([y,x],[x,y])"), "x·y");
    assert_eq!(eval_result("potential([y*z, x*z, x*y], [x,y,z])"), "x·y·z");
    assert_eq!(eval_result("potential([2*x, 3*y^2], [x,y])"), "y^3 + x^2");
    // Metamórfico: potential ∘ gradient recupera f (mod constante).
    assert_eq!(
        eval_result("potential(gradient(x^2*y + 3*x, [x,y]), [x,y])"),
        "y·x^2 + 3·x"
    );
    // No conservativo → decline honesto; trig conservativo → decline honesto
    // (el verificador es poly-only: residual NOMBRADO, no wrong-answer).
    for probe in [
        "potential([-y,x],[x,y])",
        "potential([y,-x],[x,y])",
        "potential([cos(y), -x*sin(y)], [x,y])",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("potential("),
            "{probe} debe declinar a eco residual, got: {r}"
        );
    }
    // Pin metamórfico V6 intacto.
    assert_eq!(
        eval_result("curl(gradient(x^2*y, [x,y,z]), [x,y,z])"),
        "[[0], [0], [0]]"
    );
}
#[test]
fn test_eval_surface_integral_verb_f5() {
    // Fase 3 · F5: ensamblador de superficie — r_u×r_v como elemento de área,
    // escalar via ‖·‖, flujo via producto punto; iteradas definidas anidadas.
    let eval_result = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Área lateral del cilindro, patch plano, flujo del cilindro, flujo plano.
    assert_eq!(
        eval_result("surface_integral(1,[x,y,z],[cos(u),sin(u),v],[u,v],[0,2*pi],[0,1])"),
        "2·pi"
    );
    assert_eq!(
        eval_result("surface_integral(1,[x,y,z],[u,v,u+v],[u,v],[0,1],[0,1])"),
        "sqrt(3)"
    );
    assert_eq!(
        eval_result("surface_integral([x,y,0],[x,y,z],[cos(u),sin(u),v],[u,v],[0,2*pi],[0,1])"),
        "2·pi"
    );
    assert_eq!(
        eval_result("surface_integral([0,0,1],[x,y,z],[u,v,0],[u,v],[0,1],[0,1])"),
        "1"
    );
    // Residuales HONESTOS pineados: el verbo jamás fuerza valor.
    assert_eq!(
        eval_result("surface_integral(1,[x,y,z],[u,v,u^2+v^2],[u,v],[0,1],[0,1])"),
        "integrate(integrate(sqrt(4·u^2 + 4·v^2 + 1), v, 0, 1), u, 0, 1)"
    );
    // Esfera: el interior computa (2π·|sin u|) y el exterior queda residual —
    // el |sin(u)| definido no pliega (dueño: backlog abs-en-integral).
    assert_eq!(
        eval_result(
            "surface_integral(1,[x,y,z],[sin(u)*cos(v),sin(u)*sin(v),cos(u)],[u,v],[0,pi],[0,2*pi])"
        ),
        "integrate(2·pi·|sin(u)|, u, 0, pi)"
    );
    // Declines: r que menciona variable del campo, params dentro de vars.
    for probe in [
        "surface_integral(x,[x,y,z],[u,v,x],[u,v],[0,1],[0,1])",
        "surface_integral(1,[x,y,u],[u,v,0],[u,v],[0,1],[0,1])",
    ] {
        let r = eval_result(probe);
        assert!(
            r.starts_with("surface_integral("),
            "{probe} debe declinar a eco residual, got: {r}"
        );
    }
}
#[test]
fn test_eval_divergence_laplacian_verbs() {
    // Fase 2 V5: the scalar-output verbs. divergence REQUIRES #components == #vars
    // (mismatch → honest residual, never undefined); laplacian = div∘grad computed
    // internally; vector-laplacian stays a named scope-out. Both carry the bounded
    // budget exemption — without it a raw sum of quotient derivatives was a FALSE
    // residual (laplacian(ln(x²+y²)) hit the anti-worsen budget).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("divergence([x^2,y^2],[x,y])"), "2·x + 2·y");
    assert_eq!(r("divergence([x*y, y*z, z*x],[x,y,z])"), "x + y + z");
    assert_eq!(r("laplacian(x^2+y^2,[x,y])"), "4");
    assert_eq!(r("laplacian(x^2+y^2+z^2,[x,y,z])"), "6");
    assert_eq!(r("laplacian(sin(x)*cos(y),[x,y])"), "-2·sin(x)·cos(y)");
    // The classic HARMONIC check: Δ ln(x²+y²) = 0 exactly (this was a false residual
    // before the bounded exemption — the budget-rejection class, chokepoint-D).
    assert_eq!(r("laplacian(ln(x^2+y^2),[x,y])"), "0");
    // Honest declines: component/var mismatch, scalar target for divergence,
    // vector-laplacian scope-out.
    assert_eq!(
        r("divergence([x^2,y^2],[x,y,z])"),
        "divergence([[x^2], [y^2]], [[x], [y], [z]])"
    );
    assert_eq!(r("divergence(x^2,[x,y])"), "divergence(x^2, [[x], [y]])");
    assert_eq!(
        r("laplacian([x,y],[x,y])"),
        "laplacian([[x], [y]], [[x], [y]])"
    );
    // Narration: localized rule names + the defining-formula keyed substep.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("divergence([x^2,y^2],[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular la divergencia"
    );
    assert!(es["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("∇·F"));
    let en = steps_json("laplacian(x^2+y^2,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the Laplacian"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("second derivatives"));
}
#[test]
fn test_eval_curl_verb() {
    // Fase 2 V6: curl 3D (3×1 column, standard sign convention) and 2D (SCALAR
    // ∂Q/∂x − ∂P/∂y — never zero-padded), alias rot, and the conservativity
    // metamorphics that tie the verbs together.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("curl([y,-x,0],[x,y,z])"), "[[0], [0], [-2]]");
    assert_eq!(r("curl([y,-x],[x,y])"), "-2"); // 2D = SCALAR (pinned convention)
    assert_eq!(r("rot([x*y, y*z, z*x],[x,y,z])"), "[[-y], [-z], [-x]]");
    // Conservativity test (the elemental half of the potential-field item): a gradient
    // field is irrotational — and div∘curl vanishes identically.
    assert_eq!(
        r("curl(gradient(x*y*z,[x,y,z]),[x,y,z])"),
        "[[0], [0], [0]]"
    );
    assert_eq!(r("curl([y,x],[x,y])"), "0");
    assert_eq!(
        r("equiv(divergence(curl([x*y, y*z, z*x],[x,y,z]),[x,y,z]), 0)"),
        "true"
    );
    // Honest declines: 2 components with 3 vars, 4D, scalar target.
    assert_eq!(
        r("curl([x,y],[x,y,z])"),
        "curl([[x], [y]], [[x], [y], [z]])"
    );
    assert_eq!(r("curl(x^2,[x,y])"), "curl(x^2, [[x], [y]])");
    // Narration: localized rule name + shape-aware formula substep (3D vs 2D).
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("curl([y,-x,0],[x,y,z])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el rotacional"
    );
    assert!(es["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("∇×F"));
    let en = steps_json("curl([y,-x],[x,y])", Some("en"));
    assert_eq!(en["steps"][0]["rule"].as_str().unwrap(), "Compute the curl");
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("2D curl (scalar)"));
}
#[test]
fn test_eval_improper_rational_integral_degree_n_denominator_divergence() {
    // The engine EXPANDS a denominator like `(x^2-1)(x^2-4)` into a single degree-4 polynomial, so the
    // factor-by-factor `Mul` walk never sees the quadratics. `nonzero_on_unbounded_interval` now splits
    // a degree-≥3 polynomial via its RATIONAL roots (`factor_rational_roots`) and certifies each factor,
    // so a pole strictly inside `[a, ∞)` is detected (`undefined`) and a `~1/x` tail diverges
    // (`infinity`) instead of a conservative residual. Removable singularities are pre-simplified by the
    // engine, so the cert never fabricates a divergence for a hole.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // SOUNDNESS: a pole at a rational root strictly inside the (unbounded) range -> divergent.
    assert_eq!(r("integrate(1/(x^3-x), x, 1/2, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(1/(x^4-1), x, 0, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(1/((x^2-1)*(x^2-4)), x, 0, oo)"), "undefined"); // poles at x=1,2
                                                                            // SOUNDNESS: a `~1/x` tail of a degree-n integrand diverges to +∞ (no fabricated finite value).
    assert_eq!(r("integrate(x^2/(x^3-x), x, 2, oo)"), "infinity");
    // A removable singularity is simplified away first, so its hole is NOT read as a pole.
    assert_eq!(r("integrate((x-1)/(x^3-x), x, 1/2, oo)"), "-ln(1/3)"); // = ln3, integrand 1/(x²+x)
}
#[test]
fn test_eval_arclength_curve() {
    // `arclength(f, x, a, b)` = ∫ₐᵇ √(1 + (df/dx)²) dx, rewritten to the definite integral and
    // evaluated by the integration engine: a clean closed form when the integrand is elementary,
    // an honest residual integral otherwise (catenary, elliptic, x³, …).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("arclength(2*x+1, x, 0, 3)"), "3·sqrt(5)"); // straight line
    assert_eq!(r("arclength(x, x, 0, 5)"), "5·sqrt(2)"); // diagonal
    assert_eq!(r("arclength(3, x, 0, 4)"), "4"); // flat line, length = b − a
    assert_eq!(r("arclength(x^2, x, 0, 1)"), "1/4·asinh(2) + 1/2·sqrt(5)"); // parabola
    assert_eq!(r("arclength(x^(3/2), x, 0, 1)"), "13/27·sqrt(13) - 8/27"); // power curve
    assert_eq!(r("arc_length(x^2, x, 0, 1)"), "1/4·asinh(2) + 1/2·sqrt(5)"); // alias
                                                                             // Honest residual integrals when the integrand is not elementary.
    assert_eq!(
        r("arclength(x^3, x, 0, 1)"),
        "integrate(sqrt(9·x^4 + 1), x, 0, 1)"
    );
}
#[test]
fn test_eval_wronskian() {
    // `wronskian([f₁,…,fₙ], x)` = det of the matrix of 0th…(n−1)th derivatives — the linear-
    // independence test. Reuses symbolic differentiation + determinant. A bounded budget exemption
    // lets the cofactor expansion commit.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("wronskian([sin(x),cos(x)], x)"), "-1");
    assert_eq!(r("wronskian([1,x,x^2], x)"), "2");
    assert_eq!(r("wronskian([1,x,x^2,x^3], x)"), "12"); // 0!·1!·2!·3!
    assert_eq!(r("wronskian([e^x,e^(2*x)], x)"), "e^(3·x)");
    assert_eq!(r("wronskian([x,x^2], x)"), "x^2");
    // Linearly DEPENDENT functions ⇒ Wronskian 0 (the key application).
    assert_eq!(r("wronskian([x,2*x], x)"), "0");
    assert_eq!(r("wronskian([sin(x),2*sin(x)], x)"), "0");
}
