use cas_engine::engine::Simplifier;
use cas_parser::parse;

fn simp(input: &str) -> String {
    let mut s = Simplifier::with_default_rules();
    let e = parse(input, &mut s.context).unwrap();
    let (r, _) = s.simplify(e);
    let cfg = cas_engine::semantics::EvalConfig::default();
    let mut budget = cas_engine::budget::Budget::preset_cli();
    let r2 = if let Ok(res) = cas_engine::const_fold::fold_constants(
        &mut s.context,
        r,
        &cfg,
        cas_engine::const_fold::ConstFoldMode::Safe,
        &mut budget,
    ) {
        res.expr
    } else {
        r
    };
    cas_ast::LaTeXExpr {
        context: &s.context,
        id: r2,
    }
    .to_latex()
}

#[test]
fn diag_cos4u() {
    let cases = [
        // What does cos(4u) simplify to?
        "cos(4*u)",
        // What does the RHS simplify to?
        "8*cos(u)^4 - 8*cos(u)^2 + 1",
        // What is the difference?
        "cos(4*u) - (8*cos(u)^4 - 8*cos(u)^2 + 1)",
        // Pythagorean identity checks
        "sin(u)^2 + cos(u)^2",
        "sin(u)^4 + cos(u)^4 + 2*sin(u)^2*cos(u)^2",
        "4*sin(u)^4 + 4*cos(u)^4 + 8*sin(u)^2*cos(u)^2 - 4",
        // What does the engine do with sin^2 + cos^2 in products?
        "(sin(u)^2 + cos(u)^2)^2",
        "sin(u)^4 + 2*sin(u)^2*cos(u)^2 + cos(u)^4",
        // Simple Pythagorean
        "sin(u)^4 + cos(u)^4",
        // cos(4u) alternative forms
        "2*cos(2*u)^2 - 1",
        "cos(2*u)",
    ];
    for c in &cases {
        eprintln!("{:60} => {}", c, simp(c));
    }
}
