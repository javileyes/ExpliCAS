//! Diagnóstico/regresión: el cociente opaco no debe colapsar a constante un
//! cociente trigonométrico no constante (wrong answer `7/3` descubierto el
//! 2026-07-31 en `integrate(cos(x)*(sin(x)+1)^2, x)`).

use cas_ast::Expr;
use cas_formatter::render_expr;
use cas_parser::parse;

const NUM: &str = "4*cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^2 + 4*cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^4 + 20/3*cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^3 + cos(x/2)*cos(x/2)*(2*(sin(x/2))^5 + (cos(x/2))^3*(sin(x/2 - x/2) + 2*sin(x/2)*cos(x/2)))";
const DEN: &str = "cos(x/2)*(cos(x/2))^6 + 3*cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^2 + 3*cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^4 + cos(x/2)*(cos(x/2))^6*(sin(x/2)/cos(x/2))^6";

#[test]
fn opaque_quotient_with_engine_callbacks_must_not_return_constant() {
    // 2,2 s desde que los simplify de estrategia llevan tope temporal
    // (L16(b), STRATEGY_SIMPLIFY_TIME_BUDGET_MS); antes ~150 s de molino.
    let mut ctx = cas_ast::Context::new();
    let num = parse(NUM, &mut ctx).expect("num");
    let den = parse(DEN, &mut ctx).expect("den");
    let expr = ctx.add(Expr::Div(num, den));

    let out = cas_math::div_expand_cancel_support::try_rewrite_div_expand_to_cancel_expr_with_thread_guards(
        &mut ctx,
        expr,
        |base_ctx, sub_frac| {
            let mut simplifier = cas_engine::Simplifier::with_default_rules();
            simplifier.context = base_ctx.clone();
            let mut options = cas_engine::SimplifyOptions::default();
            options.time_budget_ms =
                Some(cas_math::div_expand_cancel_support::STRATEGY_SIMPLIFY_TIME_BUDGET_MS);
            let (simplified, _) = simplifier.simplify_with_options(sub_frac, options);
            Some((simplifier.context, simplified))
        },
        cas_engine::expand,
        |expanded_ctx, expanded_num, expanded_den| {
            let mut simplifier = cas_engine::Simplifier::with_default_rules();
            simplifier.context = expanded_ctx;
            let mut options = cas_engine::SimplifyOptions::default();
            options.time_budget_ms =
                Some(cas_math::div_expand_cancel_support::STRATEGY_SIMPLIFY_TIME_BUDGET_MS);
            let (simplified_num, _) =
                simplifier.simplify_with_options(expanded_num, options.clone());
            let (simplified_den, _) = simplifier.simplify_with_options(expanded_den, options);
            Some((simplifier.context, simplified_num, simplified_den))
        },
    );

    match out {
        Some(rw) => {
            let rendered = render_expr(&ctx, rw.rewritten);
            println!("KIND {:?}", rw.kind);
            println!("REWRITTEN {rendered}");
            // El cociente N/D vale 1,148 en x=0,7 y 2,190 en x=1,3: NO es
            // constante. Cualquier resultado constante es un wrong answer.
            let is_constant = matches!(ctx.get(rw.rewritten), Expr::Number(_) | Expr::Constant(_));
            assert!(
                !is_constant,
                "el cociente opaco colapsó un cociente NO constante a la constante {rendered}"
            );
        }
        None => println!("NONE"),
    }
}

#[test]
#[ignore = "diagnóstico (~85 s): simplify TOP-LEVEL directo del monstruo trig — coste de orquestación fuera del alcance de L16(b); compara nombre neutro vs __opq0, deben dar el MISMO resultado"]
fn nested_simplifier_name_sensitivity_probe() {
    for (name, src) in [("q", "(4*q*q^6*(sin(x/2)/q)^2 + 4*q*q^6*(sin(x/2)/q)^4 + 20/3*q*q^6*(sin(x/2)/q)^3 + q*q*(2*sin(x/2)^5 + q^3*(sin(x/2-x/2) + 2*q*sin(x/2))))/(q*q^6 + 3*q*q^6*(sin(x/2)/q)^2 + 3*q*q^6*(sin(x/2)/q)^4 + q*q^6*(sin(x/2)/q)^6)"), ("__opq0", "(4*__opq0*__opq0^6*(sin(x/2)/__opq0)^2 + 4*__opq0*__opq0^6*(sin(x/2)/__opq0)^4 + 20/3*__opq0*__opq0^6*(sin(x/2)/__opq0)^3 + __opq0*__opq0*(2*sin(x/2)^5 + __opq0^3*(sin(x/2-x/2) + 2*__opq0*sin(x/2))))/(__opq0*__opq0^6 + 3*__opq0*__opq0^6*(sin(x/2)/__opq0)^2 + 3*__opq0*__opq0^6*(sin(x/2)/__opq0)^4 + __opq0*__opq0^6*(sin(x/2)/__opq0)^6)")] {
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let expr = parse(src, &mut simplifier.context).expect("parse");
        let (simplified, steps) = simplifier.simplify(expr);
        println!("VAR {name} -> {}", render_expr(&simplifier.context, simplified));
        if name == "__opq0" {
            for st in &steps {
                println!(
                    "  STEP [{}] -> {}",
                    st.rule_name,
                    render_expr(&simplifier.context, st.after)
                );
            }
        }
    }
}
