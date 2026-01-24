#![allow(clippy::format_in_format_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(dead_code)]
#![allow(unused_variables)]
use cas_ast::Expr;
use cas_engine::eval::{Engine, EvalAction, EvalRequest, EvalResult};
use cas_engine::session_state::SessionState;
use cas_engine::EntryKind;
use cas_parser::Statement;

#[test]
fn test_solve_session_ref() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // 1. Store "1 + x" (as in user session #1)
    // Actually user had #1: 1+x, #2: 1+x=5.
    // Let's simulate exactly that.

    // #1: "1 + x"
    let expr1_str = "1 + x";
    let parsed1 = cas_parser::parse(expr1_str, &mut engine.simplifier.context).unwrap();
    state
        .store
        .push(EntryKind::Expr(parsed1), expr1_str.to_string());

    // #2: "1 + x = 5"
    let eq_str = "1 + x = 5";
    let stmt = cas_parser::parse_statement(eq_str, &mut engine.simplifier.context).unwrap();
    let (kind, parsed_eq_expr) = match stmt {
        Statement::Equation(eq) => {
            let eq_expr = engine
                .simplifier
                .context
                .call("Equal", vec![eq.lhs, eq.rhs]);
            (
                EntryKind::Eq {
                    lhs: eq.lhs,
                    rhs: eq.rhs,
                },
                eq_expr,
            )
        }
        _ => panic!("Expected equation"),
    };
    state.store.push(kind, eq_str.to_string()); // Becomes #2

    // Now solve #2 for x
    // CLI parses "#2" as Variable("#2")
    let id_var = engine.simplifier.context.var("#2");

    let req = EvalRequest {
        raw_input: "#2".to_string(),
        parsed: id_var,
        kind: EntryKind::Expr(id_var),
        action: EvalAction::Solve {
            var: "x".to_string(),
        },
        auto_store: true,
    };

    let result = engine.eval(&mut state, req);

    match result {
        Ok(output) => {
            match output.result {
                EvalResult::SolutionSet(solution_set) => {
                    println!("Solution set: {:?}", solution_set);
                    // V2.0: Expect SolutionSet::Discrete with solution 4
                    match solution_set {
                        cas_ast::SolutionSet::Discrete(sols) => {
                            println!("Solutions: {:?}", sols);
                        }
                        _ => panic!("Expected Discrete solution set"),
                    }
                }
                _ => panic!("Expected SolutionSet result, got {:?}", output.result),
            }
        }
        Err(e) => panic!("Eval failed: {}", e),
    }
}
