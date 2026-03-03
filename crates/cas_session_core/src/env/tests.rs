use super::*;
use cas_ast::Expr;
use cas_parser::parse;

fn contains_integer(ctx: &Context, root: ExprId, value: i64) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Number(n) => {
                if n.is_integer() && n.to_integer() == value.into() {
                    return true;
                }
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

#[test]
fn test_substitute_simple() {
    let mut ctx = Context::new();
    let mut env = Environment::new();

    // a = 2
    let two = ctx.num(2);
    env.set("a".to_string(), two);

    // a + 1
    let expr = parse("a + 1", &mut ctx).unwrap();

    // substitute → 2 + 1 (or 1 + 2 due to ordering)
    let result = substitute(&mut ctx, &env, expr);
    assert!(cas_ast::traversal::collect_variables(&ctx, result).is_empty());
    assert!(contains_integer(&ctx, result, 1));
    assert!(contains_integer(&ctx, result, 2));
}

#[test]
fn test_substitute_transitive() {
    let mut ctx = Context::new();
    let mut env = Environment::new();

    // b = 3
    let three = ctx.num(3);
    env.set("b".to_string(), three);

    // a = b + 1
    let a_expr = parse("b + 1", &mut ctx).unwrap();
    env.set("a".to_string(), a_expr);

    // a * 2 → (3 + 1) * 2
    let expr = parse("a * 2", &mut ctx).unwrap();
    let result = substitute(&mut ctx, &env, expr);
    assert!(cas_ast::traversal::collect_variables(&ctx, result).is_empty());
    assert!(contains_integer(&ctx, result, 3));
    assert!(contains_integer(&ctx, result, 2));
}

#[test]
fn test_cycle_direct_no_hang() {
    let mut ctx = Context::new();
    let mut env = Environment::new();

    // a = a + 1 (direct cycle)
    let a_expr = parse("a + 1", &mut ctx).unwrap();
    env.set("a".to_string(), a_expr);

    // This should NOT hang - cycle detection should break it
    let expr = parse("a * 2", &mut ctx).unwrap();
    let _result = substitute(&mut ctx, &env, expr);
    // If we got here, no infinite loop - test passes
}

#[test]
fn test_cycle_indirect_no_hang() {
    let mut ctx = Context::new();
    let mut env = Environment::new();

    // a = b + 1
    let a_expr = parse("b + 1", &mut ctx).unwrap();
    env.set("a".to_string(), a_expr);

    // b = a + 1 (indirect cycle)
    let b_expr = parse("a + 1", &mut ctx).unwrap();
    env.set("b".to_string(), b_expr);

    // This should NOT hang
    let expr = parse("a * 2", &mut ctx).unwrap();
    let _result = substitute(&mut ctx, &env, expr);
    // If we got here, no infinite loop - test passes
}

#[test]
fn test_shadow_variable() {
    let mut ctx = Context::new();
    let mut env = Environment::new();

    // x = 3
    let three = ctx.num(3);
    env.set("x".to_string(), three);

    // x^2 with shadow ["x"] → should remain x^2, not 9
    let expr = parse("x^2", &mut ctx).unwrap();
    let result = substitute_with_shadow(&mut ctx, &env, expr, &["x"]);
    let vars = cas_ast::traversal::collect_variables(&ctx, result);
    assert!(vars.contains("x"), "x should stay unresolved when shadowed");
}

#[test]
fn test_is_reserved() {
    assert!(is_reserved("let"));
    assert!(is_reserved("sin"));
    assert!(is_reserved("pi"));
    assert!(is_reserved("Pi")); // case insensitive
    assert!(!is_reserved("myvar"));
    assert!(!is_reserved("a"));
}

#[test]
fn test_env_operations() {
    let mut env = Environment::new();
    let mut ctx = Context::new();
    let expr_id = ctx.num(0);

    env.set("a".to_string(), expr_id);
    assert!(env.contains("a"));
    assert_eq!(env.get("a"), Some(expr_id));
    assert_eq!(env.len(), 1);

    env.unset("a");
    assert!(!env.contains("a"));
    assert!(env.is_empty());
}
