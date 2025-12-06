use cas_ast::{Context, DisplayExpr, Expr};
use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn debug_pi_forms() {
    let mut simplifier = Simplifier::with_default_rules();

    // Test pi/4
    let expr = parse("pi/4", &mut simplifier.context).unwrap();
    println!("\n=== pi/4 ===");
    println!("Top expr: {:?}", simplifier.context.get(expr));
    debug_expr_recursive(&simplifier.context, expr, 0);

    // Test pi/2
    let expr2 = parse("pi/2", &mut simplifier.context).unwrap();
    println!("\n=== pi/2 ===");
    println!("Top expr: {:?}", simplifier.context.get(expr2));
    debug_expr_recursive(&simplifier.context, expr2, 0);

    // Test cot(pi/4)
    let expr3 = parse("cot(pi/4)", &mut simplifier.context).unwrap();
    println!("\n=== cot(pi/4) ===");
    println!("Top expr: {:?}", simplifier.context.get(expr3));
    if let Expr::Function(name, args) = simplifier.context.get(expr3) {
        println!("Function: {}", name);
        if args.len() > 0 {
            println!("Arg[0]: {:?}", simplifier.context.get(args[0]));
            debug_expr_recursive(&simplifier.context, args[0], 1);
        }
    }
}

fn debug_expr_recursive(ctx: &Context, expr_id: cas_ast::ExprId, depth: usize) {
    let indent = "  ".repeat(depth);
    match ctx.get(expr_id) {
        Expr::Mul(l, r) => {
            println!("{}Mul:", indent);
            println!("{}  Left: {:?}", indent, ctx.get(*l));
            debug_expr_recursive(ctx, *l, depth + 2);
            println!("{}  Right: {:?}", indent, ctx.get(*r));
            debug_expr_recursive(ctx, *r, depth + 2);
        }
        Expr::Div(num, den) => {
            println!("{}Div:", indent);
            println!("{}  Num: {:?}", indent, ctx.get(*num));
            debug_expr_recursive(ctx, *num, depth + 2);
            println!("{}  Den: {:?}", indent, ctx.get(*den));
            debug_expr_recursive(ctx, *den, depth + 2);
        }
        _ => {}
    }
}
