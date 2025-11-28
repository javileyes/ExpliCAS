use cas_ast::{Expr, Visitor, ExprId, Context};
use std::collections::HashSet;

pub struct VariableCollector {
    pub vars: HashSet<String>,
}

impl VariableCollector {
    pub fn new() -> Self {
        Self {
            vars: HashSet::new(),
        }
    }
}

impl Visitor for VariableCollector {
    fn visit_variable(&mut self, name: &str) {
        self.vars.insert(name.to_string());
    }
}

pub struct DepthVisitor {
    pub depth: usize,
    current_depth: usize,
}

impl DepthVisitor {
    pub fn new() -> Self {
        Self {
            depth: 0,
            current_depth: 0,
        }
    }
}

impl Visitor for DepthVisitor {
    fn visit_expr(&mut self, ctx: &Context, expr: ExprId) {
        self.current_depth += 1;
        if self.current_depth > self.depth {
            self.depth = self.current_depth;
        }
        
        // Manual dispatch to traverse children
        match ctx.get(expr) {
            Expr::Number(n) => self.visit_number(n),
            Expr::Constant(c) => self.visit_constant(c),
            Expr::Variable(name) => self.visit_variable(name),
            Expr::Add(l, r) => self.visit_add(ctx, *l, *r),
            Expr::Sub(l, r) => self.visit_sub(ctx, *l, *r),
            Expr::Mul(l, r) => self.visit_mul(ctx, *l, *r),
            Expr::Div(l, r) => self.visit_div(ctx, *l, *r),
            Expr::Pow(b, e) => self.visit_pow(ctx, *b, *e),
            Expr::Neg(e) => self.visit_neg(ctx, *e),
            Expr::Function(name, args) => self.visit_function(ctx, name, args),
        }
        
        self.current_depth -= 1;
    }
}
