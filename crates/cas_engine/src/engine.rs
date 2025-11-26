use crate::rule::Rule;
use crate::step::Step;
use cas_ast::Expr;
use std::rc::Rc;

pub struct Simplifier {
    rules: Vec<Box<dyn Rule>>,
}

impl Simplifier {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        self.rules.push(rule);
    }

    pub fn simplify(&self, mut expr: Rc<Expr>) -> (Rc<Expr>, Vec<Step>) {
        let mut steps = Vec::new();
        let mut changed = true;

        // Naive loop until no more changes. 
        // In a real CAS, we need better strategies (bottom-up, top-down, etc.)
        while changed {
            changed = false;
            for rule in &self.rules {
                if let Some(rewrite) = rule.apply(&expr) {
                    steps.push(Step::new(
                        &rewrite.description,
                        rule.name(),
                        expr.clone(),
                        rewrite.new_expr.clone(),
                    ));
                    expr = rewrite.new_expr;
                    changed = true;
                    break; // Restart loop after a change
                }
            }
            
            // If no top-level rule applied, try to simplify children (recursion)
            if !changed {
                match expr.as_ref() {
                    Expr::Add(l, r) => {
                        let (new_l, l_steps) = self.simplify(l.clone());
                        let (new_r, r_steps) = self.simplify(r.clone());
                        if new_l != *l || new_r != *r {
                            steps.extend(l_steps);
                            steps.extend(r_steps);
                            expr = Expr::add(new_l, new_r);
                            changed = true;
                        }
                    }
                    Expr::Sub(l, r) => {
                        let (new_l, l_steps) = self.simplify(l.clone());
                        let (new_r, r_steps) = self.simplify(r.clone());
                        if new_l != *l || new_r != *r {
                            steps.extend(l_steps);
                            steps.extend(r_steps);
                            expr = Expr::sub(new_l, new_r);
                            changed = true;
                        }
                    }
                    Expr::Mul(l, r) => {
                        let (new_l, l_steps) = self.simplify(l.clone());
                        let (new_r, r_steps) = self.simplify(r.clone());
                        if new_l != *l || new_r != *r {
                            steps.extend(l_steps);
                            steps.extend(r_steps);
                            expr = Expr::mul(new_l, new_r);
                            changed = true;
                        }
                    }
                    Expr::Div(l, r) => {
                        let (new_l, l_steps) = self.simplify(l.clone());
                        let (new_r, r_steps) = self.simplify(r.clone());
                        if new_l != *l || new_r != *r {
                            steps.extend(l_steps);
                            steps.extend(r_steps);
                            expr = Expr::div(new_l, new_r);
                            changed = true;
                        }
                    }
                    Expr::Pow(b, e) => {
                        let (new_b, b_steps) = self.simplify(b.clone());
                        let (new_e, e_steps) = self.simplify(e.clone());
                        if new_b != *b || new_e != *e {
                            steps.extend(b_steps);
                            steps.extend(e_steps);
                            expr = Expr::pow(new_b, new_e);
                            changed = true;
                        }
                    }
                    Expr::Neg(e) => {
                        let (new_e, e_steps) = self.simplify(e.clone());
                        if new_e != *e {
                            steps.extend(e_steps);
                            expr = Expr::neg(new_e);
                            changed = true;
                        }
                    }
                    Expr::Function(name, args) => {
                        let mut new_args = Vec::new();
                        let mut args_changed = false;
                        for arg in args {
                            let (new_arg, arg_steps) = self.simplify(arg.clone());
                            if new_arg != *arg {
                                args_changed = true;
                                steps.extend(arg_steps);
                            }
                            new_args.push(new_arg);
                        }
                        if args_changed {
                            expr = Rc::new(Expr::Function(name.clone(), new_args));
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        (expr, steps)
    }
}
