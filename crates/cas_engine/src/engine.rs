use crate::rule::Rule;
use crate::step::Step;
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::Zero;

pub struct Simplifier {
    rules: Vec<Box<dyn Rule>>,
    pub collect_steps: bool,
}

impl Simplifier {
    pub fn new() -> Self {
        Self { 
            rules: Vec::new(),
            collect_steps: true,
        }
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
                    if self.collect_steps {
                        steps.push(Step::new(
                            &rewrite.description,
                            rule.name(),
                            expr.clone(),
                            rewrite.new_expr.clone(),
                        ));
                    }
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
    pub fn are_equivalent(&self, a: Rc<Expr>, b: Rc<Expr>) -> bool {
        let diff = Expr::sub(a, b);
        
        // Force expansion of the difference to handle cases like (x+1)^2 - (x^2+2x+1)
        // We wrap it in expand() so that ExpandRule (if present) triggers.
        let expanded_diff = Rc::new(Expr::Function("expand".to_string(), vec![diff]));
        
        let (simplified_diff, _) = self.simplify(expanded_diff);
        
        // If expansion failed (e.g. non-polynomial), we might get expand(...) back.
        // In that case, check the inner expression.
        let result_expr = if let Expr::Function(name, args) = simplified_diff.as_ref() {
            if name == "expand" && args.len() == 1 {
                &args[0]
            } else {
                &simplified_diff
            }
        } else {
            &simplified_diff
        };

        match result_expr.as_ref() {
            Expr::Number(n) => n.is_zero(),
            _ => false,
        }
    }
}
