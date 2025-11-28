use crate::rule::Rule;
use crate::step::Step;
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::Zero;
use std::collections::{HashMap, HashSet};

pub struct Simplifier {
    rules: HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: Vec<Rc<dyn Rule>>,
    pub collect_steps: bool,
}

impl Simplifier {
    pub fn new() -> Self {
        Self { 
            rules: HashMap::new(),
            global_rules: Vec::new(),
            collect_steps: true,
        }
    }

    pub fn with_default_rules() -> Self {
        let mut s = Self::new();
        s.register_default_rules();
        s
    }

    pub fn register_default_rules(&mut self) {
        use crate::rules::*;
        
        arithmetic::register(self);
        canonicalization::register(self);
        exponents::register(self);
        logarithms::register(self);
        trigonometry::register(self);
        polynomial::register(self);
        algebra::register(self);
        calculus::register(self);
        functions::register(self);
        grouping::register(self);
    }

    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        let rule_rc: Rc<dyn Rule> = rule.into();
        if let Some(targets) = rule_rc.target_types() {
            for target in targets {
                self.rules.entry(target.to_string()).or_default().push(rule_rc.clone());
            }
        } else {
            self.global_rules.push(rule_rc);
        }
    }

    pub fn simplify(&self, expr: Rc<Expr>) -> (Rc<Expr>, Vec<Step>) {
        let mut transformer = SimplificationTransformer {
            simplifier: self,
            steps: Vec::new(),
        };
        let new_expr = transformer.transform_expr(expr);
        (new_expr, transformer.steps)
    }

    pub fn are_equivalent(&self, a: Rc<Expr>, b: Rc<Expr>) -> bool {
        let diff = Expr::sub(a, b);
        let expanded_diff = Rc::new(Expr::Function("expand".to_string(), vec![diff]));
        let (simplified_diff, _) = self.simplify(expanded_diff);
        
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
            _ => {
                let vars = self.collect_variables(result_expr);
                let mut var_map = HashMap::new();
                for var in vars {
                    var_map.insert(var, 1.23456789);
                }
                
                let val = result_expr.eval_f64(&var_map);
                if val.is_nan() {
                    return false;
                }
                val.abs() < 1e-9
            }
        }
    }

    fn collect_variables(&self, expr: &Rc<Expr>) -> HashSet<String> {
        use crate::visitors::VariableCollector;
        use cas_ast::Visitor;
        
        let mut collector = VariableCollector::new();
        collector.visit_expr(expr);
        collector.vars
    }
}

struct SimplificationTransformer<'a> {
    simplifier: &'a Simplifier,
    steps: Vec<Step>,
}

use cas_ast::visitor::Transformer;

impl<'a> Transformer for SimplificationTransformer<'a> {
    fn transform_expr(&mut self, expr: Rc<Expr>) -> Rc<Expr> {
        // 1. Simplify children first (Bottom-Up)
        let expr_with_simplified_children = match expr.as_ref() {
            Expr::Number(_) => self.transform_number(expr.clone()),
            Expr::Constant(_) => self.transform_constant(expr.clone()),
            Expr::Variable(_) => self.transform_variable(expr.clone()),
            Expr::Add(l, r) => self.transform_add(expr.clone(), l, r),
            Expr::Sub(l, r) => self.transform_sub(expr.clone(), l, r),
            Expr::Mul(l, r) => self.transform_mul(expr.clone(), l, r),
            Expr::Div(l, r) => self.transform_div(expr.clone(), l, r),
            Expr::Pow(b, e) => self.transform_pow(expr.clone(), b, e),
            Expr::Neg(e) => self.transform_neg(expr.clone(), e),
            Expr::Function(name, args) => self.transform_function(expr.clone(), name, args),
        };

        // 2. Apply rules to the current node
        self.apply_rules(expr_with_simplified_children)
    }
}

impl<'a> SimplificationTransformer<'a> {
    fn apply_rules(&mut self, mut expr: Rc<Expr>) -> Rc<Expr> {
        loop {
            let mut changed = false;
            let variant = get_variant_name(&expr);
            
            // Try specific rules
            if let Some(specific_rules) = self.simplifier.rules.get(variant) {
                for rule in specific_rules {
                    if let Some(rewrite) = rule.apply(&expr) {
                        if self.simplifier.collect_steps {
                            self.steps.push(Step::new(
                                &rewrite.description,
                                rule.name(),
                                expr.clone(),
                                rewrite.new_expr.clone(),
                            ));
                        }
                        expr = rewrite.new_expr;
                        changed = true;
                        break; // Restart loop with new expression
                    }
                }
            }
            
            if changed { 
                // If changed, re-simplify the whole tree
                return self.transform_expr(expr);
            }

            // Try global rules
            for rule in &self.simplifier.global_rules {
                if let Some(rewrite) = rule.apply(&expr) {
                    if self.simplifier.collect_steps {
                        self.steps.push(Step::new(
                            &rewrite.description,
                            rule.name(),
                            expr.clone(),
                            rewrite.new_expr.clone(),
                        ));
                    }
                    expr = rewrite.new_expr;
                    changed = true;
                    break; // Restart loop
                }
            }
            
            if changed {
                // If changed, re-simplify the whole tree
                return self.transform_expr(expr);
            }

            // If no rules applied, we are done with this node
            return expr;
        }
    }
}

fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Variable(_) => "Variable",
        Expr::Number(_) => "Number",
        Expr::Constant(_) => "Constant",
    }
}
