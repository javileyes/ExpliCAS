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
        
        // Arithmetic
        self.add_rule(Box::new(arithmetic::AddZeroRule));
        self.add_rule(Box::new(arithmetic::MulOneRule));
        self.add_rule(Box::new(arithmetic::MulZeroRule));
        self.add_rule(Box::new(arithmetic::CombineConstantsRule));
        
        // Canonicalization
        self.add_rule(Box::new(canonicalization::CanonicalizeNegationRule));
        self.add_rule(Box::new(canonicalization::CanonicalizeAddRule));
        self.add_rule(Box::new(canonicalization::CanonicalizeMulRule));
        self.add_rule(Box::new(canonicalization::CanonicalizeRootRule));
        self.add_rule(Box::new(canonicalization::AssociativityRule));
        
        // Exponents
        self.add_rule(Box::new(exponents::ProductPowerRule));
        self.add_rule(Box::new(exponents::PowerPowerRule));
        self.add_rule(Box::new(exponents::EvaluatePowerRule));
        self.add_rule(Box::new(exponents::ZeroOnePowerRule));
        
        // Logarithms
        self.add_rule(Box::new(logarithms::EvaluateLogRule));
        self.add_rule(Box::new(logarithms::ExponentialLogRule));
        self.add_rule(Box::new(logarithms::SplitLogExponentsRule));
        
        // Trigonometry
        self.add_rule(Box::new(trigonometry::EvaluateTrigRule));
        self.add_rule(Box::new(trigonometry::PythagoreanIdentityRule));
        self.add_rule(Box::new(trigonometry::AngleIdentityRule));
        self.add_rule(Box::new(trigonometry::TanToSinCosRule));
        self.add_rule(Box::new(trigonometry::DoubleAngleRule));
        
        // Polynomial
        self.add_rule(Box::new(polynomial::DistributeRule));
        self.add_rule(Box::new(polynomial::AnnihilationRule));
        self.add_rule(Box::new(polynomial::CombineLikeTermsRule));
        self.add_rule(Box::new(polynomial::BinomialExpansionRule));
        
        // Algebra
        self.add_rule(Box::new(algebra::SimplifyFractionRule));
        self.add_rule(Box::new(algebra::NestedFractionRule));
        self.add_rule(Box::new(algebra::ExpandRule));
        self.add_rule(Box::new(algebra::FactorRule));
        self.add_rule(Box::new(algebra::FactorDifferenceSquaresRule));
        
        // Calculus
        self.add_rule(Box::new(calculus::IntegrateRule));
        
        // Functions
        self.add_rule(Box::new(functions::EvaluateAbsRule));
        
        // Grouping
        self.add_rule(Box::new(grouping::CollectRule));
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
        self.simplify_rec(expr)
    }

    fn simplify_rec(&self, expr: Rc<Expr>) -> (Rc<Expr>, Vec<Step>) {
        let mut steps = Vec::new();

        // 1. Simplify children first (Bottom-Up)
        let (mut current_expr, child_steps) = self.simplify_children(expr);
        steps.extend(child_steps);

        // 2. Apply rules to the current node
        // We loop here to handle multiple rule applications on the same node level 
        // before potentially recursing if the structure changes significantly?
        // Actually, if we recurse on change, we don't need a loop here.
        // But to avoid deep recursion for simple chains (A->B->C), a loop is better.
        
        let mut changed = true;
        while changed {
            changed = false;
            let variant = get_variant_name(&current_expr);
            
            // Try specific rules
            if let Some(specific_rules) = self.rules.get(variant) {
                for rule in specific_rules {
                    if let Some(rewrite) = rule.apply(&current_expr) {
                        if self.collect_steps {
                            steps.push(Step::new(
                                &rewrite.description,
                                rule.name(),
                                current_expr.clone(),
                                rewrite.new_expr.clone(),
                            ));
                        }
                        current_expr = rewrite.new_expr;
                        changed = true;
                        break; // Restart loop with new expression
                    }
                }
            }
            
            if changed { continue; }

            // Try global rules
            for rule in &self.global_rules {
                if let Some(rewrite) = rule.apply(&current_expr) {
                    if self.collect_steps {
                        steps.push(Step::new(
                            &rewrite.description,
                            rule.name(),
                            current_expr.clone(),
                            rewrite.new_expr.clone(),
                        ));
                    }
                    current_expr = rewrite.new_expr;
                    changed = true;
                    break; // Restart loop
                }
            }
            
            // If we changed, we might need to re-simplify children if the structure changed?
            // If A -> B, and B has children, are they simplified?
            // If the rule just modified the top level (e.g. x+x -> 2x), children are same (x).
            // If the rule introduced new structure (e.g. (x+1)^2 -> x^2+2x+1), the new children (x^2, 2x, 1) are likely simplified or simple.
            // But strictly speaking, we should recurse.
            // However, full recursion might be expensive.
            // Let's stick to the loop for now. If a rule produces something unsimplified, 
            // the next iteration of the loop will see it as the current node.
            // But if it produces a tree, we only look at the root of that tree.
            // Correctness requires checking children of the result.
            // So: if changed, we should probably call simplify_rec on the result?
            // But that risks infinite recursion if a rule cycles.
            // The original code restarted the loop from the top.
            // Here we are at a specific node.
            
            if changed {
                // To be safe and correct (Bottom-Up), if we change the node, we should treat it as a new subtree.
                // So we should recurse.
                let (new_expr_rec, new_steps_rec) = self.simplify_rec(current_expr);
                steps.extend(new_steps_rec);
                return (new_expr_rec, steps);
            }
        }

        (current_expr, steps)
    }

    fn simplify_children(&self, expr: Rc<Expr>) -> (Rc<Expr>, Vec<Step>) {
        let mut steps = Vec::new();
        let new_expr = match expr.as_ref() {
            Expr::Add(l, r) => {
                let (new_l, l_steps) = self.simplify_rec(l.clone());
                let (new_r, r_steps) = self.simplify_rec(r.clone());
                steps.extend(l_steps);
                steps.extend(r_steps);
                if new_l != *l || new_r != *r {
                    Expr::add(new_l, new_r)
                } else {
                    expr.clone()
                }
            }
            Expr::Sub(l, r) => {
                let (new_l, l_steps) = self.simplify_rec(l.clone());
                let (new_r, r_steps) = self.simplify_rec(r.clone());
                steps.extend(l_steps);
                steps.extend(r_steps);
                if new_l != *l || new_r != *r {
                    Expr::sub(new_l, new_r)
                } else {
                    expr.clone()
                }
            }
            Expr::Mul(l, r) => {
                let (new_l, l_steps) = self.simplify_rec(l.clone());
                let (new_r, r_steps) = self.simplify_rec(r.clone());
                steps.extend(l_steps);
                steps.extend(r_steps);
                if new_l != *l || new_r != *r {
                    Expr::mul(new_l, new_r)
                } else {
                    expr.clone()
                }
            }
            Expr::Div(l, r) => {
                let (new_l, l_steps) = self.simplify_rec(l.clone());
                let (new_r, r_steps) = self.simplify_rec(r.clone());
                steps.extend(l_steps);
                steps.extend(r_steps);
                if new_l != *l || new_r != *r {
                    Expr::div(new_l, new_r)
                } else {
                    expr.clone()
                }
            }
            Expr::Pow(b, e) => {
                let (new_b, b_steps) = self.simplify_rec(b.clone());
                let (new_e, e_steps) = self.simplify_rec(e.clone());
                steps.extend(b_steps);
                steps.extend(e_steps);
                if new_b != *b || new_e != *e {
                    Expr::pow(new_b, new_e)
                } else {
                    expr.clone()
                }
            }
            Expr::Neg(e) => {
                let (new_e, e_steps) = self.simplify_rec(e.clone());
                steps.extend(e_steps);
                if new_e != *e {
                    Expr::neg(new_e)
                } else {
                    expr.clone()
                }
            }
            Expr::Function(name, args) => {
                let mut new_args = Vec::new();
                let mut args_changed = false;
                for arg in args {
                    let (new_arg, arg_steps) = self.simplify_rec(arg.clone());
                    if new_arg != *arg {
                        args_changed = true;
                    }
                    steps.extend(arg_steps);
                    new_args.push(new_arg);
                }
                if args_changed {
                    Rc::new(Expr::Function(name.clone(), new_args))
                } else {
                    expr.clone()
                }
            }
            _ => expr.clone(),
        };
        (new_expr, steps)
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
