use crate::rule::Rule;
use crate::step::Step;
use crate::profiler::RuleProfiler;
use cas_ast::{Expr, ExprId, Context};
use std::rc::Rc;
use num_traits::{Zero, ToPrimitive};
use std::collections::{HashMap, HashSet};

use tracing::debug;

pub struct Simplifier {
    pub context: Context,
    rules: HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: Vec<Rc<dyn Rule>>,
    pub collect_steps: bool,
    pub allow_numerical_verification: bool,
    pub debug_mode: bool,
    disabled_rules: HashSet<String>,
    pub enable_polynomial_strategy: bool,
    pub profiler: RuleProfiler,
}

impl Simplifier {
    pub fn new() -> Self {
        Self { 
            context: Context::new(),
            rules: HashMap::new(),
            global_rules: Vec::new(),
            collect_steps: true,
            allow_numerical_verification: true,
            debug_mode: false,
            disabled_rules: HashSet::new(),
            enable_polynomial_strategy: true,
            profiler: RuleProfiler::new(false),  // Disabled by default
        }
    }

    pub fn with_default_rules() -> Self {
        let mut s = Self::new();
        s.register_default_rules();
        s
    }

    pub fn enable_debug(&mut self) {
        self.debug_mode = true;
    }

    pub fn disable_debug(&mut self) {
        self.debug_mode = false;
    }

    pub fn debug(&self, msg: &str) {
        // Use tracing for structured logging.
        // We still check debug_mode to allow per-instance toggling if needed,
        // but ideally this should be controlled by RUST_LOG.
        // For now, let's log if EITHER debug_mode is on OR tracing is enabled at debug level.
        // Actually, let's just delegate to tracing. The subscriber will filter.
        debug!("{}", msg);
    }

    pub fn disable_rule(&mut self, rule_name: &str) {
        self.disabled_rules.insert(rule_name.to_string());
    }

    pub fn enable_rule(&mut self, rule_name: &str) {
        self.disabled_rules.remove(rule_name);
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

    pub fn get_all_rule_names(&self) -> Vec<String> {
        let mut names = HashSet::new();
        
        for rule in &self.global_rules {
            names.insert(rule.name().to_string());
        }
        
        for rules in self.rules.values() {
            for rule in rules {
                names.insert(rule.name().to_string());
            }
        }
        
        let mut sorted_names: Vec<String> = names.into_iter().collect();
        sorted_names.sort();
        sorted_names
    }

    pub fn simplify(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        let mut orchestrator = crate::orchestrator::Orchestrator::new();
        orchestrator.enable_polynomial_strategy = self.enable_polynomial_strategy;
        orchestrator.simplify(expr_id, self)
    }

    pub fn apply_rules_loop(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        let rules = &self.rules;
        let global_rules = &self.global_rules;
        let collect_steps = self.collect_steps;
        
        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules,
            global_rules,
            disabled_rules: &self.disabled_rules,
            collect_steps,
            steps: Vec::new(),
            cache: HashMap::new(),
            current_path: Vec::new(),
            profiler: &mut self.profiler,
        };
        
        let new_expr = local_transformer.transform_expr_recursive(expr_id);
        (new_expr, local_transformer.steps)
    }

    pub fn are_equivalent(&mut self, a: ExprId, b: ExprId) -> bool {
        let diff = self.context.add(Expr::Sub(a, b));
        let expand_str = "expand".to_string();
        let expanded_diff = self.context.add(Expr::Function(expand_str.clone(), vec![diff]));
        let (simplified_diff, _) = self.simplify(expanded_diff);
        
        let result_expr = {
            let expr = self.context.get(simplified_diff);
            if let Expr::Function(name, args) = expr {
                if name == "expand" && args.len() == 1 {
                    args[0]
                } else {
                    simplified_diff
                }
            } else {
                simplified_diff
            }
        };

        let expr = self.context.get(result_expr);
        match expr {
            Expr::Number(n) => n.is_zero(),
            _ => {
                if !self.allow_numerical_verification {
                    return false;
                }
                let vars = self.collect_variables(result_expr);
                let mut var_map = HashMap::new();
                for var in vars {
                    var_map.insert(var, 1.23456789);
                }
                
                if let Some(val) = eval_f64(&self.context, result_expr, &var_map) {
                    val.abs() < 1e-9
                } else {
                    false
                }
            }
        }
    }

    fn collect_variables(&self, expr_id: ExprId) -> HashSet<String> {
        use crate::visitors::VariableCollector;
        use cas_ast::Visitor;
        
        let mut collector = VariableCollector::new();
        collector.visit_expr(&self.context, expr_id);
        collector.vars
    }
}

fn eval_f64(ctx: &Context, expr: ExprId, var_map: &HashMap<String, f64>) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(n) => n.to_f64(),
        Expr::Variable(v) => var_map.get(v).cloned(),
        Expr::Add(l, r) => Some(eval_f64(ctx, *l, var_map)? + eval_f64(ctx, *r, var_map)?),
        Expr::Sub(l, r) => Some(eval_f64(ctx, *l, var_map)? - eval_f64(ctx, *r, var_map)?),
        Expr::Mul(l, r) => Some(eval_f64(ctx, *l, var_map)? * eval_f64(ctx, *r, var_map)?),
        Expr::Div(l, r) => Some(eval_f64(ctx, *l, var_map)? / eval_f64(ctx, *r, var_map)?),
        Expr::Pow(b, e) => Some(eval_f64(ctx, *b, var_map)?.powf(eval_f64(ctx, *e, var_map)?)),
        Expr::Neg(e) => Some(-eval_f64(ctx, *e, var_map)?),
        Expr::Function(name, args) => {
             let arg_vals: Option<Vec<f64>> = args.iter().map(|a| eval_f64(ctx, *a, var_map)).collect();
             let arg_vals = arg_vals?;
             match name.as_str() {
                 "sin" => Some(arg_vals.get(0)?.sin()),
                 "cos" => Some(arg_vals.get(0)?.cos()),
                 "tan" => Some(arg_vals.get(0)?.tan()),
                 "exp" => Some(arg_vals.get(0)?.exp()),
                 "ln" => Some(arg_vals.get(0)?.ln()),
                 "sqrt" => Some(arg_vals.get(0)?.sqrt()),
                 "abs" => Some(arg_vals.get(0)?.abs()),
                 _ => None,
             }
        },
        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Infinity => Some(f64::INFINITY),
            cas_ast::Constant::Undefined => Some(f64::NAN),
        }
    }
}

struct LocalSimplificationTransformer<'a> {
    context: &'a mut Context,
    rules: &'a HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: &'a Vec<Rc<dyn Rule>>,
    disabled_rules: &'a HashSet<String>,
    collect_steps: bool,
    steps: Vec<Step>,
    cache: HashMap<ExprId, ExprId>,
    current_path: Vec<crate::step::PathStep>,
    profiler: &'a mut RuleProfiler,
}

use cas_ast::visitor::Transformer;

impl<'a> Transformer for LocalSimplificationTransformer<'a> {
    fn transform_expr(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        self.transform_expr_recursive(id)
    }
}

impl<'a> LocalSimplificationTransformer<'a> {
    fn indent(&self) -> String {
        "  ".repeat(self.current_path.len())
    }

    fn transform_expr_recursive(&mut self, id: ExprId) -> ExprId {
        // Use tracing for debug logging
        let expr = self.context.get(id);
        debug!("{}[DEBUG] Visiting: {:?}", self.indent(), expr);

        // println!("Visiting: {:?} {:?}", id, self.context.get(id));
        // println!("Simplifying: {:?}", id);
        if let Some(&cached) = self.cache.get(&id) {
            return cached;
        }

        // 1. Simplify children first (Bottom-Up)
        let expr = self.context.get(id).clone();
        
        let expr_with_simplified_children = match expr {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => id,
            Expr::Add(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();
                
                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();
                
                if new_l != l || new_r != r { 
                    self.context.add(Expr::Add(new_l, new_r)) 
                } else { 
                    id 
                }
            },
            Expr::Sub(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();
                
                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();
                
                if new_l != l || new_r != r { self.context.add(Expr::Sub(new_l, new_r)) } else { id }
            },
            Expr::Mul(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();
                
                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();
                
                if new_l != l || new_r != r { self.context.add(Expr::Mul(new_l, new_r)) } else { id }
            },
            Expr::Div(l, r) => {
                self.current_path.push(crate::step::PathStep::Left);
                let new_l = self.transform_expr_recursive(l);
                self.current_path.pop();
                
                self.current_path.push(crate::step::PathStep::Right);
                let new_r = self.transform_expr_recursive(r);
                self.current_path.pop();
                
                if new_l != l || new_r != r { self.context.add(Expr::Div(new_l, new_r)) } else { id }
            },
            Expr::Pow(b, e) => {
                self.current_path.push(crate::step::PathStep::Base);
                let new_b = self.transform_expr_recursive(b);
                self.current_path.pop();
                
                self.current_path.push(crate::step::PathStep::Exponent);
                let new_e = self.transform_expr_recursive(e);
                self.current_path.pop();
                
                if new_b != b || new_e != e { self.context.add(Expr::Pow(new_b, new_e)) } else { id }
            },
            Expr::Neg(e) => {
                self.current_path.push(crate::step::PathStep::Inner);
                let new_e = self.transform_expr_recursive(e);
                self.current_path.pop();
                
                if new_e != e { self.context.add(Expr::Neg(new_e)) } else { id }
            },
            Expr::Function(name, args) => {
                let mut new_args = Vec::new();
                let mut changed = false;
                for (i, arg) in args.iter().enumerate() {
                    self.current_path.push(crate::step::PathStep::Arg(i));
                    let new_arg = self.transform_expr_recursive(*arg);
                    self.current_path.pop();
                    
                    if new_arg != *arg { changed = true; }
                    new_args.push(new_arg);
                }
                if changed { self.context.add(Expr::Function(name, new_args)) } else { id }
            },
        };

        // 2. Apply rules
        let result = self.apply_rules(expr_with_simplified_children);
        self.cache.insert(id, result);
        result
    }

    fn apply_rules(&mut self, mut expr_id: ExprId) -> ExprId {
        loop {
            let mut changed = false;
            let variant = get_variant_name(self.context.get(expr_id));
            // println!("apply_rules for {:?} variant: {}", expr_id, variant);
            // Try specific rules
            if let Some(specific_rules) = self.rules.get(variant) {
                for rule in specific_rules {
                    if self.disabled_rules.contains(rule.name()) {
                        continue;
                    }
                    if let Some(rewrite) = rule.apply(self.context, expr_id) {
                        // Record rule application for profiling
                        self.profiler.record(rule.name());
                        
                        println!("Rule '{}' applied: {:?} -> {:?}", rule.name(), expr_id, rewrite.new_expr);
                        debug!("{}[DEBUG] Rule '{}' applied: {:?} -> {:?}", self.indent(), rule.name(), expr_id, rewrite.new_expr);
                        if self.collect_steps {
                            self.steps.push(Step::new(
                                &rewrite.description,
                                rule.name(),
                                expr_id,
                                rewrite.new_expr,
                                self.current_path.clone(),
                                Some(self.context),
                            ));
                        }
                        expr_id = rewrite.new_expr;
                        changed = true;
                        break; 
                    }
                }
            }
            
            if changed { 
                return self.transform_expr_recursive(expr_id);
            }

            // Try global rules
            for rule in self.global_rules {
                if self.disabled_rules.contains(rule.name()) {
                    continue;
                }
                if let Some(rewrite) = rule.apply(self.context, expr_id) {
                    // Record rule application for profiling
                    self.profiler.record(rule.name());
                    
                    debug!("{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}", self.indent(), rule.name(), expr_id, rewrite.new_expr);
                    if self.collect_steps {
                        self.steps.push(Step::new(
                            &rewrite.description,
                            rule.name(),
                            expr_id,
                            rewrite.new_expr,
                            self.current_path.clone(),
                            Some(self.context),
                        ));
                    }
                    expr_id = rewrite.new_expr;
                    changed = true;
                    break; 
                }
            }
            
            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            return expr_id;
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
