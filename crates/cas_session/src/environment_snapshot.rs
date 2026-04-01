use serde::{Deserialize, Serialize};

use crate::env::Environment;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingSnapshot {
    pub name: String,
    pub expr: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBindingSnapshot {
    pub name: String,
    pub params: Vec<String>,
    pub expr: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub bindings: Vec<BindingSnapshot>,
    pub functions: Vec<FunctionBindingSnapshot>,
}

impl EnvironmentSnapshot {
    pub fn from_env(env: &Environment) -> Self {
        Self {
            bindings: env
                .list()
                .into_iter()
                .map(|(name, expr)| BindingSnapshot {
                    name: name.to_string(),
                    expr: expr.index() as u32,
                })
                .collect(),
            functions: env
                .list_functions()
                .into_iter()
                .map(|(name, binding)| FunctionBindingSnapshot {
                    name: name.to_string(),
                    params: binding.params.clone(),
                    expr: binding.expr.index() as u32,
                })
                .collect(),
        }
    }

    pub fn into_env(self) -> Environment {
        let mut env = Environment::new();
        for binding in self.bindings {
            env.set(binding.name, cas_ast::ExprId::from_raw(binding.expr));
        }
        for binding in self.functions {
            env.set_function(
                binding.name,
                binding.params,
                cas_ast::ExprId::from_raw(binding.expr),
            );
        }
        env
    }
}
