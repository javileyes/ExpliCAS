//! Scoped display transforms for context-aware rendering.
//!
//! This module provides the infrastructure for transforming expressions during
//! display based on the current evaluation context (scope). For example,
//! `x^(1/2)` can be displayed as `sqrt(x)` only when rendering quadratic formula results.
//!
//! The key principle: **canonical form is never changed**. These are display-only transforms.

use crate::{Context, Expr, ExprId};

// =============================================================================
// ScopeTag: Labels for evaluation context
// =============================================================================

/// Tags that identify the current evaluation/display context.
/// Used to determine which display transforms should be active.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScopeTag {
    /// A simplification rule produced this result (e.g., "QuadraticFormula")
    Rule(&'static str),
    /// Solver strategy produced this result (e.g., "isolate")
    Solver(&'static str),
    /// REPL command context (e.g., "solve")
    Command(&'static str),
}

// =============================================================================
// DisplayTransform: Trait for render-time transformations
// =============================================================================

/// A display-only transformation that can intercept expression rendering.
/// These never modify the AST - they only affect the output string.
pub trait DisplayTransform: Send + Sync {
    /// Unique name for this transform (for debugging/logging)
    fn name(&self) -> &'static str;

    /// Check if this transform should be active given the current scopes
    fn applies(&self, scopes: &[ScopeTag]) -> bool;

    /// Try to render an expression. Returns Some(string) if handled, None to fallback.
    /// The `render_child` callback allows recursive rendering with the same transforms.
    fn try_render(
        &self,
        ctx: &Context,
        id: ExprId,
        render_child: &dyn Fn(ExprId) -> String,
    ) -> Option<String>;
}

// =============================================================================
// DisplayTransformRegistry: Collection of active transforms
// =============================================================================

/// Registry holding all available display transforms.
#[derive(Default)]
pub struct DisplayTransformRegistry {
    transforms: Vec<Box<dyn DisplayTransform>>,
}

impl DisplayTransformRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Create registry with default transforms enabled
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(HalfPowerAsSqrt));
        registry
    }

    /// Register a new transform
    pub fn register(&mut self, transform: Box<dyn DisplayTransform>) {
        self.transforms.push(transform);
    }

    /// Get transforms that apply to the given scopes
    pub fn active_for(&self, scopes: &[ScopeTag]) -> Vec<&dyn DisplayTransform> {
        self.transforms
            .iter()
            .filter(|t| t.applies(scopes))
            .map(|t| t.as_ref())
            .collect()
    }
}

// =============================================================================
// ScopedRenderer: Recursive renderer with transforms
// =============================================================================

/// Renderer that applies scoped transforms during expression rendering.
pub struct ScopedRenderer<'a> {
    pub ctx: &'a Context,
    pub scopes: &'a [ScopeTag],
    active_transforms: Vec<&'a dyn DisplayTransform>,
}

impl<'a> ScopedRenderer<'a> {
    /// Create a new scoped renderer with transforms filtered by active scopes
    pub fn new(
        ctx: &'a Context,
        scopes: &'a [ScopeTag],
        registry: &'a DisplayTransformRegistry,
    ) -> Self {
        let active_transforms = registry.active_for(scopes);
        Self {
            ctx,
            scopes,
            active_transforms,
        }
    }

    /// Render an expression, applying any active transforms
    pub fn render(&self, id: ExprId) -> String {
        // Try each active transform
        for transform in &self.active_transforms {
            if let Some(rendered) = transform.try_render(self.ctx, id, &|child| self.render(child))
            {
                return rendered;
            }
        }

        // Fallback to standard display
        crate::DisplayExpr {
            context: self.ctx,
            id,
        }
        .to_string()
    }
}

// =============================================================================
// HalfPowerAsSqrt: Transform ^(1/2) to sqrt() in quadratic context
// =============================================================================

/// Transform that displays `x^(1/2)` as `sqrt(x)` when in QuadraticFormula context.
pub struct HalfPowerAsSqrt;

impl DisplayTransform for HalfPowerAsSqrt {
    fn name(&self) -> &'static str {
        "HalfPowerAsSqrt"
    }

    fn applies(&self, scopes: &[ScopeTag]) -> bool {
        scopes
            .iter()
            .any(|s| matches!(s, ScopeTag::Rule("QuadraticFormula")))
    }

    fn try_render(
        &self,
        ctx: &Context,
        id: ExprId,
        render_child: &dyn Fn(ExprId) -> String,
    ) -> Option<String> {
        // Match Pow(base, exp) where exp is 1/2
        if let Expr::Pow(base, exp) = ctx.get(id) {
            if is_half(ctx, *exp) {
                let base_str = render_child(*base);
                // Use sqrt() or √ depending on pretty mode
                let prefix = crate::display::unicode_root_prefix(2);
                return Some(format!("{}({})", prefix, base_str));
            }
        }
        // Also handle Neg(Pow(base, 1/2)) -> -sqrt(base)
        if let Expr::Neg(inner) = ctx.get(id) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if is_half(ctx, *exp) {
                    let base_str = render_child(*base);
                    let prefix = crate::display::unicode_root_prefix(2);
                    return Some(format!("-{}({})", prefix, base_str));
                }
            }
        }
        None
    }
}

/// Check if an expression represents the value 1/2
fn is_half(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        // Direct 1/2 as Number
        Expr::Number(n) => *n == num_rational::BigRational::new(1.into(), 2.into()),
        // Div(1, 2)
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                && matches!(ctx.get(*den), Expr::Number(d) if d.is_integer() && *d == num_rational::BigRational::from_integer(2.into()))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_tag_equality() {
        assert_eq!(
            ScopeTag::Rule("QuadraticFormula"),
            ScopeTag::Rule("QuadraticFormula")
        );
        assert_ne!(ScopeTag::Rule("QuadraticFormula"), ScopeTag::Rule("Other"));
    }

    #[test]
    fn test_half_power_applies_only_in_quadratic() {
        let transform = HalfPowerAsSqrt;

        // Should apply with QuadraticFormula scope
        assert!(transform.applies(&[ScopeTag::Rule("QuadraticFormula")]));

        // Should not apply without it
        assert!(!transform.applies(&[]));
        assert!(!transform.applies(&[ScopeTag::Rule("Other")]));
        assert!(!transform.applies(&[ScopeTag::Command("solve")]));
    }

    #[test]
    fn test_is_half_with_div() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half_div = ctx.add(Expr::Div(one, two));

        assert!(is_half(&ctx, half_div), "is_half should match Div(1, 2)");
    }

    #[test]
    fn test_is_half_with_number() {
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));

        assert!(is_half(&ctx, half), "is_half should match Number(1/2)");
    }

    #[test]
    fn test_sqrt_rendering_with_div_exponent() {
        let mut ctx = Context::new();
        let base = ctx.num(3);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two));
        let pow = ctx.add(Expr::Pow(base, half));

        let scopes = vec![ScopeTag::Rule("QuadraticFormula")];
        let registry = DisplayTransformRegistry::with_defaults();
        let renderer = ScopedRenderer::new(&ctx, &scopes, &registry);

        let result = renderer.render(pow);
        // Should be sqrt(3) not 3^(1/2)
        assert!(
            result.contains("sqrt") || result.contains("√"),
            "Expected sqrt notation, got: {}",
            result
        );
    }
}
