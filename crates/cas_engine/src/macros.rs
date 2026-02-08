#[macro_export]
macro_rules! define_rule {
    // Full form with targets and phase
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr, // Option<Vec<&str>>
        $phase:expr,   // PhaseMask
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }
        }
    };
    // Full form with targets, phase, AND importance
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr, // Option<Vec<&str>>
        $phase:expr,   // PhaseMask
        importance: $importance:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }
        }
    };
    // Form with phase but no targets
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        phase: $phase:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            None,
            $phase,
            | $ctx, $arg | $body
        );
    };
    // Form with targets but no phase (default: CORE | POST)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }
        }
    };
    // Simplest form: no targets, no phase (default: CORE | POST)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            None,
            | $ctx, $arg | $body
        );
    };

    // =========================================================================
    // DOMAIN-AWARE FORMS (3-arg closures with parent_ctx access)
    // =========================================================================

    // Full form with targets, phase, AND parent_ctx access
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr, // Option<Vec<&str>>
        $phase:expr,   // PhaseMask
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                // This rule uses apply_with_context, not apply_simple
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }
        }
    };

    // Full form with targets, phase, importance AND parent_ctx access
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr, // Option<Vec<&str>>
        $phase:expr,   // PhaseMask
        importance: $importance:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                // This rule uses apply_with_context, not apply_simple
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }
        }
    };

    // Domain-aware form with targets but no phase (default: CORE | POST)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            $targets,
            $crate::phase::PhaseMask::CORE | $crate::phase::PhaseMask::POST,
            | $ctx, $arg, $parent_ctx | $body
        );
    };

    // Simplest domain-aware form: no targets, no phase
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            None,
            $crate::phase::PhaseMask::CORE | $crate::phase::PhaseMask::POST,
            | $ctx, $arg, $parent_ctx | $body
        );
    };

    // Domain-aware form with importance, no targets, no phase
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        importance: $importance:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $crate::define_rule!(
            $(#[$meta])*
            $struct_name,
            $name_str,
            None,
            $crate::phase::PhaseMask::CORE | $crate::phase::PhaseMask::POST,
            importance: $importance,
            | $ctx, $arg, $parent_ctx | $body
        );
    };

    // Simple 2-arg form with importance (no targets, no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        importance: $importance:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }
        }
    };

    // =========================================================================
    // SOLVE_SAFETY FORMS (domain-aware with solve_safety classification)
    // =========================================================================

    // Domain-aware form with solve_safety, no targets, no phase
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        solve_safety: $safety:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn solve_safety(&self) -> $crate::solve_safety::SolveSafety {
                $safety
            }
        }
    };

    // Domain-aware form with solve_safety AND importance
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        solve_safety: $safety:expr,
        importance: $importance:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }

            fn solve_safety(&self) -> $crate::solve_safety::SolveSafety {
                $safety
            }
        }
    };

    // Simple 2-arg form with solve_safety (no parent_ctx, no targets)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        solve_safety: $safety:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn solve_safety(&self) -> $crate::solve_safety::SolveSafety {
                $safety
            }
        }
    };

    // Domain-aware form with targets AND solve_safety (no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr,
        solve_safety: $safety:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn solve_safety(&self) -> $crate::solve_safety::SolveSafety {
                $safety
            }
        }
    };

    // =========================================================================
    // PRIORITY FORMS
    // =========================================================================

    // Simple 2-arg form with priority only (no targets, no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        priority: $priority:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn priority(&self) -> i32 {
                $priority
            }
        }
    };

    // Simple 2-arg form with priority + importance (no targets, no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        priority: $priority:expr,
        importance: $importance:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn priority(&self) -> i32 {
                $priority
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }
        }
    };

    // Domain-aware 3-arg form with priority only (no targets, no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        priority: $priority:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn priority(&self) -> i32 {
                $priority
            }
        }
    };

    // Domain-aware 3-arg form with priority + importance (no targets, no phase)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        priority: $priority:expr,
        importance: $importance:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn priority(&self) -> i32 {
                $priority
            }

            fn importance(&self) -> $crate::step::ImportanceLevel {
                $importance
            }
        }
    };

    // Full form with targets, phase, AND priority (2-arg)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr,
        $phase:expr,
        priority: $priority:expr,
        | $ctx:ident, $arg:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, $ctx: &mut cas_ast::Context, $arg: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }

            fn priority(&self) -> i32 {
                $priority
            }
        }
    };

    // Full form with targets, phase, AND priority (3-arg, domain-aware)
    (
        $(#[$meta:meta])*
        $struct_name:ident,
        $name_str:expr,
        $targets:expr,
        $phase:expr,
        priority: $priority:expr,
        | $ctx:ident, $arg:ident, $parent_ctx:ident | $body:block
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::rule::SimpleRule for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn apply_simple(&self, _ctx: &mut cas_ast::Context, _expr: cas_ast::ExprId) -> Option<$crate::rule::Rewrite> {
                unreachable!("This rule uses apply_with_context")
            }

            fn apply_with_context(
                &self,
                $ctx: &mut cas_ast::Context,
                $arg: cas_ast::ExprId,
                $parent_ctx: &$crate::parent_context::ParentContext,
            ) -> Option<$crate::rule::Rewrite> {
                $body
            }

            fn target_types(&self) -> Option<Vec<&str>> {
                $targets
            }

            fn allowed_phases(&self) -> $crate::phase::PhaseMask {
                $phase
            }

            fn priority(&self) -> i32 {
                $priority
            }
        }
    };
}
