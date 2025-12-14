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
}
