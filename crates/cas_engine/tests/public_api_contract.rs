// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_solver as real_cas_solver;

mod cas_solver {
    #[allow(unused_imports)]
    pub use cas_engine::*;

    pub mod session_api {
        pub mod solve {
            #[allow(unused_imports)]
            pub use crate::real_cas_solver::session_api::solve::{
                format_verify_summary_lines, format_verify_summary_lines_with_hints,
            };
        }
    }
}

#[path = "../../cas_solver/tests/public_api_contract.rs"]
mod public_api_contract;
