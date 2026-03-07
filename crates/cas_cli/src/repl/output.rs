//! Output types for ReplCore - structured messages instead of direct printing.
//!
//! This allows the REPL logic to be decoupled from I/O, making it testable
//! and reusable for web/TUI/API contexts.

mod adapters;
mod core_result;
mod messages;

pub use self::adapters::*;
pub use self::core_result::*;
pub use self::messages::*;
