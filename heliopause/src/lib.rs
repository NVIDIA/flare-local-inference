//! CPython preparation hooks and native ownership at the package boundary.

pub use helios_core::*;

mod allocator;
mod embedding_tables;
mod export;
mod lifetime;
#[cfg_attr(target_os = "windows", path = "processes_windows.rs")]
mod processes;
mod python;
