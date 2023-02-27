//! `aarch64`-specific accelerated code.

#[cfg(all(feature = "aarch64_neon", target_arch = "aarch64"))]
pub use crate::decode::neon::NeonDecoder;
#[cfg(all(feature = "aarch64_neon", target_arch = "aarch64"))]
pub use crate::encode::neon::NeonEncoder;
