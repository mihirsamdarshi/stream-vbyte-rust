//! `aarch64`-specific accelerated code.

#[cfg(feature = "aarch64_neon")]
pub use crate::decode::neon::NeonDecoder;
#[cfg(feature = "aarch64_neon")]
pub use crate::encode::neon::NeonEncoder;
