//! `x86_64`-specific accelerated code.

#[cfg(all(feature = "x86_ssse3", target_arch = "x86_64"))]
pub use crate::decode::ssse3::Ssse3;
#[cfg(all(feature = "x86_sse41", target_arch = "x86_64"))]
pub use crate::encode::sse41::Sse41;
