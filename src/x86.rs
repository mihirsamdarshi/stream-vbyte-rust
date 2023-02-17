//! x86-specific accelerated code.

#[cfg(feature = "x86_ssse3")]
pub use crate::decode::ssse3::Ssse3;
#[cfg(feature = "x86_sse41")]
pub use crate::encode::sse41::Sse41;
