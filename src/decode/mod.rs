use crate::scalar;

pub mod cursor;

#[cfg(feature = "x86_ssse3")]
pub mod ssse3;

#[cfg(feature = "aarch64_neon")]
pub mod neon;

#[cfg(test)]
mod tests;

#[cfg(any(
    not(any(feature = "x86_ssse3", feature = "aarch64_neon")),
    all(feature = "x86_ssse3", feature = "aarch64_neon")
))]
pub type StreamVbyteDecoder = scalar::Scalar;

#[cfg(all(feature = "x86_ssse3", not(feature = "aarch64_neon")))]
pub type StreamVbyteDecoder = ssse3::Ssse3;

#[cfg(all(feature = "aarch64_neon", not(feature = "x86_ssse3")))]
pub type StreamVbyteDecoder = neon::NeonDecoder;

/// Decode bytes to numbers.
pub trait Decoder {
    type DecodedQuad;

    /// Decode encoded numbers in complete quads.
    ///
    /// Only control bytes with 4 corresponding encoded numbers will be provided
    /// as input (i.e. no trailing partial quad).
    ///
    /// `control_bytes` are the control bytes that correspond to `encoded_nums`.
    ///
    /// `max_control_bytes_to_decode` may be greater than the number of control
    /// bytes remaining, in which case only the remaining control bytes will
    /// be decoded.
    ///
    /// Implementations may decode at most `max_control_bytes_to_decode` control
    /// bytes, but may decode fewer.
    ///
    /// `nums_already_decoded` is the number of numbers that have already been
    /// decoded in the `DecodeCursor.decode` invocation.
    ///
    /// Returns a tuple of the number of numbers decoded (always a multiple of
    /// 4; at most `4 * max_control_bytes_to_decode`) and the number of
    /// bytes read from `encoded_nums`.
    fn decode_quads<S: DecodeQuadSink<Self>>(
        control_bytes: &[u8],
        encoded_nums: &[u8],
        max_control_bytes_to_decode: usize,
        nums_already_decoded: usize,
        sink: &mut S,
    ) -> (usize, usize);
}

/// For decoders that wish to support slice-based features like the top-level
/// `decode()` or DecodeCursor's `decode_slice()`.
pub trait WriteQuadToSlice: Decoder {
    /// Write a quad into a size-4 slice.
    fn write_quad_to_slice(quad: Self::DecodedQuad, slice: &mut [u32]);
}

/// Receives numbers decoded via a Decoder in `DecodeCursor.decode_sink()` that
/// weren't handed to `DecodeQuadSink.on_quad()`, whether because the `Decoder`
/// implementation doesn't have a natural quad representation, or because the
/// numbers are part of a trailing partial quad.
pub trait DecodeSingleSink {
    /// `nums_decoded` is the number of numbers that have already been decoded
    /// before this number in the current invocation of
    /// `DecodeCursor.decode_sink()`.
    fn on_number(&mut self, num: u32, nums_decoded: usize);
}

/// Receives numbers decoded via a Decoder in `DecodeCursor.decode_sink()`.
///
/// Since stream-vbyte is oriented around groups of 4 numbers, some decoders
/// will expose decoded numbers in some decoder-specific datatype. Or, if that
/// is not applicable for a particular `Decoder` implementation, all decoded
/// numbers will instead be passed to `DecodeSingleSink.on_number()`.
pub trait DecodeQuadSink<D: Decoder + ?Sized>: DecodeSingleSink {
    /// `nums_decoded` is the number of numbers that have already been decoded
    /// before this quad in the current invocation of
    /// `DecodeCursor.decode_sink()`.
    fn on_quad(&mut self, quad: D::DecodedQuad, nums_decoded: usize);
}

/// A sink for writing to a slice.
pub(crate) struct SliceDecodeSink<'a> {
    output: &'a mut [u32],
}

impl<'a> SliceDecodeSink<'a> {
    /// Create a new sink that wraps a slice.
    ///
    /// `output` must be at least as big as the
    fn new(output: &'a mut [u32]) -> SliceDecodeSink<'a> {
        SliceDecodeSink { output }
    }
}

impl<'a> DecodeSingleSink for SliceDecodeSink<'a> {
    #[inline]
    fn on_number(&mut self, num: u32, nums_decoded: usize) {
        self.output[nums_decoded] = num;
    }
}

impl<'a, D: Decoder + WriteQuadToSlice> DecodeQuadSink<D> for SliceDecodeSink<'a> {
    fn on_quad(&mut self, quad: D::DecodedQuad, nums_decoded: usize) {
        D::write_quad_to_slice(quad, &mut self.output[nums_decoded..(nums_decoded + 4)]);
    }
}

/// Decode `count` numbers from `input`, writing them to `output`.
///
/// The `count` must be the same as the number of items originally encoded.
///
/// `output` must be at least of size 4, and must be large enough for all
/// `count` numbers.
///
/// Returns the number of bytes read from `input`.
pub fn decode<D: Decoder + WriteQuadToSlice>(
    input: &[u8],
    count: usize,
    output: &mut [u32],
) -> usize {
    let mut cursor = cursor::DecodeCursor::new(&input, count);

    assert_eq!(
        count,
        cursor.decode_slice::<D>(output),
        "output buffer was not large enough"
    );

    cursor.input_consumed()
}

#[inline]
pub fn decode_num_scalar(len: usize, input: &[u8]) -> u32 {
    let mut buf = [0_u8; 4];
    buf[0..len].copy_from_slice(&input[0..len]);

    u32::from_le_bytes(buf)
}
