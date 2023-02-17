use crate::{
    decode::{decode_num_scalar, DecodeQuadSink, Decoder, WriteQuadToSlice},
    encode::{encode_num_scalar, Encoder},
    tables,
};

/// Encoder/Decoder that works on every platform, at the cost of speed compared
/// to the SIMD accelerated versions.
pub struct Scalar;

impl Encoder for Scalar {
    // This implementation encodes all provided input numbers.
    fn encode_quads(
        input: &[u32],
        control_bytes: &mut [u8],
        encoded_nums: &mut [u8],
    ) -> (usize, usize) {
        let mut bytes_written = 0;
        let mut nums_encoded = 0;

        for quads_encoded in control_bytes {
            let num0 = input[nums_encoded];
            let num1 = input[nums_encoded + 1];
            let num2 = input[nums_encoded + 2];
            let num3 = input[nums_encoded + 3];

            let len0 = encode_num_scalar(num0, &mut encoded_nums[bytes_written..]);
            let len1 = encode_num_scalar(num1, &mut encoded_nums[bytes_written + len0..]);
            let len2 = encode_num_scalar(num2, &mut encoded_nums[bytes_written + len0 + len1..]);
            let len3 = encode_num_scalar(
                num3,
                &mut encoded_nums[bytes_written + len0 + len1 + len2..],
            );

            // this is a few percent faster in my testing than using
            // control_bytes.iter_mut()
            *quads_encoded =
                ((len0 - 1) | (len1 - 1) << 2 | (len2 - 1) << 4 | (len3 - 1) << 6) as u8;

            bytes_written += len0 + len1 + len2 + len3;
            nums_encoded += 4;
        }

        (nums_encoded, bytes_written)
    }
}

impl Decoder for Scalar {
    // Quads are decoded one at a time anyway so no need to bundle them up only to
    // un-bundle them. Instead, we just call on_number for each decoded number.
    type DecodedQuad = UnusedQuad;

    // This implementation decodes all provided encoded data.
    fn decode_quads<S: DecodeQuadSink<Self>>(
        control_bytes: &[u8],
        encoded_nums: &[u8],
        control_bytes_to_decode: usize,
        nums_already_decoded: usize,
        sink: &mut S,
    ) -> (usize, usize) {
        let mut bytes_read: usize = 0;
        let mut nums_decoded: usize = nums_already_decoded;
        let control_byte_limit = std::cmp::min(control_bytes.len(), control_bytes_to_decode);

        for &control_byte in control_bytes[0..control_byte_limit].iter() {
            let (len0, len1, len2, len3) =
                tables::DECODE_LENGTH_PER_NUM_TABLE[control_byte as usize];
            let len0 = len0 as usize;
            let len1 = len1 as usize;
            let len2 = len2 as usize;
            let len3 = len3 as usize;

            sink.on_number(
                decode_num_scalar(len0, &encoded_nums[bytes_read..]),
                nums_decoded,
            );
            sink.on_number(
                decode_num_scalar(len1, &encoded_nums[bytes_read + len0..]),
                nums_decoded + 1,
            );
            sink.on_number(
                decode_num_scalar(len2, &encoded_nums[bytes_read + len0 + len1..]),
                nums_decoded + 2,
            );
            sink.on_number(
                decode_num_scalar(len3, &encoded_nums[bytes_read + len0 + len1 + len2..]),
                nums_decoded + 3,
            );

            bytes_read += len0 + len1 + len2 + len3;
            nums_decoded += 4;
        }

        (nums_decoded - nums_already_decoded, bytes_read)
    }
}

impl WriteQuadToSlice for Scalar {
    fn write_quad_to_slice(_quad: Self::DecodedQuad, _slice: &mut [u32]) {
        // scalar decoding doesn't use quads, so this will never be called
        unreachable!()
    }
}

/// `Scalar` decoder produces numbers one by one, so there is no quad to
/// unbundle. Any implementations of `DecodedQuadSink<EmptyQuad>` can safely use
/// `unreachable!()` or equivalent.
pub struct UnusedQuad;

/// The Scalar decoder doesn't use quads, but the type checker requires that
/// there be a `DecodeQuadSink<Scalar>` impl for a sink nonetheless. This macro
/// will generate an appropriate stub impl for a sink type.
#[macro_export]
macro_rules! decode_quad_scalar {
    ($sink:ty) => {
        impl $crate::decode::DecodeQuadSink<stream_vbyte::scalar::Scalar> for $sink {
            fn on_quad(&mut self, _: stream_vbyte::scalar::UnusedQuad, _: usize) {
                unreachable!()
            }
        }
    };
}
