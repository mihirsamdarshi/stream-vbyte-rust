//! Encode and decode `u32`s with the Stream VByte format.
//!
//! ```
//! use stream_vbyte::*;
//!
//! let nums: Vec<u32> = (0..12345).collect();
//! let mut encoded_data = Vec::new();
//! // make some space to encode into
//! encoded_data.resize(5 * nums.len(), 0x0);
//!
//! // use Scalar implementation that works on any hardware
//! let encoded_len = encode::<Scalar>(&nums, &mut encoded_data);
//! println!("Encoded {} u32s into {} bytes", nums.len(), encoded_len);
//!
//! let mut decoded_nums = Vec::new();
//! decoded_nums.resize(nums.len(), 0);
//! // now decode
//! decode::<Scalar>(&encoded_data, nums.len(), &mut decoded_nums);
//!
//! assert_eq!(nums, decoded_nums);
//!
//! ```
//!
extern crate byteorder;

use std::cmp;
use byteorder::{ByteOrder, LittleEndian};

mod tables;
mod cursor;
pub use cursor::DecodeCursor;

#[cfg(feature = "x86_ssse3")]
pub mod x86;

pub trait Encoder {
    /// Encode complete quads of input numbers.
    /// `control_bytes` will be exactly as long as the number of complete 4-number quads in `input`.
    /// Control bytes are written to `control_bytes` and encoded numbers to `encoded_nums`.
    /// Returns the total bytes written to `encoded_nums`.
    fn encode_quads(input: &[u32], control_bytes: &mut [u8], encoded_nums: &mut [u8]) -> usize;
}

pub trait Decoder {
    /// Decode encoded numbers in complete quads. Only control bytes that have all 4 lengths set
    /// and their corresponding 4 encoded numbers will be provided (i.e. no trailing partial quad).
    ///
    /// `control_bytes` is the control bytes that correspond to `encoded_nums`.
    /// `output` is the buffer to write decoded numbers into.
    /// `control_bytes_to_decode * 4` must be no greater than the length of `output`. It may
    /// be greater than the number of control bytes remaining.
    ///
    /// Implementations should decode up to `control_bytes_to_decode` control bytes, but may decode
    /// fewer.
    ///
    /// Returns the number of numbers decoded and the number of bytes read from
    /// `encoded_nums`.
    fn decode_quads(control_bytes: &[u8], encoded_nums: &[u8], output: &mut [u32],
                    control_bytes_to_decode: usize) -> (usize, usize);
}

/// Regular ol' byte shuffling.
/// Works on every platform, but it's not the quickest.
pub struct Scalar;

impl Encoder for Scalar {
    fn encode_quads(input: &[u32], control_bytes: &mut [u8], encoded_nums: &mut [u8]) -> usize {
        let mut bytes_written = 0;
        let mut nums_encoded = 0;

        for quads_encoded in 0..control_bytes.len() {
            let num0 = input[nums_encoded];
            let num1 = input[nums_encoded + 1];
            let num2 = input[nums_encoded + 2];
            let num3 = input[nums_encoded + 3];

            let len0 = encode_num_scalar(num0, &mut encoded_nums[bytes_written..]);
            let len1 = encode_num_scalar(num1, &mut encoded_nums[bytes_written + len0..]);
            let len2 = encode_num_scalar(num2, &mut encoded_nums[bytes_written + len0 + len1..]);
            let len3 = encode_num_scalar(num3, &mut encoded_nums[bytes_written + len0 + len1 + len2..]);

            // this is a few percent faster in my testing than using control_bytes.iter_mut()
            control_bytes[quads_encoded] = ((len0 - 1) | (len1 - 1) << 2 | (len2 - 1) << 4 | (len3 - 1) << 6) as u8;

            bytes_written += len0 + len1 + len2 + len3;
            nums_encoded += 4;
        }

        bytes_written
    }
}

impl Decoder for Scalar {
    fn decode_quads(control_bytes: &[u8], encoded_nums: &[u8], output: &mut [u32],
                    control_bytes_to_decode: usize) -> (usize, usize) {
        debug_assert!(control_bytes_to_decode * 4 <= output.len());

        let mut bytes_read: usize = 0;
        let mut nums_decoded: usize = 0;
        let control_byte_limit = cmp::min(control_bytes.len(), control_bytes_to_decode);

        for &control_byte in control_bytes[0..control_byte_limit].iter() {
            let (len0, len1, len2, len3) = tables::DECODE_LENGTH_PER_NUM_TABLE[control_byte as usize];
            let len0 = len0 as usize;
            let len1 = len1 as usize;
            let len2 = len2 as usize;
            let len3 = len3 as usize;

            output[nums_decoded] = decode_num_scalar(len0, &encoded_nums[bytes_read..]);
            output[nums_decoded + 1] = decode_num_scalar(len1, &encoded_nums[bytes_read + len0..]);
            output[nums_decoded + 2] = decode_num_scalar(len2, &encoded_nums[bytes_read + len0 + len1..]);
            output[nums_decoded + 3] = decode_num_scalar(len3, &encoded_nums[bytes_read + len0 + len1 + len2..]);

            bytes_read += len0 + len1 + len2 + len3;
            nums_decoded += 4;
        }

        (nums_decoded, bytes_read)
    }
}

/// Encode the `input` slice into the `output` slice.
///
/// If you don't have specific knowledge of the input that would let you determine the encoded
/// length ahead of time, make `output` 5x as long as `input`. The worst-case encoded length is 4
/// bytes per `u32` plus another byte for every 4 `u32`s, including any trailing partial 4-some.
///
/// Returns the number of bytes written to the `output` slice.
pub fn encode<E: Encoder>(input: &[u32], output: &mut [u8]) -> usize {
    if input.len() == 0 {
        return 0;
    }

    let shape = encoded_shape(input.len());

    let (control_bytes, encoded_bytes) = output.split_at_mut(shape.control_bytes_len);

    let mut num_bytes_written = E::encode_quads(&input[..],
                                                &mut control_bytes[0..shape.complete_control_bytes_len],
                                                &mut encoded_bytes[..]);

    // last control byte, if there were leftovers
    if shape.leftover_numbers > 0 {
        let mut control_byte = 0;
        let mut nums_encoded = shape.complete_control_bytes_len * 4;

        for i in 0..shape.leftover_numbers {
            let num = input[nums_encoded];
            let len = encode_num_scalar(num, &mut encoded_bytes[num_bytes_written..]);

            control_byte |= ((len - 1) as u8) << (i * 2);

            num_bytes_written += len;
            nums_encoded += 1;
        }
        control_bytes[shape.complete_control_bytes_len] = control_byte;
    }

    control_bytes.len() + num_bytes_written
}

/// Decode `count` numbers from `input`, writing them to `output`. The `count` must be the same
/// as the number of items originally encoded.
///
/// `output` must be at least of size 4, and must be large enough for all `count` numbers.
///
/// Returns the number of bytes read from `input`.
pub fn decode<D: Decoder>(input: &[u8], count: usize, output: &mut [u32]) -> usize {
    let mut cursor = DecodeCursor::new(&input, count);

    assert_eq!(count, cursor.decode::<D>(output), "output buffer was not large enough");

    cursor.input_consumed()
}

#[derive(Debug, PartialEq)]
struct EncodedShape {
    control_bytes_len: usize,
    complete_control_bytes_len: usize,
    leftover_numbers: usize
}

fn encoded_shape(count: usize) -> EncodedShape {
    EncodedShape {
        control_bytes_len: (count + 3) / 4,
        complete_control_bytes_len: count / 4,
        leftover_numbers: count % 4
    }
}

fn encode_num_scalar(num: u32, output: &mut [u8]) -> usize {
    // this will calculate 0_u32 as taking 0 bytes, so ensure at least 1 byte
    let len = cmp::max(1_usize, 4 - num.leading_zeros() as usize / 8);
    let mut buf = [0_u8; 4];
    LittleEndian::write_u32(&mut buf, num);

    for i in 0..len {
        output[i] = buf[i];
    }

    len
}

fn decode_num_scalar(len: usize, input: &[u8]) -> u32 {
    let mut buf = [0_u8; 4];
    &buf[0..len].copy_from_slice(&input[0..len]);

    LittleEndian::read_u32(&buf)
}


fn cumulative_encoded_len(control_bytes: &[u8]) -> usize {
    // sum could only overflow with an invalid encoding because the sum can be no larger than
    // the complete length of the encoded data, which fits in a usize
    control_bytes.iter()
        .map({ |&b| tables::DECODE_LENGTH_PER_QUAD_TABLE[b as usize] as usize })
        .sum()
}

#[cfg(test)]
mod tests;
