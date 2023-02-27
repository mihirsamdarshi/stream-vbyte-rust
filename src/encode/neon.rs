use std::arch::aarch64::{
    vclzq_u32, vld1_u32, vld1_u8, vld1q_u32, vld1q_u8, vmul_u32, vqsubq_u32, vqtbl1_u8, vqtbl1q_u8,
    vreinterpret_u32_u8, vreinterpretq_u8_u32, vshrq_n_u32, vst1_u32, vst1q_u8,
};

use super::Encoder;
use crate::tables::NEON_ENCODE_SHUFFLE_TABLE;

/// Encoder using SSE4.1 instructions.
pub struct NeonEncoder;

const ONES: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
// multiplicand to achieve shifts by multiplication
const SHIFT: i32 = 3;
const SHIFTS: [i32; 4] = [SHIFT, SHIFT, SHIFT, SHIFT];
// translate 3-bit bytemaps into lane codes. Last 8 will never be used.
// 0 = 1 byte encoded num, 1 = 2 byte, etc.
// These are concatenated into the control byte, and also used to sum to find
// the total length. The ordering of these codes is determined by how the
// bytemap is calculated; see comments below.
#[rustfmt::skip]
const LANECODES: [u8; 16] = [
    0, 3, 2, 3,
    1, 3, 2, 3,
    128, 128, 128, 128,
    128, 128, 128, 128];
// gather high bytes from each lane, 2 copies
const GATHER_LO: [u8; 8] = [12, 8, 4, 0, 12, 8, 4, 0];
// mul-shift magic
// concatenate 2-bit lane codes into high byte
const CONCAT: u32 = 1 | 1 << 10 | 1 << 20 | 1 << 30;
// sum lane codes in high byte
const SUM: u32 = 1 | 1 << 8 | 1 << 16 | 1 << 24;
const AGGREGATORS: [u32; 2] = [CONCAT, SUM];

impl Encoder for NeonEncoder {
    fn encode_quads(input: &[u32], control_bytes: &mut [u8], output: &mut [u8]) -> (usize, usize) {
        let mut nums_encoded: usize = 0;

        let mut code_and_length: [u32; 2] = [0, 0];

        unsafe {
            let inq = vld1q_u32(input.as_ptr() as *const u32);

            let shifts = vld1q_u32(SHIFTS.as_ptr() as *const u32);

            let clzbytes = vshrq_n_u32(vclzq_u32(inq), 3);
            let lanecodes = vqsubq_u32(shifts, clzbytes);

            let lanebytes = vreinterpretq_u8_u32(lanecodes);
            let gather_lo = vld1_u8(GATHER_LO.as_ptr() as *const u8);
            let aggregators = vld1_u32(AGGREGATORS.as_ptr() as *const u32);
            let lobytes = vqtbl1_u8(lanebytes, gather_lo);
            let mulshift = vreinterpret_u32_u8(lobytes);

            vst1_u32(
                code_and_length.as_mut_ptr() as *mut u32,
                vmul_u32(mulshift, aggregators),
            );
        };

        let (code, length) = (code_and_length[0] >> 24, 4 + (code_and_length[1] >> 24));

        unsafe {
            let encoding_shuffle =
                vld1q_u8(NEON_ENCODE_SHUFFLE_TABLE[code as usize].as_ptr() as *const u8);
            let inu = vreinterpretq_u8_u32(vld1q_u32(input.as_ptr() as *const u32));

            vst1q_u8(
                output.as_mut_ptr() as *mut u8,
                vqtbl1q_u8(inu, encoding_shuffle),
            );
        }

        // Encoding writes 16 bytes at a time, but if numbers are encoded with 1 byte
        // each, that means the last 3 quads could write past what is actually
        // necessary. So, don't process the last few control bytes.
        let control_byte_limit = control_bytes.len().saturating_sub(3);

        (nums_encoded, length as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn encodes_all_but_last_3_control_bytes() {
        // cover the whole byte length range
        let nums: Vec<u32> = (0..32).map(|i| 1 << i).collect();
        let mut encoded = Vec::new();
        let mut decoded: Vec<u32> = Vec::new();

        for control_bytes_len in 0..(nums.len() / 4 + 1) {
            encoded.clear();
            encoded.resize(nums.len() * 5, 0xFF);
            decoded.clear();
            decoded.resize(nums.len(), 54321);

            let (nums_encoded, bytes_written) = {
                let (control_bytes, num_bytes) = encoded.split_at_mut(control_bytes_len);

                NeonEncoder::encode_quads(&nums[0..4 * control_bytes_len], control_bytes, num_bytes)
            };

            let control_bytes_written = nums_encoded / 4;

            assert_eq!(
                cumulative_encoded_len(&encoded[0..control_bytes_written]),
                bytes_written
            );

            // the last control byte written may not have populated all 16 output bytes with
            // encoded nums, depending on the length required. Any unused
            // trailing bytes will have had 0 written, but nothing beyond that
            // 16 should be touched.

            let length_before_final_control_byte =
                cumulative_encoded_len(&encoded[0..control_bytes_written.saturating_sub(1)]);

            let bytes_written_for_final_control_byte =
                bytes_written - length_before_final_control_byte;
            let trailing_zero_len = if control_bytes_written > 0 {
                16 - bytes_written_for_final_control_byte
            } else {
                0
            };

            assert!(&encoded[control_bytes_len + bytes_written
                ..control_bytes_len + bytes_written + trailing_zero_len]
                .iter()
                .all(|&i| i == 0));
            assert!(
                &encoded[control_bytes_len + bytes_written + trailing_zero_len..]
                    .iter()
                    .all(|&i| i == 0xFF)
            );
        }
    }
}
