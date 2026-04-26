/// SPX rANS Core — Phase 1 Rust port of core/rans.py
///
/// Covers:
///   • PDF building (histogram → normalized 12-bit frequencies)
///   • Shard-mode selection (mirrors rans_selector._decide_shard_mode_core)
///   • 4-way interleaved rANS encode (parallel over shards via Rayon)
///   • 4-way interleaved rANS decode (single-threaded per shard)
///   • PDF table compact/expand serialization
///   • Shard payload packing
///   • Slot-lookup table construction

use rayon::prelude::*;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const L_LOWER: u64 = 1u64 << 31;
const M_BITS: u64 = 12;
const M_MASK: u64 = (1u64 << M_BITS) - 1;
pub const L_NORM_THRESHOLD: u64 = (L_LOWER >> M_BITS) << 8; // 134_217_728
const HEADER_PENALTY_BITS: f64 = 120.0;

// ---------------------------------------------------------------------------
// MAGIC_LUT  — (2^64-1) / f  for f in 1..=4096
// ---------------------------------------------------------------------------

static MAGIC_LUT: OnceLock<Vec<u64>> = OnceLock::new();

pub fn get_magic_lut() -> &'static [u64] {
    MAGIC_LUT.get_or_init(|| {
        let mut lut = vec![0u64; 4097];
        for f in 1..=4096usize {
            lut[f] = u64::MAX / f as u64;
        }
        lut
    })
}

// ---------------------------------------------------------------------------
// Core arithmetic
// ---------------------------------------------------------------------------

#[inline(always)]
fn mul_hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

// ---------------------------------------------------------------------------
// PDF builder  (mirrors rans_selector.build_pdf_from_counts)
// ---------------------------------------------------------------------------

fn build_pdf_from_counts(counts: &[u64], width: usize) -> Vec<u64> {
    let total: u64 = counts.iter().sum();
    let active = width.min(256);
    let mut pdf = vec![0u64; 256];

    if total == 0 {
        let v = (4096u64 / active as u64).max(1);
        for i in 0..active {
            pdf[i] = v;
        }
        let sum: u64 = pdf[..active].iter().sum();
        let rem = 4096i64 - sum as i64;
        if rem >= 0 {
            pdf[0] = pdf[0].saturating_add(rem as u64);
        } else {
            pdf[0] = pdf[0].saturating_sub((-rem) as u64);
        }
        return pdf;
    }

    let mut cur_sum: i64 = 0;
    for i in 0..active {
        let v_f = (counts[i] as f64 / total as f64) * 4096.0;
        let mut v = v_f.round() as u64;
        if counts[i] > 0 && v == 0 {
            v = 1;
        }
        pdf[i] = v;
        cur_sum += v as i64;
    }

    let diff = 4096i64 - cur_sum;
    let peak = pdf[..active]
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let new_peak = pdf[peak] as i64 + diff;
    if new_peak >= 1 {
        pdf[peak] = new_peak as u64;
    } else {
        pdf[peak] = 1;
        let mut remaining = (1i64 - new_peak) as u64;
        for i in 0..active {
            if remaining == 0 {
                break;
            }
            if i != peak && pdf[i] > 1 {
                let cut = (pdf[i] - 1).min(remaining);
                pdf[i] -= cut;
                remaining -= cut;
            }
        }
    }

    pdf
}

// ---------------------------------------------------------------------------
// Cross-entropy  (mirrors rans_selector.calculate_cross_entropy)
// ---------------------------------------------------------------------------

fn cross_entropy(counts: &[u64], pdf: &[u64]) -> f64 {
    let mut e = 0.0f64;
    for i in 0..256 {
        let c = counts[i];
        if c > 0 {
            let p = pdf[i];
            if p > 0 {
                e += c as f64 * (12.0 - (p as f64).log2());
            } else {
                e += c as f64 * 24.0; // impossible-symbol penalty
            }
        }
    }
    e
}

// ---------------------------------------------------------------------------
// Shard mode selector  (mirrors rans_selector._decide_shard_mode_core)
//
// templates: flat array of shape (n_templates, 256)
// Returns (mode: u8, pdf: Vec<u64>[256])
// ---------------------------------------------------------------------------

pub fn decide_shard_mode(
    counts: &[u64],
    width: usize,
    penalty: f64,
    templates: &[u64],
    n_templates: usize,
    disable_templates: bool,
) -> (u8, Vec<u64>) {
    // Mode 3: mono-symbol (only symbol 0 present)
    let non_zero: u64 = counts[1..].iter().sum();
    if non_zero == 0 {
        let mut mono = vec![0u64; 256];
        mono[0] = 4096;
        return (3, mono);
    }

    let dense = build_pdf_from_counts(counts, width);

    if disable_templates || n_templates == 0 {
        return (0, dense);
    }

    // Evaluate empirical templates (Mode 4-33)
    let mut best_mode = 0u8;
    let mut min_emp = f64::MAX;
    for tid in 0..n_templates {
        let tpl = &templates[tid * 256..(tid + 1) * 256];
        let bits = cross_entropy(counts, tpl);
        if bits < min_emp {
            min_emp = bits;
            best_mode = (4 + tid) as u8;
        }
    }

    let dense_bits = cross_entropy(counts, &dense) + penalty;

    if min_emp < dense_bits {
        let tid = (best_mode - 4) as usize;
        let tpl = &templates[tid * 256..(tid + 1) * 256];
        // Guard: template must not assign zero probability to any present symbol
        for i in 0..256 {
            if counts[i] > 0 && tpl[i] == 0 {
                return (0, dense);
            }
        }
        return (best_mode, tpl.to_vec());
    }

    (0, dense)
}

// ---------------------------------------------------------------------------
// build_pdf_tables_from_shards
//
// Parallel histogram collection + parallel PDF building.
// Returns flat arrays:
//   cum_freqs  : n_shards × 257  (row-major)
//   sym_freqs  : n_shards × 256  (row-major)
//   modes      : n_shards
// ---------------------------------------------------------------------------

pub fn build_pdf_tables_from_shards(
    data: &[u8],
    shard_offsets: &[u32],
    shard_lengths: &[u32],
    shard_widths: &[u16],
    templates: &[u64],
    n_templates: usize,
    disable_templates: bool,
) -> (Vec<u64>, Vec<u64>, Vec<u8>) {
    let n = shard_lengths.len();

    // Step 1 — parallel per-shard histograms
    let mut hists = vec![0u64; n * 256];
    hists
        .chunks_mut(256)
        .zip(shard_offsets.iter().zip(shard_lengths.iter()))
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(hist, (&off, &len))| {
            let start = off as usize;
            for &b in &data[start..start + len as usize] {
                hist[b as usize] += 1;
            }
        });

    // Step 2 — parallel PDF decisions
    let results: Vec<(Vec<u64>, Vec<u64>, u8)> = (0..n)
        .into_par_iter()
        .map(|sid| {
            let hist = &hists[sid * 256..(sid + 1) * 256];
            let width = shard_widths[sid] as usize;
            let total: u64 = hist.iter().sum();

            if total > 0 {
                let (mode, pdf) = decide_shard_mode(
                    hist,
                    width,
                    HEADER_PENALTY_BITS,
                    templates,
                    n_templates,
                    disable_templates,
                );
                let mut cum = vec![0u64; 257];
                let mut acc = 0u64;
                for j in 0..256 {
                    cum[j] = acc;
                    acc += pdf[j];
                }
                cum[256] = acc;
                (pdf, cum, mode)
            } else {
                let mut pdf = vec![0u64; 256];
                pdf[0] = 4096;
                let mut cum = vec![0u64; 257];
                for j in 1..=256 {
                    cum[j] = 4096;
                }
                (pdf, cum, 3u8)
            }
        })
        .collect();

    let mut sym_freqs = vec![0u64; n * 256];
    let mut cum_freqs = vec![0u64; n * 257];
    let mut modes = vec![0u8; n];

    for (sid, (pdf, cum, mode)) in results.into_iter().enumerate() {
        sym_freqs[sid * 256..(sid + 1) * 256].copy_from_slice(&pdf);
        cum_freqs[sid * 257..(sid + 1) * 257].copy_from_slice(&cum);
        modes[sid] = mode;
    }

    (cum_freqs, sym_freqs, modes)
}

// ---------------------------------------------------------------------------
// rANS encode helper
// ---------------------------------------------------------------------------

#[inline(always)]
fn encode_sym(sym: u8, st: &mut u64, cfreqs: &[u64], sfreqs: &[u64], magic: &[u64], bs: &mut Vec<u8>) {
    let s = sym as usize;
    let f = sfreqs[s];
    let cf = cfreqs[s];
    let x_max = L_NORM_THRESHOLD * f;
    while *st >= x_max {
        bs.push((*st & 0xFF) as u8);
        *st >>= 8;
    }
    let q = mul_hi(*st, magic[f as usize]);
    let r = *st - q * f;
    let (q, r) = if r >= f { (q + 1, r - f) } else { (q, r) };
    *st = (q << M_BITS) + cf + r;
}

// ---------------------------------------------------------------------------
// rans_encode_shards_parallel
//
// Parallel rANS encoding (one shard per Rayon worker).
// Returns:
//   final_states  : n_shards × 4  (flat)
//   bitstreams    : flat byte buffer (offsets described by bs_offsets)
//   bs_offsets    : n_shards
//   bs_lengths    : n_shards
// ---------------------------------------------------------------------------

pub fn rans_encode_shards_parallel(
    shard_data: &[u8],
    shard_offsets: &[u32],
    shard_lengths: &[u32],
    all_cum_freqs: &[u64],  // n_shards × 257 flat
    all_sym_freqs: &[u64],  // n_shards × 256 flat
    initial_state: u64,
) -> (Vec<u64>, Vec<u8>, Vec<u32>, Vec<u32>) {
    let n = shard_lengths.len();
    let magic = get_magic_lut();

    // Pre-compute output buffer slot offsets
    let mut bs_offsets = vec![0u32; n];
    let mut off = 0u32;
    for i in 0..n {
        bs_offsets[i] = off;
        // Worst-case expansion: every byte emits 2 output bytes + 1024 header slack
        off = off
            .saturating_add(shard_lengths[i].saturating_mul(2).saturating_add(1024));
    }
    let total_bs = off as usize;

    // Parallel encode — each shard gets its own Vec<u8> then we copy into flat buf
    let results: Vec<([u64; 4], Vec<u8>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let n_val = shard_lengths[i] as usize;
            let sfreqs = &all_sym_freqs[i * 256..(i + 1) * 256];
            let cfreqs = &all_cum_freqs[i * 257..(i + 1) * 257];

            if n_val == 0 || sfreqs[0] == 4096 {
                return ([initial_state; 4], Vec::new());
            }

            let data_start = shard_offsets[i] as usize;
            let mut st = [initial_state; 4];
            let mut bs: Vec<u8> = Vec::with_capacity(n_val * 2 + 256);

            let rem = n_val % 4;
            let tail = n_val - rem;

            // Tail symbols (last `rem` symbols, reverse order, lane = j % 4)
            for j in (tail..n_val).rev() {
                let lane = j % 4;
                encode_sym(shard_data[data_start + j], &mut st[lane], cfreqs, sfreqs, magic, &mut bs);
            }

            // Main 4-way blocks: groups of 4 processed in reverse, within each
            // group encoding lane 3 first then 2, 1, 0
            let mut j = tail as isize - 4;
            while j >= 0 {
                let base = data_start + j as usize;
                encode_sym(shard_data[base + 3], &mut st[3], cfreqs, sfreqs, magic, &mut bs);
                encode_sym(shard_data[base + 2], &mut st[2], cfreqs, sfreqs, magic, &mut bs);
                encode_sym(shard_data[base + 1], &mut st[1], cfreqs, sfreqs, magic, &mut bs);
                encode_sym(shard_data[base + 0], &mut st[0], cfreqs, sfreqs, magic, &mut bs);
                j -= 4;
            }

            (st, bs)
        })
        .collect();

    // Assemble flat output
    let mut final_states = vec![0u64; n * 4];
    let mut bitstreams = vec![0u8; total_bs];
    let mut bs_lengths = vec![0u32; n];

    for (i, (states, bs)) in results.into_iter().enumerate() {
        final_states[i * 4..(i + 1) * 4].copy_from_slice(&states);
        let bl = bs.len();
        let dst = bs_offsets[i] as usize;
        bitstreams[dst..dst + bl].copy_from_slice(&bs);
        bs_lengths[i] = bl as u32;
    }

    (final_states, bitstreams, bs_offsets, bs_lengths)
}

// ---------------------------------------------------------------------------
// rans_decode_4way_core  (single-threaded, called per-shard from Python executor)
// ---------------------------------------------------------------------------

pub fn rans_decode_4way_core(
    st0: u64,
    st1: u64,
    st2: u64,
    st3: u64,
    bitstream: &[u8],
    cum_freqs: &[u64],    // 257 entries
    symbol_freqs: &[u64], // 256 entries
    slot_lookup: &[u8],   // 4096 entries
    out: &mut [u8],
) {
    let target = out.len();
    let mut ptr = bitstream.len() as isize - 1;
    let mut st = [st0, st1, st2, st3];

    let blocks = target / 4;
    let remainder = target % 4;

    // Mono-symbol fast path
    if symbol_freqs[0] == 4096 {
        out.fill(slot_lookup[0]);
        return;
    }

    let mut idx = 0usize;

    macro_rules! step {
        ($lane:expr) => {{
            let s = slot_lookup[(st[$lane] & M_MASK) as usize];
            out[idx] = s;
            idx += 1;
            st[$lane] = symbol_freqs[s as usize] * (st[$lane] >> M_BITS)
                + (st[$lane] & M_MASK)
                - cum_freqs[s as usize];
            if st[$lane] < L_LOWER && ptr >= 0 {
                st[$lane] = (st[$lane] << 8) | bitstream[ptr as usize] as u64;
                ptr -= 1;
                if st[$lane] < L_LOWER && ptr >= 0 {
                    st[$lane] = (st[$lane] << 8) | bitstream[ptr as usize] as u64;
                    ptr -= 1;
                }
            }
        }};
    }

    for _ in 0..blocks {
        step!(0);
        step!(1);
        step!(2);
        step!(3);
    }
    if remainder >= 1 { step!(0); }
    if remainder >= 2 { step!(1); }
    #[allow(unused_assignments)]
    if remainder >= 3 { step!(2); }
}

// ---------------------------------------------------------------------------
// rans_decode_shards_parallel
//
// Symmetric counterpart to rans_encode_shards_parallel.
// Accepts the flat shard payload written by pack_shard_payloads, plus the
// pre-built PDF tables and slot-lookup tables.
//
// Step 1 — sequential header parse: reads 36B per shard (4×u64 states + u32
//   bitstream length) to locate each shard's bitstream in the payload.
// Step 2 — Rayon parallel decode: all shards decoded concurrently into a
//   pre-allocated output buffer using non-overlapping slice writes.
//
// Returns (flat_residuals, bytes_consumed).
// ---------------------------------------------------------------------------

pub fn rans_decode_shards_parallel(
    payload: &[u8],
    shard_counts: &[u32],  // n total (channels × shards, flat)
    cum_freqs: &[u64],     // n × 257 flat
    sym_freqs: &[u64],     // n × 256 flat
    lookups: &[u8],        // n × 4096 flat
) -> (Vec<u8>, usize) {
    let n = shard_counts.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Step 1: sequential header parse — determine bitstream positions
    // Each shard header: 4 × 8 bytes (u64 LE states) + 4 bytes (u32 LE length)
    let mut shard_meta: Vec<([u64; 4], usize, usize)> = Vec::with_capacity(n);
    let mut p = 0usize;
    for _ in 0..n {
        let s0 = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap()); p += 8;
        let s1 = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap()); p += 8;
        let s2 = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap()); p += 8;
        let s3 = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap()); p += 8;
        let bs_len = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize; p += 4;
        shard_meta.push(([s0, s1, s2, s3], p, bs_len));
        p += bs_len;
    }
    let bytes_consumed = p;

    // Step 2: pre-compute per-shard output offsets (prefix sum of counts)
    let mut out_offsets = vec![0usize; n];
    let mut total = 0usize;
    for i in 0..n {
        out_offsets[i] = total;
        total += shard_counts[i] as usize;
    }

    // Step 3: parallel decode — each shard writes to a non-overlapping output region
    let mut out = vec![0u8; total];
    // SAFETY: out_offsets partitions `out` into non-overlapping regions.
    // Each par_iter task writes only to its assigned slice. `usize` is Send.
    let out_base = out.as_mut_ptr() as usize;
    shard_meta.par_iter().enumerate().for_each(|(idx, (states, bs_off, bs_len))| {
        let count = shard_counts[idx] as usize;
        if count == 0 { return; }
        let bs = &payload[*bs_off..*bs_off + *bs_len];
        let cf = &cum_freqs[idx * 257..(idx + 1) * 257];
        let sf = &sym_freqs[idx * 256..(idx + 1) * 256];
        let sl = &lookups[idx * 4096..(idx + 1) * 4096];
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_base + out_offsets[idx]) as *mut u8, count)
        };
        rans_decode_4way_core(states[0], states[1], states[2], states[3], bs, cf, sf, sl, out_slice);
    });

    (out, bytes_consumed)
}

// ---------------------------------------------------------------------------
// pack_shard_payloads  (in-place serialization into pre-allocated buffer)
//
// Each shard slot layout: [4×u64 states (32B)] [u32 length (4B)] [bitstream]
// ---------------------------------------------------------------------------

pub fn pack_shard_payloads(
    final_states: &[u64],     // n × 4 flat
    bs_lengths: &[u32],
    bs_offsets: &[u32],
    bitstreams: &[u8],
    out: &mut [u8],
    write_offsets: &[u32],
) {
    let n = bs_lengths.len();
    for i in 0..n {
        let mut dst = write_offsets[i] as usize;
        for lane in 0..4usize {
            let st = final_states[i * 4 + lane];
            for byte in 0..8usize {
                out[dst] = ((st >> (byte * 8)) & 0xFF) as u8;
                dst += 1;
            }
        }
        let bl = bs_lengths[i] as usize;
        out[dst]     = (bl & 0xFF) as u8;
        out[dst + 1] = ((bl >> 8) & 0xFF) as u8;
        out[dst + 2] = ((bl >> 16) & 0xFF) as u8;
        out[dst + 3] = ((bl >> 24) & 0xFF) as u8;
        dst += 4;
        let src = bs_offsets[i] as usize;
        out[dst..dst + bl].copy_from_slice(&bitstreams[src..src + bl]);
    }
}

// ---------------------------------------------------------------------------
// build_all_lookups  (parallel slot → symbol LUT for all (channel, shard) pairs)
//
// Input  cum_freqs: (n_ch × n_sh × 257) flat
// Output lookups:  (n_ch × n_sh × 4096) flat
// ---------------------------------------------------------------------------

pub fn build_all_lookups(cum_freqs: &[u64], n_ch: usize, n_sh: usize) -> Vec<u8> {
    let n_total = n_ch * n_sh;
    let results: Vec<(usize, Vec<u8>)> = (0..n_total)
        .into_par_iter()
        .map(|idx| {
            let cf = &cum_freqs[idx * 257..(idx + 1) * 257];
            let mut lk = vec![0u8; 4096];
            for sym in 0u8..=255 {
                let start = cf[sym as usize] as usize;
                let end = cf[sym as usize + 1] as usize;
                if end > start && end <= 4096 {
                    lk[start..end].fill(sym);
                }
            }
            (idx, lk)
        })
        .collect();

    let mut lookups = vec![0u8; n_total * 4096];
    for (idx, lk) in results {
        lookups[idx * 4096..(idx + 1) * 4096].copy_from_slice(&lk);
    }
    lookups
}

// ---------------------------------------------------------------------------
// compact_pdf_tables  (serialize custom PDFs to bytes; Mode 3/4-33 emit nothing)
// ---------------------------------------------------------------------------

pub fn compact_pdf_tables(
    sym_freqs: &[u64],    // n_shards × 256 flat
    shard_widths: &[u16], // n_shards
    shard_modes: &[u8],   // n_shards
) -> Vec<u8> {
    let n = shard_widths.len();
    let mut out: Vec<u8> = Vec::new();

    for s in 0..n {
        let mode = shard_modes[s];
        if mode >= 3 {
            continue; // templates and empty shards need no PDF bytes
        }

        let w = shard_widths[s] as usize;
        let freqs = &sym_freqs[s * 256..(s + 1) * 256];

        let indices: Vec<u8> = (0..w).filter(|&i| freqs[i] > 0).map(|i| i as u8).collect();
        let n_nz = indices.len();

        let cost_dense = w * 2;
        let cost_sparse = 2 + n_nz * 3;

        if cost_dense <= cost_sparse {
            out.push(0); // dense sub-mode
            for i in 0..w {
                let v = freqs[i] as u16;
                out.push((v & 0xFF) as u8);
                out.push(((v >> 8) & 0xFF) as u8);
            }
        } else {
            out.push(1); // sparse sub-mode
            let n16 = n_nz as u16;
            out.push((n16 & 0xFF) as u8);
            out.push(((n16 >> 8) & 0xFF) as u8);
            out.extend_from_slice(&indices);
            for &idx in &indices {
                let v = freqs[idx as usize] as u16;
                out.push((v & 0xFF) as u8);
                out.push(((v >> 8) & 0xFF) as u8);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// expand_pdf_tables  (deserialize PDFs; mirror of compact_pdf_tables)
//
// templates: (n_templates × 256) flat
// Returns expanded: n_shards × 256 flat
// ---------------------------------------------------------------------------

pub fn expand_pdf_tables(
    data: &[u8],
    shard_widths: &[u16],
    shard_modes: &[u8],
    templates: &[u64],
    n_templates: usize,
) -> Vec<u64> {
    let n = shard_widths.len();
    let mut out = vec![0u64; n * 256];
    let mut ptr = 0usize;

    for s in 0..n {
        let mode = shard_modes[s];
        let base = s * 256;

        if mode >= 4 {
            let tid = (mode - 4) as usize;
            if tid < n_templates {
                out[base..base + 256].copy_from_slice(&templates[tid * 256..(tid + 1) * 256]);
            } else {
                out[base] = 4096;
            }
        } else if mode == 3 {
            out[base] = 4096;
        } else {
            // Mode 0 — custom PDF
            if ptr >= data.len() {
                out[base] = 4096;
                continue;
            }
            let sub = data[ptr];
            ptr += 1;

            if sub == 0 {
                // Dense
                let w = shard_widths[s] as usize;
                let needed = w * 2;
                if ptr + needed > data.len() {
                    out[base] = 4096;
                    continue;
                }
                for i in 0..w {
                    let lo = data[ptr + i * 2] as u64;
                    let hi = data[ptr + i * 2 + 1] as u64;
                    out[base + i] = lo | (hi << 8);
                }
                ptr += needed;
            } else if sub == 1 {
                // Sparse
                if ptr + 2 > data.len() {
                    out[base] = 4096;
                    continue;
                }
                let n_nz = (data[ptr] as u16 | ((data[ptr + 1] as u16) << 8)) as usize;
                ptr += 2;
                let needed = n_nz + n_nz * 2;
                if ptr + needed > data.len() {
                    out[base] = 4096;
                    continue;
                }
                let syms = &data[ptr..ptr + n_nz];
                ptr += n_nz;
                for k in 0..n_nz {
                    let sym = syms[k] as usize;
                    let lo = data[ptr + k * 2] as u64;
                    let hi = data[ptr + k * 2 + 1] as u64;
                    out[base + sym] = lo | (hi << 8);
                }
                ptr += n_nz * 2;
            } else {
                out[base] = 4096;
            }
        }
    }

    out
}

// ===========================================================================
// Phase 2: Bitplane rANS  (port of core/rans_bitplane.py @njit kernels)
// ===========================================================================

const N_SPATIAL: usize = 64; // 2-bit L|U<<2|NW<<4 → 64 spatial contexts

#[inline(always)]
fn med_edge_tuned(a: u8, b: u8, c: u8) -> u8 {
    let max_ab = a.max(b) as i32;
    let min_ab = a.min(b) as i32;
    let diff = max_ab - min_ab;
    let ci = c as i32;
    // Python/Numba computes (int(a)+int(b)-ci) in uint64, so when a+b < c it wraps
    // to a huge value, causing max(min_ab, huge)=huge, min(max_ab, huge)=max_ab.
    // Replicate that: if a+b < c, p = max_ab.
    let sum_ab = a as i32 + b as i32;
    let p = if sum_ab < ci { max_ab } else { (sum_ab - ci).clamp(min_ab, max_ab) };
    let is_smooth = ((diff >= 1) & (diff <= 3)) as i32;
    let is_high = (ci > max_ab + 50) as i32;
    let is_low = (ci < min_ab - 50) as i32;
    (p + is_smooth * (is_high * (max_ab - p) + is_low * (min_ab - p))) as u8
}

#[inline(always)]
fn from_zigzag_bp(z: u8) -> i32 {
    // Unsigned right-shift first (matches Python: np.int8(z >> 1) ^ -(np.int8(z & 1)))
    let half = (z >> 1) as i32;
    let sign = -((z & 1) as i32);
    half ^ sign
}

// s_lut: row-major (511,511) flat; d_lut: row-major (256,4) flat
#[inline(always)]
fn get_context_id(ag: u8, bg: u8, cg: u8, intensity_idx: u8, s_lut: &[u8], d_lut: &[u8]) -> u8 {
    let da = ag as i32 - cg as i32 + 255;
    let db = bg as i32 - cg as i32 + 255;
    let pk = s_lut[da as usize * 511 + db as usize];
    d_lut[pk as usize * 4 + intensity_idx as usize]
}

// Quantize 4-symbol counts to 12-bit rANS frequencies (sum = 4096)
fn rescale_to_rans_freqs_bp(counts: &[u64]) -> ([u16; 4], [u16; 5]) {
    let total: u64 = counts.iter().sum();
    let mut f = [0u16; 4];
    let mut cf = [0u16; 5];
    if total > 0 {
        let mut nf = [0i64; 4];
        let mut acc = 0i64;
        for s in 0..4 {
            let v = (counts[s] as f64 * 4096.0 / total as f64).round() as i64;
            nf[s] = v.max(1);
            acc += nf[s];
        }
        let diff = 4096i64 - acc;
        let peak = nf.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
        nf[peak] = (nf[peak] + diff).max(1);
        let mut acc_cf = 0u16;
        for s in 0..4 {
            f[s] = nf[s] as u16;
            cf[s] = acc_cf;
            acc_cf += f[s];
        }
        cf[4] = acc_cf;
    } else {
        for s in 0..4 { f[s] = 1024; }
        for s in 0..5 { cf[s] = (s as u16) * 1024; }
    }
    (f, cf)
}

// ---------------------------------------------------------------------------
// bp_build_pdf_sharded
// ---------------------------------------------------------------------------

pub fn bp_build_pdf_sharded(
    resid_flat: &[u8],
    gray_flat:  &[u8],
    s_lut:      &[u8],
    i_lut:      &[u8],
    d_lut:      &[u8],
    h:          usize,
    w:          usize,
    n_ctx:      usize,
    is_chroma:  bool,
) -> (Vec<u16>, Vec<u16>) {
    let stride = w + 2;
    let counts = (1..=h).into_par_iter()
        .fold(
            || vec![0u64; 4 * n_ctx * 4],
            |mut local, pi| {
                for pj in 1..=w {
                    let ag = gray_flat[pi * stride + pj - 1];
                    let bg = gray_flat[(pi - 1) * stride + pj];
                    let cg = gray_flat[(pi - 1) * stride + pj - 1];
                    let intensity = if is_chroma {
                        gray_flat[pi * stride + pj]
                    } else {
                        med_edge_tuned(ag, bg, cg)
                    };
                    let sid = get_context_id(ag, bg, cg, i_lut[intensity as usize], s_lut, d_lut) as usize;
                    let r_l = resid_flat[pi * stride + pj - 1];
                    let r_u = resid_flat[(pi - 1) * stride + pj];
                    let r_n = resid_flat[(pi - 1) * stride + pj - 1];
                    let px  = resid_flat[pi * stride + pj];
                    for k in 0..4usize {
                        let shift = (k * 2) as u8;
                        let l_k = (r_l >> shift) & 3;
                        let u_k = (r_u >> shift) & 3;
                        let n_k = (r_n >> shift) & 3;
                        let bp_ctx = l_k as usize | ((u_k as usize) << 2) | ((n_k as usize) << 4);
                        let ctx = sid * N_SPATIAL + bp_ctx;
                        let sym = ((px >> shift) & 3) as usize;
                        local[k * n_ctx * 4 + ctx * 4 + sym] += 1;
                    }
                }
                local
            },
        )
        .reduce(
            || vec![0u64; 4 * n_ctx * 4],
            |mut a, b| { for i in 0..a.len() { a[i] += b[i]; } a },
        );

    let mut f_flat  = vec![0u16; 4 * n_ctx * 4];
    let mut cf_flat = vec![0u16; 4 * n_ctx * 5];
    for k in 0..4usize {
        for c in 0..n_ctx {
            let base = k * n_ctx * 4 + c * 4;
            let (fr, cfr) = rescale_to_rans_freqs_bp(&counts[base..base + 4]);
            f_flat[base..base + 4].copy_from_slice(&fr);
            let cfb = k * n_ctx * 5 + c * 5;
            cf_flat[cfb..cfb + 5].copy_from_slice(&cfr);
        }
    }
    (f_flat, cf_flat)
}

// ---------------------------------------------------------------------------
// bp_build_pdf_sharded_rgb
// ---------------------------------------------------------------------------

pub fn bp_build_pdf_sharded_rgb(
    gr_flat:     &[u8],
    rd_flat:     &[u8],
    bd_flat:     &[u8],
    gr_ref_flat: &[u8],
    s_lut:       &[u8],
    i_lut:       &[u8],
    d_lut:       &[u8],
    h:           usize,
    w:           usize,
    n_ctx:       usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let stride = w + 2;
    let sz = 3 * 4 * n_ctx * 4;
    let counts = (1..=h).into_par_iter()
        .fold(
            || vec![0u64; sz],
            |mut local, pi| {
                for pj in 1..=w {
                    let ag = gr_ref_flat[pi * stride + pj - 1];
                    let bg = gr_ref_flat[(pi - 1) * stride + pj];
                    let cg = gr_ref_flat[(pi - 1) * stride + pj - 1];
                    let sid_gr = get_context_id(ag, bg, cg, i_lut[med_edge_tuned(ag, bg, cg) as usize], s_lut, d_lut) as usize;
                    let sid_ch = get_context_id(ag, bg, cg, i_lut[gr_ref_flat[pi * stride + pj] as usize], s_lut, d_lut) as usize;

                    let rl_gr = gr_flat[pi * stride + pj - 1]; let ru_gr = gr_flat[(pi-1)*stride+pj]; let rn_gr = gr_flat[(pi-1)*stride+pj-1]; let px_gr = gr_flat[pi*stride+pj];
                    let rl_rd = rd_flat[pi * stride + pj - 1]; let ru_rd = rd_flat[(pi-1)*stride+pj]; let rn_rd = rd_flat[(pi-1)*stride+pj-1]; let px_rd = rd_flat[pi*stride+pj];
                    let rl_bd = bd_flat[pi * stride + pj - 1]; let ru_bd = bd_flat[(pi-1)*stride+pj]; let rn_bd = bd_flat[(pi-1)*stride+pj-1]; let px_bd = bd_flat[pi*stride+pj];

                    for k in 0..4usize {
                        let sh = (k * 2) as u8;
                        let bp_gr = (rl_gr>>sh)as usize&3|(((ru_gr>>sh)as usize&3)<<2)|(((rn_gr>>sh)as usize&3)<<4);
                        local[0*4*n_ctx*4 + k*n_ctx*4 + (sid_gr*N_SPATIAL+bp_gr)*4 + ((px_gr>>sh)&3)as usize] += 1;
                        let bp_rd = (rl_rd>>sh)as usize&3|(((ru_rd>>sh)as usize&3)<<2)|(((rn_rd>>sh)as usize&3)<<4);
                        local[1*4*n_ctx*4 + k*n_ctx*4 + (sid_ch*N_SPATIAL+bp_rd)*4 + ((px_rd>>sh)&3)as usize] += 1;
                        let bp_bd = (rl_bd>>sh)as usize&3|(((ru_bd>>sh)as usize&3)<<2)|(((rn_bd>>sh)as usize&3)<<4);
                        local[2*4*n_ctx*4 + k*n_ctx*4 + (sid_ch*N_SPATIAL+bp_bd)*4 + ((px_bd>>sh)&3)as usize] += 1;
                    }
                }
                local
            },
        )
        .reduce(|| vec![0u64; sz], |mut a, b| { for i in 0..a.len() { a[i] += b[i]; } a });

    let make_tables = |ch: usize| -> (Vec<u16>, Vec<u16>) {
        let mut f_flat  = vec![0u16; 4 * n_ctx * 4];
        let mut cf_flat = vec![0u16; 4 * n_ctx * 5];
        for k in 0..4usize {
            for c in 0..n_ctx {
                let base = ch * 4 * n_ctx * 4 + k * n_ctx * 4 + c * 4;
                let (fr, cfr) = rescale_to_rans_freqs_bp(&counts[base..base + 4]);
                let fb  = k * n_ctx * 4 + c * 4;
                let cfb = k * n_ctx * 5 + c * 5;
                f_flat[fb..fb + 4].copy_from_slice(&fr);
                cf_flat[cfb..cfb + 5].copy_from_slice(&cfr);
            }
        }
        (f_flat, cf_flat)
    };
    let (f_gr, cf_gr) = make_tables(0);
    let (f_rd, cf_rd) = make_tables(1);
    let (f_bd, cf_bd) = make_tables(2);
    (f_gr, cf_gr, f_rd, cf_rd, f_bd, cf_bd)
}

// ---------------------------------------------------------------------------
// bp_encode_sharded
// ---------------------------------------------------------------------------

pub fn bp_encode_sharded(
    resid_flat: &[u8],
    gray_flat:  &[u8],
    all_cf:     &[u16],  // (4, n_ctx, 5) flat
    all_sf:     &[u16],  // (4, n_ctx, 4) flat
    s_lut:      &[u8],
    i_lut:      &[u8],
    d_lut:      &[u8],
    h:          usize,
    w:          usize,
    n_ctx:      usize,
    is_chroma:  bool,
) -> (Vec<u64>, Vec<u8>) {
    let stride = w + 2;
    let magic  = get_magic_lut();
    let mut st = [L_LOWER; 4];
    let mut bs: Vec<u8> = Vec::with_capacity(h * w * 4 + 64);

    for pi in (1..=h).rev() {
        for pj in (1..=w).rev() {
            let ag = gray_flat[pi * stride + pj - 1];
            let bg = gray_flat[(pi - 1) * stride + pj];
            let cg = gray_flat[(pi - 1) * stride + pj - 1];
            let intensity = if is_chroma { gray_flat[pi * stride + pj] } else { med_edge_tuned(ag, bg, cg) };
            let sid = get_context_id(ag, bg, cg, i_lut[intensity as usize], s_lut, d_lut) as usize;

            let r_l = resid_flat[pi * stride + pj - 1];
            let r_u = resid_flat[(pi - 1) * stride + pj];
            let r_n = resid_flat[(pi - 1) * stride + pj - 1];
            let px  = resid_flat[pi * stride + pj];

            for layer in (0..4usize).rev() {
                let shift = (layer * 2) as u8;
                let l_k = (r_l >> shift) & 3;
                let u_k = (r_u >> shift) & 3;
                let n_k = (r_n >> shift) & 3;
                let ctx = sid * N_SPATIAL + (l_k as usize | ((u_k as usize) << 2) | ((n_k as usize) << 4));
                let sym = ((px >> shift) & 3) as usize;
                let f  = all_sf[layer * n_ctx * 4 + ctx * 4 + sym] as u64;
                let cf = all_cf[layer * n_ctx * 5 + ctx * 5 + sym] as u64;
                while st[layer] >= L_NORM_THRESHOLD * f {
                    bs.push((st[layer] & 0xFF) as u8);
                    st[layer] >>= 8;
                }
                let q = mul_hi(st[layer], magic[f as usize]);
                let r = st[layer] - q * f;
                let (q, r) = if r >= f { (q + 1, r - f) } else { (q, r) };
                st[layer] = (q << M_BITS) + r + cf;
            }
        }
    }
    (st.to_vec(), bs)
}

// ---------------------------------------------------------------------------
// bp_decode_sharded
// ---------------------------------------------------------------------------

pub fn bp_decode_sharded(
    bitstream: &[u8],
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:         usize,
    w:         usize,
    all_cf:    &[u16],  // (4, n_ctx, 5) flat
    all_sf:    &[u16],  // (4, n_ctx, 4) flat
    s_lut:     &[u8],
    i_lut:     &[u8],
    d_lut:     &[u8],
) -> Vec<u8> {
    let n_ctx  = all_cf.len() / 20; // 4 layers × 5 entries
    let stride = w + 2;
    let mask   = M_MASK;
    let mut st = [st0, st1, st2, st3];
    let mut ptr = bitstream.len() as isize - 1;
    let mut resid = vec![0u8; stride * (h + 2)];
    let mut orig  = vec![0u8; stride * (h + 2)];

    for pi in 1..=h {
        for pj in 1..=w {
            let ag = orig[pi * stride + pj - 1];
            let bg = orig[(pi - 1) * stride + pj];
            let cg = orig[(pi - 1) * stride + pj - 1];
            let p_g = med_edge_tuned(ag, bg, cg);
            let sid = get_context_id(ag, bg, cg, i_lut[p_g as usize], s_lut, d_lut) as usize;
            let r_l = resid[pi * stride + pj - 1];
            let r_u = resid[(pi - 1) * stride + pj];
            let r_n = resid[(pi - 1) * stride + pj - 1];

            let mut syms = [0u8; 4];
            for layer in 0..4usize {
                let shift = (layer * 2) as u8;
                let l_k = (r_l >> shift) & 3;
                let u_k = (r_u >> shift) & 3;
                let n_k = (r_n >> shift) & 3;
                let ctx = sid * N_SPATIAL + (l_k as usize | ((u_k as usize) << 2) | ((n_k as usize) << 4));
                let cf_base = layer * n_ctx * 5 + ctx * 5;
                let sf_base = layer * n_ctx * 4 + ctx * 4;
                let slot = st[layer] & mask;
                let sym = (slot >= all_cf[cf_base + 1] as u64) as u8
                        + (slot >= all_cf[cf_base + 2] as u64) as u8
                        + (slot >= all_cf[cf_base + 3] as u64) as u8;
                st[layer] = all_sf[sf_base + sym as usize] as u64 * (st[layer] >> M_BITS)
                    + slot - all_cf[cf_base + sym as usize] as u64;
                if st[layer] < L_LOWER && ptr >= 0 {
                    st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    if st[layer] < L_LOWER && ptr >= 0 {
                        st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    }
                }
                syms[layer] = sym;
            }

            let px = syms[0] | (syms[1] << 2) | (syms[2] << 4) | (syms[3] << 6);
            resid[pi * stride + pj] = px;
            orig[pi * stride + pj] = ((from_zigzag_bp(px) + p_g as i32) & 0xFF) as u8;
        }
    }

    let mut out = vec![0u8; h * w];
    for row in 0..h {
        let src = (row + 1) * stride + 1;
        out[row * w..(row + 1) * w].copy_from_slice(&resid[src..src + w]);
    }
    out
}

// ---------------------------------------------------------------------------
// bp_decode_sharded_debug  — same as bp_decode_sharded but also takes
// the original resid_flat and returns the first mismatch info:
// (row, col, ptr_at_entry, st0..st3_at_entry, expected_px, got_px)
// Returns None if decode is perfect.
// ---------------------------------------------------------------------------

pub fn bp_decode_sharded_debug(
    bitstream:    &[u8],
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:            usize,
    w:            usize,
    all_cf:       &[u16],
    all_sf:       &[u16],
    s_lut:        &[u8],
    i_lut:        &[u8],
    d_lut:        &[u8],
    orig_resid:   &[u8],  // original resid_flat (padded h+2 × w+2)
    orig_gray:    &[u8],  // original gray_flat  (padded h+2 × w+2)
) -> (Vec<u8>, Option<(usize,usize,isize,[u64;4],u8,u8)>) {
    let n_ctx  = all_cf.len() / 20;
    let stride = w + 2;
    let mask   = M_MASK;
    let mut st = [st0, st1, st2, st3];
    let mut ptr = bitstream.len() as isize - 1;
    let mut resid = vec![0u8; stride * (h + 2)];
    let mut orig_rec = vec![0u8; stride * (h + 2)];
    let mut first_mismatch: Option<(usize,usize,isize,[u64;4],u8,u8)> = None;

    for pi in 1..=h {
        for pj in 1..=w {
            let ag = orig_rec[pi * stride + pj - 1];
            let bg = orig_rec[(pi - 1) * stride + pj];
            let cg = orig_rec[(pi - 1) * stride + pj - 1];
            let p_g = med_edge_tuned(ag, bg, cg);
            let sid = get_context_id(ag, bg, cg, i_lut[p_g as usize], s_lut, d_lut) as usize;
            let r_l = resid[pi * stride + pj - 1];
            let r_u = resid[(pi - 1) * stride + pj];
            let r_n = resid[(pi - 1) * stride + pj - 1];

            // snapshot state BEFORE decode (for mismatch reporting)
            let ptr_before = ptr;
            let st_before = st;

            let mut syms = [0u8; 4];
            for layer in 0..4usize {
                let shift = (layer * 2) as u8;
                let l_k = (r_l >> shift) & 3;
                let u_k = (r_u >> shift) & 3;
                let n_k = (r_n >> shift) & 3;
                let ctx = sid * N_SPATIAL + (l_k as usize | ((u_k as usize) << 2) | ((n_k as usize) << 4));
                let cf_base = layer * n_ctx * 5 + ctx * 5;
                let sf_base = layer * n_ctx * 4 + ctx * 4;
                let slot = st[layer] & mask;
                let sym = (slot >= all_cf[cf_base + 1] as u64) as u8
                        + (slot >= all_cf[cf_base + 2] as u64) as u8
                        + (slot >= all_cf[cf_base + 3] as u64) as u8;
                st[layer] = all_sf[sf_base + sym as usize] as u64 * (st[layer] >> M_BITS)
                    + slot - all_cf[cf_base + sym as usize] as u64;
                if st[layer] < L_LOWER && ptr >= 0 {
                    st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    if st[layer] < L_LOWER && ptr >= 0 {
                        st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    }
                }
                syms[layer] = sym;
            }

            let px = syms[0] | (syms[1] << 2) | (syms[2] << 4) | (syms[3] << 6);
            resid[pi * stride + pj] = px;
            orig_rec[pi * stride + pj] = ((from_zigzag_bp(px) + p_g as i32) & 0xFF) as u8;

            let expected_px = orig_resid[pi * stride + pj];
            if first_mismatch.is_none() && px != expected_px {
                first_mismatch = Some((pi, pj, ptr_before, st_before, expected_px, px));
            }
            let _ = orig_gray;
        }
    }

    let mut out = vec![0u8; h * w];
    for row in 0..h {
        let src = (row + 1) * stride + 1;
        out[row * w..(row + 1) * w].copy_from_slice(&resid[src..src + w]);
    }
    (out, first_mismatch)
}

// ---------------------------------------------------------------------------
// bp_decode_sharded_with_ref
// ---------------------------------------------------------------------------

pub fn bp_decode_sharded_with_ref(
    bitstream:   &[u8],
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:           usize,
    w:           usize,
    all_cf:      &[u16],
    all_sf:      &[u16],
    ref_ch_flat: &[u8],  // (h+2)×(w+2) flat
    s_lut:       &[u8],
    i_lut:       &[u8],
    d_lut:       &[u8],
) -> Vec<u8> {
    let n_ctx  = all_cf.len() / 20;
    let stride = w + 2;
    let mask   = M_MASK;
    let mut st = [st0, st1, st2, st3];
    let mut ptr = bitstream.len() as isize - 1;
    let mut resid = vec![0u8; stride * (h + 2)];

    for pi in 1..=h {
        for pj in 1..=w {
            let ag = ref_ch_flat[pi * stride + pj - 1];
            let bg = ref_ch_flat[(pi - 1) * stride + pj];
            let cg = ref_ch_flat[(pi - 1) * stride + pj - 1];
            let sid = get_context_id(ag, bg, cg, i_lut[ref_ch_flat[pi * stride + pj] as usize], s_lut, d_lut) as usize;
            let r_l = resid[pi * stride + pj - 1];
            let r_u = resid[(pi - 1) * stride + pj];
            let r_n = resid[(pi - 1) * stride + pj - 1];

            let mut syms = [0u8; 4];
            for layer in 0..4usize {
                let shift = (layer * 2) as u8;
                let l_k = (r_l >> shift) & 3;
                let u_k = (r_u >> shift) & 3;
                let n_k = (r_n >> shift) & 3;
                let ctx = sid * N_SPATIAL + (l_k as usize | ((u_k as usize) << 2) | ((n_k as usize) << 4));
                let cf_base = layer * n_ctx * 5 + ctx * 5;
                let sf_base = layer * n_ctx * 4 + ctx * 4;
                let slot = st[layer] & mask;
                let sym = (slot >= all_cf[cf_base + 1] as u64) as u8
                        + (slot >= all_cf[cf_base + 2] as u64) as u8
                        + (slot >= all_cf[cf_base + 3] as u64) as u8;
                st[layer] = all_sf[sf_base + sym as usize] as u64 * (st[layer] >> M_BITS)
                    + slot - all_cf[cf_base + sym as usize] as u64;
                if st[layer] < L_LOWER && ptr >= 0 {
                    st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    if st[layer] < L_LOWER && ptr >= 0 {
                        st[layer] = (st[layer] << 8) | bitstream[ptr as usize] as u64; ptr -= 1;
                    }
                }
                syms[layer] = sym;
            }
            resid[pi * stride + pj] = syms[0] | (syms[1] << 2) | (syms[2] << 4) | (syms[3] << 6);
        }
    }

    let mut out = vec![0u8; h * w];
    for row in 0..h {
        let src = (row + 1) * stride + 1;
        out[row * w..(row + 1) * w].copy_from_slice(&resid[src..src + w]);
    }
    out
}

// ===========================================================================
// Phase 3: Fused Pass1  (port of core/sharding.py fused_rct_p1_*)
// ===========================================================================

// Maps (curr - pred) & 0xFF to ZigZag symbol; mirrors Python to_zigzag.
#[inline(always)]
fn to_zigzag(delta: u8) -> u8 {
    let s = delta as i8;
    ((s << 1) ^ (s >> 7)) as u8
}

// Shared alpha-channel predictor pass used by both gray and RGB paths.
fn alpha_pass(h: usize, w: usize, a_ch: &[u8]) -> (Vec<u8>, u64, f64) {
    let mut a_res  = vec![0u8; h * w];
    let mut a_hits = 0u64;
    let mut a_sum  = 0.0f64;
    for i in 0..h {
        let mut ag_a = 0u8;
        let mut cg_a = 0u8;
        for j in 0..w {
            let bg_a    = if i > 0 { a_ch[(i - 1) * w + j] } else { 0u8 };
            let pg_a    = med_edge_tuned(ag_a, bg_a, cg_a);
            let r_a_val = a_ch[i * w + j];
            let r_a     = to_zigzag(r_a_val.wrapping_sub(pg_a));
            a_res[i * w + j] = r_a;
            a_sum  += (r_a_val as f64 - pg_a as f64).abs();
            a_hits += (r_a == 0) as u64;
            cg_a = bg_a;
            ag_a = r_a_val;
        }
    }
    (a_res, a_hits, a_sum)
}

pub struct P1GrayResult {
    pub gr_ch_p:            Vec<u8>,   // (h+2)*(w+2)
    pub gr_res:             Vec<u8>,   // (h+2)*(w+2)
    pub shard_counts:       Vec<u32>,  // 3 * n_shards
    pub shard_stats:        Vec<u32>,  // 3 * n_shards * 256
    pub shard_offsets:      Vec<u32>,  // 3 * n_shards
    pub row_global_offsets: Vec<u32>,  // h * 3 * n_shards
    pub hits:               Vec<u32>,  // 3
    pub sums:               Vec<u64>,  // 3
    pub a_res:              Vec<u8>,   // h * w, empty when !is_rgba
    pub a_hits:             u64,
    pub a_sum:              f64,
}

pub fn fused_rct_p1_gray(
    h: usize,
    w: usize,
    gray_raw: &[u8],
    a_ch:     &[u8],
    is_rgba:  bool,
    n_shards: usize,
    s_lut:    &[u8],
    i_lut:    &[u8],
    d_lut:    &[u8],
) -> P1GrayResult {
    let stride   = w + 2;
    let pad_size = (h + 2) * stride;

    let mut gr_ch_p = vec![0u8; pad_size];
    for i in 0..h {
        let dst = (i + 1) * stride + 1;
        gr_ch_p[dst..dst + w].copy_from_slice(&gray_raw[i * w..(i + 1) * w]);
    }

    let mut gr_res       = vec![0u8; pad_size];
    let mut row_ptrs_all = vec![0u32; h * n_shards];

    let n_chunks   = 16.min(h).max(1);
    let chunk_size = (h + n_chunks - 1) / n_chunks;

    struct ChunkStats { hist: Vec<u32>, h_acc: u32, s_acc: u64 }

    let gr_res_base = gr_res.as_mut_ptr() as usize;
    let rp_base     = row_ptrs_all.as_mut_ptr() as usize;

    // SAFETY: chunk c owns rows [c*chunk_size, end_i). Writes to gr_res[pi*stride..]
    // and row_ptrs_all[i*n_shards..] cover non-overlapping regions across chunks.
    // gr_ch_p / s_lut / i_lut / d_lut are read-only for the duration of this scope.
    let chunk_stats: Vec<ChunkStats> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start_i = c * chunk_size;
            let end_i   = (start_i + chunk_size).min(h);
            let mut hist  = vec![0u32; n_shards * 256];
            let mut h_acc = 0u32;
            let mut s_acc = 0u64;
            for i in start_i..end_i {
                let pi = i + 1;
                let rp: &mut [u32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        (rp_base as *mut u32).add(i * n_shards), n_shards)
                };
                for pj in 1..=w {
                    let ag   = gr_ch_p[pi * stride + pj - 1];
                    let bg   = gr_ch_p[(pi - 1) * stride + pj];
                    let cg   = gr_ch_p[(pi - 1) * stride + pj - 1];
                    let pg   = med_edge_tuned(ag, bg, cg);
                    let ctx  = get_context_id(ag, bg, cg, i_lut[pg as usize], s_lut, d_lut) as usize;
                    let curr = gr_ch_p[pi * stride + pj];
                    let zz   = to_zigzag(curr.wrapping_sub(pg));
                    unsafe { *(gr_res_base as *mut u8).add(pi * stride + pj) = zz; }
                    rp[ctx] += 1;
                    hist[ctx * 256 + zz as usize] += 1;
                    h_acc += (zz == 0) as u32;
                    s_acc += (curr as i64 - pg as i64).unsigned_abs() as u64;
                }
            }
            ChunkStats { hist, h_acc, s_acc }
        })
        .collect();

    let mut global_hist = vec![0u32; n_shards * 256];
    let mut total_hits  = 0u32;
    let mut total_sums  = 0u64;
    for cs in &chunk_stats {
        for j in 0..global_hist.len() { global_hist[j] += cs.hist[j]; }
        total_hits += cs.h_acc;
        total_sums += cs.s_acc;
    }

    let mut shard_counts_1ch = vec![0u32; n_shards];
    for i in 0..h {
        for s in 0..n_shards { shard_counts_1ch[s] += row_ptrs_all[i * n_shards + s]; }
    }

    let mut shard_offsets_1ch = vec![0u32; n_shards];
    let mut cur = 0u32;
    for s in 0..n_shards { shard_offsets_1ch[s] = cur; cur += shard_counts_1ch[s]; }

    let mut row_go_1ch = vec![0u32; h * n_shards];
    for s in 0..n_shards {
        let mut cur = shard_offsets_1ch[s];
        for i in 0..h {
            row_go_1ch[i * n_shards + s] = cur;
            cur += row_ptrs_all[i * n_shards + s];
        }
    }

    // Expand 1-channel results into 3-channel layout expected by Python callers.
    let mut shard_counts       = vec![0u32; 3 * n_shards];
    let mut shard_stats        = vec![0u32; 3 * n_shards * 256];
    let mut shard_offsets      = vec![0u32; 3 * n_shards];
    let mut row_global_offsets = vec![0u32; h * 3 * n_shards];

    shard_counts[..n_shards].copy_from_slice(&shard_counts_1ch);
    shard_offsets[..n_shards].copy_from_slice(&shard_offsets_1ch);
    shard_stats[..n_shards * 256].copy_from_slice(&global_hist);
    for i in 0..h {
        for s in 0..n_shards {
            // Layout [h][3][n_shards]; ch=0 → index = i*3*n_shards + s
            row_global_offsets[i * 3 * n_shards + s] = row_go_1ch[i * n_shards + s];
        }
    }

    let mut hits = vec![0u32; 3];
    let mut sums = vec![0u64; 3];
    hits[0] = total_hits;
    sums[0] = total_sums;

    let (a_res, a_hits, a_sum) =
        if is_rgba && !a_ch.is_empty() { alpha_pass(h, w, a_ch) }
        else { (vec![], 0, 0.0) };

    P1GrayResult {
        gr_ch_p, gr_res,
        shard_counts, shard_stats, shard_offsets, row_global_offsets,
        hits, sums, a_res, a_hits, a_sum,
    }
}

pub struct P1RgbResult {
    pub gr_ch_p:            Vec<u8>,
    pub rd_ch_p:            Vec<u8>,
    pub bd_ch_p:            Vec<u8>,
    pub gr_res:             Vec<u8>,
    pub rd_res:             Vec<u8>,
    pub bd_res:             Vec<u8>,
    pub ch_hists:           Vec<u32>,  // 3 * 256  per-channel pixel value histogram
    pub shard_counts:       Vec<u32>,  // 3 * n_shards
    pub shard_stats:        Vec<u32>,  // 3 * n_shards * 256
    pub shard_offsets:      Vec<u32>,  // 3 * n_shards
    pub row_global_offsets: Vec<u32>,  // h * 3 * n_shards
    pub hits:               Vec<u32>,  // 3
    pub sums:               Vec<u64>,  // 3
    pub a_res:              Vec<u8>,
    pub a_hits:             u64,
    pub a_sum:              f64,
}

pub fn fused_rct_p1_rgb(
    h: usize,
    w: usize,
    rgb_raw:  &[u8],   // (h, w, 3) C-contiguous
    a_ch:     &[u8],
    is_rgba:  bool,
    n_shards: usize,
    s_lut:    &[u8],
    i_lut:    &[u8],
    d_lut:    &[u8],
) -> P1RgbResult {
    let stride   = w + 2;
    let pad_size = (h + 2) * stride;

    let mut gr_ch_p = vec![0u8; pad_size];
    let mut rd_ch_p = vec![0u8; pad_size];
    let mut bd_ch_p = vec![0u8; pad_size];

    // RCT: G unchanged, R→R-G, B→B-G (all mod 256)
    for i in 0..h {
        for j in 0..w {
            let base = (i * w + j) * 3;
            let r = rgb_raw[base];
            let g = rgb_raw[base + 1];
            let b = rgb_raw[base + 2];
            let dst = (i + 1) * stride + 1 + j;
            gr_ch_p[dst] = g;
            rd_ch_p[dst] = r.wrapping_sub(g);
            bd_ch_p[dst] = b.wrapping_sub(g);
        }
    }

    let mut gr_res       = vec![0u8; pad_size];
    let mut rd_res       = vec![0u8; pad_size];
    let mut bd_res       = vec![0u8; pad_size];
    let mut row_ptrs_all = vec![0u32; h * 3 * n_shards]; // [h, 3, n_shards]

    let n_chunks   = 16.min(h).max(1);
    let chunk_size = (h + n_chunks - 1) / n_chunks;

    struct ChunkStats {
        hist:    Vec<u32>,   // 3 * n_shards * 256
        ch_hist: Vec<u32>,   // 3 * 256
        h_acc:   [u32; 3],
        s_acc:   [u64; 3],
    }

    let gr_res_base = gr_res.as_mut_ptr() as usize;
    let rd_res_base = rd_res.as_mut_ptr() as usize;
    let bd_res_base = bd_res.as_mut_ptr() as usize;
    let rp_base     = row_ptrs_all.as_mut_ptr() as usize;

    // SAFETY: same non-overlapping row-range invariant as fused_rct_p1_gray.
    let chunk_stats: Vec<ChunkStats> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start_i = c * chunk_size;
            let end_i   = (start_i + chunk_size).min(h);
            let mut hist    = vec![0u32; 3 * n_shards * 256];
            let mut ch_hist = vec![0u32; 3 * 256];
            let mut h_acc   = [0u32; 3];
            let mut s_acc   = [0u64; 3];

            for i in start_i..end_i {
                let pi = i + 1;
                let rp: &mut [u32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        (rp_base as *mut u32).add(i * 3 * n_shards), 3 * n_shards)
                };
                for pj in 1..=w {
                    let idx    = pi * stride + pj;
                    let idx_l  = pi * stride + pj - 1;
                    let idx_u  = (pi - 1) * stride + pj;
                    let idx_ul = (pi - 1) * stride + pj - 1;

                    // Green — context uses predicted green value
                    let ag = gr_ch_p[idx_l]; let bg = gr_ch_p[idx_u]; let cg = gr_ch_p[idx_ul];
                    let vg   = gr_ch_p[idx];
                    let pg   = med_edge_tuned(ag, bg, cg);
                    let ctxg = get_context_id(ag, bg, cg, i_lut[pg as usize], s_lut, d_lut) as usize;
                    let zz_g = to_zigzag(vg.wrapping_sub(pg));
                    unsafe { *(gr_res_base as *mut u8).add(idx) = zz_g; }
                    rp[ctxg] += 1;
                    hist[ctxg * 256 + zz_g as usize] += 1;
                    ch_hist[vg as usize] += 1;
                    h_acc[0] += (zz_g == 0) as u32;
                    s_acc[0] += (vg as i64 - pg as i64).unsigned_abs() as u64;

                    // Chroma intensity uses ACTUAL green value (mirrors Python)
                    let idx_v = i_lut[vg as usize];

                    // Red-diff
                    let a1 = rd_ch_p[idx_l]; let b1 = rd_ch_p[idx_u]; let c1 = rd_ch_p[idx_ul];
                    let v1   = rd_ch_p[idx];
                    let p1   = med_edge_tuned(a1, b1, c1);
                    let ctx1 = get_context_id(a1, b1, c1, idx_v, s_lut, d_lut) as usize;
                    let zz1  = to_zigzag(v1.wrapping_sub(p1));
                    unsafe { *(rd_res_base as *mut u8).add(idx) = zz1; }
                    rp[n_shards + ctx1] += 1;
                    hist[n_shards * 256 + ctx1 * 256 + zz1 as usize] += 1;
                    ch_hist[256 + v1 as usize] += 1;
                    h_acc[1] += (zz1 == 0) as u32;
                    s_acc[1] += (v1 as i64 - p1 as i64).unsigned_abs() as u64;

                    // Blue-diff
                    let a2 = bd_ch_p[idx_l]; let b2 = bd_ch_p[idx_u]; let c2 = bd_ch_p[idx_ul];
                    let v2   = bd_ch_p[idx];
                    let p2   = med_edge_tuned(a2, b2, c2);
                    let ctx2 = get_context_id(a2, b2, c2, idx_v, s_lut, d_lut) as usize;
                    let zz2  = to_zigzag(v2.wrapping_sub(p2));
                    unsafe { *(bd_res_base as *mut u8).add(idx) = zz2; }
                    rp[2 * n_shards + ctx2] += 1;
                    hist[2 * n_shards * 256 + ctx2 * 256 + zz2 as usize] += 1;
                    ch_hist[2 * 256 + v2 as usize] += 1;
                    h_acc[2] += (zz2 == 0) as u32;
                    s_acc[2] += (v2 as i64 - p2 as i64).unsigned_abs() as u64;
                }
            }
            ChunkStats { hist, ch_hist, h_acc, s_acc }
        })
        .collect();

    let mut global_hist = vec![0u32; 3 * n_shards * 256];
    let mut ch_hists    = vec![0u32; 3 * 256];
    let mut total_hits  = [0u32; 3];
    let mut total_sums  = [0u64; 3];
    for cs in &chunk_stats {
        for j in 0..global_hist.len() { global_hist[j] += cs.hist[j]; }
        for j in 0..ch_hists.len()   { ch_hists[j]    += cs.ch_hist[j]; }
        for c in 0..3 { total_hits[c] += cs.h_acc[c]; total_sums[c] += cs.s_acc[c]; }
    }

    let mut shard_counts = vec![0u32; 3 * n_shards];
    for i in 0..h {
        for c in 0..3usize {
            for s in 0..n_shards {
                shard_counts[c * n_shards + s] +=
                    row_ptrs_all[i * 3 * n_shards + c * n_shards + s];
            }
        }
    }

    let mut shard_offsets = vec![0u32; 3 * n_shards];
    for c in 0..3usize {
        let mut cur = 0u32;
        for s in 0..n_shards {
            shard_offsets[c * n_shards + s] = cur;
            cur += shard_counts[c * n_shards + s];
        }
    }

    let mut row_global_offsets = vec![0u32; h * 3 * n_shards];
    for c in 0..3usize {
        for s in 0..n_shards {
            let mut cur = shard_offsets[c * n_shards + s];
            for i in 0..h {
                row_global_offsets[i * 3 * n_shards + c * n_shards + s] = cur;
                cur += row_ptrs_all[i * 3 * n_shards + c * n_shards + s];
            }
        }
    }

    let mut hits = vec![0u32; 3];
    let mut sums = vec![0u64; 3];
    for c in 0..3 { hits[c] = total_hits[c]; sums[c] = total_sums[c]; }

    let (a_res, a_hits, a_sum) =
        if is_rgba && !a_ch.is_empty() { alpha_pass(h, w, a_ch) }
        else { (vec![], 0, 0.0) };

    P1RgbResult {
        gr_ch_p, rd_ch_p, bd_ch_p,
        gr_res, rd_res, bd_res,
        ch_hists,
        shard_counts, shard_stats: global_hist, shard_offsets, row_global_offsets,
        hits, sums, a_res, a_hits, a_sum,
    }
}

// ===========================================================================
// Phase 5: Pass-2 encode, shard-decode, and inverse transforms
// (ports of core/sharding.py and core/transform.py @njit kernels)
// ===========================================================================

// Inverse ZigZag: maps symbol z → residual as u8 (= from_zigzag(z) & 0xFF)
#[inline(always)]
fn from_zigzag_u8(z: u8) -> u8 {
    let half = (z >> 1) as i32;
    let sign = -((z & 1) as i32);
    (half ^ sign) as u8
}

// ---------------------------------------------------------------------------
// shard_pass_2_rgb
// Parallel gather pass (encode): writes zigzag residuals into pre-allocated
// shard buffers. row_global_offsets[h,3,n_shards] guarantees that each row i
// owns a non-overlapping write range in shard_gr/rd/bd.
// ---------------------------------------------------------------------------
pub fn shard_pass_2_rgb(
    h: usize,
    w: usize,
    gr_ch: &[u8],
    rd_ch: &[u8],
    bd_ch: &[u8],
    row_global_offsets: &[u32],
    shard_gr: &mut [u8],
    shard_rd: &mut [u8],
    shard_bd: &mut [u8],
    s_lut: &[u8],
    i_lut: &[u8],
    d_lut: &[u8],
) {
    let n_shards = row_global_offsets.len() / (h * 3);
    let stride   = w + 2;

    let sgr_base = shard_gr.as_mut_ptr() as usize;
    let srd_base = shard_rd.as_mut_ptr() as usize;
    let sbd_base = shard_bd.as_mut_ptr() as usize;

    // SAFETY: row_global_offsets encodes non-overlapping write regions per row.
    (0..h).into_par_iter().for_each(|i| {
        let pi       = i + 1;
        let row_base = i * 3 * n_shards;
        let mut l_gr: Vec<u32> = row_global_offsets[row_base..row_base + n_shards].to_vec();
        let mut l_rd: Vec<u32> = row_global_offsets[row_base + n_shards..row_base + 2 * n_shards].to_vec();
        let mut l_bd: Vec<u32> = row_global_offsets[row_base + 2 * n_shards..row_base + 3 * n_shards].to_vec();

        for j in 0..w {
            let pj     = j + 1;
            let idx    = pi * stride + pj;
            let idx_l  = pi * stride + pj - 1;
            let idx_u  = (pi - 1) * stride + pj;
            let idx_ul = (pi - 1) * stride + pj - 1;

            let ag = gr_ch[idx_l]; let bg = gr_ch[idx_u]; let cg = gr_ch[idx_ul];
            let vg = gr_ch[idx];
            let pg   = med_edge_tuned(ag, bg, cg);
            let ctxg = get_context_id(ag, bg, cg, i_lut[pg as usize], s_lut, d_lut) as usize;
            let zz_g = to_zigzag(vg.wrapping_sub(pg));
            let idx_v = i_lut[vg as usize];
            unsafe { *(sgr_base as *mut u8).add(l_gr[ctxg] as usize) = zz_g; }
            l_gr[ctxg] += 1;

            let a1 = rd_ch[idx_l]; let b1 = rd_ch[idx_u]; let c1 = rd_ch[idx_ul];
            let v1   = rd_ch[idx];
            let p1   = med_edge_tuned(a1, b1, c1);
            let ctx1 = get_context_id(a1, b1, c1, idx_v, s_lut, d_lut) as usize;
            let zz1  = to_zigzag(v1.wrapping_sub(p1));
            unsafe { *(srd_base as *mut u8).add(l_rd[ctx1] as usize) = zz1; }
            l_rd[ctx1] += 1;

            let a2 = bd_ch[idx_l]; let b2 = bd_ch[idx_u]; let c2 = bd_ch[idx_ul];
            let v2   = bd_ch[idx];
            let p2   = med_edge_tuned(a2, b2, c2);
            let ctx2 = get_context_id(a2, b2, c2, idx_v, s_lut, d_lut) as usize;
            let zz2  = to_zigzag(v2.wrapping_sub(p2));
            unsafe { *(sbd_base as *mut u8).add(l_bd[ctx2] as usize) = zz2; }
            l_bd[ctx2] += 1;
        }
    });
}

// ---------------------------------------------------------------------------
// shard_pass_2_gray  (single-channel variant)
// ---------------------------------------------------------------------------
pub fn shard_pass_2_gray(
    h: usize,
    w: usize,
    gr_ch: &[u8],
    row_global_offsets: &[u32],
    shard_gr: &mut [u8],
    s_lut: &[u8],
    i_lut: &[u8],
    d_lut: &[u8],
) {
    let n_shards = row_global_offsets.len() / (h * 3);
    let stride   = w + 2;
    let sgr_base = shard_gr.as_mut_ptr() as usize;

    (0..h).into_par_iter().for_each(|i| {
        let pi       = i + 1;
        let row_base = i * 3 * n_shards;
        let mut l_gr: Vec<u32> = row_global_offsets[row_base..row_base + n_shards].to_vec();

        for j in 0..w {
            let pj  = j + 1;
            let idx = pi * stride + pj;
            let ag  = gr_ch[pi * stride + pj - 1];
            let bg  = gr_ch[(pi - 1) * stride + pj];
            let cg  = gr_ch[(pi - 1) * stride + pj - 1];
            let vg  = gr_ch[idx];
            let pg   = med_edge_tuned(ag, bg, cg);
            let ctxg = get_context_id(ag, bg, cg, i_lut[pg as usize], s_lut, d_lut) as usize;
            let zz_g = to_zigzag(vg.wrapping_sub(pg));
            unsafe { *(sgr_base as *mut u8).add(l_gr[ctxg] as usize) = zz_g; }
            l_gr[ctxg] += 1;
        }
    });
}

// ---------------------------------------------------------------------------
// reconstruct_shards_rgb  (decode)
// Sequential scan per channel; rd and bd are independent after green is done.
// off_gr/rd/bd: per-shard base offsets in res_gr/rd/bd (relative to channel start).
// Returns trimmed (h, w) arrays for gr, rd, bd.
// ---------------------------------------------------------------------------
pub fn reconstruct_shards_rgb(
    h: usize,
    w: usize,
    res_gr: &[u8],
    res_rd: &[u8],
    res_bd: &[u8],
    off_gr: &[u32],
    off_rd: &[u32],
    off_bd: &[u32],
    s_lut: &[u8],
    i_lut: &[u8],
    d_lut: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let stride   = w + 2;
    let pad_size = (h + 2) * stride;
    let n_shards = off_gr.len();

    // --- Green channel (sequential scan) ---
    let mut gr_rec = vec![0u8; pad_size];
    let mut p_gr   = off_gr.to_vec();
    for pi in 1..=h {
        for pj in 1..=w {
            let idx    = pi * stride + pj;
            let idx_l  = pi * stride + pj - 1;
            let idx_u  = (pi - 1) * stride + pj;
            let idx_ul = (pi - 1) * stride + pj - 1;
            let ag = gr_rec[idx_l]; let bg = gr_rec[idx_u]; let cg = gr_rec[idx_ul];
            let pg  = med_edge_tuned(ag, bg, cg);
            let ctx = get_context_id(ag, bg, cg, i_lut[pg as usize], s_lut, d_lut) as usize;
            let zz  = res_gr[p_gr[ctx] as usize];
            p_gr[ctx] += 1;
            gr_rec[idx] = from_zigzag_u8(zz).wrapping_add(pg);
        }
    }

    // --- Red-diff channel ---
    let mut rd_rec = vec![0u8; pad_size];
    let mut p_rd   = off_rd.to_vec();
    for pi in 1..=h {
        for pj in 1..=w {
            let idx    = pi * stride + pj;
            let idx_l  = pi * stride + pj - 1;
            let idx_u  = (pi - 1) * stride + pj;
            let idx_ul = (pi - 1) * stride + pj - 1;
            let cur_g  = gr_rec[idx];
            let a = rd_rec[idx_l]; let b = rd_rec[idx_u]; let c = rd_rec[idx_ul];
            let p   = med_edge_tuned(a, b, c);
            let ctx = get_context_id(a, b, c, i_lut[cur_g as usize], s_lut, d_lut) as usize;
            let zz  = res_rd[p_rd[ctx] as usize];
            p_rd[ctx] += 1;
            rd_rec[idx] = from_zigzag_u8(zz).wrapping_add(p);
        }
    }

    // --- Blue-diff channel ---
    let mut bd_rec = vec![0u8; pad_size];
    let mut p_bd   = off_bd.to_vec();
    for pi in 1..=h {
        for pj in 1..=w {
            let idx    = pi * stride + pj;
            let idx_l  = pi * stride + pj - 1;
            let idx_u  = (pi - 1) * stride + pj;
            let idx_ul = (pi - 1) * stride + pj - 1;
            let cur_g  = gr_rec[idx];
            let a = bd_rec[idx_l]; let b = bd_rec[idx_u]; let c = bd_rec[idx_ul];
            let p   = med_edge_tuned(a, b, c);
            let ctx = get_context_id(a, b, c, i_lut[cur_g as usize], s_lut, d_lut) as usize;
            let zz  = res_bd[p_bd[ctx] as usize];
            p_bd[ctx] += 1;
            bd_rec[idx] = from_zigzag_u8(zz).wrapping_add(p);
        }
    }

    // Trim padding to (h, w)
    let mut gr_out = vec![0u8; h * w];
    let mut rd_out = vec![0u8; h * w];
    let mut bd_out = vec![0u8; h * w];
    for i in 0..h {
        let src = (i + 1) * stride + 1;
        gr_out[i * w..(i + 1) * w].copy_from_slice(&gr_rec[src..src + w]);
        rd_out[i * w..(i + 1) * w].copy_from_slice(&rd_rec[src..src + w]);
        bd_out[i * w..(i + 1) * w].copy_from_slice(&bd_rec[src..src + w]);
    }

    let _ = n_shards; // derived from off_gr.len(), used implicitly
    (gr_out, rd_out, bd_out)
}

// ---------------------------------------------------------------------------
// restore_channels  (inverse G-sub RCT, parallel over rows)
// gr_rec/rd_rec/bd_rec/a_ch are (h, w) row-major flat arrays.
// Returns (h, w, channels) flat.
// ---------------------------------------------------------------------------
pub fn restore_channels(
    gr_rec: &[u8],
    rd_rec: &[u8],
    bd_rec: &[u8],
    a_ch:   &[u8],
    is_rgba:     bool,
    is_grayscale: bool,
    apply_gsub:  bool,
    h: usize,
    w: usize,
) -> Vec<u8> {
    let channels = if is_rgba { 4 } else { 3 };
    let mut rgb  = vec![0u8; h * w * channels];

    rgb.par_chunks_mut(w * channels)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..w {
                let g = gr_rec[i * w + j];
                let (r_out, g_out, b_out) = if is_grayscale {
                    (g, g, g)
                } else if apply_gsub {
                    (rd_rec[i * w + j].wrapping_add(g), g, bd_rec[i * w + j].wrapping_add(g))
                } else {
                    (rd_rec[i * w + j], g, bd_rec[i * w + j])
                };
                let base       = j * channels;
                out_row[base]     = r_out;
                out_row[base + 1] = g_out;
                out_row[base + 2] = b_out;
                if is_rgba { out_row[base + 3] = a_ch[i * w + j]; }
            }
        });

    rgb
}

// ---------------------------------------------------------------------------
// reconstruct_2d_inplace  (sequential inverse MED prediction)
// res_ch and rec are (h, w) row-major flat arrays; writes result into rec.
// ---------------------------------------------------------------------------
pub fn reconstruct_2d_inplace(h: usize, w: usize, res_ch: &[u8], rec: &mut [u8]) {
    for i in 0..h {
        for j in 0..w {
            let a    = if j > 0       { rec[i * w + j - 1]           } else { 0u8 };
            let b    = if i > 0       { rec[(i - 1) * w + j]         } else { 0u8 };
            let c    = if i > 0 && j > 0 { rec[(i - 1) * w + j - 1] } else { 0u8 };
            let pred = med_edge_tuned(a, b, c);
            rec[i * w + j] = from_zigzag_u8(res_ch[i * w + j]).wrapping_add(pred);
        }
    }
}

// ---------------------------------------------------------------------------
// reconstruct_2d_channels  (allocates and returns)
// ---------------------------------------------------------------------------
pub fn reconstruct_2d_channels(h: usize, w: usize, res_ch: &[u8]) -> Vec<u8> {
    let mut rec = vec![0u8; h * w];
    if h > 0 && w > 0 {
        reconstruct_2d_inplace(h, w, res_ch, &mut rec);
    }
    rec
}
