mod rans_core;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyArray1, PyArray2, PyArray3,
    PyUntypedArrayMethods,
};
use ndarray::{Array2, Array3};


// ---------------------------------------------------------------------------
// Rayon thread pool configuration
// ---------------------------------------------------------------------------

#[pyfunction]
fn set_rayon_threads(n: usize) -> PyResult<()> {
    // build_global errors silently if the pool was already initialized —
    // that is acceptable behaviour (first call wins).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    Ok(())
}

// ---------------------------------------------------------------------------
// get_magic_lut — exposes the precomputed LUT as a numpy array so that
// rans_bitplane.py (still Numba) can keep using _MAGIC_LUT as a numpy argument
// ---------------------------------------------------------------------------

#[pyfunction]
fn get_magic_lut<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let lut = rans_core::get_magic_lut();
    Ok(PyArray1::from_slice(py, lut))
}

// ---------------------------------------------------------------------------
// build_pdf_tables_from_shards
//
// Python signature (mirrors old rans.build_pdf_tables_from_shards):
//   (data_flat, shard_offsets, shard_lengths, shard_widths, templates, disable_templates)
//   -> (cum_freqs[n,257], sym_freqs[n,256], modes[n])
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data_flat, shard_offsets, shard_lengths, shard_widths, templates, disable_templates))]
fn build_pdf_tables_from_shards<'py>(
    py: Python<'py>,
    data_flat: PyReadonlyArray1<'py, u8>,
    shard_offsets: PyReadonlyArray1<'py, u32>,
    shard_lengths: PyReadonlyArray1<'py, u32>,
    shard_widths: PyReadonlyArray1<'py, u16>,
    templates: PyReadonlyArray2<'py, u64>,
    disable_templates: bool,
) -> PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<u64>>, Bound<'py, PyArray1<u8>>)> {
    let data = data_flat.as_slice()?;
    let offs = shard_offsets.as_slice()?;
    let lens = shard_lengths.as_slice()?;
    let widths = shard_widths.as_slice()?;
    let n_shards = lens.len();

    let tpl_arr = templates.as_array();
    let tpl_shape = tpl_arr.shape();
    let n_tpl = tpl_shape[0];
    let tpl_flat: Vec<u64> = tpl_arr.iter().copied().collect();

    let (cum, sym, modes) = py.allow_threads(|| {
        rans_core::build_pdf_tables_from_shards(
            data, offs, lens, widths,
            &tpl_flat, n_tpl, disable_templates,
        )
    });

    let cum_arr = Array2::from_shape_vec((n_shards, 257), cum)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sym_arr = Array2::from_shape_vec((n_shards, 256), sym)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        PyArray2::from_owned_array(py, cum_arr),
        PyArray2::from_owned_array(py, sym_arr),
        PyArray1::from_vec(py, modes),
    ))
}

// ---------------------------------------------------------------------------
// rans_encode_shards_parallel
//
// Python signature (mirrors old rans.rans_encode_shards_parallel):
//   (data_flat, shard_offsets, shard_lengths, cum_freqs[n,257],
//    sym_freqs[n,256], initial_state)
//   -> (final_states[n,4], bitstreams_flat[M], bs_offsets[n], bs_lengths[n])
// ---------------------------------------------------------------------------

#[pyfunction]
fn rans_encode_shards_parallel<'py>(
    py: Python<'py>,
    data_flat: PyReadonlyArray1<'py, u8>,
    shard_offsets: PyReadonlyArray1<'py, u32>,
    shard_lengths: PyReadonlyArray1<'py, u32>,
    cum_freqs: PyReadonlyArray2<'py, u64>,
    sym_freqs: PyReadonlyArray2<'py, u64>,
    initial_state: u64,
) -> PyResult<(
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let data = data_flat.as_slice()?;
    let offs = shard_offsets.as_slice()?;
    let lens = shard_lengths.as_slice()?;
    let n = lens.len();

    let cf_arr = cum_freqs.as_array();
    let sf_arr = sym_freqs.as_array();
    let cf: Vec<u64> = cf_arr.iter().copied().collect();
    let sf: Vec<u64> = sf_arr.iter().copied().collect();

    let (states, bs, bs_offs, bs_lens) = py.allow_threads(|| {
        rans_core::rans_encode_shards_parallel(data, offs, lens, &cf, &sf, initial_state)
    });

    let states_arr = Array2::from_shape_vec((n, 4), states)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        PyArray2::from_owned_array(py, states_arr),
        PyArray1::from_vec(py, bs),
        PyArray1::from_vec(py, bs_offs),
        PyArray1::from_vec(py, bs_lens),
    ))
}

// ---------------------------------------------------------------------------
// rans_decode_4way_core
//
// Python signature (mirrors old rans.rans_decode_4way_core):
//   (st0, st1, st2, st3, bitstream[M], cum_freqs[257],
//    sym_freqs[256], slot_lookup[4096], out[N])  -> None  (in-place)
//
// `out` is a writable numpy view; we use a const-ptr cast because Python's
// ThreadPoolExecutor passes non-overlapping slices of the same base array.
// ---------------------------------------------------------------------------

#[pyfunction]
fn rans_decode_4way_core<'py>(
    _py: Python<'py>,
    st0: u64,
    st1: u64,
    st2: u64,
    st3: u64,
    bitstream: PyReadonlyArray1<'py, u8>,
    cum_freqs: PyReadonlyArray1<'py, u64>,
    sym_freqs: PyReadonlyArray1<'py, u64>,
    slot_lookup: PyReadonlyArray1<'py, u8>,
    out: PyReadonlyArray1<'py, u8>,
) -> PyResult<()> {
    let bs = bitstream.as_slice()?;
    let cf = cum_freqs.as_slice()?;
    let sf = sym_freqs.as_slice()?;
    let sl = slot_lookup.as_slice()?;
    let out_ro = out.as_slice()?;
    let out_len = out_ro.len();
    // SAFETY: Python ThreadPoolExecutor ensures each thread writes to a
    // distinct non-overlapping slice; *const → *mut cast is sound here.
    // No GIL release needed — this function is serial (no Rayon inside).
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ro.as_ptr() as *mut u8, out_len) };
    rans_core::rans_decode_4way_core(st0, st1, st2, st3, bs, cf, sf, sl, out_slice);

    Ok(())
}

// ---------------------------------------------------------------------------
// rans_decode_shards_parallel
//
// Python signature:
//   (payload[K], shard_counts[n], cum_freqs[n,257], sym_freqs[n,256],
//    lookups[n,4096])
//   -> (residuals[total_res], bytes_consumed: int)
// ---------------------------------------------------------------------------

#[pyfunction]
fn rans_decode_shards_parallel<'py>(
    py: Python<'py>,
    payload: PyReadonlyArray1<'py, u8>,
    shard_counts: PyReadonlyArray1<'py, u32>,
    cum_freqs: PyReadonlyArray2<'py, u64>,
    sym_freqs: PyReadonlyArray2<'py, u64>,
    lookups: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize)> {
    let pl = payload.as_slice()?;
    let sc = shard_counts.as_slice()?;
    let cf: Vec<u64> = cum_freqs.as_array().iter().copied().collect();
    let sf: Vec<u64> = sym_freqs.as_array().iter().copied().collect();
    let lk: Vec<u8>  = lookups.as_array().iter().copied().collect();

    let (residuals, bytes_consumed) = py.allow_threads(|| {
        rans_core::rans_decode_shards_parallel(pl, sc, &cf, &sf, &lk)
    });

    Ok((PyArray1::from_vec(py, residuals), bytes_consumed))
}

// ---------------------------------------------------------------------------
// pack_shard_payloads
//
// Python signature:
//   (final_states[n,4], bs_lengths[n], bs_offsets[n],
//    bitstreams_flat[M], out[K], shard_write_offsets[n])  -> None  (in-place)
// ---------------------------------------------------------------------------

#[pyfunction]
fn pack_shard_payloads<'py>(
    _py: Python<'py>,
    final_states: PyReadonlyArray2<'py, u64>,
    bs_lengths: PyReadonlyArray1<'py, u32>,
    bs_offsets: PyReadonlyArray1<'py, u32>,
    bitstreams_flat: PyReadonlyArray1<'py, u8>,
    out: PyReadonlyArray1<'py, u8>,
    write_offsets: PyReadonlyArray1<'py, u32>,
) -> PyResult<()> {
    let states_arr = final_states.as_array();
    let states: Vec<u64> = states_arr.iter().copied().collect();
    let lens = bs_lengths.as_slice()?;
    let offs = bs_offsets.as_slice()?;
    let bs = bitstreams_flat.as_slice()?;
    let out_ro = out.as_slice()?;
    let out_len = out_ro.len();
    let wo = write_offsets.as_slice()?;
    // Serial function — no GIL release needed.
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ro.as_ptr() as *mut u8, out_len) };
    rans_core::pack_shard_payloads(&states, lens, offs, bs, out_slice, wo);

    Ok(())
}

// ---------------------------------------------------------------------------
// build_all_lookups
//
// Python signature:
//   (all_cum_freqs[n_ch, n_sh, 257])  -> lookups[n_ch, n_sh, 4096]
// ---------------------------------------------------------------------------

#[pyfunction]
fn build_all_lookups<'py>(
    py: Python<'py>,
    all_cum_freqs: PyReadonlyArray3<'py, u64>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = all_cum_freqs.as_array();
    let shape = arr.shape();
    let n_ch = shape[0];
    let n_sh = shape[1];

    let flat: Vec<u64> = arr.iter().copied().collect();

    let lookups = py.allow_threads(|| rans_core::build_all_lookups(&flat, n_ch, n_sh));

    let arr3 = Array3::from_shape_vec((n_ch, n_sh, 4096), lookups)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyArray3::from_owned_array(py, arr3))
}

// ---------------------------------------------------------------------------
// compact_pdf_tables
//
// Python signature:
//   (all_sym_freqs[n,256], shard_widths[n], shard_modes[n])  -> bytes_array[M]
// ---------------------------------------------------------------------------

#[pyfunction]
fn compact_pdf_tables<'py>(
    py: Python<'py>,
    all_sym_freqs: PyReadonlyArray2<'py, u64>,
    shard_widths: PyReadonlyArray1<'py, u16>,
    shard_modes: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let sf_arr = all_sym_freqs.as_array();
    let sf: Vec<u64> = sf_arr.iter().copied().collect();
    let widths = shard_widths.as_slice()?;
    let modes = shard_modes.as_slice()?;

    let out = py.allow_threads(|| rans_core::compact_pdf_tables(&sf, widths, modes));

    Ok(PyArray1::from_vec(py, out))
}

// ---------------------------------------------------------------------------
// expand_pdf_tables
//
// Python signature:
//   (compacted[M], shard_widths[n], shard_modes[n], templates[t,256])
//   -> expanded[n, 256]
// ---------------------------------------------------------------------------

#[pyfunction]
fn expand_pdf_tables<'py>(
    py: Python<'py>,
    compacted: PyReadonlyArray1<'py, u8>,
    shard_widths: PyReadonlyArray1<'py, u16>,
    shard_modes: PyReadonlyArray1<'py, u8>,
    templates: PyReadonlyArray2<'py, u64>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let data = compacted.as_slice()?;
    let widths = shard_widths.as_slice()?;
    let modes = shard_modes.as_slice()?;
    let n = widths.len();

    let tpl_arr = templates.as_array();
    let n_tpl = tpl_arr.shape()[0];
    let tpl_flat: Vec<u64> = tpl_arr.iter().copied().collect();

    let out = py.allow_threads(|| {
        rans_core::expand_pdf_tables(data, widths, modes, &tpl_flat, n_tpl)
    });

    let arr = Array2::from_shape_vec((n, 256), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyArray2::from_owned_array(py, arr))
}

// ---------------------------------------------------------------------------
// bp_build_pdf_sharded
//
// Python signature:
//   (resid_flat[(h+2)*(w+2)], gray_flat[(h+2)*(w+2)], s_lut[511*511],
//    i_lut[256], d_lut[256*4], h, w, n_ctx, is_chroma)
//   -> (f[4,n_ctx,4] u16, cf[4,n_ctx,5] u16)
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_build_pdf_sharded<'py>(
    py: Python<'py>,
    resid_flat: PyReadonlyArray1<'py, u8>,
    gray_flat:  PyReadonlyArray1<'py, u8>,
    s_lut:      PyReadonlyArray1<'py, u8>,
    i_lut:      PyReadonlyArray1<'py, u8>,
    d_lut:      PyReadonlyArray1<'py, u8>,
    h:          usize,
    w:          usize,
    n_ctx:      usize,
    is_chroma:  bool,
) -> PyResult<(Bound<'py, PyArray3<u16>>, Bound<'py, PyArray3<u16>>)> {
    let rf = resid_flat.as_slice()?;
    let gf = gray_flat.as_slice()?;
    let sl = s_lut.as_slice()?;
    let il = i_lut.as_slice()?;
    let dl = d_lut.as_slice()?;
    let (f_flat, cf_flat) = py.allow_threads(|| {
        rans_core::bp_build_pdf_sharded(rf, gf, sl, il, dl, h, w, n_ctx, is_chroma)
    });
    let f_arr  = Array3::from_shape_vec((4, n_ctx, 4), f_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cf_arr = Array3::from_shape_vec((4, n_ctx, 5), cf_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray3::from_owned_array(py, f_arr),
        PyArray3::from_owned_array(py, cf_arr),
    ))
}

// ---------------------------------------------------------------------------
// bp_build_pdf_sharded_rgb
//
// Python signature:
//   (gr_flat, rd_flat, bd_flat, gr_ref_flat, s_lut, i_lut, d_lut, h, w, n_ctx)
//   -> (f_gr, cf_gr, f_rd, cf_rd, f_bd, cf_bd)  each (4,n_ctx,4) or (4,n_ctx,5)
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_build_pdf_sharded_rgb<'py>(
    py: Python<'py>,
    gr_flat:     PyReadonlyArray1<'py, u8>,
    rd_flat:     PyReadonlyArray1<'py, u8>,
    bd_flat:     PyReadonlyArray1<'py, u8>,
    gr_ref_flat: PyReadonlyArray1<'py, u8>,
    s_lut:       PyReadonlyArray1<'py, u8>,
    i_lut:       PyReadonlyArray1<'py, u8>,
    d_lut:       PyReadonlyArray1<'py, u8>,
    h:           usize,
    w:           usize,
    n_ctx:       usize,
) -> PyResult<(
    Bound<'py, PyArray3<u16>>, Bound<'py, PyArray3<u16>>,
    Bound<'py, PyArray3<u16>>, Bound<'py, PyArray3<u16>>,
    Bound<'py, PyArray3<u16>>, Bound<'py, PyArray3<u16>>,
)> {
    let grf = gr_flat.as_slice()?;
    let rdf = rd_flat.as_slice()?;
    let bdf = bd_flat.as_slice()?;
    let grrf = gr_ref_flat.as_slice()?;
    let sl2 = s_lut.as_slice()?;
    let il2 = i_lut.as_slice()?;
    let dl2 = d_lut.as_slice()?;
    let (f_gr, cf_gr, f_rd, cf_rd, f_bd, cf_bd) = py.allow_threads(|| {
        rans_core::bp_build_pdf_sharded_rgb(grf, rdf, bdf, grrf, sl2, il2, dl2, h, w, n_ctx)
    });
    let mk3 = |v: Vec<u16>, d2: usize| Array3::from_shape_vec((4, n_ctx, d2), v)
        .map_err(|e| PyValueError::new_err(e.to_string()));
    Ok((
        PyArray3::from_owned_array(py, mk3(f_gr, 4)?),
        PyArray3::from_owned_array(py, mk3(cf_gr, 5)?),
        PyArray3::from_owned_array(py, mk3(f_rd, 4)?),
        PyArray3::from_owned_array(py, mk3(cf_rd, 5)?),
        PyArray3::from_owned_array(py, mk3(f_bd, 4)?),
        PyArray3::from_owned_array(py, mk3(cf_bd, 5)?),
    ))
}

// ---------------------------------------------------------------------------
// bp_encode_sharded
//
// Python signature:
//   (resid_flat, gray_flat, all_cf[4,n_ctx,5], all_sf[4,n_ctx,4],
//    s_lut, i_lut, d_lut, h, w, n_ctx, is_chroma)
//   -> (states[4] u64, bitstream[M] u8)
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_encode_sharded<'py>(
    py: Python<'py>,
    resid_flat: PyReadonlyArray1<'py, u8>,
    gray_flat:  PyReadonlyArray1<'py, u8>,
    all_cf:     PyReadonlyArray3<'py, u16>,
    all_sf:     PyReadonlyArray3<'py, u16>,
    s_lut:      PyReadonlyArray1<'py, u8>,
    i_lut:      PyReadonlyArray1<'py, u8>,
    d_lut:      PyReadonlyArray1<'py, u8>,
    h:          usize,
    w:          usize,
    n_ctx:      usize,
    is_chroma:  bool,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u8>>)> {
    let rf  = resid_flat.as_slice()?;
    let gf  = gray_flat.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;
    let cf_vec: Vec<u16> = all_cf.as_array().iter().copied().collect();
    let sf_vec: Vec<u16> = all_sf.as_array().iter().copied().collect();
    let (states, bs) = py.allow_threads(|| {
        rans_core::bp_encode_sharded(rf, gf, &cf_vec, &sf_vec, sl, il, dl, h, w, n_ctx, is_chroma)
    });
    Ok((
        PyArray1::from_vec(py, states),
        PyArray1::from_vec(py, bs),
    ))
}

// ---------------------------------------------------------------------------
// bp_decode_sharded
//
// Python signature:
//   (bitstream[M], st0, st1, st2, st3, h, w,
//    all_cf[4,n_ctx,5], all_sf[4,n_ctx,4], s_lut, i_lut, d_lut)
//   -> resid[h,w] u8
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_decode_sharded<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'py, u8>,
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:   usize,
    w:   usize,
    all_cf: PyReadonlyArray3<'py, u16>,
    all_sf: PyReadonlyArray3<'py, u16>,
    s_lut:  PyReadonlyArray1<'py, u8>,
    i_lut:  PyReadonlyArray1<'py, u8>,
    d_lut:  PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let bs  = bitstream.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;
    let cf_vec: Vec<u16> = all_cf.as_array().iter().copied().collect();
    let sf_vec: Vec<u16> = all_sf.as_array().iter().copied().collect();
    let resid = py.allow_threads(|| {
        rans_core::bp_decode_sharded(bs, st0, st1, st2, st3, h, w, &cf_vec, &sf_vec, sl, il, dl)
    });
    let arr = Array2::from_shape_vec((h, w), resid)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

// ---------------------------------------------------------------------------
// bp_decode_sharded_with_ref
//
// Python signature:
//   (bitstream[M], st0..st3, h, w,
//    all_cf[4,n_ctx,5], all_sf[4,n_ctx,4],
//    ref_ch_flat[(h+2)*(w+2)], s_lut, i_lut, d_lut)
//   -> resid[h,w] u8
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_decode_sharded_with_ref<'py>(
    py: Python<'py>,
    bitstream:   PyReadonlyArray1<'py, u8>,
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:   usize,
    w:   usize,
    all_cf:      PyReadonlyArray3<'py, u16>,
    all_sf:      PyReadonlyArray3<'py, u16>,
    ref_ch_flat: PyReadonlyArray1<'py, u8>,
    s_lut:       PyReadonlyArray1<'py, u8>,
    i_lut:       PyReadonlyArray1<'py, u8>,
    d_lut:       PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let bs   = bitstream.as_slice()?;
    let rcf  = ref_ch_flat.as_slice()?;
    let sl   = s_lut.as_slice()?;
    let il   = i_lut.as_slice()?;
    let dl   = d_lut.as_slice()?;
    let cf_vec: Vec<u16> = all_cf.as_array().iter().copied().collect();
    let sf_vec: Vec<u16> = all_sf.as_array().iter().copied().collect();
    let resid = py.allow_threads(|| {
        rans_core::bp_decode_sharded_with_ref(bs, st0, st1, st2, st3, h, w, &cf_vec, &sf_vec, rcf, sl, il, dl)
    });
    let arr = Array2::from_shape_vec((h, w), resid)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

// ---------------------------------------------------------------------------
// bp_decode_sharded_debug
//
// Python signature:
//   (bitstream, st0..st3, h, w, all_cf, all_sf, s_lut, i_lut, d_lut,
//    orig_resid_flat, orig_gray_flat)
//   -> (resid[h,w], mismatch: None | (row,col,ptr,st0..st3,expected,got))
// ---------------------------------------------------------------------------

#[pyfunction]
fn bp_decode_sharded_debug<'py>(
    py: Python<'py>,
    bitstream: PyReadonlyArray1<'py, u8>,
    st0: u64, st1: u64, st2: u64, st3: u64,
    h:   usize,
    w:   usize,
    all_cf:        PyReadonlyArray3<'py, u16>,
    all_sf:        PyReadonlyArray3<'py, u16>,
    s_lut:         PyReadonlyArray1<'py, u8>,
    i_lut:         PyReadonlyArray1<'py, u8>,
    d_lut:         PyReadonlyArray1<'py, u8>,
    orig_resid_flat: PyReadonlyArray1<'py, u8>,
    orig_gray_flat:  PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, PyObject)> {
    let bs  = bitstream.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;
    let orf = orig_resid_flat.as_slice()?;
    let ogf = orig_gray_flat.as_slice()?;
    let cf_vec: Vec<u16> = all_cf.as_array().iter().copied().collect();
    let sf_vec: Vec<u16> = all_sf.as_array().iter().copied().collect();
    let (resid, mismatch) = py.allow_threads(|| {
        rans_core::bp_decode_sharded_debug(
            bs, st0, st1, st2, st3, h, w, &cf_vec, &sf_vec, sl, il, dl, orf, ogf
        )
    });
    let arr = Array2::from_shape_vec((h, w), resid)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let py_mismatch = match mismatch {
        None => py.None(),
        Some((row, col, ptr, st, exp, got)) => {
            pyo3::types::PyTuple::new(py, &[
                row.into_pyobject(py)?.into_any().unbind(),
                col.into_pyobject(py)?.into_any().unbind(),
                ptr.into_pyobject(py)?.into_any().unbind(),
                st[0].into_pyobject(py)?.into_any().unbind(),
                st[1].into_pyobject(py)?.into_any().unbind(),
                st[2].into_pyobject(py)?.into_any().unbind(),
                st[3].into_pyobject(py)?.into_any().unbind(),
                (exp as u32).into_pyobject(py)?.into_any().unbind(),
                (got as u32).into_pyobject(py)?.into_any().unbind(),
            ])?.into_any().unbind()
        }
    };
    Ok((PyArray2::from_owned_array(py, arr), py_mismatch))
}

// ---------------------------------------------------------------------------
// p1_gray
//
// Python signature (replaces sharding.fused_rct_p1_gray):
//   (h, w, gray_raw[h,w], a_ch[h,w|0,0], is_rgba, n_shards,
//    s_lut[511,511], i_lut[256], d_lut[256,4])
//   -> (gr_ch_p[h+2,w+2], sc[3,n], ss[3,n,256], so[3,n],
//       rgo[h,3,n], hits[3], sums[3],
//       (gr_res[h+2,w+2], empty[0,0], empty[0,0], a_res[h,w|0,0]),
//       (a_hits:u64, a_sum:f64))
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, gray_raw, a_ch, is_rgba, n_shards, s_lut, i_lut, d_lut))]
fn p1_gray<'py>(
    py:       Python<'py>,
    h:        usize,
    w:        usize,
    gray_raw: PyReadonlyArray2<'py, u8>,
    a_ch:     PyReadonlyArray2<'py, u8>,
    is_rgba:  bool,
    n_shards: usize,
    s_lut:    PyReadonlyArray2<'py, u8>,
    i_lut:    PyReadonlyArray1<'py, u8>,
    d_lut:    PyReadonlyArray2<'py, u8>,
) -> PyResult<PyObject> {
    let gray  = gray_raw.as_slice()?;
    let a_raw = a_ch.as_slice()?;
    let sl    = s_lut.as_slice()?;
    let il    = i_lut.as_slice()?;
    let dl    = d_lut.as_slice()?;

    let res = py.allow_threads(|| {
        rans_core::fused_rct_p1_gray(h, w, gray, a_raw, is_rgba, n_shards, sl, il, dl)
    });

    let hp2 = h + 2;
    let wp2 = w + 2;
    let e   = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());

    let gr_ch_p = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.gr_ch_p).map_err(e)?).into_any();
    let sc = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((3, n_shards), res.shard_counts).map_err(e)?).into_any();
    let ss = PyArray3::from_owned_array(py,
        Array3::from_shape_vec((3, n_shards, 256), res.shard_stats).map_err(e)?).into_any();
    let so = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((3, n_shards), res.shard_offsets).map_err(e)?).into_any();
    let rgo = PyArray3::from_owned_array(py,
        Array3::from_shape_vec((h, 3, n_shards), res.row_global_offsets).map_err(e)?).into_any();
    let hits = PyArray1::from_vec(py, res.hits).into_any();
    let sums = PyArray1::from_vec(py, res.sums).into_any();

    let gr_res = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.gr_res).map_err(e)?).into_any();
    let empty1 = PyArray2::<u8>::from_owned_array(py, Array2::zeros((0usize, 0usize))).into_any();
    let empty2 = PyArray2::<u8>::from_owned_array(py, Array2::zeros((0usize, 0usize))).into_any();
    let a_res_py = if is_rgba && !res.a_res.is_empty() {
        PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), res.a_res).map_err(e)?).into_any()
    } else {
        PyArray2::<u8>::from_owned_array(py, Array2::zeros((0usize, 0usize))).into_any()
    };
    let res_cached = pyo3::types::PyTuple::new(py, [gr_res, empty1, empty2, a_res_py])?.into_any();

    let a_metrics = pyo3::types::PyTuple::new(py, [
        res.a_hits.into_pyobject(py)?.into_any(),
        res.a_sum.into_pyobject(py)?.into_any(),
    ])?.into_any();

    Ok(pyo3::types::PyTuple::new(py,
        [gr_ch_p, sc, ss, so, rgo, hits, sums, res_cached, a_metrics]
    )?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// p1_rgb
//
// Python signature (replaces sharding.fused_rct_p1_rgb):
//   (h, w, rgb_raw[h,w,3], a_ch[h,w|0,0], is_rgba, n_shards,
//    s_lut[511,511], i_lut[256], d_lut[256,4])
//   -> (gr_ch_p, rd_ch_p, bd_ch_p, ch_hists[3,256],
//       sc[3,n], ss[3,n,256], so[3,n], rgo[h,3,n],
//       hits[3], sums[3],
//       (gr_res, rd_res, bd_res, a_res),
//       (a_hits:u64, a_sum:f64))
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, rgb_raw, a_ch, is_rgba, n_shards, s_lut, i_lut, d_lut))]
fn p1_rgb<'py>(
    py:       Python<'py>,
    h:        usize,
    w:        usize,
    rgb_raw:  PyReadonlyArray3<'py, u8>,
    a_ch:     PyReadonlyArray2<'py, u8>,
    is_rgba:  bool,
    n_shards: usize,
    s_lut:    PyReadonlyArray2<'py, u8>,
    i_lut:    PyReadonlyArray1<'py, u8>,
    d_lut:    PyReadonlyArray2<'py, u8>,
) -> PyResult<PyObject> {
    let rgb   = rgb_raw.as_slice()?;
    let a_raw = a_ch.as_slice()?;
    let sl    = s_lut.as_slice()?;
    let il    = i_lut.as_slice()?;
    let dl    = d_lut.as_slice()?;

    let res = py.allow_threads(|| {
        rans_core::fused_rct_p1_rgb(h, w, rgb, a_raw, is_rgba, n_shards, sl, il, dl)
    });

    let hp2 = h + 2;
    let wp2 = w + 2;
    let e   = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());

    let gr_ch_p = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.gr_ch_p).map_err(e)?).into_any();
    let rd_ch_p = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.rd_ch_p).map_err(e)?).into_any();
    let bd_ch_p = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.bd_ch_p).map_err(e)?).into_any();
    let ch_hists = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((3, 256), res.ch_hists).map_err(e)?).into_any();
    let sc = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((3, n_shards), res.shard_counts).map_err(e)?).into_any();
    let ss = PyArray3::from_owned_array(py,
        Array3::from_shape_vec((3, n_shards, 256), res.shard_stats).map_err(e)?).into_any();
    let so = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((3, n_shards), res.shard_offsets).map_err(e)?).into_any();
    let rgo = PyArray3::from_owned_array(py,
        Array3::from_shape_vec((h, 3, n_shards), res.row_global_offsets).map_err(e)?).into_any();
    let hits = PyArray1::from_vec(py, res.hits).into_any();
    let sums = PyArray1::from_vec(py, res.sums).into_any();

    let gr_res = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.gr_res).map_err(e)?).into_any();
    let rd_res = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.rd_res).map_err(e)?).into_any();
    let bd_res = PyArray2::from_owned_array(py,
        Array2::from_shape_vec((hp2, wp2), res.bd_res).map_err(e)?).into_any();
    let a_res_py = if is_rgba && !res.a_res.is_empty() {
        PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), res.a_res).map_err(e)?).into_any()
    } else {
        PyArray2::<u8>::from_owned_array(py, Array2::zeros((0usize, 0usize))).into_any()
    };
    let res_cached = pyo3::types::PyTuple::new(py,
        [gr_res, rd_res, bd_res, a_res_py])?.into_any();

    let a_metrics = pyo3::types::PyTuple::new(py, [
        res.a_hits.into_pyobject(py)?.into_any(),
        res.a_sum.into_pyobject(py)?.into_any(),
    ])?.into_any();

    Ok(pyo3::types::PyTuple::new(py,
        [gr_ch_p, rd_ch_p, bd_ch_p, ch_hists, sc, ss, so, rgo, hits, sums, res_cached, a_metrics]
    )?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// decide_shard_mode
//
// Python signature (mirrors rans_selector.decide_shard_mode):
//   (counts[256], width, header_penalty_bits, templates[n,256], disable_templates)
//   -> (mode: int, pdf: ndarray[u64, 256])
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (counts, width, header_penalty_bits, templates, disable_templates))]
fn decide_shard_mode<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, u64>,
    width: usize,
    header_penalty_bits: f64,
    templates: PyReadonlyArray2<'py, u64>,
    disable_templates: bool,
) -> PyResult<(u8, Bound<'py, PyArray1<u64>>)> {
    let c = counts.as_slice()?;
    if c.len() != 256 {
        return Err(PyValueError::new_err("counts must have exactly 256 elements"));
    }
    let tpl_arr = templates.as_array();
    let n_tpl = tpl_arr.shape()[0];
    let tpl_flat: Vec<u64> = tpl_arr.iter().copied().collect();

    let (mode, pdf) = py.allow_threads(|| {
        rans_core::decide_shard_mode(c, width, header_penalty_bits, &tpl_flat, n_tpl, disable_templates)
    });

    Ok((mode, PyArray1::from_vec(py, pdf)))
}

// ---------------------------------------------------------------------------
// p2_rgb  (Phase 5a: encode pass-2, RGB)
//
// Python signature:
//   (h, w, gr_ch[h+2,w+2], rd_ch[h+2,w+2], bd_ch[h+2,w+2],
//    row_global_offsets[h,3,n], shard_gr[M], shard_rd[M], shard_bd[M],
//    s_lut[511,511], i_lut[256], d_lut[256,4])  -> None  (in-place)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, gr_ch, rd_ch, bd_ch, row_global_offsets, shard_gr, shard_rd, shard_bd, s_lut, i_lut, d_lut))]
fn p2_rgb<'py>(
    py:                  Python<'py>,
    h:                   usize,
    w:                   usize,
    gr_ch:               PyReadonlyArray2<'py, u8>,
    rd_ch:               PyReadonlyArray2<'py, u8>,
    bd_ch:               PyReadonlyArray2<'py, u8>,
    row_global_offsets:  PyReadonlyArray3<'py, u32>,
    shard_gr:            PyReadonlyArray1<'py, u8>,
    shard_rd:            PyReadonlyArray1<'py, u8>,
    shard_bd:            PyReadonlyArray1<'py, u8>,
    s_lut:               PyReadonlyArray2<'py, u8>,
    i_lut:               PyReadonlyArray1<'py, u8>,
    d_lut:               PyReadonlyArray2<'py, u8>,
) -> PyResult<()> {
    let gr  = gr_ch.as_slice()?;
    let rd  = rd_ch.as_slice()?;
    let bd  = bd_ch.as_slice()?;
    let rgo = row_global_offsets.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;

    // SAFETY: row_global_offsets guarantees non-overlapping row writes.
    let sgr_ro  = shard_gr.as_slice()?;
    let srd_ro  = shard_rd.as_slice()?;
    let sbd_ro  = shard_bd.as_slice()?;
    let sgr_mut = unsafe { std::slice::from_raw_parts_mut(sgr_ro.as_ptr() as *mut u8, sgr_ro.len()) };
    let srd_mut = unsafe { std::slice::from_raw_parts_mut(srd_ro.as_ptr() as *mut u8, srd_ro.len()) };
    let sbd_mut = unsafe { std::slice::from_raw_parts_mut(sbd_ro.as_ptr() as *mut u8, sbd_ro.len()) };

    py.allow_threads(|| {
        rans_core::shard_pass_2_rgb(h, w, gr, rd, bd, rgo, sgr_mut, srd_mut, sbd_mut, sl, il, dl);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// p2_gray  (Phase 5a: encode pass-2, grayscale)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, gr_ch, row_global_offsets, shard_gr, s_lut, i_lut, d_lut))]
fn p2_gray<'py>(
    py:                 Python<'py>,
    h:                  usize,
    w:                  usize,
    gr_ch:              PyReadonlyArray2<'py, u8>,
    row_global_offsets: PyReadonlyArray3<'py, u32>,
    shard_gr:           PyReadonlyArray1<'py, u8>,
    s_lut:              PyReadonlyArray2<'py, u8>,
    i_lut:              PyReadonlyArray1<'py, u8>,
    d_lut:              PyReadonlyArray2<'py, u8>,
) -> PyResult<()> {
    let gr  = gr_ch.as_slice()?;
    let rgo = row_global_offsets.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;

    let sgr_ro  = shard_gr.as_slice()?;
    let sgr_mut = unsafe { std::slice::from_raw_parts_mut(sgr_ro.as_ptr() as *mut u8, sgr_ro.len()) };

    py.allow_threads(|| {
        rans_core::shard_pass_2_gray(h, w, gr, rgo, sgr_mut, sl, il, dl);
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// reconstruct_shards_rgb  (Phase 5b: shard decode, RGB)
//
// Python signature:
//   (h, w, res_gr[M], res_rd[M], res_bd[M],
//    off_gr[n], off_rd[n], off_bd[n],
//    s_lut[511,511], i_lut[256], d_lut[256,4])
//   -> (gr_rec[h,w], rd_rec[h,w], bd_rec[h,w])
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, res_gr, res_rd, res_bd, off_gr, off_rd, off_bd, s_lut, i_lut, d_lut))]
fn reconstruct_shards_rgb<'py>(
    py:     Python<'py>,
    h:      usize,
    w:      usize,
    res_gr: PyReadonlyArray1<'py, u8>,
    res_rd: PyReadonlyArray1<'py, u8>,
    res_bd: PyReadonlyArray1<'py, u8>,
    off_gr: PyReadonlyArray1<'py, u32>,
    off_rd: PyReadonlyArray1<'py, u32>,
    off_bd: PyReadonlyArray1<'py, u32>,
    s_lut:  PyReadonlyArray2<'py, u8>,
    i_lut:  PyReadonlyArray1<'py, u8>,
    d_lut:  PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>)> {
    let gr  = res_gr.as_slice()?;
    let rd  = res_rd.as_slice()?;
    let bd  = res_bd.as_slice()?;
    let ogr = off_gr.as_slice()?;
    let ord = off_rd.as_slice()?;
    let obd = off_bd.as_slice()?;
    let sl  = s_lut.as_slice()?;
    let il  = i_lut.as_slice()?;
    let dl  = d_lut.as_slice()?;

    let (gr_out, rd_out, bd_out) = py.allow_threads(|| {
        rans_core::reconstruct_shards_rgb(h, w, gr, rd, bd, ogr, ord, obd, sl, il, dl)
    });

    let e = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok((
        PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), gr_out).map_err(e)?),
        PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), rd_out).map_err(e)?),
        PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), bd_out).map_err(e)?),
    ))
}

// ---------------------------------------------------------------------------
// restore_channels  (Phase 5c: inverse G-sub RCT)
//
// Python signature:
//   (gr_rec[h,w], rd_rec[h,w], bd_rec[h,w], a_ch[h,w|0,0],
//    is_rgba, is_grayscale, apply_gsub)
//   -> rgb[h,w,3|4]
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (gr_rec, rd_rec, bd_rec, a_ch, is_rgba, is_grayscale, apply_gsub))]
fn restore_channels<'py>(
    py:          Python<'py>,
    gr_rec:      PyReadonlyArray2<'py, u8>,
    rd_rec:      PyReadonlyArray2<'py, u8>,
    bd_rec:      PyReadonlyArray2<'py, u8>,
    a_ch:        PyReadonlyArray2<'py, u8>,
    is_rgba:     bool,
    is_grayscale: bool,
    apply_gsub:  bool,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let shape    = gr_rec.shape();
    let h        = shape[0];
    let w        = shape[1];
    let channels = if is_rgba { 4usize } else { 3 };

    let gr = gr_rec.as_slice()?;
    let rd = rd_rec.as_slice()?;
    let bd = bd_rec.as_slice()?;
    let a  = a_ch.as_slice()?;

    let rgb = py.allow_threads(|| {
        rans_core::restore_channels(gr, rd, bd, a, is_rgba, is_grayscale, apply_gsub, h, w)
    });

    let e   = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let arr = Array3::from_shape_vec((h, w, channels), rgb).map_err(e)?;
    Ok(PyArray3::from_owned_array(py, arr))
}

// ---------------------------------------------------------------------------
// reconstruct_2d_channels  (Phase 5c: sequential inverse MED prediction)
//
// Python signature:
//   (h, w, res_ch[h,w])  -> rec[h,w]
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h, w, res_ch))]
fn reconstruct_2d_channels<'py>(
    py:     Python<'py>,
    h:      usize,
    w:      usize,
    res_ch: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let rc  = res_ch.as_slice()?;
    let rec = py.allow_threads(|| rans_core::reconstruct_2d_channels(h, w, rc));
    let e   = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok(PyArray2::from_owned_array(py, Array2::from_shape_vec((h, w), rec).map_err(e)?))
}

// ---------------------------------------------------------------------------
// Legacy stub kept for backwards-compat with any external callers
// ---------------------------------------------------------------------------

#[pyfunction]
fn get_backend_info() -> PyResult<(String, String)> {
    Ok(("v8.3.2-rans-rust".to_string(), env!("CARGO_PKG_VERSION").to_string()))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn spx_rans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_backend_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_magic_lut, m)?)?;
    m.add_function(wrap_pyfunction!(build_pdf_tables_from_shards, m)?)?;
    m.add_function(wrap_pyfunction!(rans_encode_shards_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(rans_decode_shards_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(rans_decode_4way_core, m)?)?;
    m.add_function(wrap_pyfunction!(pack_shard_payloads, m)?)?;
    m.add_function(wrap_pyfunction!(build_all_lookups, m)?)?;
    m.add_function(wrap_pyfunction!(compact_pdf_tables, m)?)?;
    m.add_function(wrap_pyfunction!(expand_pdf_tables, m)?)?;
    m.add_function(wrap_pyfunction!(bp_decode_sharded_debug, m)?)?;
    m.add_function(wrap_pyfunction!(bp_build_pdf_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(bp_build_pdf_sharded_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(bp_encode_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(bp_decode_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(bp_decode_sharded_with_ref, m)?)?;
    m.add_function(wrap_pyfunction!(p1_gray, m)?)?;
    m.add_function(wrap_pyfunction!(p1_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(decide_shard_mode, m)?)?;
    m.add_function(wrap_pyfunction!(p2_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(p2_gray, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_shards_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(restore_channels, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_2d_channels, m)?)?;
    Ok(())
}
