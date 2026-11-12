"""Golden model of dasIT: the processing chain of beamform_image.ipynb on the bundled example data, stage by stage.

    python make_golden.py <dasIT checkout> <output folder> [--label NAME]

Runs the notebook's cells (same calls, same order, same settings; the four blanks of the teaching notebook are filled
from example_data/.../transducer.csv, as the extended notebook does) on frame 0 of CIRS_phantom.h5 and writes
    golden.npz     the result of every stage (float32 / complex64 where the stage is large)
    golden.json    settings, library versions, per stage: shape, dtype, SHA-256 of the full-precision array, statistics,
                   run time and peak memory; and every deviation from the notebook that was needed to make it run
The stage results are what a refactored dasIT has to reproduce (or deviate from for a stated reason).
Works for both branch layouts (main: interp_lateral, apo 'rec'; usdatarecycler: interpolate_bf, apo 'hanning').
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("checkout")
p.add_argument("out")
p.add_argument("--label", default="")
p.add_argument("--frame", type=int, default=0)
a = p.parse_args()
ROOT, OUT = Path(a.checkout).resolve(), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
DATA = ROOT / "example_data" / "CIRSphantom_GE9LD_VVantage"

import matplotlib

matplotlib.use("Agg")
import scipy
import pandas
import h5py
from dasIT.data.loader import RFDataloader, TDloader, TGCloader
from dasIT.features.transducer import transducer
from dasIT.features.medium import medium
from dasIT.features.tgc import tg_compensation
from dasIT.src.delays import planewave_delays
from dasIT.src import apodization as apo_module
from dasIT.src.das_bf import RXbeamformer
from dasIT.features import signal as sig_module
from dasIT.features import image as image_module

SETTINGS = dict(adc_ratio=4, pinmapbase=1, elevation_focus=0.028, focus_number=None, totalnr_planewaves=1,
                planewave_angle_interval=[0, 0], axial_cutoff_wavelength=5, speed_of_sound_ms=1540,
                max_depth_wavelength=177, axial_extrapolation_coef=1.05, attenuation_coefficient=0.75, attenuation_power=1.5,
                tgc_mode="points", filter_type="gaussian", filter_order=10, lateral_interp_scale=3, dbrange=58, frame=a.frame)
stages, arrays, deviations = {}, {}, []


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def stage(name, fn, keep=True):
    """Run one stage, record time, peak memory (numpy allocations) and a fingerprint of its result."""
    tracemalloc.start()
    t = time.perf_counter()
    x = fn()
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    v = np.asarray(x)
    mag = np.abs(v)
    stages[name] = dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha(v), seconds=round(dt, 3), peak_mb=round(peak / 1e6, 1),
                        result_mb=round(v.nbytes / 1e6, 1), min=float(mag.min()), max=float(mag.max()), mean=float(mag.mean()),
                        nonfinite=int((~np.isfinite(v)).sum()))
    if keep:
        arrays[name] = v.astype(np.complex64 if np.iscomplexobj(v) else (np.float32 if v.dtype.kind == "f" else v.dtype))
    print(f"{name:14s} {str(v.shape):22s} {str(v.dtype):11s} {dt:7.2f} s  peak {peak / 1e6:8.1f} MB  mean|x| {mag.mean():.6g}")
    return x


# ---- transducer and medium (notebook cells 13 and 16)
td = TDloader(str(DATA / "transducer.csv")).transducer
col = lambda name, dtype="float": td[name].dropna().to_numpy(dtype=dtype, copy=False)
SETTINGS.update(center_frequency_hz=float(col("center frequency")[0]), transducer_elements_nr=int(col("number of elements")[0]),
                element_pitch_m=float(col("element pitch")[0]), bandwidth_hz=col("bandwidth").tolist())
T = transducer(center_frequency_hz=SETTINGS["center_frequency_hz"], bandwidth_hz=col("bandwidth"), adc_ratio=SETTINGS["adc_ratio"],
               transducer_elements_nr=SETTINGS["transducer_elements_nr"], element_pitch_m=SETTINGS["element_pitch_m"],
               pinmap=col("pinmap", "int"), pinmapbase=1, elevation_focus=SETTINGS["elevation_focus"], focus_number=None,
               totalnr_planewaves=1, planewave_angle_interval=[0, 0], axial_cutoff_wavelength=5, speed_of_sound_ms=1540)
M = medium(speed_of_sound_ms=1540, center_frequency=T.center_frequency, sampling_frequency=T.sampling_frequency,
           max_depth_wavelength=177, lateral_transducer_element_spacing=T.lateral_transducer_spacing,
           axial_extrapolation_coef=1.05, attenuation_coefficient=0.75, attenuation_power=1.5)
geometry = dict(sampling_frequency_hz=T.sampling_frequency, wavelength_m=T.wavelength, start_depth_samples=T.start_depth_rec_samples,
                rx_echo_samples=M.rx_echo_totalnr_samples, element_x_first_last_m=[float(T.lateral_transducer_spacing[0]), float(T.lateral_transducer_spacing[-1])],
                element_spacing_m=float(np.diff(T.lateral_transducer_spacing)[0]), grid_nz=int(M.medium[1].size), grid_nx=int(M.medium[0].size),
                grid_dz_m=float(M.medium[1][1, 0] - M.medium[1][0, 0]), grid_z_last_m=float(M.medium[1][-1, 0]))
print("geometry:", geometry)

# ---- load, zero the start, crop, sort (cell 19)
RF = RFDataloader(str(DATA / "CIRS_phantom.h5"))
raw_shape = list(RF.signal.shape)
RF.signal[:T.start_depth_rec_samples, :, :] = 0
RF.signal = RF.signal[:M.rx_echo_totalnr_samples, T.transducer_pinmap, :]
stage("rf", lambda: RF.signal[:, :, a.frame])

# ---- TGC (cell 26). main takes the loader object, usdatarecycler the table itself
ctrl = TGCloader(controlpt_path=str(DATA / "tgc_cntrl_pt.csv"))
try:
    TGC = tg_compensation(signals=RF.signal, medium=M, center_frequency=T.center_frequency, cntrl_points=ctrl, mode="points")
    tgc_signals = TGC.signals
except Exception:                                    # noqa: BLE001
    TGC = tg_compensation(signals=RF.signal, medium=M, center_frequency=T.center_frequency, cntrl_points=ctrl.tgc_control_points, mode="points")
    tgc_signals = TGC.signals
    deviations.append("tg_compensation takes the control-point table (cntrl_points=loader.tgc_control_points), not the loader")
stage("tgc", lambda: tgc_signals[:, :, a.frame])

# ---- filter (cell 29)
kw = dict(signals=tgc_signals, fcutoff_band=T.bandwidth, fsampling=T.sampling_frequency, type="gaussian", order=10)
try:
    filtered = sig_module.RFfilter(**kw).signal
except Exception as e:                               # noqa: BLE001
    deviations.append(f"RFfilter fails as written in the notebook ({type(e).__name__}: {e}). Golden uses the same coefficients "
                      f"applied along the sample axis with a kernel of shape (order, 1, 1), as at tag v1.0")
    f = object.__new__(sig_module.RFfilter)
    f._fcutoff_band, f._fsampling, f._ftype, f._forder = T.bandwidth, T.sampling_frequency, "gaussian", 10
    filtered = sig_module.convolve(tgc_signals, f.bandpass_firwin().reshape(10, 1, 1), mode="same")
f = object.__new__(sig_module.RFfilter)
f._fcutoff_band, f._fsampling, f._ftype, f._forder = T.bandwidth, T.sampling_frequency, "gaussian", 10
arrays["filter_coefficients"] = np.ravel(f.bandpass_firwin()).astype(np.float64)
stage("filtered", lambda: filtered[:, :, a.frame])

# ---- analytic signal (cell 32)
analytic = stage("analytic", lambda: sig_module.analytic_signal(np.squeeze(filtered), interp=False)[:, :, a.frame])

# ---- apodization (cell 36) and delays (cell 38)
apo_name = "rec" if hasattr(apo_module.apodization, "single_channel_apodization") else "hanning"
SETTINGS["apodization"] = apo_name
if apo_name != "rec":
    deviations.append("apodization 'rec' of the notebook does not exist on this branch, 'hanning' is used")
APO = stage("apodization", lambda: apo_module.apodization(delays=None, medium=M.medium, transducer=T, apo=apo_name, angles=T.planewave_angles()).table, keep=False)
arrays["apodization_elements_per_depth"] = (np.asarray(APO)[:, 96, :, 0] != 0).sum(axis=1).astype(np.int32)
arrays["apodization_centre_column"] = np.asarray(APO)[:, 96, :, 0].astype(np.float32)
try:
    D = planewave_delays(medium=M.medium, sos=M.speed_of_sound, fsampling=T.sampling_frequency, angles=T.planewave_angles(),
                         max_depth_samples=M.rx_echo_totalnr_samples)
    SETTINGS["max_depth_samples"] = M.rx_echo_totalnr_samples
except TypeError:
    D = planewave_delays(medium=M.medium, sos=M.speed_of_sound, fsampling=T.sampling_frequency, angles=T.planewave_angles())
    SETTINGS["max_depth_samples"] = "1360, fixed in delays.py"
delays = stage("delays", lambda: D.sample_delays, keep=False)
arrays["delays_centre_column"] = np.asarray(delays)[:, 96, :, 0].astype(np.int32)        # depth x element, pixel column 96
stages["delays"]["set_to_zero_by_the_depth_limit"] = int((np.asarray(delays) == 0).sum())

# ---- beamforming (cell 40)
def beamform():
    s = analytic
    s = np.expand_dims(s, 2)
    s = np.repeat(s, s.shape[1], axis=2)
    s = np.expand_dims(s, 3)
    return RXbeamformer(signals=s, delays=D.sample_delays, apodization=APO).frame


bf = stage("beamformed", beamform)
env = stage("envelope", lambda: abs(bf))

# ---- lateral interpolation (cell 42) and log compression (inside plot_signal_grid)
if hasattr(image_module, "interp_lateral"):
    interp = stage("interpolated", lambda: image_module.interp_lateral(signals=env, transducer=T, medium=M, scale=3).signals_lateral_interp)
else:
    interp = stage("interpolated", lambda: image_module.interpolate_bf(signals=env, transducer=T, medium=M, axial_scale=1, lateral_scale=3).signals_interp)
    deviations.append("interp_lateral does not exist on this branch, interpolate_bf(axial_scale=1, lateral_scale=3) is used")
with np.errstate(divide="ignore", invalid="ignore"):
    stage("log_image", lambda: sig_module.logcompression(interp, SETTINGS["dbrange"]))

np.savez_compressed(OUT / "golden.npz", **arrays)
manifest = dict(label=a.label, checkout=str(ROOT), data_file_sha256=hashlib.sha256((DATA / "CIRS_phantom.h5").read_bytes()).hexdigest(),
                raw_shape_samples_channels_frames=raw_shape, settings=SETTINGS, geometry=geometry,
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pandas.__version__, h5py=h5py.__version__),
                deviations_from_the_notebook=deviations, stages=stages,
                note="golden.npz holds float32 / complex64 copies; the SHA-256 values are of the full-precision arrays")
(OUT / "golden.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print("deviations:", *deviations, sep="\n  - ")
print("written:", OUT / "golden.npz", OUT / "golden.json")
