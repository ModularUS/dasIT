"""The processing chain of beamform_image.ipynb as one function, for the tests.

Same calls, same order and same settings as the notebook, on one frame of the bundled example data. The blanks of
the teaching notebook (centre frequency, number of elements, pitch) are read from transducer.csv.
Returns the result of every stage, so a test can compare each of them with the golden model in tests/golden/.
"""
from pathlib import Path

import numpy as np

from dasIT.data.loader import RFDataloader, TDloader, TGCloader
from dasIT.features.transducer import transducer
from dasIT.features.medium import medium
from dasIT.features.tgc import tg_compensation
from dasIT.features.signal import RFfilter, analytic_signal, logcompression
from dasIT.features.image import interpolate_bf
from dasIT.src.delays import planewave_delays
from dasIT.src.apodization import apodization
from dasIT.src.das_bf import RXbeamformer

EXAMPLE = Path(__file__).resolve().parents[1] / "example_data" / "CIRSphantom_GE9LD_VVantage"
STAGES = ("rf", "tgc", "filtered", "analytic", "apodization", "delays", "beamformed", "envelope", "interpolated", "log_image")


def run_chain(apo="hanning", max_depth_samples=None, frame=0, lateral_scale=3, dbrange=58, data=EXAMPLE):
    """Run the chain on `frame`. apo: 'hanning', 'blackman' or 'rec'. max_depth_samples: the sample limit of the delay
    table (default: the cropped record length). Returns {stage name: array} for the names in STAGES."""
    data = Path(data)
    out = {}
    td = TDloader(str(data / "transducer.csv")).transducer
    column = lambda name, dtype="float": td[name].dropna().to_numpy(dtype=dtype, copy=False)

    T = transducer(center_frequency_hz=float(column("center frequency")[0]), bandwidth_hz=column("bandwidth"), adc_ratio=4,
                   transducer_elements_nr=int(column("number of elements")[0]), element_pitch_m=float(column("element pitch")[0]),
                   pinmap=column("pinmap", "int"), pinmapbase=1, elevation_focus=0.028, focus_number=None,
                   totalnr_planewaves=1, planewave_angle_interval=[0, 0], axial_cutoff_wavelength=5, speed_of_sound_ms=1540)
    M = medium(speed_of_sound_ms=1540, center_frequency=T.center_frequency, sampling_frequency=T.sampling_frequency,
               max_depth_wavelength=177, lateral_transducer_element_spacing=T.lateral_transducer_spacing,
               axial_extrapolation_coef=1.05, attenuation_coefficient=0.75, attenuation_power=1.5)

    # load, zero the samples before the first echo, crop to the imaging depth, sort the channels
    RF = RFDataloader(str(data / "CIRS_phantom.h5"))
    RF.signal[:T.start_depth_rec_samples, :, :] = 0
    RF.signal = RF.signal[:M.rx_echo_totalnr_samples, T.transducer_pinmap, :]
    out["rf"] = RF.signal[:, :, frame]

    control_points = TGCloader(controlpt_path=str(data / "tgc_cntrl_pt.csv")).tgc_control_points
    tgc = tg_compensation(signals=RF.signal, medium=M, center_frequency=T.center_frequency, cntrl_points=control_points, mode="points").signals
    out["tgc"] = tgc[:, :, frame]

    filtered = RFfilter(signals=tgc, fcutoff_band=T.bandwidth, fsampling=T.sampling_frequency, type="gaussian", order=10).signal
    out["filtered"] = filtered[:, :, frame]

    out["analytic"] = analytic_signal(np.squeeze(filtered), interp=False)[:, :, frame]

    out["apodization"] = apodization(delays=None, medium=M.medium, transducer=T, apo=apo, angles=T.planewave_angles()).table
    out["delays"] = planewave_delays(medium=M.medium, sos=M.speed_of_sound, fsampling=T.sampling_frequency, angles=T.planewave_angles(),
                                     max_depth_samples=max_depth_samples or M.rx_echo_totalnr_samples).sample_delays

    signals = np.expand_dims(out["analytic"], 2)
    signals = np.repeat(signals, signals.shape[1], axis=2)
    signals = np.expand_dims(signals, 3)
    out["beamformed"] = RXbeamformer(signals=signals, delays=out["delays"], apodization=out["apodization"]).frame
    out["envelope"] = abs(out["beamformed"])

    out["interpolated"] = interpolate_bf(signals=out["envelope"], transducer=T, medium=M, axial_scale=1, lateral_scale=lateral_scale).signals_interp
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_image"] = logcompression(out["interpolated"], dbrange)

    out["_geometry"] = dict(fs=T.sampling_frequency, c=M.speed_of_sound, x=np.asarray(T.lateral_transducer_spacing), z=np.asarray(M.medium[1][:, 0]),
                            pitch=T.element_pitch, n_elements=T.transducer_elements, elevation_focus=T.elevation_focus)
    return out
