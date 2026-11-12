"""Inputs that used to fail late or with an unrelated message, and the two interfaces kept from dasIT v1.0."""
import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from chain import EXAMPLE
from dasIT.data.loader import TDloader, TGCloader
from dasIT.features.image import interp_lateral, interpolate_bf
from dasIT.features.medium import medium
from dasIT.features.tgc import tg_compensation
from dasIT.features.transducer import transducer
from dasIT.src.apodization import apodization
from dasIT.visualization.image_callback import plot_signal_grid


def small_setup(**kw):
    """A 16-element probe and a shallow medium: enough for the interface, built in milliseconds."""
    T = transducer(center_frequency_hz=5e6, bandwidth_hz=np.array([3e6, 7e6]), adc_ratio=4, transducer_elements_nr=16,
                   element_pitch_m=3e-4, elevation_focus=0.02, speed_of_sound_ms=1540, **kw)
    M = medium(speed_of_sound_ms=1540, center_frequency=T.center_frequency, sampling_frequency=T.sampling_frequency,
               max_depth_wavelength=20, lateral_transducer_element_spacing=T.lateral_transducer_spacing,
               attenuation_coefficient=0.75, attenuation_power=1.5)
    return T, M


def test_unknown_apodization_names_the_choices():
    T, M = small_setup(pinmap=np.arange(1, 17))
    with pytest.raises(ValueError, match="'rec', 'hanning' or 'blackman'"):
        apodization(medium=M.medium, transducer=T, apo="mask", angles=T.planewave_angles())
    for name in ("rec", "hanning", "blackman"):
        assert apodization(medium=M.medium, transducer=T, apo=name, angles=T.planewave_angles()).table.shape == (80, 16, 16, 1)


def test_unknown_tgc_mode_names_the_choices():
    T, M = small_setup(pinmap=np.arange(1, 17))
    with pytest.raises(ValueError, match="'points' or 'alpha'"):
        tg_compensation(signals=np.ones((160, 16, 1)), medium=M, center_frequency=T.center_frequency, mode="curve")


def test_transducer_without_pinmap_keeps_the_channel_order():
    T, _ = small_setup()
    assert np.array_equal(T.transducer_pinmap, np.arange(16))
    assert np.array_equal(small_setup(pinmap=np.arange(1, 17))[0].transducer_pinmap, np.arange(16))


def test_transducer_takes_one_element_arrays_as_read_from_csv():
    T = transducer(center_frequency_hz=np.array([5e6]), bandwidth_hz=np.array([3e6, 7e6]), adc_ratio=4,
                   transducer_elements_nr=np.array([16.0]), element_pitch_m=np.array([3e-4]), speed_of_sound_ms=1540)
    assert (T.center_frequency, T.transducer_elements, T.element_pitch) == (5e6, 16, 3e-4)


@pytest.mark.skipif(not (EXAMPLE / "tgc_cntrl_pt.csv").exists(), reason="example data not present")
def test_tgc_accepts_the_loader_and_its_table():
    _, M = small_setup()
    loader = TGCloader(controlpt_path=str(EXAMPLE / "tgc_cntrl_pt.csv"))
    x = np.ones((160, 16, 2))
    a = tg_compensation(signals=x, medium=M, center_frequency=5e6, cntrl_points=loader, mode="points").signals
    b = tg_compensation(signals=x, medium=M, center_frequency=5e6, cntrl_points=loader.tgc_control_points, mode="points").signals
    assert np.array_equal(a, b) and a[0, 0, 0] == 153 and a.shape == x.shape


def test_interp_lateral_is_the_lateral_part_of_interpolate_bf():
    T, M = small_setup()
    env = np.abs(np.random.default_rng(0).standard_normal((40, 16)))
    old = interp_lateral(signals=env, transducer=T, medium=M, scale=3)
    new = interpolate_bf(signals=env, transducer=T, medium=M, axial_scale=1, lateral_scale=3)
    assert old.signals_lateral_interp.shape == (40, 48)
    assert np.array_equal(old.signals_lateral_interp, new.signals_interp)
    assert np.array_equal(old.imagegrid_mm[0], new.imagegrid_mm[0])


def test_axial_clip_of_zero_rows_keeps_the_image():
    image = np.abs(np.random.default_rng(1).standard_normal((40, 48))) + 1
    grid = [np.linspace(0, 4.8, 48), np.linspace(0, 5.9, 40)]
    with contextlib.redirect_stdout(io.StringIO()):
        plot_signal_grid(signals=image, axis_vectors_xz=grid, axial_clip=[0, None], compression=True, dbrange=40)
    assert len(grid[1]) == 40                                 # nothing was cut
    matplotlib.pyplot.close("all")
