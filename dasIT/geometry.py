'''
Licensed under the Apache License, Version 2.0, see the LICENSE file of the dasIT repository.

Geometry of an acquisition in SI units: where the elements are and when the transmitted wave reaches a point.
Coordinates: x along the array with 0 at its centre [m], z into the medium with 0 at the array surface [m].
'''

import numpy as np


def linear_array(n_elements, pitch):
    """x positions [m] of the element centres of a linear array, centred on x = 0.
    n_elements = 4, pitch = 1 mm gives -1.5, -0.5, 0.5, 1.5 mm: neighbours are exactly one pitch apart."""
    return (np.arange(n_elements) - (n_elements - 1) / 2) * pitch


def plane_wave(angle_deg=0.0, x0=0.0):
    """Transmit delay of a plane wave steered by angle_deg (positive: towards +x).

    Returns a function tx(x, z, c) -> time [s] at which the wavefront reaches the point (x, z), counted from the
    moment it passes the array surface at x0. x0 = 0 puts that moment at the array centre; a system that starts its
    clock when the first element fires passes the x of that element.

                x0
        ────────●──────── array, z = 0            tx = ( z cos(angle) + (x - x0) sin(angle) ) / c
               /
              /  wavefront, tilted by the angle
    """
    angle = np.radians(angle_deg)

    def tx(x, z, c):
        return (z * np.cos(angle) + (x - x0) * np.sin(angle)) / c

    return tx
