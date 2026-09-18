"""SIA 261 elastic response spectrum (section 16.2.3, edition 2020).

The code describes the seismic action with a 475-year (10% in 50 years)
design ground acceleration agd per seismic zone, amplified by the ground
class and by the importance of the structure, and shaped by four
period-dependent branches:

    T <= TB:        Se = agd * gamma_f * S * (1 + T / TB * (2.5 * eta - 1))
    TB <= T <= TC:  Se = 2.5 * eta * agd * gamma_f * S
    TC <= T <= TD:  Se = 2.5 * eta * agd * gamma_f * S * TC / T
    T >= TD:        Se = 2.5 * eta * agd * gamma_f * S * TC * TD / T**2

with eta the damping correction (1.0 at 5% damping). This is a uniform-hazard
envelope, not the spectrum of any single earthquake, which is what makes the
comparison with a scenario spectrum interesting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Design ground acceleration agd [m/s2] per seismic zone (section 16.2.1, Annex F map).
SEISMIC_ZONES = {"Z1a": 0.6, "Z1b": 0.8, "Z2": 1.0, "Z3a": 1.3, "Z3b": 1.6}

# Importance factor gamma_f per structure class (Bauwerksklasse, Table 25):
# I ordinary buildings, II larger occupancy or important infrastructure,
# III lifeline facilities such as hospitals and fire stations. The 2020
# partial revision raised class III from the 1.4 of the 2014 edition.
IMPORTANCE_FACTORS = {"I": 1.0, "II": 1.2, "III": 1.5}
STRUCTURE_CLASSES = tuple(IMPORTANCE_FACTORS)


@dataclass(frozen=True)
class GroundClass:
    soil_factor: float  # S
    tb: float  # s, start of the plateau
    tc: float  # s, end of the plateau
    td: float  # s, start of the constant-displacement branch
    vs30_range: tuple[float, float]  # m/s, indicative


# Table 24. The 2014 revision raised S for the softer classes and moved the
# plateau to shorter periods than the 2003 edition had; 2020 kept these values. Class E
# is a shallow (5-30 m) soft layer over rock and shares its Vs30 range with C
# and D; class F needs a site-specific study.
GROUND_CLASSES = {
    "A": GroundClass(1.00, 0.07, 0.25, 2.0, (800.0, float("inf"))),
    "B": GroundClass(1.20, 0.08, 0.35, 2.0, (500.0, 800.0)),
    "C": GroundClass(1.45, 0.10, 0.40, 2.0, (300.0, 500.0)),
    "D": GroundClass(1.70, 0.10, 0.50, 2.0, (150.0, 300.0)),
    "E": GroundClass(1.70, 0.09, 0.25, 2.0, (150.0, 500.0)),
}


def damping_correction(damping_percent: float) -> float:
    """eta = sqrt(10 / (5 + xi)) >= 0.55, equal to 1 at the 5% reference damping."""
    return max(float(np.sqrt(10.0 / (5.0 + damping_percent))), 0.55)


def ground_class_for_vs30(vs30: float) -> str:
    """The class whose Vs30 range contains `vs30`, ignoring the profile-dependent class E."""
    for name, ground in GROUND_CLASSES.items():
        low, high = ground.vs30_range
        if name != "E" and low <= vs30 < high:
            return name
    raise ValueError(f"Vs30 = {vs30} m/s is softer than any SIA 261 ground class A-D (class F needs a site study)")


def elastic_spectrum(
    periods: np.ndarray,
    agd: float,
    ground_class: str,
    structure_class: str = "I",
    damping_percent: float = 5.0,
) -> np.ndarray:
    """Horizontal elastic response spectrum Se(T) in m/s2 for the given periods in s."""
    ground = GROUND_CLASSES[ground_class]
    gamma_f = IMPORTANCE_FACTORS[structure_class]
    eta = damping_correction(damping_percent)
    t = np.asarray(periods, dtype=float)
    plateau = 2.5 * eta * agd * gamma_f * ground.soil_factor
    # T = 0 is a legitimate input (the PGA branch), so keep it out of the divisions.
    t_safe = np.maximum(t, ground.tb)
    return np.select(
        [t <= ground.tb, t <= ground.tc, t <= ground.td],
        [
            agd * gamma_f * ground.soil_factor * (1 + t / ground.tb * (2.5 * eta - 1)),
            plateau,
            plateau * ground.tc / t_safe,
        ],
        default=plateau * ground.tc * ground.td / t_safe**2,
    )
