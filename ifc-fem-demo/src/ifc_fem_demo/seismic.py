"""Stage 4: a reproducible earthquake scenario for the house.

One specific earthquake (magnitude, epicentre, depth, mechanism) is put next
to what the code asks for at the same site:

- the median and median + 1 sigma response spectrum that the Akkar,
  Sandikkaya & Bommer (2014) ground-motion model predicts for that event at
  the site, given its distance and Vs30, and
- the SIA 261 elastic response spectrum for the seismic zone and ground
  class, for each structure class I, II and III.

The house's natural periods from the PyNite model then say where on those
spectra this particular building sits and what base shear that implies.
Every input is a dataclass field, so a scenario is reproduced by its
parameters alone (or by the command line flags of `python -m ifc_fem_demo.seismic`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ifc_fem_demo import ground_motion, sia261, structural
from ifc_fem_demo.add_window import WINDOW_MODEL_PATH

STANDARD_GRAVITY = 9.80665  # m/s2 per g
EARTH_RADIUS_KM = 6371.0
REPORT_PERIODS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Earthquake:
    magnitude: float = 5.0  # Mw
    epicentre_lat: float = 46.35  # deg, Sierre area
    epicentre_lon: float = 7.40
    depth_km: float = 10.0  # hypocentral depth
    rake_deg: float = 0.0  # 0 strike-slip, -90 normal, +90 reverse

    @property
    def mechanism(self) -> str:
        normal, reverse = ground_motion.mechanism_flags(self.rake_deg)
        return "normal" if normal else "reverse" if reverse else "strike-slip"


@dataclass(frozen=True)
class Site:
    lat: float = 46.292  # deg, Sierre town centre
    lon: float = 7.532
    vs30: float = 400.0  # m/s, within SIA 261 ground class C
    ground_class: str = "C"  # SIA 261 ground class A-E


@dataclass(frozen=True)
class DesignCode:
    zone: str = "Z3b"  # SIA 261 seismic zone
    structure_class: str = "I"  # SIA 261 Bauwerksklasse I, II or III
    damping_percent: float = 5.0

    @property
    def agd(self) -> float:
        return sia261.SEISMIC_ZONES[self.zone]

    @property
    def importance_factor(self) -> float:
        return sia261.IMPORTANCE_FACTORS[self.structure_class]


@dataclass(frozen=True)
class Scenario:
    earthquake: Earthquake = Earthquake()
    site: Site = Site()
    code: DesignCode = DesignCode()


def epicentral_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a spherical Earth (haversine formula)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    half_chord = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord)))


def log_interpolate(periods: np.ndarray, values: np.ndarray, period: float) -> float:
    """Value at `period`, interpolating log-log between tabulated periods and clamping outside."""
    return float(np.exp(np.interp(np.log(period), np.log(periods), np.log(values))))


@dataclass(frozen=True)
class ScenarioSpectra:
    """Both spectra at the site, in m/s2, on the ground-motion model's period grid."""

    scenario: Scenario
    epicentral_km: float
    hypocentral_km: float
    periods: np.ndarray  # s, 0.01 to 4
    median: np.ndarray  # scenario, median
    sigma: np.ndarray  # scenario, total standard deviation of ln(SA)
    pga_median: float
    pga_sigma: float
    code_by_class: dict[str, np.ndarray]  # SIA 261 elastic spectrum Se(T) per structure class
    code_pga_by_class: dict[str, float]  # Se(T = 0) = agd * gamma_f * S

    @property
    def code(self) -> np.ndarray:
        """Se(T) for the structure class the scenario is configured with."""
        return self.code_by_class[self.scenario.code.structure_class]

    @property
    def code_pga(self) -> float:
        return self.code_pga_by_class[self.scenario.code.structure_class]

    @property
    def median_plus_sigma(self) -> np.ndarray:
        return self.median * np.exp(self.sigma)

    @property
    def pga_plus_sigma(self) -> float:
        return self.pga_median * np.exp(self.pga_sigma)

    def code_at(self, period: float) -> dict[str, float]:
        """SIA 261 spectral acceleration at one period, per structure class."""
        code, ground_class = self.scenario.code, self.scenario.site.ground_class
        return {
            structure_class: float(sia261.elastic_spectrum(np.array([period]), code.agd, ground_class, structure_class, code.damping_percent)[0])
            for structure_class in sia261.STRUCTURE_CLASSES
        }

    def at(self, period: float) -> tuple[float, float, float]:
        """(median, median + 1 sigma, SIA 261 for the configured class) spectral accelerations at one period."""
        return (
            log_interpolate(self.periods, self.median, period),
            log_interpolate(self.periods, self.median_plus_sigma, period),
            self.code_at(period)[self.scenario.code.structure_class],
        )


def scenario_spectra(scenario: Scenario = Scenario()) -> ScenarioSpectra:
    quake, site, code = scenario.earthquake, scenario.site, scenario.code
    epicentral = epicentral_distance_km(quake.epicentre_lat, quake.epicentre_lon, site.lat, site.lon)
    hypocentral = float(np.hypot(epicentral, quake.depth_km))
    prediction = ground_motion.predict(quake.magnitude, hypocentral, quake.rake_deg, site.vs30)
    accelerations = prediction.median * STANDARD_GRAVITY
    periods = prediction.periods[1:]
    spectra = {
        structure_class: sia261.elastic_spectrum(np.append(0.0, periods), code.agd, site.ground_class, structure_class, code.damping_percent)
        for structure_class in sia261.STRUCTURE_CLASSES
    }
    return ScenarioSpectra(
        scenario=scenario,
        epicentral_km=epicentral,
        hypocentral_km=hypocentral,
        periods=periods,
        median=accelerations[1:],
        sigma=prediction.sigma[1:],
        pga_median=float(accelerations[0]),
        pga_sigma=float(prediction.sigma[0]),
        code_by_class={structure_class: spectrum[1:] for structure_class, spectrum in spectra.items()},
        code_pga_by_class={structure_class: float(spectrum[0]) for structure_class, spectrum in spectra.items()},
    )


@dataclass(frozen=True)
class HouseResponse:
    """Spectral accelerations at the house's dominant period along one horizontal axis."""

    axis: str
    period: float  # s
    mass_share: float  # effective modal mass of that mode over the seismic mass
    seismic_mass: float  # kg
    median: float  # m/s2
    median_plus_sigma: float
    code: float  # SIA 261, configured structure class
    code_by_class: dict[str, float]

    def base_shear(self, acceleration: float) -> float:
        """N, if the whole seismic mass responded at `acceleration` (a rigid-box estimate)."""
        return acceleration * self.seismic_mass


def house_response(spectra: ScenarioSpectra, modes: structural.VibrationModes) -> list[HouseResponse]:
    responses = []
    for axis, name in enumerate("XY"):
        period = modes.dominant_period(axis)
        responses.append(
            HouseResponse(
                name, period, float(modes.effective_mass_fraction[:, axis].max()), modes.seismic_mass, *spectra.at(period), spectra.code_at(period)
            )
        )
    return responses


SCENARIO_LEVELS = ("median", "median+1sigma")
LEVELS = SCENARIO_LEVELS + tuple(f"SIA {structure_class}" for structure_class in sia261.STRUCTURE_CLASSES)


def levels(response: HouseResponse) -> dict[str, float]:
    """Spectral acceleration at the house's period, per level: scenario median, + 1 sigma, SIA 261 per structure class."""
    code = {f"SIA {structure_class}": a for structure_class, a in response.code_by_class.items()}
    return {"median": response.median, "median+1sigma": response.median_plus_sigma} | code


def lateral_loads(responses: list[HouseResponse]) -> list[structural.LateralLoad]:
    """Ten equivalent static cases: each horizontal axis at every level."""
    return [
        structural.LateralLoad(f"EQ {r.axis} {level}", "XY".index(r.axis), acceleration, level)
        for r in responses
        for level, acceleration in levels(r).items()
    ]


def governing_case(results: dict[str, structural.AnalysisResult]) -> structural.AnalysisResult:
    return max(results.values(), key=lambda r: r.max_von_mises)


def describe(spectra: ScenarioSpectra) -> str:
    quake, site, code = spectra.scenario.earthquake, spectra.scenario.site, spectra.scenario.code
    suggested = sia261.ground_class_for_vs30(site.vs30)
    class_note = "" if suggested == site.ground_class else f" (Vs30 alone would suggest class {suggested})"
    return "\n".join(
        [
            f"Earthquake: Mw {quake.magnitude:.1f}, {quake.mechanism} (rake {quake.rake_deg:g} deg), "
            f"epicentre {quake.epicentre_lat:.3f} N {quake.epicentre_lon:.3f} E, {quake.depth_km:g} km deep",
            f"Site: {site.lat:.3f} N {site.lon:.3f} E, Vs30 = {site.vs30:g} m/s, SIA 261 ground class {site.ground_class}{class_note}",
            f"Distances: epicentral {spectra.epicentral_km:.1f} km, hypocentral {spectra.hypocentral_km:.1f} km",
            f"Ground motion: Akkar, Sandikkaya & Bommer (2014), hypocentral form, {code.damping_percent:g}% damping",
            f"Code: SIA 261 zone {code.zone} (agd = {code.agd:g} m/s2), S = {sia261.GROUND_CLASSES[site.ground_class].soil_factor:g}, "
            f"structure classes " + ", ".join(f"{c} (gamma_f = {f:.1f})" for c, f in sia261.IMPORTANCE_FACTORS.items())
            + f"; configured class {code.structure_class}",
        ]
    )


def spectra_table(spectra: ScenarioSpectra, periods=REPORT_PERIODS) -> str:
    classes = sia261.STRUCTURE_CLASSES
    configured = spectra.scenario.code.structure_class
    header = (
        f"{'T [s]':<8}{'median':>10}{'+1 sigma':>10}" + "".join(f"{'SIA ' + c:>10}" for c in classes)
        + f"{'sigma ln':>10}{'SIA ' + configured + '/median':>17}"
    )
    rows = [header, "-" * len(header)]

    def row(label: str, median: float, plus_sigma: float, code: dict[str, float], sigma: float) -> str:
        return (
            f"{label:<8}{median:>10.3f}{plus_sigma:>10.3f}" + "".join(f"{code[c]:>10.3f}" for c in classes)
            + f"{sigma:>10.3f}{code[configured] / median:>17.1f}"
        )

    rows.append(row("PGA", spectra.pga_median, spectra.pga_plus_sigma, spectra.code_pga_by_class, spectra.pga_sigma))
    for period in periods:
        median, plus_sigma, _ = spectra.at(period)
        sigma = np.log(log_interpolate(spectra.periods, np.exp(spectra.sigma), period))
        rows.append(row(f"{period:g}", median, plus_sigma, spectra.code_at(period), sigma))
    rows.append("\nSpectral accelerations in m/s2.")
    return "\n".join(rows)


def house_table(responses: list[HouseResponse]) -> str:
    """One row per level, the acceleration at each axis's dominant period and the rigid-box base shear it implies."""
    columns = "".join(f"{f'{r.axis}: a [m/s2]':>16}{'V [kN]':>9}" for r in responses)
    header = f"{'level':<16}{columns}"
    rows = [
        "; ".join(f"{r.axis}: T1 = {r.period:.4f} s, {r.mass_share:.0%} of the mass" for r in responses),
        "",
        header,
        "-" * len(header),
    ]
    per_axis = [levels(r) for r in responses]
    for level in LEVELS:
        cells = "".join(f"{accelerations[level]:>16.3f}{r.base_shear(accelerations[level]) / 1e3:>9.0f}" for r, accelerations in zip(responses, per_axis))
        rows.append(f"{level:<16}{cells}")
    rows.append(f"\nSeismic mass {responses[0].seismic_mass / 1e3:.1f} t; base shear V = SA x mass, as if the whole house moved as one rigid box.")
    return "\n".join(rows)


def main(
    scenario: Scenario = Scenario(),
    model_path: Path = WINDOW_MODEL_PATH,
    mesh_size: float = 0.2,
) -> tuple[ScenarioSpectra, list[HouseResponse]]:
    spectra = scenario_spectra(scenario)
    print(describe(spectra))
    modes = structural.vibration_modes(model_path, mesh_size)
    responses = house_response(spectra, modes)
    print(f"House: {len(modes.periods)} modes, longest period {modes.periods[0]:.4f} s")
    return spectra, responses


def _add_dataclass_arguments(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    for f in fields(cls):
        parser.add_argument(f"--{prefix}{f.name.replace('_', '-')}", type=type(f.default), default=f.default, help=f"default {f.default}")


def scenario_from_args(argv: list[str] | None = None) -> Scenario:
    parser = argparse.ArgumentParser(description="Seismic scenario spectra against SIA 261, with the house's periods.")
    _add_dataclass_arguments(parser, Earthquake)
    _add_dataclass_arguments(parser, Site, prefix="site-")
    _add_dataclass_arguments(parser, DesignCode)
    args = vars(parser.parse_args(argv))
    pick = lambda cls, prefix="": cls(**{f.name: args[prefix + f.name] for f in fields(cls)})
    return Scenario(pick(Earthquake), pick(Site, "site_"), pick(DesignCode))


if __name__ == "__main__":
    from ifc_fem_demo import render_3d
    from ifc_fem_demo.render_spectra import OUTPUT_DIR, render_spectra

    spectra, responses = main(scenario_from_args())
    print("\n" + spectra_table(spectra) + "\n\n" + house_table(responses))
    print(f"\nWrote {render_spectra(spectra, responses, OUTPUT_DIR / 'seismic_spectra.png')}")
    loads = lateral_loads(responses)
    results = structural.analyse_lateral(WINDOW_MODEL_PATH, loads)
    print("\n" + structural.lateral_table(results, loads))
    render_3d.render_seismic_comparison(results, loads, OUTPUT_DIR / "seismic_stress_comparison.png")
