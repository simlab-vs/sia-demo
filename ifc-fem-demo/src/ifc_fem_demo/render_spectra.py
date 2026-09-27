"""Stage 4 figure: the scenario spectrum against the SIA 261 spectra of each structure class, with the house's periods marked."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ifc_fem_demo import sia261
from ifc_fem_demo.seismic import HouseResponse, ScenarioSpectra

matplotlib.use("Agg")

OUTPUT_DIR = Path("output")
SCENARIO_COLOUR = "#2a78d6"
# One hue, light to dark, because the structure classes are ordered by importance factor.
CODE_COLOURS = {"I": "#f09a70", "II": "#eb6834", "III": "#a8420f"}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
PERIOD_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]


def render_spectra(spectra: ScenarioSpectra, responses: list[HouseResponse], output_path: Path, dpi: int = 150) -> Path:
    quake, site, code = spectra.scenario.earthquake, spectra.scenario.site, spectra.scenario.code
    t, median, plus_sigma = spectra.periods, spectra.median, spectra.median_plus_sigma

    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(t, median, plus_sigma, color=SCENARIO_COLOUR, alpha=0.12, linewidth=0)
    ax.plot(t, median, color=SCENARIO_COLOUR, linewidth=2, label=f"Mw {quake.magnitude:.1f} scenario, median (PGA {spectra.pga_median:.2f} m/s²)")
    ax.plot(
        t, plus_sigma, color=SCENARIO_COLOUR, linewidth=2, linestyle=(0, (4, 2)),
        label=f"scenario, median + 1 sigma (PGA {spectra.pga_plus_sigma:.2f} m/s²)",
    )
    for structure_class, colour in CODE_COLOURS.items():
        gamma_f = sia261.IMPORTANCE_FACTORS[structure_class]
        ax.plot(
            t, spectra.code_by_class[structure_class], color=colour, linewidth=2,
            label=f"SIA 261 elastic, zone {code.zone}, ground class {site.ground_class}, structure class {structure_class}, "
            f"γf = {gamma_f:.1f} (agd·γf·S = {spectra.code_pga_by_class[structure_class]:.2f} m/s²)",
        )

    _label(ax, t[np.argmax(plus_sigma)], plus_sigma.max(), "median + 1 sigma", above=True)
    _label(ax, t[np.argmax(median)], median.max(), "median", above=False)
    for structure_class, spectrum in spectra.code_by_class.items():
        plateau = t[spectrum >= spectrum.max() - 1e-9]
        _label(ax, np.sqrt(plateau[0] * plateau[-1]), spectrum.max(), f"SIA 261 class {structure_class}", above=True)

    top = 1.12 * max(plus_sigma.max(), max(spectrum.max() for spectrum in spectra.code_by_class.values()))
    # The two periods are usually close, so their labels are stacked instead of side by side.
    for row, r in enumerate(sorted(responses, key=lambda r: r.period)):
        ax.axvline(r.period, color=TEXT_SECONDARY, linewidth=1)
        ax.annotate(
            f"house {r.axis}: T = {r.period:.3f} s", (r.period, top), xytext=(4, -4 - 14 * row), textcoords="offset points",
            color=TEXT_SECONDARY, fontsize=8.5, ha="left", va="top",
        )

    ax.set_xscale("log")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, top)
    ax.set_xticks(PERIOD_TICKS, [f"{tick:g}" for tick in PERIOD_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("Period T [s]", color=TEXT)
    ax.set_ylabel(f"Spectral acceleration [m/s²], {code.damping_percent:g}% damping", color=TEXT)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GRID)

    ax.set_title(
        f"Mw {quake.magnitude:.1f} {quake.mechanism} earthquake, {spectra.hypocentral_km:.0f} km hypocentral distance, Vs30 {site.vs30:g} m/s "
        f"— against the SIA 261 elastic spectra",
        loc="left", fontsize=10.5, color=TEXT,
    )
    # Below the plot: inside it the legend would sit on the code plateau or the scenario band.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), frameon=False, fontsize=9, labelcolor=TEXT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _label(ax: plt.Axes, x: float, y: float, text: str, above: bool) -> None:
    ax.annotate(
        text, (x, y), xytext=(0, 6 if above else -6), textcoords="offset points",
        ha="center", va="bottom" if above else "top", fontsize=9, color=TEXT,
    )
