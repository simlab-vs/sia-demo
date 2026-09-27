"""Compare the two survey campaigns: per-element cloud-to-cloud distances and deck soffit changes.

Writes build/campaign_comparison.json and build/campaign_comparison.png.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from bridge_analysis import ArcAlignment, VerticalProfile, lowest_surface, station_bins
from load_scans import CACHE_DIR, campaigns, load_element

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SAMPLE = 200_000


def cloud_to_cloud(reference: np.ndarray, compared: np.ndarray, rng) -> dict:
    sample = compared[rng.choice(len(compared), min(SAMPLE, len(compared)), replace=False)]
    d, idx = cKDTree(reference).query(sample, distance_upper_bound=1.0)
    ok = np.isfinite(d)
    shift = np.median(sample[ok] - reference[idx[ok]], axis=0)
    return {
        "median_m": float(np.median(d[ok])),
        "p95_m": float(np.percentile(d[ok], 95)),
        "unmatched_share": float(1 - ok.mean()),
        "median_shift_xyz_m": shift.round(4).tolist(),
    }


def soffit_profile(align: ArcAlignment, deck: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    s, o = align.station_offset(deck[:, :2])
    stations, idx = station_bins(s, lo, hi)
    z = np.full(len(stations), np.nan)
    near_axis = np.abs(o) < 2.5
    for i in range(len(stations)):
        m = near_axis & (idx == i)
        if m.sum() > 50:
            z[i] = lowest_surface(deck[m, 2])
    return stations, z


def main() -> None:
    first, second = campaigns()[0], campaigns()[-1]
    model = json.loads((Path("build") / first / "bridge_model.json").read_text())
    a = model["alignment"]
    align = ArcAlignment(tuple(a["centre"]), a["radius"], a["direction"], a["theta0"], a["length"])
    rng = np.random.default_rng(0)

    keys = sorted(f.stem for f in (CACHE_DIR / first).glob("*.npz"))
    c2c = {k: cloud_to_cloud(load_element(first, k), load_element(second, k), rng) for k in keys}

    deck = {c: np.concatenate([load_element(c, k) for k in keys if k.startswith("Tab_")]) for c in (first, second)}
    lo, hi = model["deck"]["s_start"] + 1, model["deck"]["s_end"] - 1
    stations, z1 = soffit_profile(align, deck[first], lo, hi)
    _, z2 = soffit_profile(align, deck[second], lo, hi)
    dz = z2 - z1
    pier_stations = [p["station"] for p in model["piers"]]
    away = np.array([min(abs(s - p) for p in pier_stations) > 3 for s in stations]) & np.isfinite(dz)

    report = {
        "reference_campaign": first,
        "compared_campaign": second,
        "cloud_to_cloud": c2c,
        "soffit_change": {
            "median_m": float(np.median(dz[away])),
            "min_m": float(np.min(dz[away])),
            "max_m": float(np.max(dz[away])),
            "std_m": float(np.std(dz[away])),
        },
    }
    Path("build/campaign_comparison.json").write_text(json.dumps(report, indent=1))

    fig, ax = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    ax[0].plot(stations[away] - model["deck"]["s_start"], dz[away] * 1000, ".", ms=3, color="#2a6fdb")
    for p, s in zip(model["piers"], pier_stations):
        ax[0].axvline(s - model["deck"]["s_start"], color="0.7", lw=0.8)
        ax[0].text(s - model["deck"]["s_start"], ax[0].get_ylim()[1], p["name"], ha="center", va="bottom")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("chainage from Est deck end [m]")
    ax[0].set_ylabel(f"soffit z ({second}) − ({first}) [mm]")
    ax[0].grid(alpha=0.3)
    names = list(c2c)
    ax[1].bar(names, [c2c[n]["median_m"] * 1000 for n in names], color="#2a6fdb", label="median")
    ax[1].plot(names, [c2c[n]["p95_m"] * 1000 for n in names], "o", color="#d0342c", label="95th percentile")
    ax[1].set_ylabel("cloud-to-cloud distance [mm]")
    ax[1].tick_params(axis="x", rotation=60)
    ax[1].legend()
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("build/campaign_comparison.png", dpi=90)

    print(json.dumps(report["soffit_change"], indent=1))
    for k, v in c2c.items():
        print(f"{k:12s} median {v['median_m'] * 1000:5.1f} mm  p95 {v['p95_m'] * 1000:6.1f} mm  shift {v['median_shift_xyz_m']}")


if __name__ == "__main__":
    main()
