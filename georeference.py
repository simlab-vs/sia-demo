"""Georeference the local survey frame to LV95 / LN02 and fetch swissALTI3D terrain.

The scans are in an unreferenced local frame. A 4-parameter transform (plan rotation,
plan translation, height offset) is estimated in two steps:

1. coarse: the deck ends are matched to the ends of the swisstopo gazetteer road
   "Pont de la Praz" (swissTLM3D);
2. fine: the scanned top surface is matched to the swissSURFACE3D surface model (which,
   unlike swissALTI3D, keeps the bridge deck) and the scanned ground around the pier bases
   to the swissALTI3D terrain model, by a grid search then robust least squares.
Writes build/<campaign>/georeference.json and caches tiles in build/swisstopo/.
"""
import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from bridge_analysis import ArcAlignment
from load_scans import CACHE_DIR, campaigns, load_element

GEOADMIN_SEARCH = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
STAC_ITEMS = "https://data.geo.admin.ch/api/stac/v0.9/collections/{collection}/items"
TILE_DIR = Path("build/swisstopo")
BRIDGE_NAME = "Pont de la Praz"
SURFACE_MODEL = "ch.swisstopo.swisssurface3d-raster"
TERRAIN_MODEL = "ch.swisstopo.swissalti3d"
TERRAIN_MARGIN = 450.0
MATCH_CELL = 1.0
PIER_RING_INNER_RADIUS = 1.5


@dataclass
class Georeference:
    """LV95 = R(rotation) @ (local_xy - pivot) + pivot_lv95 ; LN02 = local_z + dz."""

    rotation: float
    pivot: np.ndarray
    pivot_lv95: np.ndarray
    dz: float

    def _matrix(self, sign=1.0) -> np.ndarray:
        c, s = np.cos(self.rotation), np.sin(sign * self.rotation)
        return np.array([[c, -s], [s, c]])

    def to_lv95(self, xyz: np.ndarray) -> np.ndarray:
        xy = (xyz[:, :2] - self.pivot) @ self._matrix().T + self.pivot_lv95
        return np.c_[xy, xyz[:, 2] + self.dz]

    def to_local(self, enz: np.ndarray) -> np.ndarray:
        xy = (enz[:, :2] - self.pivot_lv95) @ self._matrix(-1).T + self.pivot
        return np.c_[xy, enz[:, 2] - self.dz]

    def as_dict(self) -> dict:
        return {
            "rotation_rad": self.rotation,
            "pivot_local": self.pivot.tolist(),
            "pivot_lv95": self.pivot_lv95.tolist(),
            "dz": self.dz,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Georeference":
        return cls(d["rotation_rad"], np.array(d["pivot_local"]), np.array(d["pivot_lv95"]), d["dz"])

    def with_offsets(self, dx: float, dy: float, dz: float, dtheta: float) -> "Georeference":
        return Georeference(self.rotation + dtheta, self.pivot, self.pivot_lv95 + [dx, dy], self.dz + dz)


def get_json(url: str, **params) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=60) as r:
        return json.load(r)


def lv95_to_wgs84(e: float, n: float) -> tuple[float, float]:
    """swisstopo approximate formulas (~1 m), sufficient for tile queries."""
    y, x = (e - 2_600_000) / 1e6, (n - 1_200_000) / 1e6
    lon = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2 - 0.0436 * y**3
    lat = 16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2 - 0.0447 * y**2 * x - 0.0140 * x**3
    return lon * 100 / 36, lat * 100 / 36


def gazetteer_road_box(name: str) -> np.ndarray:
    results = get_json(GEOADMIN_SEARCH, searchText=name, type="locations", sr=2056)["results"]
    road = next(r for r in results if r["attrs"]["objectclass"] == "TLM_STRASSE")
    box = road["attrs"]["geom_st_box2d"].removeprefix("BOX(").removesuffix(")")
    return np.array([[float(v) for v in corner.split()] for corner in box.split(",")])


def fetch_tiles(collection: str, bbox_lv95: tuple[float, float, float, float], gsd: float) -> list[Path]:
    """Download the latest release of every 1 km tile of `collection` intersecting the box."""
    lon0, lat0 = lv95_to_wgs84(bbox_lv95[0], bbox_lv95[1])
    lon1, lat1 = lv95_to_wgs84(bbox_lv95[2], bbox_lv95[3])
    items = get_json(STAC_ITEMS.format(collection=collection), bbox=f"{lon0},{lat0},{lon1},{lat1}", limit=100)["features"]
    latest: dict[str, tuple[str, str]] = {}
    for item in items:
        year, tile = item["id"].split("_")[-2:]
        for asset in item["assets"].values():
            if asset["href"].endswith(".tif") and asset.get("eo:gsd") == gsd and year > latest.get(tile, ("",))[0]:
                latest[tile] = (year, asset["href"])
    out_dir = TILE_DIR / collection
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile, (_, href) in sorted(latest.items()):
        path = out_dir / Path(href).name
        if not path.exists():
            urllib.request.urlretrieve(href, path)
        paths.append(path)
    return paths


@dataclass
class Raster:
    """Elevation raster in LV95 with cell-centre coordinates (east increasing, north increasing)."""

    east: np.ndarray
    north: np.ndarray
    z: np.ndarray

    @classmethod
    def mosaic(cls, paths: list[Path], bbox_lv95=None) -> "Raster":
        sources = [rasterio.open(p) for p in paths]
        z, transform = merge(sources, bounds=bbox_lv95, nodata=np.nan, dtype="float32")
        for s in sources:
            s.close()
        z = z[0][::-1]
        rows, cols = z.shape
        east = transform.c + transform.a * (np.arange(cols) + 0.5)
        north = transform.f + transform.e * (rows - np.arange(rows) - 0.5)
        return cls(east, north, z)

    def sampler(self) -> RegularGridInterpolator:
        return RegularGridInterpolator((self.north, self.east), self.z, bounds_error=False, fill_value=np.nan)

    def sample(self, en: np.ndarray) -> np.ndarray:
        return self.sampler()(en[:, ::-1])


def load_scan(campaign: str) -> np.ndarray:
    return np.concatenate([np.load(f)["xyz"] for f in sorted((CACHE_DIR / campaign).glob("*.npz"))])


def top_surface(xyz: np.ndarray, cell: float, lowest=False) -> np.ndarray:
    """One point per occupied plan cell: cell centre with the highest (or lowest) z."""
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    order = np.lexsort((xyz[:, 2] if lowest else -xyz[:, 2], ij[:, 1], ij[:, 0]))
    ij, z = ij[order], xyz[order, 2]
    first = np.r_[True, np.any(np.diff(ij, axis=0) != 0, axis=1)]
    return np.c_[(ij[first] + 0.5) * cell, z[first]]


def local_deck_ends(model: dict) -> dict[str, np.ndarray]:
    a = model["alignment"]
    align = ArcAlignment(tuple(a["centre"]), a["radius"], a["direction"], a["theta0"], a["length"])
    return {name: np.asarray(align.point(model["deck"][key], 0.0)) for name, key in (("Est", "s_start"), ("Ouest", "s_end"))}


def coarse_georeference(model: dict) -> Georeference:
    local_est, local_ouest = local_deck_ends(model).values()
    box = gazetteer_road_box(BRIDGE_NAME)
    lv95_ouest, lv95_est = box[np.argsort(box[:, 0])]
    local_dir, lv95_dir = local_ouest - local_est, lv95_ouest - lv95_est
    rotation = np.arctan2(lv95_dir[1], lv95_dir[0]) - np.arctan2(local_dir[1], local_dir[0])
    pivot = (local_est + local_ouest) / 2
    return Georeference(float(rotation), pivot, (lv95_est + lv95_ouest) / 2, 0.0)


def surface_residuals(geo: Georeference, points: np.ndarray, surface: RegularGridInterpolator) -> np.ndarray:
    enz = geo.to_lv95(points)
    return enz[:, 2] - surface(enz[:, [1, 0]])


def pier_ground_points(campaign: str, model: dict) -> np.ndarray:
    """Lowest scan points per 0.5 m cell in a ring of bare ground around each pier base."""
    rings = []
    for pier in model["piers"]:
        pts = load_element(campaign, f"Pil_{pier['name']}")
        around = np.linalg.norm(pts[:, :2] - pier["centre"], axis=1) > PIER_RING_INNER_RADIUS
        rings.append(top_surface(pts[around & (pts[:, 2] < pier["z_bottom"] + 0.6)], 0.5, lowest=True))
    return np.concatenate(rings)


@dataclass
class MatchTargets:
    """Scan top surface against swissSURFACE3D and scan ground around pier bases against swissALTI3D.

    The deck alone leaves the position along the bridge axis unconstrained (a long, smooth
    ridge); the valley floor at the pier bases, which drops ~15 m from pier A to D, fixes it.
    """

    top: np.ndarray
    ground: np.ndarray
    dsm: RegularGridInterpolator
    dtm: RegularGridInterpolator

    def residuals(self, geo: Georeference) -> tuple[np.ndarray, np.ndarray]:
        return surface_residuals(geo, self.top, self.dsm), surface_residuals(geo, self.ground, self.dtm)


def spread(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    return float(np.median(np.abs(r - np.median(r))))


def fine_georeference(coarse: Georeference, targets: MatchTargets) -> tuple[Georeference, dict]:
    """Grid search over plan offsets and rotation, then robust least squares on all four parameters."""

    def score(p):
        r_top, r_ground = targets.residuals(coarse.with_offsets(p[0], p[1], 0.0, p[2]))
        height_gap = abs(np.nanmedian(r_top) - np.nanmedian(r_ground))
        return spread(r_top) + spread(r_ground) + height_gap, -(np.nanmedian(r_top) + np.nanmedian(r_ground)) / 2

    grid = [(dx, dy, dtheta) for dtheta in np.radians(np.arange(-2.0, 2.01, 0.25))
            for dx in np.arange(-12, 12.1, 1.0) for dy in np.arange(-12, 12.1, 1.0)]
    scores = [score(p) for p in grid]
    best = int(np.argmin([s for s, _ in scores]))
    start = (*grid[best][:2], scores[best][1], grid[best][2])
    ground_weight = np.sqrt(len(targets.top) / len(targets.ground))

    def residuals(p):
        r_top, r_ground = targets.residuals(coarse.with_offsets(*p))
        return np.nan_to_num(np.r_[r_top, ground_weight * r_ground])

    fit = least_squares(residuals, start, loss="soft_l1", f_scale=0.25, x_scale=[1, 1, 1, 1e-3])
    geo = coarse.with_offsets(*fit.x)
    r_top, r_ground = (r[np.isfinite(r)] for r in targets.residuals(geo))
    stats = {
        "coarse_to_fine_shift_m": [float(v) for v in fit.x[:2]],
        "coarse_to_fine_rotation_deg": float(np.degrees(fit.x[3])),
        "surface_model": residual_stats(r_top, 0.25),
        "terrain_model_pier_ground": residual_stats(r_ground, 0.25),
    }
    return geo, stats


def residual_stats(r: np.ndarray, tolerance: float) -> dict:
    return {
        "cells": int(len(r)),
        "median_m": float(np.median(r)),
        "median_abs_m": float(np.median(np.abs(r))),
        f"within_{round(tolerance * 100)}cm": float(np.mean(np.abs(r) < tolerance)),
    }


def terrain_bbox(geo: Georeference) -> tuple[float, float, float, float]:
    centre = geo.to_lv95(np.c_[[geo.pivot], [0.0]])[0]
    return (centre[0] - TERRAIN_MARGIN, centre[1] - TERRAIN_MARGIN, centre[0] + TERRAIN_MARGIN, centre[1] + TERRAIN_MARGIN)


def load_georeference(campaign: str) -> Georeference:
    return Georeference.from_dict(json.loads((Path("build") / campaign / "georeference.json").read_text())["transform"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default=campaigns()[-1], choices=campaigns())
    args = parser.parse_args()
    out_dir = Path("build") / args.campaign
    model = json.loads((out_dir / "bridge_model.json").read_text())

    coarse = coarse_georeference(model)
    bbox = terrain_bbox(coarse)
    dsm_paths, dtm_paths = fetch_tiles(SURFACE_MODEL, bbox, 0.5), fetch_tiles(TERRAIN_MODEL, bbox, 0.5)
    targets = MatchTargets(
        top_surface(load_scan(args.campaign), MATCH_CELL),
        pier_ground_points(args.campaign, model),
        Raster.mosaic(dsm_paths, bbox).sampler(),
        Raster.mosaic(dtm_paths, bbox).sampler(),
    )
    geo, fit_stats = fine_georeference(coarse, targets)

    report = {
        "campaign": args.campaign,
        "crs": "LV95 (EPSG:2056) / LN02 heights",
        "transform": geo.as_dict(),
        "rotation_deg": float(np.degrees(geo.rotation)),
        "fit": fit_stats,
        "surface_model_tiles": [p.name for p in dsm_paths],
        "terrain_model_tiles": [p.name for p in dtm_paths],
        "deck_ends_lv95": {name: geo.to_lv95(np.r_[xy, 0.0][None])[0, :2].round(2).tolist() for name, xy in local_deck_ends(model).items()},
    }
    (out_dir / "georeference.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
