"""Extract a parametric, measured description of Pont de la Praz from the point clouds.

Output: build/<campaign>/bridge_model.json, consumed by build_ifc.py.

Conventions
-----------
* Station ``s`` runs along the horizontal alignment from the Est end to the Ouest end.
* Offset ``o`` is the lateral distance to the alignment, positive to the left (IFC convention).
* Deck cross-sections are expressed as (o, h) where ``h`` is the height above the box soffit.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import least_squares
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from load_scans import campaigns, load_element, load_elements

PIER_NAMES = "ABCDEFGH"
STATION_STEP = 1.0
SECTION_BIN = 0.04

# Values that cannot be observed by a terrestrial scan (inside the box, under the asphalt, behind walls).
ASSUMPTIONS = {
    "pavement_thickness": 0.10,
    "box_top_slab_thickness": 0.30,
    "box_bottom_slab_thickness": 0.22,
    "box_web_thickness": 0.45,
    "box_top_haunch": [0.60, 0.25],
    "box_bottom_haunch": [0.30, 0.15],
    "abutment_wall_thickness": 1.00,
    "wing_wall_thickness": 0.50,
    "gallery_back_wall_thickness": 0.50,
    "abutment_bearing_setback": 0.50,
}


@dataclass
class ArcAlignment:
    centre: tuple[float, float]
    radius: float
    direction: int  # +1 counter-clockwise (curving left), -1 clockwise (curving right)
    theta0: float  # polar angle (around the centre) of station 0
    length: float = 0.0

    def station_offset(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.atleast_2d(xy) - self.centre
        dtheta = np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - self.theta0)))
        return self.direction * dtheta * self.radius, -self.direction * (np.hypot(v[:, 0], v[:, 1]) - self.radius)

    def point(self, s, o=0.0) -> np.ndarray:
        s, o = np.broadcast_arrays(np.asarray(s, float), np.asarray(o, float))
        theta = self.theta0 + self.direction * s / self.radius
        r = self.radius - self.direction * o
        return np.stack([self.centre[0] + r * np.cos(theta), self.centre[1] + r * np.sin(theta)], axis=-1)

    def heading(self, s) -> np.ndarray:
        return self.theta0 + self.direction * np.asarray(s, float) / self.radius + self.direction * np.pi / 2


@dataclass
class VerticalProfile:
    """Box soffit elevation z = c0 + c1*s + c2*s^2 along the alignment."""

    coeffs: tuple[float, float, float]
    rms: float
    is_parabolic: bool

    def soffit(self, s) -> np.ndarray:
        c0, c1, c2 = self.coeffs
        s = np.asarray(s, float)
        return c0 + c1 * s + c2 * s**2


# --------------------------------------------------------------------------- geometry fitting


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    a = np.c_[2 * xy, np.ones(len(xy))]
    cx, cy, k = np.linalg.lstsq(a, (xy**2).sum(1), rcond=None)[0]
    guess = [cx, cy, np.sqrt(k + cx**2 + cy**2)]
    fit = least_squares(lambda q: np.hypot(xy[:, 0] - q[0], xy[:, 1] - q[1]) - q[2], guess)
    return fit.x[:2], fit.x[2], fit.fun


def robust_polyfit(x: np.ndarray, y: np.ndarray, deg: int, n_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(x), bool)
    for _ in range(5):
        p = np.polyfit(x[keep], y[keep], deg)
        res = y - np.polyval(p, x)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < max(n_sigma * mad, 0.005)
    return p, keep


def station_bins(s: np.ndarray, lo: float, hi: float, step: float = STATION_STEP):
    edges = np.arange(lo, hi + step, step)
    idx = np.digitize(s, edges) - 1
    return edges[:-1] + step / 2, idx


def lowest_surface(z: np.ndarray, band: float = 0.08) -> float:
    zmin = np.percentile(z, 2)
    return float(np.median(z[z < zmin + band]))


def fit_alignment(deck: np.ndarray, est_ref: np.ndarray, ouest_ref: np.ndarray) -> tuple[ArcAlignment, dict]:
    xy = deck[:, :2]
    centre_xy = xy.mean(0)
    _, vecs = np.linalg.eigh(np.cov((xy - centre_xy).T))
    u, v = vecs[:, 1], vecs[:, 0]
    t, n = (xy - centre_xy) @ u, (xy - centre_xy) @ v
    mids = []
    for k in np.unique(np.floor(t / 2.0)):
        m = np.floor(t / 2.0) == k
        if m.sum() > 500:
            lo, hi = np.percentile(n[m], [0.5, 99.5])
            mids.append(centre_xy + u * (k * 2 + 1) + v * (lo + hi) / 2)
    centre, radius, _ = fit_circle(np.array(mids))

    for _ in range(3):
        align = ArcAlignment(tuple(centre), radius, 1, 0.0)
        theta_est = align.station_offset(est_ref)[0][0] / radius
        align.theta0 = theta_est
        direction = 1 if align.station_offset(ouest_ref)[0][0] > 0 else -1
        align.direction = direction
        s, o = align.station_offset(xy)
        stations, idx = station_bins(s, s.min(), s.max())
        edge_mid, width, st = [], [], []
        for i, sc in enumerate(stations):
            m = idx == i
            if m.sum() < 300:
                continue
            lo, hi = np.percentile(o[m], [0.2, 99.8])
            edge_mid.append((lo + hi) / 2)
            width.append(hi - lo)
            st.append(sc)
        edge_mid, width, st = map(np.array, (edge_mid, width, st))
        good = np.abs(width - np.median(width)) < 0.15
        centre, radius, residuals = fit_circle(align.point(st[good], edge_mid[good]))

    align = ArcAlignment(tuple(float(c) for c in centre), float(radius), 1, 0.0)
    align.theta0 = align.station_offset(est_ref)[0][0] / radius
    align.direction = 1 if align.station_offset(ouest_ref)[0][0] > 0 else -1
    stats = {
        "deck_width_median": float(np.median(width)),
        "centreline_rms": float(np.sqrt(np.mean(residuals**2))),
        "centreline_max_abs": float(np.abs(residuals).max()),
        "stations_used": int(good.sum()),
    }
    return align, stats


def fit_vertical(align: ArcAlignment, deck: np.ndarray, support_stations: np.ndarray) -> tuple[VerticalProfile, dict]:
    s, o = align.station_offset(deck[:, :2])
    near_box_axis = np.abs(o) < 2.5
    stations, idx = station_bins(s, s.min(), s.max())
    samples = []
    for i, sc in enumerate(stations):
        if np.min(np.abs(support_stations - sc)) < 3.0:
            continue
        m = near_box_axis & (idx == i)
        if m.sum() > 50:
            samples.append((sc, lowest_surface(deck[m, 2])))
    samples = np.array(samples)
    quad, keep_q = robust_polyfit(samples[:, 0], samples[:, 1], 2)
    lin, keep_l = robust_polyfit(samples[:, 0], samples[:, 1], 1)
    span = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 200)
    sag = np.max(np.abs(np.polyval(quad, span) - np.polyval(lin, span)))
    is_parabolic = bool(sag > 0.02)
    coeffs = quad[::-1] if is_parabolic else np.r_[lin[::-1], 0.0]
    keep = keep_q if is_parabolic else keep_l
    res = samples[keep, 1] - np.polyval(coeffs[::-1], samples[keep, 0])
    profile = VerticalProfile(tuple(float(c) for c in coeffs), float(np.sqrt(np.mean(res**2))), is_parabolic)
    stats = {
        "soffit_samples": int(len(samples)),
        "soffit_outliers_rejected": int((~keep).sum()),
        "parabola_vs_line_max_deviation": float(sag),
        "soffit_rms": profile.rms,
    }
    return profile, stats


# --------------------------------------------------------------------------- deck cross-section


def cell_extrema(key_station: np.ndarray, key_bin: np.ndarray, h: np.ndarray, n_bins: int):
    """Per (station, offset-bin) min and max of h, returned as dense arrays with NaN for empty cells."""
    n_st = key_station.max() + 1
    flat = key_station * n_bins + key_bin
    order = np.argsort(flat, kind="stable")
    flat, hs = flat[order], h[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    lo = np.full(n_st * n_bins, np.nan)
    hi = np.full(n_st * n_bins, np.nan)
    lo[flat[starts]] = np.minimum.reduceat(hs, starts)
    hi[flat[starts]] = np.maximum.reduceat(hs, starts)
    return lo.reshape(n_st, n_bins), hi.reshape(n_st, n_bins)


def extract_deck_envelopes(align, profile, deck, s_range, support_stations):
    s, o = align.station_offset(deck[:, :2])
    stations, idx = station_bins(s, *s_range)
    usable = np.array([np.min(np.abs(support_stations - sc)) > 5.0 for sc in stations])

    rows_o, rows_h, rows_st = [], [], []
    for i, sc in enumerate(stations):
        m = idx == i
        if not usable[i] or m.sum() < 1000:
            continue
        oi, zi = o[m], deck[m, 2]
        lo, hi = np.percentile(oi, [0.2, 99.8])
        oi = oi - (lo + hi) / 2
        soffit = lowest_surface(zi[np.abs(oi) < 2.5])
        rows_o.append(oi)
        rows_h.append(zi - soffit)
        rows_st.append(np.full(len(oi), len(rows_st)))
    o_all, h_all, st_all = np.concatenate(rows_o), np.concatenate(rows_h), np.concatenate(rows_st)

    edges = np.arange(-8.2, 8.2 + SECTION_BIN, SECTION_BIN)
    centres = edges[:-1] + SECTION_BIN / 2
    b = np.digitize(o_all, edges) - 1
    ok = (b >= 0) & (b < len(centres))
    lo, hi = cell_extrema(st_all[ok], b[ok], h_all[ok], len(centres))
    coverage = np.sum(~np.isnan(lo), axis=0)
    valid = coverage >= max(10, 0.2 * len(rows_st))
    lower = np.nanmedian(lo, axis=0)
    hi[~(hi > lower + 0.15)] = np.nan
    upper = np.nanmedian(hi, axis=0)
    valid &= ~np.isnan(upper)
    return centres[valid], lower[valid], upper[valid], len(rows_st)


def fit_road_lines(o, upper):
    """Fit the two carriageway surfaces and locate the feet of the edge and median barriers."""
    lines, feet = {}, {}
    for side, (a, b) in {"left": (-6.5, -1.3), "right": (1.0, 6.5)}.items():
        m = (o > a) & (o < b)
        p, _ = robust_polyfit(o[m], upper[m], 1)
        lines[side] = p
        above = upper - np.polyval(p, o) > 0.04
        outward = o < a if side == "left" else o > b
        inward = (o > b) if side == "left" else (o < a)
        out_idx = np.flatnonzero(outward & above)
        in_idx = np.flatnonzero(inward & above)
        feet[side] = {
            "edge": float(o[out_idx.max()] if side == "left" else o[out_idx.min()]),
            "median": float(o[in_idx.min()] if side == "left" else o[in_idx.max()]),
        }
    return lines, feet


def assumed_box_void(web_bottom, web_top, slab_top_line):
    """Inner void of the single-cell box: webs, slabs and haunches are assumed (not observable)."""
    t_top, t_bot, t_web = (ASSUMPTIONS[k] for k in ("box_top_slab_thickness", "box_bottom_slab_thickness", "box_web_thickness"))
    (hx_top, hy_top), (hx_bot, hy_bot) = ASSUMPTIONS["box_top_haunch"], ASSUMPTIONS["box_bottom_haunch"]

    def inner_web_o(side, h):
        (o0, h0), (o1, h1) = web_bottom[side], web_top[side]
        return o0 + (o1 - o0) * (h - h0) / (h1 - h0) - np.sign(o0) * t_web

    h_bot = t_bot
    h_top_l = slab_top_line(inner_web_o("left", 0)) - t_top
    h_top_r = slab_top_line(inner_web_o("right", 0)) - t_top
    bl, br = inner_web_o("left", h_bot), inner_web_o("right", h_bot)
    tl, tr = inner_web_o("left", h_top_l), inner_web_o("right", h_top_r)
    pts = [
        (bl + hx_bot, h_bot), (br - hx_bot, h_bot), (inner_web_o("right", h_bot + hy_bot), h_bot + hy_bot),
        (inner_web_o("right", h_top_r - hy_top), h_top_r - hy_top), (tr - hx_top, h_top_r),
        (tl + hx_top, h_top_l), (inner_web_o("left", h_top_l - hy_top), h_top_l - hy_top),
        (inner_web_o("left", h_bot + hy_bot), h_bot + hy_bot),
    ]
    return Polygon(pts)


def polygon_coords(geom) -> dict:
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    geom = orient(geom, 1.0)
    return {
        "exterior": np.round(np.asarray(geom.exterior.coords)[:-1], 4).tolist(),
        "interiors": [np.round(np.asarray(r.coords)[:-1], 4).tolist() for r in geom.interiors],
        "area": round(geom.area, 4),
    }


def build_deck_section(o, lower, upper):
    lower_line = np.c_[o, lower]
    upper_line = np.c_[o, upper][::-1]
    outer = Polygon(np.r_[lower_line, upper_line]).buffer(0).simplify(0.01)

    lines, feet = fit_road_lines(o, upper)
    t_pav = ASSUMPTIONS["pavement_thickness"]
    med_l, med_r = feet["left"]["median"], feet["right"]["median"]

    def road(oo):
        return np.polyval(lines["left"], oo) if oo <= (med_l + med_r) / 2 else np.polyval(lines["right"], oo)

    def slab_top(oo):
        return road(oo) - t_pav

    slab_top_pts = [(-9, slab_top(-9)), (med_l, slab_top(med_l)), (med_r, slab_top(med_r)), (9, slab_top(9))]
    below_slab_top = Polygon(slab_top_pts + [(9, -1), (-9, -1)])
    girder_outer = outer.intersection(below_slab_top)
    above = outer.difference(below_slab_top)

    edge_l, edge_r = feet["left"]["edge"], feet["right"]["edge"]
    barrier_left = above.intersection(box(-9, -1, edge_l, 9))
    barrier_right = above.intersection(box(edge_r, -1, 9, 9))
    barrier_median = above.intersection(box(med_l, -1, med_r, 9))
    road_poly = Polygon(
        [(edge_l, slab_top(edge_l)), (med_l, slab_top(med_l)), (med_l, road(med_l)), (edge_l, road(edge_l))]
    ).union(Polygon([(med_r, slab_top(med_r)), (edge_r, slab_top(edge_r)), (edge_r, road(edge_r)), (med_r, road(med_r))]))
    pavement_left, pavement_right = (outer.intersection(road_poly).intersection(box(a, -1, b, 9)) for a, b in ((edge_l, med_l), (med_r, edge_r)))

    soffit_mask = lower < 0.05
    web_bottom_l, web_bottom_r = o[soffit_mask].min(), o[soffit_mask].max()
    web_top = {}
    for side, sign in (("left", -1), ("right", 1)):
        on_web = (lower > 0.8) & (lower < 2.3) & (sign * o > 0)
        p = np.polyfit(lower[on_web], o[on_web], 1)
        h_top = float(np.interp(sign * 3.95, o, lower))
        web_top[side] = (float(np.polyval(p, h_top)), h_top)
    web_bottom = {"left": (float(web_bottom_l), 0.0), "right": (float(web_bottom_r), 0.0)}

    void = assumed_box_void(web_bottom, web_top, slab_top).intersection(girder_outer.buffer(-0.15))
    girder = girder_outer.difference(void)

    road_axis_height = float(np.polyval(lines["right"], 0.0))
    return {
        "outer": polygon_coords(outer),
        "girder": polygon_coords(girder),
        "barrier_left": polygon_coords(barrier_left),
        "barrier_right": polygon_coords(barrier_right),
        "barrier_median": polygon_coords(barrier_median),
        "pavement_left": polygon_coords(pavement_left),
        "pavement_right": polygon_coords(pavement_right),
        "measured": {
            "overall_width": float(o.max() - o.min()),
            "structural_depth_at_axis": road_axis_height,
            "soffit_width": float(web_bottom_r - web_bottom_l),
            "web_bottom_left": web_bottom["left"],
            "web_bottom_right": web_bottom["right"],
            "web_top_left": web_top["left"],
            "web_top_right": web_top["right"],
            "crossfall_left": float(lines["left"][0]),
            "crossfall_right": float(lines["right"][0]),
            "road_left_height_at_axis": float(np.polyval(lines["left"], 0.0)),
            "road_right_height_at_axis": road_axis_height,
            "barrier_left_foot": edge_l,
            "barrier_right_foot": edge_r,
            "median_barrier_feet": [med_l, med_r],
            "barrier_left_height": float(upper[o < edge_l].max() - road(edge_l)),
            "barrier_right_height": float(upper[o > edge_r].max() - road(edge_r)),
            "median_barrier_height": float(upper[(o > med_l) & (o < med_r)].max() - road(med_r)),
        },
        "slab_top_polyline": [list(p) for p in slab_top_pts],
        "envelope": {"o": o.round(3).tolist(), "lower": lower.round(4).tolist(), "upper": upper.round(4).tolist()},
    }


def deck_end_stations(align, profile, section_measured: dict, end_points: dict[str, np.ndarray]) -> dict[str, float]:
    """The deck ends where points stop lying on the inclined box webs."""
    ends = {}
    for side, pts in end_points.items():
        s, o = align.station_offset(pts[:, :2])
        h = pts[:, 2] - profile.soffit(s)
        on_web = np.zeros(len(s), bool)
        for web in ("left", "right"):
            (o_bot, _), (o_top, h_top) = section_measured[f"web_bottom_{web}"], section_measured[f"web_top_{web}"]
            expected = o_bot + (o_top - o_bot) * h / h_top
            on_web |= (np.abs(o - expected) < 0.06) & (h > 0.8) & (h < h_top - 0.3)
        ends[side] = float(np.percentile(s[on_web], 0.5) if side == "est" else np.percentile(s[on_web], 99.5))
    return ends


# --------------------------------------------------------------------------- piers


def measure_pier(name: str, xyz: np.ndarray, align: ArcAlignment, profile: VerticalProfile) -> dict:
    c0 = xyz[:, :2].mean(0)
    _, vecs = np.linalg.eigh(np.cov((xyz[:, :2] - c0).T))
    u = vecs[:, 1]
    v = np.array([-u[1], u[0]])
    a, b, z = (xyz[:, :2] - c0) @ u, (xyz[:, :2] - c0) @ v, xyz[:, 2]
    z_lo, z_hi = np.percentile(z, [0.5, 99.5])

    slices = []
    for zz in np.arange(z_lo + 1.5, z_hi - 2.0, 0.5):
        m = np.abs(z - zz) < 0.05
        if m.sum() < 200:
            continue
        a_lo, a_hi = np.percentile(a[m], [0.3, 99.7])
        b_lo, b_hi = np.percentile(b[m], [0.3, 99.7])
        slices.append((a_lo, a_hi, b_lo, b_hi, zz))
    slices = np.array(slices)
    a_mid = np.median((slices[:, 0] + slices[:, 1]) / 2)
    b_mid = np.median((slices[:, 2] + slices[:, 3]) / 2)
    width = float(np.median(slices[:, 1] - slices[:, 0]))
    thickness = float(np.median(slices[:, 3] - slices[:, 2]))
    a, b = a - a_mid, b - b_mid
    centre = c0 + u * a_mid + v * b_mid

    chamfers = []
    for *_, zz in slices:
        m = np.abs(z - zz) < 0.05
        for sa in (1, -1):
            for sb in (1, -1):
                corner = (sa * a[m] > 0) & (sb * b[m] > 0)
                if corner.sum() > 20:
                    reach = np.percentile((sa * a[m][corner] + sb * b[m][corner]), 99.5)
                    chamfers.append(width / 2 + thickness / 2 - reach)
    chamfer = float(np.clip(np.median(chamfers), 0.0, thickness / 3))

    face = (np.abs(np.abs(b) - thickness / 2) < 0.1) & (np.abs(a) < 1.2)
    zf = np.sort(z[face])
    gaps = np.flatnonzero(np.diff(zf) > 0.15)
    mid_idx = np.searchsorted(zf, np.median(zf))
    upper_gaps = gaps[gaps > mid_idx]
    z_top = float(zf[upper_gaps[0]] if len(upper_gaps) else zf[-1])
    body = (np.abs(a) < width / 2 - 0.3) & (np.abs(np.abs(b) - thickness / 2) < 0.12)
    z_bottom = float(np.percentile(z[body], 0.3))

    above = (z > z_top + 0.05) & (np.abs(a) < 1.0) & (np.abs(b) < thickness / 2)
    soffit = lowest_surface(z[above]) if above.sum() > 50 else float(profile.soffit(align.station_offset(centre)[0][0]))

    bearings = []
    gap = (z > z_top + 0.03) & (z < soffit - 0.03) & (np.abs(a) < width / 2 + 0.2) & (np.abs(b) < thickness / 2 + 0.3)
    for sa in (-1, 1):
        m = gap & (sa * a > 1.0)
        if m.sum() < 30:
            continue
        a_lo, a_hi = np.percentile(a[m], [1, 99])
        b_lo, b_hi = np.percentile(b[m], [1, 99])
        bearings.append({
            "a": float((a_lo + a_hi) / 2), "b": float((b_lo + b_hi) / 2),
            "size_a": float(a_hi - a_lo), "size_b": float(min(b_hi - b_lo, thickness)),
            "z_bottom": z_top, "z_top": soffit, "points": int(m.sum()),
        })

    s, o = align.station_offset(centre)
    radial_heading = float(align.heading(s[0]) + np.pi / 2)
    skew = float(np.degrees(np.angle(np.exp(1j * (np.arctan2(u[1], u[0]) - radial_heading)))))
    skew = (skew + 90) % 180 - 90
    return {
        "name": name,
        "centre": [float(centre[0]), float(centre[1])],
        "long_axis": [float(u[0]), float(u[1])],
        "station": float(s[0]),
        "offset": float(o[0]),
        "skew_to_radial_deg": skew,
        "width": width,
        "thickness": thickness,
        "chamfer": chamfer,
        "chamfer_spread": float(np.std(chamfers)),
        "z_bottom": z_bottom,
        "z_top": z_top,
        "height": z_top - z_bottom,
        "soffit_above": float(soffit),
        "bearings": bearings,
        "points": int(len(xyz)),
        "slices_used": int(len(slices)),
    }


# --------------------------------------------------------------------------- abutments


def dominant_value(values: np.ndarray, bin_size: float = 0.05) -> float:
    edges = np.arange(values.min(), values.max() + bin_size, bin_size)
    counts, edges = np.histogram(values, edges)
    k = np.argmax(counts)
    sel = values[(values >= edges[k]) & (values < edges[k + 1])]
    return float(np.median(sel))


def measure_abutment_est(align, profile, pts, s_deck_end, road_height, slab_top_polyline) -> dict:
    s, o = align.station_offset(pts[:, :2])
    z = pts[:, 2]
    soffit_end = float(profile.soffit(s_deck_end))

    wall_window = (s > s_deck_end - 10) & (s < s_deck_end - 5) & (np.abs(o) < 9.5) & (z < soffit_end + 2.0)
    s_wall = dominant_value(s[wall_window])
    at_wall = np.abs(s - s_wall) < 0.15
    o_min, o_max = np.percentile(o[at_wall & (z < soffit_end + 2.0)], [0.5, 99.5])
    z_ground_wall = float(np.percentile(z[at_wall], 1))

    road_top = soffit_end + road_height
    slab_zone = (s > s_wall + 0.5) & (s < s_deck_end - 0.8) & (np.abs(o) > 1.5) & (np.abs(o) < 9)
    z_slab_under = dominant_value(z[slab_zone & (z > soffit_end + 1.5) & (z < road_top - 0.2)], 0.02)
    top_o, top_h = np.array(slab_top_polyline).T
    below_top = np.interp(o, top_o, top_h) - (z - profile.soffit(s))
    slab_thickness = dominant_value(below_top[slab_zone & (below_top > 0.2) & (below_top < 1.5)], 0.02)

    wings = {}
    for side, edge in (("left", o_max), ("right", o_min)):
        m = (np.abs(o - edge) < 0.4) & (s < s_wall - 0.3) & (z < z_slab_under - 0.3)
        if m.sum() < 200:
            continue
        s_start = float(np.percentile(s[m], 1))
        wings[side] = {
            "s_start": s_start, "s_end": s_wall, "o_outer": float(edge),
            "z_bottom": float(np.percentile(z[m], 0.5)),
            "z_top": float(profile.soffit(s_start) + road_height - ASSUMPTIONS["pavement_thickness"]),
        }

    block = (s > s_deck_end - 0.5) & (s < s_deck_end + 5) & (np.abs(o) < 4) & (z < soffit_end - 0.1)
    seat_candidates = z[block & (z > soffit_end - 1.0)]
    z_seat = dominant_value(seat_candidates, 0.02)
    walls = block & (z > z_seat - 0.9) & (z < z_seat - 0.1)
    s_lo, s_hi = np.percentile(s[walls], [0.5, 99.5])
    ob_lo, ob_hi = np.percentile(o[walls], [0.5, 99.5])
    z_ground_block = float(np.percentile(z[block & (z < z_seat)], 0.5))

    return {
        "front_wall": {"s_face": s_wall, "o_min": float(o_min), "o_max": float(o_max), "z_bottom": z_ground_wall, "z_top": z_slab_under},
        "wing_walls": wings,
        "cover_slab": {"s_start": s_wall, "s_end": s_deck_end, "o_min": float(o_min), "o_max": float(o_max),
                       "z_underside": z_slab_under, "thickness": slab_thickness},
        "support_block": {"s_start": float(s_lo), "s_end": float(s_hi), "o_min": float(ob_lo), "o_max": float(ob_hi), "z_bottom": z_ground_block, "z_top": z_seat},
        "seat_elevation": z_seat,
        "soffit_at_end": soffit_end,
        "points": int(len(pts)),
    }


def measure_abutment_ouest(align, profile, pts, s_deck_end, road_height) -> dict:
    s, o = align.station_offset(pts[:, :2])
    z = pts[:, 2]
    soffit_end = float(profile.soffit(s_deck_end))

    road_top = soffit_end + road_height
    front_window = (s > s_deck_end - 0.5) & (s < s_deck_end + 0.5) & (np.abs(o) < 9.5) & (z < soffit_end - 0.4)
    s_front = dominant_value(s[front_window])
    at_front = (np.abs(s - s_front) < 0.2) & (z < soffit_end - 0.4)
    z_ground = float(np.percentile(z[at_front], 1))

    behind = (s > s_front + 0.2) & (s < s_front + 4) & (np.abs(o) < 9)
    gallery_floor = lowest_surface(z[behind & (z > z_ground + 0.3)], band=0.06)
    in_gallery = behind & (s > s_front + 1.0) & (z > gallery_floor + 0.3) & (z < road_top - 0.5)
    s_back_inner = dominant_value(s[in_gallery])
    floor_pts = behind & (np.abs(z - gallery_floor) < 0.05) & (s < s_back_inner)
    s_gallery_front = float(np.percentile(s[floor_pts], 0.5))
    gallery_zone = behind & (s > s_gallery_front + 0.2) & (s < s_back_inner - 0.2)
    gallery_ceiling = dominant_value(z[gallery_zone & (z > gallery_floor + 2.5) & (z < road_top - 0.3)], 0.02)
    ceiling_pts = gallery_zone & (np.abs(z - gallery_ceiling) < 0.05)
    g_o_lo, g_o_hi = np.percentile(o[ceiling_pts], [0.5, 99.5])
    side_walls = (s > s_front) & (s < s_back_inner) & (z > gallery_floor + 0.5) & (z < gallery_ceiling - 0.5)
    o_min, o_max = np.percentile(o[side_walls], [0.2, 99.8])

    return {
        "body": {
            "s_front": s_front, "s_back": s_back_inner + ASSUMPTIONS["gallery_back_wall_thickness"],
            "o_min": float(o_min), "o_max": float(o_max), "z_bottom": z_ground,
            "z_top": float(road_top - ASSUMPTIONS["pavement_thickness"]),
        },
        "gallery": {
            "s_start": s_gallery_front, "s_end": s_back_inner, "o_min": float(g_o_lo), "o_max": float(g_o_hi),
            "z_floor": gallery_floor, "z_ceiling": gallery_ceiling,
        },
        "soffit_at_end": soffit_end,
        "points": int(len(pts)),
    }


# --------------------------------------------------------------------------- main


def analyse(campaign: str) -> dict:
    deck = load_elements(campaign, "Tab_")
    cul_est, cul_ouest = load_element(campaign, "Cul_Est"), load_element(campaign, "Cul_Ouest")
    pier_xyz = {n: load_element(campaign, f"Pil_{n}") for n in PIER_NAMES}

    align, align_stats = fit_alignment(deck, cul_est[:, :2].mean(0), cul_ouest[:, :2].mean(0))
    all_xy = np.concatenate([deck[:, :2], cul_est[:, :2], cul_ouest[:, :2]])
    s_all, _ = align.station_offset(all_xy)
    start = np.floor(s_all.min()) - 1.0
    align.theta0 += align.direction * start / align.radius
    align.length = float(np.ceil(s_all.max() - start) + 1.0)

    pier_centres = np.array([p[:, :2].mean(0) for p in pier_xyz.values()])
    pier_stations = align.station_offset(pier_centres)[0]
    profile, vert_stats = fit_vertical(align, deck, pier_stations)

    section_range = (pier_stations.min() - 20, pier_stations.max() + 20)
    o_env, lower, upper, n_sections = extract_deck_envelopes(align, profile, deck, section_range, pier_stations)
    section = build_deck_section(o_env, lower, upper)
    ends = deck_end_stations(align, profile, section["measured"], {
        "est": load_element(campaign, "Tab_Est_A"),
        "ouest": load_element(campaign, "Tab_H_Ouest"),
    })

    piers = [measure_pier(n, pier_xyz[n], align, profile) for n in PIER_NAMES]
    names = ["Est"] + list(PIER_NAMES) + ["Ouest"]
    stations = [ends["est"]] + [p["station"] for p in piers] + [ends["ouest"]]
    spans = [{"name": f"{a}-{b}", "s_start": s0, "s_end": s1, "length": s1 - s0}
             for a, b, s0, s1 in zip(names[:-1], names[1:], stations[:-1], stations[1:])]

    # Abutments are static and partly hidden, so all campaigns (same survey frame) are pooled to maximise coverage.
    def pooled(*keys):
        return np.concatenate([load_element(c, k) for c in campaigns() for k in keys])

    abutments = {
        "Est": measure_abutment_est(align, profile, pooled("Cul_Est", "Tab_Est_A"), ends["est"],
                                    section["measured"]["structural_depth_at_axis"], section["slab_top_polyline"]),
        "Ouest": measure_abutment_ouest(align, profile, pooled("Cul_Ouest", "Tab_H_Ouest"), ends["ouest"],
                                        section["measured"]["structural_depth_at_axis"]),
    }

    return {
        "campaign": campaign,
        "coordinate_system": "local survey frame of the E57 files (metres, Z up); not georeferenced",
        "assumptions": ASSUMPTIONS,
        "alignment": asdict(align) | {"stats": align_stats},
        "vertical": asdict(profile) | {"stats": vert_stats, "grade_line_height_above_soffit": section["measured"]["structural_depth_at_axis"]},
        "deck": {
            "s_start": ends["est"], "s_end": ends["ouest"], "length": ends["ouest"] - ends["est"],
            "sections_stacked": n_sections, "section": section, "spans": spans,
        },
        "piers": piers,
        "abutments": abutments,
        "point_counts": {"deck": int(len(deck)), "Cul_Est": int(len(cul_est)), "Cul_Ouest": int(len(cul_ouest))},
        "abutment_campaigns_pooled": campaigns(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default=campaigns()[-1], choices=campaigns())
    args = parser.parse_args()
    model = analyse(args.campaign)
    out = Path("build") / args.campaign / "bridge_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
