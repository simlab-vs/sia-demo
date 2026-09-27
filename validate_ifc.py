"""Validate the generated IFC: schema conformance, geometry, and scan-to-model deviations.

Writes build/<campaign>/validation.json plus preview images.
"""
import argparse
import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.validate
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from bridge_analysis import ArcAlignment
from load_scans import campaigns, load_element, load_elements

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

SURFACE_SAMPLE_SPACING = 0.03
SCAN_SAMPLE = 400_000

SCAN_GROUPS = {
    "Superstructure": lambda c: load_elements(c, "Tab_"),
    "Culée Est": lambda c: load_element(c, "Cul_Est"),
    "Culée Ouest": lambda c: load_element(c, "Cul_Ouest"),
    **{f"Pile {n}": (lambda c, n=n: load_element(c, f"Pil_{n}")) for n in "ABCDEFGH"},
}


def schema_errors(ifc: ifcopenshell.file) -> list[str]:
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(ifc, logger, express_rules=True)
    return [f"{s['type']}: {s['message']}"[:300] for s in logger.statements]


def tessellate(ifc: ifcopenshell.file) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    meshes = {}
    it = ifcopenshell.geom.iterator(settings, ifc)
    if it.initialize():
        while True:
            shape = it.get()
            verts = np.asarray(shape.geometry.verts).reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
            if len(faces):
                meshes[shape.id] = (verts, faces)
            if not it.next():
                break
    return meshes


def sample_surface(verts: np.ndarray, faces: np.ndarray, spacing: float) -> np.ndarray:
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    counts = np.maximum(1, np.ceil(areas / spacing**2)).astype(int)
    idx = np.repeat(np.arange(len(tri)), counts)
    r1, r2 = np.random.rand(2, len(idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    t = tri[idx]
    return t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])


def container_name(element) -> str:
    for rel in element.ContainedInStructure or []:
        part = rel.RelatingStructure
        if part.is_a("IfcBridgePart") and part.PredefinedType in ("PIER", "ABUTMENT"):
            return part.Name
        return part.Name
    return "?"


def deviations(ifc, meshes, campaign) -> dict:
    groups: dict[str, list[np.ndarray]] = {}
    for eid, (verts, faces) in meshes.items():
        name = container_name(ifc.by_id(eid))
        groups.setdefault(name, []).append(sample_surface(verts, faces, SURFACE_SAMPLE_SPACING))
    all_surface = cKDTree(np.concatenate([np.concatenate(v) for v in groups.values()]))

    report = {}
    rng = np.random.default_rng(0)
    for name, loader in SCAN_GROUPS.items():
        scan = loader(campaign)
        scan = scan[rng.choice(len(scan), min(SCAN_SAMPLE, len(scan)), replace=False)]
        d, _ = all_surface.query(scan, distance_upper_bound=5.0)
        d = np.where(np.isinf(d), 5.0, d)
        report[name] = {
            "points": int(len(scan)),
            "median_m": float(np.median(d)),
            "p90_m": float(np.percentile(d, 90)),
            "within_5cm": float(np.mean(d < 0.05)),
            "within_10cm": float(np.mean(d < 0.10)),
            "beyond_50cm": float(np.mean(d > 0.50)),
        }
    return report


def render_preview(meshes, out: Path, elev=22, azim=-60, bounds=None, title=None):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection="3d")
    for verts, faces in meshes.values():
        tri = verts[faces]
        if bounds is not None and not np.any((tri[..., 0] > bounds[0]) & (tri[..., 0] < bounds[1])):
            continue
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.clip(normals @ np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]), 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[shade, shade, shade * 0.97], edgecolor="none"))
    pts = np.concatenate([v for v, _ in meshes.values()])
    if bounds is not None:
        pts = pts[(pts[:, 0] > bounds[0]) & (pts[:, 0] < bounds[1])]
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def slice_mesh(verts_soz: np.ndarray, faces: np.ndarray, axis: int, value: float) -> np.ndarray:
    """Intersect a mesh given in (s, o, z) coordinates with the plane coordinate[axis] == value."""
    keep = [i for i in range(3) if i != axis]
    tri = verts_soz[faces]
    d = tri[..., axis] - value
    crossing = (d.min(1) < 0) & (d.max(1) > 0)
    segments = []
    for t, dd in zip(tri[crossing], d[crossing]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] < 0) != (dd[j] < 0):
                w = dd[i] / (dd[i] - dd[j])
                pts.append(t[i] + w * (t[j] - t[i]))
        if len(pts) == 2:
            segments.append([pts[0][keep], pts[1][keep]])
    return np.asarray(segments).reshape(-1, 2, 2)


def render_section_overlays(meshes, model: dict, campaign: str, out: Path):
    a = model["alignment"]
    align = ArcAlignment(tuple(a["centre"]), a["radius"], a["direction"], a["theta0"], a["length"])

    def to_soz(xyz):
        s, o = align.station_offset(xyz[:, :2])
        return np.c_[s, o, xyz[:, 2]]

    meshes_soz = [(to_soz(v), f) for v, f in meshes.values()]
    deck = load_elements(campaign, "Tab_")
    piers = {p["name"]: p for p in model["piers"]}
    s_mid = (piers["C"]["station"] + piers["D"]["station"]) / 2
    ab_est, ab_ouest = model["abutments"]["Est"], model["abutments"]["Ouest"]
    views = [
        ("Deck cross-section, mid-span C-D", deck, 0, s_mid, 0.15, None),
        ("Cross-section at pier D", np.concatenate([deck, load_element(campaign, "Pil_D")]), 0, piers["D"]["station"], 0.15, (None, (piers["D"]["z_top"] - 6, None))),
        ("Culée Est, longitudinal section o = +5 m", np.concatenate([deck, load_element(campaign, "Cul_Est")]), 1, 5.0, 0.2,
         ((ab_est["front_wall"]["s_face"] - 6, model["deck"]["s_start"] + 10), None)),
        ("Culée Ouest, longitudinal section o = -4 m", np.concatenate([deck, load_element(campaign, "Cul_Ouest")]), 1, -4.0, 0.2,
         ((model["deck"]["s_end"] - 8, ab_ouest["body"]["s_back"] + 3), None)),
    ]
    fig, axes = plt.subplots(len(views), 1, figsize=(16, 7 * len(views)))
    for ax, (title, pts, axis, value, tol, window) in zip(axes, views):
        soz = to_soz(pts)
        keep = [i for i in range(3) if i != axis]
        m = np.abs(soz[:, axis] - value) < tol
        if window:
            for k, lim in zip(keep, window):
                if lim:
                    m &= (soz[:, k] > lim[0]) & (soz[:, k] < (np.inf if lim[1] is None else lim[1]))
        ax.scatter(soz[m, keep[0]], soz[m, keep[1]], s=0.3, c="0.55", label="scan points")
        segs = np.concatenate([slice_mesh(v, f, axis, value) for v, f in meshes_soz])
        ax.add_collection(LineCollection(segs, colors="#d0342c", linewidths=1.0, label="IFC model"))
        if window:
            if window[0]:
                ax.set_xlim(*window[0])
            if window[1]:
                ax.set_ylim(window[1][0], window[1][1] or soz[m, keep[1]].max() + 0.5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(["station s [m]", "offset o [m] (+ left)", "z [m]"][keep[0]])
        ax.set_ylabel("z [m]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", markerscale=10)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default=campaigns()[-1], choices=campaigns())
    args = parser.parse_args()
    out_dir = Path("build") / args.campaign
    ifc = ifcopenshell.open(out_dir / "pont_de_la_praz.ifc")

    errors = schema_errors(ifc)
    meshes = tessellate(ifc)
    products = [p for p in ifc.by_type("IfcProduct") if p.Representation and not p.is_a("IfcAlignment") and not p.is_a("IfcLinearElement")]
    missing = [f"{p.is_a()} {p.Name}" for p in products if p.id() not in meshes]
    report = {
        "schema": ifc.schema_identifier,
        "schema_errors": errors,
        "elements": {cls: len(ifc.by_type(cls)) for cls in ("IfcBeam", "IfcColumn", "IfcBearing", "IfcWall", "IfcSlab", "IfcRailing", "IfcCourse", "IfcBridgePart", "IfcAlignment", "IfcReferent")},
        "products_with_geometry": len(products),
        "tessellated": len(meshes),
        "failed_geometry": missing,
        "scan_to_model": deviations(ifc, meshes, args.campaign),
    }
    (out_dir / "validation.json").write_text(json.dumps(report, indent=1))
    render_section_overlays(meshes, json.loads((out_dir / "bridge_model.json").read_text()), args.campaign, out_dir / "section_overlays.png")
    render_preview(meshes, out_dir / "preview.png")
    render_preview(meshes, out_dir / "preview_est.png", elev=15, azim=-35, bounds=(-225, -150))
    render_preview(meshes, out_dir / "preview_ouest.png", elev=15, azim=-130, bounds=(135, 190))
    print(json.dumps({k: v for k, v in report.items() if k != "scan_to_model"}, indent=1))
    for name, r in report["scan_to_model"].items():
        print(f"{name:15s} median {r['median_m'] * 100:5.1f} cm  p90 {r['p90_m'] * 100:6.1f} cm  <5cm {r['within_5cm']:.0%}  <10cm {r['within_10cm']:.0%}  >50cm {r['beyond_50cm']:.1%}")


if __name__ == "__main__":
    main()
