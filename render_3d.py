"""Render rotating 3D views (VTK) of the IFC model, alone or with the scans and swissALTI3D terrain.

Writes build/<campaign>/viz/<variant>.mp4 plus a still <variant>.png for each variant.
Needs build/<campaign>/georeference.json (georeference.py) for the terrain variants.
"""
import argparse
import json
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import matplotlib
import numpy as np
import pyvista as pv

from georeference import TERRAIN_MODEL, Raster, fetch_tiles, load_georeference
from load_scans import CACHE_DIR, campaigns

VARIANTS = {
    "bim": {"scan": False, "terrain": False},
    "bim_scan": {"scan": True, "terrain": False},
    "bim_terrain": {"scan": False, "terrain": True},
    "bim_scan_terrain": {"scan": True, "terrain": True},
}
WINDOW = (1920, 1080)
FRAMES = 360
FRAMERATE = 30
SCAN_POINTS = 5_000_000
TERRAIN_RADIUS = 300.0
TERRAIN_STEP = 2
TERRAIN_BASE_DEPTH = 15.0
CONTOUR_INTERVAL = 5.0
ORBIT = {"bridge": (440.0, 150.0), "terrain": (500.0, 210.0)}
TERRAIN_COLOURS = matplotlib.colors.LinearSegmentedColormap.from_list(
    "jura_pasture", ["#3f5e34", "#5d7d43", "#7f9a55", "#a3ad6c", "#c2b98a"]
)
SKIRT_COLOUR = "#8a7560"
MODEL_OUTLINE = "#ff7a1a"
SKY = ("#dfe9f3", "#9fbad6")


def ifc_meshes(ifc_path: Path) -> list[tuple[pv.PolyData, tuple[float, float, float]]]:
    """World-coordinate tessellation of every product, merged per surface-style colour."""
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    by_colour: dict[tuple, list[pv.PolyData]] = {}
    ifc = ifcopenshell.open(ifc_path)
    it = ifcopenshell.geom.iterator(settings, ifc)
    if it.initialize():
        while True:
            geometry = it.get().geometry
            faces = np.asarray(geometry.faces, dtype=int).reshape(-1, 3)
            if len(faces):
                verts = np.asarray(geometry.verts).reshape(-1, 3)
                colours = [tuple(round(c, 3) for c in m.diffuse.components) for m in geometry.materials] or [(0.7, 0.7, 0.7)]
                ids = np.asarray(geometry.material_ids, dtype=int)
                for i, colour in enumerate(colours):
                    own = faces[ids == i] if len(ids) == len(faces) else faces
                    if len(own):
                        by_colour.setdefault(colour, []).append(pv.PolyData(verts, np.c_[np.full(len(own), 3), own]))
            if not it.next():
                break
    return [(pv.merge(parts).clean(), colour) for colour, parts in by_colour.items()]


def scan_cloud(campaign: str, max_points: int) -> pv.PolyData:
    files = sorted((CACHE_DIR / campaign).glob("*.npz"))
    xyz = np.concatenate([np.load(f)["xyz"] for f in files])
    rgb = np.concatenate([np.load(f)["rgb"] for f in files])
    keep = np.random.default_rng(0).choice(len(xyz), min(max_points, len(xyz)), replace=False)
    cloud = pv.PolyData(xyz[keep])
    cloud["rgb"] = rgb[keep]
    return cloud


def terrain_mesh(campaign: str, centre: np.ndarray) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """swissALTI3D disc around the bridge, moved into the local survey frame, with contours and a base skirt."""
    geo = load_georeference(campaign)
    centre_lv95 = geo.to_lv95(np.r_[centre, 0.0][None])[0]
    bbox = (*(centre_lv95[:2] - TERRAIN_RADIUS), *(centre_lv95[:2] + TERRAIN_RADIUS))
    dtm = Raster.mosaic(fetch_tiles(TERRAIN_MODEL, bbox, 0.5), bbox)
    east, north = np.meshgrid(dtm.east[::TERRAIN_STEP], dtm.north[::TERRAIN_STEP])
    height = dtm.z[::TERRAIN_STEP, ::TERRAIN_STEP]
    local = geo.to_local(np.c_[east.ravel(), north.ravel(), height.ravel()])
    grid = pv.StructuredGrid(*(local[:, i].reshape(east.shape) for i in range(3)))
    grid["altitude"] = height.ravel(order="F")
    grid["distance"] = np.linalg.norm(grid.points[:, :2] - centre, axis=1)
    disc = grid.extract_surface(algorithm="dataset_surface").clip_scalar(scalars="distance", value=TERRAIN_RADIUS)
    disc = disc.extract_surface(algorithm="dataset_surface").triangulate()
    levels = np.arange(np.ceil(disc["altitude"].min() / CONTOUR_INTERVAL), disc["altitude"].max() / CONTOUR_INTERVAL) * CONTOUR_INTERVAL
    contours = disc.contour(isosurfaces=levels, scalars="altitude")
    contours.points[:, 2] += 0.15
    base = pv.Plane(center=(*centre, disc.bounds.z_min - TERRAIN_BASE_DEPTH), direction=(0, 0, 1), i_size=4 * TERRAIN_RADIUS, j_size=4 * TERRAIN_RADIUS)
    rim = disc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    skirt = rim.extrude_trim((0, 0, -1), base)
    return disc.compute_normals(), contours, skirt


def bridge_centre(model: dict) -> np.ndarray:
    xy = np.array([p["centre"] for p in model["piers"]])
    return xy.mean(axis=0)


def build_scene(plotter: pv.Plotter, meshes, cloud, terrain, with_scan: bool) -> None:
    plotter.set_background(SKY[0], top=SKY[1])
    for mesh, colour in meshes:
        plotter.add_mesh(mesh, color=colour, specular=0.15, ambient=0.15, opacity=0.4 if with_scan else 1.0)
        if with_scan:
            plotter.add_mesh(mesh.extract_feature_edges(feature_angle=35), color=MODEL_OUTLINE, line_width=1.5)
    if cloud is not None:
        plotter.add_mesh(cloud, scalars="rgb", rgb=True, point_size=2.0)
    if terrain is not None:
        surface, contours, skirt = terrain
        clim = np.percentile(surface["altitude"], [1, 99])
        plotter.add_mesh(surface, scalars="altitude", cmap=TERRAIN_COLOURS, clim=clim, smooth_shading=True, show_scalar_bar=False, ambient=0.2, diffuse=0.8, specular=0.05)
        plotter.add_mesh(contours, color="#2f3a28", opacity=0.3, line_width=1.0)
        plotter.add_mesh(skirt, color=SKIRT_COLOUR, ambient=0.3)
    if with_scan:
        plotter.enable_depth_peeling(number_of_peels=4)


def orbit_camera(plotter: pv.Plotter, focus: np.ndarray, angle: float, radius: float, height: float) -> None:
    plotter.camera.focal_point = focus
    plotter.camera.position = focus + [radius * np.cos(angle), radius * np.sin(angle), height]
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 30
    plotter.camera.clipping_range = (5, 5 * radius)


def render_variant(out_dir: Path, name: str, meshes, cloud, terrain, focus: np.ndarray, start_angle: float, frames: int) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    build_scene(plotter, meshes, cloud, terrain, with_scan=cloud is not None)
    radius, height = ORBIT["terrain" if terrain is not None else "bridge"]
    orbit_camera(plotter, focus, start_angle, radius, height)
    plotter.screenshot(out_dir / f"{name}.png")
    if frames:
        plotter.open_movie(out_dir / f"{name}.mp4", framerate=FRAMERATE, quality=8, macro_block_size=8)
        for angle in start_angle + np.linspace(0, 2 * np.pi, frames, endpoint=False):
            orbit_camera(plotter, focus, angle, radius, height)
            plotter.write_frame()
    plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default=campaigns()[-1], choices=campaigns())
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--frames", type=int, default=FRAMES, help="0 renders the stills only")
    args = parser.parse_args()
    build_dir = Path("build") / args.campaign
    out_dir = build_dir / "viz"
    out_dir.mkdir(exist_ok=True)

    model = json.loads((build_dir / "bridge_model.json").read_text())
    centre = bridge_centre(model)
    focus = np.r_[centre, np.mean([p["z_bottom"] + p["height"] for p in model["piers"]]) - 8.0]
    start_angle = np.arctan2(*(np.array(model["alignment"]["centre"]) - centre)[::-1]) + np.radians(35)

    meshes = ifc_meshes(build_dir / "pont_de_la_praz.ifc")
    needs = {k for v in args.variants for k, on in VARIANTS[v].items() if on}
    cloud = scan_cloud(args.campaign, SCAN_POINTS) if "scan" in needs else None
    terrain = terrain_mesh(args.campaign, centre) if "terrain" in needs else None
    for name in args.variants:
        spec = VARIANTS[name]
        render_variant(
            out_dir, name, meshes, cloud if spec["scan"] else None, terrain if spec["terrain"] else None,
            focus, start_angle, args.frames,
        )
        print(f"rendered {out_dir / name}")


if __name__ == "__main__":
    main()
