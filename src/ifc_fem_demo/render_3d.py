"""Stage 3: VTK renders of the house and of the structural results.

Rendering goes through PyVista, the Python layer over VTK: turntable videos
of the IFC geometry (with and without the window), a side-by-side view of
the von Mises stress on the deformed structure, and a turntable of the
stressed house. Frames are encoded to MP4 with imageio's bundled ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import ifcopenshell
import numpy as np
import pyvista as pv

from ifc_fem_demo.ifc_geometry import world_mesh
from ifc_fem_demo.structural import COMBO, AnalysisResult

pv.OFF_SCREEN = True

OUTPUT_DIR = Path("output")
STRESS_CMAP = "Reds"  # sequential: one hue, light to dark
BACKGROUND = ("#f6f5f0", "#dbe6ef")  # ground tint, sky tint
GROUND_COLOR = "#e4e2d8"
TIMBER_COLOR = "#b98a5a"
CONCRETE_COLOR = "#d6d2c6"
GLASS_COLOR = "#a9d3ec"

# Rendered IFC classes and their look; openings and spatial elements are skipped.
ELEMENT_STYLE = {
    "IfcWall": (CONCRETE_COLOR, 1.0),
    "IfcSlab": (CONCRETE_COLOR, 1.0),
    "IfcBeam": (TIMBER_COLOR, 1.0),
    "IfcColumn": (TIMBER_COLOR, 1.0),
    "IfcWindow": (GLASS_COLOR, 0.55),
    "IfcCurtainWall": (GLASS_COLOR, 0.55),
}
ROOF_COLOR = "#8d6b4a"


def element_polydata(element: ifcopenshell.entity_instance) -> pv.PolyData:
    vertices, faces = world_mesh(element)
    return pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())


def _style(element: ifcopenshell.entity_instance) -> tuple[str, float]:
    if element.is_a("IfcSlab") and element.PredefinedType == "ROOF":
        return ROOF_COLOR, 1.0
    return ELEMENT_STYLE[element.is_a()]


def new_plotter(window_size: tuple[int, int] = (960, 720), **kwargs) -> pv.Plotter:
    # Frame sizes are multiples of 16 so the H.264 encoder need not pad them.
    plotter = pv.Plotter(off_screen=True, window_size=window_size, **kwargs)
    plotter.set_background(BACKGROUND[0], top=BACKGROUND[1])
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def add_ground(plotter: pv.Plotter, low: np.ndarray, high: np.ndarray) -> None:
    centre = (low + high) / 2
    # A disc rather than a square: its edge reads as a level horizon from every
    # azimuth of the turntable instead of a tilting straight line.
    ground = pv.Circle(radius=4 * (high - low).max(), resolution=180)
    ground.translate((centre[0], centre[1], low[2]), inplace=True)
    plotter.add_mesh(ground, color=GROUND_COLOR, ambient=0.4, diffuse=0.6, specular=0.0)


def add_house(plotter: pv.Plotter, model: ifcopenshell.file) -> tuple[np.ndarray, np.ndarray]:
    """Draw every renderable element; returns the scene's bounding box."""
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    glazing = []
    for ifc_class in ELEMENT_STYLE:
        for element in model.by_type(ifc_class):
            mesh = element_polydata(element)
            low, high = np.minimum(low, mesh.bounds[::2]), np.maximum(high, mesh.bounds[1::2])
            color, opacity = _style(element)
            if opacity < 1.0:
                glazing.append((mesh, color, opacity))
                continue
            plotter.add_mesh(mesh, color=color, ambient=0.25, diffuse=0.8, specular=0.1)
    # Translucent surfaces are added last so the opaque walls behind them are drawn first.
    for mesh, color, opacity in glazing:
        plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True)
    add_ground(plotter, low, high)
    return low, high


def frame_scene(plotter: pv.Plotter, low: np.ndarray, high: np.ndarray, azimuth: float = -35.0, elevation: float = 22.0) -> None:
    """Point the camera at the scene centre from the given angles (degrees)."""
    centre = (low + high) / 2
    distance = 3.4 * (high - low).max()
    direction = np.array(
        [np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)), -np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)), np.sin(np.radians(elevation))]
    )
    plotter.camera_position = [tuple(centre + distance * direction), tuple(centre), (0, 0, 1)]
    plotter.camera.view_angle = 26


def write_turntable(plotter: pv.Plotter, output_path: Path, n_frames: int, fps: int) -> Path:
    """Orbit the camera once about the vertical axis through the focal point."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_movie(str(output_path), framerate=fps, quality=8)
    for _ in range(n_frames):
        plotter.write_frame()
        plotter.camera.Azimuth(360 / n_frames)  # VTK's incremental rotation about the view-up axis
    plotter.close()
    print(f"Wrote {output_path} ({n_frames} frames at {fps} fps)")
    return output_path


def render_rotation(model_path: Path, output_path: Path, n_frames: int = 96, fps: int = 24) -> Path:
    plotter = new_plotter()
    low, high = add_house(plotter, ifcopenshell.open(str(model_path)))
    frame_scene(plotter, low, high)
    return write_turntable(plotter, output_path, n_frames, fps)


def deformation_scale(results: list[AnalysisResult], visible_fraction: float = 0.04) -> float:
    """One round scale factor making the largest displacement a visible fraction of the house."""
    size = max((r.mesh.points.max(axis=0) - r.mesh.points.min(axis=0)).max() for r in results)
    largest = max(r.max_displacement for r in results)
    raw = visible_fraction * size / largest
    return float(round(raw, -int(np.floor(np.log10(raw)))))


def structural_polydata(result: AnalysisResult, scale: float) -> tuple[pv.PolyData, pv.PolyData]:
    """(shell quads with stress as cell data, frame members as tubes), both deformed."""
    points = result.mesh.points + scale * result.displacements
    quads = result.mesh.quads
    shells = pv.PolyData(points, np.column_stack([np.full(len(quads), 4), quads]).ravel())
    shells.cell_data["von Mises [MPa]"] = result.von_mises / 1e6

    members = result.mesh.members
    bars = pv.PolyData(points, lines=np.column_stack([np.full(len(members), 2), members]).ravel())
    return shells, bars.tube(radius=0.07)


def add_structure(plotter: pv.Plotter, result: AnalysisResult, scale: float, clim: tuple[float, float], show_scalar_bar: bool) -> None:
    shells, bars = structural_polydata(result, scale)
    plotter.add_mesh(
        shells,
        scalars="von Mises [MPa]",
        cmap=STRESS_CMAP,
        clim=clim,
        show_edges=True,
        edge_color="#ffffff",
        edge_opacity=0.35,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title": "von Mises [MPa]", "vertical": True, "position_x": 0.83, "position_y": 0.3, "width": 0.04, "height": 0.45, "n_labels": 5, "fmt": "%.2f", "color": "#333333"},
    )
    plotter.add_mesh(bars, color=TIMBER_COLOR, ambient=0.25, diffuse=0.8, specular=0.1)


def add_wind_arrows(plotter: pv.Plotter, result: AnalysisResult, wall_name: str) -> None:
    """A few arrows in front of the windward wall, so the lateral load is visible."""
    wall = next(panel for panel in result.structure.panels if panel.name == wall_name)
    normal = wall.surface.normal
    if normal @ (result.mesh.points.mean(axis=0) - wall.surface.origin) < 0:
        normal = -normal
    starts = np.array([wall.surface.point(s, t) - 1.1 * normal for s in (0.2, 0.5, 0.8) for t in (0.3, 0.7)])
    arrows = pv.PolyData(starts)
    arrows["direction"] = np.tile(normal, (len(starts), 1))
    plotter.add_mesh(arrows.glyph(orient="direction", scale=False, factor=0.8, geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.08, tip_length=0.3)), color="#4a6fa5")


def render_stress_comparison(
    results: dict[str, AnalysisResult],
    output_path: Path = OUTPUT_DIR / "stress_comparison.png",
    wall_name: str = "Wall South",
) -> Path:
    """The two cases side by side, same colour scale, same deformation scale."""
    cases = list(results.values())
    scale = deformation_scale(cases)
    clim = (0.0, max(r.max_von_mises for r in cases) / 1e6)

    plotter = new_plotter(window_size=(1600, 720), shape=(1, 2), border=False)
    for column, result in enumerate(cases):
        plotter.subplot(0, column)
        add_structure(plotter, result, scale, clim, show_scalar_bar=column == len(cases) - 1)
        add_wind_arrows(plotter, result, wall_name)
        low, high = result.mesh.points.min(axis=0), result.mesh.points.max(axis=0)
        add_ground(plotter, low, high)
        frame_scene(plotter, low, high)
        plotter.add_text(
            f"{result.case.replace('_', ' ')}: {COMBO} combination, deformation x{scale:g}\n"
            f"peak {result.max_von_mises / 1e6:.2f} MPa in {result.max_stress_element}, "
            f"{result.corner_von_mises / 1e6:.2f} MPa at the window corners",
            position="upper_left",
            font_size=11,
            color="#333333",
        )
    plotter.link_views()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"Wrote {output_path}")
    return output_path


def render_stress_rotation(result: AnalysisResult, output_path: Path, n_frames: int = 96, fps: int = 24) -> Path:
    scale = deformation_scale([result])
    plotter = new_plotter()
    add_structure(plotter, result, scale, (0.0, result.max_von_mises / 1e6), show_scalar_bar=True)
    low, high = result.mesh.points.min(axis=0), result.mesh.points.max(axis=0)
    add_ground(plotter, low, high)
    frame_scene(plotter, low, high)
    plotter.add_text(f"{result.case.replace('_', ' ')}: deformation x{scale:g}", position="upper_left", font_size=11, color="#333333")
    return write_turntable(plotter, output_path, n_frames, fps)
