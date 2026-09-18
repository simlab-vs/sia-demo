"""Stage 2: 3D shell-and-frame analysis of the house with PyNite.

Walls and roof panels become quad shell elements, the ridge beam and posts
become frame members, and the ground slab becomes the fixed support. The
model is linear, static and coarse on purpose: the point is the data flow
IFC -> structural idealisation -> mesh -> solve -> stress field, and how a
window opening redistributes stress into its corners.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.element
import numpy as np
import scipy.sparse
from Pynite import FEModel3D

from ifc_fem_demo.add_window import TARGET_WALL_NAME, WINDOW_MODEL_PATH
from ifc_fem_demo.build_model import CLT, CONCRETE, GLULAM
from ifc_fem_demo.ifc_geometry import axis_segment, mid_surface

COMBO = "SLS"
PLANE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Material:
    youngs_modulus: float  # Pa
    poisson_ratio: float
    shear_modulus: float  # Pa, only used by the frame members
    density: float  # kg/m3


# Structural properties looked up by the IfcMaterial name assigned in Stage 1.
# CLT and glulam are treated as isotropic, which is crude but keeps the demo simple.
MATERIALS = {
    CONCRETE: Material(30e9, 0.2, 12.5e9, 2400.0),
    CLT: Material(11e9, 0.3, 0.69e9, 500.0),
    GLULAM: Material(11.6e9, 0.3, 0.72e9, 420.0),
}


@dataclass(frozen=True)
class Loads:
    gravity: float = 9.81  # m/s2
    snow_pressure: float = 2.5e3  # Pa on the roof panels, a heavy alpine snow load
    wind_pressure: float = 1.5e3  # Pa on the windward wall, pushing inwards
    windward_wall: str = TARGET_WALL_NAME


@dataclass(frozen=True)
class Rect:
    """A rectangle in space: origin + s * u_edge + t * v_edge for s, t in [0, 1]."""

    origin: np.ndarray
    u_edge: np.ndarray
    v_edge: np.ndarray

    @classmethod
    def from_corners(cls, corners: np.ndarray) -> Rect:
        return cls(corners[0], corners[1] - corners[0], corners[3] - corners[0])

    @property
    def corners(self) -> np.ndarray:
        o, u, v = self.origin, self.u_edge, self.v_edge
        return np.array([o, o + u, o + u + v, o + v])

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u_edge, self.v_edge)
        return n / np.linalg.norm(n)

    def point(self, s: float, t: float) -> np.ndarray:
        return self.origin + s * self.u_edge + t * self.v_edge

    @property
    def area(self) -> float:
        return float(np.linalg.norm(np.cross(self.u_edge, self.v_edge)))

    def parameters(self, point: np.ndarray) -> tuple[float, float] | None:
        """(s, t) of `point` if it lies in the rectangle's plane, else None."""
        offset = point - self.origin
        if abs(offset @ self.normal) > PLANE_TOLERANCE:
            return None
        return offset @ self.u_edge / (self.u_edge @ self.u_edge), offset @ self.v_edge / (self.v_edge @ self.v_edge)

    def contains(self, point: np.ndarray) -> bool:
        """True if `point` lies on this rectangle (not on its edges)."""
        st = self.parameters(point)
        return st is not None and all(PLANE_TOLERANCE < c < 1 - PLANE_TOLERANCE for c in st)

    def on_boundary(self, point: np.ndarray) -> bool:
        st = self.parameters(point)
        if st is None or any(c < -PLANE_TOLERANCE or c > 1 + PLANE_TOLERANCE for c in st):
            return False
        return any(abs(c) < PLANE_TOLERANCE or abs(c - 1) < PLANE_TOLERANCE for c in st)


@dataclass
class Panel:
    name: str
    surface: Rect
    thickness: float
    material: str
    openings: list[Rect] = field(default_factory=list)


@dataclass(frozen=True)
class Bar:
    name: str
    start: np.ndarray
    end: np.ndarray
    width: float  # horizontal section dimension
    depth: float  # vertical section dimension
    material: str


@dataclass
class Structure:
    panels: list[Panel]
    bars: list[Bar]

    @property
    def control_points(self) -> np.ndarray:
        """Points where elements meet; mesh lines must pass through them."""
        points = [panel.surface.corners for panel in self.panels]
        points += [opening.corners for panel in self.panels for opening in panel.openings]
        points += [np.array([bar.start, bar.end]) for bar in self.bars]
        return np.vstack(points)


def _material_name(element: ifcopenshell.entity_instance) -> str:
    return ifcopenshell.util.element.get_material(element).Name


def _is_structural_slab(slab: ifcopenshell.entity_instance) -> bool:
    # A ground-bearing slab (BASESLAB) is the support, not a plate to analyse.
    return slab.PredefinedType != "BASESLAB"


def join_walls_at_centrelines(walls: list[Panel], tolerance: float) -> None:
    """Extend or trim each wall's ends onto the mid-planes of the walls it meets.

    The IFC boxes butt against each other's faces; the shell model wants their
    mid-surfaces to meet, which is the usual idealisation for a wall panel model.
    """
    for wall in walls:
        rect = wall.surface
        along = rect.u_edge / np.linalg.norm(rect.u_edge)
        for other in walls:
            other_normal = other.surface.normal
            if other is wall or abs(along @ other_normal) < 0.9:
                continue
            for end_index in (0, 1):
                end = rect.point(end_index, 0.0)
                gap = (other.surface.origin - end) @ other_normal
                if 0 < abs(gap) <= tolerance:
                    shift = along * gap / (along @ other_normal)
                    origin = rect.origin + shift if end_index == 0 else rect.origin
                    u_edge = rect.u_edge - shift if end_index == 0 else rect.u_edge + shift
                    rect = Rect(origin, u_edge, rect.v_edge)
        wall.surface = rect


def structure_from_ifc(model: ifcopenshell.file) -> Structure:
    """Idealise the IFC products as mid-surface panels and axis bars."""
    walls = []
    for wall in model.by_type("IfcWall"):
        corners, thickness = mid_surface(wall)
        openings = [Rect.from_corners(mid_surface(rel.RelatedOpeningElement)[0]) for rel in wall.HasOpenings]
        walls.append(Panel(wall.Name, Rect.from_corners(corners), thickness, _material_name(wall), openings))
    join_walls_at_centrelines(walls, tolerance=max(wall.thickness for wall in walls))

    roof_panels = []
    for slab in filter(_is_structural_slab, model.by_type("IfcSlab")):
        corners, thickness = mid_surface(slab)
        roof_panels.append(Panel(slab.Name, Rect.from_corners(corners), thickness, _material_name(slab)))

    bars = []
    for element in model.by_type("IfcBeam") + model.by_type("IfcColumn"):
        start, end, (width, depth) = axis_segment(element)
        bars.append(Bar(element.Name, start, end, width, depth, _material_name(element)))
    return Structure(walls + roof_panels, bars)


def _grid_lines(rect: Rect, edge: np.ndarray, other_edge: np.ndarray, planes: list[np.ndarray], mesh_size: float) -> np.ndarray:
    """Parameter values along `edge` where the panel is cut, then refined to `mesh_size`.

    Every element corner in the model defines a plane x = c, y = c or z = c.
    Where such a plane crosses this panel along a grid line, the line is kept,
    so neighbouring panels (and openings) end up sharing nodes.
    """
    cuts = {0.0, 1.0}
    for axis in range(3):
        if abs(edge[axis]) < PLANE_TOLERANCE or abs(other_edge[axis]) > PLANE_TOLERANCE:
            continue
        for value in planes[axis]:
            s = (value - rect.origin[axis]) / edge[axis]
            if PLANE_TOLERANCE < s < 1 - PLANE_TOLERANCE:
                cuts.add(float(s))
    cuts = sorted(cuts)
    # Two planes can cut the panel along the same line (a wall top is both a
    # z = c plane and, on the roof, a y = c plane); keep one to avoid slivers.
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] > 1e-6]

    length = np.linalg.norm(edge)
    lines = [0.0]
    for start, end in zip(cuts[:-1], cuts[1:]):
        # Neighbouring panels only share nodes if they divide a common stretch
        # into the same number of cells, so an interval that is 9 cells long up
        # to floating point noise must give 9, never 10.
        divisions = max(1, math.ceil((end - start) * length / mesh_size - 1e-6))
        lines.extend(np.linspace(start, end, divisions + 1)[1:])
    return np.array(lines)


@dataclass
class FeMesh:
    """The PyNite model plus plain arrays describing it, for post-processing and rendering."""

    model: FEModel3D
    node_names: list[str]
    points: np.ndarray  # (n_nodes, 3)
    quads: np.ndarray  # (n_quads, 4) node indices
    quad_names: list[str]
    quad_panels: list[str]  # IFC element name each quad belongs to
    members: np.ndarray  # (n_members, 2) node indices
    member_names: list[str]
    applied_force: dict[str, np.ndarray] = field(default_factory=dict)  # resultant per load case, N

    def combo_force(self, combo: str) -> np.ndarray:
        factors = self.model.load_combos[combo].factors
        return sum((factor * self.applied_force[case] for case, factor in factors.items()), np.zeros(3))

    def free_nodes(self) -> np.ndarray:
        return np.array([not self.model.nodes[name].support_DX for name in self.node_names])

    def displacements(self, combo: str = COMBO) -> np.ndarray:
        nodes = self.model.nodes
        return np.array([[nodes[n].DX[combo], nodes[n].DY[combo], nodes[n].DZ[combo]] for n in self.node_names])

    def quad_centres(self) -> np.ndarray:
        return self.points[self.quads].mean(axis=1)


class MeshBuilder:
    def __init__(self, structure: Structure, mesh_size: float, extra_control_points: np.ndarray | None = None):
        self.structure = structure
        self.mesh_size = mesh_size
        self.extra_control_points = extra_control_points
        self.model = FEModel3D()
        for name, m in MATERIALS.items():
            self.model.add_material(name, m.youngs_modulus, m.shear_modulus, m.poisson_ratio, m.density)
        self.node_index: dict[tuple[int, int, int], int] = {}
        self.points: list[np.ndarray] = []
        self.quads: list[list[int]] = []
        self.quad_names: list[str] = []
        self.quad_panels: list[str] = []
        self.members: list[list[int]] = []
        self.member_names: list[str] = []

    def node_at(self, point: np.ndarray) -> int:
        key = tuple(int(round(c * 1e3)) for c in point)
        if key not in self.node_index:
            self.node_index[key] = len(self.points)
            self.points.append(np.asarray(point, dtype=float))
            self.model.add_node(f"N{len(self.points)}", *map(float, point))
        return self.node_index[key]

    def _node_name(self, index: int) -> str:
        return f"N{index + 1}"

    def add_panel(self, panel: Panel, planes: list[np.ndarray]) -> None:
        rect = panel.surface
        s_lines = _grid_lines(rect, rect.u_edge, rect.v_edge, planes, self.mesh_size)
        t_lines = _grid_lines(rect, rect.v_edge, rect.u_edge, planes, self.mesh_size)
        for i in range(len(s_lines) - 1):
            for j in range(len(t_lines) - 1):
                centre = rect.point((s_lines[i] + s_lines[i + 1]) / 2, (t_lines[j] + t_lines[j + 1]) / 2)
                if any(opening.contains(centre) for opening in panel.openings):
                    continue
                # Nodes are created per kept quad, so an opening leaves no orphan
                # nodes behind (PyNite would flag those as unstable).
                cell = [(s_lines[i], t_lines[j]), (s_lines[i + 1], t_lines[j]), (s_lines[i + 1], t_lines[j + 1]), (s_lines[i], t_lines[j + 1])]
                corners = [self.node_at(rect.point(s, t)) for s, t in cell]
                name = f"Q{len(self.quads) + 1}"
                self.model.add_quad(name, *map(self._node_name, corners), panel.thickness, panel.material)
                self.quads.append(corners)
                self.quad_names.append(name)
                self.quad_panels.append(panel.name)

    def add_bar(self, bar: Bar) -> None:
        """One PyNite member per stretch between the nodes already lying on the bar's axis.

        Members only connect at their end nodes, so a beam under a meshed roof
        has to be split wherever the roof mesh has a node on the ridge line.
        """
        w, d = bar.width, bar.depth
        # PyNite's local y axis of a horizontal member is global Y, so Iy governs
        # bending in the vertical plane (the strong axis of the ridge beam).
        self.model.add_section(bar.name, A=w * d, Iy=w * d**3 / 12, Iz=d * w**3 / 12, J=w * d * (w**2 + d**2) / 12)

        axis = bar.end - bar.start
        length_squared = axis @ axis
        for endpoint in (bar.start, bar.end):
            self.node_at(endpoint)
        on_axis = []
        for index, point in enumerate(self.points):
            offset = point - bar.start
            fraction = offset @ axis / length_squared
            distance = np.linalg.norm(offset - fraction * axis)
            if distance < PLANE_TOLERANCE and -PLANE_TOLERANCE <= fraction <= 1 + PLANE_TOLERANCE:
                on_axis.append((fraction, index))
        on_axis.sort()

        for k, ((_, i), (_, j)) in enumerate(zip(on_axis[:-1], on_axis[1:])):
            name = f"{bar.name} {k + 1}"
            self.model.add_member(name, self._node_name(i), self._node_name(j), bar.material, bar.name)
            self.members.append([i, j])
            self.member_names.append(name)

    def build(self) -> FeMesh:
        control = self.structure.control_points
        if self.extra_control_points is not None:
            control = np.vstack([control, self.extra_control_points])
        planes = [np.unique(np.round(control[:, axis], 6)) for axis in range(3)]
        for panel in self.structure.panels:
            self.add_panel(panel, planes)
        for bar in self.structure.bars:
            self.add_bar(bar)
        return FeMesh(
            self.model,
            [self._node_name(i) for i in range(len(self.points))],
            np.array(self.points),
            np.array(self.quads),
            self.quad_names,
            self.quad_panels,
            np.array(self.members).reshape(-1, 2),
            self.member_names,
        )


def fix_base(mesh: FeMesh) -> None:
    base_level = mesh.points[:, 2].min()
    for name, point in zip(mesh.node_names, mesh.points):
        if abs(point[2] - base_level) < PLANE_TOLERANCE:
            mesh.model.def_support(name, True, True, True, True, True, True)


def _quad_area(mesh: FeMesh, quad: np.ndarray) -> float:
    p = mesh.points[quad]
    return 0.5 * float(np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1])))


def _lump_on_nodes(mesh: FeMesh, node_indices, force: np.ndarray, case: str) -> None:
    """Spread a total force equally over the given nodes."""
    share = force / len(node_indices)
    for index in node_indices:
        for direction, component in zip(("FX", "FY", "FZ"), share):
            if component != 0.0:
                mesh.model.add_node_load(mesh.node_names[index], direction, float(component), case=case)
    mesh.applied_force[case] = mesh.applied_force.get(case, np.zeros(3)) + force


def _panel_weights(mesh: FeMesh, structure: Structure, gravity: float):
    """(quad, weight) for every shell element: density x g x thickness x area."""
    panels = {panel.name: panel for panel in structure.panels}
    for quad, panel_name in zip(mesh.quads, mesh.quad_panels):
        panel = panels[panel_name]
        yield quad, MATERIALS[panel.material].density * gravity * panel.thickness * _quad_area(mesh, quad)


def _member_weight(mesh: FeMesh, gravity: float) -> float:
    return sum(MATERIALS[m.material.name].density * gravity * m.section.A * m.L() for m in mesh.model.members.values())


def _inward_normal(panel: Panel, house_centre: np.ndarray) -> np.ndarray:
    normal = panel.surface.normal
    return normal if normal @ (house_centre - panel.surface.origin) > 0 else -normal


def apply_loads(mesh: FeMesh, structure: Structure, loads: Loads) -> None:
    """Dead, snow and wind as lumped nodal loads, combined 1:1:1 for a serviceability check."""
    panels = {panel.name: panel for panel in structure.panels}
    down = np.array([0.0, 0.0, -1.0])

    for quad, weight in _panel_weights(mesh, structure, loads.gravity):
        _lump_on_nodes(mesh, quad, down * weight, "Dead")
    for quad, panel_name in zip(mesh.quads, mesh.quad_panels):
        if panel_name.startswith("Roof"):
            _lump_on_nodes(mesh, quad, down * loads.snow_pressure * _quad_area(mesh, quad), "Snow")

    windward = panels[loads.windward_wall]
    inward = _inward_normal(windward, mesh.points.mean(axis=0))
    for quad, panel_name in zip(mesh.quads, mesh.quad_panels):
        if panel_name == loads.windward_wall:
            _lump_on_nodes(mesh, quad, inward * loads.wind_pressure * _quad_area(mesh, quad), "Wind")
    # The window pane is not modelled, but it still catches wind and hands it
    # to the edges of its opening, so that share goes onto the perimeter nodes.
    for opening in windward.openings:
        perimeter = [i for i, point in enumerate(mesh.points) if opening.on_boundary(point)]
        _lump_on_nodes(mesh, perimeter, inward * loads.wind_pressure * opening.area, "Wind")

    mesh.model.add_member_self_weight("FZ", -loads.gravity, case="Dead")
    mesh.applied_force["Dead"] += down * _member_weight(mesh, loads.gravity)

    mesh.model.add_load_combo(COMBO, {"Dead": 1.0, "Snow": 1.0, "Wind": 1.0})


@dataclass(frozen=True)
class LateralLoad:
    """Equivalent static seismic action: the dead mass accelerated along one horizontal axis.

    The house is far stiffer than the corner period of any spectrum here, so it
    moves as a rigid box and every kilogram sees the same spectral acceleration.
    """

    name: str
    axis: int  # 0 = X, 1 = Y
    acceleration: float  # m/s2
    level: str = ""  # where the acceleration comes from, e.g. "median" or "SIA II"

    @property
    def unit_case(self) -> str:
        return UNIT_INERTIA_CASES[self.axis]

    @property
    def direction(self) -> np.ndarray:
        return np.eye(3)[self.axis]


UNIT_INERTIA_CASES = ("EQX", "EQY")


def apply_unit_inertia(mesh: FeMesh, structure: Structure) -> None:
    """Load cases EQX and EQY: the dead mass under 1 m/s2 along X and Y, on the nodes free to move.

    Mass lumped on the fixed base nodes goes straight into the ground, exactly
    as the modal analysis drops it, so an equivalent static combination
    produces the base shear SA x seismic mass.
    """
    free = mesh.free_nodes()
    for axis, case in enumerate(UNIT_INERTIA_CASES):
        direction = np.eye(3)[axis]
        for quad, mass in _panel_weights(mesh, structure, gravity=1.0):
            moving = [index for index in quad if free[index]]
            if moving:
                _lump_on_nodes(mesh, moving, direction * mass * len(moving) / len(quad), case)
        mesh.model.add_member_self_weight("FX" if axis == 0 else "FY", 1.0, case=case)
        mesh.applied_force[case] += direction * _member_weight(mesh, gravity=1.0)


def add_lateral_combos(mesh: FeMesh, lateral_loads: Iterable[LateralLoad]) -> None:
    """Dead weight plus the accelerated mass; snow and wind are not combined with the earthquake."""
    for load in lateral_loads:
        mesh.model.add_load_combo(load.name, {"Dead": 1.0, load.unit_case: load.acceleration})


def von_mises_per_quad(mesh: FeMesh, structure: Structure, combo: str = COMBO) -> np.ndarray:
    """Von Mises stress at each quad centre, on whichever face is more stressed.

    PyNite gives membrane stresses and bending moments per unit length at the
    element centre; the bending stress on the faces is 6 M / t^2.
    """
    thickness = {panel.name: panel.thickness for panel in structure.panels}
    stresses = np.empty(len(mesh.quads))
    for k, (name, panel_name) in enumerate(zip(mesh.quad_names, mesh.quad_panels)):
        quad = mesh.model.quads[name]
        membrane = quad.membrane(0, 0, combo_name=combo).ravel()
        bending = 6 * quad.moment(0, 0, combo_name=combo).ravel() / thickness[panel_name] ** 2
        faces = np.array([membrane + bending, membrane - bending])
        sx, sy, txy = faces.T
        stresses[k] = np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2).max()
    return stresses


def total_reaction(mesh: FeMesh, combo: str = COMBO) -> np.ndarray:
    reactions = np.zeros(3)
    for node in mesh.model.nodes.values():
        reactions += [node.RxnFX[combo], node.RxnFY[combo], node.RxnFZ[combo]]
    return reactions


def reaction_imbalance(mesh: FeMesh, combo: str = COMBO) -> float:
    """|sum of reactions + applied loads| relative to the applied loads: ~0 means equilibrium."""
    applied = mesh.combo_force(combo)
    return float(np.linalg.norm(total_reaction(mesh, combo) + applied) / np.linalg.norm(applied))


@dataclass
class AnalysisResult:
    case: str
    combo: str
    structure: Structure
    mesh: FeMesh
    displacements: np.ndarray  # (n_nodes, 3), m
    von_mises: np.ndarray  # (n_quads,), Pa
    max_displacement: float  # m
    max_lateral_displacement: float  # m, in the horizontal plane
    max_von_mises: float  # Pa, over the whole house
    max_stress_element: str
    wall_max_von_mises: float  # Pa, in the wall that gets the window
    wall_max_location: tuple[float, float, float]
    corner_von_mises: float  # Pa, within a couple of elements of the window corners
    reaction: np.ndarray  # (3,), N, sum of the support reactions
    reaction_imbalance: float

    def base_shear(self, axis: int) -> float:
        """N, the horizontal force the ground has to resist along `axis` (0 = X, 1 = Y)."""
        return float(abs(self.reaction[axis]))

    @property
    def n_quads(self) -> int:
        return len(self.mesh.quads)

    @property
    def n_members(self) -> int:
        return len(self.mesh.members)


def stress_near_corners(result_stress: np.ndarray, centres: np.ndarray, openings: list[Rect], radius: float) -> float:
    """Largest stress among elements whose centre is within `radius` of an opening corner.

    The re-entrant corner is a stress singularity in the elastic model, so
    the value there is mesh-dependent; comparing the same neighbourhood in
    the solid and the opened wall on the same mesh is still meaningful.
    """
    corners = np.array([corner for opening in openings for corner in opening.corners])
    distance_to_corner = np.linalg.norm(centres[:, None, :] - corners[None, :, :], axis=2).min(axis=1)
    return float(result_stress[distance_to_corner <= radius].max())


def solve(
    model_path: Path,
    mesh_size: float = 0.2,
    loads: Loads = Loads(),
    reference_openings: list[Rect] | None = None,
    lateral_loads: Iterable[LateralLoad] = (),
) -> tuple[Structure, FeMesh]:
    """Mesh one IFC model, load it and solve every combination in one go."""
    structure = structure_from_ifc(ifcopenshell.open(str(model_path)))
    # Meshing the solid wall with the opening's outline as extra grid lines gives
    # both cases the same mesh, so their results differ only through the opening.
    reference_corners = np.vstack([rect.corners for rect in reference_openings]) if reference_openings else None
    mesh = MeshBuilder(structure, mesh_size, reference_corners).build()
    fix_base(mesh)
    apply_loads(mesh, structure, loads)
    lateral_loads = list(lateral_loads)
    if lateral_loads:
        apply_unit_inertia(mesh, structure)
        add_lateral_combos(mesh, lateral_loads)
    mesh.model.analyze(check_statics=False, sparse=True)
    return structure, mesh


def result_for_combo(
    structure: Structure,
    mesh: FeMesh,
    case: str,
    combo: str,
    mesh_size: float,
    wall_name: str = TARGET_WALL_NAME,
    reference_openings: list[Rect] | None = None,
) -> AnalysisResult:
    displacements = mesh.displacements(combo)
    von_mises = von_mises_per_quad(mesh, structure, combo)
    quad_panels = np.array(mesh.quad_panels)
    hottest = int(np.argmax(von_mises))

    wall = next(panel for panel in structure.panels if panel.name == wall_name)
    in_wall = np.flatnonzero(quad_panels == wall_name)
    wall_hottest = in_wall[np.argmax(von_mises[in_wall])]
    centres = mesh.quad_centres()
    openings = wall.openings or reference_openings or []
    corner_stress = stress_near_corners(von_mises[in_wall], centres[in_wall], openings, 1.5 * mesh_size) if openings else 0.0

    return AnalysisResult(
        case=case,
        combo=combo,
        structure=structure,
        mesh=mesh,
        displacements=displacements,
        von_mises=von_mises,
        max_displacement=float(np.linalg.norm(displacements, axis=1).max()),
        max_lateral_displacement=float(np.linalg.norm(displacements[:, :2], axis=1).max()),
        max_von_mises=float(von_mises[hottest]),
        max_stress_element=mesh.quad_panels[hottest],
        wall_max_von_mises=float(von_mises[wall_hottest]),
        wall_max_location=tuple(float(c) for c in centres[wall_hottest]),
        corner_von_mises=corner_stress,
        reaction=total_reaction(mesh, combo),
        reaction_imbalance=reaction_imbalance(mesh, combo),
    )


def analyse(
    model_path: Path,
    case: str,
    mesh_size: float = 0.2,
    loads: Loads = Loads(),
    wall_name: str = TARGET_WALL_NAME,
    reference_openings: list[Rect] | None = None,
) -> AnalysisResult:
    """Solve one IFC model under the serviceability combination; `reference_openings` says where to probe a wall that has none."""
    structure, mesh = solve(model_path, mesh_size, loads, reference_openings)
    return result_for_combo(structure, mesh, case, COMBO, mesh_size, wall_name, reference_openings)


def analyse_lateral(
    model_path: Path,
    lateral_loads: Iterable[LateralLoad],
    mesh_size: float = 0.2,
    loads: Loads = Loads(),
    wall_name: str = TARGET_WALL_NAME,
) -> dict[str, AnalysisResult]:
    """One solve of the model under dead weight plus each equivalent static lateral load."""
    lateral_loads = list(lateral_loads)
    structure, mesh = solve(model_path, mesh_size, loads, lateral_loads=lateral_loads)
    return {load.name: result_for_combo(structure, mesh, load.name, load.name, mesh_size, wall_name) for load in lateral_loads}


def lateral_table(results: dict[str, AnalysisResult], lateral_loads: Iterable[LateralLoad]) -> str:
    header = (
        f"{'case':<24}{'a [m/s2]':>10}{'base shear [kN]':>17}{'lateral [mm]':>14}"
        f"{'house vM [MPa]':>16}{'wall vM [MPa]':>15}{'corners [MPa]':>15}  house peak in"
    )
    rows = [header, "-" * len(header)]
    for load in lateral_loads:
        r = results[load.name]
        rows.append(
            f"{r.case:<24}{load.acceleration:>10.3f}{r.base_shear(load.axis) / 1e3:>17.1f}"
            f"{r.max_lateral_displacement * 1e3:>14.3f}{r.max_von_mises / 1e6:>16.3f}"
            f"{r.wall_max_von_mises / 1e6:>15.3f}{r.corner_von_mises / 1e6:>15.3f}  {r.max_stress_element}"
        )
    rows.append("\nEach case is dead weight plus the seismic mass accelerated along one axis, as a rigid box (no behaviour factor).")
    return "\n".join(rows)


@dataclass(frozen=True)
class VibrationModes:
    """Natural periods of the house and how much of its mass each mode moves."""

    periods: np.ndarray  # (n_modes,), s, longest first
    effective_mass_fraction: np.ndarray  # (n_modes, 3), share of the seismic mass moved along X, Y, Z
    seismic_mass: float  # kg, mass free to move: dead weight above the fixed base

    def dominant_period(self, axis: int) -> float:
        """Period of the mode that carries the most mass along `axis` (0 = X, 1 = Y)."""
        return float(self.periods[np.argmax(self.effective_mass_fraction[:, axis])])


def vibration_modes(model_path: Path, mesh_size: float = 0.2, n_modes: int = 12, loads: Loads = Loads()) -> VibrationModes:
    """Modal analysis of a fresh model: dead load becomes lumped mass at the nodes.

    PyNite's modal solve overwrites the static results, hence the fresh model.
    The effective modal mass is (phi' M r)^2 / (phi' M phi) for a unit rigid
    translation r; summed over all modes it would reach the whole seismic mass.
    """
    structure = structure_from_ifc(ifcopenshell.open(str(model_path)))
    mesh = MeshBuilder(structure, mesh_size).build()
    fix_base(mesh)
    apply_loads(mesh, structure, loads)
    model = mesh.model
    model.add_load_combo("Mass", {"Dead": 1.0})
    model.analyze_modal(num_modes=n_modes, mass_combo_name="Mass", mass_direction="Z", gravity=loads.gravity)

    mass_matrix = scipy.sparse.csr_matrix(model.M("Mass", "Z", loads.gravity, sparse=True))
    nodes = [model.nodes[name] for name in mesh.node_names]
    free = np.array([not node.support_DX for node in nodes])
    rigid = np.zeros((3, 6 * len(nodes)))
    for axis in range(3):
        rigid[axis, axis::6] = free
    total_mass = rigid @ mass_matrix @ rigid.T
    seismic_mass = float(total_mass[2, 2])

    fractions = np.empty((len(model.frequencies), 3))
    for k in range(len(model.frequencies)):
        combo = f"Mode {k + 1}"
        shape = np.array([[n.DX[combo], n.DY[combo], n.DZ[combo], n.RX[combo], n.RY[combo], n.RZ[combo]] for n in nodes]).ravel()
        generalised_mass = shape @ mass_matrix @ shape
        fractions[k] = (rigid @ mass_matrix @ shape) ** 2 / generalised_mass / np.diag(total_mass)
    return VibrationModes(1.0 / np.asarray(model.frequencies, dtype=float), fractions, seismic_mass)


def summary_table(results: dict[str, AnalysisResult], wall_name: str = TARGET_WALL_NAME) -> str:
    header = (
        f"{'case':<14}{'quads':>7}{'bars':>6}{'max |u| [mm]':>14}{'lateral [mm]':>14}"
        f"{'house vM [MPa]':>16}{'wall vM [MPa]':>15}{'corners [MPa]':>15}  house peak in"
    )
    rows = [header, "-" * len(header)]
    for r in results.values():
        rows.append(
            f"{r.case:<14}{r.n_quads:>7}{r.n_members:>6}{r.max_displacement * 1e3:>14.3f}"
            f"{r.max_lateral_displacement * 1e3:>14.3f}{r.max_von_mises / 1e6:>16.3f}"
            f"{r.wall_max_von_mises / 1e6:>15.3f}{r.corner_von_mises / 1e6:>15.3f}  {r.max_stress_element}"
        )
    solid, opened = results["solid"], results["with_opening"]
    rows.append(
        f"\n'corners' is the stress next to the window corners of {wall_name} (same spot in the solid wall)."
        f"\nStress at the window corners, with/without opening: {opened.corner_von_mises / solid.corner_von_mises:.2f}x"
    )
    return "\n".join(rows)


def main(
    base_path: Path,
    window_path: Path = WINDOW_MODEL_PATH,
    mesh_size: float = 0.2,
) -> dict[str, AnalysisResult]:
    opened = analyse(window_path, "with_opening", mesh_size)
    wall = next(panel for panel in opened.structure.panels if panel.name == TARGET_WALL_NAME)
    solid = analyse(base_path, "solid", mesh_size, reference_openings=wall.openings)
    results = {"solid": solid, "with_opening": opened}
    for r in results.values():
        print(f"  {r.case:<13} {r.n_quads:5d} quads, {r.n_members:3d} bars, reaction imbalance {r.reaction_imbalance:.1e}")
    return results


if __name__ == "__main__":
    from ifc_fem_demo.build_model import BASE_MODEL_PATH

    print(summary_table(main(BASE_MODEL_PATH)))
