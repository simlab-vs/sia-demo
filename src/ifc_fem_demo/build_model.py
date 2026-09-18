"""Stage 1a: build a small house as an IFC4 model.

The house is a concrete box (four walls, an interior shear wall, a ground
slab) under a pitched roof of two cross-laminated timber panels. The panels
rest on the long walls at the eaves and on a glulam ridge beam that stands on
a post at each gable. The gable triangles are glazed and carry no load.

Lengths are metres, which is also the unit declared in the IFC project, so
the numbers written to the file are the numbers you see here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.profile
import ifcopenshell.api.project
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import numpy as np

MODEL_DIR = Path("model")
BASE_MODEL_PATH = MODEL_DIR / "base_building.ifc"

CONCRETE = "Concrete C30/37"
GLULAM = "Glulam GL24h"
CLT = "CLT 5-layer"
GLASS = "Glass"


@dataclass(frozen=True)
class HouseDimensions:
    length: float = 5.0  # outer footprint along X
    width: float = 4.0  # outer footprint along Y
    wall_height: float = 3.0
    wall_thickness: float = 0.2
    slab_thickness: float = 0.2
    roof_pitch: float = 0.75  # rise over run, about 37 degrees
    roof_thickness: float = 0.15
    roof_overhang: float = 0.4  # in plan, beyond the wall centre lines
    ridge_beam_width: float = 0.16
    ridge_beam_depth: float = 0.40
    post_size: float = 0.16
    glazing_thickness: float = 0.05
    shear_wall_x: float = 3.5  # centre line of the interior wall

    @property
    def half_thickness(self) -> float:
        return self.wall_thickness / 2

    @property
    def ridge_height(self) -> float:
        """Height of the roof mid-surface at the ridge, above the wall centre lines."""
        run = self.width / 2 - self.half_thickness
        return self.wall_height + self.roof_pitch * run

    @property
    def roof_slope_angle(self) -> float:
        return float(np.arctan(self.roof_pitch))


def placement_matrix(x: float, y: float, z: float, rotation_deg: float = 0.0) -> np.ndarray:
    """4x4 world matrix: translate to (x, y, z) and rotate about Z."""
    angle = np.radians(rotation_deg)
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def placement_from_axes(origin, x_axis, y_axis) -> np.ndarray:
    """4x4 world matrix with the given local X and Y axes (Z follows the right-hand rule)."""
    x_axis = np.asarray(x_axis, dtype=float) / np.linalg.norm(x_axis)
    y_axis = np.asarray(y_axis, dtype=float) / np.linalg.norm(y_axis)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = x_axis, y_axis, np.cross(x_axis, y_axis)
    matrix[:3, 3] = origin
    return matrix


def create_spatial_hierarchy(model: ifcopenshell.file) -> ifcopenshell.entity_instance:
    project = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name="IFC FEM Demo")
    ifcopenshell.api.unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"})

    site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingStorey", name="Ground Floor")

    ifcopenshell.api.aggregate.assign_object(model, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(model, products=[building], relating_object=site)
    ifcopenshell.api.aggregate.assign_object(model, products=[storey], relating_object=building)
    return storey


def create_body_context(model: ifcopenshell.file) -> ifcopenshell.entity_instance:
    model_context = ifcopenshell.api.context.add_context(model, context_type="Model")
    return ifcopenshell.api.context.add_context(
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_context,
    )


class HouseBuilder:
    """Adds products to one storey of an IFC file, in world coordinates."""

    def __init__(self, model: ifcopenshell.file, dims: HouseDimensions):
        self.model = model
        self.dims = dims
        self.storey = create_spatial_hierarchy(model)
        self.body = create_body_context(model)
        self.materials = {
            name: ifcopenshell.api.material.add_material(model, name=name, category=category)
            for name, category in ((CONCRETE, "concrete"), (GLULAM, "wood"), (CLT, "wood"), (GLASS, "glass"))
        }

    def _place(self, product, representation, matrix: np.ndarray, material: str, container=None) -> None:
        ifcopenshell.api.geometry.assign_representation(self.model, product=product, representation=representation)
        if container is None:
            ifcopenshell.api.spatial.assign_container(self.model, products=[product], relating_structure=self.storey)
        else:
            ifcopenshell.api.aggregate.assign_object(self.model, products=[product], relating_object=container)
        ifcopenshell.api.geometry.edit_object_placement(self.model, product=product, matrix=matrix)
        ifcopenshell.api.material.assign_material(
            self.model, products=[product], type="IfcMaterial", material=self.materials[material]
        )

    def add_wall(self, name: str, length: float, matrix: np.ndarray) -> ifcopenshell.entity_instance:
        wall = ifcopenshell.api.root.create_entity(self.model, ifc_class="IfcWall", name=name)
        # An extruded rectangle: `length` along local X, `thickness` along local +Y,
        # `height` along local Z. The wall's origin is its bottom-left-front corner.
        shape = ifcopenshell.api.geometry.add_wall_representation(
            self.model,
            context=self.body,
            length=length,
            height=self.dims.wall_height,
            thickness=self.dims.wall_thickness,
        )
        self._place(wall, shape, matrix, CONCRETE)
        return wall

    def add_ground_slab(self) -> ifcopenshell.entity_instance:
        d = self.dims
        # BASESLAB says "ground-bearing": the structural stage treats it as the
        # support the walls stand on rather than as a plate to analyse.
        slab = ifcopenshell.api.root.create_entity(
            self.model, ifc_class="IfcSlab", predefined_type="BASESLAB", name="Ground Slab"
        )
        footprint = [(0.0, 0.0), (d.length, 0.0), (d.length, d.width), (0.0, d.width)]
        shape = ifcopenshell.api.geometry.add_slab_representation(
            self.model, context=self.body, depth=d.slab_thickness, polyline=footprint
        )
        # The slab is extruded upwards from its placement, so drop it by one
        # thickness to make its top face the storey level (z = 0) where the walls start.
        self._place(slab, shape, placement_matrix(0.0, 0.0, -d.slab_thickness), CONCRETE)
        return slab

    def add_roof(self) -> ifcopenshell.entity_instance:
        """Two sloped CLT panels aggregated into an IfcRoof."""
        d = self.dims
        roof = ifcopenshell.api.root.create_entity(self.model, ifc_class="IfcRoof", name="Roof")
        ifcopenshell.api.spatial.assign_container(self.model, products=[roof], relating_structure=self.storey)
        ifcopenshell.api.geometry.edit_object_placement(self.model, product=roof, matrix=np.eye(4))

        length = d.length - d.wall_thickness + 2 * d.roof_overhang
        run = d.width / 2 - d.half_thickness + d.roof_overhang
        panel_width = float(run * np.hypot(1.0, d.roof_pitch))
        x_start = d.half_thickness - d.roof_overhang
        ridge_y = d.width / 2
        # Each panel's local X runs along the ridge and local Y down the slope
        # from the ridge to the eave; local Z is then the outward normal.
        for name, down_slope in (("Roof South", (0.0, -1.0, -d.roof_pitch)), ("Roof North", (0.0, 1.0, -d.roof_pitch))):
            panel = ifcopenshell.api.root.create_entity(
                self.model, ifc_class="IfcSlab", predefined_type="ROOF", name=name
            )
            outline = [(0.0, 0.0), (length, 0.0), (length, panel_width), (0.0, panel_width)]
            shape = ifcopenshell.api.geometry.add_slab_representation(
                self.model, context=self.body, depth=d.roof_thickness, polyline=outline
            )
            matrix = placement_from_axes((x_start, ridge_y, d.ridge_height), (1.0, 0.0, 0.0), down_slope)
            # The slab is extruded along +Z from its placement plane, so start it
            # half a thickness below the mid-surface, which is what the FE model uses.
            matrix[:3, 3] -= matrix[:3, 2] * d.roof_thickness / 2
            self._place(panel, shape, matrix, CLT, container=roof)
        return roof

    def _add_bar(self, ifc_class: str, name: str, width: float, depth: float, start, end, material: str):
        """A rectangular section extruded from `start` to `end` (both on the axis)."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        axis = end - start
        bar = ifcopenshell.api.root.create_entity(self.model, ifc_class=ifc_class, name=name)
        profile = ifcopenshell.api.profile.add_parameterized_profile(self.model, ifc_class="IfcRectangleProfileDef")
        profile.XDim, profile.YDim = width, depth
        # Profiles extrude along local Z with the section centred on the axis
        # (cardinal point 5). Local Y is chosen as world Z for a horizontal bar so
        # the section's `depth` is its vertical dimension.
        shape = ifcopenshell.api.geometry.add_profile_representation(
            self.model, context=self.body, profile=profile, depth=float(np.linalg.norm(axis))
        )
        local_y = (0.0, 0.0, 1.0) if abs(axis[2]) < 1e-9 else (0.0, 1.0, 0.0)
        local_x = np.cross(local_y, axis)
        self._place(bar, shape, placement_from_axes(start, local_x, local_y), material)
        return bar

    def add_ridge_beam_and_posts(self) -> list[ifcopenshell.entity_instance]:
        d = self.dims
        x_west, x_east = d.half_thickness, d.length - d.half_thickness
        ridge = (d.width / 2, d.ridge_height)
        beam = self._add_bar(
            "IfcBeam",
            "Ridge Beam",
            d.ridge_beam_width,
            d.ridge_beam_depth,
            (x_west - d.roof_overhang, *ridge),
            (x_east + d.roof_overhang, *ridge),
            GLULAM,
        )
        # The posts run up to the beam axis (inside the beam) so that beam,
        # post and roof mid-surface share one point in the structural model.
        posts = [
            self._add_bar("IfcColumn", name, d.post_size, d.post_size, (x, ridge[0], d.wall_height), (x, *ridge), GLULAM)
            for name, x in (("Post West", x_west), ("Post East", x_east))
        ]
        return [beam, *posts]

    def add_gable_glazing(self) -> list[ifcopenshell.entity_instance]:
        """Triangular glass infill above each gable wall; non-load-bearing."""
        d = self.dims
        y0, y1 = d.half_thickness, d.width - d.half_thickness
        # The roof underside at the ridge: mid-surface minus half a thickness on the slope.
        apex = float(d.ridge_height - d.roof_thickness / (2 * np.cos(d.roof_slope_angle)))
        triangle = [(y0, d.wall_height), (y1, d.wall_height), (d.width / 2, apex), (y0, d.wall_height)]
        glazing = []
        for name, x in (("Gable Glazing West", d.half_thickness), ("Gable Glazing East", d.length - d.half_thickness)):
            panel = ifcopenshell.api.root.create_entity(self.model, ifc_class="IfcCurtainWall", name=name)
            profile = ifcopenshell.api.profile.add_arbitrary_profile(self.model, profile=triangle)
            shape = ifcopenshell.api.geometry.add_profile_representation(
                self.model, context=self.body, profile=profile, depth=d.glazing_thickness, cardinal_point=None
            )
            # Profile X along world Y, profile Y along world Z; the extrusion
            # (world X) is centred on the gable wall's centre line.
            matrix = placement_from_axes((x - d.glazing_thickness / 2, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            self._place(panel, shape, matrix, GLASS)
            glazing.append(panel)
        return glazing


def build_model(dims: HouseDimensions = HouseDimensions()) -> ifcopenshell.file:
    model = ifcopenshell.api.project.create_file(version="IFC4")
    house = HouseBuilder(model, dims)

    t = dims.wall_thickness
    # The south and north walls run the full length; the east, west and shear
    # walls fit between them so the corners do not overlap.
    inner_width = dims.width - 2 * t
    wall_specs = [
        ("Wall South", dims.length, placement_matrix(0.0, 0.0, 0.0)),
        ("Wall North", dims.length, placement_matrix(0.0, dims.width - t, 0.0)),
        # A wall rotated by 90 degrees has its thickness pointing to local +Y,
        # which is world -X, hence the +t shift.
        ("Wall West", inner_width, placement_matrix(t, t, 0.0, rotation_deg=90)),
        ("Wall East", inner_width, placement_matrix(dims.length, t, 0.0, rotation_deg=90)),
        # An interior wall across the short direction: it stiffens the box
        # against wind on the long walls, which is what a shear wall is for.
        # It sits off-centre, clear of the window that Stage 1b cuts.
        ("Shear Wall", inner_width, placement_matrix(dims.shear_wall_x + t / 2, t, 0.0, rotation_deg=90)),
    ]
    for name, length, matrix in wall_specs:
        house.add_wall(name, length, matrix)
    house.add_ground_slab()
    house.add_roof()
    house.add_ridge_beam_and_posts()
    house.add_gable_glazing()
    return model


def main(output_path: Path = BASE_MODEL_PATH) -> Path:
    model = build_model()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))
    counts = {ifc_class: len(model.by_type(ifc_class)) for ifc_class in ("IfcWall", "IfcSlab", "IfcBeam", "IfcColumn")}
    print(f"Wrote {output_path}: " + ", ".join(f"{n} {c[3:].lower()}s" for c, n in counts.items()))
    return output_path


if __name__ == "__main__":
    main()
