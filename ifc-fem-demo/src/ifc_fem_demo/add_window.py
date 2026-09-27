"""Stage 1b: cut a window opening into one wall and fill it with an IfcWindow.

IFC models openings as separate `IfcOpeningElement` products related to the
host element through `IfcRelVoidsElement`. The window is another product
that sits in the opening through `IfcRelFillsElement`. Nothing is subtracted
from the wall's own geometry; viewers apply the boolean when rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.util.element
import ifcopenshell.util.placement
import ifcopenshell.util.representation
import numpy as np

from ifc_fem_demo.build_model import BASE_MODEL_PATH, MODEL_DIR
from ifc_fem_demo.ifc_geometry import local_extents

WINDOW_MODEL_PATH = MODEL_DIR / "building_with_window.ifc"
TARGET_WALL_NAME = "Wall South"


@dataclass(frozen=True)
class WindowDimensions:
    width: float = 1.2
    height: float = 1.5
    sill_height: float = 0.9


def add_window(
    model: ifcopenshell.file,
    wall: ifcopenshell.entity_instance,
    window_dims: WindowDimensions = WindowDimensions(),
) -> tuple[ifcopenshell.entity_instance, ifcopenshell.entity_instance]:
    body = ifcopenshell.util.representation.get_context(model, "Model", "Body", "MODEL_VIEW")
    storey = ifcopenshell.util.element.get_container(wall)
    wall_length, wall_thickness, _ = local_extents(wall)
    wall_matrix = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)

    # Opening: a box as thick as the wall, centred along the wall length.
    # `add_wall_representation` is just a convenient extruded rectangle here.
    opening = ifcopenshell.api.root.create_entity(model, ifc_class="IfcOpeningElement", name="Window Opening")
    opening_shape = ifcopenshell.api.geometry.add_wall_representation(
        model, context=body, length=window_dims.width, height=window_dims.height, thickness=wall_thickness
    )
    ifcopenshell.api.geometry.assign_representation(model, product=opening, representation=opening_shape)
    ifcopenshell.api.feature.add_feature(model, feature=opening, element=wall)

    # The opening's placement is expressed relative to the wall by the API; we
    # hand it the world matrix, obtained by shifting along the wall's own axes.
    local_offset = np.array([(wall_length - window_dims.width) / 2, 0.0, window_dims.sill_height, 1.0])
    opening_matrix = wall_matrix.copy()
    opening_matrix[:, 3] = wall_matrix @ local_offset
    ifcopenshell.api.geometry.edit_object_placement(model, product=opening, matrix=opening_matrix)

    # Window: a thin pane in the middle of the opening; a real project would
    # use a proper IfcWindowType with lining/panel parameters.
    pane_thickness = 0.05
    window = ifcopenshell.api.root.create_entity(model, ifc_class="IfcWindow", name="Window")
    window.OverallWidth = window_dims.width
    window.OverallHeight = window_dims.height
    window_shape = ifcopenshell.api.geometry.add_wall_representation(
        model, context=body, length=window_dims.width, height=window_dims.height, thickness=pane_thickness
    )
    ifcopenshell.api.geometry.assign_representation(model, product=window, representation=window_shape)
    ifcopenshell.api.feature.add_filling(model, opening=opening, element=window)
    ifcopenshell.api.spatial.assign_container(model, products=[window], relating_structure=storey)

    pane_offset = np.array([0.0, (wall_thickness - pane_thickness) / 2, 0.0, 1.0])
    window_matrix = opening_matrix.copy()
    window_matrix[:, 3] = opening_matrix @ pane_offset
    ifcopenshell.api.geometry.edit_object_placement(model, product=window, matrix=window_matrix)
    return opening, window


def find_wall(model: ifcopenshell.file, name: str = TARGET_WALL_NAME) -> ifcopenshell.entity_instance:
    return next(wall for wall in model.by_type("IfcWall") if wall.Name == name)


def main(input_path: Path = BASE_MODEL_PATH, output_path: Path = WINDOW_MODEL_PATH) -> Path:
    model = ifcopenshell.open(str(input_path))
    wall = find_wall(model)
    dims = WindowDimensions()
    opening, window = add_window(model, wall, dims)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))

    print(f"Wrote {output_path}")
    print(f"  wall    {wall.GlobalId}  {wall.Name}")
    print(f"  opening {opening.GlobalId}  {dims.width} m x {dims.height} m, sill at {dims.sill_height} m")
    print(f"  window  {window.GlobalId}  fills opening via IfcRelFillsElement")
    return output_path


if __name__ == "__main__":
    main()
