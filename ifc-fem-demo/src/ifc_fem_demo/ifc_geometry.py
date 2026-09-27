"""Small helpers to read geometry back out of an IFC model.

Rather than parsing IfcExtrudedAreaSolid attributes by hand, we let
ifcopenshell's geometry engine triangulate the element and work with the
resulting vertices. That is robust to how the representation was authored.
"""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import numpy as np

_LOCAL = ifcopenshell.geom.settings()
_WORLD = ifcopenshell.geom.settings()
_WORLD.set("use-world-coords", True)


def local_vertices(element: ifcopenshell.entity_instance) -> np.ndarray:
    """Triangulated vertices (n, 3) in the element's own placement frame."""
    shape = ifcopenshell.geom.create_shape(_LOCAL, element)
    return np.array(shape.geometry.verts).reshape(-1, 3)


def world_vertices(element: ifcopenshell.entity_instance) -> np.ndarray:
    """Triangulated vertices (n, 3) in project (world) coordinates."""
    shape = ifcopenshell.geom.create_shape(_WORLD, element)
    return np.array(shape.geometry.verts).reshape(-1, 3)


def local_extents(element: ifcopenshell.entity_instance) -> tuple[float, float, float]:
    """(extent_x, extent_y, extent_z) of the element in its own frame.

    For a wall built by ifcopenshell.api this is (length, thickness, height).
    """
    vertices = local_vertices(element)
    return tuple(float(v) for v in vertices.max(axis=0) - vertices.min(axis=0))


def vertices_in_frame_of(
    element: ifcopenshell.entity_instance, frame_element: ifcopenshell.entity_instance
) -> np.ndarray:
    """Vertices of `element` expressed in the placement frame of `frame_element`."""
    frame = ifcopenshell.util.placement.get_local_placement(frame_element.ObjectPlacement)
    world = world_vertices(element)
    homogeneous = np.column_stack([world, np.ones(len(world))])
    return (np.linalg.inv(frame) @ homogeneous.T).T[:, :3]


def world_mesh(element: ifcopenshell.entity_instance) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, triangle_faces) for `element` in world coordinates.

    `faces` indexes rows of `vertices`, ready for a Poly3DCollection:
    `vertices[faces]` gives one (n_triangles, 3, 3) array of triangle corners.
    """
    shape = ifcopenshell.geom.create_shape(_WORLD, element)
    vertices = np.array(shape.geometry.verts).reshape(-1, 3)
    faces = np.array(shape.geometry.faces).reshape(-1, 3)
    return vertices, faces


def to_world(element: ifcopenshell.entity_instance, local_points: np.ndarray) -> np.ndarray:
    """Map (n, 3) points from the element's placement frame to world coordinates."""
    frame = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
    homogeneous = np.column_stack([local_points, np.ones(len(local_points))])
    return (frame @ homogeneous.T).T[:, :3]


def mid_surface(element: ifcopenshell.entity_instance) -> tuple[np.ndarray, float]:
    """World corners (4, 3) of a plate-like element's mid-surface, and its thickness.

    Every plate-like element in this demo (walls, slabs, openings) is a box in
    its own frame, so the thinnest side of its local bounding box is the
    thickness and the other two sides span the mid-surface. The corners go
    around the rectangle, so corner 1 - corner 0 and corner 3 - corner 0 are
    its two edges.
    """
    vertices = local_vertices(element)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = high - low
    thin = int(np.argmin(extent))
    first, second = (axis for axis in range(3) if axis != thin)

    origin = low.copy()
    origin[thin] = (low[thin] + high[thin]) / 2
    first_edge, second_edge = np.zeros(3), np.zeros(3)
    first_edge[first], second_edge[second] = extent[first], extent[second]
    corners = np.array([origin, origin + first_edge, origin + first_edge + second_edge, origin + second_edge])
    return to_world(element, corners), float(extent[thin])


def axis_segment(element: ifcopenshell.entity_instance) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """World start and end of a bar-like element's axis, plus its section (width, depth).

    The longest side of the local bounding box is the axis; the axis passes
    through the centre of the section.
    """
    vertices = local_vertices(element)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = high - low
    long = int(np.argmax(extent))

    centre = (low + high) / 2
    start, end = centre.copy(), centre.copy()
    start[long], end[long] = low[long], high[long]
    section = tuple(float(extent[axis]) for axis in range(3) if axis != long)
    start_world, end_world = to_world(element, np.array([start, end]))
    return start_world, end_world, section
