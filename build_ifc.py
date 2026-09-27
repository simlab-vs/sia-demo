"""Build an IFC 4.3 (IFC4X3_ADD2) bridge model from build/<campaign>/bridge_model.json."""
import argparse
import json
import math
from pathlib import Path

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.alignment
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.style
import ifcopenshell.api.unit
import ifcopenshell.guid
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from bridge_analysis import ArcAlignment, VerticalProfile
from load_scans import campaigns

SWEEP_STEP = 2.5
PROJECT_NAME = "Pont de la Praz"

MATERIALS = {
    "concrete": ("Béton", "concrete", (0.74, 0.74, 0.71)),
    "barrier": ("Béton (glissière)", "concrete", (0.86, 0.86, 0.83)),
    "asphalt": ("Revêtement bitumineux", "asphalt", (0.22, 0.22, 0.24)),
    "bearing": ("Appareil d'appui", "steel", (0.90, 0.45, 0.10)),
}


class BridgeIfc:
    def __init__(self, model: dict):
        self.m = model
        a, v = model["alignment"], model["vertical"]
        self.align = ArcAlignment(tuple(a["centre"]), a["radius"], a["direction"], a["theta0"], a["length"])
        self.profile = VerticalProfile(tuple(v["coeffs"]), v["rms"], v["is_parabolic"])
        self.grade_height = v["grade_line_height_above_soffit"]
        self.station0 = model["deck"]["s_start"]
        self.f = ifcopenshell.file(schema="IFC4X3_ADD2")
        self.styles = {}
        self._setup_project()
        self._setup_materials()

    # ------------------------------------------------------------------ scaffolding

    def _setup_project(self):
        f = self.f
        self.project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name=PROJECT_NAME)
        units = [ifcopenshell.api.unit.add_si_unit(f, unit_type=t) for t in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT", "PLANEANGLEUNIT")]
        ifcopenshell.api.unit.assign_unit(f, units=units)
        model_ctx = ifcopenshell.api.context.add_context(f, context_type="Model")
        self.body_ctx = ifcopenshell.api.context.add_context(
            f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model_ctx
        )
        ifcopenshell.api.context.add_context(f, context_type="Model", context_identifier="Axis", target_view="GRAPH_VIEW", parent=model_ctx)

        self.site = self._spatial("IfcSite", "Site Pont de la Praz", self.project,
                                  description="Local survey coordinate system of the laser scans (not georeferenced)")
        self.bridge = self._spatial("IfcBridge", PROJECT_NAME, self.site, predefined_type="GIRDER",
                                    description="Continuous single-cell box girder bridge, 9 spans, curved in plan")
        self.superstructure = self._spatial("IfcBridgePart", "Superstructure", self.bridge, predefined_type="SUPERSTRUCTURE")
        self.substructure = self._spatial("IfcBridgePart", "Substructure", self.bridge, predefined_type="SUBSTRUCTURE")

    def _spatial(self, ifc_class, name, parent, predefined_type=None, description=None):
        entity = ifcopenshell.api.root.create_entity(self.f, ifc_class=ifc_class, name=name, predefined_type=predefined_type)
        entity.Description = description
        if ifc_class == "IfcBridgePart":
            entity.UsageType = "VERTICAL" if predefined_type in ("SUPERSTRUCTURE", "SUBSTRUCTURE") else "LONGITUDINAL"
        entity.ObjectPlacement = self._placement()
        ifcopenshell.api.aggregate.assign_object(self.f, products=[entity], relating_object=parent)
        return entity

    def _setup_materials(self):
        self.materials = {}
        for key, (name, category, rgb) in MATERIALS.items():
            material = ifcopenshell.api.material.add_material(self.f, name=name, category=category)
            style = ifcopenshell.api.style.add_style(self.f, name=name)
            ifcopenshell.api.style.add_surface_style(
                self.f, style=style, ifc_class="IfcSurfaceStyleShading",
                attributes={"SurfaceColour": {"Name": None, "Red": rgb[0], "Green": rgb[1], "Blue": rgb[2]}, "Transparency": 0.0},
            )
            ifcopenshell.api.style.assign_material_style(self.f, material=material, style=style, context=self.body_ctx)
            self.materials[key] = (material, style)

    # ------------------------------------------------------------------ geometry helpers

    def _placement(self, origin=(0.0, 0.0, 0.0), x_dir=(1.0, 0.0)):
        f = self.f
        axis = f.createIfcAxis2Placement3D(
            f.createIfcCartesianPoint([float(c) for c in origin]),
            f.createIfcDirection((0.0, 0.0, 1.0)),
            f.createIfcDirection((float(x_dir[0]), float(x_dir[1]), 0.0)),
        )
        return f.createIfcLocalPlacement(None, axis)

    def _frame_at(self, s: float, z: float = 0.0):
        """Local frame on the alignment: X along the tangent, Y to the left, Z up."""
        heading = float(self.align.heading(s))
        xy = self.align.point(s)
        return self._placement((xy[0], xy[1], z), (math.cos(heading), math.sin(heading)))

    def _polyline_profile(self, pts, name=None):
        f = self.f
        pts = [[float(x), float(y)] for x, y in pts]
        curve = f.createIfcIndexedPolyCurve(f.createIfcCartesianPointList2D(pts + [pts[0]]), None, False)
        return f.createIfcArbitraryClosedProfileDef("AREA", name, curve)

    def _rect_profile(self, x0, x1, y0, y1):
        return self._polyline_profile([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def _extrusion(self, profile, depth, z0=0.0):
        f = self.f
        position = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, float(z0))), None, None)
        return f.createIfcExtrudedAreaSolid(profile, position, f.createIfcDirection((0.0, 0.0, 1.0)), float(depth))

    def _shape(self, items, rep_type):
        rep = self.f.createIfcShapeRepresentation(self.body_ctx, "Body", rep_type, items)
        return self.f.createIfcProductDefinitionShape(None, None, [rep])

    def _sweep_mesh(self, polygon: dict, s0: float, s1: float):
        """Triangulated solid of a vertical cross-section (o, h above soffit) swept along the alignment."""
        poly = orient(Polygon(polygon["exterior"], polygon["interiors"]), 1.0)
        rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
        stations = np.linspace(s0, s1, max(2, math.ceil((s1 - s0) / SWEEP_STEP) + 1))
        ring_pts = np.concatenate(rings)
        n_ring = len(ring_pts)

        coords = []
        for s in stations:
            xy = self.align.point(np.full(n_ring, s), ring_pts[:, 0])
            z = self.profile.soffit(s) + ring_pts[:, 1]
            coords.append(np.c_[xy, z])
        coords = np.concatenate(coords)

        faces = []
        offset = 0
        for ring in rings:
            n = len(ring)
            for k in range(len(stations) - 1):
                base, nxt = k * n_ring + offset, (k + 1) * n_ring + offset
                for i in range(n):
                    j = (i + 1) % n
                    faces.append((base + i, base + j, nxt + j))
                    faces.append((base + i, nxt + j, nxt + i))
            offset += n

        index = {tuple(np.round(p, 6)): i for i, p in enumerate(ring_pts)}
        last = (len(stations) - 1) * n_ring
        for tri in shapely.constrained_delaunay_triangles(poly).geoms:
            t = orient(tri, 1.0)
            ids = [index[tuple(np.round(p, 6))] for p in np.asarray(t.exterior.coords)[:3]]
            faces.append((ids[2], ids[1], ids[0]))
            faces.append((last + ids[0], last + ids[1], last + ids[2]))

        f = self.f
        point_list = f.createIfcCartesianPointList3D(coords.round(5).tolist())
        index_list = [[a + 1, b + 1, c + 1] for a, b, c in faces]
        return f.createIfcTriangulatedFaceSet(point_list, None, True, index_list, None)

    # ------------------------------------------------------------------ element helpers

    def _element(self, ifc_class, name, container, predefined_type=None, material="concrete", description=None, object_type=None):
        element = ifcopenshell.api.root.create_entity(self.f, ifc_class=ifc_class, name=name, predefined_type=predefined_type)
        element.Description = description
        if object_type:
            element.ObjectType = object_type
        ifcopenshell.api.spatial.assign_container(self.f, products=[element], relating_structure=container)
        mat, style = self.materials[material]
        ifcopenshell.api.material.assign_material(self.f, products=[element], material=mat)
        self.styles[element.id()] = style
        return element

    def _set_shape(self, element, placement, shape):
        element.ObjectPlacement = placement
        element.Representation = shape
        for rep in shape.Representations:
            ifcopenshell.api.style.assign_representation_styles(self.f, shape_representation=rep, styles=[self.styles[element.id()]])

    def _psets(self, element, survey: dict, dims: dict | None = None, qto: tuple[str, dict] | None = None):
        f = self.f
        pset = ifcopenshell.api.pset.add_pset(f, product=element, name="PTB_ScanSurvey")
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties={"Campaign": self.m["campaign"], **survey})
        if dims:
            pset = ifcopenshell.api.pset.add_pset(f, product=element, name="PTB_MeasuredDimensions")
            ifcopenshell.api.pset.edit_pset(f, pset=pset, properties={k: round(float(v), 4) for k, v in dims.items()})
        if qto:
            q = ifcopenshell.api.pset.add_qto(f, product=element, name=qto[0])
            ifcopenshell.api.pset.edit_qto(f, qto=q, properties={k: round(float(v), 4) for k, v in qto[1].items()})

    # ------------------------------------------------------------------ alignment

    def build_alignment(self):
        f, al, c = self.f, self.align, self.profile.coeffs
        alignment = ifcopenshell.api.alignment.create(f, "Axe Pont de la Praz", include_vertical=True, start_station=-self.station0)
        horizontal = ifcopenshell.api.alignment.get_horizontal_layout(alignment)
        vertical = ifcopenshell.api.alignment.get_vertical_layout(alignment)
        x0, y0 = al.point(0.0)
        ifcopenshell.api.alignment.create_layout_segment(f, horizontal, f.createIfcAlignmentHorizontalSegment(
            StartPoint=f.createIfcCartesianPoint((float(x0), float(y0))),
            StartDirection=float(al.heading(0.0)),
            StartRadiusOfCurvature=al.direction * al.radius,
            EndRadiusOfCurvature=al.direction * al.radius,
            SegmentLength=al.length,
            PredefinedType="CIRCULARARC",
        ))
        g0, g1 = c[1], c[1] + 2 * c[2] * al.length
        ifcopenshell.api.alignment.create_layout_segment(f, vertical, f.createIfcAlignmentVerticalSegment(
            StartDistAlong=0.0,
            HorizontalLength=al.length,
            StartHeight=c[0] + self.grade_height,
            StartGradient=g0,
            EndGradient=g1,
            RadiusOfCurvature=1 / (2 * c[2]) if self.profile.is_parabolic else None,
            PredefinedType="PARABOLICARC" if self.profile.is_parabolic else "CONSTANTGRADIENT",
        ))
        alignment.Description = (
            f"Fitted to the deck edges: circular arc R={al.radius:.1f} m (rms {self.m['alignment']['stats']['centreline_rms'] * 1000:.0f} mm); "
            f"grade line = top of pavement at axis, {self.grade_height:.3f} m above the box soffit"
        )
        ifcopenshell.api.spatial.reference_structure(f, products=[alignment], relating_structure=self.bridge)
        self.alignment = alignment

    def chainage(self, s: float) -> float:
        """Station value with 0+000 at the Est end of the deck."""
        return s - self.station0

    def add_referent(self, station: float, name: str, product):
        ifcopenshell.api.alignment.add_stationing_referent(
            self.f, alignment=self.alignment, distance_along=station, station=self.chainage(station), name=name, positioned_product=product
        )

    # ------------------------------------------------------------------ superstructure

    def build_deck(self):
        deck, section = self.m["deck"], self.m["deck"]["section"]
        meas = section["measured"]
        girder_area = section["girder"]["area"]
        for span in deck["spans"]:
            beam = self._element(
                "IfcBeam", f"Caisson travée {span['name']}", self.superstructure, "GIRDER_SEGMENT",
                description="Single-cell prestressed box girder segment between support axes (continuous girder)",
            )
            self._set_shape(beam, self._placement(), self._shape([self._sweep_mesh(section["girder"], span["s_start"], span["s_end"])], "Tessellation"))
            self._psets(
                beam,
                {"SourceScans": f"Tab_{span['name'].replace('-', '_')}", "GeometrySource": "outer shape measured; internal void assumed",
                 "SectionsStacked": deck["sections_stacked"]},
                {"StationStart": self.chainage(span["s_start"]), "StationEnd": self.chainage(span["s_end"]), "SpanLength": span["length"],
                 "OverallWidth": meas["overall_width"], "DepthAtAxis": meas["structural_depth_at_axis"], "SoffitWidth": meas["soffit_width"]},
                ("Qto_BeamBaseQuantities", {"Length": span["length"], "CrossSectionArea": girder_area, "NetVolume": girder_area * span["length"]}),
            )

        s0 = self.m["abutments"]["Est"]["front_wall"]["s_face"]
        s1 = self.m["abutments"]["Ouest"]["body"]["s_back"]
        for key, name, height_key in (
            ("barrier_left", "Glissière béton gauche", "barrier_left_height"),
            ("barrier_right", "Glissière béton droite", "barrier_right_height"),
            ("barrier_median", "Glissière béton centrale", "median_barrier_height"),
        ):
            rail = self._element("IfcRailing", name, self.superstructure, "GUARDRAIL", material="barrier",
                                 description="Concrete safety barrier (New Jersey profile)")
            self._set_shape(rail, self._placement(), self._shape([self._sweep_mesh(section[key], s0, s1)], "Tessellation"))
            self._psets(rail, {"SourceScans": "Tab_*", "GeometrySource": "measured"},
                        {"HeightAboveRoad": meas[height_key], "CrossSectionArea": section[key]["area"]},
                        ("Qto_RailingBaseQuantities", {"Length": s1 - s0}))

        for key, name in (("pavement_left", "Revêtement chaussée gauche"), ("pavement_right", "Revêtement chaussée droite")):
            course = self._element("IfcCourse", name, self.superstructure, "PAVEMENT", material="asphalt",
                                   description="Wearing course incl. waterproofing; top surface measured, thickness assumed")
            self._set_shape(course, self._placement(), self._shape([self._sweep_mesh(section[key], s0, s1)], "Tessellation"))
            side = key.split("_")[1]
            self._psets(course, {"SourceScans": "Tab_*", "GeometrySource": "top measured; thickness assumed"},
                        {"Crossfall": meas[f"crossfall_{side}"], "AssumedThickness": self.m["assumptions"]["pavement_thickness"]},
                        ("Qto_CourseBaseQuantities", {"Length": s1 - s0, "Thickness": self.m["assumptions"]["pavement_thickness"],
                                                      "GrossVolume": section[key]["area"] * (s1 - s0)}))

    # ------------------------------------------------------------------ substructure

    def _standard_bearing(self):
        measured = [b for p in self.m["piers"] for b in p["bearings"] if b["size_a"] < 1.8]
        return {
            "a": float(np.median([abs(b["a"]) for b in measured])),
            "size_a": float(np.median([b["size_a"] for b in measured])),
            "size_b": float(np.median([b["size_b"] for b in measured])),
        }

    def _bearing(self, name, container, placement, a, b, size_a, size_b, z0, z1, survey):
        bearing = self._element("IfcBearing", name, container, "NOTDEFINED", material="bearing",
                                description="Bearing incl. plinth (lumped); type not identifiable from the scan")
        profile = self._rect_profile(a - size_a / 2, a + size_a / 2, b - size_b / 2, b + size_b / 2)
        self._set_shape(bearing, placement, self._shape([self._extrusion(profile, z1 - z0, z0)], "SweptSolid"))
        self._psets(bearing, survey, {"SizeLongAxis": size_a, "SizeTransverse": size_b, "Height": z1 - z0})
        return bearing

    def build_piers(self):
        std = self._standard_bearing()
        for p in self.m["piers"]:
            part = self._spatial("IfcBridgePart", f"Pile {p['name']}", self.substructure, predefined_type="PIER")
            column = self._element("IfcColumn", f"Fût pile {p['name']}", part, "PIERSTEM",
                                   description="Wall pier, octagonal section (rectangle with chamfered corners)")
            w, t, c = p["width"] / 2, p["thickness"] / 2, p["chamfer"]
            octagon = [(-w + c, -t), (w - c, -t), (w, -t + c), (w, t - c), (w - c, t), (-w + c, t), (-w, t - c), (-w, -t + c)]
            placement = self._placement((*p["centre"], p["z_bottom"]), p["long_axis"])
            profile = self._polyline_profile(octagon, name=f"Octagon {p['width']:.2f}x{p['thickness']:.2f}")
            self._set_shape(column, placement, self._shape([self._extrusion(profile, p["height"])], "SweptSolid"))
            area = shapely.Polygon(octagon).area
            self._psets(
                column,
                {"SourceScans": f"Pil_{p['name']}", "PointCount": p["points"], "GeometrySource": "measured (foundation not visible)",
                 "SlicesUsed": p["slices_used"]},
                {"Station": self.chainage(p["station"]), "OffsetFromAxis": p["offset"], "SkewToRadialDeg": p["skew_to_radial_deg"],
                 "Width": p["width"], "Thickness": p["thickness"], "Chamfer": p["chamfer"], "Height": p["height"],
                 "BottomElevation": p["z_bottom"], "TopElevation": p["z_top"]},
                ("Qto_ColumnBaseQuantities", {"Length": p["height"], "CrossSectionArea": area, "NetVolume": area * p["height"]}),
            )
            top_frame = self._placement((*p["centre"], 0.0), p["long_axis"])
            heading = float(self.align.heading(p["station"]))
            long_axis_to_left = np.sign(np.dot(p["long_axis"], (-math.sin(heading), math.cos(heading))))
            for side in (-1, 1):
                measured = next((b for b in p["bearings"] if np.sign(b["a"]) == side and b["size_a"] < 1.8), None)
                a = measured["a"] if measured else side * std["a"]
                b = measured["b"] if measured else 0.0
                self._bearing(f"Appui pile {p['name']} {'gauche' if side * long_axis_to_left > 0 else 'droite'}", part, top_frame, a, b,
                              std["size_a"], std["size_b"], p["z_top"], p["soffit_above"],
                              {"SourceScans": f"Pil_{p['name']}", "GeometrySource": "position measured; size = median over piers" if measured else "assumed"})
            self.add_referent(p["station"], f"Axe pile {p['name']}", part)

    def build_abutment_est(self):
        ab = self.m["abutments"]["Est"]
        A = self.m["assumptions"]
        part = self._spatial("IfcBridgePart", "Culée Est", self.substructure, predefined_type="ABUTMENT")
        fw = ab["front_wall"]
        frame = self._frame_at(fw["s_face"])
        survey = {"SourceScans": "Cul_Est, Tab_Est_A", "PointCount": ab["points"]}

        wall = self._element("IfcWall", "Mur de front culée Est", part, "SOLIDWALL")
        profile = self._rect_profile(-A["abutment_wall_thickness"], 0.0, fw["o_min"], fw["o_max"])
        self._set_shape(wall, frame, self._shape([self._extrusion(profile, fw["z_top"] - fw["z_bottom"], fw["z_bottom"])], "SweptSolid"))
        self._psets(wall, survey | {"GeometrySource": "front face, width and height measured; thickness assumed"},
                    {"Station": self.chainage(fw["s_face"]), "Width": fw["o_max"] - fw["o_min"], "Height": fw["z_top"] - fw["z_bottom"]})

        for side, wing in ab["wing_walls"].items():
            t = A["wing_wall_thickness"]
            o0, o1 = (wing["o_outer"] - t, wing["o_outer"]) if side == "left" else (wing["o_outer"], wing["o_outer"] + t)
            ww = self._element("IfcWall", f"Mur en aile culée Est {'gauche' if side == 'left' else 'droit'}", part, "RETAININGWALL")
            profile = self._rect_profile(wing["s_start"] - fw["s_face"], -A["abutment_wall_thickness"], o0, o1)
            self._set_shape(ww, frame, self._shape([self._extrusion(profile, wing["z_top"] - wing["z_bottom"], wing["z_bottom"])], "SweptSolid"))
            self._psets(ww, survey | {"GeometrySource": "outer face and visible length measured; thickness assumed"},
                        {"VisibleLength": wing["s_end"] - wing["s_start"]})

        cs = ab["cover_slab"]
        slab = self._element("IfcSlab", "Dalle de couverture culée Est", part, "USERDEFINED", object_type="ABUTMENT_COVER_SLAB",
                             description="Slab spanning between the abutment wall and the deck end")
        top_o, top_h = np.array(self.m["deck"]["section"]["slab_top_polyline"]).T
        inner = (top_o > cs["o_min"]) & (top_o < cs["o_max"])
        o_top = np.r_[cs["o_min"], top_o[inner], cs["o_max"]]
        h_top = np.interp(o_top, top_o, top_h)
        outline = list(zip(o_top, h_top - cs["thickness"])) + list(zip(o_top[::-1], h_top[::-1]))
        outline = np.asarray(orient(Polygon(outline), 1.0).exterior.coords)[:-1].tolist()
        self._set_shape(slab, self._placement(), self._shape([self._sweep_mesh({"exterior": outline, "interiors": []}, cs["s_start"], cs["s_end"])], "Tessellation"))
        self._psets(slab, survey | {"GeometrySource": "thickness and extent measured; parallel to the pavement"},
                    {"Length": cs["s_end"] - cs["s_start"], "Thickness": cs["thickness"]})

        sb = ab["support_block"]
        block = self._element("IfcWall", "Sommier d'appui culée Est", part, "USERDEFINED", object_type="BEARING_SEAT_BLOCK",
                              description="Support block under the deck end (internal chambers simplified as solid)")
        profile = self._rect_profile(sb["s_start"] - fw["s_face"], sb["s_end"] - fw["s_face"], sb["o_min"], sb["o_max"])
        self._set_shape(block, frame, self._shape([self._extrusion(profile, sb["z_top"] - sb["z_bottom"], sb["z_bottom"])], "SweptSolid"))
        self._psets(block, survey | {"GeometrySource": "outline measured; simplified solid"},
                    {"SeatElevation": sb["z_top"], "GroundElevation": sb["z_bottom"]})

        std = self._standard_bearing()
        s_bearing = sb["s_end"] - A["abutment_bearing_setback"] - std["size_b"] / 2
        bearing_frame = self._frame_at(s_bearing)
        for side in (-1, 1):
            self._bearing(f"Appui culée Est {'gauche' if side > 0 else 'droite'}", part, bearing_frame, 0.0, side * std["a"],
                          std["size_b"], std["size_a"], sb["z_top"], float(self.profile.soffit(s_bearing)),
                          {"SourceScans": "Cul_Est", "GeometrySource": "seat level measured; position and size assumed"})
        self.add_referent(self.m["deck"]["s_start"], "Extrémité tablier Est", part)

    def build_abutment_ouest(self):
        ab = self.m["abutments"]["Ouest"]
        part = self._spatial("IfcBridgePart", "Culée Ouest", self.substructure, predefined_type="ABUTMENT")
        body, gal = ab["body"], ab["gallery"]
        s_front = max(body["s_front"], self.m["deck"]["s_end"])
        frame = self._frame_at(s_front)
        f = self.f
        solid = self._extrusion(self._rect_profile(0.0, body["s_back"] - s_front, body["o_min"], body["o_max"]),
                                body["z_top"] - body["z_bottom"], body["z_bottom"])
        void = self._extrusion(self._rect_profile(gal["s_start"] - s_front, gal["s_end"] - s_front, gal["o_min"], gal["o_max"]),
                               gal["z_ceiling"] - gal["z_floor"], gal["z_floor"])
        wall = self._element("IfcWall", "Corps de culée Ouest", part, "SOLIDWALL",
                             description="Abutment body with inspection gallery behind the deck end")
        self._set_shape(wall, frame, self._shape([f.createIfcBooleanResult("DIFFERENCE", solid, void)], "CSG"))
        self._psets(wall, {"SourceScans": "Cul_Ouest, Tab_H_Ouest", "PointCount": ab["points"],
                           "GeometrySource": "front face, width, gallery measured; back wall thickness assumed"},
                    {"Station": self.chainage(s_front), "Width": body["o_max"] - body["o_min"], "GalleryWidth": gal["s_end"] - gal["s_start"],
                     "GalleryHeight": gal["z_ceiling"] - gal["z_floor"], "GalleryFloorElevation": gal["z_floor"],
                     "GroundElevation": body["z_bottom"]})
        self.add_referent(self.m["deck"]["s_end"], "Extrémité tablier Ouest", part)

    # ------------------------------------------------------------------ assembly

    def build(self) -> ifcopenshell.file:
        self.build_alignment()
        self.build_deck()
        self.build_abutment_est()
        self.build_piers()
        self.build_abutment_ouest()
        return self.f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=campaigns()[-1], choices=campaigns())
    args = parser.parse_args()
    model = json.loads((Path("build") / args.campaign / "bridge_model.json").read_text())
    ifc = BridgeIfc(model).build()
    out = Path("build") / args.campaign / "pont_de_la_praz.ifc"
    ifc.write(str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
