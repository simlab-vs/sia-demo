# ifc-fem-demo

A teaching demo that connects **BIM modelling in IFC** with a **structural check** and
**3D visualisation**, all in Python. It builds a small house as an IFC4 model, cuts a
window into one wall, turns both versions into a shell-and-frame finite element model with
[PyNite](https://github.com/JWock82/PyNite), and renders geometry and results with VTK.

The point of the demo is the data flow: an IFC file is the single source of truth, the
structural model is derived from it rather than drawn again, and the results are shown on
the real building geometry.

```bash
uv sync                    # create the environment
uv run python main.py      # run all stages end to end (about 15 seconds)
uv run pytest              # run the tests
```

Outputs land in `model/` (IFC files) and `output/` (videos and images).

## What gets built

| Stage | Module | Produces |
|---|---|---|
| 1a. IFC house | `build_model.py` | `model/base_building.ifc` |
| 1b. Window | `add_window.py` | `model/building_with_window.ifc` |
| 2. Structural check | `structural.py` | summary table on stdout |
| 3. Renders | `render_3d.py` | `output/rotation_no_window.mp4`, `output/rotation_with_window.mp4`, `output/stress_comparison.png`, `output/stress_with_window.mp4` |

`pipeline.py` runs them in order; `main.py` is the entry point.

### The house

A 5 m × 4 m concrete box with a pitched timber roof:

- four exterior `IfcWall`s (3 m high, 0.2 m thick) and an interior **shear wall** across
  the short direction, placed off-centre so it stays clear of the window;
- a ground-bearing `IfcSlab` typed `BASESLAB`;
- a **roof**: an `IfcRoof` aggregating two sloped `IfcSlab`s of type `ROOF` (CLT panels,
  pitch 0.75, 0.4 m overhang) resting on the long walls and on a glulam ridge beam
  (`IfcBeam`) that stands on a post (`IfcColumn`) at each gable;
- glazed gable triangles (`IfcCurtainWall`) that carry no load;
- `IfcMaterial`s for concrete, CLT, glulam and glass, assigned to every element;
- the spatial hierarchy Project → Site → Building → Storey with SI units in metres.

Stage 1b picks the south wall and adds an `IfcOpeningElement` (1.2 m × 1.5 m, sill at
0.9 m, centred) related to the wall through `IfcRelVoidsElement`, then an `IfcWindow`
filling it through `IfcRelFillsElement`. The wall's own geometry is untouched; the opening
is a separate product that viewers subtract when rendering. Both files pass ifcopenshell's
schema validation.

### The structural model

Stage 2 reads each IFC file back and solves the whole house as one 3D model in PyNite:

1. **Idealisation.** Walls and roof slabs become their mid-surfaces (the thinnest side of
   each element's local bounding box is its thickness), the beam and posts become their
   axes, and the opening becomes a rectangle in the wall's mid-plane. Wall ends are extended
   or trimmed onto the mid-planes of the walls they meet, the usual centre-line
   idealisation. The `BASESLAB` becomes the fixed support. Material properties are looked
   up by `IfcMaterial` name; non-structural elements are skipped by IFC class.
2. **Meshing.** Every element corner defines a plane `x = c`, `y = c` or `z = c`. Each
   panel is cut wherever such a plane crosses it along a grid line, then refined to about
   0.2 m. That one rule makes neighbouring walls, the roof, the opening and the ridge beam
   share nodes. The solid wall is meshed with the opening's outline too, so the two cases
   differ only through the opening itself.
3. **Elements and loads.** Walls and roof are PyNite quads (membrane plus MITC4 bending);
   the ridge beam is split into members between the roof nodes on the ridge. Loads are
   self-weight, 2.5 kPa of snow on the roof and 1.5 kPa of wind on the south wall, lumped
   onto nodes and combined 1:1:1. The window pane is not modelled, but its share of the
   wind is passed to the opening's edges.
4. **Post-processing.** Von Mises stress per quad at the element centre, on whichever face
   is more stressed (membrane stress ± 6 M / t²). The summary reports peak displacement,
   lateral drift, the peak stress in the house and in the south wall, the stress next to
   the window corners in both models, and an equilibrium check of reactions against
   applied loads.

Typical output:

```
case            quads  bars  max |u| [mm]  lateral [mm]  house vM [MPa]  wall vM [MPa]  corners [MPa]  house peak in
solid            2676    30         0.101         0.050           0.389          0.287          0.125  Roof North
with_opening     2628    30         0.101         0.049           0.389          0.289          0.213  Roof North

Stress at the window corners, with/without opening: 1.70x
```

The opening raises the stress at its corners by about 1.7x while the rest of the house is
unchanged. The house-wide peak sits in the roof panel where it bends over the wall top,
a local bending effect rather than a design value. A sharp re-entrant corner
is a stress singularity in the linear elastic model, so corner values depend on the mesh;
the comparison is qualitative and this is a demo, not a design check.

### The renders

Stage 3 renders off screen through PyVista, the Python layer over VTK:

- **Turntables** of the IFC geometry, with and without the window: the camera orbits once
  around the house's vertical axis (96 frames, 24 fps), with concrete, timber and
  translucent glazing told apart by colour.
- **Stress comparison**: both structural models side by side on their deformed shape,
  same colour scale, same deformation scale, wind arrows on the loaded wall, mesh edges
  visible.
- **Stress turntable** of the deformed house with the window.

Frames are encoded to MP4 by imageio's bundled ffmpeg, so no system install is needed.

## Design decisions worth knowing

- **Geometry is read from IFC through ifcopenshell's geometry engine**, never by parsing
  representation attributes. That is why an arbitrary profile, a rotated placement or an
  opening all work the same way.
- **The gable triangles are glazing, not structure.** The roof therefore spans from eave
  to ridge and needs the ridge beam and posts. This keeps every structural panel a
  rectangle, which is what makes the conforming mesh rule so short.
- **The ridge beam axis sits on the roof mid-surface** so beam, posts and roof share
  nodes without rigid links. In the IFC the beam therefore pokes slightly above the roof.
- **Both cases use the same mesh.** Without that, a re-entrant corner or a wall junction
  gives different numbers for reasons that have nothing to do with the window.
- **Sequential single-hue colour map** for stress, with one shared scale across the two
  cases, so lighter and darker mean the same thing in both panels.

## How it got here

The first version analysed one wall in 2D plane stress with gmsh and scikit-fem and
plotted the field with matplotlib. That answered the original question (how does an
opening change the stress in a wall) but left the rest of the house out. The current
version replaced the solver with PyNite so the whole house, roof and shear wall included,
is analysed as a 3D shell-and-frame model, and replaced matplotlib with VTK so the results
are shown on the real, deformed geometry. The IFC stage kept its structure and gained the
roof, shear wall, beam, posts and glazing.

## Dependencies

| Package | Role |
|---|---|
| `ifcopenshell` | create, edit, validate and read back the IFC model |
| `PyNiteFEA` | 3D finite element analysis: quad shells for walls and roof, frame members for beam and posts |
| `pyvista` | VTK rendering: turntable videos and stress fields on the deformed structure |
| `imageio[ffmpeg]` | encode rendered frames into MP4 (bundles its own ffmpeg binary) |
| `pytest` (dev) | tests |

## Layout

```
main.py                         # uv run python main.py
src/ifc_fem_demo/
  build_model.py                # stage 1a: base IFC house
  add_window.py                 # stage 1b: opening + window
  ifc_geometry.py               # read geometry back from IFC (mid-surfaces, axes, meshes)
  structural.py                 # stage 2: idealise, mesh, solve with PyNite, compare
  render_3d.py                  # stage 3: VTK turntables and stress views
  pipeline.py                   # runs everything and prints a summary table
tests/test_demo.py              # IFC validity/content, structural and render sanity checks
```

The tests build both IFC files in a temporary directory, validate them, check that the
structure read back from IFC is connected, run a coarse analysis to confirm equilibrium
and the corner stress increase, and render short videos and the comparison image.
