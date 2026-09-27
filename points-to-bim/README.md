# Pont de la Praz: point cloud to IFC 4.3

Turns the per-element terrestrial laser scans in `data/` (E57, 5 cm resampled) into a
parametric IFC 4.3 (`IFC4X3_ADD2`) bridge model with an alignment, and checks the model
against the scans.

## Pipeline

```sh
uv run python load_scans.py                              # E57 -> build/cache/<campaign>/*.npz
uv run python bridge_analysis.py --campaign 2022-10-02   # measurements -> build/<campaign>/bridge_model.json
uv run python build_ifc.py       --campaign 2022-10-02   # -> build/<campaign>/pont_de_la_praz.ifc
uv run python validate_ifc.py    --campaign 2022-10-02   # schema, geometry, scan-to-model deviations, previews
uv run python compare_campaigns.py                       # 2022-03-01 vs 2022-10-02
uv run python georeference.py    --campaign 2022-10-02   # local frame -> LV95/LN02, fetches swisstopo tiles
uv run python render_3d.py       --campaign 2022-10-02   # VTK orbit videos -> build/<campaign>/viz/
```

`--campaign` defaults to the latest campaign.

## What is measured vs assumed

| Part | Measured from the scans | Assumed (not visible to a TLS) |
|---|---|---|
| Alignment | circular arc (R ≈ 3.96 km), parabolic grade (~1.1 %) fitted to deck edges and soffit | – |
| Box girder | full outer section (stacked from ~290 sections), soffit, webs, cantilevers, edge beams | inner void: slabs 0.30/0.22 m, webs 0.45 m, haunches |
| Barriers, pavement | New Jersey barrier profiles, crossfalls, road surface | pavement thickness 0.10 m |
| Piers | octagonal section, orientation, base and top elevations | foundations (buried) |
| Bearings | position between pier top and soffit | size (median over piers), type |
| Culée Est | wall face, width, ground, cover slab thickness, support block, visible wing walls | wall and wing wall thicknesses |
| Culée Ouest | front face, width, gallery (floor, ceiling, extent) | back wall thickness |

Every IFC element carries `PTB_ScanSurvey` (source scans, `GeometrySource`) and
`PTB_MeasuredDimensions` property sets, plus standard `Qto_*` quantities where applicable.
Stationing starts at 0+000 at the Est deck end.

Coordinates are the local survey frame of the E57 files (not georeferenced; no `IfcMapConversion`).

## Georeferencing and terrain

The bridge is the A9 viaduct at Ballaigues (swissTLM3D road "Pont de la Praz"). `georeference.py`
estimates a 4-parameter transform (plan rotation and translation, height offset) from the local
frame to LV95 / LN02. It starts from the gazetteer road extent, then fits the scanned top surface
to swissSURFACE3D (which keeps the deck) and the scanned ground around the pier bases to
swissALTI3D. The deck constrains the position across the bridge; the valley floor, which drops
~15 m from pier A to D, constrains it along the axis. Result: rotation −170.8°, height offset
+785.7 m, residuals ~12 cm (median, surface model) and ~15 cm (terrain at the pier bases).

`render_3d.py` renders, with VTK (via PyVista), a 12 s orbit per variant: `bim`, `bim_scan` (the scans in
true colour over the translucent model, with the model edges in orange), and the same two over
a 600 m disc of swissALTI3D terrain (0.5 m tiles, shown at 1 m, 5 m contours) moved into the local frame.
Abutment measurements pool both campaigns, which share the same frame, because each campaign
only covers part of those hidden zones.
