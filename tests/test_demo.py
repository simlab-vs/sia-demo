from pathlib import Path

import ifcopenshell
import ifcopenshell.validate
import imageio.v2
import numpy as np
import pytest

from ifc_fem_demo import add_window, render_3d, structural

COARSE_MESH = 0.4  # keeps the structural tests to a few seconds


@pytest.fixture(scope="session")
def results(model_paths: tuple[Path, Path]) -> dict[str, structural.AnalysisResult]:
    return structural.main(*model_paths, mesh_size=COARSE_MESH)


def assert_valid_ifc(path: Path) -> ifcopenshell.file:
    model = ifcopenshell.open(str(path))
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger, express_rules=True)
    assert logger.statements == [], logger.statements
    return model


def test_base_model_contains_expected_entities(model_paths: tuple[Path, Path]) -> None:
    model = assert_valid_ifc(model_paths[0])
    assert model.schema == "IFC4"
    assert {u.Name for u in model.by_type("IfcSIUnit")} >= {"METRE"}
    assert {w.Name for w in model.by_type("IfcWall")} == {"Wall South", "Wall North", "Wall East", "Wall West", "Shear Wall"}
    assert sorted(s.PredefinedType for s in model.by_type("IfcSlab")) == ["BASESLAB", "ROOF", "ROOF"]
    assert len(model.by_type("IfcRoof")) == 1
    assert len(model.by_type("IfcBeam")) == 1
    assert len(model.by_type("IfcColumn")) == 2
    assert len(model.by_type("IfcCurtainWall")) == 2
    assert model.by_type("IfcOpeningElement") == []


def test_window_model_contains_opening_and_window(model_paths: tuple[Path, Path]) -> None:
    model = assert_valid_ifc(model_paths[1])
    assert len(model.by_type("IfcWall")) == 5
    assert len(model.by_type("IfcOpeningElement")) == 1
    assert len(model.by_type("IfcWindow")) == 1

    voids = model.by_type("IfcRelVoidsElement")
    fills = model.by_type("IfcRelFillsElement")
    assert len(voids) == 1 and len(fills) == 1
    assert voids[0].RelatingBuildingElement.Name == add_window.TARGET_WALL_NAME
    assert fills[0].RelatingOpeningElement == voids[0].RelatedOpeningElement


def test_structure_read_from_ifc_is_connected(model_paths: tuple[Path, Path]) -> None:
    structure = structural.structure_from_ifc(ifcopenshell.open(str(model_paths[1])))
    panels = {panel.name: panel for panel in structure.panels}
    assert set(panels) == {"Wall South", "Wall North", "Wall East", "Wall West", "Shear Wall", "Roof South", "Roof North"}
    assert {bar.name for bar in structure.bars} == {"Ridge Beam", "Post West", "Post East"}
    assert len(panels["Wall South"].openings) == 1

    # Walls are joined on their centre lines: the south wall now runs between
    # the west and east wall mid-planes, and the shear wall reaches the south wall.
    south, west, shear = panels["Wall South"].surface, panels["Wall West"].surface, panels["Shear Wall"].surface
    assert south.origin[0] == pytest.approx(west.origin[0])
    assert shear.origin[1] == pytest.approx(south.origin[1])


def test_roof_shares_every_wall_top_node(results: dict[str, structural.AnalysisResult]) -> None:
    mesh = results["with_opening"].mesh
    panels = np.array(mesh.quad_panels)
    for wall, roof in (("Wall South", "Roof South"), ("Wall North", "Roof North")):
        wall_nodes = set(mesh.quads[panels == wall].ravel())
        roof_nodes = set(mesh.quads[panels == roof].ravel())
        top_of_wall = {i for i in wall_nodes if mesh.points[i, 2] == pytest.approx(mesh.points[list(wall_nodes), 2].max())}
        assert top_of_wall <= roof_nodes, f"{wall} has top nodes the {roof} does not share"


def test_opening_raises_stress_at_its_corners(results: dict[str, structural.AnalysisResult]) -> None:
    solid, opened = results["solid"], results["with_opening"]
    for result in (solid, opened):
        assert result.reaction_imbalance < 1e-8, "reactions do not balance the applied loads"
        assert result.max_displacement > 0
        assert result.n_members > 0

    assert opened.n_quads < solid.n_quads
    assert opened.corner_von_mises > 1.2 * solid.corner_von_mises
    # Away from the window the two models are the same structure on the same mesh.
    assert opened.max_von_mises == pytest.approx(solid.max_von_mises, rel=0.05)


def test_renders_produce_playable_video_and_images(
    model_paths: tuple[Path, Path], results: dict[str, structural.AnalysisResult], tmp_path: Path
) -> None:
    video_path = render_3d.render_rotation(model_paths[1], tmp_path / "rotation.mp4", n_frames=6, fps=6)
    stress_video = render_3d.render_stress_rotation(results["with_opening"], tmp_path / "stress.mp4", n_frames=6, fps=6)
    image_path = render_3d.render_stress_comparison(results, tmp_path / "comparison.png")

    for path in (video_path, stress_video):
        assert imageio.v2.get_reader(str(path)).count_frames() == 6
    assert image_path.stat().st_size > 0
