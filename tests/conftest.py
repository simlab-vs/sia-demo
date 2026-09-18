from pathlib import Path

import pytest

from ifc_fem_demo import add_window, build_model


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    model_dir = tmp_path_factory.mktemp("model")
    base_path = build_model.main(model_dir / "base_building.ifc")
    window_path = add_window.main(base_path, model_dir / "building_with_window.ifc")
    return base_path, window_path
