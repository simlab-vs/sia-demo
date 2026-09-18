"""Run the whole demo: build the IFC house, cut the window, analyse both, render."""

from __future__ import annotations

import time

from ifc_fem_demo import add_window, build_model, render_3d, structural


def main() -> None:
    started = time.perf_counter()

    print("== Stage 1a: build base IFC model ==")
    base_path = build_model.main()

    print("\n== Stage 1b: add window opening ==")
    window_path = add_window.main(base_path)

    print("\n== Stage 2: PyNite shell-and-frame analysis, with and without the window ==")
    results = structural.main(base_path, window_path)

    print("\n== Stage 3: VTK renders ==")
    output = render_3d.OUTPUT_DIR
    render_3d.render_rotation(base_path, output / "rotation_no_window.mp4")
    render_3d.render_rotation(window_path, output / "rotation_with_window.mp4")
    render_3d.render_stress_comparison(results, output / "stress_comparison.png")
    render_3d.render_stress_rotation(results["with_opening"], output / "stress_with_window.mp4")

    print("\n== Summary ==")
    print(structural.summary_table(results))
    print(f"\nTotal time: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
