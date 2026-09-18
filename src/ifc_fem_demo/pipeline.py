"""Run the whole demo: build the IFC house, cut the window, analyse both, render, run the seismic scenario and its static load."""

from __future__ import annotations

import time

from ifc_fem_demo import add_window, build_model, render_3d, render_spectra, seismic, structural


def main(scenario: seismic.Scenario = seismic.Scenario()) -> None:
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

    print("\n== Stage 4: seismic scenario against SIA 261 ==")
    spectra, responses = seismic.main(scenario, window_path)
    render_spectra.render_spectra(spectra, responses, output / "seismic_spectra.png")

    print("\n== Stage 5: equivalent static lateral load on the PyNite model ==")
    loads = seismic.lateral_loads(responses)
    seismic_results = structural.analyse_lateral(window_path, loads)
    render_3d.render_seismic_comparison(seismic_results, loads, output / "seismic_stress_comparison.png")
    render_3d.render_stress_rotation(seismic.governing_case(seismic_results), output / "seismic_stress.mp4")

    print("\n== Summary ==")
    print(structural.summary_table(results))
    print("\n" + seismic.spectra_table(spectra) + "\n\n" + seismic.house_table(responses))
    print("\n" + structural.lateral_table(seismic_results, loads))
    print(f"\nTotal time: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
