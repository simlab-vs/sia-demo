from pathlib import Path

import numpy as np
import pytest

from ifc_fem_demo import ground_motion, render_spectra, seismic, sia261, structural

COARSE_MESH = 0.4

# Spot values from the verification tables shipped with OpenQuake hazardlib
# (tests/gsim/data/AKKAR14/AKKAR_2014_RHYPO_*.csv), which were generated with
# the authors' own code: (Mw, rake, R_hyp km, Vs30) -> SA in g at PGA, 0.2, 1.0, 4.0 s.
AUTHORS_MEDIANS = {
    (5.0, 0.0, 10.0, 300.0): (1.40241636e-01, 3.13151453e-01, 3.77078197e-02, 2.17561438e-03),
    (5.0, 0.0, 0.0, 570.0): (3.23358939e-01, 6.84660812e-01, 4.59669369e-02, 2.62311614e-03),
    (6.0, 90.0, 50.0, 800.0): (2.96509858e-02, 5.31626192e-02, 1.54002273e-02, 1.85413241e-03),
    (7.0, -90.0, 20.0, 180.0): (2.31835082e-01, 5.20104880e-01, 3.55537356e-01, 7.47989839e-02),
    (8.0, 0.0, 100.0, 1100.0): (5.71650866e-02, 1.02154670e-01, 7.59473530e-02, 2.45055308e-02),
}
AUTHORS_TOTAL_SIGMA = (7.34713611e-01, 7.94765500e-01, 7.99667181e-01, 7.17702027e-01)
CHECK_PERIODS = (0.0, 0.2, 1.0, 4.0)


def _at(prediction: ground_motion.Prediction, period: float) -> tuple[float, float]:
    index = int(np.flatnonzero(np.isclose(prediction.periods, period))[0])
    return float(prediction.median[index]), float(prediction.sigma[index])


def test_ground_motion_matches_the_authors_verification_values() -> None:
    for (magnitude, rake, rhyp, vs30), expected in AUTHORS_MEDIANS.items():
        prediction = ground_motion.predict(magnitude, rhyp, rake, vs30)
        for period, median_g, sigma in zip(CHECK_PERIODS, expected, AUTHORS_TOTAL_SIGMA):
            got_median, got_sigma = _at(prediction, period)
            assert got_median == pytest.approx(median_g, rel=1e-6), (magnitude, rake, rhyp, vs30, period)
            assert got_sigma == pytest.approx(sigma, rel=1e-6)


def test_ground_motion_period_grid_and_scatter() -> None:
    prediction = ground_motion.predict(5.0, 15.0, 0.0, 400.0)
    assert prediction.periods[0] == 0.0
    assert prediction.periods[1] == 0.01 and prediction.periods[-1] == 4.0
    assert np.all(np.diff(prediction.periods) > 0)
    assert np.allclose(prediction.median_plus_sigma, prediction.median * np.exp(prediction.sigma))
    assert np.all(prediction.sigma > 0.6) and np.all(prediction.sigma < 0.9)


def test_mechanism_and_site_terms_behave_as_in_the_paper() -> None:
    assert ground_motion.mechanism_flags(0.0) == (0, 0)
    assert ground_motion.mechanism_flags(-90.0) == (1, 0)
    assert ground_motion.mechanism_flags(90.0) == (0, 1)
    assert ground_motion.mechanism_flags(180.0) == (0, 0)

    rock = ground_motion.predict(5.0, 15.0, 0.0, 750.0)
    stiff, stiffer = (ground_motion.predict(5.0, 15.0, 0.0, v) for v in (1000.0, 1500.0))
    assert np.allclose(stiff.median, stiffer.median), "no de-amplification beyond Vs30 = 1000 m/s"
    assert np.all(stiff.median < rock.median)

    # Soil response is non-linear: a soft site amplifies weak motion and, in this
    # model, even de-amplifies the short-period part of strong motion.
    weak_ratio = ground_motion.predict(5.0, 80.0, 0.0, 200.0).median / ground_motion.predict(5.0, 80.0, 0.0, 750.0).median
    strong_ratio = ground_motion.predict(7.0, 5.0, 0.0, 200.0).median / ground_motion.predict(7.0, 5.0, 0.0, 750.0).median
    assert weak_ratio[0] > 1.0 > strong_ratio[0]

    with pytest.raises(ValueError):
        ground_motion.predict(3.5, 15.0, 0.0, 400.0)
    with pytest.raises(ValueError):
        ground_motion.predict(5.0, 250.0, 0.0, 400.0)


def test_sia261_spectrum_branches() -> None:
    agd, ground = 1.6, sia261.GROUND_CLASSES["C"]
    t = np.array([0.0, ground.tb, (ground.tb + ground.tc) / 2, ground.tc, ground.td, 2 * ground.td])
    se = sia261.elastic_spectrum(t, agd, "C")
    plateau = 2.5 * agd * ground.soil_factor
    assert se[0] == pytest.approx(agd * ground.soil_factor)
    assert se[1] == pytest.approx(plateau) and se[2] == pytest.approx(plateau) and se[3] == pytest.approx(plateau)
    assert se[4] == pytest.approx(plateau * ground.tc / ground.td)
    assert se[5] == pytest.approx(se[4] / 4), "constant-displacement branch falls with 1/T^2"

    fine = np.linspace(0.0, 4.0, 4001)
    assert np.all(np.abs(np.diff(sia261.elastic_spectrum(fine, agd, "C"))) < 0.05), "no jumps between branches"
    assert np.allclose(sia261.elastic_spectrum(fine, agd, "C", "III"), 1.5 * sia261.elastic_spectrum(fine, agd, "C"))
    assert np.all(sia261.elastic_spectrum(fine, agd, "D") >= sia261.elastic_spectrum(fine, agd, "A"))
    assert sia261.damping_correction(5.0) == 1.0
    assert sia261.damping_correction(100.0) == 0.55
    assert sia261.elastic_spectrum(np.array([0.3]), agd, "C", damping_percent=10.0) < plateau
    with pytest.raises(KeyError):
        sia261.elastic_spectrum(fine, agd, "F")


def test_ground_class_from_vs30() -> None:
    assert [sia261.ground_class_for_vs30(v) for v in (900.0, 600.0, 400.0, 200.0)] == ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        sia261.ground_class_for_vs30(100.0)


def test_distances() -> None:
    assert seismic.epicentral_distance_km(46.0, 7.0, 46.0, 7.0) == 0.0
    assert seismic.epicentral_distance_km(46.0, 7.0, 47.0, 7.0) == pytest.approx(111.19, abs=0.05)
    spectra = seismic.scenario_spectra(seismic.Scenario())
    assert spectra.epicentral_km == pytest.approx(12.0, abs=0.2)
    assert spectra.hypocentral_km == pytest.approx(np.hypot(spectra.epicentral_km, 10.0))


def test_default_scenario_spectra_are_plausible() -> None:
    spectra = seismic.scenario_spectra()
    assert spectra.periods[0] == 0.01 and spectra.periods[-1] == 4.0
    assert np.allclose(spectra.median_plus_sigma, spectra.median * np.exp(spectra.sigma))
    assert spectra.pga_median == pytest.approx(spectra.at(0.01)[0], rel=0.05), "PGA and SA(0.01 s) nearly coincide"
    assert 0.3 < spectra.pga_median < 2.0, "a Mw 5 at 15 km gives a few percent of g"
    assert spectra.periods[np.argmax(spectra.median)] < 0.3, "moderate earthquakes peak at short periods"
    assert spectra.code_pga == pytest.approx(1.6 * 1.45)
    assert spectra.code_pga_by_class == pytest.approx({"I": 1.6 * 1.45, "II": 1.2 * 1.6 * 1.45, "III": 1.5 * 1.6 * 1.45})
    assert np.all(spectra.code > spectra.median_plus_sigma), "the 475-year design spectrum envelopes this scenario"

    median, plus_sigma, code = spectra.at(0.15)
    assert median < plus_sigma < code
    assert code == pytest.approx(2.5 * 1.6 * 1.45)


def test_scenario_parameters_change_the_answer() -> None:
    base = seismic.scenario_spectra()
    stronger = seismic.scenario_spectra(seismic.Scenario(earthquake=seismic.Earthquake(magnitude=6.0)))
    deeper = seismic.scenario_spectra(seismic.Scenario(earthquake=seismic.Earthquake(depth_km=20.0)))
    reverse = seismic.scenario_spectra(seismic.Scenario(earthquake=seismic.Earthquake(rake_deg=90.0)))
    softer = seismic.scenario_spectra(seismic.Scenario(site=seismic.Site(vs30=200.0, ground_class="D")))
    important = seismic.scenario_spectra(seismic.Scenario(code=seismic.DesignCode(structure_class="III")))
    assert np.all(stronger.median > base.median)
    assert np.all(deeper.median < base.median)
    assert reverse.pga_median > base.pga_median
    assert softer.pga_median > base.pga_median and np.all(softer.code >= base.code)
    assert np.allclose(important.code, 1.5 * base.code) and np.allclose(important.median, base.median)


def test_command_line_reproduces_a_scenario() -> None:
    scenario = seismic.scenario_from_args(["--magnitude", "5.5", "--depth-km", "8", "--site-vs30", "250", "--site-ground-class", "D", "--zone", "Z2"])
    assert scenario.earthquake == seismic.Earthquake(magnitude=5.5, depth_km=8.0)
    assert scenario.site == seismic.Site(vs30=250.0, ground_class="D")
    assert scenario.code == seismic.DesignCode(zone="Z2")
    assert seismic.scenario_from_args([]) == seismic.Scenario()


@pytest.fixture(scope="module")
def modes(model_paths: tuple[Path, Path]) -> structural.VibrationModes:
    return structural.vibration_modes(model_paths[1], mesh_size=COARSE_MESH)


def test_vibration_modes_of_the_house(modes: structural.VibrationModes) -> None:
    assert np.all(np.diff(modes.periods) <= 0), "longest period first"
    assert 0.005 < modes.periods[0] < 0.1, "a squat concrete box is very stiff"
    fractions = modes.effective_mass_fraction
    assert fractions.shape == (len(modes.periods), 3)
    assert np.all(fractions >= 0) and np.all(fractions.sum(axis=0) <= 1.0 + 1e-9)
    assert 15e3 < modes.seismic_mass < 60e3, "dead weight of walls and roof, in kg"
    for axis in (0, 1):
        assert fractions[:, axis].max() > 0.1
        assert modes.dominant_period(axis) in modes.periods


def test_house_response_and_figure(modes: structural.VibrationModes, tmp_path: Path) -> None:
    spectra = seismic.scenario_spectra()
    responses = seismic.house_response(spectra, modes)
    assert [r.axis for r in responses] == ["X", "Y"]
    for r in responses:
        assert r.median < r.median_plus_sigma < r.code
        assert r.code == r.code_by_class["I"] < r.code_by_class["II"] < r.code_by_class["III"]
        assert r.code_by_class["III"] == pytest.approx(1.5 * r.code_by_class["I"])
        assert r.base_shear(r.code) == pytest.approx(r.code * modes.seismic_mass)
    assert "PGA" in seismic.spectra_table(spectra)
    assert "base shear" in seismic.house_table(responses)

    image = render_spectra.render_spectra(spectra, responses, tmp_path / "spectra.png")
    assert image.stat().st_size > 10_000


@pytest.fixture(scope="module")
def lateral(model_paths: tuple[Path, Path]) -> tuple[list[structural.LateralLoad], dict[str, structural.AnalysisResult]]:
    loads = [structural.LateralLoad("EQ X", 0, 1.0), structural.LateralLoad("EQ Y", 1, 1.0), structural.LateralLoad("EQ X double", 0, 2.0)]
    return loads, structural.analyse_lateral(model_paths[1], loads, mesh_size=COARSE_MESH)


def test_lateral_load_is_the_seismic_mass_times_acceleration(
    lateral: tuple[list[structural.LateralLoad], dict[str, structural.AnalysisResult]], modes: structural.VibrationModes
) -> None:
    loads, results = lateral
    for load in loads:
        r = results[load.name]
        assert r.combo == load.name
        assert r.reaction_imbalance < 1e-9
        assert r.base_shear(load.axis) == pytest.approx(load.acceleration * modes.seismic_mass, rel=0.02)
        assert abs(r.reaction[1 - load.axis]) < 1e-6 * r.base_shear(load.axis), "no shear across the load"
        assert r.max_lateral_displacement > 0


def test_lateral_response_is_linear_in_the_acceleration(
    lateral: tuple[list[structural.LateralLoad], dict[str, structural.AnalysisResult]],
) -> None:
    _, results = lateral
    single, double = results["EQ X"], results["EQ X double"]
    assert double.base_shear(0) == pytest.approx(2 * single.base_shear(0))
    assert double.reaction[2] == pytest.approx(single.reaction[2]), "dead weight is the same in both"
    assert double.max_lateral_displacement > 1.5 * single.max_lateral_displacement


def test_scenario_lateral_cases_and_figure(
    lateral: tuple[list[structural.LateralLoad], dict[str, structural.AnalysisResult]], modes: structural.VibrationModes, tmp_path: Path
) -> None:
    from ifc_fem_demo import render_3d

    responses = seismic.house_response(seismic.scenario_spectra(), modes)
    loads = seismic.lateral_loads(responses)
    assert seismic.LEVELS == ("median", "median+1sigma", "SIA I", "SIA II", "SIA III")
    assert [load.name for load in loads] == [f"EQ {a} {level}" for a in "XY" for level in seismic.LEVELS]
    x = responses[0]
    assert [load.acceleration for load in loads[:5]] == [x.median, x.median_plus_sigma, *x.code_by_class.values()]
    assert render_3d.inertia_arrow_colour(loads[0]) == render_spectra.SCENARIO_COLOUR
    assert render_3d.inertia_arrow_colour(loads[4]) == render_spectra.CODE_COLOURS["III"]

    test_loads, results = lateral
    assert seismic.governing_case(results) is results["EQ X double"]
    assert "base shear" in structural.lateral_table(results, test_loads)
    image = render_3d.render_seismic_comparison(results, test_loads, tmp_path / "seismic.png")
    assert image.stat().st_size > 10_000
