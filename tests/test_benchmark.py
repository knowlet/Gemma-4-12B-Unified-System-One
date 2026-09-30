import json

import pytest

from s1.backends import UniformBackend, UnsupportedRequest
from s1.benchmark import compare_reports, evaluate, load_cases, write_report
from s1.training import fit_temperature, permute_request


def test_uniform_known_metrics(benchmark_case):
    report = evaluate(UniformBackend(), [benchmark_case])
    assert report["metrics"]["n"] == 3
    assert report["metrics"]["acc"] == pytest.approx(2 / 3)
    assert report["metrics"]["brier"] == pytest.approx(0.5)
    assert report["metrics"]["nll"] == pytest.approx(0.69314718)
    assert report["metrics"]["score_mae"] == 5
    assert report["coverage"] == 1


def test_choice_tie_is_preserved_with_distribution_based_metrics(benchmark_case):
    class TiedChoice(UniformBackend):
        def predict(self, request):
            result = super().predict(request)
            result["answers"]["route"]["choice"] = "technical"
            return result

    report = evaluate(TiedChoice(), [benchmark_case])
    record = next(r for r in report["records"] if r["question_id"] == "route")
    assert record["choice"] == "technical"
    assert record["probabilities"] == {"billing": 0.5, "technical": 0.5}
    assert report["by_type"]["choice"]["acc"] == 1.0


def test_failures_and_common_subset(benchmark_case):
    class Broken:
        name = "broken"

        def predict(self, request):
            raise UnsupportedRequest("text only")

    good = evaluate(UniformBackend(), [benchmark_case])
    bad = evaluate(Broken(), [benchmark_case])
    assert bad["coverage"] == 0
    assert bad["counts"] == {"unsupported": 3}
    assert bad["metrics"] == {"n": 0}
    comparison = compare_reports([good, bad])
    assert comparison["common_decisions"] == 0
    assert comparison["metrics"]["uniform"] == {"n": 0}
    bad["dataset_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_reports([good, bad])


def test_report_roundtrip(benchmark_case, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(benchmark_case.model_dump_json() + "\n")
    loaded = load_cases(dataset)
    report = evaluate(UniformBackend(), loaded)
    write_report(report, tmp_path / "report.json")
    assert (
        json.loads((tmp_path / "report.json").read_text())["dataset_sha256"]
        == report["dataset_sha256"]
    )
    dataset.write_text(benchmark_case.model_dump_json() + "\n" + benchmark_case.model_dump_json())
    with pytest.raises(ValueError, match="unique"):
        load_cases(dataset)


def test_test_data_cannot_be_calibration(benchmark_case):
    benchmark_case.split = "calibration"
    with pytest.raises(ValueError, match="test-only"):
        evaluate(UniformBackend(), [benchmark_case])


def test_temperature_fit_and_permutation(decision_request):
    import random

    fit = fit_temperature([[8, 0], [8, 0], [8, 0], [8, 0]], [0, 0, 0, 1])
    assert fit["temperature"] > 1
    assert fit["nll_after"] < fit["nll_before"]
    changed = permute_request(decision_request, random.Random(1))
    assert changed.questions[0].labels() != decision_request.questions[0].labels()
    assert changed.questions[0].criteria == decision_request.questions[0].criteria
    assert changed.questions[2] == decision_request.questions[2]
