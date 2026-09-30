from pathlib import Path

from s1.evaluation.datasets import load_cases
from s1.evaluation.experiments import perturb_dataset, prepare_sweep


def test_order_perturbations_retain_semantic_gold_and_groups(tmp_path):
    source = Path(__file__).resolve().parents[2] / "examples/benchmarks/text-contract.jsonl"
    output = tmp_path / "permuted.jsonl"
    perturb_dataset(source, output, seed=3)
    for left, right in zip(load_cases(source), load_cases(output)):
        assert left.gold == right.gold
        assert left.group_id == right.group_id
        assert {q.id for q in left.request.questions} == {q.id for q in right.request.questions}


def test_sweeps_cover_contract_bounds_and_are_explicit_fixtures(tmp_path):
    manifest = prepare_sweep(tmp_path / "sweep", cases_per_cell=2)
    assert manifest["quality_claim"] is False
    assert len(manifest["cells"]) == 21
    cases = load_cases(tmp_path / "sweep/n-64.jsonl")
    assert len(cases[0].request.questions) == 64
    cases = load_cases(tmp_path / "sweep/k-52.jsonl")
    assert len(cases[0].request.questions[0].labels()) == 52
