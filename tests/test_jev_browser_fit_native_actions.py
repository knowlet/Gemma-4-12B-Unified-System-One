"""Historical179 orchestration; genuine tiny CPU updates are not12B qualification."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jev_browser_fit_native_actions as FIT
import pytest
from local_evidence import require_local_evidence
from test_jev_browser_fit_actions import tiny as tiny

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MANIFEST = (
    ROOT
    / "artifacts/jev-runtime/owned-public-mixed-composition-20261018-01/actual-admit-02/mixed-manifest-A.json"
)


@pytest.fixture
def fake_boot_observation(monkeypatch):
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = "11111111-2222-4333-8444-555555555555"
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == boot_path:
            return boot_id + "\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(FIT.Path, "read_text", read_text)
    return boot_id


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 281},
        {"seed": True},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"prompt_format": "raw-json-v1"},
        {"prompt_format": None},
    ],
)
def test_bad_options_refuse_before_environment_or_output(tmp_path, monkeypatch, options):
    def forbidden():
        raise AssertionError("environment/model must not be entered")

    monkeypatch.setattr(FIT, "_environment", forbidden)
    output = tmp_path / "fit"
    with pytest.raises(ValueError):
        FIT.fit_native_actions(historical_root=ROOT, output_dir=output, **options)
    assert not output.exists()


def test_default_cli_is_dry_without_environment_or_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(FIT, "_environment", lambda: pytest.fail("dry must not enter fitter"))
    output = tmp_path / "uncreated"
    assert FIT.main(["--output-dir", str(output)]) == 0
    assert (
        json.loads(capsys.readouterr().out)["status"] == "DRY_NO_DATA_PROCESSOR_MODEL_OR_TRAINING"
    )
    assert not output.exists()


def test_real_constructor_arguments_preserve_pinned_temperature_and_base(monkeypatch):
    from s1.browser_action_adapter import BASE_ID, BASE_REVISION, LORA

    calls = {}
    torch = SimpleNamespace(
        set_num_threads=lambda n: calls.update(threads=n),
        manual_seed=lambda n: calls.update(seed=n),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(enable_cudnn_sdp=lambda value: calls.update(cudnn=value))
        ),
    )

    def construct(name, **kwargs):
        calls.update(name=name, kwargs=kwargs)
        return "constructor-not-entered"

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "s1.unified", SimpleNamespace(UnifiedDecisionModel=construct))
    assert FIT._construct_model(42) == "constructor-not-entered"
    assert calls["name"] == BASE_ID and calls["seed"] == 42 and calls["threads"] == 2
    assert calls["cudnn"] is False
    assert calls["kwargs"] == {
        "revision": BASE_REVISION,
        "device": "cuda",
        "precision": "bfloat16",
        "quantization": "none",
        "max_context": 8192,
        "temperature": FIT._policy().artifact.TEMPERATURE,
        "attn_implementation": "sdpa",
        "lora": {**LORA, "revision": BASE_REVISION},
    }


def test_initial_resource_failure_is_preserved_before_data_or_model(
    tmp_path, monkeypatch, fake_boot_observation
):
    class Refuse:
        def check(self, phase, scope):
            assert phase == "before-any-project-framework-or-parent-payload" and scope == "cpu"
            raise ValueError("original resource admission refused")

    def gate(_path, boot_id):
        assert boot_id == fake_boot_observation
        return Refuse()

    monkeypatch.setattr(FIT, "_environment", lambda: None)
    monkeypatch.setattr(FIT, "_gate", gate)
    monkeypatch.setattr(FIT, "_construct_model", lambda *_: pytest.fail("must not load model"))
    output = tmp_path / "fit"
    with pytest.raises(ValueError, match="original resource admission"):
        FIT.fit_native_actions(historical_root=ROOT, output_dir=output)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed" and receipt["model_loads"] == 0
    assert receipt["training_updates"] == 0 and not (output / "inputs").exists()


def test_overlapping_parent_refuses_before_any_parent_write(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "adapter").mkdir(parents=True)
    (parent / "receipt.json").write_text("immutable parent sentinel")
    before = sorted(str(p.relative_to(parent)) for p in parent.rglob("*"))
    monkeypatch.setattr(FIT, "_environment", lambda: pytest.fail("must reject before setup"))
    with pytest.raises(ValueError, match="immutable initial parent"):
        FIT.fit_native_actions(
            historical_root=ROOT,
            output_dir=parent / "child",
            initial_action_adapter=parent / "adapter",
        )
    assert sorted(str(p.relative_to(parent)) for p in parent.rglob("*")) == before
    assert (parent / "receipt.json").read_text() == "immutable parent sentinel"


def test_captured_package_and_closed_evidence_outputs_refuse_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(FIT, "_environment", lambda: pytest.fail("must reject before setup"))
    package_output = ROOT / "src/s1/native-fit-test-must-not-be-created"
    with pytest.raises(ValueError, match="captured source"):
        FIT.fit_native_actions(historical_root=ROOT, output_dir=package_output)
    assert not package_output.exists()
    historical = tmp_path / "historical"
    old = historical / FIT.HISTORICAL_INPUT_TREES[0]
    old.mkdir(parents=True)
    with pytest.raises(ValueError, match="immutable historical input tree"):
        FIT.fit_native_actions(historical_root=historical, output_dir=old / "child")
    assert not (old / "child").exists()


class NativeToyTokenizer:
    """Short synthetic tokenization, with the actual serialized native envelope."""

    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, -1, 1, 106
    all_special_ids = [0, 1, 50, 106]
    operations = {
        name: 20 + i
        for i, name in enumerate(
            ("CLICK", "DONE", "BLOCKED", "WAIT", "SCROLL_DOWN", "SCROLL_UP", "TYPE_TEXT", "SELECT")
        )
    }

    def __init__(self):
        self.prompt, self.targets = "", {}
        self.available = [i for i in range(40, 512) if i not in self.all_special_ids]

    def convert_tokens_to_ids(self, token):
        return 106 if token == "<turn|>" else -1

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<turn|>":
            return [106]
        if self.prompt and text.startswith(self.prompt):
            tail = text[len(self.prompt) :]
            return [1, 3, 4] + (self.encode(tail) if tail else [])
        if text.endswith("<turn|>"):
            return self.encode(text.removesuffix("<turn|>")) + [106]
        action = json.loads(text)
        target = action["target"]
        if target is None:
            target_id = 13
        else:
            if target not in self.targets:
                self.targets[target] = self.available[len(self.targets)]
            target_id = self.targets[target]
        return [10, self.operations[action["operation"]], 11, target_id, 12]


class NativeToyProcessor:
    def __init__(self, torch):
        self.torch, self.tokenizer = torch, NativeToyTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        from s1.browser_action_prompt import native_prompt

        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.tokenizer.targets.clear()
        self.tokenizer.prompt = native_prompt(messages[0]["content"][0]["text"])
        return self.tokenizer.prompt

    def __call__(self, *, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        ids = self.torch.tensor([self.tokenizer.encode(text)])
        return {
            "input_ids": ids,
            "attention_mask": self.torch.ones_like(ids),
            "mm_token_type_ids": self.torch.zeros_like(ids),
        }


@pytest.fixture
def native_toy(request, monkeypatch, fake_boot_observation):
    from s1 import browser_native_action_data as data

    require_local_evidence(ROOT / data.COMPONENT["path"], HISTORICAL_MANIFEST)
    tiny = request.getfixturevalue("tiny")

    import jev_browser_fit_actions as kernel
    import torch

    tiny.lm.to(dtype=torch.bfloat16)
    for parameter in tiny.lm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    tiny.processor = NativeToyProcessor(torch)
    tiny.tok = tiny.processor.tokenizer
    tiny.lm.get_base_model().generation_config.eos_token_id = [1, 106, 50]
    initial = copy.deepcopy(tiny, {id(torch): torch})
    gates = []

    class FakeCPUGate:
        def check(self, phase, scope):
            gates.append((phase, scope))

    def gate(_path, boot_id):
        assert boot_id == fake_boot_observation
        return FakeCPUGate()

    def check(model, fitter):
        # Only these explicitly mocked base/shape/resource checks differ from
        # production. The unchanged trainer still checks frozen parameters/head,
        # FP32 q/v-only gradients, actual updates and held-aside forwards.
        assert fitter is kernel and model.device == "cpu"
        assert model.head.weight.dtype == torch.bfloat16 and not model.head.weight.requires_grad

    monkeypatch.setattr(FIT, "_environment", lambda: None)
    monkeypatch.setattr(FIT, "_gate", gate)

    def checkpoint():
        assert gates[-1] == ("before-checkpoint-payload-hash-and-BF16-loader", "load")
        gates.append(("checkpoint-hash", "fixture"))
        return {"scope": "tiny-test-no-12B-checkpoint"}

    def construct(_):
        assert gates[-1] == ("before-any-BF16-model-construction", "train")
        assert gates[-2] == ("checkpoint-hash", "fixture")
        return tiny

    monkeypatch.setattr(FIT, "_checkpoint", checkpoint)
    monkeypatch.setattr(FIT, "_construct_model", construct)
    monkeypatch.setattr(FIT, "_check_model", check)
    return tiny, initial, gates


def lines(path):
    return [json.loads(x) for x in path.read_bytes().splitlines() if x.strip()]


@pytest.mark.local_evidence
def test_genuine_tiny_one_pass_uses_all179_and_exports_matching_native_profile(
    tmp_path, native_toy
):
    from s1.browser_action_adapter import verify_action_adapter
    from s1.browser_action_prompt import expected_format

    model, _, gates = native_toy
    before = {n: p.detach().clone() for n, p in model.lm.named_parameters() if p.requires_grad}
    output = tmp_path / "native"
    receipt = FIT.fit_native_actions(historical_root=ROOT, output_dir=output)
    assert receipt["stage"] == "complete" and receipt["training_updates"] == 142
    assert receipt["training"]["development_updates"] == 0
    assert receipt["training"]["development_examples"] == 37
    assert receipt["training"]["frozen_base_versions_unchanged"]
    assert (
        receipt["training"]["original_head_sha256_before"]
        == receipt["training"]["original_head_sha256_after"]
    )
    train, dev = (
        lines(output / "compiled-train.jsonl"),
        lines(output / "compiled-development.jsonl"),
    )
    assert len(train) == 142 and len(dev) == 37
    assert {r["case_id"] for r in lines(output / "updates.jsonl")} == {r["case_id"] for r in train}
    assert {r["prompt_representation"]["format"] for r in train + dev} == {
        "raw-json-v1",
        FIT.FORMAT,
    }
    assert all(
        r["prompt_format"] == FIT.FORMAT and r["native_stop_token_id"] == 106 for r in train + dev
    )
    assert any(
        not p.detach().equal(before[n]) for n, p in model.lm.named_parameters() if p.requires_grad
    )
    verified = verify_action_adapter(output / "adapter", expected_prompt_format=FIT.FORMAT)
    assert expected_format(verified.telemetry(), FIT.FORMAT) == FIT.FORMAT
    with pytest.raises(ValueError):
        verify_action_adapter(output / "adapter", expected_prompt_format="raw-json-v1")
    assert sum(phase.startswith("before-BF16-processor-compile-") for phase, _ in gates) == 179
    assert sum(phase == "before-native-train-forward" for phase, _ in gates) == 142
    assert sum(phase == "before-native-development-forward" for phase, _ in gates) == 37
    assert not model.lm._forward_pre_hooks


@pytest.mark.local_evidence
def test_genuine_native_warm_start_loads_exact_parent_and_preserves_development(
    tmp_path, native_toy, monkeypatch
):
    model, initial, _ = native_toy
    parent = tmp_path / "parent"
    FIT.fit_native_actions(historical_root=ROOT, output_dir=parent, steps=1)
    monkeypatch.setattr(FIT, "_construct_model", lambda _: initial)
    child = tmp_path / "child"
    receipt = FIT.fit_native_actions(
        historical_root=ROOT, output_dir=child, steps=1, initial_action_adapter=parent / "adapter"
    )
    check = receipt["initial_action_adapter"]["initial_tensor_verification"]
    assert check["all_tensors_exact"] and check["frozen_base_versions_unchanged"]
    assert receipt["training"]["development_updates"] == 0
    assert receipt["training_updates"] == 1 and not model.lm._forward_pre_hooks


@pytest.mark.local_evidence
def test_native_partial_update_keeps_primary_and_has_no_valid_manifest(
    tmp_path, native_toy, monkeypatch
):
    import jev_browser_fit_actions as kernel

    model, _, _ = native_toy
    original, calls = kernel.action_loss, 0

    def failure(model, row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("primary native update failure")
        return original(model, row)

    monkeypatch.setattr(kernel, "action_loss", failure)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="primary native update failure"):
        FIT.fit_native_actions(historical_root=ROOT, output_dir=output, steps=2)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed" and receipt["partial_training"]["completed_updates"] == 1
    assert receipt["training_updates"] == 1
    assert not (output / "adapter/manifest.json").exists()
    assert not model.lm._forward_pre_hooks and not model.backbone._forward_pre_hooks
