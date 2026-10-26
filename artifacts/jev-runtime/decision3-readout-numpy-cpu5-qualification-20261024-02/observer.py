"""Read-only final-norm observer and original FP32 readout replay.

No framework is imported by this module. The owned caller passes its already
authenticated genuine runtime and Torch module. No encoder is called here.
"""

from __future__ import annotations

import math

ABS_TOL = 1e-6
REL_TOL = 1e-5
WIDTH = 1024
CODES = 255


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_identity(tensor):
    return (
        id(tensor),
        tuple(tensor.shape),
        str(tensor.dtype),
        tensor.device.type,
        tensor.data_ptr(),
        tensor._version,
    )


def baseline_identity(runtime):
    """Metadata/version witnesses; not a hash of live baseline tensor values."""
    return {
        "readout": tensor_identity(runtime.readout),
        "backbone": tuple(tensor_identity(p) for p in runtime.backbone.parameters()),
        "encoder": id(runtime.encoder),
        "norm": id(runtime.backbone.language_model.norm),
    }


class FinalNormObserver:
    """Copy only [B, D] last-position values; always return None to Torch."""

    def __init__(self, torch, rows, padded_width):
        require(1 <= rows <= 4 and padded_width > 0, "invalid declared partial batch")
        self.torch = torch
        self.rows, self.padded_width = rows, padded_width
        self.first_layer_events = []
        self.norm_events = []
        self.features = None

    def _check(self, tensor):
        require(
            tuple(tensor.shape) == (self.rows, self.padded_width, WIDTH)
            and tensor.device.type == "cpu"
            and tensor.dtype == self.torch.bfloat16
            and not self.torch.cuda.is_initialized(),
            "genuine complete CPU BF16 partial-batch shape differs",
        )

    def first_layer(self, _module, args):
        require(len(self.first_layer_events) == 0 and args, "extra/missing encoder pass")
        self._check(args[0])
        self.first_layer_events.append(list(args[0].shape))
        return None

    def __call__(self, _module, _args, output):
        require(len(self.norm_events) == 0, "extra final-norm invocation")
        self._check(output)
        # The original runtime uses exactly this final-position slice. Only the
        # detached copy is retained, never [B, L, D] or the original output.
        features = output[:, -1, :].detach().to(device="cpu", dtype=self.torch.float32).clone()
        require(
            tuple(features.shape) == (self.rows, WIDTH)
            and bool(self.torch.isfinite(features).all().item()),
            "finite complete last-position features required",
        )
        self.features = features
        self.norm_events.append(list(output.shape))
        return None

    def require_single_pass(self):
        require(
            len(self.first_layer_events) == len(self.norm_events) == 1
            and self.features is not None,
            "exactly one original encoder/final-norm pass required",
        )


def replay_probabilities(runtime, torch, observer, counts):
    """Original h.float() @ W0.T, legal 255-code mask, T=1, softmax.

    This calls no encoder/backbone/prepare/system_one method and mutates neither
    the baseline readout nor the captured feature tensor.
    """
    observer.require_single_pass()
    require(
        len(counts) == observer.rows
        and all(type(n) is int and 1 <= n <= CODES for n in counts)
        and runtime.temperature == 1.0
        and runtime.readout.dtype == torch.float32
        and tuple(runtime.readout.shape) == (CODES, WIDTH),
        "original FP32 readout/legal counts/temperature required",
    )
    with torch.inference_mode():
        logits = observer.features.float() @ runtime.readout.T
        limit = torch.as_tensor(list(counts), device=runtime.device)[:, None]
        invalid = torch.arange(CODES, device=runtime.device)[None] >= limit
        masked = logits.masked_fill(invalid, float("-inf"))
        rows = torch.softmax(masked / runtime.temperature, dim=-1).cpu().tolist()
    observer.require_single_pass()
    require(len(rows) == len(counts), "replay row count differs")
    require(
        all(all(value == 0 for value in row[count:]) for row, count in zip(rows, counts)),
        "illegal codes have nonzero probability",
    )
    return [row[:count] for row, count in zip(rows, counts)]


def compare_distributions(raw, questions, replayed):
    """Check every legal probability and exact ordered argmax, no gold labels."""
    require(len(replayed) == len(questions), "incomplete replay rows")
    rows = []
    for (key, question), predicted in zip(questions.items(), replayed):
        labels = list(question["criteria"])
        answer = raw["answers"][key]
        actual = answer["probabilities"]
        require(
            list(actual) == labels and len(predicted) == len(labels),
            "original complete legal code/label order differs",
        )
        require(
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in predicted)
            and abs(sum(predicted) - 1) <= ABS_TOL,
            "replayed finite normalized distribution required",
        )
        errors = []
        for label, value in zip(labels, predicted):
            original = actual[label]
            require(
                type(original) in (int, float) and math.isfinite(original),
                "nonfinite original probability",
            )
            error = abs(value - original)
            require(
                error <= ABS_TOL + REL_TOL * abs(original),
                "original/replay probability differs beyond predeclared tolerance",
            )
            errors.append(error)
        replay_choice = labels[max(range(len(predicted)), key=predicted.__getitem__)]
        native_choice = max(actual, key=actual.__getitem__)
        require(
            answer["choice"] == native_choice == replay_choice,
            "native/replay ordered argmax differs",
        )
        rows.append({"head": key, "legal_options": len(labels), "max_absolute_error": max(errors)})
    return {
        "all_legal_probabilities_matched": True,
        "all_ordered_argmaxes_matched": True,
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "comparison": "abs(replay-native) <= 1e-6 + 1e-5*abs(native)",
        "rows": rows,
        "semantic_correctness_evaluated": False,
    }
