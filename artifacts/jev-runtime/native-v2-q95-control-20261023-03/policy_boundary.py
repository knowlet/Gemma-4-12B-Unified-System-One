"""The unchanged checked whole20s ASGI boundary; no new policy semantics."""

from checked_source import module

selected = module(
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/policy_boundary.py",
    "4ed323e24cff140caef36c6557fe3286154e1c11b92f39536615eab6f1d3a8ef",
    "_native_q95_pin_selected_boundary",
)
ControlState, PolicyBoundary = selected.ControlState, selected.PolicyBoundary
