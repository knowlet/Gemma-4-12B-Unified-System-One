"""Synthetic fixture isolation when both binding plugins are collected."""

import jev_decision3_candidate_browser_binding as candidate
import jev_decision3_semantic_browser_binding as semantic
import jev_decision3_semantic_operation_audit as audit

pytest_plugins = (
    "test_jev_decision3_semantic_browser_binding",
    "test_jev_decision3_candidate_browser_binding",
)


def test_candidate_and_semantic_fixtures_remain_distinct(
    candidate_binding_context, semantic_binding_context
):
    candidate_instance = candidate_binding_context.make()
    semantic_instance = semantic_binding_context.make()
    try:
        assert type(candidate_instance) is candidate.CandidateBrowserBinding
        assert type(semantic_instance) is semantic.SemanticBrowserBinding
        assert semantic_binding_context.lease.held is None
        assert audit.SOURCE_ONLY is False
    finally:
        candidate_instance.close()
        semantic_instance.close()
