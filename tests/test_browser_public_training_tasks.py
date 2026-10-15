"""Fixed development task identities and detached private route metadata."""

import hashlib
import unicodedata
from urllib.parse import urlsplit

import pytest

from s1.browser_public_training_tasks import (
    PRIVATE_ROUTE_HINT_REVISION,
    PROPOSAL_SHA256,
    get_public_training_task,
    public_training_tasks,
    tasks,
)


def test_exact_twelve_tasks_and_split_site_separation():
    records = tasks()
    assert len(records) == 12
    assert len({task["goal_id"] for task in records}) == 12
    assert sum(task["split"] == "train" for task in records) == 8
    assert sum(task["split"] == "development" for task in records) == 4
    for field in ("site_group", "layout_group"):
        train = {task[field] for task in records if task["split"] == "train"}
        development = {task[field] for task in records if task["split"] == "development"}
        assert train.isdisjoint(development)
    assert PROPOSAL_SHA256 == "e3c2c9833709f3d5c05a320c13403f46aa8b8938acaf7e867d52470928f4bdfa"


@pytest.mark.parametrize("task", tasks(), ids=lambda task: task["goal_id"])
def test_goal_hash_url_and_private_bootstrap_route_are_bound(task):
    normalized = " ".join(unicodedata.normalize("NFKC", task["goal"]).split())
    assert hashlib.sha256(normalized.encode()).hexdigest() == task["goal_sha256"]
    target = urlsplit(task["primary_target_url"])
    assert target.scheme == "https"
    assert [target.hostname] == task["allowed_top_level_hosts"]
    assert target.path == task["target_path"] == task["route_paths"][-1]
    other = get_public_training_task(task["bootstrap_goal_id"])
    assert task["bootstrap_target_path"] == other["target_path"] != task["target_path"]
    assert task["bootstrap_route_paths"] == other["route_paths"]
    assert task["split"] == other["split"]
    assert task["site_group"] == other["site_group"]
    assert target.hostname not in {
        "docs.python.org",
        "science.nasa.gov",
        "arxiv.org",
        "en.wikipedia.org",
    }


def test_task_dictionaries_are_detached_and_changed_hints_rejected():
    first = tasks()
    original = public_training_tasks()
    first[0]["route_paths"].clear()
    first[0]["goal"] = "CHANGED"
    assert tasks() == original
    with pytest.raises(ValueError, match="differs"):
        get_public_training_task(first[0])
    with pytest.raises(ValueError, match="unknown"):
        get_public_training_task("not-one-of-the-twelve")


def test_mdn_goal_requires_syntax_and_signature_without_an_explanation_condition():
    mdn = [task for task in tasks() if task["judge_kind"] == "mdn_method"]
    assert len(mdn) == 4
    assert all(task["target_heading"] == "Syntax" for task in mdn)
    assert all("explanation" not in task["goal"] for task in mdn)


def test_private_mdn_category_routes_keep_partner_recovery_hint_consistent():
    assert PRIVATE_ROUTE_HINT_REVISION == "mdn-specifications-forward-routes-v2"
    categories = {
        "mdn-intersectionobserver-observe": "/en-US/docs/Web/API/Intersection_Observer_API",
        "mdn-resizeobserver-observe": "/en-US/docs/Web/API/Resize_Observer_API",
        "mdn-urlsearchparams-get": "/en-US/docs/Web/API/URL_API",
    }
    for task in tasks():
        if task["judge_kind"] != "mdn_method":
            assert "index_section_fragment" not in task
            continue
        other = get_public_training_task(task["bootstrap_goal_id"])
        expected = "specifications" if task["goal_id"] in categories else "interfaces"
        assert task["index_section_fragment"] == expected
        assert task["bootstrap_index_section_fragment"] == other["index_section_fragment"]
        if task["goal_id"] in categories:
            assert task["route_paths"][0] == categories[task["goal_id"]]
