"""Independent public-web judges use fresh, scoped DOM evidence only."""

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jev_public_web_cases", ROOT / "scripts" / "jev_public_web_cases.py"
)
PUBLIC_WEB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLIC_WEB
SPEC.loader.exec_module(PUBLIC_WEB)

SEARCH = "public-arxiv-title-filter-results-v1"
PAPER = "public-arxiv-original-paper-open-v1"
PYTHON = "public-python-counter-most-common-v1"
NASA = "public-nasa-mars-facts-moons-v1"
CASE_IDS = (SEARCH, PAPER, PYTHON, NASA)
PHRASE = '"Attention Is All You Need"'


def element(text, *, tag="div", element_id="", x=20, y=100, width=600, height=40):
    return {
        "tag": tag,
        "id": element_id,
        "text": text,
        "css_visible": True,
        "rect": {"x": x, "y": y, "width": width, "height": height},
        "center_unobscured": True,
        "intersection_unobscured": True,
    }


def observation(case_id):
    """Construct realistic article/form evidence, independent of judge helpers."""
    result = {
        "url": "",
        "document_title": "",
        "ready_state": "complete",
        "viewport": {"width": 1200, "height": 800},
        "main": element("Main document content", tag="main", height=1100, y=0),
        "headings": [],
        "blocked_elements": [],
        "overlays": [],
    }
    if case_id == SEARCH:
        result.update(
            url="https://arxiv.org/search/?abstracts=hide&query=%22Attention+Is+All+You+Need%22&searchtype=title&size=50",
            document_title="Search | arXiv e-print repository",
            headings=[element("Search", tag="h1", y=20)],
            arxiv_search={
                "forms": [
                    {
                        "action": "https://arxiv.org/search/",
                        "query": {**element("", tag="input", y=90), "value": PHRASE},
                        "searchtype": {
                            **element("Title", tag="select", y=140),
                            "value": "title",
                        },
                        "hide": {
                            **element("Hide abstracts", tag="input", y=190),
                            "value": "hide",
                            "checked": True,
                        },
                    }
                ],
                "summaries": [
                    element(
                        'Search results for Title: "Attention Is All You Need"', tag="h1", y=260
                    )
                ],
                "results": [
                    {
                        **element("Attention Is All You Need", tag="li", y=350),
                        "title": "Attention Is All You Need",
                        "links": ["https://arxiv.org/abs/1706.03762"],
                    }
                ],
            },
        )
    elif case_id == PAPER:
        title = element("Title: Attention Is All You Need", tag="h1", y=120)
        result.update(
            url="https://arxiv.org/abs/1706.03762v7",
            document_title="[1706.03762] Attention Is All You Need",
            headings=[title],
            arxiv_paper={
                "title": title,
                "authors": element("Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar", y=190),
                "identifier": element("arXiv:1706.03762v7 [cs.CL]", y=70),
                "submission_metadata": [
                    {
                        **element(
                            "[Submitted on 12 Jun 2017 (v1), last revised 2 Aug 2023 (this version, v7)]",
                            y=250,
                        ),
                        "kind": "dateline",
                    }
                ],
                "abstract": element(
                    "Abstract: We propose a new simple network architecture, the Transformer.",
                    tag="blockquote",
                    y=330,
                    height=280,
                ),
            },
        )
    elif case_id == PYTHON:
        paragraph = element(
            "Return a list of the n most common elements and their counts from the most common to the least.",
            tag="p",
            y=760,
            height=240,
        )
        paragraph["center_unobscured"] = False
        result.update(
            url="https://docs.python.org/3.14/library/collections.html",
            document_title="collections — Container datatypes — Python 3.14 documentation",
            headings=[element("collections — Container datatypes", tag="h1", y=-1800)],
            python={
                "counter_section": element(
                    "Counter objects",
                    tag="section",
                    element_id="counter-objects",
                    y=-1200,
                    height=3000,
                ),
                "method": element(
                    "most_common(n=None)",
                    tag="dt",
                    element_id="collections.Counter.most_common",
                    y=690,
                ),
                "description": element(paragraph["text"], tag="dd", y=740, height=1000),
                "first_paragraph": paragraph,
            },
        )
    elif case_id == NASA:
        result.update(
            url="https://science.nasa.gov/mars/facts/#moons",
            document_title="Mars Facts - NASA Science",
            headings=[element("Mars Facts", tag="h1", y=-1500)],
            nasa={
                "sections": [
                    {
                        "heading": element("Moons", tag="h2", y=160),
                        "content_before_next_h2": "Mars has two small moons, Phobos and Deimos.",
                    },
                    {
                        "heading": element("Rings", tag="h2", y=950),
                        "content_before_next_h2": "Mars has no rings.",
                    },
                ]
            },
        )
    else:
        raise AssertionError(case_id)
    return result


def judge(case_id, observed):
    return PUBLIC_WEB.validate_public_web_observation(
        PUBLIC_WEB.get_public_web_case(case_id), observed
    )


def assert_rejected(case_id, observed):
    result = judge(case_id, observed)
    assert result["success"] is False
    assert result["mismatches"]
    return result


def replace_url(observed, **parts):
    observed["url"] = urlunsplit(urlsplit(observed["url"])._replace(**parts))


def required_visible(case_id, observed):
    return {
        SEARCH: lambda: observed["arxiv_search"]["summaries"][0],
        PAPER: lambda: observed["arxiv_paper"]["title"],
        PYTHON: lambda: observed["python"]["method"],
        NASA: lambda: observed["nasa"]["sections"][0]["heading"],
    }[case_id]()


def test_manifest_contains_only_the_four_proposed_read_only_cases():
    cases = tuple(PUBLIC_WEB.PUBLIC_WEB_CASES.values())
    assert {case.case_id for case in cases} == set(CASE_IDS)
    assert len(cases) == len(CASE_IDS)
    manifest = PUBLIC_WEB.case_manifest()
    assert isinstance(manifest, list)
    assert json.loads(json.dumps(manifest)) == manifest
    assert {item["case_id"] for item in manifest} == set(CASE_IDS)
    for case in cases:
        assert dataclasses.is_dataclass(case)
        assert PUBLIC_WEB.get_public_web_case(case.case_id) == case
        entry = PUBLIC_WEB.case_manifest(case)
        assert isinstance(entry, dict)
        assert entry["case_id"] == case.case_id
        assert entry["goal"] == case.goal
        assert entry["start_url"] == case.start_url
        assert set(entry["allowed_top_level_hosts"]) == set(case.allowed_top_level_hosts)
        assert json.loads(json.dumps(entry)) == entry
        with pytest.raises(dataclasses.FrozenInstanceError):
            case.goal = "changed"
    with pytest.raises(ValueError):
        PUBLIC_WEB.get_public_web_case("public-debian-curl-exact-trixie-results-v1")
    with pytest.raises(ValueError):
        PUBLIC_WEB.get_public_web_case("unknown")


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_valid_scoped_observation_passes_and_is_not_mutated(case_id):
    observed = observation(case_id)
    before = copy.deepcopy(observed)
    result = judge(case_id, observed)
    assert result["success"] is True
    assert result["mismatches"] == {}
    assert result["observed"] == observed
    assert observed == before


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("change", ["host", "path", "http", "credentials"])
def test_url_requires_expected_https_host_and_case_path(case_id, change):
    observed = observation(case_id)
    if change == "host":
        replace_url(observed, netloc=urlsplit(observed["url"]).hostname + ".example.org")
    elif change == "path":
        replace_url(observed, path="/unrelated/")
    elif change == "http":
        replace_url(observed, scheme="http")
    else:
        replace_url(observed, netloc="user:password@" + urlsplit(observed["url"]).netloc)
    assert_rejected(case_id, observed)


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("change", ["loading", "no_main", "challenge", "blocked", "overlay"])
def test_incomplete_challenge_or_replaced_documents_fail(case_id, change):
    observed = observation(case_id)
    if change == "loading":
        observed["ready_state"] = "loading"
    elif change == "no_main":
        observed["main"] = None
    elif change == "challenge":
        observed["document_title"] = "Just a moment... Verify you are human"
    elif change == "blocked":
        observed["blocked_elements"] = [element("Access Denied: verify you are human", tag="h1")]
    else:
        observed["overlays"] = [element("Please sign in", width=1200, height=800, x=0, y=0)]
    assert_rejected(case_id, observed)


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("change", ["hidden", "zero_area", "offscreen", "occluded"])
def test_required_visible_evidence_checks_css_geometry_and_occlusion(case_id, change):
    observed = observation(case_id)
    target = required_visible(case_id, observed)
    if change == "hidden":
        target["css_visible"] = False
    elif change == "zero_area":
        target["rect"]["width"] = 0
    elif change == "offscreen":
        target["rect"]["y"] = observed["viewport"]["height"] + 1
    else:
        target["center_unobscured"] = False
    assert_rejected(case_id, observed)


@pytest.mark.parametrize(
    "parameter,wrong", [("query", "other"), ("searchtype", "all"), ("abstracts", "show")]
)
def test_arxiv_search_url_filters_are_required_and_unambiguous(parameter, wrong):
    observed = observation(SEARCH)
    values = {"query": PHRASE, "searchtype": "title", "abstracts": "hide"}
    values[parameter] = wrong
    replace_url(observed, query=urlencode(values))
    assert_rejected(SEARCH, observed)


@pytest.mark.parametrize(
    "suffix", ["&searchtype=title", "&abstracts=hide", "&query=%22Attention+Is+All+You+Need%22"]
)
def test_arxiv_search_rejects_duplicate_required_query_parameters(suffix):
    observed = observation(SEARCH)
    observed["url"] += suffix
    assert_rejected(SEARCH, observed)


@pytest.mark.parametrize(
    "change",
    [
        "stale_query",
        "wrong_field",
        "unchecked_hide",
        "wrong_action",
        "duplicate_form",
        "no_results",
        "wrong_result_id",
        "wrong_summary",
    ],
)
def test_arxiv_search_requires_applied_form_and_real_result_evidence(change):
    observed = observation(SEARCH)
    search = observed["arxiv_search"]
    form = search["forms"][0]
    if change == "stale_query":
        form["query"]["value"] = "A different search"
    elif change == "wrong_field":
        form["searchtype"]["value"] = "all"
    elif change == "unchecked_hide":
        form["hide"]["checked"] = False
    elif change == "wrong_action":
        form["action"] = "https://arxiv.org/search/advanced"
    elif change == "duplicate_form":
        other = copy.deepcopy(form)
        form["searchtype"]["value"] = "all"
        other["query"]["value"] = "A different search"
        search["forms"].append(other)
    elif change == "no_results":
        search["results"] = []
    elif change == "wrong_result_id":
        search["results"][0]["links"] = ["https://arxiv.org/abs/1706.03763"]
    else:
        search["summaries"][0]["text"] = 'Search results for Author: "Attention Is All You Need"'
    assert_rejected(SEARCH, observed)


def test_arxiv_search_does_not_freeze_result_count_or_require_other_forms_to_match():
    observed = observation(SEARCH)
    observed["arxiv_search"]["summaries"][0]["text"] = (
        'Showing 1–2 of 2 results for Title: "Attention Is All You Need"'
    )
    header_form = copy.deepcopy(observed["arxiv_search"]["forms"][0])
    header_form["action"] = "https://arxiv.org/search/header"
    header_form["query"]["value"] = ""
    header_form["searchtype"] = None
    header_form["hide"] = None
    observed["arxiv_search"]["forms"].append(header_form)
    assert judge(SEARCH, observed)["success"] is True


@pytest.mark.parametrize(
    "path",
    [
        "/pdf/1706.03762",
        "/html/1706.03762",
        "/abs/1706.03763",
        "/abs/1706.03762v0",
        "/abs/1706.03762extra",
    ],
)
def test_original_paper_rejects_related_documents_and_wrong_identifiers(path):
    observed = observation(PAPER)
    replace_url(observed, path=path)
    assert_rejected(PAPER, observed)


@pytest.mark.parametrize(
    "change",
    [
        "wrong_title",
        "wrong_author",
        "wrong_identifier",
        "wrong_year",
        "missing_submission",
        "unscoped_year",
        "empty_abstract",
        "label_only_abstract",
    ],
)
def test_original_paper_requires_scoped_identity_submission_and_abstract(change):
    observed = observation(PAPER)
    paper = observed["arxiv_paper"]
    if change == "wrong_title":
        paper["title"]["text"] = "Title: Attention Is All You Need for Vision"
    elif change == "wrong_author":
        paper["authors"]["text"] = "Authors: A Different Person"
    elif change == "wrong_identifier":
        paper["identifier"]["text"] = "arXiv:1706.03763"
    elif change == "wrong_year":
        paper["submission_metadata"][0]["text"] = (
            "Submitted on 12 Jun 2018 (v1), last revised in 2023"
        )
    elif change == "missing_submission":
        paper["submission_metadata"] = []
    elif change == "unscoped_year":
        paper["submission_metadata"][0]["text"] = "Last revised in 2023"
        observed["main"]["text"] += " Related work from 2017"
    elif change == "empty_abstract":
        paper["abstract"]["text"] = "  \n  "
    else:
        paper["abstract"]["text"] = "Abstract:"
    assert_rejected(PAPER, observed)


def test_original_paper_strips_only_the_normal_leading_title_label():
    observed = observation(PAPER)
    observed["arxiv_paper"]["title"]["text"] = "  Title:  Attention\n Is All You Need  "
    assert judge(PAPER, observed)["success"] is True
    observed["arxiv_paper"]["title"]["text"] = "Attention Is All You Need Title:"
    assert_rejected(PAPER, observed)


def test_original_paper_submission_history_v1_is_valid_with_later_revisions():
    observed = observation(PAPER)
    observed["arxiv_paper"]["submission_metadata"] = [
        {
            **element("[v1] Mon, 12 Jun 2017 17:57:34 UTC; [v7] Wed, 2 Aug 2023", y=1500),
            "kind": "submission-history",
        }
    ]
    assert judge(PAPER, observed)["success"] is True


@pytest.mark.parametrize(
    "kind,text",
    [
        ("dateline", "Submitted on 12 Jun 2016 (v1), last revised 2 Aug 2017"),
        ("submission-history", "[v1] Mon, 12 Jun 2016 17:57:34 UTC; [v2] Wed, 2 Aug 2017"),
    ],
)
def test_original_submission_year_is_not_a_later_revision_year(kind, text):
    observed = observation(PAPER)
    observed["arxiv_paper"]["submission_metadata"] = [{**element(text), "kind": kind}]
    assert_rejected(PAPER, observed)


@pytest.mark.parametrize(
    "path",
    [
        "/3/library/collections.html",
        "/3.13/library/collections.html",
        "/3.14/search.html",
        "/3.14/library/datatypes.html",
    ],
)
def test_python_requires_the_pinned_minor_version_and_reference_page(path):
    observed = observation(PYTHON)
    replace_url(observed, path=path)
    assert_rejected(PYTHON, observed)


@pytest.mark.parametrize(
    "change",
    [
        "wrong_h1",
        "missing_section",
        "wrong_section",
        "missing_method",
        "wrong_method_id",
        "wrong_signature",
        "missing_description",
        "empty_description",
        "missing_paragraph",
        "empty_paragraph",
        "paragraph_outside_viewport",
        "paragraph_occluded",
    ],
)
def test_python_requires_scoped_method_signature_and_visible_description(change):
    observed = observation(PYTHON)
    content = observed["python"]
    if change == "wrong_h1":
        observed["headings"][0]["text"] = "datatypes — Data Types"
    elif change == "missing_section":
        content["counter_section"] = None
    elif change == "wrong_section":
        content["counter_section"]["id"] = "deque-objects"
    elif change == "missing_method":
        content["method"] = None
    elif change == "wrong_method_id":
        content["method"]["id"] = "collections.Counter"
    elif change == "wrong_signature":
        content["method"]["text"] = "Counter([iterable-or-mapping])"
    elif change == "missing_description":
        content["description"] = None
    elif change == "empty_description":
        content["description"]["text"] = " "
    elif change == "missing_paragraph":
        content["first_paragraph"] = None
    elif change == "empty_paragraph":
        content["first_paragraph"]["text"] = " "
    elif change == "paragraph_outside_viewport":
        content["first_paragraph"]["rect"]["y"] = 801
    else:
        content["first_paragraph"]["intersection_unobscured"] = False
    assert_rejected(PYTHON, observed)


def test_python_first_paragraph_needs_intersection_rather_than_visible_center():
    observed = observation(PYTHON)
    paragraph = observed["python"]["first_paragraph"]
    assert paragraph["rect"]["y"] < observed["viewport"]["height"]
    assert paragraph["rect"]["y"] + paragraph["rect"]["height"] / 2 > observed["viewport"]["height"]
    assert paragraph["center_unobscured"] is False
    assert judge(PYTHON, observed)["success"] is True
    replace_url(observed, fragment="collections.Counter.most_common")
    assert judge(PYTHON, observed)["success"] is True


@pytest.mark.parametrize(
    "change",
    [
        "wrong_h1",
        "h2_instead_of_h1",
        "wrong_section",
        "wrong_heading_tag",
        "missing_phobos",
        "missing_deimos",
        "names_in_next_section",
        "no_sections",
    ],
)
def test_nasa_requires_main_h1_and_exact_moons_section_before_next_h2(change):
    observed = observation(NASA)
    sections = observed["nasa"]["sections"]
    if change == "wrong_h1":
        observed["headings"][0]["text"] = "Mars Exploration"
    elif change == "h2_instead_of_h1":
        observed["headings"][0]["tag"] = "h2"
    elif change == "wrong_section":
        sections[0]["heading"]["text"] = "Moons and Rings"
    elif change == "wrong_heading_tag":
        sections[0]["heading"]["tag"] = "a"
    elif change == "missing_phobos":
        sections[0]["content_before_next_h2"] = "The small moon Deimos orbits Mars."
    elif change == "missing_deimos":
        sections[0]["content_before_next_h2"] = "The small moon Phobos orbits Mars."
    elif change == "names_in_next_section":
        sections[0]["content_before_next_h2"] = "Two natural satellites orbit Mars."
        sections[1]["content_before_next_h2"] = "Phobos and Deimos appear in related links."
    else:
        observed["nasa"]["sections"] = []
    assert_rejected(NASA, observed)


def test_nasa_accepts_moons_reached_by_scrolling_without_a_fragment():
    observed = observation(NASA)
    replace_url(observed, fragment="")
    assert observed["headings"][0]["rect"]["y"] < 0
    assert judge(NASA, observed)["success"] is True


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_missing_required_visible_element_fails(case_id):
    observed = observation(case_id)
    if case_id == SEARCH:
        observed["arxiv_search"]["summaries"] = []
    elif case_id == PAPER:
        observed["arxiv_paper"]["title"] = None
    elif case_id == PYTHON:
        observed["python"]["method"] = None
    else:
        observed["nasa"]["sections"][0]["heading"] = None
    assert_rejected(case_id, observed)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_hidden_challenge_markers_and_offscreen_overlays_do_not_replace_article(case_id):
    observed = observation(case_id)
    marker = element("Access Denied")
    marker["css_visible"] = False
    observed["blocked_elements"] = [marker]
    observed["overlays"] = [element("Menu", y=900)]
    assert judge(case_id, observed)["success"] is True


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_nonpositive_viewport_cannot_pass_even_when_element_flags_claim_visible(case_id):
    observed = observation(case_id)
    observed["viewport"]["height"] = 0
    assert_rejected(case_id, observed)


def test_search_original_result_may_be_offscreen_when_real_results_are_in_view():
    observed = observation(SEARCH)
    original = observed["arxiv_search"]["results"][0]
    original["rect"]["y"] = 1600
    original["center_unobscured"] = False
    original["intersection_unobscured"] = False
    observed["arxiv_search"]["results"].insert(
        0,
        {
            **element("Attention Is All You Need for Another Application", tag="li", y=350),
            "title": "Attention Is All You Need for Another Application",
            "links": ["https://arxiv.org/abs/2501.12345"],
        },
    )
    assert judge(SEARCH, observed)["success"] is True
    observed["arxiv_search"]["results"].pop(0)
    assert_rejected(SEARCH, observed)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_verifier_reads_one_fresh_observation_per_call(case_id):
    class Browser:
        def __init__(self):
            self.calls = []
            self.current = observation(case_id)

        def evaluate(self, expression):
            assert isinstance(expression, str)
            self.calls.append(expression)
            return copy.deepcopy(self.current)

    browser = Browser()
    case = PUBLIC_WEB.get_public_web_case(case_id)
    assert PUBLIC_WEB.verify_public_web_case(browser, case)["success"] is True
    assert len(browser.calls) == 1
    browser.current["url"] = "https://example.org/unrelated"
    assert PUBLIC_WEB.verify_public_web_case(browser, case)["success"] is False
    assert len(browser.calls) == 2


def test_verifier_propagates_evaluation_failure_without_reusing_an_observation():
    failure = RuntimeError("fresh document evaluation failed")

    class Browser:
        calls = 0

        def evaluate(self, _expression):
            self.calls += 1
            raise failure

    browser = Browser()
    with pytest.raises(RuntimeError) as caught:
        PUBLIC_WEB.verify_public_web_case(browser, PUBLIC_WEB.get_public_web_case(NASA))
    assert caught.value is failure
    assert browser.calls == 1
