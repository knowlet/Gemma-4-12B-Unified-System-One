"""Fresh heldout judges: goal mismatches remain separate from technical errors."""

import copy
import dataclasses
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import jev_heldout_public_cases as HELDOUT
import pytest

IDS = tuple(HELDOUT.PUBLIC_WEB_CASES)


def element(text, tag="div", element_id="", y=150, height=30):
    return {
        "text": text,
        "tag": tag,
        "id": element_id,
        "css_visible": True,
        "rect": {"x": 20, "y": y, "width": 600, "height": height},
        "center_unobscured": True,
        "intersection_unobscured": True,
    }


def observation(case_id):
    case = HELDOUT.get_public_web_case(case_id)
    kind = case.judge_kind
    observed = {
        "url": "",
        "ready_state": "complete",
        "document_title": "Reference",
        "viewport": {"width": 1120, "height": 780},
        "main": element("Ordinary target document", "main", y=0, height=3000),
        "headings": [],
        "blocked_elements": [],
        "overlays": [],
    }
    if kind in HELDOUT._PATHS:
        observed["url"] = "https://" + case.allowed_top_level_hosts[0] + HELDOUT._PATHS[kind]
        identities = {
            "sqlite": element("Window Functions"),
            "curl": element("curl man page", "h1"),
            "numpy": element("numpy.reshape(a, shape=None, order='C')", "dt", "numpy.reshape"),
            "pandas": element("Series.map(arg, na_action=None)", "dt", "pandas.Series.map"),
        }
        labels = {
            "sqlite": "row_number()",
            "numpy": "order {'C', 'F', 'A'}, optional",
            "pandas": "na_action {None, ignore}",
            "curl": "--retry-max-time <seconds>",
        }
        descriptions = {
            "sqlite": "Rows in the current partition start at 1 according to ORDER BY.",
            "numpy": "Read using C-like or Fortran-like index order, with C/F/A options.",
            "pandas": "If ignore, propagate NaN values without passing them to the mapping correspondence.",
            "curl": "The retry timer is reset before the first transfer attempt.",
        }
        item = {
            "entry": element(labels[kind], "p" if kind == "curl" else "dt"),
            "description": element(descriptions[kind], "p" if kind == "curl" else "dd", y=200),
            "paragraph": element(descriptions[kind], "p", y=200),
        }
        if kind == "curl":
            item["anchor"] = "--retry-max-time"
        observed["definition"] = {"identity": identities[kind], "definitions": [item]}
    else:
        term = "asthma" if kind == "pubmed_search" else "migraine"
        filters = ["simsearch2.ffrft"] + (["pubt.clinicaltrial"] if kind == "pubmed_search" else [])
        titles = "Clinical Trial, Free full text" if kind == "pubmed_search" else "Free full text"
        observed["url"] = "https://pubmed.ncbi.nlm.nih.gov/?" + urlencode(
            {"term": term, "filter": filters}, doseq=True
        )
        observed["pubmed"] = {
            "forms": [
                {
                    "action": "https://pubmed.ncbi.nlm.nih.gov/",
                    "query": {**element("", "input"), "value": term},
                    "applied_filter_inputs": filters,
                }
            ],
            "applied_message": element("Filters applied: " + titles),
            "applied_titles": titles,
            "result_count": element("1,234", "span"),
            "citations": [
                {
                    **element("A real citation", "article", y=350, height=200),
                    "title": element("A real citation", "a", y=350),
                    "href": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                }
            ],
        }
    return observed


def judge(case_id, data):
    return HELDOUT.validate_public_web_observation(case_id, data)


def reject(case_id, data):
    result = judge(case_id, data)
    assert result["success"] is False
    assert result["mismatches"]
    return result


def test_six_fixed_cases_match_predeclared_goals_if_proposal_is_present():
    assert IDS == tuple(f"heldout-public-{i:03d}" for i in range(6))
    manifest = HELDOUT.case_manifest()
    assert json.loads(json.dumps(manifest)) == manifest
    assert all(item["selection_sha256"] == HELDOUT.SELECTION_SHA256 for item in manifest)
    proposal = Path(__file__).resolve().parents[1] / (
        "artifacts/jev-runtime/public-native-heldout-plan-20261018-01/proposal.json"
    )
    if proposal.exists():
        import hashlib

        assert hashlib.sha256(proposal.read_bytes()).hexdigest() == HELDOUT.SELECTION_SHA256
        for item, selected in zip(
            manifest, json.loads(proposal.read_bytes())["cases"], strict=True
        ):
            for key in ("case_id", "goal", "start_url", "allowed_top_level_hosts"):
                assert item[key] == selected[key]
    with pytest.raises(ValueError):
        HELDOUT.get_public_web_case("public-python-counter-most-common-v1")
    case = HELDOUT.get_public_web_case(IDS[0])
    with pytest.raises(ValueError):
        HELDOUT.case_manifest(dataclasses.replace(case, goal="changed"))


@pytest.mark.parametrize("case_id", IDS)
def test_complete_scoped_evidence_passes_without_mutation(case_id):
    data = observation(case_id)
    before = copy.deepcopy(data)
    result = judge(case_id, data)
    assert result == {"success": True, "observed": data, "mismatches": {}}
    assert data == before


@pytest.mark.parametrize("case_id", IDS)
@pytest.mark.parametrize("change", ["host", "path", "http", "credentials", "port", "control"])
def test_wrong_document_cannot_pass_even_with_all_goal_text(case_id, change):
    data = observation(case_id)
    parsed = urlsplit(data["url"])
    replacement = {
        "host": {"netloc": "example.org"},
        "path": {"path": "/wrong"},
        "http": {"scheme": "http"},
        "credentials": {"netloc": "user@" + parsed.netloc},
        "port": {"netloc": parsed.netloc + ":444"},
    }
    data["url"] = (
        data["url"] + "\n"
        if change == "control"
        else urlunsplit(parsed._replace(**replacement[change]))
    )
    reject(case_id, data)


@pytest.mark.parametrize("case_id", IDS)
@pytest.mark.parametrize("change", ["loading", "empty", "challenge", "overlay", "nan"])
def test_incomplete_or_obscured_page_is_not_goal_satisfaction(case_id, change):
    data = observation(case_id)
    if change == "loading":
        data["ready_state"] = "loading"
    elif change == "empty":
        data["main"]["text"] = ""
    elif change == "challenge":
        data["blocked_elements"] = [element("Verify you are human")]
    elif change == "overlay":
        data["overlays"] = [element("Filter dialog", "dialog")]
    else:
        data["viewport"]["width"] = float("nan")
    reject(case_id, data)


@pytest.mark.parametrize("case_id", IDS[:4])
@pytest.mark.parametrize(
    "change",
    [
        "offscreen",
        "hidden",
        "covered",
        "description_offscreen",
        "wrong_definition",
        "missing_description",
        "wrong_identity",
        "query",
    ],
)
def test_docs_require_the_actual_visible_entry_and_own_description(case_id, change):
    data = observation(case_id)
    item = data["definition"]["definitions"][0]
    if change == "offscreen":
        item["entry"]["rect"]["y"] = 9000
    elif change == "hidden":
        item["entry"]["css_visible"] = False
    elif change == "covered":
        item["entry"]["center_unobscured"] = False
    elif change == "description_offscreen":
        item["paragraph"]["rect"]["y"] = 9000
    elif change == "wrong_definition":
        item["paragraph"]["text"] = "A different parameter's explanation."
    elif change == "missing_description":
        item["description"] = None
    elif change == "query":
        data["url"] += "?search=target"
    else:
        data["definition"]["identity"]["text"] = "Unrelated document or method"
    reject(case_id, data)


@pytest.mark.parametrize("case_id", IDS[4:])
@pytest.mark.parametrize(
    "change",
    [
        "typed_only",
        "wrong_field",
        "missing_filter",
        "extra_filter",
        "duplicate_filter",
        "pending_filters",
        "wrong_banner",
        "no_banner",
        "zero_results",
        "wrong_citation",
        "offscreen_results",
    ],
)
def test_search_requires_submitted_query_applied_filters_and_real_results(case_id, change):
    data = observation(case_id)
    kind = data["pubmed"]
    parsed = urlsplit(data["url"])
    from urllib.parse import parse_qs

    query = parse_qs(parsed.query)
    if change == "typed_only":
        query["term"] = ["asthma" if case_id == IDS[5] else "unsubmitted"]
    elif change == "wrong_field":
        kind["forms"][0]["query"]["value"] = "unsubmitted edit"
    elif change == "missing_filter":
        query["filter"] = []
    elif change == "extra_filter":
        query["filter"].append("pubt.review")
    elif change == "duplicate_filter":
        query["filter"].append(query["filter"][0])
    elif change == "pending_filters":
        kind["forms"][0]["applied_filter_inputs"] = []
    elif change == "wrong_banner":
        kind["applied_titles"] = (
            "Clinical Trial" if case_id == IDS[4] else "Clinical Trial, Free full text"
        )
    elif change == "no_banner":
        kind["applied_message"]["css_visible"] = False
    elif change == "zero_results":
        kind["result_count"]["text"] = "0"
    elif change == "wrong_citation":
        kind["citations"][0]["href"] = "https://pubmed.ncbi.nlm.nih.gov/help/"
    else:
        kind["citations"][0]["rect"]["y"] = 9000
    data["url"] = urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))
    reject(case_id, data)


@pytest.mark.parametrize("case_id", IDS)
def test_unchanged_start_url_cannot_pass_final_goal(case_id):
    data = observation(case_id)
    data["url"] = HELDOUT.get_public_web_case(case_id).start_url
    reject(case_id, data)


def test_fresh_evaluate_once_no_cached_snapshot_and_technical_errors_propagate():
    class Browser:
        def __init__(self):
            self.current = observation(IDS[0])
            self.calls = []

        def evaluate(self, expression):
            self.calls.append(expression)
            if self.current is None:
                raise RuntimeError("fresh DOM unavailable")
            return self.current

    browser = Browser()
    assert HELDOUT.verify_public_web_case(browser, IDS[0])["success"]
    browser.current["url"] = "https://example.org/"
    assert not HELDOUT.verify_public_web_case(browser, IDS[0])["success"]
    browser.current = None
    with pytest.raises(RuntimeError, match="fresh DOM unavailable"):
        HELDOUT.verify_public_web_case(browser, IDS[0])
    assert len(browser.calls) == 3
    for case_id in IDS:
        expression = HELDOUT.observation_expression(case_id)
        assert "elementFromPoint" in expression
        assert "nextElementSibling" in expression
        for forbidden in ("scrollIntoView", ".click(", "dispatchEvent", "__jev", "location.href ="):
            assert forbidden not in expression


def diagnostic_html(case_id):
    """Small HTML mirrors inspected official structure, not a policy or model test."""
    observed = observation(case_id)
    kind = HELDOUT.get_public_web_case(case_id).judge_kind
    if kind in ("sqlite", "curl"):
        identity = (
            '<div class="fancy_title">Window Functions</div>'
            if kind == "sqlite"
            else "<h1>curl man page</h1>"
        )
        entry = (
            (
                '<dt id="target"><p><b>row_number()</b></p></dt><dd><p>'
                + observed["definition"]["definitions"][0]["paragraph"]["text"]
                + "</p></dd>"
            )
            if kind == "sqlite"
            else (
                '<p class="level0" id="target"><a name="--retry-max-time"></a>'
                '<span>--retry-max-time &lt;seconds&gt;</span></p><p class="level1">'
                + observed["definition"]["definitions"][0]["paragraph"]["text"]
                + "</p>"
            )
        )
        body = (
            '<div class="'
            + ("fancy" if kind == "sqlite" else "contents")
            + '">'
            + identity
            + "<dl>"
            + entry
            + "</dl></div>"
        )
    elif kind in ("numpy", "pandas"):
        identity = "numpy.reshape" if kind == "numpy" else "pandas.Series.map"
        label = "order" if kind == "numpy" else "na_action"
        signature = observed["definition"]["identity"]["text"]
        text = observed["definition"]["definitions"][0]["paragraph"]["text"]
        body = f'<main><h1>{identity}</h1><dl><dt id="{identity}">{signature}</dt><dd><dl class="field-list"><dt>Parameters</dt><dd><dl><dt id="target"><strong>{label}</strong></dt><dd><p>{text}</p></dd></dl></dd></dl></dd></dl></main>'
    else:
        evidence = observed["pubmed"]
        form = evidence["forms"][0]
        inputs = "".join(
            f'<input type="hidden" name="filter" value="{value}">'
            for value in form["applied_filter_inputs"]
        )
        body = f'<main><h1>Search results</h1><form action="/"><input name="term" value="{form["query"]["value"]}"><div class="selected-filters">{inputs}</div></form><div class="results-amount"><span class="value">1,234</span></div><section class="search-results-list"><div class="applied-filters-message">Filters applied: <span class="filter-titles">{evidence["applied_titles"]}</span></div><article class="full-docsum" id="target"><a class="docsum-title" href="/12345678/">A real citation</a></article></section></main>'
    return (
        "<!doctype html><style>body{font:18px sans-serif;margin:20px}#target{margin-top:"
        + ("1200px" if kind in HELDOUT._PATHS else "20px")
        + ";}p{max-width:650px}</style>"
        + body
    )


@pytest.fixture(scope="module")
def chromium():
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for CPU-only local HTML Chromium checks")
    playwright = pytest.importorskip("playwright.sync_api")
    path = os.environ.get(
        "JEV_CHROMIUM_EXECUTABLE",
        str(
            Path.home()
            / (".cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell")
        ),
    )
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=path, headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.mark.parametrize("case_id", IDS)
def test_real_chromium_fresh_js_uses_owned_definition_and_never_scrolls(chromium, case_id):
    page = chromium.new_page(viewport={"width": 1120, "height": 780})
    try:
        url = observation(case_id)["url"]
        page.route(
            "**/*",
            lambda route: route.fulfill(content_type="text/html", body=diagnostic_html(case_id)),
        )
        page.goto(url)
        if case_id in IDS[:4]:
            assert not HELDOUT.verify_public_web_case(page, case_id)["success"]
            page.locator("#target").scroll_into_view_if_needed()
        before = page.content(), page.evaluate("scrollY")
        assert HELDOUT.verify_public_web_case(page, case_id)["success"]
        assert (page.content(), page.evaluate("scrollY")) == before
        page.locator("#target").evaluate("node => node.style.display = 'none'")
        assert not HELDOUT.verify_public_web_case(page, case_id)["success"]
    finally:
        page.close()
