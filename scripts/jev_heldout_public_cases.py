"""Predeclared public transfer tasks, independent of serving and training.

Selection/exposure evidence: artifacts/jev-runtime/public-native-heldout-plan-20261018-01.
These cases are evaluation-only. The pinned original module supplies common pure
DOM visibility helpers; its four exposed case definitions are never selected here.
"""

from __future__ import annotations

from dataclasses import asdict
from urllib.parse import parse_qs

from jev_public_web_cases import (
    _BLOCKED_TEXT,
    _READ_COMMON_JS,
    PublicWebCase,
    _finite_number,
    _in_view,
    _mapping,
    _rendered,
    _sequence,
    _text,
    _url,
)

SELECTION_SHA256 = "4596e56d2f769f283ae606e42390236e82a2a7bbb59199a815f5ed2f7b13881c"
SCOPE = (
    "predeclared websites absent from audited browser-adaptation artifacts; "
    "base pretraining exposure unknown"
)
PUBLIC_WEB_CASES = {
    case.case_id: case
    for case in (
        PublicWebCase(
            "heldout-public-000",
            "Open the SQLite window functions reference, then bring the row_number() built-in "
            "function entry and its description into view. Stop there.",
            "https://www.sqlite.org/docs.html",
            ("www.sqlite.org", "sqlite.org"),
            "sqlite",
        ),
        PublicWebCase(
            "heldout-public-001",
            "In the NumPy 2.3 documentation, open numpy.reshape and bring the order parameter "
            "and its description of C and Fortran indexing into view. Stop with that parameter visible.",
            "https://numpy.org/doc/2.3/reference/routines.array-manipulation.html",
            ("numpy.org",),
            "numpy",
        ),
        PublicWebCase(
            "heldout-public-002",
            "Open the curl command-line man page and bring the --retry-max-time option heading "
            "and its explanation into view. Stop on that option.",
            "https://curl.se/docs/",
            ("curl.se",),
            "curl",
        ),
        PublicWebCase(
            "heldout-public-003",
            "In the pandas 2.3 Series reference, open Series.map and bring the na_action "
            "parameter and the explanation of ignore into view. Stop there.",
            "https://pandas.pydata.org/pandas-docs/version/2.3/reference/series.html",
            ("pandas.pydata.org",),
            "pandas",
        ),
        PublicWebCase(
            "heldout-public-004",
            "Search PubMed for asthma. Apply both the Clinical Trial and Free full text "
            "sidebar filters, then leave the filtered search results open.",
            "https://pubmed.ncbi.nlm.nih.gov/",
            ("pubmed.ncbi.nlm.nih.gov",),
            "pubmed_search",
        ),
        PublicWebCase(
            "heldout-public-005",
            "Change the PubMed search to migraine. Keep Free full text enabled, remove the "
            "Clinical Trial filter, and leave the resulting search results open.",
            "https://pubmed.ncbi.nlm.nih.gov/?term=asthma&filter=pubt.clinicaltrial"
            "&filter=simsearch2.ffrft",
            ("pubmed.ncbi.nlm.nih.gov",),
            "pubmed_change",
        ),
    )
}


def get_public_web_case(case_id: str) -> PublicWebCase:
    try:
        return PUBLIC_WEB_CASES[case_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown heldout public-web case: {case_id!r}") from exc


def _case(case):
    if isinstance(case, str):
        return get_public_web_case(case)
    if not isinstance(case, PublicWebCase) or PUBLIC_WEB_CASES.get(case.case_id) != case:
        raise ValueError("A fixed heldout public-web case is required")
    return case


def case_manifest(case=None):
    if case is None:
        return [case_manifest(value) for value in PUBLIC_WEB_CASES.values()]
    result = asdict(_case(case))
    result["allowed_top_level_hosts"] = list(result["allowed_top_level_hosts"])
    return {**result, "selection_sha256": SELECTION_SHA256, "scope": SCOPE}


_DEFINITIONS_JS = r"""
  const norm = node => text(node).replace(/\s+/g, ' ').trim();
  const outsideNavigation = node => !!node && !node.closest('nav,aside,header,footer');
  const definition = entry => {
    const description = entry?.nextElementSibling?.tagName === 'DD'
      ? entry.nextElementSibling : null;
    return {entry:element(entry), description:element(description),
      paragraph:element(description?.querySelector('p'))};
  };
"""
_READ_CASE_JS = {
    "sqlite": r"""
  result.definition = {identity:element(main?.querySelector('.fancy_title')),
    definitions:main ? [...main.querySelectorAll('dt')].filter(node =>
      outsideNavigation(node) && norm(node) === 'row_number()').map(definition) : []};
""",
    "numpy": r"""
  const api = main?.querySelector('dt[id="numpy.reshape"]');
  const body = api?.nextElementSibling?.tagName === 'DD' ? api.nextElementSibling : null;
  result.definition = {identity:element(api), definitions:body ?
    [...body.querySelectorAll('dl.field-list > dd > dl > dt')].filter(node =>
      norm(node.querySelector('strong')) === 'order').map(definition) : []};
""",
    "pandas": r"""
  const api = main?.querySelector('dt[id="pandas.Series.map"]');
  const body = api?.nextElementSibling?.tagName === 'DD' ? api.nextElementSibling : null;
  result.definition = {identity:element(api), definitions:body ?
    [...body.querySelectorAll('dl.field-list > dd > dl > dt')].filter(node =>
      norm(node.querySelector('strong')) === 'na_action').map(definition) : []};
""",
    "curl": r"""
  const entry = main?.querySelector('p.level0:has(a[name="--retry-max-time"])');
  const paragraph = entry?.nextElementSibling?.matches('p.level1')
    ? entry.nextElementSibling : null;
  result.definition = {identity:element(main?.querySelector('h1')),
    definitions:entry ? [{entry:element(entry), description:element(paragraph),
      paragraph:element(paragraph), anchor:entry.querySelector('a[name]')?.name}] : []};
""",
}
_PUBMED_JS = r"""
  result.pubmed = {
    forms:main ? [...main.querySelectorAll('form')].filter(form =>
      form.querySelector('input[name="term"]')).map(form => ({action:form.action,
        query:control(form.querySelector('input[name="term"]')),
        applied_filter_inputs:[...form.querySelectorAll(
          '.selected-filters input[type="hidden"][name="filter"]')].map(node => node.value)
      })) : [],
    applied_message:element(main?.querySelector('.search-results-list .applied-filters-message')),
    applied_titles:text(main?.querySelector('.search-results-list .filter-titles')),
    result_count:element(main?.querySelector('.results-amount .value')),
    citations:main ? [...main.querySelectorAll('.search-results-list article.full-docsum')]
      .map(node => ({...element(node),
        title:element(node.querySelector('a.docsum-title[href]')),
        href:node.querySelector('a.docsum-title[href]')?.href})) : []
  };
"""
_PATHS = {
    "sqlite": "/windowfunctions.html",
    "numpy": "/doc/2.3/reference/generated/numpy.reshape.html",
    "curl": "/docs/manpage.html",
    "pandas": "/pandas-docs/version/2.3/reference/api/pandas.Series.map.html",
}


def observation_expression(case):
    """One synchronous read of the current document; no snapshot cache or writes."""
    case = _case(case)
    selector = {"sqlite": ".fancy", "curl": ".contents"}.get(
        case.judge_kind, 'main, [role="main"], #content'
    )
    common = _READ_COMMON_JS.replace(
        "document.querySelector('main, [role=\"main\"], #content')",
        f"document.querySelector('{selector}')",
        1,
    )
    extra = _PUBMED_JS if case.judge_kind.startswith("pubmed_") else _READ_CASE_JS[case.judge_kind]
    return common + _DEFINITIONS_JS + extra + "return result;\n})()"


def validate_public_web_observation(case, observed):
    """Independent final goal satisfaction; the unchanged runner additionally requires DONE."""
    case = _case(case)
    data, mismatches = _mapping(observed), {}

    def check(key, condition, expected, actual=None):
        if not condition:
            mismatches[key] = {"expected": expected, "observed": actual}

    viewport = _mapping(data.get("viewport"))
    check(
        "viewport",
        all(_finite_number(viewport.get(k)) and viewport[k] > 0 for k in ("width", "height")),
        "positive finite viewport",
        viewport,
    )
    check(
        "ready_state",
        data.get("ready_state") in ("interactive", "complete"),
        "loaded DOM",
        data.get("ready_state"),
    )
    main = _mapping(data.get("main"))
    check(
        "main",
        _rendered(main) and bool(_text(main.get("text"))),
        "nonempty rendered target content",
        main,
    )
    headings = _sequence(data.get("headings"))
    check("headings", isinstance(data.get("headings"), list), "fresh heading list", headings)
    for key in ("blocked_elements", "overlays"):
        values = data.get(key)
        check(
            key,
            isinstance(values, list)
            and not any(_in_view(item, viewport, center=False) for item in _sequence(values)),
            "no visible challenge/login/overlay",
            values,
        )
    status_text = " ".join(
        [
            _text(data.get("document_title")),
            *[_text(_mapping(h).get("text")) for h in headings if _rendered(h)],
            _text(main.get("text"))[:2000],
        ]
    )
    check(
        "page_status",
        not _BLOCKED_TEXT.search(status_text),
        "ordinary target page",
        status_text[:300],
    )
    parsed = _url(data.get("url"), case.allowed_top_level_hosts)
    check("url", parsed is not None, "exact allowed HTTPS host", data.get("url"))
    if case.judge_kind in _PATHS:
        check(
            "path",
            bool(parsed and parsed.path == _PATHS[case.judge_kind] and not parsed.query),
            _PATHS[case.judge_kind] + " without query",
            data.get("url"),
        )
        definition = _mapping(data.get("definition"))
        identity = _mapping(definition.get("identity"))
        text = _text(identity.get("text"))
        identity_ok = {
            "sqlite": identity.get("tag") == "div" and text == "Window Functions",
            "curl": identity.get("tag") == "h1" and text == "curl man page",
            "numpy": identity.get("tag") == "dt"
            and identity.get("id") == "numpy.reshape"
            and "numpy.reshape(" in "".join(text.split()),
            "pandas": identity.get("tag") == "dt"
            and identity.get("id") == "pandas.Series.map"
            and "Series.map(" in "".join(text.split()),
        }[case.judge_kind]
        check(
            "document_identity",
            _rendered(identity) and identity_ok,
            "rendered exact document/API identity",
            identity,
        )
        matches = []
        for item in _sequence(definition.get("definitions")):
            item = _mapping(item)
            entry, description, paragraph = [
                _mapping(item.get(k)) for k in ("entry", "description", "paragraph")
            ]
            label = _text(entry.get("text"))
            explanation = _text(paragraph.get("text")).casefold()
            if case.judge_kind == "sqlite":
                content_ok = label == "row_number()" and all(
                    value in explanation for value in ("current partition", "order by", "1")
                )
            elif case.judge_kind == "numpy":
                content_ok = label.startswith("order") and all(
                    value in explanation for value in ("c-like", "fortran-like", "index order")
                )
            elif case.judge_kind == "pandas":
                content_ok = label.startswith("na_action") and all(
                    value in explanation
                    for value in ("ignore", "nan", "without passing", "mapping")
                )
            else:
                content_ok = (
                    label == "--retry-max-time <seconds>"
                    and item.get("anchor") == "--retry-max-time"
                    and all(
                        value in explanation for value in ("retry timer", "first transfer attempt")
                    )
                )
            expected_tag = "p" if case.judge_kind == "curl" else "dt"
            if (
                content_ok
                and entry.get("tag") == expected_tag
                and _in_view(entry, viewport)
                and description.get("tag") == ("p" if case.judge_kind == "curl" else "dd")
                and _rendered(description)
                and paragraph.get("tag") == "p"
                and _in_view(paragraph, viewport, center=False)
            ):
                matches.append(item)
        check(
            "visible_definition",
            bool(matches),
            "exact entry and its own explanation in view",
            definition.get("definitions"),
        )
    else:
        _validate_pubmed(case, data, parsed, viewport, check)
    return {"success": not mismatches, "observed": observed, "mismatches": mismatches}


def _validate_pubmed(case, data, parsed, viewport, check):
    expected_term = "asthma" if case.judge_kind == "pubmed_search" else "migraine"
    expected_filters = {"simsearch2.ffrft"}
    expected_titles = {"Free full text"}
    if case.judge_kind == "pubmed_search":
        expected_filters.add("pubt.clinicaltrial")
        expected_titles.add("Clinical Trial")
    query = parse_qs(parsed.query, keep_blank_values=True) if parsed else {}
    terms, filters = query.get("term", []), query.get("filter", [])
    check("path", bool(parsed and parsed.path == "/"), "PubMed search root", data.get("url"))
    check(
        "submitted_term",
        len(terms) == 1 and terms[0].strip().casefold() == expected_term,
        expected_term,
        terms,
    )
    check(
        "applied_filter_url",
        len(filters) == len(expected_filters) and set(filters) == expected_filters,
        sorted(expected_filters),
        filters,
    )
    kind = _mapping(data.get("pubmed"))
    matching_forms = []
    for form in _sequence(kind.get("forms")):
        form = _mapping(form)
        action = _url(form.get("action"), case.allowed_top_level_hosts)
        control = _mapping(form.get("query"))
        applied = _sequence(form.get("applied_filter_inputs"))
        if (
            action
            and action.path == "/"
            and not action.query
            and control.get("tag") == "input"
            and _rendered(control)
            and _text(control.get("value")).casefold() == expected_term
            and len(applied) == len(expected_filters)
            and set(applied) == expected_filters
        ):
            matching_forms.append(form)
    check(
        "applied_search_form",
        bool(matching_forms),
        "one rendered search form agrees with submitted term and applied filters",
        kind.get("forms"),
    )
    message = _mapping(kind.get("applied_message"))
    titles = [value.strip() for value in _text(kind.get("applied_titles")).split(",")]
    check(
        "applied_filter_ui",
        _rendered(message)
        and "Filters applied:" in _text(message.get("text"))
        and len(titles) == len(expected_titles)
        and set(titles) == expected_titles,
        "rendered results-level applied filter banner with exact labels",
        message,
    )
    count = _mapping(kind.get("result_count"))
    digits = _text(count.get("text")).replace(",", "")
    check(
        "result_count",
        _rendered(count) and digits.isascii() and digits.isdigit() and int(digits) > 0,
        "positive rendered result count",
        count,
    )
    citations = []
    for citation in _sequence(kind.get("citations")):
        citation = _mapping(citation)
        link = _url(citation.get("href"), case.allowed_top_level_hosts)
        title = _mapping(citation.get("title"))
        if (
            link
            and link.path.strip("/").isascii()
            and link.path.strip("/").isdigit()
            and not link.query
            and citation.get("tag") == "article"
            and _in_view(citation, viewport, center=False)
            and _rendered(title)
            and bool(_text(title.get("text")))
        ):
            citations.append(citation)
    check(
        "citation_results",
        bool(citations),
        "ordinary citation results in view",
        kind.get("citations"),
    )


def verify_public_web_case(browser, case):
    case = _case(case)
    return validate_public_web_observation(case, browser.evaluate(observation_expression(case)))
