"""Fixed read-only public-web goals and independent fresh-DOM judges.

This module uses only the standard library. Its observations and expected values
belong to external verification, never to an agent prompt or action policy. These
four cases do not establish public-web task performance or browser readiness.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class PublicWebCase:
    case_id: str
    goal: str
    start_url: str
    allowed_top_level_hosts: tuple[str, ...]
    judge_kind: str


PUBLIC_WEB_CASES = {
    case.case_id: case
    for case in (
        PublicWebCase(
            "public-arxiv-title-filter-results-v1",
            'Search arXiv for the exact phrase "Attention Is All You Need" in the Title field, '
            "choose Hide abstracts, and leave the search results open.",
            "https://arxiv.org/search/",
            ("arxiv.org",),
            "arxiv_search",
        ),
        PublicWebCase(
            "public-arxiv-original-paper-open-v1",
            "Find and open the arXiv abstract page for the original 2017 paper Attention Is All "
            "You Need by Ashish Vaswani and colleagues. Stop with the abstract page open.",
            "https://arxiv.org/search/",
            ("arxiv.org",),
            "arxiv_paper",
        ),
        PublicWebCase(
            "public-python-counter-most-common-v1",
            "In the Python 3.14 documentation, open the collections reference and bring the "
            "Counter.most_common() method signature and description into view.",
            "https://docs.python.org/3.14/library/datatypes.html",
            ("docs.python.org",),
            "python",
        ),
        PublicWebCase(
            "public-nasa-mars-facts-moons-v1",
            "Open NASA Science's Mars Facts page and bring its Moons section into view. "
            "Stop on that section.",
            "https://science.nasa.gov/solar-system/",
            ("science.nasa.gov",),
            "nasa",
        ),
    )
}


def get_public_web_case(case_id: str) -> PublicWebCase:
    try:
        return PUBLIC_WEB_CASES[case_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown public-web case: {case_id!r}") from exc


def _case(case: PublicWebCase | str) -> PublicWebCase:
    if isinstance(case, str):
        return get_public_web_case(case)
    if not isinstance(case, PublicWebCase) or PUBLIC_WEB_CASES.get(case.case_id) != case:
        raise ValueError("A fixed public-web case is required")
    return case


def case_manifest(case: PublicWebCase | str | None = None) -> dict | list[dict]:
    """Return independent JSON-safe configuration, without policy annotations."""
    if case is None:
        return [case_manifest(value) for value in PUBLIC_WEB_CASES.values()]
    result = asdict(_case(case))
    result["allowed_top_level_hosts"] = list(result["allowed_top_level_hosts"])
    return result


# Every expression below only reads the current document. In particular, it does
# not scroll, navigate, dispatch events, assign DOM properties, or consult the
# upstream snapshot's cached node map. The same evaluation reads URL and content.
_READ_COMMON_JS = r"""
(() => {
  const main = document.querySelector('main, [role="main"], #content');
  const text = element => element?.innerText ?? '';
  const cssVisible = element => {
    if (!element || !element.isConnected) return false;
    if (typeof element.checkVisibility === 'function' &&
        !element.checkVisibility({checkOpacity:true, checkVisibilityCSS:true})) return false;
    for (let node=element; node; node=node.parentElement) {
      const style = getComputedStyle(node);
      if (node.hidden || style.display === 'none' || style.visibility === 'hidden' ||
          style.visibility === 'collapse' || style.contentVisibility === 'hidden' ||
          Number(style.opacity) === 0) return false;
    }
    return true;
  };
  const unobscured = (element, x, y) => {
    if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return false;
    const hit = document.elementFromPoint(x, y);
    return !!hit && (hit === element || element.contains(hit) || hit.contains(element));
  };
  const element = node => {
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    const left = Math.max(0, rect.left), top = Math.max(0, rect.top);
    const right = Math.min(innerWidth, rect.right), bottom = Math.min(innerHeight, rect.bottom);
    return {tag:node.tagName.toLowerCase(), id:node.id, text:text(node),
      css_visible:cssVisible(node), rect:{x:rect.left,y:rect.top,width:rect.width,height:rect.height},
      center_unobscured:unobscured(node, rect.left+rect.width/2, rect.top+rect.height/2),
      intersection_unobscured:right>left && bottom>top &&
        unobscured(node, (left+right)/2, (top+bottom)/2)};
  };
  const control = node => node ? {...element(node), value:node.value,
    checked:node.checked === true} : null;
  const headings = main ? [...main.querySelectorAll('h1,h2')].map(element) : [];
  const blocked = '#challenge-form,#cf-challenge-running,#cf-chl-widget,' +
    '[id^="cf-chl-"],.cf-error-details,.g-recaptcha,.h-captcha,' +
    'iframe[src*="recaptcha"][title*="challenge"],' +
    'iframe[src*="hcaptcha"][title*="challenge"],input[type="password"]';
  const result = {url:location.href, document_title:document.title,
    ready_state:document.readyState, viewport:{width:innerWidth,height:innerHeight},
    main:element(main), headings,
    blocked_elements:[...document.querySelectorAll(blocked)].map(element),
    overlays:[...document.querySelectorAll(
      '[role="dialog"],dialog[open],.modal.is-active,.modal.show')].map(element)};
"""

_READ_CASE_JS = {
    "arxiv_search": r"""
  result.arxiv_search = {
    forms:main ? [...main.querySelectorAll('form')].filter(form =>
      form.querySelector('input[name="query"]') && form.querySelector('select[name="searchtype"]')
    ).map(form => ({action:form.action,
      query:control(form.querySelector('input[name="query"]')),
      searchtype:control(form.querySelector('select[name="searchtype"]')),
      hide:control(form.querySelector('input[name="abstracts"][value="hide"]'))})) : [],
    summaries:headings,
    results:main ? [...main.querySelectorAll('li.arxiv-result')].map(item => ({
      ...element(item), title:text(item.querySelector('p.title')),
      links:[...item.querySelectorAll('p.list-title a[href]')].map(link => link.href)
    })) : []
  };
""",
    "arxiv_paper": r"""
  result.arxiv_paper = {
    title:element(main?.querySelector('h1.title')),
    authors:element(main?.querySelector('.authors')),
    identifier:element(main?.querySelector('.arxivid')),
    submission_metadata:main ? [...main.querySelectorAll('.dateline,.submission-history')]
      .map(node => ({...element(node),
        kind:node.classList.contains('dateline') ? 'dateline' : 'submission-history'})) : [],
    abstract:element(main?.querySelector('blockquote.abstract'))
  };
""",
    "python": r"""
  const method = main?.querySelector('dt[id="collections.Counter.most_common"]');
  const description = method?.nextElementSibling?.tagName === 'DD' ? method.nextElementSibling : null;
  result.python = {
    counter_section:element(main?.querySelector('section#counter-objects')),
    method:element(method), description:element(description),
    first_paragraph:element(description?.querySelector('p'))
  };
""",
    "nasa": r"""
  const sectionText = heading => {
    const walker = document.createTreeWalker(main, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let active = false, node;
    const parts = [];
    while ((node=walker.nextNode())) {
      if (node === heading) { active=true; continue; }
      if (!active) continue;
      if (node.nodeType === Node.ELEMENT_NODE && node.tagName === 'H2') break;
      if (node.nodeType !== Node.TEXT_NODE || heading.contains(node)) continue;
      const parent = node.parentElement;
      if (parent && !parent.closest('nav,aside,header,footer,script,style') && cssVisible(parent) &&
          parent.getClientRects().length) parts.push(node.textContent);
    }
    return parts.join(' ');
  };
  result.nasa = {sections:main ? [...main.querySelectorAll('h2')].map(heading => ({
    heading:element(heading), content_before_next_h2:sectionText(heading)
  })) : []};
""",
}


def _text(value) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _sequence(value) -> list:
    return value if isinstance(value, list) else []


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _rendered(element) -> bool:
    element = _mapping(element)
    rect = _mapping(element.get("rect"))
    return (
        element.get("css_visible") is True
        and all(_finite_number(rect.get(key)) for key in ("x", "y", "width", "height"))
        and rect["width"] > 0
        and rect["height"] > 0
    )


def _in_view(element, viewport, *, center=True) -> bool:
    if not _rendered(element):
        return False
    viewport = _mapping(viewport)
    if not all(
        _finite_number(viewport.get(key)) and viewport[key] > 0 for key in ("width", "height")
    ):
        return False
    rect = element["rect"]
    if center:
        return (
            0 <= rect["x"] + rect["width"] / 2 < viewport["width"]
            and 0 <= rect["y"] + rect["height"] / 2 < viewport["height"]
            and element.get("center_unobscured") is True
        )
    return (
        rect["x"] < viewport["width"]
        and rect["x"] + rect["width"] > 0
        and rect["y"] < viewport["height"]
        and rect["y"] + rect["height"] > 0
        and element.get("intersection_unobscured") is True
    )


def _url(value, hosts):
    if not isinstance(value, str) or value != value.strip() or re.search(r"[\x00-\x20\x7f]", value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            return None
        return parsed
    except ValueError:
        return None


def _paper_url(value) -> bool:
    parsed = _url(value, ("arxiv.org",))
    return bool(
        parsed
        and not parsed.query
        and re.fullmatch(r"/abs/1706\.03762(?:v[1-9][0-9]*)?/?", parsed.path)
    )


_BLOCKED_TEXT = re.compile(
    r"\b(?:access denied|request blocked|robot check|verify (?:that )?you are human|"
    r"checking your browser|enable javascript and cookies|just a moment|service unavailable|"
    r"temporarily unavailable|under maintenance|page not found|404 not found|403 forbidden|"
    r"internal server error|sign in to continue|log in to continue)\b",
    re.IGNORECASE,
)


def validate_public_web_observation(case: PublicWebCase | str, observed: dict) -> dict:
    """Judge a structured fresh DOM read; never use a goal claim or action history.

    Rectangles are viewport-relative. Required center visibility and paragraph
    intersection are recomputed here, rather than accepting a single pass flag.
    CSS-rendered content outside the viewport is allowed only where the case
    explicitly requires DOM presence (for example the original paper result).
    """
    case = _case(case)
    data, mismatches = _mapping(observed), {}

    def check(key, condition, expected, actual=None):
        if not condition:
            mismatches[key] = {"expected": expected, "observed": actual}

    viewport = _mapping(data.get("viewport"))
    check(
        "viewport",
        all(_finite_number(viewport.get(k)) and viewport[k] > 0 for k in ("width", "height")),
        "positive finite viewport dimensions",
        data.get("viewport"),
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
        "nonempty rendered main content",
        data.get("main"),
    )
    headings = _sequence(data.get("headings"))
    check(
        "headings",
        isinstance(data.get("headings"), list),
        "main-content heading observations",
        data.get("headings"),
    )
    for key in ("blocked_elements", "overlays"):
        values = data.get(key)
        check(key + "_observation", isinstance(values, list), "fresh DOM marker list", values)
        visible = [item for item in _sequence(values) if _in_view(item, viewport, center=False)]
        check(key, not visible, "no visible access/login challenge or blocking overlay", visible)
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
        "ordinary target document",
        status_text[:300],
    )
    parsed = _url(data.get("url"), case.allowed_top_level_hosts)
    check("url", parsed is not None, "HTTPS URL on the case's exact allowed host", data.get("url"))
    kind = _mapping(data.get(case.judge_kind))
    if case.judge_kind == "arxiv_search":
        phrase = '"Attention Is All You Need"'
        query = parse_qs(parsed.query, keep_blank_values=True) if parsed else {}
        expected_query = {"query": [phrase], "searchtype": ["title"], "abstracts": ["hide"]}
        check(
            "path",
            bool(parsed and parsed.path.removesuffix("/") == "/search"),
            "/search/",
            parsed.path if parsed else None,
        )
        check(
            "query",
            all(query.get(k) == v for k, v in expected_query.items())
            and set(query) <= {"query", "searchtype", "abstracts", "order", "size"}
            and all(len(values) == 1 and bool(values[0]) for values in query.values()),
            "quoted title phrase, title field, hidden abstracts; no duplicate filters",
            query,
        )
        forms = []
        for form in _sequence(kind.get("forms")):
            form = _mapping(form)
            action = _url(form.get("action"), case.allowed_top_level_hosts)
            controls = [_mapping(form.get(key)) for key in ("query", "searchtype", "hide")]
            if (
                action
                and action.path.removesuffix("/") == "/search"
                and not action.query
                and all(_in_view(control, viewport) for control in controls)
                and controls[0].get("tag") == "input"
                and controls[0].get("value") == phrase
                and controls[1].get("tag") == "select"
                and controls[1].get("value") == "title"
                and controls[2].get("tag") == "input"
                and controls[2].get("value") == "hide"
                and controls[2].get("checked") is True
            ):
                forms.append(form)
        check(
            "applied_form",
            bool(forms),
            "one coherent visible main search form with all requested values",
            kind.get("forms"),
        )
        summaries = [
            item
            for item in _sequence(kind.get("summaries"))
            if (
                _mapping(item).get("tag") in ("h1", "h2")
                and _in_view(item, viewport)
                and "attention is all you need" in _text(item.get("text")).casefold()
                and re.search(r"\btitle\b", _text(item.get("text")), re.IGNORECASE)
                and re.search(r"\bresults?\b", _text(item.get("text")), re.IGNORECASE)
            )
        ]
        check(
            "result_summary",
            bool(summaries),
            "visible result heading identifying this title search",
            kind.get("summaries"),
        )
        results = [
            item
            for item in _sequence(kind.get("results"))
            if (
                _rendered(item)
                and _mapping(item).get("tag") == "li"
                and _text(item.get("title"))
                and _text(item.get("text"))
            )
        ]
        check(
            "results",
            any(_in_view(item, viewport, center=False) for item in results),
            "nonempty rendered result list in view",
            kind.get("results"),
        )
        check(
            "original_paper_result",
            any(_paper_url(link) for item in results for link in _sequence(item.get("links"))),
            "a real result links to arXiv:1706.03762",
            [item.get("links") for item in results],
        )
    elif case.judge_kind == "arxiv_paper":
        check(
            "paper_url",
            _paper_url(data.get("url")),
            "the exact arXiv abstract ID, optionally versioned",
            data.get("url"),
        )
        title = _mapping(kind.get("title"))
        title_text = re.sub(r"^Title:\s*", "", _text(title.get("text")))
        check(
            "paper_title",
            title.get("tag") == "h1"
            and _in_view(title, viewport)
            and title_text == "Attention Is All You Need",
            "exact original title heading in view",
            title,
        )
        authors = _mapping(kind.get("authors"))
        check(
            "paper_authors",
            _rendered(authors) and "Ashish Vaswani" in _text(authors.get("text")),
            "rendered Ashish Vaswani author entry",
            authors,
        )
        identifier = _mapping(kind.get("identifier"))
        check(
            "paper_identifier",
            _rendered(identifier)
            and bool(
                re.search(
                    r"\barXiv:\s*1706\.03762(?:v[1-9][0-9]*)?\b", _text(identifier.get("text"))
                )
            ),
            "rendered arXiv:1706.03762 identifier",
            identifier,
        )
        original_year = False
        for metadata in _sequence(kind.get("submission_metadata")):
            metadata = _mapping(metadata)
            pattern = (
                r"\bSubmitted on \d{1,2} [A-Za-z]+ 2017\b"
                if metadata.get("kind") == "dateline"
                else r"\[v1\]\s*(?:[A-Za-z]{3},?\s+)?\d{1,2}\s+[A-Za-z]+\s+2017\b"
                if metadata.get("kind") == "submission-history"
                else r"(?!)"
            )
            original_year |= _rendered(metadata) and bool(
                re.search(pattern, _text(metadata.get("text")), re.IGNORECASE)
            )
        check(
            "original_submission",
            original_year,
            "original submission metadata from 2017",
            kind.get("submission_metadata"),
        )
        abstract = _mapping(kind.get("abstract"))
        abstract_text = re.sub(r"^Abstract:\s*", "", _text(abstract.get("text")))
        check(
            "abstract",
            abstract.get("tag") == "blockquote" and _rendered(abstract) and bool(abstract_text),
            "nonempty rendered article abstract",
            abstract,
        )
    elif case.judge_kind == "python":
        check(
            "path",
            bool(parsed and parsed.path == "/3.14/library/collections.html" and not parsed.query),
            "Python 3.14 collections reference without query",
            data.get("url"),
        )
        check(
            "collections_heading",
            any(
                _mapping(h).get("tag") == "h1"
                and _rendered(h)
                and "collections" in _text(h.get("text")).casefold()
                and "container datatypes" in _text(h.get("text")).casefold()
                for h in headings
            ),
            "main collections — Container datatypes heading",
            headings,
        )
        section = _mapping(kind.get("counter_section"))
        check(
            "counter_section",
            section.get("tag") == "section"
            and section.get("id") == "counter-objects"
            and _rendered(section),
            "rendered Counter section",
            section,
        )
        method = _mapping(kind.get("method"))
        signature = re.sub(r"\s+", "", _text(method.get("text"))).removesuffix("¶")
        check(
            "method_signature",
            method.get("tag") == "dt"
            and method.get("id") == "collections.Counter.most_common"
            and _in_view(method, viewport)
            and signature
            in {
                "most_common(n=None)",
                "Counter.most_common(n=None)",
                "collections.Counter.most_common(n=None)",
            },
            "exact most_common(n=None) signature centered in view",
            method,
        )
        description = _mapping(kind.get("description"))
        paragraph = _mapping(kind.get("first_paragraph"))
        paragraph_text = _text(paragraph.get("text")).casefold()
        check(
            "method_description",
            description.get("tag") == "dd"
            and _rendered(description)
            and bool(_text(description.get("text")))
            and paragraph.get("tag") == "p"
            and _in_view(paragraph, viewport, center=False)
            and all(word in paragraph_text for word in ("most common", "elements", "counts")),
            "corresponding first description paragraph at least partly visible",
            paragraph,
        )
    elif case.judge_kind == "nasa":
        check(
            "path",
            bool(parsed and parsed.path.removesuffix("/") == "/mars/facts" and not parsed.query),
            "NASA Mars Facts article without query",
            data.get("url"),
        )
        check(
            "facts_heading",
            any(
                _mapping(h).get("tag") == "h1"
                and _rendered(h)
                and _text(h.get("text")) == "Mars Facts"
                for h in headings
            ),
            "exact main Mars Facts heading",
            headings,
        )
        sections = []
        for section in _sequence(kind.get("sections")):
            section = _mapping(section)
            heading = _mapping(section.get("heading"))
            content = _text(section.get("content_before_next_h2"))
            if (
                heading.get("tag") == "h2"
                and _text(heading.get("text")) == "Moons"
                and _in_view(heading, viewport)
                and re.search(r"\bPhobos\b", content)
                and re.search(r"\bDeimos\b", content)
            ):
                sections.append(section)
        check(
            "moons_section",
            bool(sections),
            "main Moons heading in view, with Phobos and Deimos before the next h2",
            kind.get("sections"),
        )
    return {"success": not mismatches, "observed": observed, "mismatches": mismatches}


def verify_public_web_case(browser, case: PublicWebCase | str) -> dict:
    """Read one fresh synchronous browser observation, then independently judge.

    Evaluation errors propagate so the harness preserves verification failures
    separately from an ordinary, observed goal mismatch.
    """
    case = _case(case)
    expression = _READ_COMMON_JS + _READ_CASE_JS[case.judge_kind] + "return result;\n})()"
    return validate_public_web_observation(case, browser.evaluate(expression))
