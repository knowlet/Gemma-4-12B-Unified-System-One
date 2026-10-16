"""Read-only private DOM facts and pure checks for twelve training goals.

The collector supplies trusted expected ownership separately. This module never
navigates, scrolls, dispatches events, consults hidden action nodes, changes DOM,
or supplies facts to an inference policy. Passing these predicates alone is not
a complete guarded-collection or fitting-eligibility proof.
"""

from __future__ import annotations

import json
import math
import re
from urllib.parse import urlsplit

from .browser_public_training_tasks import get_public_training_task

OBSERVATION_SCHEMA = "jev-public-native-training-dom-observation-v3"
OWNERSHIP_FIELDS = frozenset(
    {"target_id", "session_id", "root_frame_id", "loader_id", "native_page_key", "url"}
)

_READ_JS = r"""
(() => {
  const spec = __PRIVATE_CONFIG__;
  const origin = performance.timeOrigin, initialURL = location.href;
  const main = document.querySelector(spec.main_selector);
  const clean = value => String(value ?? '').replace(/\s+/g, ' ').trim();
  const normalized = value => clean(value).replace(/^\d+(?:\.\d+)*[.)]?\s*/, '')
    .toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
  const parent = node => node?.parentElement || node?.getRootNode?.()?.host || null;
  const path = node => {
    const parts = [];
    for (let current=node; current && current.nodeType===1; current=parent(current)) {
      if (parts.length >= 128) return null;
      const tag=current.tagName.toLowerCase(), lightParent=current.parentElement;
      const root=!lightParent ? current.getRootNode?.() : null;
      const siblings=[...(lightParent?.children || root?.children || [current])]
        .filter(child=>child.tagName===current.tagName);
      parts.unshift(tag+'['+(siblings.indexOf(current)+1)+']');
      if (!lightParent && root?.host) parts.unshift('#shadow-root');
    }
    return parts.length ? '/'+parts.join('/') : null;
  };
  const hit = (x,y) => {
    let node=x>=0 && y>=0 && x<innerWidth && y<innerHeight ? document.elementFromPoint(x,y) : null;
    const documentPath=path(node), chain=node ? [path(node)] : [];
    let depth=0;
    while (node?.shadowRoot?.mode==='open' && depth++<128) {
      const next=node.shadowRoot.elementFromPoint(x,y);
      if (!next || next===node) break;
      node=next;chain.push(path(node));
    }
    return {x,y,path:path(node),document_path:documentPath,chain,overflow:depth>128};
  };
  const fact = node => {
    if (!node) return null;
    const rect=node.getBoundingClientRect(), styles=[], ancestors=[];
    let current=node;
    for (; current && styles.length<128; current=parent(current)) {
      const style=getComputedStyle(current);
      ancestors.push(path(current));
      styles.push({display:style.display,visibility:style.visibility,opacity:style.opacity,
        content_visibility:style.contentVisibility || 'visible',hidden:current.hidden===true});
    }
    const raw=node.innerText ?? node.textContent ?? '';
    const left=Math.max(0,rect.left), top=Math.max(0,rect.top);
    const right=Math.min(innerWidth,rect.right), bottom=Math.min(innerHeight,rect.bottom);
    return {path:path(node),tag:node.tagName.toLowerCase(),id:node.id,text:raw.slice(0,32000),
      text_truncated:raw.length>32000,connected:node.isConnected,
      rect:{x:rect.left,y:rect.top,width:rect.width,height:rect.height},
      client_rect_count:node.getClientRects().length,style_chain:styles,style_overflow:!!current,
      composed_ancestor_paths:ancestors,
      center_hit:hit(rect.left+rect.width/2,rect.top+rect.height/2),
      intersection_hit:hit((left+right)/2,(top+bottom)/2)};
  };
  const excluded = node => !!node.closest('nav,aside,header,footer,script,style');
  const headings=main ? [...main.querySelectorAll('h1,h2,h3,h4,h5,h6')].filter(node=>!excluded(node)) : [];
  const heading=spec.target_heading ? headings.find(node=>normalized(node.innerText)===normalized(spec.target_heading)) : null;
  const ownNodes = (sectionHeading, predicate, maximum) => {
    if (!main || !sectionHeading) return [];
    const walker=document.createTreeWalker(main,NodeFilter.SHOW_ELEMENT), found=[];
    let active=false,node;
    while ((node=walker.nextNode())) {
      if (node===sectionHeading) {active=true;continue;}
      if (!active || sectionHeading.contains(node)) continue;
      // Opening content belongs to this heading only until the next heading,
      // including child subsections with a deeper level.
      if (/^H[1-6]$/.test(node.tagName)) break;
      if (!excluded(node) && predicate(node)) {
        found.push(node);
        if (found.length>=maximum) break;
      }
    }
    return found;
  };
  const target={heading:fact(heading),signature:null,signature_shadow:null,paragraph:null,paragraphs:[],row:null,
    matching_function:null,owner_container_path:null};
  if (spec.judge_kind==='mdn_method') {
    const pattern=new RegExp('(^|[^a-zA-Z0-9_$])'+spec.signature_name+'\\s*\\(');
    const signatures=ownNodes(heading,node=>node.tagName==='PRE' && pattern.test(node.textContent),16);
    target.signature=fact(signatures[0]);
    if (target.signature) {
      target.signature.owner_heading_path=path(heading);
      target.signature.source='light-dom';
    } else {
      // Only hosts encountered in this heading's own light-DOM interval may
      // supply a signature. No other section or closed root is searched.
      const hosts=ownNodes(heading,node=>node.tagName==='MDN-CODE-EXAMPLE',16);
      for (const host of hosts) {
        const root=host.shadowRoot;
        if (!root || root.mode!=='open') continue;
        const pre=[...root.querySelectorAll('pre')].find(candidate=>{
          const code=candidate.querySelector('code');
          return code && pattern.test(code.innerText ?? code.textContent ?? '');
        });
        if (!pre) continue;
        const code=pre.querySelector('code');
        target.signature={...fact(pre),owner_heading_path:path(heading),source:'open-shadow-root'};
        target.signature_shadow={root_mode:root.mode,host:fact(host),code:fact(code),
          root_path:path(host)+'/#shadow-root',owner_heading_path:path(heading)};
        break;
      }
    }
  } else if (spec.judge_kind==='overlay') {
    const fn=main ? [...main.querySelectorAll('code.function')].find(node=>clean(node.textContent)==='overlay') : null;
    const row=fn?.closest('tr'), cell=fn?.closest('td,th'), signature=fn?.closest('p');
    const paragraphs=cell ? [...cell.querySelectorAll('p')].filter(node=>clean(node.textContent)) : [];
    const index=paragraphs.indexOf(signature), explanation=index>=0 ? paragraphs[index+1] : null;
    target.row=fact(row); target.matching_function=fact(fn); target.signature=fact(signature);
    target.owner_container_path=path(cell);
    target.paragraphs=paragraphs.slice(0,16).map(fact); target.paragraph=fact(explanation);
    for (const item of [target.signature,target.paragraph,...target.paragraphs]) {
      if (item) item.owner_container_path=path(cell);
    }
  } else {
    target.paragraphs=ownNodes(heading,node=>node.tagName==='P' && clean(node.textContent),64).map(node=>
      ({...fact(node),owner_heading_path:path(heading)}));
    target.paragraph=target.paragraphs[0] ?? null;
  }
  // Git's manual identifies its command in NAME, not in an article h1.
  const nameHeading=spec.identity_command ? headings.find(node=>normalized(node.innerText)==='name') : null;
  const nameParagraphs=ownNodes(nameHeading,node=>node.tagName==='P' && clean(node.textContent),1)
    .map(node=>({...fact(node),owner_heading_path:path(nameHeading)}));
  const article_identity={heading:fact(nameHeading),paragraph:nameParagraphs[0] ?? null,
    paragraphs:nameParagraphs};
  const blockerSelector='#challenge-form,#cf-challenge-running,[id^="cf-chl-"],'+
    '.cf-error-details,.g-recaptcha,.h-captcha,input[type="password"]';
  return {schema_id:'jev-public-native-training-dom-observation-v3',url:location.href,
    document_title:document.title,document_time_origin:performance.timeOrigin,
    document_read:{url_start:initialURL,url_end:location.href,
      time_origin_start:origin,time_origin_end:performance.timeOrigin},
    ready_state:document.readyState,scroll_x:scrollX,scroll_y:scrollY,
    viewport:{width:innerWidth,height:innerHeight},main:fact(main),
    headings:headings.slice(0,128).map(fact),target,article_identity,
    blockers:[...document.querySelectorAll(blockerSelector)].slice(0,32).map(fact)};
})()
"""


def observe_public_training(browser, task):
    """One read-only evaluation; collector adds actual ownership afterward."""
    task = get_public_training_task(task)
    config = {
        key: task[key]
        for key in ("main_selector", "target_heading", "judge_kind", "signature_name")
    }
    config["identity_command"] = (
        task["main_identity"] if task["site_group"] == "git-reference" else None
    )
    return browser.evaluate(
        _READ_JS.replace("__PRIVATE_CONFIG__", json.dumps(config, ensure_ascii=False))
    )


def _mapping(value):
    return value if type(value) is dict else {}


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _words(value):
    return " ".join(re.findall(r"[^\W_]+", _text(value).casefold()))


def _heading_text(value):
    return _words(re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", _text(value)))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e16


def _node_path(value):
    return isinstance(value, str) and bool(
        len(value.split("/")) <= 129
        and re.fullmatch(
            r"(?:/[a-z][a-z0-9:_-]*\[[1-9][0-9]*\])+(?:/#shadow-root"
            r"(?:/[a-z][a-z0-9:_-]*\[[1-9][0-9]*\])+)*",
            value,
        )
    )


def _within(path, parent):
    return (
        _node_path(path)
        and _node_path(parent)
        and (path == parent or path.startswith(parent + "/"))
    )


def _rect(value):
    return (
        type(value) is dict
        and set(value) == {"x", "y", "width", "height"}
        and all(_number(value[key]) and abs(value[key]) <= 1e9 for key in value)
        and value["width"] > 0
        and value["height"] > 0
    )


def _rendered(node, *, allow_contents=False):
    node = _mapping(node)
    styles = node.get("style_chain")
    box = (
        _rect(node.get("rect"))
        and type(node.get("client_rect_count")) is int
        and node["client_rect_count"] > 0
    )
    if allow_contents and type(styles) is list and styles and type(styles[0]) is dict:
        rect = node.get("rect")
        box = box or (
            styles[0].get("display") == "contents"
            and type(rect) is dict
            and set(rect) == {"x", "y", "width", "height"}
            and all(_number(value) for value in rect.values())
            and rect["width"] == rect["height"] == 0
            and type(node.get("client_rect_count")) is int
            and node["client_rect_count"] == 0
        )
    if (
        node.get("connected") is not True
        or not _node_path(node.get("path"))
        or not box
        or node.get("style_overflow") is not False
        or type(styles) is not list
        or not 1 <= len(styles) <= 128
    ):
        return False
    for style in styles:
        if (
            type(style) is not dict
            or style.get("hidden") is not False
            or not isinstance(style.get("display"), str)
            or style["display"] in {"", "none"}
            or style.get("visibility") != "visible"
            or style.get("content_visibility") not in {"visible", "auto"}
            or not isinstance(style.get("opacity"), str)
        ):
            return False
        try:
            opacity = float(style["opacity"])
        except ValueError:
            return False
        if not math.isfinite(opacity) or not 0 < opacity <= 1:
            return False
    return True


def _viewport(value):
    return type(value) is dict and all(
        _number(value.get(key)) and 0 < value[key] <= 100_000 for key in ("width", "height")
    )


def _visible(node, viewport, *, opening=False):
    if not _rendered(node) or not _viewport(viewport):
        return False
    rect = node["rect"]
    if opening:
        left, top = max(0, rect["x"]), max(0, rect["y"])
        right = min(viewport["width"], rect["x"] + rect["width"])
        bottom = min(viewport["height"], rect["y"] + rect["height"])
        if rect["y"] < 0 or right <= left or bottom - top < min(16, rect["height"]):
            return False
        x, y = (left + right) / 2, (top + bottom) / 2
        hit = _mapping(node.get("intersection_hit"))
    else:
        x, y = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        hit = _mapping(node.get("center_hit"))
    return (
        0 <= x < viewport["width"]
        and 0 <= y < viewport["height"]
        and _number(hit.get("x"))
        and _number(hit.get("y"))
        and abs(hit["x"] - x) < 1e-4
        and abs(hit["y"] - y) < 1e-4
        and _within(hit.get("path"), node["path"])
    )


def _composed_ancestors(node):
    """Require every style's composed ancestor, including the light-DOM host."""
    current = node.get("path")
    if not _node_path(current):
        return False
    expected = []
    while current:
        expected.append(current)
        current = current.rsplit("/", 1)[0]
        if current.endswith("/#shadow-root"):
            current = current.removesuffix("/#shadow-root")
    return (
        node.get("composed_ancestor_paths") == expected
        and type(node.get("style_chain")) is list
        and len(node["style_chain"]) == len(expected)
    )


def _shadow_hit(node, host, root_path):
    hit = _mapping(node.get("center_hit"))
    chain = hit.get("chain")
    return (
        hit.get("document_path") == host.get("path")
        and hit.get("overflow") is False
        and type(chain) is list
        and len(chain) == 2
        and chain[0] == host.get("path")
        and chain[-1] == hit.get("path")
        and _within(chain[-1], node.get("path"))
        and chain[-1].startswith(root_path + "/")
        and chain[-1].count("/#shadow-root/") == 1
    )


def _mdn_signature_source(target, heading, signature, main, viewport, name):
    source = signature.get("source")
    shadow = target.get("signature_shadow")
    signature_path = signature.get("path")
    if source == "light-dom":
        return (
            shadow is None and _node_path(signature_path) and "/#shadow-root/" not in signature_path
        )
    shadow = _mapping(shadow)
    host, code = (_mapping(shadow.get(key)) for key in ("host", "code"))
    root_path = shadow.get("root_path")
    return (
        source == "open-shadow-root"
        and _node_path(signature_path)
        and shadow.get("root_mode") == "open"
        and host.get("tag") == "mdn-code-example"
        and _within(host.get("path"), main.get("path"))
        and "/#shadow-root/" not in host.get("path", "")
        and root_path == host.get("path", "") + "/#shadow-root"
        and signature.get("path", "").startswith(root_path + "/")
        and signature.get("path", "").count("/#shadow-root/") == 1
        and shadow.get("owner_heading_path") == heading.get("path")
        and _rendered(host, allow_contents=True)
        and code.get("tag") == "code"
        and _within(code.get("path"), signature.get("path"))
        and code.get("text_truncated") is False
        and bool(
            re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"\s*\(", _text(code.get("text")))
        )
        and _visible(code, viewport)
        and _composed_ancestors(signature)
        and _composed_ancestors(code)
        and _composed_ancestors(host)
        and _shadow_hit(signature, host, root_path)
        and _shadow_hit(code, host, root_path)
    )


def _url(value, task, *, target=False):
    if not isinstance(value, str) or value != value.strip() or re.search(r"[\x00-\x20\x7f]", value):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname in task["allowed_top_level_hosts"]
            and parsed.username is None
            and parsed.password is None
            and parsed.port in {None, 443}
            and not parsed.query
            and (not target or parsed.path == task["target_path"])
        )
    except ValueError:
        return False


def _valid_ownership(binding):
    if type(binding) is not dict or set(binding) != OWNERSHIP_FIELDS:
        return False
    if not all(
        isinstance(binding[key], str) and 0 < len(binding[key]) <= 4096
        for key in OWNERSHIP_FIELDS - {"native_page_key"}
    ):
        return False
    key = binding["native_page_key"]
    return (
        type(key) is list
        and len(key) == 7
        and _number(key[0])
        and key[0] > 0
        and key[1] == binding["url"]
        and all(_number(key[index]) for index in (2, 3, 4, 5))
        and key[4] > 0
        and key[5] > 0
        and type(key[6]) is list
    )


def validate_public_training_observation(task, observation, *, expected_ownership=None):
    """Recompute site predicates and require a separately trusted owned binding.

    ``observation['expected_ownership']`` and any claimed ``success`` are ignored.
    The collector must bind actual target/session/root/loader/native page key and
    pass its trusted expectation explicitly. Control identities in page_key[6]
    remain native collector/guard evidence, not reconstructed by this DOM judge.
    """
    task, data = get_public_training_task(task), _mapping(observation)
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    viewport = _mapping(data.get("viewport"))
    check("observation_schema", data.get("schema_id") == OBSERVATION_SCHEMA)
    check("viewport", _viewport(viewport))
    check("document_ready", data.get("ready_state") == "complete")
    check("target_document_url", _url(data.get("url"), task, target=True))
    expected_ok = _valid_ownership(expected_ownership)
    check("trusted_expected_ownership", expected_ok)
    actual = data.get("ownership")
    check(
        "owned_context", expected_ok and _valid_ownership(actual) and actual == expected_ownership
    )
    key = expected_ownership["native_page_key"] if expected_ok else [None] * 7
    raw_key = [
        data.get("document_time_origin"),
        data.get("url"),
        data.get("scroll_x"),
        data.get("scroll_y"),
        viewport.get("width"),
        viewport.get("height"),
    ]
    read = _mapping(data.get("document_read"))
    check(
        "same_native_document",
        expected_ok
        and all(_number(raw_key[index]) for index in (0, 2, 3, 4, 5))
        and raw_key == key[:6]
        and read.get("url_start") == read.get("url_end") == data.get("url")
        and read.get("time_origin_start") == read.get("time_origin_end") == raw_key[0],
    )
    title, main = _text(data.get("document_title")), _mapping(data.get("main"))
    # A display:contents main has no box; its identity and target descendants
    # still have to pass their own box, CSS, ownership and visibility checks.
    check("main_rendered", _rendered(main, allow_contents=True))
    identity = _words(task["main_identity"])
    check("document_title_identity", bool(identity) and identity in _words(title))
    headings = data.get("headings") if type(data.get("headings")) is list else []
    identity_nodes = [
        node
        for node in headings
        if type(node) is dict
        and _rendered(node)
        and _within(node.get("path"), main.get("path"))
        and node.get("tag") in {"h1", "h2", "h3"}
        and identity in _heading_text(node.get("text"))
        and node.get("text_truncated") is False
    ]
    if task["judge_kind"] == "mdn_method":
        identity_nodes = [
            node
            for node in identity_nodes
            if node.get("tag") == "h1"
            and re.search(
                r"\b" + re.escape(task["signature_name"]) + r"\s*\(\)", _text(node.get("text"))
            )
        ]
    if task["site_group"] == "git-reference":
        article = _mapping(data.get("article_identity"))
        name_heading, name_paragraph = (
            _mapping(article.get(key)) for key in ("heading", "paragraph")
        )
        name_paragraphs = article.get("paragraphs")
        document_identity = (
            _rendered(name_heading)
            and name_heading in headings
            and name_heading.get("tag") in {"h1", "h2", "h3", "h4", "h5", "h6"}
            and _heading_text(name_heading.get("text")) == "name"
            and name_heading.get("text_truncated") is False
            and _within(name_heading.get("path"), main.get("path"))
            and _rendered(name_paragraph)
            and name_paragraph.get("tag") == "p"
            and name_paragraph.get("text_truncated") is False
            and _within(name_paragraph.get("path"), main.get("path"))
            and name_paragraph.get("owner_heading_path") == name_heading.get("path")
            and type(name_paragraphs) is list
            and len(name_paragraphs) == 1
            and name_paragraphs[0] == name_paragraph
            and re.match(
                r"^" + re.escape(task["main_identity"]) + r"(?:\(\d\))?\s+[-–—]\s+\S",
                _text(name_paragraph.get("text")),
            )
        )
    else:
        document_identity = bool(identity_nodes)
    check("main_document_identity", document_identity)
    blockers = data.get("blockers")
    check(
        "no_visible_access_blocker",
        type(blockers) is list and not any(_visible(node, viewport) for node in blockers),
    )
    check(
        "no_access_error_title",
        not re.search(
            r"\b(?:access denied|request blocked|verify you are human|just a moment|403 forbidden|404 not found|page not found)\b",
            title,
            re.IGNORECASE,
        ),
    )
    target = _mapping(data.get("target"))
    heading, signature, paragraph = (
        _mapping(target.get(key)) for key in ("heading", "signature", "paragraph")
    )
    if task["target_heading"] is not None:
        check(
            "requested_heading",
            _heading_text(heading.get("text")) == _heading_text(task["target_heading"])
            and heading.get("tag") in {"h1", "h2", "h3", "h4", "h5", "h6"}
            and heading.get("text_truncated") is False
            and _within(heading.get("path"), main.get("path"))
            and _visible(heading, viewport),
        )
    if task["judge_kind"] in {"mdn_method", "overlay"}:
        check(
            "requested_signature",
            signature.get("text_truncated") is False
            and _within(signature.get("path"), main.get("path"))
            and bool(
                re.search(
                    r"(?<![A-Za-z0-9_$])" + re.escape(task["signature_name"]) + r"\s*\(",
                    _text(signature.get("text")),
                )
            )
            and _visible(signature, viewport),
        )
    if task["judge_kind"] == "mdn_method":
        check(
            "signature_in_syntax_section",
            bool(heading)
            and signature.get("tag") == "pre"
            and signature.get("owner_heading_path") == heading.get("path"),
        )
        check(
            "signature_source_ownership",
            _mdn_signature_source(
                target, heading, signature, main, viewport, task["signature_name"]
            ),
        )
    elif task["judge_kind"] == "overlay":
        row, function = _mapping(target.get("row")), _mapping(target.get("matching_function"))
        cell = target.get("owner_container_path")
        paragraphs = target.get("paragraphs")
        check(
            "matching_overlay_row",
            _rendered(row)
            and _visible(function, viewport)
            and row.get("tag") == "tr"
            and function.get("tag") == "code"
            and signature.get("tag") == "p"
            and _text(function.get("text")) == "overlay"
            and _within(function.get("path"), signature.get("path"))
            and _within(signature.get("path"), cell)
            and _within(cell, row.get("path"))
            and _within(row.get("path"), main.get("path")),
        )
        check(
            "own_overlay_explanation",
            type(paragraphs) is list
            and len(paragraphs) >= 2
            and paragraphs[0] == signature
            and paragraphs[1] == paragraph
            and signature.get("owner_container_path")
            == paragraph.get("owner_container_path")
            == cell
            and paragraph.get("tag") == "p"
            and _within(paragraph.get("path"), cell)
            and len(_text(paragraph.get("text")).split()) >= 3
            and paragraph.get("text_truncated") is False
            and _visible(paragraph, viewport, opening=True),
        )
    else:
        paragraphs = target.get("paragraphs")
        check(
            "first_own_explanatory_paragraph",
            type(paragraphs) is list
            and bool(paragraphs)
            and paragraphs[0] == paragraph
            and paragraph.get("tag") == "p"
            and paragraph.get("text_truncated") is False
            and paragraph.get("owner_heading_path") == heading.get("path")
            and _within(paragraph.get("path"), main.get("path"))
            and len(_text(paragraph.get("text")).split()) >= 3
            and _visible(paragraph, viewport, opening=True),
        )
    return {
        "schema_id": "jev-public-native-training-dom-verdict-v3",
        "goal_id": task["goal_id"],
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scope": "private DOM and supplied trusted ownership checks only; not full collection validation",
    }
