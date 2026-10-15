"""Pure private predicates and read-only observer source; no live browser calls."""

import copy
import json
import re
import shutil
import subprocess

import pytest

from s1.browser_public_training_evidence import (
    OBSERVATION_SCHEMA,
    observe_public_training,
    validate_public_training_observation,
)
from s1.browser_public_training_tasks import get_public_training_task, tasks

MAIN = "/html[1]/body[1]/main[1]"


def node(path, text, *, tag="p", x=40, y=100, width=700, height=40):
    return {
        "path": path,
        "tag": tag,
        "id": "",
        "text": text,
        "text_truncated": False,
        "connected": True,
        "rect": {"x": x, "y": y, "width": width, "height": height},
        "client_rect_count": 1,
        "style_chain": [
            {
                "display": "block",
                "visibility": "visible",
                "opacity": "1",
                "content_visibility": "visible",
                "hidden": False,
            }
        ],
        "style_overflow": False,
        "center_hit": {"x": x + width / 2, "y": y + height / 2, "path": path},
        "intersection_hit": {"x": x + width / 2, "y": y + height / 2, "path": path},
    }


def observation(task):
    task = get_public_training_task(task)
    url, origin = task["primary_target_url"], 1_790_000_000_000
    owner = {
        "target_id": "owned-target",
        "session_id": "owned-session",
        "root_frame_id": "owned-root",
        "loader_id": "owned-loader",
        "native_page_key": [origin, url, 0, 200, 1024, 768, []],
        "url": url,
    }
    title = task["main_identity"]
    if task["judge_kind"] == "mdn_method":
        title += ": " + task["signature_name"] + "() method"
    main = node(MAIN, "Main document content", tag="main", y=-200, height=2000)
    identity = node(MAIN + "/h1[1]", title, tag="h1", y=-100)
    heading = node(MAIN + "/h2[1]", task["target_heading"] or "Function table", tag="h2")
    paragraph = node(
        MAIN + "/p[1]", "This is the first explanatory paragraph for the requested section.", y=240
    )
    paragraph["owner_heading_path"] = heading["path"]
    target = {
        "heading": heading,
        "signature": None,
        "paragraph": paragraph,
        "paragraphs": [paragraph],
        "row": None,
        "matching_function": None,
        "owner_container_path": None,
    }
    if task["judge_kind"] == "mdn_method":
        signature = node(
            MAIN + "/pre[1]", task["signature_name"] + "(value)", tag="pre", y=160, height=60
        )
        signature["owner_heading_path"] = heading["path"]
        target.update(signature=signature, paragraph=None, paragraphs=[])
    elif task["judge_kind"] == "overlay":
        row = node(
            MAIN + "/table[1]/tbody[1]/tr[1]", "overlay function row", tag="tr", y=120, height=180
        )
        cell = row["path"] + "/td[1]"
        signature = node(
            cell + "/p[1]",
            "overlay ( string text PLACING replacement text FROM start integer ) → text",
            y=140,
            height=60,
        )
        signature["owner_container_path"] = cell
        paragraph = node(
            cell + "/p[2]",
            "Replaces the substring of the original string with the new substring.",
            y=220,
        )
        paragraph["owner_container_path"] = cell
        function = node(
            signature["path"] + "/code[1]", "overlay", tag="code", y=140, width=80, height=20
        )
        target.update(
            heading=None,
            signature=signature,
            paragraph=paragraph,
            paragraphs=[signature, paragraph],
            row=row,
            matching_function=function,
            owner_container_path=cell,
        )
    raw = {
        "schema_id": OBSERVATION_SCHEMA,
        "url": url,
        "document_title": title + " - Official Documentation",
        "document_time_origin": origin,
        "document_read": {
            "url_start": url,
            "url_end": url,
            "time_origin_start": origin,
            "time_origin_end": origin,
        },
        "ready_state": "complete",
        "scroll_x": 0,
        "scroll_y": 200,
        "viewport": {"width": 1024, "height": 768},
        "main": main,
        "headings": [identity, heading],
        "target": target,
        "blockers": [],
        "ownership": copy.deepcopy(owner),
    }
    return raw, owner


@pytest.mark.parametrize("task", tasks(), ids=lambda task: task["goal_id"])
def test_all_twelve_exact_task_shapes_have_recomputed_private_checks(task):
    raw, owner = observation(task)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["success"], verdict["failed_checks"]
    assert all(verdict["checks"].values())
    assert "not full collection validation" in verdict["scope"]


def test_missing_trusted_ownership_and_self_asserted_expectation_never_pass():
    task = tasks()[0]
    raw, owner = observation(task)
    raw["success"] = True
    raw["expected_ownership"] = owner
    verdict = validate_public_training_observation(task, raw)
    assert not verdict["success"]
    assert not verdict["checks"]["trusted_expected_ownership"]


@pytest.mark.parametrize("key", ["target_id", "session_id", "root_frame_id", "loader_id", "url"])
def test_foreign_owned_context_is_rejected(key):
    task = tasks()[0]
    raw, owner = observation(task)
    raw["ownership"][key] = "foreign-context"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "owned_context"
    ]


@pytest.mark.parametrize("key", ["document_time_origin", "scroll_x", "scroll_y"])
def test_private_read_must_match_current_native_page_key(key):
    task = tasks()[0]
    raw, owner = observation(task)
    raw[key] += 1
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "same_native_document"
    ]


def test_read_cannot_cross_document_reload_or_change_native_viewport():
    task = tasks()[0]
    for mutate in (
        lambda raw: raw["document_read"].update(time_origin_end=1),
        lambda raw: raw["viewport"].update(width=900),
    ):
        raw, owner = observation(task)
        mutate(raw)
        assert not validate_public_training_observation(task, raw, expected_ownership=owner)[
            "success"
        ]


@pytest.mark.parametrize(
    "url",
    [
        "https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver#syntax",
        "https://developer.mozilla.org.evil.test/en-US/docs/Web/API/IntersectionObserver/observe#syntax",
        "http://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver/observe",
        "https://user:pass@developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver/observe",
    ],
)
def test_bare_fragment_related_path_or_foreign_url_is_not_completion(url):
    task = tasks()[0]
    raw, owner = observation(task)
    raw["url"] = url
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "target_document_url"
    ]


@pytest.mark.parametrize("field", ["heading", "signature"])
def test_mdn_required_heading_and_signature_must_be_unobscured(field):
    task = tasks()[0]
    raw, owner = observation(task)
    raw["target"][field]["center_hit"]["path"] = "/html[1]/body[1]/dialog[1]"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize(
    "method", ["get(value)", "unobserve(value)", "observe is only mentioned in prose"]
)
def test_mdn_signature_is_correct_method_not_related_text(method):
    task = tasks()[0]
    raw, owner = observation(task)
    raw["target"]["signature"]["text"] = method
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "requested_signature"
    ]


def test_mdn_requires_right_interface_and_syntax_owner_but_no_explanation():
    task = tasks()[0]
    raw, owner = observation(task)
    assert raw["target"]["paragraph"] is None
    assert validate_public_training_observation(task, raw, expected_ownership=owner)["success"]
    raw["headings"][0]["text"] = "ResizeObserver: observe() method"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "main_document_identity"
    ]
    raw, owner = observation(task)
    raw["target"]["signature"]["owner_heading_path"] = MAIN + "/h2[2]"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "signature_in_syntax_section"
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True, 1e12])
def test_bad_target_rectangles_cannot_be_proved_visible(value):
    task = tasks()[0]
    raw, owner = observation(task)
    raw["target"]["signature"]["rect"]["height"] = value
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize(
    "mutation",
    ["offscreen", "hidden", "transparent_ancestor", "truncated", "detached", "wrong_section"],
)
def test_requested_section_facts_are_recomputed_instead_of_trusting_success(mutation):
    task = get_public_training_task("postgres18-check-constraints")
    raw, owner = observation(task)
    raw["success"] = True
    target = raw["target"]["heading"]
    if mutation == "offscreen":
        target["rect"]["y"] = 5000
    elif mutation == "hidden":
        target["style_chain"][0]["display"] = "none"
    elif mutation == "transparent_ancestor":
        target["style_chain"].append(
            {
                "display": "block",
                "visibility": "visible",
                "opacity": "0",
                "content_visibility": "visible",
                "hidden": False,
            }
        )
    elif mutation == "truncated":
        target["text_truncated"] = True
    elif mutation == "detached":
        target["connected"] = False
    else:
        target["text"] = "Foreign Keys"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


def test_postgres_overlay_requires_matching_row_and_its_first_explanation():
    task = get_public_training_task("postgres18-string-overlay")
    raw, owner = observation(task)
    raw["target"]["matching_function"]["text"] = "substring"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "matching_overlay_row"
    ]
    raw, owner = observation(task)
    raw["target"]["paragraph"]["owner_container_path"] = MAIN + "/table[1]/tbody[1]/tr[2]/td[1]"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "own_overlay_explanation"
    ]


def test_first_own_paragraph_must_be_visible_and_belong_to_requested_section():
    task = get_public_training_task("rust-stack-heap")
    raw, owner = observation(task)
    raw["target"]["paragraph"]["owner_heading_path"] = MAIN + "/h2[2]"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "first_own_explanatory_paragraph"
    ]
    raw, owner = observation(task)
    raw["target"]["paragraph"]["rect"]["y"] = -30
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


def test_correct_description_on_wrong_git_command_document_is_rejected():
    task = get_public_training_task("git-restore-description")
    raw, owner = observation(task)
    raw["headings"][0]["text"] = "git-worktree(1)"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "main_document_identity"
    ]


def test_access_challenge_and_visible_password_controls_fail_closed():
    task = tasks()[0]
    raw, owner = observation(task)
    raw["blockers"] = [node(MAIN + "/input[1]", "", tag="input")]
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["checks"][
        "no_visible_access_blocker"
    ]
    raw, owner = observation(task)
    raw["document_title"] = "Just a moment"
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize(
    "field", ["target_id", "session_id", "root_frame_id", "loader_id", "native_page_key"]
)
def test_missing_expected_root_context_fields_fail_closed(field):
    task = tasks()[0]
    raw, owner = observation(task)
    owner.pop(field)
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


def test_observer_performs_one_readonly_evaluation_and_contains_no_policy_labels(tmp_path):
    class Browser:
        def __init__(self):
            self.expressions = []

        def evaluate(self, expression):
            self.expressions.append(expression)
            return {"raw": "not a verdict"}

    browser = Browser()
    assert observe_public_training(browser, tasks()[0]) == {"raw": "not a verdict"}
    assert len(browser.expressions) == 1
    script = browser.expressions[0]
    assert "document_time_origin" in script and "elementFromPoint" in script
    assert "getBoundingClientRect" in script and "getComputedStyle" in script
    for forbidden in (
        "scrollIntoView",
        "scrollTo(",
        "scrollBy(",
        "dispatchEvent",
        "__jevFast",
        "fetch(",
        "window.open",
        "appendChild",
        "setAttribute",
    ):
        assert forbidden not in script
    assert not re.search(r"(?:location\.href|scrollX|scrollY)\s*=(?!=)", script)
    assert "teacher_choice" not in script and "gold" not in script
    if shutil.which("node"):
        path = tmp_path / "readonly-observer.js"
        path.write_text(script)
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    private_config = script.split("const spec = ", 1)[1].split(";", 1)[0]
    assert json.loads(private_config)["target_heading"] == "Syntax"


def dom_observation(task, *, child_heading=False, own_content=False, overlay_name_visible=True):
    """Execute the real observer against a small CPU-only DOM API mock.

    Facts, paths and hit tests come from the observer JS. The mock supplies a
    document tree and CSS rectangles; it never manufactures an owner verdict.
    """
    if not shutil.which("node"):
        pytest.skip("Node is required for the actual observer DOM mock")

    class Browser:
        def evaluate(self, expression):
            return expression

    script = observe_public_training(Browser(), task)
    program = r"""
const spec=__TASK__, childHeading=__CHILD__, ownContent=__OWN__, visibleName=__VISIBLE__;
globalThis.location={href:spec.primary_target_url};
globalThis.performance={timeOrigin:1790000000000};
globalThis.innerWidth=1024;globalThis.innerHeight=768;
globalThis.scrollX=0;globalThis.scrollY=200;
globalThis.NodeFilter={SHOW_ELEMENT:1};
function node(tag,text,y,height=40,width=700) {
  return {tagName:tag.toUpperCase(),nodeType:1,innerText:text,textContent:text,id:'',
    hidden:false,isConnected:true,children:[],parentElement:null,
    contains(other){for(;other;other=other.parentElement)if(other===this)return true;return false;},
    closest(selector){const tags=selector.split(',');for(let n=this;n;n=n.parentElement)
      if(tags.includes(n.tagName.toLowerCase()))return n;return null;},
    getBoundingClientRect(){return {left:40,top:y,right:40+width,bottom:y+height,width,height};},
    getClientRects(){return [this.getBoundingClientRect()];}};
}
function add(parent,child){parent.children.push(child);child.parentElement=parent;return child;}
function descendants(root){return root.children.flatMap(n=>[n,...descendants(n)]);}
const html=node('html','',0,768),body=add(html,node('body','',0,768));
const main=add(body,node('main','',-200,2000));
const identity=spec.main_identity+(spec.judge_kind==='mdn_method' ? ': '+spec.signature_name+'() method' : '');
add(main,node('h1',identity,-100));
if(spec.judge_kind==='overlay') {
  const row=add(add(add(main,node('table','',-40,350)),node('tbody','',-40,350)),node('tr','',-40,350));
  const cell=add(row,node('td','',-40,350));
  const signature=add(cell,node('p','overlay ( string text PLACING replacement text FROM start integer ) → text',visibleName?120:-40,visibleName?60:100));
  const fn=add(signature,node('code','overlay',visibleName?140:-30,20,80));fn.functionClass=true;
  add(cell,node('p','Replaces the substring of the original string with the new substring.',visibleName?220:100));
} else {
  add(main,node('h2',spec.target_heading,100));
  const content=()=>spec.judge_kind==='mdn_method'
    ? node('pre',spec.signature_name+'(value)',240,60)
    : node('p',childHeading&&!ownContent
      ? 'This explanation belongs only to the different child subsection.'
      : 'This is the opening explanation directly owned by the requested section.',240);
  if(ownContent)add(main,content());
  if(childHeading)add(main,node('h3','A Different Child Subsection',180));
  if(!ownContent)add(main,content());
}
const all=[html,body,main,...descendants(main)];
main.querySelectorAll=selector=>selector==='code.function'
  ? descendants(main).filter(n=>n.functionClass)
  : descendants(main).filter(n=>/^H[1-6]$/.test(n.tagName));
for(const cell of all.filter(n=>n.tagName==='TD'))cell.querySelectorAll=()=>descendants(cell).filter(n=>n.tagName==='P');
globalThis.getComputedStyle=()=>({display:'block',visibility:'visible',opacity:'1',contentVisibility:'visible'});
globalThis.document={title:identity+' - Official Documentation',readyState:'complete',
  querySelector(){return main;},querySelectorAll(){return [];},
  createTreeWalker(root){const nodes=descendants(root);let i=0;return {nextNode(){return nodes[i++]||null;}};},
  elementFromPoint(x,y){return [...all].reverse().find(n=>{const r=n.getBoundingClientRect();
    return x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;})||null;}};
console.log(JSON.stringify(eval(__SCRIPT__)));
"""
    for key, value in (
        ("__TASK__", task),
        ("__CHILD__", child_heading),
        ("__OWN__", own_content),
        ("__VISIBLE__", overlay_name_visible),
        ("__SCRIPT__", script),
    ):
        program = program.replace(key, json.dumps(value))
    result = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True)
    raw = json.loads(result.stdout)
    _, owner = observation(task)
    raw["ownership"] = copy.deepcopy(owner)
    return raw, owner


@pytest.mark.parametrize("task_id", ["rust-stack-heap", "mdn-intersectionobserver-observe"])
def test_actual_dom_observer_does_not_borrow_child_subsection_content(task_id):
    task = get_public_training_task(task_id)
    raw, owner = dom_observation(task, child_heading=True)
    field = "signature" if task["judge_kind"] == "mdn_method" else "paragraph"
    assert raw["target"][field] is None
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize("task_id", ["rust-stack-heap", "mdn-intersectionobserver-observe"])
def test_actual_dom_observer_keeps_own_content_before_child_heading(task_id):
    task = get_public_training_task(task_id)
    raw, owner = dom_observation(task, child_heading=True, own_content=True)
    assert validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize("visible", [False, True])
def test_actual_dom_overlay_requires_function_name_visible_with_signature_and_explanation(visible):
    task = get_public_training_task("postgres18-string-overlay")
    raw, owner = dom_observation(task, overlay_name_visible=visible)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["success"] is visible
    assert verdict["checks"]["matching_overlay_row"] is visible
