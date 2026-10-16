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
        signature["source"] = "light-dom"
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
    if task["site_group"] == "git-reference":
        name_heading = node(MAIN + "/section[1]/h2[1]", "NAME", tag="h2", y=-180)
        name_paragraph = node(
            MAIN + "/section[1]/p[1]", task["main_identity"] + " - The documented command.", y=-130
        )
        name_paragraph["owner_heading_path"] = name_heading["path"]
        raw["headings"].append(name_heading)
        raw["article_identity"] = {
            "heading": name_heading,
            "paragraph": name_paragraph,
            "paragraphs": [name_paragraph],
        }
    return raw, owner


@pytest.mark.parametrize("task", tasks(), ids=lambda task: task["goal_id"])
def test_all_twelve_exact_task_shapes_have_recomputed_private_checks(task):
    raw, owner = observation(task)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["success"], verdict["failed_checks"]
    assert raw["schema_id"] == "jev-public-native-training-dom-observation-v3"
    assert verdict["schema_id"] == "jev-public-native-training-dom-verdict-v3"
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
    raw["article_identity"]["paragraph"]["text"] = "git-worktree - Manage multiple working trees."
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


def dom_observation(
    task,
    *,
    child_heading=False,
    own_content=False,
    overlay_name_visible=True,
    main_display="block",
    zero_main_box=False,
    git_command=None,
    git_name_child=False,
    mdn_shadow=None,
):
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
const mainDisplay=__DISPLAY__, zeroMainBox=__ZERO_BOX__, gitCommand=__GIT_COMMAND__, gitNameChild=__GIT_CHILD__;
const shadowMode=__SHADOW__;
globalThis.location={href:spec.primary_target_url};
globalThis.performance={timeOrigin:1790000000000};
globalThis.innerWidth=1024;globalThis.innerHeight=768;
globalThis.scrollX=0;globalThis.scrollY=200;
globalThis.NodeFilter={SHOW_ELEMENT:1};
function node(tag,text,y,height=40,width=700) {
  return {tagName:tag.toUpperCase(),nodeType:1,innerText:text,textContent:text,id:'',
    hidden:false,isConnected:true,children:[],parentElement:null,
    getRootNode(){return this._root || this.parentElement?.getRootNode() || globalThis.document;},
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
if(zeroMainBox) {
  main.getBoundingClientRect=()=>({left:0,top:0,right:0,bottom:0,width:0,height:0});
  main.getClientRects=()=>[];
}
const identity=spec.main_identity+(spec.judge_kind==='mdn_method' ? ': '+spec.signature_name+'() method' : '');
if(spec.site_group!=='git-reference')add(main,node('h1',identity,-100));
if(spec.site_group==='git-reference') {
  add(main,node('h2','NAME',-180));
  if(gitNameChild)add(main,node('h3','A Different Child Subsection',-140));
  add(main,node('p',(gitCommand??spec.main_identity)+' - The documented command.',-100));
  add(main,node('h2','SYNOPSIS',-40));
  add(main,node('pre','git restore [options]',0));
  add(main,node('h2',spec.target_heading,100));
  add(main,node('p','This is the opening explanation of the requested command.',200));
} else if(spec.judge_kind==='overlay') {
  const row=add(add(add(main,node('table','',-40,350)),node('tbody','',-40,350)),node('tr','',-40,350));
  const cell=add(row,node('td','',-40,350));
  const signature=add(cell,node('p','overlay ( string text PLACING replacement text FROM start integer ) → text',visibleName?120:-40,visibleName?60:100));
  const fn=add(signature,node('code','overlay',visibleName?140:-30,20,80));fn.functionClass=true;
  add(cell,node('p','Replaces the substring of the original string with the new substring.',visibleName?220:100));
} else {
  add(main,node('h2',spec.target_heading,100));
  if(shadowMode && spec.judge_kind==='mdn_method') {
    if(childHeading)add(main,node('h3','A Different Child Subsection',180));
    const host=add(main,node('mdn-code-example',shadowMode==='light_text_only'?spec.signature_name+'(value)':'',230,100));
    const root={host,mode:shadowMode==='closed'?'closed':'open',children:[]};
    if(shadowMode!=='closed' && shadowMode!=='light_text_only')host.shadowRoot=root;
    const pre=node('pre',spec.signature_name+'(value)',240,80,600);pre._root=root;
    root.children.push(pre);
    const code=add(pre,node('code',shadowMode==='wrong_signature'?'unrelated(value)':spec.signature_name+'(value)',245,60,500));
    if(shadowMode==='missing_code')pre.children=[];
    if(shadowMode==='hidden_host')host._display='none';
    if(shadowMode==='hidden_code')code._visibility='hidden';
    if(shadowMode==='offscreen') {
      for(const n of [host,pre,code])n.getBoundingClientRect=()=>({left:40,top:900,right:640,bottom:980,width:600,height:80});
    }
    if(shadowMode==='disconnected')code.isConnected=false;
    root.querySelectorAll=()=>descendants(root).filter(n=>n.tagName==='PRE');
    pre.querySelector=()=>pre.children.find(n=>n.tagName==='CODE')||null;
    root.elementFromPoint=(x,y)=>[...descendants(root)].reverse().find(n=>{const r=n.getBoundingClientRect();
      return x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;})||null;
    if(shadowMode==='occluded')root.elementFromPoint=()=>null;
  } else {
  const content=()=>spec.judge_kind==='mdn_method'
    ? node('pre',spec.signature_name+'(value)',240,60)
    : node('p',childHeading&&!ownContent
      ? 'This explanation belongs only to the different child subsection.'
      : 'This is the opening explanation directly owned by the requested section.',240);
  if(ownContent)add(main,content());
  if(childHeading)add(main,node('h3','A Different Child Subsection',180));
  if(!ownContent)add(main,content());
  }
}
const all=[html,body,main,...descendants(main)];
main.querySelectorAll=selector=>selector==='code.function'
  ? descendants(main).filter(n=>n.functionClass)
  : descendants(main).filter(n=>/^H[1-6]$/.test(n.tagName));
for(const cell of all.filter(n=>n.tagName==='TD'))cell.querySelectorAll=()=>descendants(cell).filter(n=>n.tagName==='P');
globalThis.getComputedStyle=n=>({display:n._display||(n===main?mainDisplay:'block'),visibility:n._visibility||'visible',opacity:'1',contentVisibility:'visible'});
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
        ("__DISPLAY__", main_display),
        ("__ZERO_BOX__", zero_main_box),
        ("__GIT_COMMAND__", git_command),
        ("__GIT_CHILD__", git_name_child),
        ("__SHADOW__", mdn_shadow),
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


def test_actual_DOM_open_shadow_code_has_composed_ownership_styles_and_deep_hits():
    task = get_public_training_task("mdn-abortcontroller-abort")
    raw, owner = dom_observation(
        task, mdn_shadow="open", main_display="contents", zero_main_box=True
    )
    signature, shadow = raw["target"]["signature"], raw["target"]["signature_shadow"]
    assert signature["source"] == "open-shadow-root"
    assert "/mdn-code-example[1]/#shadow-root/pre[1]" in signature["path"]
    assert shadow["root_mode"] == "open"
    assert signature["composed_ancestor_paths"][1] == shadow["host"]["path"]
    assert signature["center_hit"]["chain"] == [shadow["host"]["path"], shadow["code"]["path"]]
    assert validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


def test_new_observer_schema_does_not_reinterpret_old_v2_observations():
    task = get_public_training_task("mdn-abortcontroller-abort")
    raw, owner = observation(task)
    raw["schema_id"] = "jev-public-native-training-dom-observation-v2"
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert not verdict["success"] and not verdict["checks"]["observation_schema"]


@pytest.mark.parametrize(
    "mode",
    [
        "closed",
        "light_text_only",
        "wrong_signature",
        "missing_code",
        "hidden_host",
        "hidden_code",
        "offscreen",
        "disconnected",
        "occluded",
    ],
)
def test_actual_DOM_shadow_requires_real_rendered_unobscured_owned_code(mode):
    task = get_public_training_task("mdn-abortcontroller-abort")
    raw, owner = dom_observation(task, mdn_shadow=mode)
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


def test_actual_DOM_shadow_never_borrows_code_after_child_heading():
    task = get_public_training_task("mdn-abortcontroller-abort")
    raw, owner = dom_observation(task, mdn_shadow="open", child_heading=True)
    assert raw["target"]["signature"] is None
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize(
    "mutation", ["host_path", "root_mode", "ancestor", "deep_hit", "heading", "truncated"]
)
def test_shadow_raw_fact_corruption_cannot_override_composed_source_ownership(mutation):
    task = get_public_training_task("mdn-abortcontroller-abort")
    raw, owner = dom_observation(task, mdn_shadow="open")
    target = raw["target"]
    if mutation == "host_path":
        target["signature_shadow"]["host"]["path"] = MAIN + "/mdn-code-example[2]"
    elif mutation == "root_mode":
        target["signature_shadow"]["root_mode"] = "closed"
    elif mutation == "ancestor":
        target["signature"]["composed_ancestor_paths"].pop(1)
    elif mutation == "deep_hit":
        target["signature"]["center_hit"]["chain"] = [target["signature"]["path"]]
    elif mutation == "heading":
        target["signature_shadow"]["owner_heading_path"] = MAIN + "/h3[1]"
    else:
        target["signature_shadow"]["code"]["text_truncated"] = True
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert not verdict["success"]
    assert not verdict["checks"]["signature_source_ownership"]


@pytest.mark.parametrize("visible", [False, True])
def test_actual_dom_overlay_requires_function_name_visible_with_signature_and_explanation(visible):
    task = get_public_training_task("postgres18-string-overlay")
    raw, owner = dom_observation(task, overlay_name_visible=visible)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["success"] is visible
    assert verdict["checks"]["matching_overlay_row"] is visible


@pytest.mark.parametrize(
    "display,accepted", [("contents", True), ("block", False), ("none", False)]
)
def test_actual_dom_display_contents_main_requires_real_visible_descendants(display, accepted):
    task = get_public_training_task("mdn-intersectionobserver-observe")
    raw, owner = dom_observation(task, own_content=True, main_display=display, zero_main_box=True)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["success"] is accepted
    assert verdict["checks"]["main_rendered"] is accepted
    if accepted:
        raw["target"]["signature"]["connected"] = False
        assert not validate_public_training_observation(task, raw, expected_ownership=owner)[
            "success"
        ]


@pytest.mark.parametrize("task_id", ["git-restore-description", "git-worktree-description"])
def test_actual_git_manual_identity_comes_from_own_NAME_paragraph_without_command_h1(task_id):
    task = get_public_training_task(task_id)
    raw, owner = dom_observation(task)
    assert all(node["tag"] != "h1" for node in raw["headings"])
    assert raw["article_identity"]["paragraph"]["text"].startswith(task["main_identity"] + " - ")
    assert validate_public_training_observation(task, raw, expected_ownership=owner)["success"]


@pytest.mark.parametrize("wrong_command", ["git-worktree", "git-restore-extra"])
def test_actual_git_NAME_rejects_other_command_even_with_correct_URL_title_and_DESCRIPTION(
    wrong_command,
):
    task = get_public_training_task("git-restore-description")
    raw, owner = dom_observation(task, git_command=wrong_command)
    verdict = validate_public_training_observation(task, raw, expected_ownership=owner)
    assert verdict["checks"]["target_document_url"]
    assert verdict["checks"]["document_title_identity"]
    assert verdict["checks"]["requested_heading"]
    assert verdict["checks"]["first_own_explanatory_paragraph"]
    assert not verdict["checks"]["main_document_identity"]
    assert not verdict["success"]


def test_actual_git_NAME_cannot_borrow_command_from_a_child_subsection():
    task = get_public_training_task("git-restore-description")
    raw, owner = dom_observation(task, git_name_child=True)
    assert raw["article_identity"]["paragraph"] is None
    assert not validate_public_training_observation(task, raw, expected_ownership=owner)["success"]
