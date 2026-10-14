#!/usr/bin/env python3
"""Run the pinned Jev agent in Chromium and independently judge its final DOM.

Only the browser-harness transport and model HTTP destination are replaced. The
upstream Agent, choose(), snapshot.js, freshness guards, and act() execute as-is.
Fixture expectations stay in the verifier; they never become model action scripts.
Install playwright and run ``playwright install chromium`` before using this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

UPSTREAM_REVISION = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
UPSTREAM_FILES = (
    "__init__.py",
    "agent.py",
    "browser.py",
    "model.py",
    "questions.py",
    "snapshot.js",
    "static/fixture.html",
)
UPSTREAM_TRAVEL_SOURCE = "jev_ultrafast/static/fixture.html"
UPSTREAM_TRAVEL_SHA256 = "e48e7d7b6225b215060692e0a28c3d0b59e465e4b94c1bc510d5f37ff0b8bfbe"
TRAVEL_CHALLENGE_SCOPE = (
    "unseen goal requirements on exposed unchanged upstream HTML, not unseen website"
)
TRAVEL_TRANSFER_SCOPE = (
    "seeded catalogue and initial-control variants derived from exposed upstream HTML; "
    "new identities and requirements, not unseen websites or unchanged upstream fixtures"
)
SEEDED_TASK_IDS = (
    "filter_open",
    "search_filter_open",
    "already_checked",
    "native_select",
    "autocomplete",
    "submit_populated",
    "premature_done",
    "completion_toggle",
    "unapplied_search",
)
DEVELOPMENT_ONLY_TASK_IDS = ("upstream_travel",)
TASK_IDS = SEEDED_TASK_IDS + DEVELOPMENT_ONLY_TASK_IDS


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    task_id: str
    goal: str
    html: str
    expected: dict[str, Any]
    needs_text: bool = False
    query_string: str = ""
    source: dict[str, Any] | None = None

    @property
    def path(self) -> str:
        return f"/tasks/{self.case_id}"

    @property
    def url_suffix(self) -> str:
        return self.path + ("?" + self.query_string if self.query_string else "")


def make_upstream_travel_case(upstream_path: str | Path, repeat: int = 0) -> FixtureCase:
    source = (Path(upstream_path) / UPSTREAM_TRAVEL_SOURCE).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != UPSTREAM_TRAVEL_SHA256:
        raise ValueError("Historical upstream travel fixture differs from its pinned source")
    return FixtureCase(
        case_id=f"development-{repeat:03d}-upstream-travel",
        task_id="upstream_travel",
        goal="Use the destination search and filters to find Design stays in Lisbon with "
        "Free cancellation, then open Casa Flora.",
        html=source.decode("utf-8"),
        expected={
            "hash": "#casa-flora",
            "name": "Casa Flora",
            "category": "Design",
            "free": True,
            "query": "Lisbon",
            "searched": True,
        },
        needs_text=True,
        query_string="scenario=travel",
        source={
            "repository": "https://github.com/browser-use/jev-ultrafast",
            "revision": UPSTREAM_REVISION,
            "path": UPSTREAM_TRAVEL_SOURCE,
            "sha256": digest,
            "scope": "known exposed development regression; excluded from holdout",
        },
    )


def make_travel_goal_challenges(upstream_path: str | Path, repeat: int = 0) -> list[FixtureCase]:
    """Evaluation-only requirements; never added to the default teaching fixtures."""
    original = make_upstream_travel_case(upstream_path, repeat)
    specifications = (
        (
            "Use the destination search and filters to find Design stays in Copenhagen with "
            "Free cancellation, then open The Glasshouse.",
            {
                "hash": "#the-glasshouse",
                "name": "The Glasshouse",
                "category": "Design",
                "free": True,
                "query": "Copenhagen",
                "searched": True,
            },
            True,
        ),
        (
            "Use the destination search and filters to find Nature stays in Lisbon without "
            "enabling Free cancellation, then open Serra Lodge.",
            {
                "hash": "#serra-lodge",
                "name": "Serra Lodge",
                "category": "Nature",
                "free": False,
                "query": "Lisbon",
                "searched": True,
            },
            True,
        ),
        (
            "Open the details for Casa Flora.",
            {"hash": "#casa-flora", "name": "Casa Flora"},
            False,
        ),
    )
    return [
        FixtureCase(
            case_id=f"travel-goal-{repeat:03d}-{index}",
            task_id="upstream_travel",
            goal=goal,
            html=original.html,
            expected=expected,
            needs_text=needs_text,
            query_string=original.query_string,
            source={**original.source, "scope": TRAVEL_CHALLENGE_SCOPE},
        )
        for index, (goal, expected, needs_text) in enumerate(specifications)
    ]


def make_travel_transfer_cases(
    upstream_path: str | Path, *, seed: int, repeat: int = 0
) -> list[FixtureCase]:
    """Evaluation-only seeded catalogues; retain the native page's event handlers.

    This is explicitly derived HTML. The original pinned regression and default
    suites remain byte-identical. The collector must not call this factory.
    """
    original = make_upstream_travel_case(upstream_path, repeat)
    rng = random.Random(f"jev-travel-transfer-v1:{seed}:{repeat}")
    names = rng.sample(
        [
            "Linden House",
            "Cedar Court",
            "Meridian Loft",
            "Harbor Rooms",
            "Willow Retreat",
            "Arden House",
            "Birch Lodge",
            "Copper Yard",
        ],
        3,
    )
    cities = rng.sample(
        ["Porto", "Kyoto", "Utrecht", "Tainan", "Bergen", "Ljubljana", "Quito", "Tallinn"], 3
    )
    categories = rng.sample(["Design", "Nature", "Coastal"], 3)
    places = [
        {
            "name": names[index],
            "city": cities[index],
            "category": categories[index],
            "free": (True, False, bool(rng.getrandbits(1)))[index],
            "price": rng.randrange(85, 260),
            "description": "An independent stay with a quiet courtyard and room to unwind.",
        }
        for index in range(3)
    ]
    start = original.html.index("const places = [")
    end = original.html.index("];", start) + 2
    catalogue = "const places = " + json.dumps(places, ensure_ascii=False) + ";"
    derived = original.html[:start] + catalogue + original.html[end:]
    initial = 'let query = "",\n        category = "all",\n        free = false,\n        searched = false;'
    if derived.count(initial) != 1:
        raise ValueError("Pinned travel initial state cannot be derived unambiguously")
    configured = derived.replace(
        initial,
        "let query = "
        + json.dumps(cities[0])
        + ", category = "
        + json.dumps(categories[0])
        + ", free = false, searched = false;",
    )

    def expected(index, **requirements):
        return {
            "hash": "#" + names[index].lower().replace(" ", "-"),
            "name": names[index],
            **requirements,
        }

    specifications = (
        (
            f"Search for {categories[0]} stays in {cities[0]}, enable Free cancellation, "
            f"apply the search, and open {names[0]}'s details.",
            expected(0, query=cities[0], category=categories[0], free=True, searched=True),
            derived,
            True,
        ),
        (
            f"Show the {categories[1]} category, keep Free cancellation unchecked, "
            f"and open {names[1]}. No destination search is requested.",
            expected(1, category=categories[1], free=False),
            derived,
            False,
        ),
        (
            f"Submit a destination search for {cities[2]} and then open {names[2]}. "
            "There is no category or cancellation-filter requirement.",
            expected(2, query=cities[2], searched=True),
            derived,
            True,
        ),
        (f"Show the detail page for {names[1]}.", expected(1), derived, False),
        (
            f"The {cities[0]} destination and {categories[0]} category are already filled in. "
            f"Submit the search and open {names[0]}; leave Free cancellation unchecked.",
            expected(0, query=cities[0], category=categories[0], free=False, searched=True),
            configured,
            False,
        ),
    )
    return [
        FixtureCase(
            case_id=f"travel-transfer-{repeat:03d}-{index}",
            task_id="upstream_travel",
            goal=goal,
            html=page,
            expected=judge,
            needs_text=needs_text,
            query_string=original.query_string,
            source={
                "repository": original.source["repository"],
                "revision": UPSTREAM_REVISION,
                "parent_path": UPSTREAM_TRAVEL_SOURCE,
                "parent_sha256": UPSTREAM_TRAVEL_SHA256,
                "sha256": hashlib.sha256(page.encode()).hexdigest(),
                "derivation": "replace catalogue data; final case also preconfigures form values",
                "scope": TRAVEL_TRANSFER_SCOPE,
            },
        )
        for index, (goal, judge, page, needs_text) in enumerate(specifications)
    ]


def _document(title: str, body: str, script: str) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:18px sans-serif;max-width:900px;margin:24px}"
        "label{display:block;margin:14px 0}button,input,select{font:inherit;padding:8px}"
        "button{margin:8px}a{display:block;margin:12px}"
        "[role=option]{padding:12px;border:1px solid #aaa;cursor:pointer}"
        "</style><body>"
        f"<h1>{html.escape(title)}</h1>{body}<script>{script}</script></body></html>"
    )


def make_unapplied_search_case(seed=20261014, repeat=0, suite_split="development"):
    """A matching initial result is not evidence that an edited search was applied."""
    if suite_split not in {"development", "holdout"}:
        raise ValueError("suite_split must be development or holdout")
    namespace = f"jev-unapplied-search-v1:{suite_split}:{seed}:{repeat}"
    rng = random.Random(int(hashlib.sha256(namespace.encode()).hexdigest(), 16))
    token = f"{rng.getrandbits(48):012x}"
    if suite_split == "development":
        locations = ["Rivergate", "Lakehaven", "Pine Borough", "Harbor District"]
        collections = ["Research", "Art", "History", "Architecture"]
        names = ["Alder Archive", "Birch Gallery", "Clover Institute", "Elm Pavilion"]
        other_categories = ["Community", "Nature"]
    else:
        locations = ["Hillcrest", "Meadowford", "Oak Quarter", "Bay Crossing"]
        collections = ["Astronomy", "Craft", "Heritage", "Engineering"]
        names = ["Hazel Museum", "Iris Workshop", "Juniper Hall", "Laurel Observatory"]
        other_categories = ["Culture", "Sport"]
    location = rng.choice(locations)
    category = rng.choice(collections)
    record = rng.choice(names)
    categories = [category, *other_categories]
    rng.shuffle(categories)
    labels = (
        ("Search area", "Collection", "Verified only", "Apply search")
        if suite_split == "development"
        else ("Location query", "Record group", "Certified only", "Update results")
    )
    query_label, category_label, checkbox_label, submit_label = labels
    options = '<option value="">All collections</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in categories
    )
    controls = [
        f'<label>{query_label} <input id="query-{token}" type="search" data-search-query></label>',
        f'<label>{category_label} <select id="group-{token}" aria-label="{category_label}">{options}</select></label>',
        f'<label><input id="verified-{token}" type="checkbox">{checkbox_label}</label>',
    ]
    rng.shuffle(controls)
    layout = (
        "display:flex;gap:12px;align-items:center;flex-wrap:wrap"
        if rng.randrange(2)
        else "display:grid;grid-template-columns:1fr 1fr;gap:6px"
    )
    body = (
        f'<form id="form-{token}" style="{layout}">{"".join(controls)}'
        f'<button type="submit">{submit_label}</button></form>'
        f'<p id="applied-{token}" data-benchmark-search-state></p>'
        f'<section id="results-{token}"></section><article id="detail-{token}"></article>'
    )
    records = [
        {"name": record, "city": location, "category": category, "verified": True},
        {"name": "Oak Center", "city": "Outer Quay", "category": category, "verified": True},
        {"name": "Ash Annex", "city": location, "category": other_categories[0], "verified": False},
    ]
    rng.shuffle(records)
    script = f"""
      const records={json.dumps(records)};
      let applied={{query:'',category:'',verified:false,search_committed:false,commit:0}};
      const summary=document.getElementById('applied-{token}');
      const results=document.getElementById('results-{token}');
      const detail=document.getElementById('detail-{token}');
      function render(){{
        summary.dataset.benchmarkSearchState=JSON.stringify(applied);
        summary.textContent=applied.search_committed
          ? 'Applied search: '+(applied.query||'all areas')+'; Collection: '+(applied.category||'all')+
              '; Verified only: '+(applied.verified?'enabled':'off')
          : 'Search not applied. Showing all records.';
        const rows=records.filter(r=>!applied.search_committed ||
          ((!applied.query || r.city.toLowerCase().includes(applied.query.trim().toLowerCase())) &&
           (!applied.category || r.category===applied.category) && (!applied.verified || r.verified)));
        results.replaceChildren();
        for(const record of rows){{
          const card=document.createElement('section');
          const info=document.createElement('p');
          info.textContent=record.name+'; Area: '+record.city+'; Collection: '+record.category+
            '; '+(record.verified?'Verified record':'Unverified record');
          const button=document.createElement('button');button.type='button';button.textContent='Open '+record.name;
          button.onclick=()=>{{
            const opened={{kind:'catalog_detail',name:record.name,query:applied.query,
              category:applied.category,verified:applied.verified,
              search_committed:applied.search_committed,opened_after_commit:applied.commit>0,
              opened_at_commit:applied.commit}};
            detail.textContent='Record details: '+record.name+'; Area: '+record.city+
              '; Applied query: '+(applied.query||'none')+'; Collection: '+(applied.category||'all')+
              '; Verified only: '+(applied.verified?'enabled':'off')+
              '; '+(applied.search_committed?'search applied':'search not applied');
            detail.dataset.benchmarkResult=JSON.stringify(opened);
            location.hash='record-'+record.name.toLowerCase().replaceAll(' ','-');
          }};
          card.append(info,button);results.append(card);
        }}
      }}
      document.getElementById('form-{token}').onsubmit=e=>{{
        e.preventDefault();
        applied={{query:document.getElementById('query-{token}').value,
          category:document.getElementById('group-{token}').value,
          verified:document.getElementById('verified-{token}').checked,
          search_committed:true,commit:applied.commit+1}};
        detail.replaceChildren();detail.removeAttribute('data-benchmark-result');
        location.hash='';render();
      }};
      render();
    """
    return FixtureCase(
        case_id=f"{suite_split}-{repeat:03d}-unapplied_search-{token}",
        task_id="unapplied_search",
        goal=f"Search for records in {location}, set {category_label} to {category}, enable {checkbox_label}, "
        f"apply the search and filters, then open the details for {record}. "
        "Stop when those details reflect the applied search.",
        html=_document(
            "Record catalog" if suite_split == "development" else "Directory lookup", body, script
        ),
        expected={
            "kind": "catalog_detail",
            "name": record,
            "query": location,
            "category": category,
            "verified": True,
            "search_committed": True,
            "opened_after_commit": True,
        },
        needs_text=True,
    )


def make_cases(
    seed: int = 20261014,
    repeat: int = 0,
    suite_split: str = "development",
    upstream_path: str | Path | None = None,
) -> list[FixtureCase]:
    """Freeze nine seeded cases; optionally append exposed upstream travel on development."""
    if suite_split not in {"development", "holdout"}:
        raise ValueError("suite_split must be development or holdout")
    namespace = f"jev-dom-v1:{suite_split}:{seed}:{repeat}"
    rng = random.Random(int(hashlib.sha256(namespace.encode()).hexdigest(), 16))
    token = f"{rng.getrandbits(48):012x}"
    region = rng.choice(["North Coast", "East Valley", "South Hills", "West Lake"])
    record = rng.choice(["Cedar Observatory", "Juniper Library", "Maple Workshop", "Willow Studio"])
    code = f"{rng.randrange(10000, 99999)}"
    selected = rng.choice(["Priority review", "Standard review", "Deferred review"])
    city = rng.choice(["Taipei Harbor", "Cedar Ridge", "Bristol Park", "Osaka Riverside"])
    expected_base = {"fixture_token": token}
    cases = []

    def add(task_id, goal, body, script, expected, needs_text=False, title=None):
        cases.append(
            FixtureCase(
                case_id=f"{suite_split}-{repeat:03d}-{task_id}-{token}",
                task_id=task_id,
                goal=goal,
                html=_document(title or task_id.replace("_", " ").title(), body, script),
                expected={**expected_base, **expected},
                needs_text=needs_text,
            )
        )

    # Candidate order and DOM IDs vary independently of the correct action.
    regions = [region, "Central Plain", "Outer Island"]
    rng.shuffle(regions)
    options = '<option value="">Choose region</option>' + "".join(
        f'<option value="r{i}">{html.escape(value)}</option>' for i, value in enumerate(regions)
    )
    for task_id, use_search in (("filter_open", False), ("search_filter_open", True)):
        search = (
            f'<label>Search records <input id="q-{token}" type="search"></label>'
            if use_search
            else ""
        )
        body = (
            f'<form id="form-{token}">{search}<label>Region '
            f'<select id="region-{token}">{options}</select></label>'
            f'<label><input id="verified-{token}" type="checkbox">Verified only</label>'
            '<button type="submit">Apply filters</button></form>'
            '<div id="results" aria-live="polite">No results loaded.</div>'
            '<article id="article"></article>'
        )
        script = f"""
          const form=document.getElementById('form-{token}');
          form.addEventListener('submit', e=>{{e.preventDefault();
            const region=document.getElementById('region-{token}');
            const verified=document.getElementById('verified-{token}').checked;
            const query=document.getElementById('q-{token}')?.value||'';
            const correct=region.selectedOptions[0].textContent==={json.dumps(region)}
              && verified && ({json.dumps(not use_search)} ||
                  query.toLowerCase().includes({json.dumps(record.lower())}));
            const rows=correct ? [{json.dumps(record)}, 'Unverified Annex'] : ['Other Record'];
            const results=document.getElementById('results');results.replaceChildren();
            results.append(document.createTextNode('Results: '+rows.length));
            for(const name of rows){{
              const link=document.createElement('a');link.href='#record';link.textContent='Open '+name;
              link.addEventListener('click',()=>{{
                const article=document.getElementById('article');
                article.textContent='Record details: '+name+'; Region: '+region.selectedOptions[0].textContent;
                article.dataset.benchmarkResult=JSON.stringify({{kind:'record',name,
                  region:region.selectedOptions[0].textContent,verified,query}});
              }});results.append(link);
            }}
          }});
        """
        goal = (
            (f"Search for {record}, " if use_search else "")
            + f"filter records to {region} with Verified only enabled, apply the filters, "
            + f"and open the details for {record}. Stop when those details are visible."
        )
        add(
            task_id,
            goal,
            body,
            script,
            {"kind": "record", "name": record, "region": region, "verified": True},
            use_search,
        )

    checkbox_labels = ["Weekly newsletter", "Product tips"]
    rng.shuffle(checkbox_labels)
    checkbox_body = "".join(
        f'<label><input id="{i}-{token}" type="checkbox" checked>{label}</label>'
        for i, label in enumerate(checkbox_labels)
    )
    add(
        "already_checked",
        "Keep Weekly newsletter enabled, disable Product tips, and save preferences. "
        "Stop only after the saved confirmation appears.",
        f'<form>{checkbox_body}<button>Save preferences</button></form><p id="receipt"></p>',
        "document.querySelector('form').onsubmit=e=>{e.preventDefault();"
        "const values=Object.fromEntries([...document.querySelectorAll('label')]"
        ".map(l=>[l.textContent.trim(),l.querySelector('input').checked]));"
        "const receipt=document.getElementById('receipt');"
        "receipt.textContent='Preferences saved: '+JSON.stringify(values);"
        "receipt.dataset.benchmarkResult=JSON.stringify({kind:'preferences',...values});};",
        {"kind": "preferences", "Weekly newsletter": True, "Product tips": False},
    )

    priorities = ["Priority review", "Standard review", "Deferred review"]
    rng.shuffle(priorities)
    select_options = '<option value="">Choose review mode</option>' + "".join(
        f'<option value="mode-{i}-{token}">{label}</option>' for i, label in enumerate(priorities)
    )
    add(
        "native_select",
        f"Set Review mode to {selected} and submit it. Verify the saved mode.",
        f"<form><label>Review mode <select>{select_options}</select></label>"
        '<button>Save review mode</button></form><p id="receipt"></p>',
        "document.querySelector('form').onsubmit=e=>{e.preventDefault();"
        "const mode=document.querySelector('select').selectedOptions[0].textContent;"
        "const receipt=document.getElementById('receipt');receipt.textContent='Saved review mode: '+mode;"
        "receipt.dataset.benchmarkResult=JSON.stringify({kind:'review',mode});};",
        {"kind": "review", "mode": selected},
    )

    city_choices = [city, city.split()[0] + " Airport", "Other Town"]
    rng.shuffle(city_choices)
    add(
        "autocomplete",
        f"Choose {city} using the Destination autocomplete suggestions, then confirm the destination. "
        "Stop after the confirmed destination is visible.",
        f'<form><label>Destination <input id="city-{token}" role="combobox" '
        'aria-controls="suggestions" aria-expanded="false" autocomplete="off"></label>'
        '<div id="suggestions" role="listbox"></div><button>Confirm destination</button></form>'
        '<p id="receipt"></p>',
        f"""
          const field=document.getElementById('city-{token}'), options=document.getElementById('suggestions');
          let chosen=null;
          field.addEventListener('input',()=>{{chosen=null;options.replaceChildren();
            field.setAttribute('aria-expanded','true');
            for(const name of {json.dumps(city_choices)}){{
              if(!field.value.trim() || !name.toLowerCase().includes(field.value.toLowerCase().trim()))continue;
              const option=document.createElement('div');option.setAttribute('role','option');
              option.textContent=name;option.addEventListener('click',()=>{{chosen=name;field.value=name;
                options.replaceChildren();field.setAttribute('aria-expanded','false');}});options.append(option);
            }}
          }});
          document.querySelector('form').onsubmit=e=>{{e.preventDefault();
            const receipt=document.getElementById('receipt');
            receipt.textContent=chosen ? 'Confirmed destination: '+chosen : 'Choose a destination suggestion first.';
            if(chosen)receipt.dataset.benchmarkResult=JSON.stringify({{kind:'destination',city:chosen}});
          }};
        """,
        {"kind": "destination", "city": city},
        True,
    )
    add(
        "submit_populated",
        f"Submit the enrollment form with Enrollment code {code}, which is already filled in. "
        "Stop only after enrollment is confirmed.",
        f'<form><label>Enrollment code <input value="{code}"></label>'
        '<button>Submit enrollment</button></form><p id="receipt">Enrollment not submitted.</p>',
        "document.querySelector('form').onsubmit=e=>{e.preventDefault();"
        "const code=document.querySelector('input').value, receipt=document.getElementById('receipt');"
        "receipt.textContent='Enrollment confirmed: '+code;"
        "receipt.dataset.benchmarkResult=JSON.stringify({kind:'enrollment',code});};",
        {"kind": "enrollment", "code": code},
    )
    report = f"Usage report {code}"
    add(
        "premature_done",
        f"Activate {report}. Stop only when that report is visibly active.",
        f'<p id="receipt">{report} is inactive.</p><button>Activate report</button>',
        f"document.querySelector('button').onclick=()=>{{const receipt=document.getElementById('receipt');"
        f"receipt.textContent={json.dumps(report + ' is active.')};"
        "receipt.dataset.benchmarkResult=JSON.stringify({kind:'report',active:true});};",
        {"kind": "report", "active": True},
    )
    completion, undo, before, after = (
        (
            "Mark complete",
            "Reset completion",
            "Completion not yet confirmed",
            "Completion confirmed",
        )
        if suite_split == "development"
        else (
            "Confirm approval",
            "Undo approval",
            "Approval awaiting confirmation",
            "Approval confirmed",
        )
    )
    add(
        "completion_toggle",
        f"Click the {completion} button. Stop when {after} is visibly shown.",
        f'<p id="receipt">{before}</p><button id="finish-{token}">{completion}</button>'
        f'<button id="undo-{token}" disabled>{undo}</button>',
        f"""
          const receipt=document.getElementById('receipt');
          document.getElementById('finish-{token}').onclick=()=>{{
            receipt.textContent={json.dumps(after)};
            receipt.dataset.benchmarkResult=JSON.stringify({{kind:'completion',completed:true}});
            document.getElementById('finish-{token}').disabled=true;
            document.getElementById('undo-{token}').disabled=false;
          }};
          document.getElementById('undo-{token}').onclick=()=>{{
            receipt.textContent={json.dumps(before)};
            receipt.dataset.benchmarkResult=JSON.stringify({{kind:'completion',completed:false}});
            document.getElementById('finish-{token}').disabled=false;
            document.getElementById('undo-{token}').disabled=true;
          }};
        """,
        {"kind": "completion", "completed": True},
        title="Local completion fixture"
        if suite_split == "development"
        else "Approval confirmation",
    )
    if suite_split == "development" and upstream_path is not None:
        cases.append(make_upstream_travel_case(upstream_path, repeat))
    # An independent namespace preserves every byte of the original nine-case pilot.
    cases.append(make_unapplied_search_case(seed, repeat, suite_split))
    return cases


def verify_fixture(browser, case: FixtureCase) -> dict[str, Any]:
    """Read final DOM afresh, without trusting DONE, agent history, or chosen targets."""
    if case.task_id == "upstream_travel":
        # These lexical variables belong to the unchanged fixture's actual event handlers.
        # query/searched distinguish a submitted search from merely filling Destination.
        observed = browser.evaluate("""(() => {
          const heading=document.querySelector('#content h1');
          return {result:{hash:location.hash,name:heading?.innerText,
            category:typeof category==='string' ? category : null,
            free:typeof free==='boolean' ? free : null,
            query:typeof query==='string' ? query : null,
            searched:typeof searched==='boolean' ? searched : null},
            result_visible:!!heading && heading.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}),
            text:document.body.innerText,url:location.href,
            typed_destination:document.querySelector('#destination')?.value ?? null,
            filters_note:document.querySelector('.fixture-note')?.innerText ?? null};
        })()""")
    elif case.task_id == "unapplied_search":
        observed = browser.evaluate("""(() => {
          const detail=document.querySelector('[data-benchmark-result]');
          const summary=document.querySelector('[data-benchmark-search-state]');
          return {result:detail ? JSON.parse(detail.dataset.benchmarkResult) : null,
            result_visible:!!detail && detail.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}),
            applied:summary ? JSON.parse(summary.dataset.benchmarkSearchState) : null,
            typed_query:document.querySelector('[data-search-query]')?.value,
            text:document.body.innerText,url:location.href};
        })()""")
    else:
        observed = browser.evaluate("""(() => {
      const result=document.querySelector('[data-benchmark-result]');
      return {result:result ? JSON.parse(result.dataset.benchmarkResult) : null,
        result_visible:!!result && result.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}),
        text:document.body.innerText, url:location.href,
        fields:[...document.querySelectorAll('input,select')].map(e=>({
          label:[...(e.labels||[])].map(l=>l.innerText.trim()).join(' '),
          value:e.value, checked:e.checked,
          selected_label:e.tagName==='SELECT' ? e.selectedOptions[0]?.textContent : null}))};
        })()""")
    result = observed.get("result") or {}

    def matches(key, actual, expected):
        if (
            case.task_id in {"unapplied_search", "upstream_travel"}
            and key == "query"
            and isinstance(actual, str)
        ):
            # The application's own search compares trimmed, case-insensitive text.
            return actual.strip().casefold() == expected.strip().casefold()
        return actual == expected

    mismatches = {
        key: {"expected": value, "observed": result.get(key)}
        for key, value in case.expected.items()
        if key != "fixture_token" and (key not in result or not matches(key, result[key], value))
    }
    if not observed.get("result_visible"):
        mismatches["visible_result"] = {"expected": True, "observed": False}
    if case.task_id == "already_checked":
        for field in observed.get("fields", []):
            expected = case.expected.get(field["label"])
            if expected is not None and field["checked"] != expected:
                mismatches[f"current_{field['label']}"] = {
                    "expected": expected,
                    "observed": field["checked"],
                }
    if case.task_id == "native_select":
        fields = observed.get("fields", [])
        current = fields[0].get("selected_label") if fields else None
        if current != case.expected["mode"]:
            mismatches["current_mode"] = {"expected": case.expected["mode"], "observed": current}
    if case.task_id == "unapplied_search":
        applied = observed.get("applied") or {}
        for key in ("query", "category", "verified", "search_committed"):
            if not matches(key, applied.get(key), case.expected[key]):
                mismatches[f"applied_{key}"] = {
                    "expected": case.expected[key],
                    "observed": applied.get(key),
                }
        if not applied.get("commit") or result.get("opened_at_commit") != applied.get("commit"):
            mismatches["opened_at_current_commit"] = {
                "expected": applied.get("commit"),
                "observed": result.get("opened_at_commit"),
            }
    return {"success": not mismatches, "observed": observed, "mismatches": mismatches}


@contextmanager
def fixture_server(cases: list[FixtureCase]):
    pages = {case.path: case.html.encode() for case in cases}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            page = pages.get(urlsplit(self.path).path)
            self.send_response(200 if page else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page or b"Not found")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class PlaywrightCDP:
    """A browser-harness CDP transport, preserving the upstream browser code."""

    def __init__(self, browser):
        self.browser = browser
        self.targets = {}
        self.sessions = {}
        self.calls = 0

    def cdp(self, method, session_id=None, **params):
        self.calls += 1
        if method == "Target.createTarget":
            target = f"target-{len(self.targets) + 1}"
            page = self.browser.new_page()
            page.goto(params.get("url", "about:blank"))
            self.targets[target] = page
            return {"targetId": target}
        if method == "Target.attachToTarget":
            target = params["targetId"]
            session = f"session-{len(self.sessions) + 1}"
            self.sessions[session] = self.targets[target].context.new_cdp_session(
                self.targets[target]
            )
            return {"sessionId": session}
        if method == "Target.closeTarget":
            page = self.targets.pop(params["targetId"], None)
            if page:
                page.close()
            return {"success": True}
        if session_id not in self.sessions:
            raise ValueError("Missing owned CDP session")
        return self.sessions[session_id].send(method, params)


def upstream_manifest(upstream_path: str | Path) -> dict[str, Any]:
    root = Path(upstream_path).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    revision = result.stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}; found {revision}")
    files = {
        str(Path("jev_ultrafast") / name): hashlib.sha256(
            (root / "jev_ultrafast" / name).read_bytes()
        ).hexdigest()
        for name in UPSTREAM_FILES
    }
    # A pinned commit alone does not prove that the working source is unchanged.
    for relative in files:
        original = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"], check=True, capture_output=True
        ).stdout
        if hashlib.sha256(original).hexdigest() != files[relative]:
            raise ValueError(f"Upstream source has local changes: {relative}")
    return {
        "repository": "https://github.com/browser-use/jev-ultrafast",
        "revision": revision,
        "source_sha256": files,
    }


@contextmanager
def load_upstream(upstream_path: str | Path, transport: PlaywrightCDP):
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "browser_harness"
        or name.startswith("browser_harness.")
        or name == "jev_ultrafast"
        or name.startswith("jev_ultrafast.")
    }
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType("browser_harness")
    package.__path__ = []
    admin = types.ModuleType("browser_harness.admin")
    admin.ensure_daemon = lambda: None
    helpers = types.ModuleType("browser_harness.helpers")
    helpers.cdp = transport.cdp
    sys.modules.update(
        {
            "browser_harness": package,
            "browser_harness.admin": admin,
            "browser_harness.helpers": helpers,
        }
    )
    sys.path.insert(0, str(Path(upstream_path).resolve()))
    try:
        yield (
            importlib.import_module("jev_ultrafast.agent"),
            importlib.import_module("jev_ultrafast.model"),
        )
    finally:
        for name in list(sys.modules):
            if (
                name == "browser_harness"
                or name.startswith("browser_harness.")
                or name == "jev_ultrafast"
                or name.startswith("jev_ultrafast.")
            ):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path.pop(0)


def _endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Model endpoint must be an explicit HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint credentials and query parameters are not allowed")
    return value.rstrip("/")


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in {"authorization", "api_key", "apikey", "access_token", "password"}
                else _redact(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


@contextmanager
def model_adapter(
    agent_module,
    model_module,
    *,
    endpoint,
    model,
    api_key,
    text_endpoint=None,
    text_model=None,
    text_api_key=None,
    text_helper=None,
):
    variables = {"TYPESAFE_MODEL": model, "TYPESAFE_API_KEY": api_key or "local-benchmark"}
    if text_endpoint:
        if not text_model:
            raise ValueError("An explicit text model is required with text_endpoint")
        variables.update(
            TEXT_MODEL_BASE_URL=_endpoint(text_endpoint),
            TEXT_MODEL=text_model,
            TEXT_MODEL_API_KEY=text_api_key or "local-benchmark",
        )
    saved_env = {key: os.environ.get(key) for key in variables}
    os.environ.update(variables)
    original_post = model_module.post_json
    original_text = agent_module.field_text
    client = getattr(model_module, "CLIENT", None)
    original_http_post = client.post if client is not None else None
    records = []
    active_record = None
    secrets = [
        api_key or "",
        text_api_key or "",
        os.environ.get("TYPESAFE_API_KEY", ""),
        os.environ.get("TEXT_MODEL_API_KEY", ""),
    ]

    def post(url, key, body):
        nonlocal active_record
        policy = url == "https://api.typesafe.ai/v1/systemone"
        destination = endpoint if policy else url
        request = {**body, "model": model} if policy else body
        record = {
            "kind": "policy" if policy else "text",
            "endpoint": destination,
            "request": _redact(request, secrets),
            "started_unix": time.time(),
        }
        records.append(record)
        active_record = record
        started = time.perf_counter()
        try:
            response = original_post(destination, key, request)
            record["response"] = _redact(response, secrets)
            if policy and (
                not isinstance(response.get("model"), str) or not response["model"].strip()
            ):
                raise ValueError("S1 endpoint omitted the reported model identity")
            return response
        except Exception as exc:
            if hasattr(exc, "generation"):
                record["generation"] = _redact(exc.generation, secrets)
            record["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            active_record = None

    def http_post(url, **kwargs):
        # Capture each real retry, while preserving upstream timeout/retry semantics.
        transport_record = {"endpoint": str(url), "request": _redact(kwargs.get("json"), secrets)}
        if active_record is not None:
            active_record.setdefault("transport_attempts", []).append(transport_record)
        started = time.perf_counter()
        try:
            response = original_http_post(url, **kwargs)
            transport_record["http_status"] = response.status_code
            try:
                transport_record["response"] = _redact(response.json(), secrets)
            except ValueError:
                transport_record["response_text"] = _redact(response.text, secrets)
            return response
        except Exception as exc:
            transport_record["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)
            raise
        finally:
            transport_record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def helper(context):
        record = {
            "kind": "text_callback",
            "request": _redact(context, secrets),
            "started_unix": time.time(),
        }
        records.append(record)
        started = time.perf_counter()
        try:
            text, metadata = text_helper(context)
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                raise ValueError("Text callback returned an invalid field value")
            if not isinstance(metadata.get("model"), str) or not metadata["model"]:
                raise ValueError("Text callback must report its actual model identity")
            metadata = {**metadata, "latency_ms": round((time.perf_counter() - started) * 1000)}
            record["response"] = _redact({"text": text, "metadata": metadata}, secrets)
            return text, metadata
        except Exception as exc:
            if hasattr(exc, "generation"):
                record["generation"] = _redact(exc.generation, secrets)
            record["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def missing_helper(_context):
        raise ValueError(
            "TYPE_TEXT requires an explicit text_endpoint/model or a real text_helper callback"
        )

    model_module.post_json = post
    if client is not None:
        client.post = http_post
    if text_helper is not None:
        agent_module.field_text = helper
    elif not text_endpoint:
        agent_module.field_text = missing_helper
    try:
        yield records, secrets
    finally:
        model_module.post_json = original_post
        if client is not None:
            client.post = original_http_post
        agent_module.field_text = original_text
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, value: Any, secrets=()):
    path.write_text(json.dumps(_redact(value, list(secrets)), indent=2, ensure_ascii=False) + "\n")


@contextmanager
def _record_runner_error(output, secrets):
    try:
        yield
    except Exception as exc:
        _write_json(
            output / "runner_error.json", {"error": f"{type(exc).__name__}: {exc}"}, secrets
        )
        raise


def run_benchmark(
    upstream_path,
    endpoint,
    model,
    output_dir,
    *,
    repeats=1,
    max_steps=16,
    seed=20261014,
    suite_split="development",
    task_ids=None,
    travel_challenges_only=False,
    travel_transfer_only=False,
    api_key=None,
    text_endpoint=None,
    text_model=None,
    text_api_key=None,
    text_helper: Callable | None = None,
    chromium_executable=None,
    live_url=None,
    live_goal=None,
    live_expected_title=None,
):
    """Run all attempts, preserving failures; an existing output folder is never overwritten.

    ``text_helper(context)`` must perform actual generation and return ``(text,
    metadata)`` with an actual model identity. No fixture expectation reaches it.
    The process-global upstream adapters require sequential calls in one process.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An explicit model identifier is required")
    if repeats < 1 or max_steps < 1:
        raise ValueError("repeats and max_steps must be positive")
    endpoint = _endpoint(endpoint)
    manifest = upstream_manifest(upstream_path)
    if suite_split not in {"development", "holdout"}:
        raise ValueError("suite_split must be development or holdout")
    if travel_challenges_only:
        if suite_split != "development":
            raise ValueError("Travel goal challenges require the development split")
        if task_ids is not None or any(
            value is not None for value in (live_url, live_goal, live_expected_title)
        ):
            raise ValueError("Travel goal challenges cannot be combined with tasks or live flags")
    if travel_transfer_only and (
        suite_split != "development"
        or travel_challenges_only
        or task_ids is not None
        or any(value is not None for value in (live_url, live_goal, live_expected_title))
    ):
        raise ValueError(
            "Travel transfer requires development and cannot mix suites, tasks or live flags"
        )
    defaults = TASK_IDS if suite_split == "development" else SEEDED_TASK_IDS
    chosen = (
        {"upstream_travel"}
        if travel_challenges_only or travel_transfer_only
        else set(defaults if task_ids is None else task_ids)
    )
    if chosen - set(TASK_IDS):
        raise ValueError(f"Unknown tasks: {sorted(chosen - set(TASK_IDS))}")
    if suite_split == "holdout" and chosen.intersection(DEVELOPMENT_ONLY_TASK_IDS):
        raise ValueError("Historical upstream_travel is development only and cannot run as holdout")
    if not chosen and not live_url:
        raise ValueError("Choose at least one task")
    if live_url:
        parsed = urlsplit(live_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "en.wikipedia.org"
            or not parsed.path.startswith("/wiki/")
        ):
            raise ValueError("Live tasks are restricted to read-only English Wikipedia articles")
        if not live_goal or not live_expected_title:
            raise ValueError("A live task needs a goal and independently expected article title")
    from playwright.sync_api import sync_playwright

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        case
        for repeat in range(repeats)
        for case in (
            make_travel_transfer_cases(upstream_path, seed=seed, repeat=repeat)
            if travel_transfer_only
            else make_travel_goal_challenges(upstream_path, repeat)
            if travel_challenges_only
            else make_cases(seed, repeat, suite_split, upstream_path=upstream_path)
        )
        if case.task_id in chosen
    ]
    configuration = {
        "upstream": manifest,
        "model_requested": model,
        "endpoint": endpoint,
        "repeats": repeats,
        "max_steps": max_steps,
        "seed": seed,
        "suite_split": suite_split,
        "task_ids": sorted(chosen),
        "travel_challenges_only": travel_challenges_only,
        "travel_transfer_only": travel_transfer_only,
        "suite_scope": TRAVEL_TRANSFER_SCOPE
        if travel_transfer_only
        else TRAVEL_CHALLENGE_SCOPE
        if travel_challenges_only
        else "Nine resettable local DOM templates; holdout has disjoint seeded variants "
        "of the same nine templates, not unseen websites. The original exposed upstream_travel "
        "regression is development only and excluded from holdout",
        "text_model_requested": text_model,
        "text_callback": text_helper is not None,
        "live_url": live_url,
        "timing_boundary": "Agent construction through independent "
        "verification, final HTML capture/write and browser cleanup; "
        "excludes JSON evidence writes",
        "transport": "headless Chromium through Playwright CDP; unchanged upstream browser.act",
        "fixture_manifest": [
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "html_sha256": hashlib.sha256(case.html.encode()).hexdigest(),
                "goal_sha256": hashlib.sha256(case.goal.encode()).hexdigest(),
                "needs_text": case.needs_text,
                "query_string": case.query_string,
                "source": case.source,
            }
            for case in cases
        ],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_json(output / "configuration.json", configuration)
    for case in cases:
        case_dir = output / "fixtures" / case.case_id
        case_dir.mkdir(parents=True)
        (case_dir / "page.html").write_text(case.html)
        _write_json(
            case_dir / "judge.json",
            {
                "goal": case.goal,
                "expected": case.expected,
                "needs_text": case.needs_text,
                "query_string": case.query_string,
                "source": case.source,
            },
        )
    outcomes = []
    startup_secrets = [api_key or "", text_api_key or "", os.environ.get("TEXT_MODEL_API_KEY", "")]
    with (
        _record_runner_error(output, startup_secrets),
        sync_playwright() as playwright,
        fixture_server(cases) as fixture_base,
    ):
        launch = {"headless": True}
        if chromium_executable:
            launch["executable_path"] = str(chromium_executable)
        browser = playwright.chromium.launch(**launch)
        transport = PlaywrightCDP(browser)
        try:
            with load_upstream(upstream_path, transport) as (agent_module, model_module):
                with model_adapter(
                    agent_module,
                    model_module,
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    text_endpoint=text_endpoint,
                    text_model=text_model,
                    text_api_key=text_api_key,
                    text_helper=text_helper,
                ) as (records, secrets):
                    task_specs = [
                        (case.case_id, case.goal, fixture_base + case.url_suffix, case)
                        for case in cases
                    ]
                    if live_url:
                        task_specs.append(("live-wikipedia", live_goal, live_url, None))
                    for case_id, goal, url, case in task_specs:
                        attempt = output / "attempts" / case_id
                        attempt.mkdir(parents=True, exist_ok=False)
                        started = time.perf_counter()
                        record_start, call_start = len(records), transport.calls
                        agent, error, verification = (
                            None,
                            None,
                            {"success": False, "error": "Agent did not start"},
                        )
                        close_error, verification_elapsed_ms = None, None
                        error_stage, verification_error, dom_error = None, None, None
                        snapshots = []
                        try:
                            agent = agent_module.Agent(url, goal, screenshots=False)
                            snapshots.append(_redact(agent.snapshot(), secrets))
                            for _ in range(max_steps):
                                if agent.state["status"] in {"done", "blocked"}:
                                    break
                                snapshots.append(_redact(agent.command("tick"), secrets))
                            if agent.state["status"] not in {"done", "blocked"}:
                                error = f"Runner stopped at the {max_steps}-decision budget"
                                error_stage = "decision_budget"
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            error_stage = "execution"
                        finally:
                            if agent is not None:
                                verification_started = time.perf_counter()
                                try:
                                    if case:
                                        verification = verify_fixture(agent.browser, case)
                                    else:
                                        observed = agent.browser.evaluate("""(() => ({url:location.href,
                                          title:document.title, heading:document.querySelector('#firstHeading')?.innerText,
                                          article:!!document.querySelector('#mw-content-text .mw-parser-output')}))()""")
                                        verification = {
                                            "success": observed.get("heading")
                                            == live_expected_title
                                            and observed.get("article") is True,
                                            "observed": observed,
                                            "expected_title": live_expected_title,
                                        }
                                except Exception as exc:
                                    verification_error = f"{type(exc).__name__}: {exc}"
                                    verification = {
                                        "success": None,
                                        "error": verification_error,
                                    }
                                    error = (
                                        f"{error}; verification failed: {verification_error}"
                                        if error
                                        else verification_error
                                    )
                                    error_stage = error_stage or "verification"
                                verification_elapsed_ms = round(
                                    (time.perf_counter() - verification_started) * 1000, 3
                                )
                                dom_stage = "dom_capture"
                                try:
                                    dom = agent.browser.evaluate(
                                        "document.documentElement.outerHTML"
                                    )
                                    dom_stage = "dom_write"
                                    (attempt / "final.html").write_text(_redact(dom or "", secrets))
                                except Exception as exc:
                                    message = f"{type(exc).__name__}: {exc}"
                                    dom_error = {"stage": dom_stage, "message": message}
                                    error = (
                                        f"{error}; {dom_stage} failed: {message}"
                                        if error
                                        else message
                                    )
                                    error_stage = error_stage or dom_stage
                                final_state = agent.snapshot()
                                try:
                                    agent.close()
                                except Exception as exc:
                                    close_error = f"{type(exc).__name__}: {exc}"
                                    error = (
                                        f"{error}; close failed: {close_error}"
                                        if error
                                        else f"Close failed: {close_error}"
                                    )
                                    error_stage = error_stage or "close"
                            else:
                                final_state = None
                            model_calls = records[record_start:]
                            policy_calls = [
                                record for record in model_calls if record["kind"] == "policy"
                            ]
                            identities = sorted(
                                {
                                    record["response"]["model"]
                                    for record in policy_calls
                                    if isinstance(record.get("response", {}).get("model"), str)
                                }
                            )
                            outcome = {
                                "case_id": case_id,
                                "task_id": case.task_id if case else "live_wikipedia",
                                "goal": goal,
                                "status": final_state["status"] if final_state else "error",
                                "goal_satisfied": None
                                if verification["success"] is None
                                else bool(verification["success"]),
                                "success": _task_success(verification, final_state, error),
                                "premature_done": _premature_done(verification, final_state),
                                "error": error,
                                "error_stage": error_stage,
                                "verification_error": verification_error,
                                "dom_error": dom_error,
                                "close_error": close_error,
                                "verification": verification,
                                "verification_elapsed_ms": verification_elapsed_ms,
                                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                                "policy_calls": len(policy_calls),
                                "text_calls": len(model_calls) - len(policy_calls),
                                "browser_protocol_calls": transport.calls - call_start,
                                "model_requested": model,
                                "models_reported": identities,
                                "action_count": len(final_state["history"]) if final_state else 0,
                            }
                            _write_json(attempt / "outcome.json", outcome, secrets)
                            _write_json(attempt / "model_calls.json", model_calls, secrets)
                            _write_json(attempt / "states.json", snapshots, secrets)
                            _write_json(attempt / "final_state.json", final_state, secrets)
                            outcomes.append(outcome)
                            _write_json(
                                output / "summary.json", _summary(outcomes, configuration), secrets
                            )
        finally:
            browser.close()
    return _summary(outcomes, configuration)


def _task_success(verification, final_state, error):
    return bool(
        verification.get("success")
        and final_state
        and final_state["status"] == "done"
        and error is None
    )


def _premature_done(verification, final_state):
    return bool(
        final_state
        and final_state["status"] == "done"
        and verification.get("success") is False
        and not verification.get("error")
    )


def _summary(outcomes, configuration):
    successes = sum(item["success"] for item in outcomes)
    return {
        "configuration": configuration,
        "attempts": len(outcomes),
        "successes": successes,
        "goals_satisfied": sum(item["goal_satisfied"] is True for item in outcomes),
        "success_rate": successes / len(outcomes) if outcomes else 0,
        "premature_done_count": sum(item["premature_done"] for item in outcomes),
        "policy_calls": sum(item["policy_calls"] for item in outcomes),
        "text_calls": sum(item["text_calls"] for item in outcomes),
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-path", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--suite-split", choices=["development", "holdout"], default="development")
    parser.add_argument("--tasks", nargs="+", choices=TASK_IDS)
    parser.add_argument("--travel-challenges-only", action="store_true")
    parser.add_argument("--travel-transfer-only", action="store_true")
    parser.add_argument("--api-key-env", default="S1_API_KEY")
    parser.add_argument("--text-endpoint")
    parser.add_argument("--text-model")
    parser.add_argument("--text-api-key-env", default="TEXT_MODEL_API_KEY")
    parser.add_argument("--chromium-executable")
    parser.add_argument("--live-url")
    parser.add_argument("--live-goal")
    parser.add_argument("--live-expected-title")
    args = parser.parse_args()
    if args.travel_challenges_only and (
        args.suite_split != "development"
        or args.travel_transfer_only
        or args.tasks is not None
        or any(
            value is not None for value in (args.live_url, args.live_goal, args.live_expected_title)
        )
    ):
        parser.error(
            "travel-challenges-only requires development and cannot use tasks or live flags"
        )
    if args.travel_transfer_only and (
        args.suite_split != "development"
        or args.travel_challenges_only
        or args.tasks is not None
        or any(
            value is not None for value in (args.live_url, args.live_goal, args.live_expected_title)
        )
    ):
        parser.error(
            "travel-transfer-only requires development and cannot mix suites, tasks or live flags"
        )
    values = vars(args)
    values["task_ids"] = values.pop("tasks")
    values["api_key"] = os.environ.get(values.pop("api_key_env"))
    values["text_api_key"] = os.environ.get(values.pop("text_api_key_env"))
    print(json.dumps(run_benchmark(**values), ensure_ascii=False))


if __name__ == "__main__":
    main()
