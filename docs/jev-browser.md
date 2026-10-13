# Real-browser evaluation with Jev

This campaign evaluates task completion in a real, isolated Chromium browser,
using browser-use/jev-ultrafast at
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. It keeps the upstream Agent,
operation/target policy, DOM snapshot and guarded executor. Playwright replaces
only the browser-harness transport. An explicit model destination replaces the
hardcoded TypeSafe endpoint.

The prior [October 13 local report](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/codex/local-jev-benchmark-20261013/docs/local-jev-benchmark-20261013/report.md)
recorded 0/3 travel and 0/5 completion tasks. BoolQ and public231 decision accuracy
are different populations and do not establish browser task success.

## October 14 local pilot

Seven completed local runs used the original nine development tasks, with one
attempt per task. The unchanged upstream travel task is included. Gemma policy arms use
`knowlet/Gemma-4-12B-Unified-System-One` revision
`a66f836b56605039fe040f330180e336d19b3362`, NF4 weights with BF16 computation.
The competitor uses `Mapika/decider-2b` revision
`533964dae8be954c5b5e19fa4948e48408094c1e` in BF16 through `decider-ai==1.4.0`.
All seven use the same pinned Gemma NF4 checkpoint for real text generation on
NVIDIA GB10.

| Policy / evidence | Successful tasks | Final goal satisfied | False DONE | Policy / text calls | Total attempt time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Native causal](../artifacts/jev-browser/20261014-local-nf4-native-03/receipt.json) | 1/9 | 4/9 | 2 | 71 / 6 | 138.585 |
| [Independent user message](../artifacts/jev-browser/20261014-local-nf4-user-01/receipt.json) | 5/9 | 5/9 | 4 | 35 / 3 | 106.460 |
| [Pinned Decider-2B](../artifacts/jev-browser/20261014-local-decider-02/receipt.json) | 4/9 | 4/9 | 2 | 55 / 2 | 48.758 |
| [Reasoned](../artifacts/jev-browser/20261014-local-nf4-reasoned-01/receipt.json) | 6/9 | 8/9 | 1 | 50 / 3 | 466.336 |
| [Compact reasoned](../artifacts/jev-browser/20261014-local-nf4-compact-01/receipt.json) | 7/9 | 7/9 | 2 | 35 / 3 | 208.208 |
| [Grounded V1](../artifacts/jev-browser/20261014-local-nf4-grounded-01/receipt.json) | 8/9 | 8/9 | 1 | 30 / 2 | 50.039 |
| [Goal-first V2](../artifacts/jev-browser/20261014-local-nf4-grounded-02/receipt.json) | 7/9 | 7/9 | 2 | 29 / 2 | 45.809 |

Total attempt time is the sum of all nine attempts, including failures, browser
navigation, model/text work, independent verification, final HTML capture and
browser cleanup. It excludes checkpoint/server setup and JSON evidence writes.
It is neither a latency percentile nor a sum restricted to successful tasks.
The separate entire-run receipt time includes setup and must not be mixed into
this column. These are descriptive measurements, with no speed-leadership claim.

The fixture hashes, seed `20261014`, 16-decision budget and real text-helper
settings match between arms. Gemma policy/helper checkpoint, NF4/BF16 format and
temperature `2.7830344470383452` are fixed; Decider retains its own published
per-type calibration. The original-nine runner hash is the same in all seven receipts.
The first three share one runtime source identity; later variants intentionally
change policy/runtime source, with each version captured and verified after its
run. The shared Gemma helper uses SDPA, and cuDNN SDPA is disabled throughout
each benchmark process. The recorded stack is Torch `2.13.0+cu130`, Transformers `5.17.0`,
bitsandbytes `0.50.2` and Playwright `1.58.0`.

Native succeeds only on saving checkbox preferences. It reaches the right final
DOM on three additional tasks but stops as BLOCKED, which is a failure under the
completion rule. Its completion-toggle run exhausts the budget. The user variant
passes search/filter/detail opening, autocomplete, populated-form submission,
report activation and the completion toggle. It still stops too early on opening
a filtered record, saving checkbox preferences, saving a selected mode and the
upstream travel task. On travel, it opens Casa Flora with Design and cancellation
enabled but never applies the Lisbon destination search.

Decider passes filter/detail opening, populated-form submission, report
activation and the completion toggle. It falsely declares DONE on the search
and autocomplete tasks, becomes BLOCKED on saving a selected mode, and exhausts
the decision budget on checkbox preferences and upstream travel. Its observed
peak CUDA allocation is 12.274 GB for the process containing both Decider and
the Gemma NF4 text helper; this is not Decider's standalone footprint.

The Decider receipt records the eager PyTorch reference path, with no CUDA
graphs, compilation or persistent prefix cache. Optional `causal-conv1d` and
`flash-linear-attention` packages are absent. Model size and these implementation
differences affect latency, so these runs establish no model-only speed ordering.

Reasoned assessment reaches eight correct final DOMs but terminates cleanly on
only six tasks: search/filter opening becomes BLOCKED after meeting the goal,
and report activation exhausts the budget after activation. Compact assessment
passes seven tasks but falsely completes autocomplete and upstream travel.
Grounded V1 passes all eight seeded tasks and falsely completes upstream travel.
Goal-first V2 additionally falsely completes saving a selected mode, reducing
its score to seven. Both grounded variants use a fixed state-reading rubric
without assessment generation. The current `grounded` mode restores V1, selected
on development results; the V2 attempt remains available as `goal-first`. Its
historical receipt retains the former mode name and original captured source.

All variants still fail the exposed upstream travel task. A higher development
pass count does not establish reliable completion or a general advantage over
Decider. Prompt placement, question independence and later policy/runtime changes
were developed on these known cases, so their individual effects and population
advantage remain unresolved. The first frozen holdout result is recorded separately below.

All seven linked receipts retain every attempt and pass post-run source
verification. These files are local campaign evidence; the receipts and raw
traces should be archived with any published comparison.

## First frozen holdout

The restored grounded V1 policy was frozen before evaluating two seeded holdout
variants of each of the nine local templates, including `unapplied_search`.
Hierarchy and stage batching were disabled. The matched pinned Decider arm uses
the same Gemma NF4 helper. No policy changes are made from these holdout outcomes.

| Policy / evidence | Successful tasks | Final goal satisfied | False DONE | Policy / text calls | Total attempt time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Frozen grounded V1](../artifacts/jev-browser/20261014-holdout-grounded-r2-01/receipt.json) | 18/18 | 18/18 | 0 | 67 / 6 | 120.902 |
| [Pinned Decider-2B](../artifacts/jev-browser/20261014-holdout-decider-r2-01/receipt.json) | 5/18 | 5/18 | 5 | 112 / 3 | 51.291 |

Both arms have identical 18-case fixture manifests, seeds, budgets, runner,
timing boundaries and helper settings. All retained fixture HTML and goal hashes
match those manifests, all 65 captured source files match their recorded hashes,
and post-run source verification passes for both. The runtime identity is
`96b6a32e74c0ab6e80230befb979ce7f62c2d43325375a020eb695603a2904d7`;
the runner and settings are preserved in the linked receipt. This timing uses
the same attempt boundary as the pilot, excluding setup. These are two variants
of nine known templates, with no unseen-website result. The exposed upstream
travel task is excluded from holdout and still fails in development. Grounded
V1 completes more tasks in this specific paired fixture population; the two
variants share nine templates and are not 18 independent task types. The
shorter Decider total includes its failures and fewer text calls, so this table
is not a comparison of successful-task latency or model-only speed. It does
not establish broad browser reliability or superiority on live websites.

## Historical failure diagnosis

The October 13 archive's 72 measured decisions retain the exact goal and stay
below the 4,096-token context limit. The text helper correctly generated Lisbon,
and the executor performed the chosen actions. The remaining observed errors are policy
and termination failures:

| Observed page | Wrong decision | Consequence |
| --- | --- | --- |
| Completion confirmed; Reset is the only enabled button | CLICK 0.778488; DONE 0.062994 | Undo a completed task |
| Lisbon typed, Design selected, cancellation unchecked | Open Casa Flora outranks Search and cancellation | Skip submission and a required filter |
| Detail explicitly says cancellation off and destination anywhere | DONE 0.614459 | False completion |

All 30 already-confirmed completion observations incorrectly select CLICK. Those
requests contain a single question, so question coupling does not explain this
specific stop failure. Positive scalar temperature changes cannot fix argmax.
Checkbox `value="on"` and `checked="false"` are also potentially confusing: the
HTML submission value is different from actual checked state.

The `user` prompt variant moves instructions and all candidates inside the
native user message, with a checked answer-letter boundary in the assistant
turn. It scores each question independently. `native` preserves the existing
causal prompt. The local pilot above measures the two variants on the development
suite. The separate `reasoned` mode additionally generates a provisional progress
assessment from the observation, history and goal before the same user-message
readout. `--analysis-style compact` requests a shorter assessment. Their extra
generation is included in attempt time and policy-response token accounting;
the table's text-call count denotes field-value generation, not those assessments.
`grounded` and `goal-first` add fixed reading rubrics without assessment generation.

## Protocol

The seven pilot rows above use the original eight seeded tasks plus unchanged
upstream travel HTML. The current default suite appends `unapplied_search`, for
ten development tasks and nine holdout tasks. The original nine pages and goals
are unchanged; earlier nine-task scores are not rewritten as ten-task results.
The new trap initially shows a matching record and checks that the search and
filters were actually submitted before opening it. A matching card and typed
query alone cannot satisfy its independent judge.

Completion has the historical button interlock: completing disables Mark
complete and enables Reset; resetting reverses it. The travel
judge checks applied search state rather than merely typed text or a property's
cancellation badge.

The nine seeded holdout templates use different values, DOM identifiers,
candidate orders and confirmation wording. They are variants of the same
templates, not unseen websites. The original exposed travel task is excluded
from holdout. Read-only Wikipedia tasks can separately test actual web navigation.

A pass requires all three conditions: a fresh independent DOM check succeeds,
the upstream agent declares DONE, and no execution/budget/cleanup error occurs.
Correct DOM at a timeout is recorded as `goal_satisfied`, not task success.
False DONE, blocked runs, provider errors and retries remain in the evidence.
Each attempt saves requests, every HTTP response/retry, generated text including
invalid raw outputs, probabilities, actions, final DOM and source hashes.

All comparison arms must share goals, fixtures, seeds, browser viewport,
action/request budgets, helper checkpoint/settings and timing boundaries.
Compare successful-task latency separately from failures. Record unsupported
requests and unknown bills explicitly. The local NF4 pilot uses its decision
checkpoint for text generation; its timings are not equivalent to the cloud
campaign's common BF16 base helper.

## Read-only live trials and candidate capacity

The [Spork-to-Spoon user trial](../artifacts/jev-browser/20261014-live-spork-spoon-user-01/receipt.json)
passed 1/1 in 16.009 seconds, with three policy calls, no field-text calls and one
executed action. Fresh verification reads the final Spoon URL, heading and article
content; the agent declares DONE and source verification passes. This earlier
user-mode trial has no matched live competitor result.

A later paired trial uses the frozen grounded V1 policy and pinned Decider on
the same Spork-to-Spoon goal, viewport, 16-decision budget, helper and runtime
source identity. Their first upstream wire inputs match exactly after replacing
the model alias, including all 37 click candidates. The
[grounded run](../artifacts/jev-browser/20261014-live-spork-spoon-grounded-01/receipt.json)
passes 1/1 in 21.147 seconds, with four policy calls and one executed action.
The [Decider run](../artifacts/jev-browser/20261014-live-spork-spoon-decider-01/receipt.json)
fails 0/1 in 24.805 seconds after 16 policy calls and eight actions. Decider opens
Spoon, then continues navigating; its final page is View source for Main Page.
Both use no field-text calls and pass source verification. These are two trials
of one live task, not a cross-site reliability estimate or latency ranking.

Read-only CPU browser observation probes on Wikipedia exposed target tables with
74 candidates on Solar System, 54 on Photosynthesis and 80 on Python. The ordinary
S1 contract supports at most 52 candidates per question and explicitly rejects
those larger target heads. No targets were pruned in the
[CPU capacity inspection](../artifacts/jev-browser/wiki-capacity-probes/20261014-cpu-01/summary.json);
that inspection has zero model calls and provides no model quality score.

The explicit hierarchy adapter uses balanced groups of at most 20 targets,
supports up to 400 targets per recognized head, and evaluates every group and
every within-group conditional. It returns the complete joint distribution
`P(target) = P(group) × P(target | group)` and its global maximum. These are
hierarchical policy probabilities; they do not retain flat-classifier calibration.
Ordinary core contracts remain unchanged. Stage requests, results and failures
are preserved. The default 20-second budget is cooperative: a blocking backend
call can finish after the budget before the adapter reports timeout.

An 80-button Chromium diagnostic passes through the real guarded browser
executor with a diagnostic policy. It validates transport and target retention;
it is not a trained-model success score.

The completed [Solar-System-to-Sun hierarchy trial](../artifacts/jev-browser/20261014-live-solar-sun-grounded-hierarchy-01/receipt.json)
fails 0/1 in 25.982 seconds. The subsequent
[batched trial](../artifacts/jev-browser/20261014-live-solar-sun-grounded-batch-01/receipt.json)
also fails 0/1, in 26.659 seconds. Both retain the identical 76-candidate request,
execute no browser action, return HTTP 504 and leave the final article on Solar
System. Source verification passes in both runs. Their 20-second cooperative
budget was not raised.

The sequential trial makes six native stage predictions. Batching requests
those same six stages in one backend method call and uses three genuine
two-row forwards. The backend returns all six raw distributions in 24.104
seconds, before the adapter rejects the over-budget call. Those outputs retain
the Sun target's high conditional probability, but no joint decision was served
and no action or task success occurred. Whole hierarchy time is 24.327 seconds
batched versus 22.769 seconds sequential; attempt time includes the browser and
verification as well. This one matched snapshot shows no speed improvement.
Per-chunk durations were not retained in this historical batch receipt; only
shared time, references and counts are available. Making the candidate table
representable has not yet made this larger-page navigation usable.
The current backend additionally retains one shared chunk ledger on its first
ordered response, so completed chunk results and timing survive a later adapter
deadline. CPU diagnostics verify that retention; no new live result is implied.

## Run locally

```bash
uv sync --extra inference --extra api --extra quantization --extra browser
uv run --no-sync playwright install chromium
git clone https://github.com/browser-use/jev-ultrafast.git /tmp/jev-pinned
git -C /tmp/jev-pinned checkout 1231850a0bf1a0c0341fe408ef1668dbbfdfac46

# Serve an existing configured model with the opt-in wire adapter.
uv run --no-sync s1 serve --backend gemma --jev-compatible \
  --jev-model-identity gemma-browser

# Policy-only checks; no text helper is silently selected.
uv run --no-sync python scripts/jev_browser_benchmark.py \
  --upstream-path /tmp/jev-pinned \
  --endpoint http://127.0.0.1:8000/v1/systemone --model gemma-browser \
  --output-dir artifacts/jev-browser/your-unique-run \
  --tasks completion_toggle premature_done submit_populated
```

For a complete local model run, install the inference/quantization extras in a
CUDA-compatible environment, then run `scripts/jev_local_model.py` with a pinned
model revision and a documented `--prompt-mode`. It generates real field values
with that same model. An existing output directory is refused.
Use a frozen source snapshot for a running campaign; changing source during a
run would invalidate its recorded hashes.

Use a Torch/CUDA build that supports the installed GPU. The GB10 pilot used
Torch `2.13.0+cu130`; the repository's standard inference extra currently pins
Torch `2.8.0`. Preserve the established GPU environment when reproducing the
GB10 pilot and record its actual versions.

```bash
python scripts/jev_local_model.py \
  --upstream-path /tmp/jev-pinned \
  --model knowlet/Gemma-4-12B-Unified-System-One \
  --revision a66f836b56605039fe040f330180e336d19b3362 \
  --quantization nf4 --prompt-mode grounded \
  --suite-split development --repeats 1 \
  --output-dir artifacts/jev-browser/your-unique-local-run
```

`--competitor decider` selects the existing pinned Decider-2B adapter for policy
decisions while retaining the same Gemma checkpoint and quantization for text.
Record the competitor telemetry, installed runtime packages and measured memory;
its historical A100 results do not predict this browser suite's success on GB10.
The retained `20261014-local-decider-01` attempt failed during checkpoint setup
before any browser decisions; it is a setup failure, with no nine-task score.
The native/user pilot receipts' `model_load_ms` includes source capture and
dependency imports through loading. The updated runner separates loader time
from `setup_through_model_load_ms`. Browser attempt timing begins after setup.

For a separate read-only Wikipedia trial, supply `--live-only`, `--live-url`,
`--live-goal` and `--live-expected-title`. For example, start at
`https://en.wikipedia.org/wiki/Main_Page`, ask to open the article about
Gödel's incompleteness theorems, and use the expected heading
`Gödel's incompleteness theorems`. The checker reads the final article heading
and content independently. Larger target tables require the explicit hierarchy
option, with the changed probability and timeout semantics described above.

Add `--jev-hierarchical` together with `--jev-compatible` when serving the API,
or `--hierarchical-candidates` to `scripts/jev_local_model.py`. Both enable the
same adapter; neither silently selects or prunes targets. Keep this option and
its branching/budget settings identical across comparison arms.

To request stage batching, add `--jev-batch-stages` with the API hierarchy flags,
or add `--batch-hierarchy-stages` with `--hierarchical-candidates` in the local
runner. This optional setting uses the backend's ordered batch method when
supported; otherwise the receipt records the sequential fallback. It preserves
every group and conditional decision, records shared inference time once, and
keeps the same cooperative time budget. No successful live-model result for
this batch setting is claimed yet.

```bash
JEV_BROWSER_TESTS=1 JEV_UPSTREAM_PATH=/tmp/jev-pinned \
  uv run --no-sync pytest -q tests/test_jev_browser_benchmark.py tests/test_jev_api.py
```

These Chromium tests use diagnostic policies to verify the harness and judge.
Their pass counts are not model task success rates.

## Matched cloud campaign

`apps/modal/jev_browser.py` prepares ephemeral A100-80GB runs for native/user
base Gemma, native/user specialist Gemma, optional reasoned arms, and pinned
Jev-Omni. It uses the same real base-model text helper for every arm and preserves
setup failures too.
An individual worker has a 60-minute limit and only one worker runs at a time.
No deployment or model publication is part of this command.

Automatic approval review rejected uploading project source to Modal because
sending workspace material to that external destination had not been explicitly
authorized. The cloud campaign is waiting for that authorization. Local tests,
public checkpoint downloads and local inference continue independently. No
successful cloud campaign is claimed.

## Remaining acceptance work

Repeat frozen comparison arms on additional disjoint variants and the expanded
development suite. Evaluate remaining early-stop and
search-submission failures. Resolve the hierarchy trials' inference cost within
their recorded budget before broader live comparisons.
Use completed-page candidate-order permutations to distinguish semantic failures
from position sensitivity. Tune
only on development cases; freeze the policy before holdout. Then run repeated,
matched competitor arms and real website tasks, with paired uncertainty and
complete error/cost populations. A small fixture win does not establish broad
browser reliability or superiority over other models.
