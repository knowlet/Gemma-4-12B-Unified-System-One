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

## Second pinned competitor: Jev-Omni

The local native Jev-Omni comparison uses
`akhilaaa3/Jev-Omni@5addda86ddee081a68fb067477ea100c221b8917`, its
hash-checked publisher prompt and decision head, BF16 backbone and BF16
autocast with FP32 head storage. It uses the same pinned Gemma NF4 field-text
helper as the grounded arm. Neither hierarchy nor prefix caching is enabled.
These are deployment configurations with different model weights and precision;
the experiment does not isolate an architecture or quantization effect.

| Policy / evidence | Successful tasks | Final goal satisfied | False DONE | Policy / text calls | Total attempt time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Grounded V1 rerun](../artifacts/jev-browser/20261014-holdout-grounded-r2-02/receipt.json) | 18/18 | 18/18 | 0 | 67 / 6 | 126.143 |
| [Native pinned Jev-Omni](../artifacts/jev-browser/20261014-holdout-jev-omni-r2-02/receipt.json) | 6/18 | 10/18 | 1 | 155 / 7 | 206.202 |

The [paired audit](../artifacts/jev-browser/20261014-paired-grounded-jev-omni-holdout-audit.json)
verifies identical fixture manifests, seed, 16-decision budget, 67 captured
source files and all 18 first wire requests after normalizing only the model
alias and local fixture-server port. Both source verifications pass with identity
`245ceb608af2248b73cd863ffc507cf57cf58c900ac4ab476a518f8cc2b70f3d`.
The nine shared template families and two seeded variants are the same holdout
population used above. The original exposed travel goal is excluded. The
grounded policy was not changed using these competitor outcomes. Four Jev-Omni
attempts have satisfied final DOM requirements without a valid DONE; they
remain unsuccessful under the same completion rule. The times include failed
attempts and different action/text-call counts, so they do not establish
model-only speed or successful-task latency.

An earlier Jev-Omni attempt was interrupted during setup when its temporary
runtime disappeared. Its [original receipt](../artifacts/jev-browser/20261014-holdout-jev-omni-r2-01/receipt.json)
and [interruption observation](../artifacts/jev-browser/20261014-holdout-jev-omni-r2-01/interruption-observation.json)
are preserved. It has no browser attempts and is not scored as 0/18. The completed
rerun reconstructs its exact captured source bytes in project-owned storage.

## Frozen-backbone readout development fit

The [first readout fit](../artifacts/jev-browser/20261014-browser-readout-fit-01/receipt.json)
extracts genuine grounded-V1 hidden states from the pinned Gemma NF4/BF16
checkpoint, then optimizes detached copies of its 52 selected LM-head rows and
bias in FP32. The optimizer updates only the sidecar, and the fit checks that
the original 52 head rows remain unchanged. The backbone and full text-helper
head are not replaced; the common field-text helper continues to use that original
model. The untrained FP32 sidecar is the matched control, so both arms use the
same projection precision, candidate IDs, softcap and deployment temperature
`2.7830344470383452`.

Training uses 368 scripted development states from 88 guarded browser episodes
(seed `20261001`, eight repeats), producing 584 question features. It includes
the exposed original travel goal and eight early-detail-opening recovery
episodes. A separate 66-state, 18-episode development collection (seed
`20261002`, two repeats) produces 102 features and excludes exposed travel.
The recorded split audit finds no shared IDs, source groups or identical
requests between these collections. Only the 584 training features enter the
optimizer; the 102 development features are retained separately. These are supervised variants of known
fixtures. They are not held-out website evaluations, and the original travel
page and goal are already present in training.

The fit performs 128 CPU optimizer steps with batch size 64, learning rate
`0.001` and seed 42, using soft cross-entropy plus `0.1` Brier loss and a `0.01`
selected-row anchor at objective temperature `1.0`. Feature extraction and fit
setup are included in its 953.377-second receipt time; that is not browser
inference latency. The fit's pre-load source capture and post-run source
verification pass. Its exact case, wire, fixture and receipt copies were
[preserved after fitting](../artifacts/jev-browser/20261014-browser-readout-fit-01/inputs-preserved-after-fit/manifest.json),
with hashes matching the recorded collections. The matching collector source
was also preserved retrospectively, before the audit fixes. These retrospective
copies do not establish a pre-collection collector/harness source identity or
pre-fit input preservation. Fit01's own runtime sources were separately captured
before model loading. Future collectors capture their sources
before collection; future fits freeze validated inputs before loading weights
and fail if their originals, frozen copies or runtime sources change.

The completed closed-loop development runs use all ten current tasks, seed
`20261014`, one attempt per task and the same 16-decision budget. Hierarchy,
stage batching and prefix caching are disabled.

| Policy / evidence | Successful tasks | Final goal satisfied | False DONE | Policy / text calls | Total attempt time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Untrained FP32 control](../artifacts/jev-browser/20261014-development-readout-control-01/receipt.json) | 8/10 | 8/10 | 1 | 45 / 3 | 202.506 |
| [Learned FP32 readout, 128 steps](../artifacts/jev-browser/20261014-development-readout-learned-01/receipt.json) | 9/10 | 9/10 | 1 | 39 / 3 | 154.257 |
| [Learned FP32 readout, 512 steps](../artifacts/jev-browser/20261014-development-readout512-01/receipt.json) | 10/10 | 10/10 | 0 | 39 / 4 | 114.628 |

The control and 128-step arms retain identical fixture manifests and the same verified runtime
source identity
`a161b4a22dc18436c11e6dce39eed974551c79465e4c288ba7b6f058eacddaca`.
Their sidecar weight hashes match the loaded manifests. The control exhausts
the budget on search/filter/detail opening despite correctly generated search
text; the learned arm completes that task and all nine seeded fixtures. Both
still falsely complete the original upstream travel task. The three real helper
outputs are correct in both arms: `Juniper Library`, `Osaka Riverside` and
`Harbor District`. Neither travel attempt invokes the helper or types Lisbon.
There is no held-out learned-readout result in this report.

The [learned travel trace](../artifacts/jev-browser/20261014-development-readout-learned-01/benchmark/attempts/development-000-upstream-travel/model_calls.json)
executes Free cancellation, the Find a stay navigation link, Design selection,
Find stays, and View Casa Flora, then chooses DONE. Find stays submits an empty
Destination. The fresh judge confirms Design, cancellation enabled and a
submitted search, but reads `query=""`; Lisbon remains unmet. The
[control trace](../artifacts/jev-browser/20261014-development-readout-control-01/benchmark/attempts/development-000-upstream-travel/model_calls.json)
opens Casa Flora immediately and chooses DONE with all four search/filter
requirements unmet.

On their matching initial travel observations, the learned operation head gives
CLICK `0.471885` versus TYPE_TEXT `0.424997`, a margin of `0.046888`, and its
click head prefers Free cancellation with `0.954946`. The control prefers
CLICK `0.273674` versus SELECT `0.216872`, and View Casa Flora with `0.595322`.
At their different final detail states, learned DONE is `0.873468` versus CLICK
`0.083619`; control DONE is `0.606754` versus CLICK `0.196141`. These observed
probabilities show that the remaining omission and false stop occur in policy
selection, with no helper or executor error. They do not identify a causal
explanation for the learned policy's missed empty query or establish calibrated
confidence on new goals.

The table sums all ten attempts, including the control's budget failure,
navigation, model/text work, verification and cleanup. Setup, checkpoint loading
and fitting are excluded. These single runs take different action paths, and
their model-loading times also differ, so the totals are not a successful-task
latency comparison, a percentile or evidence of a general speed advantage.
The 128-step result is development progress with a known travel failure.
The longer fit below resolves that exposed task in one development run;
learned generalization remains unestablished.

## Longer readout fit on unchanged features

The [512-step CPU replay](../artifacts/jev-browser/20261014-browser-readout-replay512-01/receipt.json)
reuses the exact original feature archive and untrained FP32 control. It fits
512 total steps from that control, with the same seed 42, learning rate `0.001`,
objective and fixed deployment temperature. The optimizer receives the same
584 training features from 368 states; the separate 102 development features
remain excluded from optimization. The new 673-state partial-progress
collection is not used. The longer-fit choice was fixed on development
observations, before the new travel-goal outcomes, and selection remains
restricted to the original development suite.

This replay takes 9.509 seconds, loads no backbone and records zero new
backbone forwards or GPU training. Its source, original/frozen collection,
original/frozen replay-input and emitted-copy integrity checks all pass.
The feature SHA
`c8873a29a413c76815d6e4636c391a885d40f22108a8a4c1068cbef474b058d0`
was captured retrospectively for replay: the original fit did not prospectively
hash its feature archive or control manifest. Freezing these bytes now does
not authenticate their identity at the original extraction time. The replay
preserves original candidate-token IDs, softcap and deployment temperature;
its inherited full-head verification is distinguished from the current
CPU control-row check.

The [completed 512-step browser run](../artifacts/jev-browser/20261014-development-readout512-01/receipt.json)
passes all ten known development tasks with zero false DONE. All ten fixture
manifests and initial wire requests match the control and 128-step runs after
normalizing only model alias and the local server port. Browser budgets, seeds
and timing boundaries also match. The decision
backend, readout, text helper, backbone wrapper and batch-scoring source files
are byte-identical; CLI and runner/case-builder sources differ. All 67 captured
source files verify, with runtime identity
`245ceb608af2248b73cd863ffc507cf57cf58c900ac4ab476a518f8cc2b70f3d`.
The loaded readout matches the saved weight SHA
`11b56c9da7750d4e748a2f0ade2befe1f9dbeb7522be0312b826647c423cd159`.

Its [original travel trace](../artifacts/jev-browser/20261014-development-readout512-01/benchmark/attempts/development-000-upstream-travel/final_state.json)
selects Design, correctly generates and types Lisbon, enables Free cancellation,
clicks Find stays, opens Casa Flora and chooses DONE. Fresh verification reads
`query="Lisbon"`, `searched=true`, Design and cancellation enabled, with the
correct visible detail page. This uses five actions, six policy calls and one
text call. All four helper outputs across the suite are correct. Hierarchy,
stage batching and prefix caching remain disabled. The table's 114.628 seconds
sums browser attempts and excludes fitting and setup. This is one successful
development run on supervised known fixtures. The separate goal recheck below
uses a changed combined inference configuration; this development result alone
establishes no held-out learned-head, general reliability or speed advantage.

## BF16 development check and travel goal variations

The [unquantized BF16 grounded run](../artifacts/jev-browser/20261014-local-bf16-grounded-01/receipt.json)
passes 9/10 development tasks, satisfies nine final goals and falsely declares
DONE on upstream travel. It uses the original decision head, the same pinned
checkpoint, grounded V1 prompt and deployment temperature, with BF16 for both
decisions and the real text helper. It makes 38 policy and four text calls;
the sum of all ten attempts is 64.719 seconds, excluding setup. Hierarchy,
stage batching, readout replacement and prefix caching are disabled.

Its [travel history](../artifacts/jev-browser/20261014-local-bf16-grounded-01/benchmark/attempts/development-000-upstream-travel/final_state.json)
executes Free cancellation, types the helper's correct `Lisbon`, selects
Design, opens Casa Flora and chooses DONE. The input still visibly contains
Lisbon before opening the details, but Find stays is never clicked. The fresh
judge reads Design and cancellation enabled, with applied `query=""` and
`searched=false`. All four BF16 helper outputs are correct: `Lisbon`,
`Juniper Library`, `Osaka Riverside` and `Harbor District`.

These traces retain three different omissions on the exposed travel goal:
the earlier NF4 grounded policy opens Casa Flora immediately, the learned
128-step NF4 sidecar submits an empty destination, and BF16 types the right
destination but omits submission. The original nine fixture/goal identities
match the historical NF4 run, and all ten match the learned-readout development
run. Runtime source versions differ between those historical runs, and helper
precision changes with BF16. These observations do not isolate a causal effect
of quantization or establish a speed ordering.

The [three travel goal variations](../artifacts/jev-browser/20261014-travel-goal-challenges-readout128-01/receipt.json)
use the same unchanged exposed upstream HTML with new requested requirements.
They retain the 128-step learned sidecar, whose weight SHA is
`fedb79fcc4924a4a024bc66ddd8971301087f12509f0961ad15fa7883a00f3d7`,
the NF4 backbone/helper, one attempt per goal and the same 16-decision budget.
They are new goals on a known page, not unseen websites or a matched competitor
comparison.

| Requested outcome | Successful | Policy / text calls | Attempt time (s) |
| --- | ---: | ---: | ---: |
| Design, Copenhagen, cancellation enabled; open The Glasshouse | 0/1 | 16 / 0 | 87.995 |
| Nature, Lisbon, cancellation left off; open Serra Lodge | 1/1 | 9 / 1 | 23.790 |
| Open Casa Flora details only | 1/1 | 2 / 0 | 4.044 |

The Copenhagen attempt enables cancellation, then repeats Find stays six times
and Find a stay navigation nine times until the budget expires. It
never types Copenhagen, selects Design or opens The Glasshouse. The final
applied destination is empty and category remains all. The Nature attempt
selects Nature, correctly generates and types Lisbon, clicks Find stays five
times, then opens Serra Lodge and stops with every requested property verified;
cancellation stays off. The direct Casa Flora request succeeds without search
or filters, which that goal does not require. Across these three attempts the
result is 2/3 with zero false DONE, 27 policy calls, one correct helper output
and 115.829 seconds of total attempt time, including the budget failure.
Both new runs verify all 67 preserved runtime source files and share source
identity `245ceb608af2248b73cd863ffc507cf57cf58c900ac4ab476a518f8cc2b70f3d`.
The challenge outcomes are not used to retune the prompt or select a longer
fit; selection remains on the original development suite.

The [512-step readout with prefix caching and suffix batch size two](../artifacts/jev-browser/20261014-travel-goal-challenges-readout512-prefix2-01/receipt.json)
subsequently passes only 1/3 on these same goal/HTML fixtures and budgets.
Hierarchy and stage batching are also enabled. It makes 21 policy calls and
one correct Lisbon helper call, with one false DONE and 52.153 seconds summed
across all three attempts. All 67 captured source files verify. The Copenhagen
attempt enables cancellation and opens The Glasshouse, then falsely stops
with category all, empty destination and no submitted search. The Nature
attempt selects Nature and eventually types/applies Lisbon, but repeatedly
clicks Find stays, then wrongly enables cancellation on its final allowed
action; it never opens Serra Lodge. Direct Casa Flora opening alone passes.

An [uncached 512-step recheck](../artifacts/jev-browser/20261014-travel-goal-challenges-readout512-full-01/receipt.json)
also scores 1/3, with the same 21 policy calls, one text call and one false DONE,
in 116.311 seconds of summed attempt time. Hierarchy and stage batching remain
enabled. Copenhagen again stops without its category, query or submission.
Nature keeps cancellation off in this run but still exhausts the budget
without opening Serra Lodge. Its 67 source files verify. The completion
failure therefore persists without prefix caching; the two inference paths
do not produce identical action histories.

The [cached-versus-full audit](../artifacts/jev-browser/20261014-readout512-prefix-versus-full-goal-audit.json)
finds identical initial wire inputs for all three goals, with maximum absolute
probability differences of approximately `0.037`–`0.041`. On the Nature goal,
the first operation changes from CLICK without caching to SELECT with caching;
later observations therefore diverge. Both configurations fail that goal.
Native prefix reuse has genuine numerical sensitivity on this NF4 deployment;
the real-model runs do not establish bit-identical scores or action histories.

This combined recheck falls below the 128-step run's 2/3 and shows that the
512-step development 10/10 has not established reliable completion on changed
goal requirements. Learned rows and the inference configuration both change,
so this comparison does not isolate a cause. These three observed goals are
now exposed development regressions; subsequent adaptations must not describe
them as held-out tests.

## Additional development recovery collection

The [partial-progress training collection](../artifacts/jev-browser/20261014-training-partial-trajectories-01/receipt.json)
contains 673 scripted states from 165 guarded episodes, using seed `20261003`
and 15 repeats, with 1,064 supervised question rows. It retains fresh development
episodes and adds 15 upstream travel recovery initializations covering every
incomplete combination of typed destination, selected Design, enabled
cancellation and submitted search.
Initialization actions are executed through the guarded browser before the
teacher observes and labels recovery states. These are 15 initialization
configurations, with 11 distinct verifier result tuples: unsent destination
text is lost when the detail page opens and is typed again during repair.

This collection includes a recovery with Design and cancellation already set,
an empty applied destination and `searched=true`, matching the missing-query
prerequisite seen in the learned development failure. The teacher returns to
all stays and supplies Lisbon. Its declared source is scripted development
supervision, with no model score. There are also 54 observed travel states with
Lisbon filled without a subsequent Find stays action, including 18 recovery
states.
In 27 of those states Design and cancellation are already correct; all 27
teacher choices click Find stays. Dataset, wire, 150 fixture-file and 67
prospective source hashes are retained and checked. The 673-state training
and separate 66-state development inventories share no IDs, source groups or
exact request fingerprints. These checks do not establish semantic independence
between variants of a template. This is newly collected supervision, with no
closed-loop quality improvement established by collection alone.

The preserved collector explicitly generates the development split. Its case
IDs and fixture HTML hashes do not overlap the recorded 18-case fixture
holdout, and none of the three new travel challenge goals or IDs occurs in
the emitted training supervision. Their literals are present in the preserved
benchmark source, whose challenge-generation function the collector does not
call. Seven exact goal strings do recur in the finite vocabulary of
the shared holdout templates, so goal-string disjointness is not claimed.
The exposed upstream travel HTML and original Lisbon goal are intentionally
training data. This inventory audit establishes what was collected; it does
not make these supervised templates an independent website population.

## Diversified travel supervision and transfer scope

The completed [diversified collection](../artifacts/jev-browser/20261014-training-diverse-goals-01/receipt.json)
contains 425 states, 108 episodes and 673 supervised questions. Four repeats
retain the nine seeded task families and add 60 travel goals plus 12 recovery
episodes on the unchanged exposed HTML. Travel supervision covers direct
opening, query only, category only, query plus category, and explicit
cancellation settings. The [independent audit](../artifacts/jev-browser/20261014-training-diverse-goals-01-audit.json)
checks all 67 source files, 96 fixture files and exact wire/label identities,
including four real corrections from wrongly enabled cancellation to off.
Against the separate 66-state development collection, IDs, source groups and
exact requests are disjoint, but five goal strings repeat. The three exposed
travel regression strings are absent; their semantic requirement combinations
are deliberately trained and must no longer be treated as held out.

The diverse fit is in progress with 512 CPU update steps, learning rate `0.001`
and seed 42 after fresh model feature extraction. It explicitly selects
`--feature-batch-size 2`; the default remains one. This option batches at most
two grounded feature rows while retaining each row's actual hidden features
and capture evidence. It applies to fresh extraction, not `--features-from`
replay. No completed diverse-fit quality result or batching speed advantage is
reported here. Use frozen sources and fresh output directories, for example:

```bash
python scripts/jev_browser_training_data.py \
  --upstream-path /tmp/jev-pinned --split train --seed 20261005 --repeats 4 \
  --travel-goal-variants --output-dir artifacts/jev-browser/your-training-goals

python scripts/jev_browser_fit_readout.py \
  --train-cases artifacts/jev-browser/your-training-goals/cases.jsonl \
  --development-cases artifacts/jev-browser/20261014-development-trajectories-01/cases.jsonl \
  --steps 512 --learning-rate 0.001 --seed 42 --feature-batch-size 2 \
  --output-dir artifacts/jev-browser/your-diverse-fit
```

The upcoming local `--travel-transfer-only --suite-split development` mode uses
a separately seeded catalogue with new property/city mappings and an additional
already-filled-control case. Choose and record its `--seed` after training
selection is frozen. It changes the catalogue and that initial form state,
preserves the pinned page's event handlers, and records parent and derived HTML
hashes. All eight available transfer property names and eight cities are absent
from the actual training payloads; evaluation factory literals in preserved
source files are not training payloads. This tests transfer within a known page
and task structure, not unseen websites. Freeze the learned artifact before
comparison, use the same fixtures and helper for other models, and retain
failures as well as successes. No transfer model result is claimed here.

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
cancellation badge. Travel query verification accepts surrounding whitespace
and case differences, matching the application's search semantics; a submitted
search remains a separate required state.

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
shared time, references and counts are available. These first two trials make
the target table representable but do not complete the navigation.

The [NF4 prefix-reuse trial with one suffix per forward](../artifacts/jev-browser/20261014-live-solar-sun-grounded-prefix-01/receipt.json)
also fails 0/1: all six stage results return from the prefix utility in 20.954
seconds, and whole hierarchy time is 21.333 seconds, exceeding the unchanged
20-second budget. It executes no action. Its attempt
takes 24.491 seconds. The [full-context BF16 batch trial](../artifacts/jev-browser/20261014-live-solar-sun-bf16-batch-01/receipt.json)
likewise returns HTTP 504 with zero actions and fails 0/1 in 27.309 seconds.

The later [NF4 prefix-reuse trial with two suffix rows per forward](../artifacts/jev-browser/20261014-live-solar-sun-grounded-prefix-batch2-01/receipt.json)
**passes 1/1** in 34.890 seconds of attempt time, with three policy calls,
one guarded click and no field-text calls. It opens target `44`, Sun, then
predicts DONE twice: the upstream freshness guard rejects the first DONE after
the page changes, re-observes, and accepts the next decision. Fresh independent
verification confirms the Sun URL, heading and article content. All three HTTP
responses succeed, and every hierarchy call stays below the same 20-second
cooperative budget. The complete run, including setup, takes 49.074 seconds.

| Policy call | Operation | Original click candidates | Policy time (s) | Full-context tokens | Physical computed tokens | Included padding tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Solar System | CLICK Sun | 76 | 12.586 | 28,543 | 13,199 | 2,491 |
| Sun, rejected stale decision | DONE | 71 | 8.835 | 26,459 | 12,360 | 2,341 |
| Sun, accepted decision | DONE | 71 | 9.097 | 26,965 | 12,515 | 2,365 |

Every call scores the operation, all four target groups and every within-group
conditional, including when DONE wins. The original target orders are preserved;
the served joint distributions cover all 76/71/71 candidates, exactly equal
the recorded group-by-conditional products, and each sums to 1. The singleton
text target retains its deterministic compatibility probability of 1. Each
call makes four genuine backbone forwards: one shared-prefix prefill and three
two-row suffix chunks, with one live cache fork at a time. No candidates or
original question contexts are pruned.

On the first call, the common prefix is 3,567 tokens. Its logical computed work
is 10,708 tokens; the table's 13,199 physical positions also count 2,491 new
padding positions. These counters sum consistently across the saved per-request
allocations and forward ledger; full-context counts retain the six original
question lengths. Cache-fork and backbone substep times are host measurements
without explicit device synchronization. The policy/attempt totals include the
completed probability work and the stale-page retry; substep times should not
be treated as independent GPU kernel timings.

The successful run's first saved request is identical to the sequential, NF4
full-context batch and one-suffix prefix requests, including model alias,
state, instructions and candidate order. The BF16 first request differs only
in model alias. The success uses the pinned checkpoint's original decision
head, grounded V1 prompt and NF4/BF16 computation; no learned readout is selected.
All 67 copied runtime source files match their hashes, and final source
verification passes with identity
`1cec18f022d567f09d64006bd1817a2344a91cb34b6db0b581c68632b8638cee`.
Runtime implementations differ across these historical runs. This is one
successful navigation on a previously inspected task, with no matched live
competitor for this target, no repeated latency distribution and no cross-site
reliability claim. It establishes that this preserved request can finish within
the existing policy-call budget using the explicit two-suffix setting.

The subsequent [learned-512 readout plus the same prefix setting](../artifacts/jev-browser/20261014-live-solar-sun-readout512-prefix2-01/receipt.json)
fails this live task with zero actions: the client reaches its unchanged
25-second read timeout, and fresh verification still reads Solar System.
No server-stage response or terminal server error is retained for that request,
so its cause is unresolved. The successful original-head live navigation and
the learned-head 10/10 development run are separate results; they do not
establish a single configuration that passes both.

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
keeps the same cooperative time budget. The successful Solar-System-to-Sun
trial above additionally enables prefix reuse and two-row suffix batching;
ordinary full-context stage batching alone still timed out in its recorded trial.

For owned text-only `user`, `grounded` or `goal-first` policies, opt into exact
common-prefix caching with `--jev-prefix-cache` when serving, or `--prefix-cache`
in the local runner. The default suffix batch size is 1. Select
`--jev-prefix-suffix-batch-size 2` or local `--prefix-suffix-batch-size 2` to score
two suffix rows per native cache fork; size 2 requires prefix caching. These
flags do not enable hierarchy or stage batching implicitly. To reproduce the
successful trial's complete combination:

```bash
python scripts/jev_local_model.py \
  --upstream-path /tmp/jev-pinned \
  --model knowlet/Gemma-4-12B-Unified-System-One \
  --revision a66f836b56605039fe040f330180e336d19b3362 \
  --quantization nf4 --prompt-mode grounded \
  --hierarchical-candidates --batch-hierarchy-stages \
  --prefix-cache --prefix-suffix-batch-size 2 \
  --live-only --live-url https://en.wikipedia.org/wiki/Solar_System \
  --live-goal 'Starting from the Solar System article, open Wikipedia’s article about the Sun. Stop only when the Sun article is open.' \
  --live-expected-title Sun \
  --output-dir artifacts/jev-browser/your-unique-solar-prefix-run
```

Keep the same checkpoint, compatible CUDA stack, viewport and recorded
20-second policy-call budget. The live page and its target table can change;
this command is a new trial, not a guarantee of the historical result.

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
