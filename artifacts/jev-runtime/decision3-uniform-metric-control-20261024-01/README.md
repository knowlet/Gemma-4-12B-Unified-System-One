# Uniform primitive metric control — held source candidate

This candidate diagnoses the scope of the official JevBench coherence metric.
It is not a model, a Decision3 adaptation, a browser agent, a ranking entry or
evidence of competitive superiority. No score has been measured here.

Status: **SOURCE ONLY, not executed or validated.** The existing NN96310 run
remains independent. No imports, tests, AST checks, formatting, builds,
evaluation, suite payload reads, host observations or model operations were
performed while authoring this candidate. The original frozen sources and
previous run qualifications are unchanged.

`runner.py` uses the standard library and the existing official
`jevbench.from_callable` / `jevbench.evaluate` API. Choice returns `1/K` for
every original criterion id, Score returns `1/L` for every original level
index, and Noul returns `0.5`. State and instruction text do not affect the
answer. Every supplied question and criterion order is retained. Unknown
types refuse instead of being silently skipped. This total three-primitive
definition is necessary because official relations themselves create Noul,
Choice and Score questions; a Choice-only callable is incomplete.

The exact selected benchmark metadata is
`decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json`, SHA-256
`b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021`.
Only its 78 source refs, two dataset-metadata refs and coherence identity are
used. Its model, adapter, CPU/RAM and old owner fields confer no authority.
The coherence revision remains
`e18733694623aa93058e279c3534f8b8e2edefa3`; the suite remains
`jevbench-mini`, with seed 0, one trial/repeat, one concurrent request,
minimize disabled, original fixed 0.05 scoring tolerance, min_n 5, bootstrap
1000 and min_coverage 0.95. No custom cases, relation subset, rewrite bank,
answer-dependent minimization or altered scorer is supplied.

A fresh in-memory cache is essential: the official API first asks a preflight
request, and several relations share identical requests. Disabling the cache
would resend them. This candidate requires the original 1248 distinct
requests / 240 cases / 50 relations / 1200 planned tests. The official suite
loader performs its existing payload hash checks. The selected source and
metadata hashes/stats are checked before and after computation; no benchmark
contents or raw results are copied into this candidate or its output.

The aggregate contains official binary/graded metrics, dimensions, groups,
relations, intervals and coverage/exclusion diagnostics. Metrics are copied,
not recomputed. Official errors are counted and canonically hashed; a new
zero-error qualification rule is not imposed. Raw Report, case ids, examples,
questions, gold, bases and error text are never written. Every output remains
`computed_unreviewed_control_only` and requires an independent current root
closure/source audit before any scoped reporting.

Default invocation is held and does not open pin metadata or official
dependencies. `--execute` also refuses immediately while `SOURCE_ONLY=True`,
before reading its own source, pins, dependencies, suite or output path.
There is no CLI override, environment override, auto-release or guardian in
this candidate. Root must first review this source, finish the existing run,
validate the fake tests and select its own execution/closure boundary. A later
explicit source change releases the hold and changes the runner SHA; root
must select that new exact SHA. The future interface is:

```text
<python> -I -S -B runner.py --execute \
  --expected-runner-sha256 <reviewed released runner SHA> \
  --output <new aggregate JSON path in an existing directory>
```

This is an interface description, not a command executed in this stage.
Output creation is exclusive and cannot replace an earlier result.
`test_runner.py` contains only synthetic wire-format, invariance/partition,
typed-projection, redaction and early-hold tests. It does not load the official
package or actual suite. The tests are authored but unexecuted.

Official source rationale:

- `schema.py:21,29,124`: the three primitives and their wire distributions.
- `api.py:51,129,165,200`: callable interface, frozen bundled suite rules,
  evaluate defaults and initial preflight.
- `backends.py:256,318,378`: answer validation and cache identity/behavior.
- `relations/rep.py:141`, `relations/bat.py:29`, `relations/log.py:46`,
  `relations/cho.py:55`: stable uniform answers can satisfy representation,
  batch, weak logical-order and conditioning laws without reading state.
- `relations/mea.py:229,247,291,376,488`: partition/marginal constraints can
  distinguish uniform answers; actual binary violations still depend on the
  original fixed tolerance and frozen selected tests.
- `score.py:50,71,108`, `results.py:74`: official binary/graded aggregation,
  five-dimension mean, coverage policy and report settings.

No actual score, browser usability, correctness or ranking follows from this
source analysis. Uniform ties also make first-argmax actions and flip fields
unsuitable as competence evidence.
