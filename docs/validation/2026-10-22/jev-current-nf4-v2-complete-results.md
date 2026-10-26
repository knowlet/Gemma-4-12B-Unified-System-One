# Matched current-NF4 V2 browser results

On 2026-10-22 the fresh native V2 adapter completed **4/10** original V7 Chromium tasks on GX10, versus **2/10** for the raw96 control. Both arms retained all ten tasks, the same NF4 text helper, seed, candidates, judges, 16-decision limit and request budgets. The known four tasks remained 1/4; the six public tasks improved from 1/6 to 3/6. Two tasks gained success and none lost success.

The comparison changes both the action adapter and prompt serialization. It does not establish which change caused the improvement. These ten public scenarios are an exposed development comparison, with no statistical superiority or generalization claim. Actual Decision3 comparison is still pending.

The V2 run made 85 policy calls and five text-helper calls, all recorded as HTTP 200. HTTP completion does not imply task completion: six tasks still failed, including two premature DONE claims, two tasks exhausting their decision budgets and two BLOCKED outcomes. Independent replay of the unchanged DOM judges confirmed every result and the paired task outcomes.

The fresh adapter followed a complete [142-update native fit with 37 zero-update development examples](jev-native-v2-complete-training-results.md). Its manifest SHA is `8d4f014be160f632936678db32be8c73d3413f301c368eb86d6c69644ac7727c`; deployment uses NF4 and `shared-elements-v2`, including its exact raw-format fallback. The original BF16 checkpoint revision is `a66f836b56605039fe040f330180e336d19b3362`.

Independent auditing verified all 703 browser output files, 90 sealed HTTP journals, ten ordered task IDs, saved DOM judges, both browser client closures and the successful selected model invocation. The owner recorded all 785 selected source/artifact hashes unchanged. The auditor independently hashed the 784 small files and checked the current 11 cache aliases; it did not repeat the full 23.9 GB base-weight hash. The selected actor and cgroup were absent, with two clean process scans and a known final host.

See the [aggregate and exact evidence references](jev-current-nf4-v2-complete-results.json), [raw control](jev-current-nf4-raw-complete-results.md) and [earlier failure diagnosis](jev-current-nf4-raw-partial-diagnosis-01.md). Raw DOM, requests, answers and weights remain local. The separate fresh BF16 results are Benchmark Heaven public231 **195/231 (84.42%)** and JevBench coherence mini **41.26/100**; they are not browser task scores.
