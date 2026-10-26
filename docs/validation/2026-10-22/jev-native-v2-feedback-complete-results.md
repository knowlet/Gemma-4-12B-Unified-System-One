# Typed stale-action feedback: no improvement observed

On 2026-10-22 the explicit feedback experiment completed **3/10** original V7 Chromium tasks, versus **4/10** for the matched native V2 control. It retained the three public-task successes and lost the Counter task, with no gains. The experiment is **not promoted**; the completed V2 configuration without feedback remains the best current NF4 result.

The experiment reused the same N3 adapter, shared-elements-v2 serializer, NF4 policy and helper, all ten task definitions, candidates, judges and request budgets. Its only declared agent change adds typed stale-attempt feedback to the next actual policy input, while preserving executed history and unknown physical input outcomes. Real web pages and timing can vary, and each arm ran once, so the observed difference does not isolate a universal causal effect.

Actual inputs carried feedback in 25 policy calls, with 15 distinct stale-attempt records. The complete run used 101 policy calls and five helper calls, all 106 recorded as HTTP 200. The additional feedback therefore reached the model but did not establish improved task completion.

The lost Counter task illustrates a completion-check gap: the model declared DONE while the requested method signature and description remained below the viewport. Both arms had two early unrecorded clicks, and their page text already differed before any feedback. Saved prompt, grammar and response checks found no invalid target, formatter fallback or failed HTTP delivery; the observations do not attribute the loss solely to feedback.

Independent auditing replayed all ten final DOM judges, joined all 106 client request bodies and statuses with sealed server journals, verified both browser client closures and confirmed the selected model invocation ended. It checked the 790-file output inventory, all 790 owner source references, independently hashed 789 small selected files, checked eleven current cache aliases and verified actor/cgroup absence with two clean scans. The auditor did not repeat the complete 23.9 GB base-weight hash.

See the [aggregate and exact evidence references](jev-native-v2-feedback-complete-results.json) and [4/10 V2 control](jev-current-nf4-v2-complete-results.md). Raw DOM, requests, answers and weights remain local. The separate JevBench scores remain public231 195/231 (84.42%) and coherence mini 41.26/100.
