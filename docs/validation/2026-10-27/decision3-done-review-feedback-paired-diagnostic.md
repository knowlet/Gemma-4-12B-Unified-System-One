# DONE review feedback: paired browser diagnostic

The original runner reported **0/1 successful attempts for both the candidate
and the baseline**. This was one previously seen development case. The two arms
used different models, and the run was not a controlled ablation of feedback.
It does not establish that feedback improved or degraded performance, and it is
not a benchmark or leaderboard score.

| Original runner result | Candidate with feedback | Baseline |
| --- | ---: | ---: |
| Successful attempts | 0/1 | 0/1 |
| Actions | 2 | 5 |
| Policy calls | 5 | 16 |
| Text calls | 1 | 1 |
| Failure stage | `public_navigation_scope` | `decision_budget` |

One review call and one constrained replan were reconciled. All five candidate
service calls had matching bindings and same-call FP64 consistency evidence.
The client capture contained six exchanges; one remained unmatched, and its
attribution is unknown. Matching evidence does not by itself establish task
success or complete transport capture. The reported outcomes are the original
runner's saved judge results; the judge has not been independently rerun.

The owned candidate and outer units exited successfully and their control
groups were absent after closure. This does not prove global GPU exclusivity.

The execution plan was pinned to SHA-256
`ad0b00919fb791302f92362173f91fa955900740b9401f7cc7cd77232184bb54`.
During subsequent CI repair, three synthetic test files listed in that plan
were modified after the run began; one changed approximately one second before
the recorded closure. Their original pinned versions can be verified at commit
`876f2e4`. These files were separate from the 59 runtime source pins. The plan's
entire pin inventory therefore cannot be described as unchanged throughout the
run, and the timing does not establish that any changed test file was executed
by the paired browser process.
