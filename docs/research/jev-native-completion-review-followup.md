# Native completion review follow-up

2026-10-22. This is a proposed separate agent protocol. A callback-only client
component now exists at
`artifacts/jev-runtime/browser-completion-review-profile-20261022-01/`.
It has no actual generic-native server route, launch profile, model run or quality
qualification. The existing ten-task results remain unchanged.

The independently replayed Counter comparison identifies a concrete failure:
the feedback arm issued DONE while the requested method signature and description
remained below the viewport. Its input text already differed before feedback,
so that comparison cannot isolate feedback as the cause. The saved diagnosis is
`artifacts/jev-runtime/native-v2-feedback-counter-diagnosis-20261022-01/summary.json`
(SHA `6b196f3a0e544536a260e1bd9e572dd8f256b4e44f9cf6404e826967666d9b13`).

The pinned jev-ultrafast `Agent.command("act")` checks page freshness before
accepting DONE, then stops. It does not independently assess whether the user's
goal is visible. Its source SHA is
`aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b`
at upstream revision `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`.

The larger Browser Use agent offers a separate optional trace judgement. At the
inspected revision, that verdict is attached for comparison with the agent's
own success report; it does not replace that report or resume navigation after
an adverse verdict. Its judgement path can also accept ground truth, which must
not be exposed to a navigation policy in this experiment. This is background
for separating self-report and verification, rather than an implementation to
copy unchanged. See the pinned primary source:
[Browser Use service.py](https://github.com/browser-use/browser-use/blob/aa093a843da5edbf323c0b4145c9b6a0db3ecde8/browser_use/agent/service.py#L1454).
The existing upstream research records its MIT license.

A prospective native reviewer would receive only the user's goal, current
observed page, actual history and an explicit statement that completion requires
the requested visible evidence. It would make a separate binary native decision
when the policy proposes DONE. All submitted heads and their order must remain
intact; there is no fixed fifteen-head browser geometry. Both S1 and Decision3
need an explicitly selected generic native-question route, since the current
browser parser requires its operation criterion. The text-helper field-value
interface must not silently become a goal judge.

The agent must retain the original DONE proposal and reviewer answer separately.
A negative or unknown review is not a successful task and must not fabricate a
different native operation. It would return control to navigation, with typed
completion feedback, while consuming the same sixteen-step budget. A stale page,
timeout or invalid review must not approve completion. Both forwards, admission
checks and any publication must fit an explicitly declared request budget.
The client prototype uses a shared twenty-second policy-and-review interval,
sixteen counted action slots, and thirty-two logical policy/review attempt slots.
Failed policy attempts remain counted. HTTP retries, backend forwards and native
question heads require separate server accounting; this client does not claim
that thirty-two logical attempts bounds those other populations.

The reviewer must never receive benchmark judges, expected routes, target DOM
selectors, success labels or a failed external judge's answer. External task
scoring remains independent and unchanged. This protocol may itself lose valid
successes or create repeated vetoes, so it requires the complete matched ten-task
comparison before promotion.

Necessary verification is a real agent flow with an ordinary synthetic long
document: false DONE is retained and vetoed without claiming success; a visible
goal may be accepted; stale, late and failed reviews remain incomplete; task and
request budgets hold. Model quality still requires real model calls after local
GPU availability returns. This is separate from the independently tested
FIND_TEXT component, which addresses navigation reachability.
