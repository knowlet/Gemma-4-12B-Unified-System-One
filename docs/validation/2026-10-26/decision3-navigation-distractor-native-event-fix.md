# Navigation training collection: native text-to-radio events

The first genuine collection attempted all 24 independent navigation-distractor
TRAIN scenes. Twenty-one completed; three mixed-control scenes were rejected
by the radio transition checker. The complete batch, including its 62 partial
native rows, remains quarantined. No example from that batch is admitted for
feature extraction or optimization, and it supplies no model or benchmark score.

Independent replay of the original hashed protocol journal and episode records
shows the same valid transition in all three failures: a trusted text `change`
event, followed by the requested radio's trusted `click`, `input`, and `change`.
The text value stays unchanged, the requested radio is selected with native
group exclusion, and the other controls, submission state and URL stay unchanged.
The original native action reports execution. The old check incorrectly required
every appended event to target the radio.

## Reference and implementation decision

| Reference | Relevant behavior | Decision |
| --- | --- | --- |
| [HTML common input events](https://html.spec.whatwg.org/multipage/input.html#common-input-element-events) | A changed text field can commit its value when focus moves away. | Account for its actual unchanged-value event before the radio click. |
| [WPT radio tests](https://github.com/web-platform-tests/wpt/blob/master/html/semantics/forms/the-input-element/radio.html) | Radio activation dispatches click, input, then change; group checked state remains exclusive. | Require the exact ordered trio and actual exclusive DOM state. |
| Existing scenario submit proof | Already accepts one unchanged trusted text change immediately before submission. | Apply the same narrow value/trust check to the observed text-to-radio transition. |

The fix accepts exactly the radio trio, or one trusted text `change` with the
unchanged before/after value and no checked state, immediately followed by that
trio. Extra, duplicated, reordered, foreign-target, changed-value or untrusted
events remain invalid. The native executor join, complete event history, actual
radio exclusion and unrelated-control checks remain required. The original
collector and existing failed snapshots are unchanged.

## Verification and evidence

The three related scenario/collection suites pass **138 tests** in 2.25 seconds.
The new coverage includes a mixed text-before-radio route and 17 invalid-event
or invalid-DOM variants. Lint and formatting checks pass. These tests check
transition contracts; they do not establish successful recollection or improved
model behavior.

The original failed owner and browser close successfully within their existing
limits. Independent failure review verifies 19 selected owner sources, all 195
frozen catalog sources, two matching original live process identity samples,
the original scoped closure, and current unit/cgroup absence. This establishes
closure for the observed owned cohort; it is not a universal containment claim.

| Evidence | SHA-256 |
| --- | --- |
| Original scene source | `16c60a00534447f1a09daf503a6db74f3dcb0b39bde7c2ec3dd855101689d91c` |
| Corrected scene source | `f2978cce57a8b7d2ade1ade96e7fd1374fa61ade31ab2d8fbe438dc0880598dd` |
| Corrected regression tests | `1e1b2e0e3eff9c6add9b3e6db30dcf5c179717c7172917e90bdf1a2a7751bcfa` |
| Original failed collection receipt | `e98e5043a36e042bb7f904598ffc5cb667961a9b995f5ba017e7367980e649cc` |
| Original protocol journal | `ab23f7b2dc1ab6f059e552b8a5c3ea13f0927b910712102adee427a6eaa698f7` |
| Independent three-failure event diagnosis | `eda4153167508b86854e2f406b352911b81bb296c43129b90b8152ecc39bf19d` |

A fresh full 24-scene collection and independent admission are still required.
The earlier [two-head browser result](decision3-goal-routed-two-head-gpu-browser-results.md)
remains 0/1 for both candidate and matched baseline. Full browser and both new
candidate JevBench scores remain unmeasured. The helper's refusal of a null
login-field value remains correct and unchanged.
