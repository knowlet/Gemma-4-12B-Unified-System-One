# Checked-control state progress training

Decision recorded on 2026-10-25 after `baa9f75`, before implementation. The
[first-case paired diagnostic](../validation/2026-10-25/decision3-firstcase-gpu-paired-results.md)
found a correctly executed initial radio selection followed by three clicks on
the same checked control. TYPE_TEXT and SELECT remained offered. Both candidate
and matched NF4 baseline failed the original task. The full comparison remains
incomplete. This observation motivates generic training coverage; the exposed
case, goal, DOM, requests and answers must not become optimizer inputs.

## Sources and decision

Searches covered public checkbox task state/reward implementations and browser
radio activation semantics. Relevant evidence was read from the original
repositories and standards, rather than search summaries. No new dependency or
upstream revision is selected.

| Reference | Mechanism read | Decision |
| --- | --- | --- |
| Existing project MIT `jev_decision3_affordance_scenarios.py` and `jev_decision3_affordance_collection.py` at `baa9f75` | Authored routes, actual DOM/event proofs, complete publisher wires, goal/ancestor exclusion and independent admission | Reuse collection contracts; preserve the four old families and their K4/K7/K8 profile. Add a separate module and explicit profile. |
| [MiniWoB++ click-checkboxes](https://github.com/Farama-Foundation/miniwob-plusplus/blob/33c3b4ddef8c6eb67c57a29663d844b1eda7e614/miniwob/html/miniwob/click-checkboxes.html), `createCheckboxes` and submit callback; [MIT license](https://github.com/Farama-Foundation/miniwob-plusplus/blob/33c3b4ddef8c6eb67c57a29663d844b1eda7e614/LICENSE) | Generates alternative controls and compares each real DOM `.checked` boolean with the requested selection on submission | Design reference only. Its partial checkbox reward is unsuitable for the unchanged Jev task judge and the required radio/text/select sequence. Copy no code or task data. |
| [WHATWG radio-button state](https://html.spec.whatwg.org/multipage/input.html#radio-button-state-(type=radio)), read 2026-10-25 | Boolean checkedness, radio-group exclusion, activation input/change events | Independently verify actual group state; never coerce wire strings such as `"false"` to booleans. |
| [W3C WebDriver draft, events](https://www.w3.org/TR/2026/WD-webdriver2-20260702/#events) | Native activation events are trusted | Semantic reference only. Keep the existing pinned CDP executor and its distinct synthetic SELECT contract; do not replace execution with page-script clicks. |

The new non-scrolling family contains a radio group, text field, select and
submit button. It rotates visible labels, desired values and DOM ordering by
seed. The authored route is unchecked desired radio CLICK, TYPE_TEXT, SELECT,
submit CLICK, then freshly verified DONE. Checked radio controls remain visible
and clickable alongside the incomplete fields. Every edit invalidates the
submitted result. Submission records actual values, rather than the desired
answers embedded in a page script.

The collector will bind the new profile, module hashes and complete generated
population into its plan, replan and independent admission. Its natural menu
is expected to be K6, including terminal choices, but real publisher observation
must establish that count. No pruning, renumbering or forced menu injection is
permitted. The old profile retains its exact K4/K7/K8 coverage requirement.
Existing collections and the 109-row fit remain separate immutable artifacts.

## Verification and limits

Source controls must reject string checkedness, incorrect or multiple checked
radios, untrusted clicks, no-effect repeated radio clicks, premature DONE,
incorrect submitted values and profile tampering. Replay must preserve original
wire bytes, all targets and criterion order. A genuine model-free browser
collection and independent admission must subsequently prove actual transitions
and naturally offered menus before any new feature capture or fitting.

All new scenarios are train-only, with the existing normalized goal hashes and
scenario/ancestor exclusions. Teacher metadata stays outside native requests.
Source controls and a scripted-teacher collection provide no model accuracy
score. Operation-only supervision can teach the switch to TYPE_TEXT or SELECT;
it cannot directly distinguish submit from radio when both are CLICK targets.
If target selection still fails, it needs separate target adaptation evidence.
Complete native browser comparison, both fresh candidate JevBench scores and
competitor superiority remain required and unproven.

## Native submit event correction

The first actual two-family run naturally offered K6 and completed radio, text
and select transitions in each scene. Both subsequent native submit clicks
produced the correct requested live and recorded values, but the new proof
rejected them. The text field was still focused: clicking submit first emitted
its trusted `change` event with the unchanged completed text, followed by the
trusted submit-button click. The verifier incorrectly required every appended
event to target the button. The entire failed collection remains quarantined.

This agrees with the [HTML common event behavior](https://html.spec.whatwg.org/multipage/input.html#common-event-behaviors):
text changes can be committed on loss of focus. Before implementation of the
correction, select the following rule: allow either the existing single button
click or one trusted text-field change before that click, only at the current
unchanged text value. Continue to require unchanged inputs, actual correct
recorded values, a complete native executor join and newly verified goal state.
Reject altered values, untrusted changes, wrong targets or types, and events
after the submit click. Do not suppress the browser event or change the executor.
Add distinguishing positive and refusal controls, then collect a fresh batch.
