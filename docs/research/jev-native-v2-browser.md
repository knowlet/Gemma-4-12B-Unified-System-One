Research decision, October 22, 2026, before implementation.

This composes the current NF4 action deployment with the unchanged V7 Chromium
runner for all four original and six public transfer tasks. It is a new NF4
experiment. The historical Q95/BF16 V7 result, 5/10, remains a reference rather
than a precision or helper matched control. Explicit raw and V2 modes allow the
verified raw parent and native V2 child to share the same current NF4 helper.

The search is deliberately limited to the already available, pinned upstream
checkout and the existing deployment research in `jev-native-v2-fit-deploy.md`.
No external implementation or dependency is copied. Two local searches examined
browser/client HTTP semantics and model/adapter/helper lifecycle separately.

| Candidate | Evidence read | Decision |
| --- | --- | --- |
| Browser Use Jev | Local upstream commit `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`, `jev_ultrafast/model.py:choose/field_text`, `questions.py:TEXT_VALUE`; local `LICENSE` is MIT, copyright 2026 Browser Use | Directly use the existing V7 runner, original request construction, 25-second HTTP client, read-only site scope and independent final-DOM judges. No new browser executor or answer repair. |
| Existing local deployment | `scripts/jev_local_model.py`, `s1.browser_action_adapter`, `s1.api.create_app`; repository MIT | Directly reuse preweight artifact verification, explicit prompt-format validation, NF4 loading, frozen adapter attachment and generated-action API. The old CLI's callback and profile choices do not provide a V7 HTTP composition. |
| Existing constrained helper | `scripts/jev_field_syntax_helper.py:prepare_helper/generate_field_text`, `jev_field_json_grammar.py`; repository MIT | Directly use the full native grammar, actual EOS requirement, 128-token/8192-context/20-second limits and original adapter-disabled context. Bind current NF4 sources without inheriting Q95 qualification. The simpler `browser_text` generator has no equivalent whole-helper deadline. |
| Existing V7 bundle | `jev_browser_profile_v7_benchmark.py:prepare_bundle/verify_bundle`, SHA `31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e` | Prepare its separate original/heldout bundles and invoke their unchanged runner CLI in separately owned CPU processes. Preserve 16 decisions, seed 20261014, viewport, navigation journals and all ten denominators. Do not replace protected module globals or weaken original bundle pins. |

The new orchestration owns a loopback API and adds only the upstream-compatible
helper endpoint. It records exact request/response bodies, joins policy server
events, serializes policy/helper access and preserves errors. A separate durable
resource journal uses the existing full BF16 load formula even for NF4:
`max(24 GiB, ceil(MemTotal/5)) + 2 * 23,919,549,408 bytes`; no resident or swap
credit. Checks precede construction and each actual native forward. These are
current execution observations, not an old training or qualification record.

The command is dry unless `--run` is explicit. Portable fake tests cover
preweight format refusal, HTTP/helper contracts, helper isolation, source/bundle
composition and complete fixed case selection. Actual NF4 startup, full ten-task
execution, client/server joins and external process/cgroup closure remain for
root to review and run after the real trained artifact exists. An in-process
receipt cannot attest that its own process has exited.

Implementation seam findings: NF4's attached guard exposes `_check()` rather
than the helper prototype's `check(full=True)` and `_check(disabled=True)` API.
An explicit current-NF4 bridge delegates the original contexts and metadata and
checks actual PEFT disabled state, frozen parameters and model/head/version/T
identity. Its `full=True` means complete checks of these declared runtime
identities; it is not Q95 full-weight or BF16 numerical qualification. The
resource hook uses the actual backbone because PEFT generation can bypass the
outer PEFT forward method. Existing PID/start ownership and cleanup are reused
from `public-affordance-matrix-harness-20261019-01/matrix_common.py`.

Portable validation: 34 fake-runtime tests passed, including real FastAPI HTTP
contracts with the original attached-guard contexts, exception restoration,
raw HTTP/API request-ID joins, refusal on policy evidence failure, all ten V7
case selections, and successful/interrupted orchestration with retained partial
group counts. Cleanup and publication failures retain the original browser
exception with secondary notes. Unconfirmed server drain stays explicitly
unconfirmed and produces no output index; a failed receipt or index publication
attempts an atomic failed-receipt update. The whole HTTP deadline raises
`TimeoutError` and returns 504 after lock acquisition, immediately before
dispatch, and after a candidate response; no expired candidate is returned as
an authorized action. The fake regressions cover each deadline boundary and
cleanup, receipt and index failures.

The final tests passed in 1.88 seconds under a 256 MiB address-space limit and
30 CPU-second limit, with JUnit at
`/tmp/jev-native-v2-browser-fixes-20261022.xml`. Ruff checks and formatting passed
under the same limits with `MALLOC_ARENA_MAX=1 RAYON_NUM_THREADS=1`; the allocator
setting avoided a prior formatter allocation failure within that cap. These
tests ran no model, CUDA, Chromium, cloud worker or actual browser request.
