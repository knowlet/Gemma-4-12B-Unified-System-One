# Matched V7: first launch refused before model execution

The first selected launch at commit `6485648` exited after 45.52 seconds at
the original host memory admission check. The actor directory was never
created, and none of the ten new CPU outputs existed. This is an infrastructure
admission refusal, with no new model result or complete browser score.

The original memory reserve, model allowance, resource caps and ComfyUI idle
requirement remain unchanged. A later host observation found one active
ComfyUI task; the authorized model unload request is deferred until that task
returns. No task was interrupted or removed from its queue by this cleanup.

Plan03 and launcher02 select a fresh `matched-02` output and nonce. The model,
source selection, unit name, original CPU outputs 31–40, browser protocol and
judges are unchanged. The launcher checks that all selected outputs are still
fresh before executing. The previous plan, intent and terminal are preserved.

Both new JevBench scores and the complete matched browser score remain
unmeasured. The two JevBench suites continue to require separate reporting.
