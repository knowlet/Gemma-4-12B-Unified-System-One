# Private model endpoint attribution

2026-10-27. A prior paired development run retained one HTTPX capture whose
model-service attribution was unknown. The capture must not expose a URL,
header, credential or case identity. Existing output consumers must retain the
old schema by default, and a transport exception cannot prove delivery.

| Reference | Evidence | Decision |
| --- | --- | --- |
| [HTTPX 0.28.1](https://github.com/encode/httpx/blob/26d48e0634e6ee9cdc0533996db289ce4b430177/httpx/_models.py), [BSD license](https://github.com/encode/httpx/blob/26d48e0634e6ee9cdc0533996db289ce4b430177/LICENSE.md) | `Response.request` returns the attached request or raises; redirects have history, and content can be decoded. | Reuse the installed HTTPX response attachment, preserve existing no-redirect and body handling. |
| [Jev pinned model](https://github.com/browser-use/jev-ultrafast/blob/1231850a0bf1a0c0341fe408ef1668dbbfdfac46/jev_ultrafast/model.py) | `post_json` uses a shared client and retries; policy and text helper use separate configured destinations. | Reuse existing runner/capture, do not copy upstream code or infer endpoint from body content. |
| Existing runner and capture | The runner configures destinations; capture rejects unsafe bodies and withholds endpoints, but cannot classify the unmatched event. | Extend only the explicitly opted-in private capture contract with a fixed destination-match enum, and keep unknown for missing, overlapping, or nonmatching destinations. |

The enum describes the attached request destination, not its semantic purpose,
delivery, complete transport coverage, model output, or task success. Exact
matches to explicitly configured, credential-free URLs must be performed in
memory; never emit the URL or its digest. Preserve retries, error propagation,
source gates, cleanup and legacy schema. Add synthetic HTTPX tests for exact,
unknown, overlapping and exception cases, plus bootstrap integration and
privacy checks. Actual service attribution still requires independent binding
and a new supervised run; prior unmatched captures cannot be reclassified.
