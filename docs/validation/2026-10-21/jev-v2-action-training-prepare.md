The action compiler and CFG validator now accept an explicit shared-element
prompt format. This closes a training-preparation gap: a LoRA trained with raw
JSON does not establish that it learned the compressed V2 representation.
Historical raw records keep their original shape; shared records authenticate
the requested format, effective raw fallback, complete request and legal actions.
The fit CLI and exported adapter manifest still use the existing raw profile.

The GX10 ML environment passed all **104** selected regression tests with no
skips. These use tiny CPU fixtures, including gradient and LoRA update checks;
they do not load the 12B model. A Torch CUDA device-enumeration OOM warning was
retained. CUDA was not initialized before or after testing; that does not prove
zero driver queries. The dedicated test unit exited successfully and was removed.

The historical-data reader authenticated 179 existing rows from 45 selected data
files: 142 train and 37 development. It preserves original wire/teacher and split
joins without constructing an old admission object or creating labels. Its 16
tests pass; the preparation driver's 13 fake tests and external guardian's 5 fake
tests also pass. These counts establish historical data consistency and tool
behavior, rather than completed native processing of that population.

The actual full-population preparation loaded the processor, then refused the
unchanged resource gate: available RAM fell to **26,534,883,328 B**, below the
**29,340,397,568 B** requirement. It completed **0 of 179 rows**, with no RAW/V2
comparisons or CFG plans. All partial evidence remains local. The original
3 GiB unit limit, zero swap limit and resource reserve were retained; ComfyUI
was observed without interrupting or unloading it.

The failed unit and both owned actors are closed. All 132 current source pins
remain unchanged. The largest recorded live cgroup peak was 1,101,082,624 B;
systemd's cached terminal peak was lower at 1,065,381,888 B. Final cgroup counters
are unavailable after removal, so no final zero-OOM claim follows.

The [aggregate result](jev-v2-action-training-prepare-results.json) records the
exact evidence references. Published files contain code, aggregate results and
hash references; raw wire bodies, processor copies, prompts and token arrays stay
local. A fresh attempt still needs stable RAM and its own output namespace.
Native V2 training/export qualification, real browser retesting and all six
Decision3 comparisons remain unfinished. No new browser result was measured;
the latest original public V7 result remains 5/10.
