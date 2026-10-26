# Navigation-distractor two-head fit and prefix regression

A fresh two-head fit reaches **202/225 operation decisions** and **47/47 CLICK
targets** on its cached TRAIN features. The same old 153 operation rows fall from
**147/153 with the previous fit to 137/153 with this fit**. The additional coverage
does not produce a uniform improvement, and these in-sample results establish no
browser success or benchmark gain.

The [complete native collection and GPU feature report](decision3-navigation-distractor-collection-results.md)
provides 72 full wires, covering CLICK menus with 14, 18 or 24 choices. This fit
preserves the earlier 153 operation and 14 CLICK-target examples byte for byte,
then appends all 72 new operation examples and all 33 new teacher-active CLICK
examples. Already-correct CLICK examples remain included. Every original target
loss mask stays false; two separate, explicitly bound supervision authorities
supply labels outside the feature-only wires. Full native menus and head order
remain unchanged.

| Cached TRAIN group | Frozen W0 / zero delta | New 10,000-step delta |
| --- | --- | --- |
| Operation, all 225 | 84/225 | 202/225 |
| Operation, old 153 | 62/153 | 137/153 |
| Operation, new 72 | 22/72 | 65/72 |
| CLICK target, all 47 | 15/47 | 47/47 |
| CLICK target, old 14 | 6/14 | 14/14 |
| CLICK target, new 33 | 9/33 | 33/33 |

The old-prefix comparison uses the
[previous independently verified fit](decision3-goal-routed-two-head-fit-results.md),
which achieved 147/153 on those same operation rows. Within this new fit, the
new SELECT operation examples fall from 3/6 with zero delta to 2/6, and the
older goal-routed TYPE_TEXT examples fall from 3/4 to 2/4. A perfect CLICK-target
TRAIN result does not resolve those operation errors.

Both FP32 deltas start at zero with shape `[255, 1024]`. The frozen optimizer
core runs 10,000 steps with learning rate 0.003, L2 0.0001, seed 20261023 and
requested batch size 64; the effective CLICK batch is 47. No earlier delta is
loaded. The encoder and frozen baseline readout stay unchanged. Unused output
rows remain exactly zero from row 8 for operation and row 24 for CLICK target.
No DEV or sealed examples enter the optimizer. DEV has informed model selection,
so these TRAIN results are not presented as an untouched evaluation.

The genuine five-core CPU fit takes 39.01038612698903 seconds under the original
8 GiB, zero-swap owner; the selected command takes 37.632246649009176 seconds.
A separate owned CPU process takes 7.0040291479672305 seconds, with a
5.117741886002477-second selected command, to independently recompute zero and
final FP64 metrics. It checks full-TRAIN loss, gradients, valid-set correctness
and margins, and rebuilds all 288 new native head mappings. The final review
reauthenticates 1,001 file identities and both original scoped closures, and
checks current unit/cgroup absence. It **does not rerun the optimizer steps**;
those steps are bound by the frozen source and original receipts.

An initial pure metadata preparation refuses a 73-entry source inventory before
starting any fit. The corrected plan projects exactly 24 historical feature-proof
JSON/log references out of the parent's direct source list while retaining all
31 Python sources. The complete 73-entry inventory remains authenticated before
and after fitting. The parent pins 49 projected entries plus five fixed entries,
within its unchanged 64-entry limit. No source bound or process limit is raised.

The [machine-readable report](decision3-navigation-distractor-two-head-fit-results.json)
contains aggregate metrics and immutable evidence hashes. At this milestone,
GPU composition and this candidate's browser behavior remain unverified. The
next numerical check retains all 72 full wires in two fixed, consecutive groups
of 36 within the existing per-owner deadline. Both JevBench scores and the full
browser score remain unmeasured for this candidate. The previous candidate's
[matched browser pair](decision3-goal-routed-two-head-gpu-browser-results.md)
remains 0/1 versus 0/1; no ranking or win is claimed.
