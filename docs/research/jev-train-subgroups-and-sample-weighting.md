# Diagnose TRAIN operation regressions before changing the objective

Research checked on 2026-10-26. The completed navigation experiment improved its
new TRAIN operation rows while reducing accuracy on the retained TRAIN prefix.
The matched browser diagnostic still failed and repeatedly clicked a checked
radio. These observations do not establish a single cause or qualify a model.
Compare frozen readouts on the same admitted TRAIN features before selecting
another fit. Preserve every row's offered operation labels and original soft
targets; keep teacher annotations separate from publisher wires.

| Reference | Relevant mechanism | Decision |
| --- | --- | --- |
| Existing frozen NumPy head core | Masked soft-target CE, zero-start FP32 residual SGD, unchanged frozen baseline | Extend a separately versioned primitive; preserve the original source and accepted artifacts. |
| [NumPy `average`](https://github.com/numpy/numpy/blob/f9f1772ce010001208475f751d19253469788410/numpy/lib/_function_base_impl.py), BSD 3-clause [license](https://github.com/numpy/numpy/blob/f9f1772ce010001208475f751d19253469788410/LICENSE.txt) | Its weighted branch divides by the sum of weights. | Keep NumPy as the existing dependency. Do not use this reduction for the proposed per-row SGD objective. |
| [PyTorch probability-target CE](https://github.com/pytorch/pytorch/blob/b43489ea06ed9a3bb80dbbd18300f8b0ef6f56a1/aten/src/ATen/native/LossNLL.cpp), BSD-style [license](https://github.com/pytorch/pytorch/blob/b43489ea06ed9a3bb80dbbd18300f8b0ef6f56a1/LICENSE) | `cross_entropy_loss_prob_target` computes log-softmax and reduces probability-target losses by row count. Its `weight` input is a class vector. | Reference the reduction design only. Do not add a framework dependency or mistake class weights for sample weights. |

Both upstream implementations and licenses were read at the linked commits.
Searches covered NumPy weighted averaging and PyTorch probability-target CE.
The installed PyTorch version's documentation was unavailable through the web
tool; the pinned source comparison is design evidence, not local runtime
verification. No upstream implementation is copied into the proposed primitive.

The prospective objective is
`sum(sample_weight[i] * soft_CE[i]) / N + 0.5 * l2 * ||delta||²`.
Uniform sampling without replacement uses the batch row count as its gradient
denominator. Normalizing each batch by its sampled weight sum would define a
different update. L2 remains independent of sample weights. Require a finite,
strictly positive FP32 weight for every admitted row, unchanged masks, zero
initial residuals, the same random sampling and explicit hyperparameters.

The first diagnostic compares W0, the old 153-row operation readout and the new
225-row readout with independent FP64 masked logits. Report valid-target
correctness, soft CE and probability margins by the old 119, old 34 and new 72
row groups, then by teacher operation and literal checked-radio/history state.
Empty groups have zero rows and null metrics. Neither goal-text matching nor
model predictions define subgroup membership. These are TRAIN diagnostics;
they cannot become browser success or either JevBench score.

Before an actual weighted fit, verify independent FP64 loss/gradient agreement,
finite differences, all-one equivalence to the original core, illegal-tail and
input immutability, invalid-weight rejection and deadline propagation. The
prospective 12-scene tall-page curriculum separately adds checked-radio SELECT
and both explicit post-radio edit orders with large dropdowns. Its seven-choice
operation menu must be observed from the native browser; it cannot be fabricated
or filtered to match the expectation. Any unexpected population is quarantined.
No new collection, weighted fit or performance improvement is established by
this research note.
