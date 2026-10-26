# Frozen Decision3 features verified on the new K7 collection

GX10 completed one frozen Decision3 model load and 48 native GPU calls on the
[accepted independent TRAIN collection](decision3-scroll-state-native-collection-results.md).
A separate audit replayed all 192 heads and 1,954 legal probabilities with FP64
math. All ordered argmaxes, native entropy/confidence and probability comparisons
passed; the maximum absolute probability error was `2.597254158409612e-7`
against absolute tolerance `1e-6` and relative tolerance `1e-5`.
The [aggregate record](decision3-scroll-state-gpu-feature-results.json) contains
the measured counts and provenance hashes.

Each calculation retained its complete original native request, ordered
criteria and actual head index. All 255 frozen output rows were calculated
before the original legal masks; illegal replay probabilities stayed zero.
Every baseline matched the pinned FP32 safetensors bytes. Feature capture did
not load teacher labels, trained residuals or candidate compositions, and did
not open an optimizer. The new population includes the actual seven-operation
menus and 11/15/23-choice SELECT menus established by browser collection.

The original owner completed successfully in 54.018 seconds overall and
52.377 seconds in its command. Separate replay verified both original child
identity samples, scoped closure and current unit/cgroup absence. This
acceptance covers frozen feature capture and numerical integrity. It grants
no fit or deployment authority and establishes no new accuracy, Full10,
coherence JevBench, Benchmark Heaven score or ranking.
