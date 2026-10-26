The October 22 CPU regression runs used the existing pinned
`run_compiler_tests.py` guardian. Its 26,119,172,096-byte reserve stopped both
runs while their observed unit peaks were 1,004,314,624 and 739,303,424 bytes.
The original closures and incomplete test results remain unchanged. These
observations do not establish the eventual full-suite peak.

The inspected worker, `native-v2-training-20261022-01/integration_tests.py`,
runs synthetic tiny CPU training and selected regression fixtures, with CUDA
hidden, two Torch CPU threads, an 8 GiB address-space limit and a 540-second
alarm. It does not construct a production 12B model. Reusing the prospective
large-model host reserve for this worker unnecessarily prevents these tests.

Use a separately named CPU-only profile: a 3 GiB cgroup hard limit, zero swap,
600-second unit deadline, 16 GiB host reserve, and startup availability of
16 GiB + 3 GiB + 64 MiB. The 16 GiB value is an explicit CPU headroom policy,
not a measured capacity proof. The controller retains real host observations,
checks the actual cgroup and systemd limits, and stops only its nonce- and
InvocationID-bound unit when the CPU floor is crossed. Existing status/host
readers are reused without modifying their globals or historical controller.
Cleanup failures retain the primary exception and incomplete closure remains
unknown. Sampling cannot guarantee against unrelated rapid allocations.

The worker checks that Torch CUDA is uninitialized before and after the suite;
this does not attest that no dependency queried the driver. A CPU suite pass
must include pytest exit zero, unchanged execution sources and closed owned
cgroup. It does not qualify GPU memory capacity, 12B training or browser
quality. The GPU reserve and 116,313,485,312-byte training startup floor remain
unchanged. No dependency or external implementation is introduced; this
extends the existing repository MIT test orchestration only.
