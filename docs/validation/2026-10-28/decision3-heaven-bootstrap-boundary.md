# Heaven CUDA bootstrap trust boundary

The reviewed source-only bootstrap has two entry modes. An independently
supervised invocation starts its fixed
`jev_decision3_jevbench_heaven_bootstrap_release_v2_20261028.py`
file with isolated Python for the supervisor. The supervisor retains its
existing systemd resource limits and starts that same release by path with
`-I -B -s` in owner mode. Execution from standard input is deliberately
unsupported. No release or new evaluation plan is created by these source
files or their synthetic tests.

Python reads and begins executing the bootstrap release before the bootstrap
can check the plan and selected source hashes. The caller must authenticate
the bootstrap release, interpreter, standard-library environment and exact
plan SHA independently before launch. A file replacement between that external
check and Python's open is not prevented. The bootstrap verifies its exact
source allowlist and gate-only release pairs before loading selected workspace
modules from the verified bytes; that does not close this entry-point race.

The pinned publisher manifest and pre/post source checks do not make dynamic
Hugging Face imports, external Python packages or native libraries an atomic,
fully authenticated supply chain. The GPU, cpuset, complete official
evaluation and actual runtime compatibility require separate supervised runs;
synthetic tests do not establish any score or real-world task success.
