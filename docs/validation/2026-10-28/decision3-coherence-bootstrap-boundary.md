# Coherence MINI CUDA bootstrap trust boundary

The source-only bootstrap has isolated supervisor and owner entry modes using
the fixed `jev_decision3_jevbench_coherence_bootstrap_release_v2_20261028.py`
path. The supervisor keeps the existing resource limits and invokes the owner
through the same bootstrap. Its source pins include the existing selective
MINI inventory and coherence configuration; the official scorer, callback
timing, population and failure handling remain those of the original owner.

The owner must use the dedicated controlled checkout at
`/tmp/jev-coherence-mini-controlled-snapshot`. Before execution, an independent
operator must prepare and control that checkout against concurrent changes.
The existing MINI loader verifies its fixed SDK source set and six required
assets: two MINI files and four public base-suite files. The base-suite files
are dependencies of the official MINI loader; including them does not run the
full base suite. Unselected examples are outside this selection. Pre/post hash
checks alone cannot rule out transient replacement of a concurrently writable
checkout.

Python opens and starts the bootstrap release before it verifies the plan and
workspace source pins. Its release, interpreter, standard-library environment
and plan SHA require independent authentication, and there remains an entry
file replacement window. The bootstrap executes selected workspace modules
from verified bytes. Dynamic Hugging Face imports, other Python packages and
native libraries are not an atomically verified, closed supply chain.

The synthetic tests do not create a release or evaluation plan, start a GPU,
read public task content, produce a benchmark score or establish runtime
compatibility. A real run additionally requires valid resource admission,
supervision and independently audited official results.
