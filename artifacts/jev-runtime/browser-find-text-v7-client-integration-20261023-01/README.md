# Optional FIND_TEXT client for the unchanged V7 benchmark

This source component selects the existing FIND_TEXT Agent inside a new isolated V7 runner. It preserves the original model adapter, task order, V7 observation helper, judges, seed20261014 and16-step budget. It has no browser or model score.

Preparation requires explicit profile `jev-v7-find-text-client-source-v1`:

```python
bundle, manifest = selector.prepare_bundle(new_directory, case_module="original", profile=selector.PROFILE)
# Independently prepare the second group with case_module="heldout".
```

The selector derives each group from the checked original V7 builder. It changes one private Agent constructor call, adds private selection hooks, and proves the complete inverse equals the original parent runner. The factory executes inside the original model adapter, so the selected class captures its installed transport wrappers. Original Agent and choose globals remain unchanged. All other parent bundle bytes are retained. The private main accepts only the complete original ordered4 or public6 tasks, one development repeat, seed20261014,16 steps and the fixed V7d9f helper.

An explicit HTTP text endpoint and model are required. The inherited FIND implementation compiles field_text from model.py and would bypass a callback installed only on agent.field_text; this profile rejects callback-only helpers before entering the original adapter. The inherited20-second helper action budget remains unchanged.

Seventeen CPU tests pass. They exercise the actual canonical Agent, choose, field_text and retry implementation plus the original model adapter, using explicit fake HTTP and fake Browser replies. They verify503 retry instrumentation, separate policy/helper records, response model identity, redaction, restoration, fixed arguments and bundle tamper refusal. There is no network, actual browser, model or GPU operation in these tests. XML suite time is4.548s; pytest wall output is4.68s. Source review is independently CLEAR.

A future separately named owner must explicitly bind this selector with the compatible c371 S1 FIND service selector, the complete source union, actual loaded canonical modules, current Q95 helper identity and liveness, runtime/cache/geometry/text limits, resource authority and final closure. This component supplies neither a launch plan nor helper admission, model qualification, actual V7 execution, ten-task score or cross-model comparison. Source preparation used temporary directories and removed them afterward. See source-parity.json, validation.json and independent-review.json for the precise scope.
