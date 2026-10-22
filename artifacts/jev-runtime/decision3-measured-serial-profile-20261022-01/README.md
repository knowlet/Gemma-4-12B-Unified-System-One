This is an unexecuted resource-policy proposal for full D3. It preserves the original 132,618,706,804-byte admission refusal and does not claim that the hardware cannot run the checkpoint.

The installed publisher path already uses meta initialization and mmap tensor slices. Its Transformers 5.17 conversion path uses `HF_DEACTIVATE_ASYNC_LOAD=1` to select deferred serial materialization. The existing `HF_ENABLE_PARALLEL_LOADING=false` setting alone does not select that path.

The proposed first budget is a separate 6 GiB CPU/memcg limit, 58 GiB authenticated CUDA allocator cap, swap zero, and the original 2 GiB framework buffer. Add both limits conservatively: no UMA overlap, resident, or swap credit. With current selected total 130,595,864,576 bytes, reserve is 26,119,172,916 bytes; admission therefore requires at least 96,986,133,300 bytes of **fresh helper-loaded MemAvailable**. A fresh shortfall refuses before constructing the model.

The existing owner supports a 6 GiB selection. New backend, bootstrap, execution, and co-resident resource-policy seams are still required because the frozen original backend and CORE correctly refuse full D3. No active source or historical formula is changed. Numerical settings, publisher code, full checkpoint, native255, batch8, temperature1, context8192, Q95 helper, text48, and all ten tasks remain explicit unchanged requirements.

First measure construction plus exact original text48 under sealed independent caps, host sampling, and selected unit closure; retain null score. The original warmup lengths are `[37,64,320,333,1000,1024]`, so this does not certify every 8192-token request. Proceed to two fresh full-task workers only after reviewing the actual measurement. A 6 GiB memcg OOM or CUDA-cap failure remains a resource-profile failure, with no limits raised in the same run.

See `profile-proposal.json` for exact arithmetic, source references, admission order, measurement fields, unchanged numerical contract, and remaining implementation seams. There is no actual launch plan or launch authorization in this bundle.
