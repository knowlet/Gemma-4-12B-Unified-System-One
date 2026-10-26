# BF16 mmap／逐層 offload：只讀可行性判斷

結論：**底層機制值得後續獨立量測；目前無法證明能在約29GB可用記憶體、8192 context及20秒預算內工作，也不能沿用原Q95／NF4／N3資格。** 這不是降低原74GB初始門檻的理由。最有價值的下一步是小型storage／hook相容性量測，再決定是否值得設計完整loader；本次沒有實作或執行該量測。

本次只讀小型config、已保存runtime metadata及Python來源，另查官方文件。沒有開啟、hash、mmap或載入23.9GB權重payload，沒有import torch、建立model／CUDA／服務、呼叫Comfy或外部運算，也沒有改既有來源或resource gate。約29GB是父任務提供的現況，沒有重新觀察主機。

## 已核對的事實

實際overlay METADATA為Transformers5.17.0、Accelerate1.15.0、safetensors0.8.0、PEFT0.21.0。已保存Q95 runtime metadata記錄torch2.13.0+cu130；本次沒有載入或重新驗證該torch runtime。

| 決策問題 | 本機來源所支持的答案 |
| --- | --- |
| 現有BF16 loader必然先複製一整份CPU anonymous weights嗎？ | **不能如此斷言。** Transformers5.17預設在meta建立模型，Linux safetensors路徑使用`safe_open(...backend="mmap",device="cpu")`與lazy slices；`_materialize_copy`實際是切片後`.to(device,dtype)`，同CPU／同dtype時沒有明寫clone，`set_param_for_module`直接包成Parameter／setattr。轉dtype、weight conversion、非mmap後端或其他載入分支仍可能配置新storage，實際anon/file比例尚未量測。 |
| S1會保留這種CPU mmap部署嗎？ | **現行BF16路徑不會維持逐層offload。** `UnifiedDecisionModel._initialize`在quantization none時執行整個`lm.to(device)`；目前呼叫選cuda。NF4路徑則明定整個模型到單GPU的device_map，且標示cpu_offload false。`from_components`仍呼叫同一_initialize，並非現成offload入口。 |
| `cpu_offload`能保留外部mmap-backed CPU state_dict嗎？ | **來源上可行，但有條件。** Accelerate接收傳入state_dict並保留引用，offload hook由weights_map取tensor；若傳入的是既有BF16 file-backed tensor且不改dtype／不clone／不原地寫入，沒有必須先複製全CPU權重的步驟。若省略state_dict，預設由模型`.to("cpu")`建立；從GPU呼叫會產生CPU副本。GPU執行的`.to(cuda)`仍會配置／傳輸GPU tensor，不等於GPU直接零拷貝讀取檔案。 |
| `device_map`與disk offload都等於保留原檔mmap嗎？ | **否。** `dispatch_model`可使用外部state_dict／safetensors offload_index；`OffloadedWeightsLoader`的safetensors_file分支直接safe_open。相反地，普通`.dat` disk offload會另寫numpy memmap，`load_offloaded_weight`再用`torch.tensor(np.memmap(...))`建立tensor，包含CPU複製。`offload_state_dict=True`是暫存記憶體控制，不能當成零拷貝保證；Transformers5.17的from_pretrained還會移除舊`low_cpu_mem_usage`／`offload_state_dict` kwargs。 |
| 所謂read-only是何種保證？ | safetensors0.8官方Rust實作對torch使用`from_file(shared=False)`；這不是把Parameter變成不可寫物件的安全邊界。唯讀使用、禁止merge／原地更新與驗證原檔完整性仍須另定。寫入或dtype轉換可失去原file backing。 |
| 自動device_map能直接當UMA實體RAM預算嗎？ | 不行。Accelerate1.15已在`get_max_memory`偵測`device_properties.is_integrated`，成功時不另加CPU額度，避免重算共享RAM。本次沒有查詢GB10實際flag，也沒有驗證CUDA allocator與host/cgroup accounting；不能假設自動值已滿足本專案實體記憶體／reserve規則。 |

來源：[S1初始化](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/src/s1/unified.py:200)、[量化配置](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/src/s1/quantization.py:58)、[Transformers載入](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/transformers/modeling_utils.py:4420)、[materialization](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/transformers/core_model_loading.py:1236)、[Accelerate cpu_offload](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/accelerate/big_modeling.py:179)、[offload storage](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/accelerate/utils/offload.py:48)、[integrated CUDA budget](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/accelerate/utils/modeling.py:757)。

官方依據：[Accelerate1.15 API](https://huggingface.co/docs/accelerate/v1.15.0/en/package_reference/big_modeling)、[Big Model Inference](https://huggingface.co/docs/accelerate/v1.15.0/en/concept_guides/big_model_inference)、[safetensors0.8 Rust storage實作](https://github.com/safetensors/safetensors/blob/v0.8.0/bindings/python/src/lib.rs)、[Accelerate1.15 offload實作](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/offload.py)。文件描述一般能力；本機實際來源優先。

## Config推算，並非峰值量測

已讀原snapshot的小型config：hidden_size3840、vocab262144、48層；40層sliding attention為8 KV heads×256維，8層full attention覆寫為1 KV head×512維，sliding_window1024，BF16。`attention_k_eq_v`不表示可以把cache容量除二：實作仍分別正規化／儲存K和V。embedding與lm_head設定tied，模型也宣告相同tie；實際storage是否仍共享需載入後驗證。

| 項目 | Config／來源估算 |
| --- | --- |
| 單份BF16 embedding／tied lm_head | 262144×3840×2＝2,013,265,920 bytes＝**1.875GiB**。tie被破壞會增加一份。 |
| 一個sliding decoder block主要矩陣 | 約**427.5MiB**，未計norm、LoRA、buffer及workspace。 |
| 一個full decoder block主要矩陣 | 約**461.25MiB**，同上；不應以平均層大小安排最大層預算。 |
| 8192-token邏輯KV，batch1／BF16 | full層128MiB，加40層各保留1023 sliding tokens，合計**447.6875MiB**。這是形狀估算，不是配置峰值。 |
| 長prefill的KV backing風險 | 已安裝DynamicSlidingWindowLayer先cat整段，再以基本切片保存last1023，沒有clone；切片view可能繼續持有整段storage。按此來源路徑，8192 prefill可達約**2.625GiB**的cache backing；native32輸出上限對應最大8160 prompt，約**2.615GiB**；Q95 helper128對應最大8064 prompt，約**2.584GiB**。首輪decode重新cat較短視窗後可改變此占用。這是來源推論，尚未量測實際storage／allocator釋放時機。 |
| MLP中間activation | 8192×15360×2約240MiB／tensor；同時存活幾份、SDPA後端與workspace另計。 |

cache證據：[DynamicLayer／SlidingWindowLayer](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/transformers/cache_utils.py:205)、[Gemma4 attention／異質層](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/transformers/models/gemma4_unified/modeling_gemma4_unified.py:342)。[PyTorch官方view規則](https://docs.pytorch.org/docs/2.14/tensor_view.html)支持基本切片共享storage的一般推論；該文件為2.14，並非本次torch2.13 runtime量測。

不能把上述尺寸相加就宣告已證明不可行：file cache是否可及時回收、UMA／CUDA匿名頁、allocator保留、cache view的生命週期、SDPA workspace與MemAvailable／cgroup歸帳都未知。但也不能把447.7MiB當峰值，或把mmap虛擬位址當成不耗實體RAM。僅head加長prefill cache backing已是約4.5GiB量級，約29GB現況沒有已證明的安全餘裕。

## 三種S1路徑須分流

| 路徑 | KV／head語義 | offload相容性限制 |
| --- | --- | --- |
| 普通S1 choice readout | backbone `use_cache=False`，只投影所選head rows；不應套用上面的生成KV總量。 | `head.weight[ids]`在module.forward之外直接取值，若head處於meta，普通pre_forward hook不會自動救援；通常須保留dense head或另設明確讀取接縫。 |
| 現有N3 browser generated JSON | NF4 base＋N3；`lm.generate(use_cache=True,max_new_tokens=32)`，8192 context、20秒合作式期限。 | 仍需完整vocab lm_head，不是A–Z列讀取。標準generate通常自動`logits_to_keep=1`，所以不能一律加上8192×vocab的4GiB BF16 logits；實際PEFT/generation分支仍需核對。grammar是在logits產出後限制token，不會免除head權重。 |
| 原Q95 BF16 text helper | base-helper語義、adapter disabled；`generate(use_cache=True,max_new_tokens=128)`、8192 context、20秒。 | 保留BF16值只是必要條件；原helper snapshot釘所有parameter identity／version／dtype及LoRA狀態，offload的meta↔device Parameter替換不符合原identity資格。 |

來源：[普通choice](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/src/s1/browser_policy.py:499)、[N3生成](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/src/s1/jev_generated.py:404)、[Q95 helper生成](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/scripts/jev_field_syntax_helper.py:541)、[generation最後一個token logits](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/overlay/transformers/generation/utils.py:2746)、[Q95完整parameter snapshot](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/scripts/jev_qualify_public_field_syntax_bf16.py:2715)。

原`attach_action_adapter`還明確要求`model.quantization == "nf4"`，並檢查head／backbone／temperature identity；不能拿BF16 offload模型直接走此入口，或重標成原NF4 N3。Accelerate的pre／post hooks會把Parameter設到meta、再從weights_map重新建立到執行device；PEFT wrapper／LoRA注入順序、hook保留、adapter enable/disable、forward evidence、tied head及身分檢查都要在新profile重新驗證。不能用整個Gemma class的preload來掩蓋直接權重存取問題，因為可能一次重新載入整個模型。

20秒也尚未成立：全CPU／disk逐層offload會在每次autoregressive forward重載未駐留權重。以約22GiB文字權重、最多32／128個生成步驟做粗略交通量估算，可能達數百GiB至TiB量級；這不是SSD實際讀量或時間下界（檔案頁可能命中cache、部分層可常駐），更不是已證明超時。需要真實傳輸／prefill／decode／fault時間才能判斷。官方也將disk列為最慢選項；API可用不代表本工作負載達標。

## 值得後續量測的最小問題

先限定獨立的新profile，保留原base revision、BF16／LoRA數值、chat template、temperature、8192 context、原20秒期限與fail-closed輸出；原Q95／NF4比較分數繼續分開。

1. 先用小型獨立fixture確認file-backed storage生命週期、同dtype不複製、外部state_dict、tie／hook／PEFT狀態及meta參數復原；不得由此宣稱12B可載入。
2. 只有取得獨立、可審核的實體記憶體預算後，才考慮真實層與head的有界傳輸量測，記錄anon/file RSS、host MemAvailable、cgroup、CUDA配置／保留、冷熱頁與單層峰值。需要自己的新准入條件；不能修改舊門檻後沿用舊qualification。
3. 再決定是否值得測長prefill及單步decode，包含實際KV storage／workspace。若這些加上原host reserve沒有餘裕，或20秒／必要輸出形狀無法成立，就停止完整loader方向。

因此目前分類是：**機制可用；既有入口不相容；目前29GB／20秒可用性尚需量測。** 這份來源研究不提供新的執行授權、准入證據或benchmark分數。

來源指紋：config SHA256 `b50e760a9107b629e1a140ab7a4392e875c37baefee022f27c98b90b9d947cf0`；S1 unified `42ba5f076af164e0e54690a735e25a272bdd678279901946c2e5accefa385a6c`；Transformers modeling_utils `38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d`；core_model_loading `c5b6bcdb6401a3cfdf825bc979d41ac3e7e075e09459021363dce5e68c54de22`；Accelerate big_modeling `aa5ff729d8177c29e757ab2fb8a4e9a2b2edd50b3a931d5d159e270066eab426`；offload `e28e36e09f05281f3a3c1a801af83cc79503362ecf0661f9de436b7792c2b85e`；cache_utils `702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c`。
