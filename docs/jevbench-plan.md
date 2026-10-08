# Jev 評測與上榜準備

更新日期：2026-10-08。這份文件區分本專案的實測、公開評測重播與榜單維護者的正式評測。尚未執行的項目保留 `not_run`，正式排名由榜單維護者確認。

## 已確認的專案狀態

[PR #6](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/6) 的 head 為 `03860a3423f4f9fb98ebf9b37f8509e33d4ae5cd`，已整合至本機 `codex/runtime-optimization`。最後查核時 PR 仍為 OPEN，merge state 為 CLEAN；Python 3.11–3.13 contracts、inference 與 CommitCheck 均成功，沒有正式審查或 inline comments。本輪沒有對外 merge 或 push；新本機變更的 GitHub CI 尚未執行。

獨立檢查通過 74 個 NVFP4、release、publication 與 Jev-Omni focused tests。封存的 NVFP4／BF16 872 個 records 與 Jev-Omni 180 個 records 的 case identity、完整機率、argmax、accuracy 一致；Jev 證據索引的 18 個檔案 hash／size 與 NVFP4 replay 的 evaluation／conversion hash 綁定有效。這些歷史 populations 在本輪做本機封存重算；下文另列新 A100 public231／coherence 推論，不將兩組 populations 混用。

| 封存比較的 population | BF16，同 B200 | NVFP4，同 B200 | Jev-Omni，A100 |
| --- | ---: | ---: | ---: |
| 新的 BoolQ test，256 cases | 230/256 | 227/256 | 未在此 population 評測 |
| 歷史 BoolQ，128 cases | 118/128 | 114/128 | 112/128 |
| 歷史 native media，52 cases | 32/52 | 31/52 | 30/52 |

BF16／NVFP4 的 B200 latency 可作同環境比較；Jev-Omni 的 A100 latency 需保留不同硬體及 runtime 的限制。上述數字沒有使用 JevBench 的 coherence suite，也不是官方上榜成績。詳見 [NVFP4 證據](validation/2026-10-07/nvfp4/README.md) 與 [Jev-Omni 證據](validation/2026-10-07/jev-omni.md)。

## 評測軌道

### JevBench：機率一致性

[JevBench](https://github.com/JevBench/jevbench) 評測機率在變形問題之間是否一致。官方 repo 在本輪固定為 `e18733694623aa93058e279c3534f8b8e2edefa3`：50 個 relations 分屬 REP、BAT、MEA、LOG、CHO 五個 dimensions。單一 test 的機率容差為 0.05；relation pass rate 先在各 dimension 平均，再平均五個 dimensions，因此不能把所有 tests 的通過數直接當 overall score。

官方報告使用 `jevbench-mini`，含 1,200 個 tests、1,248 個預先凍結 requests；`jevbench-240` 含 12,000 個 tests。套件可接 `POST /v1/systemone` 或 in-process callable。每個問題都要保留其 id 與完整候選機率，不得從預期答案或 relation id 產生輸出。詳見 [官方協定與 scoring](https://github.com/JevBench/jevbench/blob/e18733694623aa93058e279c3534f8b8e2edefa3/README.md) 與 [官方重現方式](https://github.com/JevBench/jevbench/blob/e18733694623aa93058e279c3534f8b8e2edefa3/docs/reproducing.md)。

Coherence、correctness 與 calibration 分開呈現。Coherence 測量變形問題之間的機率一致性；它不表示 confidence 與真實正確率相符，calibration 另用同一 population 的 ECE／Brier 等指標。官方 README 的 uniform reference 雖然 coherence 為 89.4，accuracy 只有 39.8%；因此不得靠趨近均勻的機率取得較高 coherence 後宣稱模型品質提升。正式評估需同時保留原始答案、coverage、各 dimension、confidence intervals 與獨立 accuracy。

### Jev 公開 accuracy 評測與 sealed 評測

另一條軌道是 [BenchmarkHeaven／fstandhartinger 的 Jev 評測](https://github.com/fstandhartinger/jevbench) 的公開 accuracy 資料與官方 sealed 評測。本輪選定 legacy 公開資料的 immutable revision `bb05a335bc809e61b20c0f745d25499a82b326fc`，含 original 72、easy 48、hard 111，共 231 個可評分 cases；canonical cases SHA256 為 `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`。資料及 labels 公開，結果作研究 screening 使用。

公開 231-case 重播、JevBench coherence 與官方 sealed 結果必須分開歸檔、分開命名；公開資料的得分不能替代 sealed 得分，也不能推算官方排名。公開重播只使用該 revision 的 scoring，保留各軸與 case-level 結果，不建立自訂 composite 或排名。正式提交的程序與已驗證來源列在 [提交草稿](validation/2026-10-08/jevbench/submission-draft.md)。

2026-10-08 查核的榜單版本為 v1.7.18，量測協定為 v1.6.1，官方完整 population 為 P300＋S1200＝1,500 個 decisions。本輪公開 repo 的 legacy 231 cases 不等於完整 P300，更不包含 S1200；完整 P300 的資料取得方式尚未確認。

[官方 capability headline 方法](https://github.com/fstandhartinger/model-market-comparison/blob/a942fe0285da04ed227f77f2e38324959f0adc3f/docs/jevbench/METHOD-CAPABILITY-SCORE-HEADLINE-2026-10-01.md) 要求成本不超過 `0.06459465517241379 USD / 1,000 decisions`，且 adjusted p50 不超過 `1.2329566404223442 seconds`。這是官方 adjusted serving 測量與 headline 條件；本輪 direct-call A100 latency 或 Modal 按時計費不能代入宣稱符合。官方完整測量、可稽核的 adjusted cost／latency 與 sealed 結果完成前，capability eligibility、官方分數與排名均保留未知。

[官方提交入口](https://benchmarkheaven.com/submit) 需要 contact email。這輪僅準備供維護者重現的 [草稿](validation/2026-10-08/jevbench/submission-draft.md)；尚未送出。

## 市場模型與資料暴露紀錄

這是可持續擴充的比較 roster。列入表示已找到公開 publisher，不表示獲官方推薦或已在本專案完成評測；模型版本、資料暴露與 runtime 必須一起保留。

| 模型與 primary source | 本輪配置／狀態 | 公平比較的限制與資料暴露 |
| --- | --- | --- |
| [本專案 BF16](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362) | 本輪 A100 baseline；causal multislot、SDPA、原有 temperature | 本輪不使用 public cases fitting、checkpoint selection 或 temperature tuning；base pretraining overlap 未知。本次重播後將其視為已曝光的開發資料。 |
| [OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B/tree/c88e18653ceb7a8770716287f55fdefc79d6b588)／[publisher code](https://github.com/OmniJev/OneJev/tree/7e3007d7829c7f59f5af15c60fb8454a11e37170) | 已完成：native MMDecisionEngine、publisher default temperature 1.0、CUDA graphs、debias=1 | 該 revision 沒有提供 calibration sidecar，使用 publisher 的 `Calibration()` default；保留 prompt／auto-fork／FP32 head。不聲稱 publisher 未看過 public benchmark。缺少 FLA／causal-conv 的 PyTorch fallback 需保留，這不是 publisher 完整最佳化 serving speed。 |
| [Decider-2B](https://huggingface.co/Mapika/decider-2b/tree/533964dae8be954c5b5e19fa4948e48408094c1e)／[publisher code](https://github.com/Mapika/decider/tree/e50e549b47e2da69223734fee4efa1ddd4528e93) | 已完成：native Decider eager reference、independent=True、shipped by-type temperatures／isolated-level policy | 與舊 campaign 的 Decider-4B 分成不同 row；FLA 0.4.2 已安裝，缺少 causal-conv 的 reference fallback 需保留。該 revision 的 `decider_config.json` 描述 public decision mixture replay 與在 regression／own-validation rows 上 fitting；不把公開測試當作從未曝光的 held-out。 |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni/tree/5addda86ddee081a68fb067477ea100c221b8917) | PR #6 歷史 128 BoolQ＋52 native media；本輪 run03 public231／coherence 已完成 | 新測量與舊 population 分開；BF16 backbone、FP32 head／softmax、publisher independent prompt、原生 processor，沒有 graph／prefix cache 或本輪 fitting；publisher 訓練暴露未知。 |
| [Torchcast Decision 12B](https://huggingface.co/torchcast-ai/torchcast-decision-12b) | 已列 roster；本輪未執行 | Publisher 公開聲明 public benchmark 曾用於 model selection，並移除 14 個 overlap cases；不能把其公開得分稱為未見過的 held-out。須另外固定 revision、source 與 submitted profile。 |
| [Cygnet recipe](https://github.com/blockbrain-ai/cygnet-recipe) | 已列 roster；本輪未執行 | 先確認實際可下載 checkpoint、recipe revision、inference policy 與 public benchmark 在 recipe 的用途；現有資料不足以填未曝光聲明。 |
| [Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) | 已列 roster；本輪未執行 | 保留 publisher prompt v2、10-option 上限及 state truncation 政策。超限、截斷與不同 context policy 需明列，不用較小成功子集代表完整 population。 |

本輪可重現配置見 [Modal campaign](../apps/modal/jevbench_campaign.py)。OneJev、Decider、Jev-Omni 的完整 code／weight pins 在該檔固定；Torchcast、Cygnet、Quyet 尚未進入執行清單，因此不填推論分數、成本或排名。

```bash
uv run --no-sync modal run apps/modal/jevbench_campaign.py \
  --run your-unique-run \
  --models s1-bf16,onejev-4b,decider-2b,jev-omni
```

上述 Modal CLI／in-process 路徑已有 S1、OneJev、Decider、Jev-Omni 四個不同 checkpoint 的完整 231-case 與 coherence 執行證據；另有 S1 independent development 與 compiled bounded profiles。命令使用各模型原生 runtime，在 A100-80GB 依序執行。每個 run id 必須唯一；已執行的 run 不覆寫。它不執行官方 sealed 評測，也不自動提交結果。此輪沒有 GPU HTTP server smoke，不將 callable 結果稱為 deployed endpoint 驗證。

逐題 profile 與 compiled profile 需要明確選擇；`s1-compiled` 的 bounded stage 不執行 coherence。只對指定模型執行 coherence 時可使用以下 CLI（各 stage 完成狀態見文末）：

```bash
uv run --no-sync modal run apps/modal/jevbench_campaign.py \
  --run your-unique-development-run --models s1-independent,s1-compiled \
  --coherence-models s1-independent

uv run --no-sync modal run apps/modal/jevbench_campaign.py \
  --run your-unique-followup-run --models jev-omni,s1-independent \
  --coherence-models jev-omni
```

共用 image 依賴固定為 `torch==2.10.0`、`torchvision==0.25.0`、`transformers==5.17.0`、`accelerate==1.15.0`、`huggingface-hub==1.33.0`、`safetensors==0.8.0`、`pillow==12.3.0`、`soundfile==0.14.0`、`librosa==0.11.0`、`sentencepiece==0.2.2`、`protobuf==6.33.6`、`numpy==2.5.3`、`pydantic==2.13.5`、`httpx==0.28.1`；Decider image 另外固定 `flash-linear-attention==0.4.2`。Python image 目標為 3.12，已完成 GPU receipts／coherence reports 記錄 Python 3.12.10；重新建立 image 後以新 receipt 為準，不把本機 Python／PyTorch 版本套用到 GPU 結果。

### 使用逐題 opt-in

本 repo 自 commit `7a91578` 起，`decide`、`benchmark`、`serve` 可明確傳 `--question-mode independent`；預設仍為 `causal_multislot`。本輪新 wheel 來自 source commit `1b581e90d4458cb658e565bc608990e8dee6174b`：`gemma_system_one-0.1.0-py3-none-any.whl`，195,197 bytes，SHA256 `8c16a6536480d6632d5f5478989ba1920d6ec2568eafe91647df2cb8591b38a1`。固定 HF model revision 原先附帶的 runtime 包沒有此選項，必須使用這份 repo code 或新 wheel。以下範例需要含該 wrapper、使用上列固定依賴的 CUDA 環境：

```bash
uv run --no-sync s1 decide examples/request.json \
  --backend gemma --model knowlet/Gemma-4-12B-Unified-System-One \
  --revision a66f836b56605039fe040f330180e336d19b3362 \
  --device cuda --precision bfloat16 --attn-implementation sdpa \
  --question-mode independent
```

CLI／API wrapper 路由重用實際 GPU campaign 已執行的 G4 independent helper；helper source 未變更。Wrapper 的 CLI／CPU ASGI route 檢查通過 283 tests；最終 full suite 為 1,240 passed、3 skipped、1 Starlette deprecation warning，Ruff／format／diff check 通過。這證明新 flag 路由到同一 helper，沒有將舊 GPU callable report 說成新 wrapper 的 GPU HTTP server smoke。

Public-01／public-02／public-03 的 executed source archive 保留了原始 callable campaign；其中的舊 CLI 不含這個新 flag。精確重現封存 GPU run 時使用該 run 的 source bytes；使用新 CLI 時使用上述 current wrapper source／wheel，並記錄新的 runtime identity，不把兩者混成同一 source revision。

## 執行與公信力要求

1. **先固定 identity。** 每一模型記錄公開模型 revision、publisher code commit、processor、calibration sidecar、precision、依賴版本、source hashes、硬體與 request manifest；保留未執行、unsupported、拒絕與錯誤資料。
2. **固定並披露 inference semantics。** 保留本專案原始 causal multi-question 的 immutable baseline；看到公開 BAT 結果後，可以開發另外命名的 `s1-independent` profile，但需明列它是觀察失敗後選擇的語義變更、重新評測並保留獨立 receipt，不能回填或替換原始成績，也不再聲稱它是 untouched held-out。Wire adapter 只做 schema mapping，不做機率投影、問句分類、relation 特判或後處理修補。
3. **先驗證協定再跑整套。** `jevbench check` 通過後，執行固定 mini suite；原始 requests／answers 與 suite、manifest hash 一起封存。缺失或失敗的 relations 保留未評分狀態，不能以較小成功子集代表整套分數。
4. **公開資料作開發資料。** 完整披露哪些 public benchmark 曾被看過、用於調參或在 publisher source 中被引用。這輪完成後不再把公開 231 cases 稱為未見過的 held-out。新的 temperature、training、prompt 選擇使用另外的開發／calibration population，正式 sealed 資料由維護者管理。
5. **品質與效能分開比較。** 不同硬體的品質可用 matched requests 與 paired intervals 比較；速度比較需要同硬體、同 precision policy、同上下文、同 batch／concurrency、冷啟動與暖推論分開。記錄成功率、SLO misses、所有 GPU stages 的時間與成本；tensor footprint 不等於最低 VRAM。
6. **讓維護者可重現。** 準備 immutable model package、安裝方式、無 gold label 的 server／callable adapter、完整 report 與 raw answers，以及資料暴露聲明。正式排名及正式收錄需由維護者確認。

## 優化次序

原始 causal multislot baseline 已完成公共 accuracy 與 coherence，結果顯示優先問題是 batch independence、probability measure 與 choice-set coherence。後續另命名的 `s1-independent` 已完成 public231／coherence，明列它是公開 BAT 失敗後選擇的新語義、與原始 baseline 分開比較；原本 causal 預設值維持不變，沒有事後改寫這次 baseline 分數。

訓練範圍也需從目前 BoolQ specialist 擴展到更廣的 Choice／Score、互補／集合／邊際機率問題；使用另外生成的訓練／calibration 資料，保留原始模型與固定 public baseline。先驗證品質改進再討論正式 sealed 評測，不以趨近 uniform 或 relation 特判換取 coherence。

不可變 candidate head rows 的 opt-in cache 已實作並完成這輪原生 A100 實驗；完整公開 231 cases 的機率差為 0，全部 label 相同，但速度尚未建立收益。Compilation 的 bounded 實驗已測出機率漂移，後續先定位原因；token-position scatter 仍是待測候選。每一候選先做 bounded 實驗，再重驗所有適用 population、mixed media、batch 與上下文邊界；測得的品質或資源收益足以支持時才成為預設值。已有 GGUF output reservation 的記憶體改善與 NVFP4 壓縮證據保留各自的硬體與測量範圍；不把記憶體節省推論成 latency 改善。

## 2026-10-08 原生 A100 baseline

`20261008-public-01/s1-bf16` 已完成。受測 weights 為 `a66f836b56605039fe040f330180e336d19b3362`，使用 NVIDIA A100-SXM4-80GB、BF16、PyTorch 2.10.0、Transformers 5.17.0、SDPA、causal multislot、52 candidate rows，temperature 為已發布的 `2.7830344470383452`。本輪不做 calibration fitting。公開 231 cases 全部 attempted、valid、scorable，coverage 與 strict schema validity 均為 100%，沒有 renormalization。

| 測量 | 結果 | 範圍 |
| --- | ---: | --- |
| Legacy public231 accuracy | 197/231＝85.2814% | 公開開發 screening；不是 P300／S1200 |
| ECE | 0.04849537 | 同一公開 231-case population |
| Request latency p50／p95 | 88.8564／498.6602 ms | 同步原生 callable；不含 model setup；不是 adjusted serving latency |
| JevBench-mini strict coherence | 40.5049% | 原始 probability tolerance 0.05 的 overall |
| Coherence 95% bootstrap interval | 37.7338–43.1087% | 官方 report 1,000 次 bootstrap |
| Coherence coverage | 50/50 checks，1,200/1,200 tests | 各 check coverage＝1；沒有 errors |
| REP／BAT／MEA／LOG／CHO | 58.3333／38.3333／15.4412／69.5833／20.8333% | 各 dimension 的 strict pass rate |

`graded.overall` 的 88.8156% 是另一種連續診斷指標，不能當作 strict coherence 的 40.5049%、不能替代榜單 headline。這次完整 baseline 顯示 correctness 與機率一致性仍有明顯落差，沒有宣稱上榜排名。

Candidate cache 的 matched 231-case 重播仍為 197/231；所有候選機率的最大差為 0，全部 labels 一致。八個固定位置跨三個公開檔案的暖推論 ABBA 實驗各記錄 48 次 baseline／cached samples；其 pooled medians 為 baseline 112.8609 ms、cached 115.0600 ms。以 task 為配對單位的 cached／baseline latency ratio 為 1.005158，95% CI 為 0.998159–1.098600；區間跨越 1，沒有可接受的速度收益。Cache 維持 opt-in，不以整套前後兩次 latency 當作控制實驗。

本機原始證據位於 `artifacts/jevbench/20261008-public-01/s1-bf16/`，包含 receipt、完整 coherence report、public／cached manifests、records／raw answers 與 candidate-cache samples。第一個實際執行版本另外保存在 `artifacts/jevbench/executed-public01/`；後續修正的 campaign 不替代這組原始 source bytes：

| 實際執行 source | SHA256 |
| --- | --- |
| `jevbench_campaign.py` | `627e73c5a1bd549838af33a78dc1ebd7881b2752de0a635c04e6d766e38550d8` |
| `run_jevbench_public.py` | `f303209795ac932e251c786642c434c2a561cfb1299e4065e96ff0247f436701` |
| `unified.py` | `42ba5f076af164e0e54690a735e25a272bdd678279901946c2e5accefa385a6c` |

Run01 的舊 receipt 只記錄 runner hash；其餘 59 個 campaign／S1 package hashes 沒有 execution receipt attestation。上述 source bytes 已保存並對照另存 snapshot，但此輪來源證明仍為 partial，不能把封存 bytes 等同完整 execution proof；public／coherence 原始答案的完整重算狀態另列。

## 四個市場模型與開發 profiles 的完整比較

以下 primary comparison 使用相同的公開 cases 與 pinned coherence suite，但保留每個 publisher 的原生 inference／temperature／kernel policy。Causal baseline、OneJev、Decider、compiled 使用 NVIDIA A100-SXM4-80GB；Jev-Omni 與 primary `s1-independent` 使用 NVIDIA A100 80GB PCIe，屬同 A100 系列但 board type 不同。P50／p95 是同步原生 callable 測量，包含各模型在這組 requests 的原生執行與 fallback；不同 board type、run 與 publisher policy 的 latency 不構成配對 speed comparison。

| Profile／實跑 | Board type | Public231 accuracy | ECE | Strict coherence（95% CI） | BAT | Callable p50／p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `s1-bf16`／public-01 | A100 SXM4 | 197/231＝85.2814% | 0.04849537 | 40.5049%（37.7338–43.1087%） | 38.3333% | 88.8564／498.6602 |
| `onejev-4b`／public-02 | A100 SXM4 | 171/231＝74.0260% | 0.08578237 | 72.5310%（69.9166–74.7447%） | 100% | 46.4446／1,187.7305 |
| `decider-2b`／public-02 | A100 SXM4 | 175/231＝75.7576% | 0.06591602 | 72.8873%（70.4568–75.1625%） | 100% | 55.7398／92.7399 |
| `jev-omni`／public-03 | A100 PCIe | 203/231＝87.8788% | 0.04966260 | 74.5474%（72.2231–76.9987%） | 100% | 85.5362／512.5903 |
| `s1-independent`／public-02 | A100 PCIe | 197/231＝85.2814% | 0.04849537 | 71.9592%（69.2672–74.3349%） | 96.6667% | 85.0734／528.5743 |
| `s1-compiled`／public-02 | A100 SXM4 | 197/231＝85.2814% | 0.05435744 | `not_run` | `not_run` | 50.7335／399.1843 |

六個 primary profiles 的 public231 都有 231 valid cases、full coverage、strict schema validity 100%、零 renormalization；完整 records、request hashes、raw answer hashes 與 pinned 官方 summary 已重算通過。五個 coherence profiles 各有完整 50 checks／1,200 tests、240 suite cases、1,248 native cache entries，沒有 errors；原始 answers 以 pinned 官方 scorer 重算，deterministic outcomes／gaps／scores 與 GPU report 一致。Compiled 的 coherence 是 `expected_not_run`。詳見 [完整 campaign audit](validation/2026-10-08/jevbench/campaign-audit.md) 與 [machine-readable audit](validation/2026-10-08/jevbench/campaign-audit.json)。重算保留 GPU Python 3.12.10／PyTorch 2.10.0 provenance，不以本機重算環境冒充原始 inference 環境。

OneJev 使用 publisher default temperature 1.0（revision 無 calibration sidecar），auto-fork、CUDA graphs、FP32 head 與 debias=1；執行時 FLA／causal-conv 缺少而走 PyTorch fallback。Decider 使用 shipped temperatures：Choice 1.164、Noul 1.624、Score 1.124，`isolated_levels=true`；FLA 0.4.2 已安裝，但 causal-conv 缺少而使用 reference fallback。這些差異保留在證據與比較中，不據此宣稱官方最佳化服務速度。

Jev-Omni 在本組公開資料觀察到最高 accuracy 點估計與 strict coherence；其 REP／BAT／MEA／LOG／CHO 為 81.3889／100／49.2647／75.4167／66.6667%。相對 S1 baseline 或 independent，配對 accuracy 差為 +2.5974 percentage points，95% source-group bootstrap CI 為 −1.2931 至 +6.7228 percentage points，跨越零，不能宣稱確定優於 S1。完整 231 paired cases／195 source groups 的描述性比較見 audit；causal S1 與 Jev-Omni 還存在 board type 差異。S1 的 accuracy 點估計高於 OneJev／Decider，而原始 causal coherence 較弱；accuracy、coherence 與 ECE 不合成新排名。各模型的訓練／選擇暴露不同，這些公開分數不能證明 sealed 排名。

`s1-independent` 重用既有 G4 helper，以每題獨立 prompt 與實際 right-padding mask 推論，保留同一 weights、temperature 與 BF16／SDPA；實際執行 `unified.py` SHA256 仍是 `42ba5f076af164e0e54690a735e25a272bdd678279901946c2e5accefa385a6c`，profile 選擇記在獨立 receipt。REP／BAT／MEA／LOG／CHO 為 73.0556／96.6667／37.9902／85.4167／66.6667%。相較原始 causal baseline 的公開 coherence 改善是一次觀察失敗後的 development 結果，不能稱為新的未曝光 held-out；PCIe 與 SXM4 board 不同，也沒有宣稱速度提升。產品 CLI 已提供明確 opt-in，但未執行 GPU HTTP server smoke，沒有宣稱服務已部署。

Public-03 的 independent supplemental stage 已完成：A100-SXM4-80GB 上 public231 仍為 197/231、ECE 0.04849537，callable p50／p95 59.0737／469.0054 ms，這次未重跑 coherence。與 public-02 independent 的 231 raw requests／responses 逐一相同，849 個候選機率最大差為 0。八個 native cases 包含文字、影像、音訊、混合媒體、長 state 與 16-question batch；29 個 argmax labels 與 sequential reference 全部相同，59 個機率的最大差為 `0.0063006282`。按模態最大差為文字 0.0063006282、影像 0.0020535067、音訊 0.0019716620、混合 0.0000000149。這是有界的逐題／batch 語義 regression，不能稱為 native 全部機率 exact，也不能以不同 run 的 latency 宣稱配對速度收益。

## Compiled bounded 實驗

`20261008-public-02/s1-compiled` 使用同一 BF16 weights、A100-SXM4-80GB 與 `decoder-default-dynamic`（`dynamic=True`、`fullgraph=False`）。Public231 為 197/231，看似與 eager 總 accuracy 相同，但四個 case labels 改變：兩個由正確變錯誤、兩個由錯誤變正確。192 個候選 distributions 改變，最大機率差為 `0.507617265`；不能以總 accuracy 相同宣稱 output parity。

最大差出現在 `easy-intent-00`：預測仍為 `track_order`，其機率由 `0.4346627` 變成 `0.9422800`。Native 八個 cases 的 29 個 labels 全部保留，但最大機率差仍為 `0.008602023`。這組 native cases 是有界的 regression 證據，不代表完整 media population 的 calibration 或 parity。

首次 public request 用時 `363.864655 seconds`，包含 cold compilation；該 public run 的 p50／p95 為 50.7335／399.1843 ms。與 baseline 的 88.8564 ms p50 是未配對的不同 run，沒有接受為速度收益，也沒有藏掉 cold compilation 成本。Compiled profile 沒有執行 coherence；維持 experimental、預設關閉。接下來若研究此路徑，先定位機率漂移並重驗 calibration／品質，再做配對暖推論與冷啟動比較。

Public-02 四個、public-03 兩個 model cells 各自的 60 個 source hashes 已對照 frozen executed archive 驗證；65 個 frozen package 檔案也與 Git `501c8a466fa6e694b184283d0574526a500b1655` bytes 相同。各 profile receipt 保留實際 source identity；run01 的 receipt source proof 仍為 partial，後續新增 CLI flag 不回填舊 GPU provenance。

OneJev／Decider／independent／compiled 證據位於 `artifacts/jevbench/20261008-public-02/`，Jev-Omni 與 independent supplemental 在 `artifacts/jevbench/20261008-public-03/`。Public-03 app `ap-c5QrLXfwBUz7jwpf3VapVq` 已 exit 0，完整 Modal log 已回收。使用各 run 的 executed source archive 與 receipts；新 CLI source／wheel 另外放在 `runtime-current`，不替代已執行 GPU source identity。

完整 [證據索引](validation/2026-10-08/jevbench/README.md)、[raw-evidence.zip](validation/2026-10-08/jevbench/raw-evidence.zip) 與 [manifest](validation/2026-10-08/jevbench/raw-evidence.manifest.json) 已封存並逐檔驗證。ZIP 含 8,219 members、1,848 public raw files、6,240 coherence cache files，以及 logs、source、audits 與 current wrapper wheel；所有 members 的 SHA256／size／CRC 通過重讀核對。

| 封存檔 | 大小 | SHA256 |
| --- | ---: | --- |
| `raw-evidence.zip` | 10,932,994 bytes；未壓縮 28,970,101 bytes | `b85b4471d655a05e5fc682d2d22c157ae86af9b42a05a926e2bd78655223f53b` |
| `raw-evidence.manifest.json` | 2,127,644 bytes | `5cea21069490c69b59f8b9f12fd76062aeb2ed3b395a0ba3fe6a86d5d8d7b44d` |

## 結果狀態

| 項目 | 本輪狀態 | 解讀 |
| --- | --- | --- |
| PR #6 review 與封存 evidence audit | 完成 | 74 focused tests 通過；CI required checks 成功 |
| Jev 公開 231-case accuracy 重播 | 四個 checkpoint＋independent／compiled profiles 全部完成 | 六個 primary profiles 各 231 valid；raw／hash／official summary audit 通過，不能推算 sealed 排名 |
| JevBench-mini coherence | 五個 profiles 完成：40.5049／72.5310／72.8873／71.9592／74.5474% | strict overall；各自全 50 checks／1,200 tests、1,248 raw cache 官方重播通過 |
| `s1-independent` development profile | public-02 完成 public231／coherence | 公開 causal BAT 結果後選擇；新語義、A100 PCIe 與原始 baseline 分列 |
| `s1-compiled` bounded stage | public-02 完成 native 八個 cases＋public231 | 四個 labels／192 distributions 改變；cold compilation 363.864655 s；experimental off，沒有 coherence 或已接受速度收益 |
| `jev-omni` 新 suite | public-03 完成 public231／coherence 與 raw audit | 203/231；strict coherence 74.5474%；與 PR #6 舊 population 分開 |
| independent native regression | public-03 完成 native 八個 cases＋public231 | 29 labels 保留、max drift 0.006300628；該 stage 不再跑 coherence |
| Final evidence archive | 完成，逐檔驗證通過 | 8,219 members；run02／03 source hashes 完整匹配；run01 receipt source proof partial 已披露 |
| 官方 sealed 評測 | `not_run` | 沒有官方分數或排名 |
| 正式提交／榜單收錄 | 未提交 | [草稿](validation/2026-10-08/jevbench/submission-draft.md) 供審閱及重現準備 |
