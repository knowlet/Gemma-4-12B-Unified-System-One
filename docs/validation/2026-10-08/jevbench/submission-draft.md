# Jev 評測提交草稿

更新日期：2026-10-08。狀態：**草稿，未寄送、未提交、未獲正式收錄**。下面的待填欄位需要實跑 receipts 或維護者確認；未填欄位不表示零分。[BenchmarkHeaven 提交表單](https://benchmarkheaven.com/submit) 需要 contact email，這份草稿不猜測使用者的 email。

## 擬供維護者審閱的摘要

我們希望評估 `knowlet/Gemma-4-12B-Unified-System-One` 的 **BF16 independent profile** 是否適合納入你們的 typed decision 評測。模型將文字、原生影像及音訊轉成 Noul、Choice、Score 的完整候選機率，推論不使用生成式答案解析。Independent 是觀察公開 causal BAT 失敗後選擇的 development 語義，原本 causal 預設與 baseline 保留；它不是未接觸 public benchmark 的候選。我們會提供固定 checkpoint revision、可重現的 S1 runtime 與標準 adapter。

我們將公開資料上的 accuracy、calibration、JevBench probability coherence 與官方 sealed 評測分開報告。Coherence 是變形問題之間的機率一致性，不能當作 calibration；confidence／correctness 的差異另列 ECE。任何公開重播結果都不聲稱是官方 sealed 得分或正式排名；資料暴露、已使用的開發資料及 adapter 與原始 runtime 的差異會隨提交揭露。

## 候選模型與 identity

| 欄位 | 值 |
| --- | --- |
| Model repository | `knowlet/Gemma-4-12B-Unified-System-One` |
| 擬提交 candidate profile | `s1-independent`：BF16／SDPA、每題獨立 prompt、實際 right-padding mask、未開啟 compile；public development exposure 已披露 |
| Candidate weights revision | `a66f836b56605039fe040f330180e336d19b3362` |
| Candidate public／coherence 結果 | 197/231＝85.2814%；strict coherence 71.9592%（95% CI 69.2672–74.3349%），BAT 96.6667%；A100 80GB PCIe |
| 已完成 baseline model revision | `a66f836b56605039fe040f330180e336d19b3362` |
| 已完成 baseline inference profile | `s1-bf16`：原始 causal multislot、SDPA、52 candidate rows；不是獨立逐題 profile |
| 已完成 GPU callable source identity | 實際執行 source bytes／SHA256 見 [本輪計畫](../../../jevbench-plan.md)；與新 CLI wrapper 發布包分開封存 |
| 新 opt-in wrapper source／wheel | Source `1b581e90d4458cb658e565bc608990e8dee6174b`；`gemma_system_one-0.1.0-py3-none-any.whl`，195,197 bytes；SHA256 `8c16a6536480d6632d5f5478989ba1920d6ec2568eafe91647df2cb8591b38a1` |
| Temperature sidecar | `s1_config.json` SHA256 `322ffa1c6bfa7c4069da43265be1e79071eee6fbcff07cd4a74f9ed04774b962`；temperature `2.7830344470383452`；processor inventory／hashes 保留於各 receipt |
| Candidate precision／quantization／hardware | BF16／未量化；primary public／coherence 使用 NVIDIA A100 80GB PCIe；supplemental public／native 使用 A100-SXM4-80GB，分開呈現 |
| Context／max questions／max options | 已執行 S1 callable：16,384 expanded tokens／64 questions／52 options；context 超限拒絕，不截斷 |
| Dependencies／kernel | 共用固定依賴見 [本輪計畫](../../../jevbench-plan.md)；GPU Python 3.12.10、PyTorch 2.10.0、Transformers 5.17.0；BF16 使用 SDPA，沒有自訂 NVFP4 kernel |
| 啟動方式 | Modal CLI／in-process callable 已執行；current checkout 的 `--question-mode independent` wrapper 已通過 CLI／CPU ASGI route 檢查；GPU HTTP server smoke 未執行 |
| Public quality report | 已完成 legacy 231 cases：197/231＝85.2814%；原始本機 evidence 路徑與 scope 見下文 |
| Candidate JevBench-mini report | 已完成 strict coherence 71.9592%，95% CI 69.2672–74.3349%；完整 50 checks／1,200 tests、1,248 cache entries，無 errors；原始 causal baseline 40.5049% 另列 |
| Official sealed score／rank | 未評測／未知 |
| Capability headline cost／latency eligibility | 未完成官方 adjusted serving 量測／未知 |
| Contact email | 待使用者指定；未提交 |
| Formal submission source／wheel／evidence URLs | 新 source／wheel identity 已固定，本機 evidence 完整封存並稽核；對外可下載 URLs 尚待發布。舊 callable source archive 不含新 CLI flag，不能當作新 wrapper 發布包 |

已有公開 BF16、MLX、GGUF 與 NVFP4 derivatives；它們的 calibration 與 runtime 不互相替代。提交前只選定一個明確模型／runtime identity，或分別提交已獨立驗證的 derivatives。NVFP4 使用自訂 `s1-transformers-nvfp4-v1` 與 pinned Blackwell kernel；僅 B200 已有執行證據，stock Transformers、vLLM、TensorRT-LLM 相容性未建立。

本輪已固定 BF16 weights revision `a66f836b56605039fe040f330180e336d19b3362`。`20261008-public-01` 已完成原始 BF16 baseline；`20261008-public-02` 已完成 OneJev-4B、Decider-2B 與逐題 development profile 的 public231／coherence，以及 compiled bounded stage；`20261008-public-03` 已完成第四個 checkpoint Jev-Omni 的 public231／coherence 與 independent native 八個 cases＋public231。三個 run 的 evidence 與完整 logs 已回收、ZIP／manifest 逐檔驗證通過，run03 app exit 0；run01 的 receipt source proof 仍為 partial，與完整 raw-answer audit 分開披露。

公開 source 使用 BenchmarkHeaven／fstandhartinger legacy revision `bb05a335bc809e61b20c0f745d25499a82b326fc` 的 231 cases。這與最新看板 v1.7.18／量測協定 v1.6.1 所述的 P300＋S1200＝1,500 decisions 不同；沒有據此計算正式分數或排名。

依 [官方 capability headline 方法](https://github.com/fstandhartinger/model-market-comparison/blob/a942fe0285da04ed227f77f2e38324959f0adc3f/docs/jevbench/METHOD-CAPABILITY-SCORE-HEADLINE-2026-10-01.md)，成本需不超過 `0.06459465517241379 USD / 1,000 decisions`、adjusted p50 需不超過 `1.2329566404223442 seconds`。我們不以 direct-call A100 latency 或 Modal 按時計費代入這些 adjusted serving 條件。

JevBench coherence 使用獨立 repo revision `e18733694623aa93058e279c3534f8b8e2edefa3` 的 mini suite，與上述 accuracy／sealed 看板分開報告。它是獨立評測專案；此草稿不聲稱 TypeSafe AI endorsement。

已完成的 BF16 baseline 公開 accuracy 為 85.2814%，但 strict coherence 只有 40.5049%，其中 BAT 38.3333%、MEA 15.4412%、CHO 20.8333%。我們保留原始結果並明列限制；這次逐題 development profile 已另外命名及重新評測，後續擴展 Choice／Score 訓練也需新 identity，不替換 causal baseline。Report 裡的 graded overall 88.8156% 不當作 strict coherence。

Baseline ECE 為 0.04849537；同步原生 callable latency p50／p95 為 88.8564／498.6602 ms，不含 setup，不是官方 adjusted serving 測量，成本未知。Opt-in candidate cache 在全部 231 cases 保持機率差為 0、labels 全相同；八個固定位置的暖推論 paired cached／baseline ratio 為 1.005158，95% CI 0.998159–1.098600，沒有可接受的速度收益，維持 opt-in。

四個不同 checkpoint 與兩個另外命名的 S1 profiles 比較如下。全部是 legacy public231 screening；coherence 是獨立 suite。SXM4 與 PCIe、不同 publisher runtime／run 的 latency 只作描述，不作配對 serving speed 排名。

| Primary profile | A100 board | Accuracy | ECE | Strict coherence（95% CI） | Callable p50／p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `s1-bf16`／public-01 | SXM4 | 197/231＝85.2814% | 0.04849537 | 40.5049%（37.7338–43.1087%） | 88.8564／498.6602 |
| `onejev-4b`／public-02 | SXM4 | 171/231＝74.0260% | 0.08578237 | 72.5310%（69.9166–74.7447%） | 46.4446／1,187.7305 |
| `decider-2b`／public-02 | SXM4 | 175/231＝75.7576% | 0.06591602 | 72.8873%（70.4568–75.1625%） | 55.7398／92.7399 |
| `jev-omni`／public-03 | PCIe | 203/231＝87.8788% | 0.04966260 | 74.5474%（72.2231–76.9987%） | 85.5362／512.5903 |
| `s1-independent`／public-02 | PCIe | 197/231＝85.2814% | 0.04849537 | 71.9592%（69.2672–74.3349%） | 85.0734／528.5743 |
| `s1-compiled`／public-02 | SXM4 | 197/231＝85.2814% | 0.05435744 | `not_run` | 50.7335／399.1843 |

OneJev 使用 publisher default temperature 1.0（revision 沒有 sidecar）、auto-fork、CUDA graphs 與 FP32 head；缺少 FLA／causal-conv 的 PyTorch fallback 保留在測量 scope 中。Decider 使用 shipped by-type temperatures／isolated-level policy；FLA 0.4.2 已安裝，causal-conv 缺少而走 reference fallback。Jev-Omni 使用 BF16 backbone、FP32 head／softmax、publisher independent prompt，沒有 graph／prefix cache 或本輪 fitting。完整 pins 與 publisher 資料暴露紀錄見 [本輪計畫](../../../jevbench-plan.md)。

Jev-Omni 的 strict REP／BAT／MEA／LOG／CHO 為 81.3889／100／49.2647／75.4167／66.6667%。它相對 S1 baseline 或 independent 的 paired accuracy 差為 +2.5974 percentage points，95% source-group bootstrap CI 為 −1.2931 至 +6.7228 percentage points，跨越零；不宣稱確定優於 S1。全 231 paired cases／195 source groups 的描述性品質比較不等於正式排名。

`s1-independent` 已完成 197/231＝85.2814%，ECE 0.04849537；strict coherence 71.9592%（95% CI 69.2672–74.3349%），REP／BAT／MEA／LOG／CHO 為 73.0556／96.6667／37.9902／85.4167／66.6667%。它使用 **NVIDIA A100 80GB PCIe**，而 causal baseline、OneJev、Decider 使用 A100-SXM4-80GB；其 callable p50／p95 為 85.0734／528.5743 ms，不能因不同 board 的 latency 點估計宣稱速度提升。

六個 primary public profiles 皆 231 valid、full coverage、strict schema validity 100%、零 renormalization，完整 records／request／raw answer hashes 與 pinned 官方 summary 重算通過。五個 coherence profiles 各全 50 checks／1,200 tests、240 suite cases、1,248 native cache entries、無 errors；以 pinned 官方 scorer 重算的 deterministic outcomes／gaps／scores 一致。Compiled coherence 是 `expected_not_run`。詳見 [完整 campaign audit](campaign-audit.md) 與 [machine-readable audit](campaign-audit.json)。原始 GPU Python 3.12.10／PyTorch 2.10.0 provenance 保留，不以本機重算環境覆蓋。

Independent 的 public-03 supplemental 使用 A100-SXM4-80GB，仍為 197/231、ECE 0.04849537，callable p50／p95 59.0737／469.0054 ms，未重跑 coherence。與 public-02 的 231 raw requests／responses 與 849 個機率全部相同，最大差為 0。Native 八個 cases 覆蓋文字、影像、音訊、混合媒體、長 state 與 16-question batch；29 argmax labels 與 sequential reference 全保留，59 個機率最大差 `0.0063006282`。文字／影像／音訊／混合的最大差分別為 0.0063006282／0.0020535067／0.0019716620／0.0000000149；native 不是完整機率 exact，也不據此宣稱配對速度收益。

我們允許根據已公開的 baseline 失敗做後續開發，例如另命名 `s1-independent`；明列它是觀察 BAT 失敗後選擇的新語義，保持原始 immutable baseline 與各 profile 的獨立 receipts，不回填原始成績。所有這輪已接觸的 public cases 都視為 development exposure，改進後不聲稱是 untouched held-out。

Independent 重用既有 G4 helper 的逐題 prompt 與 right-padding mask，保留原始 weights、temperature 及 `unified.py` source bytes；沒有將 causal 預設值改成逐題。實際 GPU callable core SHA256 為 `42ba5f076af164e0e54690a735e25a272bdd678279901946c2e5accefa385a6c`。這份草稿以 independent 作擬提交候選；正式提交前仍須公開固定 runtime source／wheel／evidence URLs，並取得 contact email 與 endpoint 要求。

本 repo 自 `7a91578` 起為 `decide`、`benchmark`、`serve` 提供 `--question-mode independent`，路由至相同、未改變的 G4 helper；上表的新 wheel 固定在 source `1b581e90`，本機 member hashes 已核對。HF 模型 pin 原附 runtime 包沒有此選項，需使用這份 repo code 或新 wheel。CLI／CPU ASGI route 檢查為 283 passed；最終 full suite 為 1,240 passed、3 skipped、1 Starlette deprecation warning，Ruff／format／diff check 通過。新本機變更的 GitHub CI 尚未執行。這不是新的 GPU HTTP server smoke。使用新 wrapper 的例子（需要該 source／wheel、固定依賴的 CUDA 環境）：

```bash
uv run --no-sync s1 decide examples/request.json \
  --backend gemma --model knowlet/Gemma-4-12B-Unified-System-One \
  --revision a66f836b56605039fe040f330180e336d19b3362 \
  --device cuda --precision bfloat16 --attn-implementation sdpa \
  --question-mode independent
```

Public-02／public-03 的 frozen source archive 重現舊 callable；不包含新 CLI flag。正式提交包會分別說明已執行 GPU source 與新 wrapper 的 provenance，不把新 source identity 套用回舊 GPU report。

Compiled profile 不作本草稿的提交候選：雖然也是 197/231，但四個 labels 改變、192 distributions 改變、最大機率漂移 `0.507617265`；首次 request cold compilation 用時 363.864655 s，普通 p50 50.7335 ms 與 baseline 是未配對不同 run，沒有接受為速度收益。Native 29 labels 保留但最大 drift `0.008602023`。Compile 維持 experimental、預設關閉，沒有將相同總 accuracy 當作 parity。

以下 Modal CLI／callable 路徑已完成四個市場 checkpoint 的 public231／coherence，另外命名的 independent／compiled 與 supplemental stages 已分開記錄。每次使用新的 run id；未執行 GPU HTTP endpoint smoke。

```bash
uv run --no-sync modal run apps/modal/jevbench_campaign.py \
  --run your-unique-run \
  --models s1-bf16,onejev-4b,decider-2b,jev-omni
```

原始完整本機 evidence 在 `artifacts/jevbench/20261008-public-01/s1-bf16/`、`artifacts/jevbench/20261008-public-02/` 及 `artifacts/jevbench/20261008-public-03/`；第一個實際執行版本在 `artifacts/jevbench/executed-public01/`，後續 run 保留各自的 executed source archive，current wrapper 在 `runtime-current` 另列。完整封存見 [本輪證據](README.md)。提交前將發布固定的 evidence index／source／wheel URLs，並依維護者要求驗證 endpoint 啟動命令。

[raw-evidence.zip](raw-evidence.zip) 共 8,219 members，10,932,994 bytes，未壓縮 28,970,101 bytes；SHA256 `b85b4471d655a05e5fc682d2d22c157ae86af9b42a05a926e2bd78655223f53b`。[外部 manifest](raw-evidence.manifest.json) 為 2,127,644 bytes，SHA256 `5cea21069490c69b59f8b9f12fd76062aeb2ed3b395a0ba3fe6a86d5d8d7b44d`。全部 members 的 hash／size／CRC 已重讀核對，包含 1,848 public raw files、6,240 coherence cache files、完整 logs／source／audits／新 wrapper wheel。

65 個 frozen package 檔案與 Git `501c8a466fa6e694b184283d0574526a500b1655` bytes 一致，run02 四個及 run03 兩個 cells 各 60/60 source hashes 匹配 execution receipts。Run01 舊 receipt 只記錄 runner hash，其他 59 個來源 hashes 未記錄；保留 bytes 與另存 snapshot 的核對仍是較弱、partial 的來源證明，不稱為完整 execution attestation。這個限制不隱藏，也不由新 wrapper 的 source 核對取代。

## 重現包應包含的內容

- 模型與 processor 的 immutable download revision、檔案 SHA256／size、模型 card 與 license／NOTICE。
- 執行所需 source 或 wheel、SHA256、固定依賴與硬體要求；受測的 calibration sidecar 與 provenance。
- 只轉換 wire schema 的 adapter；每題完整機率、question／option ids、Score level order 與上下文拒絕政策。
- 對應的 suite／cases／request manifest、raw answers、錯誤／unsupported 明細與 coverage。
- JevBench dimensions、confidence intervals 與 accuracy 的獨立報表；不同硬體／runtime 的 latency 分開呈現。
- 已接觸或用於調整的 public benchmark／publisher heuristics 聲明；正式 sealed 資料不由本專案索取答案或用於調參。

## 目前可提供的背景證據

[PR #6](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/6) 已有已公開 NVFP4 derivative 的 B200 conversion／save-reload／bundled-wheel replay 證據，以及固定 Jev-Omni revision 的 A100 比較。PR head `03860a3` 的既有 required CI 成功、merge state CLEAN，已本機整合，沒有對外 merge 或 push。這些歷史資料是 BoolQ 與小型 native media regression population，不能代替 Jev official sealed 或本輪 JevBench coherence suite。完整 scope 與驗證見 [本輪計畫](../../../jevbench-plan.md)、[NVFP4](../../2026-10-07/nvfp4/README.md) 與 [Jev-Omni](../../2026-10-07/jev-omni.md)。

## 待維護者確認的事項

1. 接受的評測／收錄途徑、模型大小或 license 條件，以及是否由維護者啟動評測。
2. Jev public accuracy 與 official sealed 評測的精確 runtime、request、format、temperature 與 latency protocol。
3. JevBench 榜單收錄是否需要追加 full suite、重新執行、原始 answers 或 confidence interval 證據。
4. 此次 public-data 暴露聲明與 inference profile 命名是否足以讓結果與現有模型公平比較。

正式收錄與排名只有在維護者確認後才更新。本草稿沒有代表使用者聯絡任何維護者。
