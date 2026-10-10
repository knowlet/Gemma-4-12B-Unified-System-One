# JevBench breakthrough：執行前固定協定

2026-10-10 僅補正公開下載連結；原始協定 bytes 在 immutable 研究 tag 保存，當時參數與數值不改寫，後續 CPU 證據見 [10/10 索引](../../2026-10-10/README.md)。

本輪在 GPU 輸出出現前固定兩階段、八個 profile、資料與配方。機器可讀來源是 [experiment config](../../../../configs/experiments/jevbench-breakthrough-20261008.json)，凍結時間為 `2026-10-08T04:57:27Z`。目前全部 profile 為 `not_run`；沒有新分數、成本、排名或已通過的驗收。Preflight 可以修正實作與契約問題，但須在 GPU 執行前完成並保存實際 source/config bytes 與 SHA256；修正不得改變本輪宣告的資料、配方或比較選擇。

目標是釐清 prompt、原 BoolQ 專項權重與 decision readout 的影響，再執行一次有界的通用型別 continuation。前一輪公開診斷已被看過；它們只能提供已揭露的研究背景，不得把公開錯題、答案、coherence case 或其分數拿來 fitting、tuning、改 recipe、挑 checkpoint。新的測試資料在本輪訓練前保留，但這不等於真實世界或 foundation pretraining 未見證據。

## 固定八個 profile

| Stage | Profile | 權重／prompt | Readout／訓練 |
| --- | --- | --- | --- |
| ablate | `released-current` | Published release／current | 原 native BF16 head，無訓練 |
| ablate | `released-user_question` | Published release／user_question | 原 native BF16 head，無訓練 |
| ablate | `base-current` | Google base／current | 原 native BF16 head，無訓練 |
| ablate | `base-user_question` | Google base／user_question | 原 native BF16 head，無訓練 |
| train | `released-user-control` | Published release／user_question | 原 native BF16 head，無訓練；同 cell control |
| train | `released-user-head-init` | Published release／user_question | 原 52 selected rows 與 bias 複製為 FP32，無更新；precision/kernel control |
| train | `released-user-head` | Published release／user_question | 獨立 FP32 selected-row head，CPU 128 updates |
| train | `released-user-lora` | Published release／user_question | Attention LoRA 128 optimizer updates；原 native BF16 head |

Published release 固定為 `knowlet/Gemma-4-12B-Unified-System-One@a66f836b56605039fe040f330180e336d19b3362`；Google base 固定為 `google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`。下載 immutable snapshot，記錄 config、tokenizer、processor 與所有 weight shard 的 identity/hash。原 checkpoint 保持不變；研究 head 與 adapter 另存，不把八個結果中公開分數最高的 profile 當作自動新版本。

`current` 保留現有題目 placement；`user_question` 把 state、原生 media、同一題目及選項放在 USER，後接 ASSISTANT 的 `Answer: (` slot。兩種模式都逐題切分，一題一次 native forward；多題 request 不共享前一題選項或答題上下文。`head-init` 的 row 值來自原 head，但 FP32 linear/softcap 運算可能改變 logits，因此必須保留此控制組，不能把 dtype 改變的效果全部歸因於訓練。

每個 stage 的四個 profile 在同一 A100-80GB function cell 依序執行；兩個 stage 的實際 A100 board 可以不同，receipt 必須記錄 device name。Backbone 為 BF16、SDPA、完整 native processor；只有兩個 head profile 採 FP32 selected-row readout，LoRA 的可訓練 adapter 採 FP32。沒有 compile、CUDA graph、candidate cache、KV cache、事後 probability constraints。Context 上限 16,384，超限拒絕且不截斷；保留 64 questions／52 options 契約。這輪沒有已接受的速度收益，也不能用跨 stage 的時間作配對 speedup。

## 固定資料與目標

合成資料由 [generator](../../../../scripts/prepare_breakthrough_data.py) 產生於 [data manifest（ZIP：research/data/manifest.json）](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip)。Generator SHA256 是 `30c126df19772a34f41979b77e8afe176f29c121fde688dffb993316f07b9abf`，manifest SHA256 是 `f408124be6c60ba79983222d1bc64a8432733b7803b69046633fa3d403268d96`，seed 為 `20261008`。Config 保存所有檔案與 canonical dataset hashes；啟動前須重驗，不接受同路徑換 bytes。

| Split | Cases／questions | Choice | Noul | Score | JSONL SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 1,024 | 342 | 341 | 341 | `c61d407f5decc76f9a53c2f8315e30a5aa278569a685187114ce2023f82f9f23` |
| calibration | 192 | 64 | 64 | 64 | `7500565b2e380849af3f4c29c9e7917f55a690419de3f24edc05123ed9812f76` |
| test | 384 | 128 | 128 | 128 | `db6cf2214832fd13289170f00e1b9e3d060373f468cf4cd264b8f27a0477c889` |

Source group 由完整 canonical finite world 計算，排除 group、case ID、request fingerprint 的跨 split 重複，不能靠換 nonce 假裝新來源。Choice 包含 2／4／8 候選，依有限人口的已知數量求機率；Noul 包含八種 propositional／conditional operation；Score 使用 3／5 levels，包含不等距 `0,2,5`，有精確數量與真實期望分數。

Train/calibration 的 inventory、census、flag/table 與直接 rating 頻率模板，和 test 的 partitioned category aggregation、nested policy records、measurement-to-score band mappings 分開。Test 六個 template families 在 generator 內事先保留。Primitive logic 與 numeric level sets 仍共享，因此測量的是合成模板遷移，不宣稱未知數學規則、廣泛實務泛化或預訓練未見。生成沒有下載 corpus、呼叫模型、讀取 public231/coherence 內容；目前 forbidden fingerprint 庫為空，不能宣稱做過外部 benchmark 精確重複 audit。

`soft_gold` 是已知有限人口的完整抽樣分布；hard `gold` 只是 oracle 最可能類別，有 tie 時用 canonical world label order。一次隨機抽樣尚未發生，故其 argmax match 不叫真實事件 accuracy。主要量測 soft CE、KL、Brier 與 Score expected MAE，並保留各型別／template strata。Exact masses、分母、world、運算與期望分數保存在 provenance，答案不進 inference request。載入要使用 `EvaluationCase`／`load_cases`，不可透過會丟棄 soft targets 的基本 `Case` view。

原 release regression 另從 `artifacts/release/datasets` mount 為 `/workspace/release-data`，manifest SHA256 `e0ab28e3435a910c19abf3a9a7e58df8b5cff7a806b6c7f4f18021db1d3804dd`。`scripts/release_training.py:load_release_data` 驗完整 manifest、source/checksum、split/group/request integrity。全體使用其 BoolQ `test.jsonl` 256、`media-test.jsonl` 52（32 MNIST image／20 FSDD audio）、historic `regression-text.jsonl` 128；既有 train2048/calibration256 只為完整 loader integrity，完全不作本輪 fitting/calibration。

BoolQ source 固定 `google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5`、upstream `90af34107399cc7a446b373dc4ee35b8001da7c2`。Release fresh256 指相對原 release 訓練與先前 passage groups 的保留資料；此前已驗過，不能再描述為本輪首次未見。Native fixture 另用 `examples/benchmarks/mps.jsonl`，8 cases／29 questions，SHA256 `23de11921c22807ef90841916a052b172a88667814239d30629ed2e71e1f971b`，含 text/image/audio/mixed；它不能取代完整 native52。

## 固定訓練與 calibration

Head 只使用 1,024 train cases 的 native `user_question` hidden features，轉為 detached FP32 CPU features。52 rows 與 bias 由原 head 初始化，無 bias 時初始化為零；backbone 和原 native head 不更新。AdamW、weight decay 0、seed42、128 steps、batch64、LR `1e-4`、clip norm1；loss 為 soft CE + `0.1 × Brier` + `0.01 × mean squared weight anchor`。Anchor 作用於 weight 相對初始化 rows；不是 bias anchor。Batch 內抽樣不重複，batch 間可重訪 train cases；128×64 不是宣稱每個 case 恰好八次。

LoRA 從 published immutable release 重新載入，rank8、alpha16、`q_proj/k_proj/v_proj/o_proj`、bias none、FP32 trainable adapters；AdamW、weight decay0、seed42、LR `1e-5`、non-reentrant gradient checkpointing、clip norm1。每個 microbatch 一個 case，梯度累積8，128 optimizer updates = 1,024 train forwards，shuffle 後完整 train population 恰好一次。Loss 為 soft CE + `0.1 × Brier`。只評最後 update128；中途 adapter snapshot 是回復與稽核資料，不能看 public 分數挑選。

每個 profile 都只用新 synthetic calibration192（每型別64）擬合自身 Choice/Noul/Score temperature，以 mean soft CE 在 `numpy.geomspace(0.1,10,81)` 取第一個最小值。保存 raw logits、targets、before/after soft CE、選中的 T 與是否命中 grid boundary。Positive T 不改同題 argmax；不能把 calibration 當分類錯題修復。Grid 與 recipe 不因 public/coherence 結果改動。

Public231 保存三組：calibration-only type T 的 native GPU 結果，以及同一 raw logits 的 unit T=1、published global T=`2.7830344470383452` replay。Global T 對每個 profile 用同一固定值作控制；它不是 Google base 自身擬合過的 T。Replay 不再呼叫模型，其 durations 是 CPU replay，不是 GPU latency。Coherence 保存 calibration-only type T 的 native 結果與 unit T raw-logit replay 兩組，完整呈現 temperature 是否把分布推向 uniform；不能只報較好看的一組。

## 全人口結果與驗收

每個 profile 必須完成 fresh synthetic test384、legacy public231、release BoolQ256、native52、historic text128、fixture8/29 questions，以及 coherence-mini 的兩個 policy。Coherence 固定 [JevBench source revision](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3)：240 cases、50 checks、1,200 tests，每組完整1,248 cached answers；報 pinned official strict overall、BAT/MEA/REP/LOG/CHO、per-check outcomes、support 與 errors，不拿 graded score 或 calibration 充當 coherence。

Public231 固定 [legacy official source revision](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc)，canonical dataset SHA256 `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`。它和 current public300 + sealed1200 的正式1,500題不同。這輪 `not_submitted`、sealed `not_run`、rank/cost 都未知；沒有 official endorsement、上榜或成本／延遲 cap 達標結論。

Head 的主要訓練比較為 `released-user-head` 對 `released-user-head-init`，避免混入 precision readout；LoRA 對 `released-user-control`。兩者仍要對原 native control 報全部 retention。Ablate四組保留完整 released/base × current/user_question 因子結果，不選公開最佳組再重開訓練。

Fresh soft metrics 必須改善，且不能傷害完整 release BoolQ/native/historic/fixture population；Score accuracy／expected-value MAE 與 LOG 是需要保留的強項。Noul false-accept/false-reject 要帶 gold 類別分母，不把 aggregate accuracy 上升當作錯誤風險已降低。所有分布與 label flips 都保留；errors、unsupported、unmatched、not-run 不得從人口消失。

比較按相同 case/question/source group 配對，以 repo 的 source-group bootstrap 報95% CI（seed42、2,000 resamples），不要把 coherence repeated slots 當獨立 samples。型別與 template support 一併報告；八個預先宣告比較仍為 descriptive，沒有 best-of-eight 或已做 multiple-comparison 調整的 superiority claim。

本協定不自造允許退化 margin。正式 noninferiority 使用既有 `src/s1/evaluation/gates.py`，要求 explicit pinned pair/population、`all_decisions`、support1.0、至少30 groups；目前 accuracy margin 為 null，formal gate 為 `not_assessed`。若沒有事先固定的 explicit gate，或 CI 跨零、完整人口不足、結果有型別／LOG退化或混合 tradeoff，結論保持 `inconclusive`，不能寫成通過 noninferiority。足夠證據的正向結果也只是新研究候選，不自動改 SDK default、發布模型或宣稱 deployment ready。

## 執行與證據保存

Preflight 完成後使用 app 的既有 local entrypoint；以下是待執行命令，不代表 GPU smoke 已完成：

```bash
uv run --no-sync modal run apps/modal/breakthrough.py --run <unique-run> --stage ablate
uv run --no-sync modal run apps/modal/breakthrough.py --run <unique-run> --stage train
```

Remote 為 Volume `gemma-unified-system-one` 的 `/vol/breakthrough/<unique-run>/<stage>`；local 匯出為 `artifacts/jevbench/breakthrough-20261008/runs/<unique-run>/<stage>`。每個新 run/stage 不覆寫現有目錄。保存 frozen config、executed source、checkpoint/data identity、calibration/heldout raw logits、全部 public policy response、release regression、native fixture、兩組 coherence cache/report、official replay audit、training update logs、head/adapter hashes與實際 GPU/dependency環境。Export 完成不等於所有 profile 成功；failed/not-run stage 與未下載 binary 需明列，訓練 artifact 仍要另驗保存。

訓練輸出不得覆蓋任何 published checkpoint。執行後只補原始 receipts、稽核與結果，不回寫本協定來合理化已看到的結果。若需要下一輪 recipe，必須新 experiment ID、先固定新獨立協定與來源，再執行。
