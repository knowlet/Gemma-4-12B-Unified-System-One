# Decision3 可變選項特徵：13 筆原生擷取與獨立重播

日期：2026-10-25（Asia/Taipei）。[上一輪收集](decision3-affordance-collection-results.md)的 13 筆完整 train-only 瀏覽器請求，已完成真正 d3-edge 原生特徵擷取及獨立數值重播。模型權重尚未更新，沒有新的瀏覽器成功率或 JevBench 分數。

## 執行與人口

使用 GX10 的原有 CPU5 owner：affinity 15–19、quota 500%、4 GiB、swap 0；每次呼叫保留原本 20 秒 hard watchdog，整批保留 120 秒。模型是未適配的 `vllm-sr/d3-edge`，revision `cab71e6ad3461b8086d51ddb572c656babd63b0b`，BF16 backbone、FP32 255×1024 readout、T=1、最大 native batch 4。

| 分塊 | 完整請求 | 實際 head 數 | 操作選項數 | 整批執行秒數 |
| --- | ---: | --- | --- | ---: |
| 第一批 | 8 | 4 筆 M2、4 筆 M4 | 4 筆 K4、4 筆 K7 | 57.3274 |
| 第二批 | 5 | 5 筆 M4 | 1 筆 K7、4 筆 K8 | 48.9694 |
| 合計 | 13 | 44 個實際 head | K4／K7／K8 | 兩次獨立載入 |

完整 wire、模型別名、head／criteria 順序及 singleton 均保留。每筆以實際 M 決定 `[M,1024]` 特徵形狀，未補成四個 head，未裁掉不參與 loss 的 target head。13 筆全部驗證後才依原序分成 8＋5；特徵擷取不讀教師標籤。

單次完整呼叫含擷取、原生驗證、Torch／NumPy 重播、來源驗證、hooks 清理及證據寫入，為 3.7349–7.3422 秒。這是新訓練場景的實測，不能代替之前 4,579-token 自造長輸入診斷或完整十場景的延遲結果。兩個 unit 的 peak 分別為 1,789,501,440 與 1,447,174,144 bytes，`memory.events.max/oom/oom_kill` 均為零；數值包括 wrapper、framework 與 cache，並非模型單獨用量。

## 獨立驗證

每次原生 forward 同時保存原始完整回覆、最終 norm 的最後位置特徵及 FP32 readout。執行中的 Torch replay 與 NumPy zero-delta projection 對照所有實際 head；兩次載入的 FP32 baseline bytes 完全相同。擷取後，Root 另以不載入 Torch／NumPy／模型的標準函式庫程式解析 NPY，再用 float64 dot product 重建合法輸出。

| 獨立重播 | head 比較 | 合法機率比較 | 最大絕對誤差 |
| --- | ---: | ---: | ---: |
| 第一批 | 24 | 72 | 5.72657545161448e-7 |
| 第二批 | 20 | 67 | 1.5103635578883612e-6 |

139 個合法機率全部符合原門檻 `abs(replay-native) <= 1e-6 + 1e-5*abs(native)`，44／44 個 ordered argmax 相同，並驗證正規化、使用量及 singleton／entropy confidence。特徵列與真實 `head_id`、`h_index`、criteria 順序及完整 wire hash 逐筆連結；沒有假設 operation 在第零列，也沒有按 K 猜測 head。

Root 同時核對原始 command intent／observed child／result、nonce、invocation、sealed config、27 個來源參照、每次來源及輸入最後驗證。原始 owner receipt 均成功，所選 unit 與觀測 cohort 已退出，cgroup 不存在、兩次掃描為空。原始 `model_qualified`、`continuation_authorized`、`whole_tree_exit_confirmed`、`no_escape_enforced` 均仍為 false。

保留一次尚未建立 owner output 的啟動前拒絕：source pool 與 owner 自帶參照重複。後續以全新 intent／輸出排除重複，再通過原 controller 的純檢查；未修改 controller、記憶體或期限門檻。新的來源也通過 7 個有界 stdlib 假測試與另一份僅來源審查；假測試不作為實際數值資格。

## 下一個模型候選

13 筆 operation 標籤另從已獨立 admission 的 authored `valid_next_targets` 匯出，與完整 wire／實際 head 位置連結。標籤沒有取用模型自己的選擇，沒有從 step ID 推測答案。此階段沒有 optimizer，所有 feature／label 產物仍為 `fit_authorized=false`、`model_score=null`。

下一步以獨立的合併訓練接受條件，將舊 96 筆與新 13 筆 operation features 合併為 109 筆；舊 K8 loader 與 immutable selection 不變。訓練後仍需真正的數值、完整瀏覽器與兩套 JevBench 評測。原模型的 Benchmark Heaven 158／231 與 coherenceMINI 70.586601／100 分開保留，本輪未重跑；新候選尚未有分數。

公開數值與證據 SHA256 見[聚合收據](decision3-affordance-feature-replay-results.json)。原始 wire、教師逐筆標籤、機率、特徵、baseline 與本機實驗產物未放入公開收據；公開報告不構成可在缺少私人產物時重播的完整封存。
