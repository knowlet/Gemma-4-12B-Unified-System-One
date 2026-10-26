# Decision3：109 筆操作資料的合併訓練候選

日期：2026-10-25（Asia/Taipei）。新候選已完成 512 步 operation-only 訓練，使用舊 96 筆與[新擷取並獨立重播的 13 筆](decision3-affordance-feature-replay-results.md)資料。整次原始 owner 執行為 3.7878 秒，產出仍為 pending；尚未通過部署數值驗證、完整瀏覽器比較或兩套 JevBench。

## 訓練與接受界線

| 項目 | 本次設定／結果 |
| --- | --- |
| 模型基底 | `vllm-sr/d3-edge`，revision `cab71e6ad3461b8086d51ddb572c656babd63b0b` |
| operation rows | 109＝舊 96＋新 13 |
| 操作選項分布 | K4：4；K7：5；K8：100 |
| 新資料的 encoder 輸入 | 完整原生 wire、M2／M4 全部 heads；44 個 head、139 個合法輸出已獨立重播 |
| optimizer 輸入 | 僅依實際 `head_id`／`h_index` 取 operation；教師標籤按當次 criteria 順序對應位置 |
| 更新範圍 | FP32 255×1024 delta；從零開始，BF16 backbone 與原 readout 不變 |
| 固定參數 | 512 步、learning rate 0.003、L2 0.0001、seed 20261023、batch 64 |
| 資源 | 原 owner09，CPU5、affinity 15–19、quota 500%、4 GiB、swap 0 |
| 訓練時間預算 | 原 fitter 的完整 120 秒 cooperative deadline；包含資料驗證、optimizer、最後驗證與寫入，owner 外層期限保持原值 |

舊 K8 loader、原始 selection、數學 core 與舊收據都保持原樣。新 loader 先完整驗證新 13 筆的所有 head，再抽 operation；新資料接受紀錄與標籤紀錄不自動授予 fitting，由另一份明確固定 hash 的 combined capability 授權。

舊資料仍完整驗證 132 個受監督 head 後選出 96 個 operation；36 個舊受監督 target head 與 31 個新 target head 不參與本次 loss。未使用 head 沒有補零標籤。新標籤只來自獨立 admission 的 authored `valid_next_targets`，不是模型自己的選擇。開發結果曾影響原有策略，`dev_used_for_model_selection=true` 如實保留；optimizer 沒有讀取 development／sealed 資料。

## 本次已驗證

新來源通過 9 個有界 stdlib 假測試、Ruff 與另一份僅來源審查後，Root 才另存 release copies。實際擬合讀取所有固定輸入、驗證舊 96 筆與新 13 筆並完成 512 步；新資料在 optimizer 前再次做完整 zero-delta 原生重播。產物始終保持 pending，未改寫資料接受紀錄或宣稱本機部署資格。

Root 另以標準函式庫核對原 execution／intent／observed child／result、capability、34 個來源參照、109 列 mapping hash、FP32 NPY 格式及有限性。delta 非零，L2 norm 為 0.8280175898546096；第 8–254 列精確為零，符合本輪最大 K8 的訓練人口。這不表示非 operation heads 已有新的部署資格，實際套用仍須驗證。

原 owner receipt 成功、所選 unit 與觀測 cohort 已退出、cgroup 不存在、兩次掃描為空。原始 `model_qualified`、`continuation_authorized`、`whole_tree_exit_confirmed`、`no_escape_enforced` 仍為 false。

optimizer 回傳的首／末 mini-batch loss 為 1.706387／0.949172，最小值 0.775372；這些不是完整訓練集評分、保留集或瀏覽器品質結果，不能當成模型勝出證據。

## 本機 GPU 後續路徑

另以獨立的 2 GiB、單 CPU core、30 秒 unit 執行不載入模型的 16×16 BF16 GPU 運算。GX10 的 NVIDIA GB10、compute capability 12.1，在 Torch 2.13.0+cu130／CUDA 13.0 下完成同步運算並得到預期值。

首次探測在清理後零配置量檢查失敗；第二版保留原失敗，使用已安裝 Torch 的目前程序 cuBLAS workspace 清理 API，再確認配置量由 32 MiB 回到零，unit 隨後退出。這僅證明此基本運算路徑，未載入 Decision3、未驗證完整模型 kernels，也沒有 GPU 模型或瀏覽器分數。

下一步需要新的候選數值資格與完整十場景比較；GPU 路徑需另外建立本機 owner／runtime pins 並做真正原生短／長輸入驗證。原 CPU5 資格與自造 CPU10 呼叫結果都不能直接轉成 GPU 資格。

本候選的 Benchmark Heaven、coherenceMINI、十場景瀏覽器分數皆為 null。原模型的兩套 JevBench 分數分開保留，未繼承給新候選。公開聚合與證據 hash 見[收據](decision3-affordance-combined-fit-results.json)；權重、逐筆標籤、特徵及原始預測留在本機私人產物。
