# Decision3 可變操作選項收集與長輸入後續

發布日期：2026-10-25（Asia/Taipei）。這次取得 13 筆通過獨立證據重播的完整原生訓練請求，涵蓋 K4、K7、K8。另有兩次自造長輸入呼叫低於 20 秒，但尚未證明穩定延遲或瀏覽器品質提升。沒有新的模型分數或勝出宣稱。

## 訓練資料缺口

既有 operation readout 訓練使用 96 個 operation 範例，全部為 K8。瀏覽器實際會提供不同數量的操作選項；Decision3 的 255 個輸出碼代表當次 criteria 的位置，不能依操作名稱重排權重。這是需要補測的資料涵蓋缺口，尚未證明是先前瀏覽器失敗的原因。

新增的收集器沿用 pinned jev-ultrafast `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` 的 Agent、操作選單、選擇與執行器，採用 settled V7 觀測。四個新寫的 train-only 場景只連到本機測試網頁，以 scripted teacher 在 HTTP 邊界提供執行指令，沒有模型推論。SELECT 沿用原執行器的 JavaScript 事件；未將它宣稱為可信的 CDP 使用者輸入。

| 已完整完成並重播的場景 | 請求數 | 實際觀測的 K |
| --- | ---: | --- |
| compact-one | 2 | 4 |
| compact-two | 2 | 4 |
| form-top | 4 | 7 |
| form-middle | 5 | 7、8 |
| 合計 | 13 | K4：4 筆；K7：5 筆；K8：4 筆 |

每個 K 都包含繼續操作與實際完成後的 DONE。4 筆請求保留 2 個原生 question heads，9 筆保留 4 個；未裁掉未監督 head 或 singleton。teacher、有效目標、證據及 operation-only loss mask 都在原生 payload 之外。

獨立 admission 重播原始 CDP 日誌、freshness guard、完整輸入順序、Agent history、場景完成條件、網路範圍與瀏覽器關閉證據，並逐筆重建 wire。修補前的驗證器可能接受 act 範圍外的輸入或缺少 DONE 後置 guard；新增回歸測試重現三條錯誤接受路徑，修補後拒絕這些情況。這版純測試為 **38 passed、1 integration deselected**，未以假測試取代實際收集。

本次 exclusion inventory 有 64 個目標 hash、101 個 scenario ID、4 個 ancestor ID，涵蓋已知 benchmark、公開保留場景及舊 prototype／DSL development 身分；不是對所有可能保留資料的普遍保證。收集與獨立 admission 均使用不變的 source-v1 副本，之後同步分支沒有改動它。

保留的嘗試人口包括：完整 Chromium 啟動 SIGTRAP，零場景、零模型呼叫；一次宿主記憶體不足的啟動前拒絕，未啟動瀏覽器；以及換用既有 matched 評測相同 headless shell 後完整完成的四場景。失敗輸出未覆寫或混入成功資料。最後執行的原始 owner closure 確認所選 unit 與觀測 cohort 已退出，兩次掃描為空；原始 `model_qualified`、`continuation_authorized`、`whole_tree_exit_confirmed` 仍為 false。

這 13 筆僅通過資料 admission／完整 wire projection，`fit_eligible=false`、`model_calls=0`、`model_score=null`。下一步需要獨立接受訓練輸入、擷取完整原生 encoder 的特徵、訓練後重跑完整瀏覽器比較及兩套 JevBench。使用方式見[收集器文件](../../decision3-affordance-training-collection.md)。

## CPU10 長輸入後續

以下為 10 月 24 日執行、10 月 25 日整理的後續結果。模型仍為未適配的 d3-edge，BF16 backbone、FP32 255×1024 readout、T=1；native batch=4，輸入長度 3117／4196／4579／3126，K=7／38／36／1。自造輸入、編碼、原生數學與 4 GiB／swap 0 資源限制維持一致，CPU affinity 為 5–9、15–19。期限仍為完整 callback 的 20 秒，未放寬。

| 新執行 | callback 秒數 | 20 秒結果 |
| --- | ---: | --- |
| 被動整層計時 | 20.1856 | 未通過 |
| 被動 attention／MLP 計時 | 17.1585；return 時 17.1600 | 通過 |
| 原版無計時 hooks 重複 | 17.0583；return 時 17.0593 | 通過 |

component 計時那次，linear attention 合計 7.6373 秒、full attention 5.5267 秒、MLP 3.4805 秒。這提供下一輪優化的成本依據，沒有更動 kernel 或模型。兩種 hook 版本及無 hook 重複，均在這兩個自造輸入的 164 個合法輸出上與先前無 hook 執行完全相同，8／8 個 argmax 相同；不延伸為一般數值等價保證。

同設定先前無 hook 呼叫為 23.3038 秒，本次重複為 17.0583 秒，宿主記憶體狀態也不同，尚不能歸因於特定優化或宣稱穩定通過。無 hook 重複的 unit peak 為 4,294,963,200 bytes，`memory.events.max=263`，沒有 OOM；不能因此推論記憶體壓力解釋延遲。這些是自造長輸入原生 callback 診斷，未執行瀏覽器動作，也未取代原本 hard-20-second、短輸入的 native smoke qualification。

## 分數與證據

| 評測 | 本輪狀態 |
| --- | --- |
| Benchmark Heaven public231 | Edge 原模型仍為 158／231，68.398268%；本輪未重跑 |
| JevBench coherenceMINI | Edge 原模型仍為 70.586601／100，1248／1248 完整；本輪未重跑 |
| 新的可變 K 候選 | 尚未訓練，兩套分數皆為 null |
| 十場景瀏覽器比較 | 本輪無新模型執行；先前 adapted Edge 因長頁面逾時未完成，完整分數仍為 null |

完整公開數值與 SHA256 依據見[聚合收據](decision3-affordance-collection-results.json)。原始 DOM、wire、教師標籤、CDP 日誌、特徵與模型資料留在私人產物；本報告只公開程式碼、聚合數值與證據 hash。先前分數的完整界線見[兩套 JevBench](../2026-10-24/decision3-edge-cpu4-coherence-complete-results.md)及[瀏覽器逾時紀錄](../2026-10-24/decision3-v7-long-page-timeout-results.md)。
