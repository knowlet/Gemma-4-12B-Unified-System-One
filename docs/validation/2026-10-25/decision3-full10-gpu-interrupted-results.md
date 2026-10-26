# Decision3：完整十場景 GPU 比較中止紀錄

日期：2026-10-25（Asia/Taipei）。本次已真正啟動原定的完整比較，目標仍是 NF4 基線 10 題與 Decision3 GPU 候選 10 題，使用原有 V7、相同常駐 NF4 helper 與固定順序。執行期間 ComfyUI 出現另一筆工作，觸發 `Comfy_became_busy_stop_own_unit_only`；原 launcher 以同一個保留的程序 handle 觀測至終態，整次為 **506.14779754402116 秒**，exit code 為 **1**。

這是中止及所選範圍清理的紀錄，沒有完成 matched score。候選尚未執行，不能記為 0/10。

| 項目 | 已核對結果 |
| --- | --- |
| 原定案例 | 基線 10 題＋候選 10 題，沒有縮減或更換案例 |
| 啟動時可用記憶體 | 85,573,648,384 bytes |
| 原完整 NF4 啟動門檻 | 74,025,380,596 bytes，保持原值 |
| 基線 | 9 筆完整 infrastructure records，其中 4 筆 `task_success=true`；第 10 筆 caller 已建立，但遭中斷 |
| Decision3 候選 | 沒有啟動服務，沒有模型呼叫；10 個預留 owner outputs 均未建立 |
| 來源 | 沒有變更所選來源 |

前置清理所中止的工作，與執行期間新出現的 ComfyUI 工作不同。本次觸發停止後，只停止自己所選的單位，沒有中止新工作；公開報告不包含工作識別資訊。

兩個所選使用者單位目前均為 not-found／inactive，原所選 cgroup 已不存在；兩次程序掃描的 matches／unreadable 均為空。Pump 的 `independently_closed=true`、children 為 0、`complete_ten_cases=false`，只表示本次沒有留下活躍的候選 child，不能當成十筆候選已完成或已通過模型／瀏覽器資格。

啟動前另有 8 個自造 launcher 控制通過，耗時 0.008 秒，並通過 Ruff 與獨立來源審查。這些控制驗證保留原程序 handle、再次中斷與紀錄寫入失敗的收尾行為，不代表模型或場景品質。啟動門檻、模型／helper 設定及 20 題目標均未因此放寬。

| 本次新候選結果 | 分數 |
| --- | --- |
| 完整十場景 matched 比較 | null |
| Benchmark Heaven | null |
| JevBench coherence | null |
| 官方排名 | 無 |

四筆基線成功只是未完成執行中的已觀測計數，不另計成功率或與未執行候選比較。先前的 [109 筆候選 GPU 數值驗證](decision3-affordance-operation-gpu-numerical-results.md)及[排程與服務準備](decision3-gpu-forward-lease-preparation-results.md)保留各自範圍；本次沒有新的 Decision3 品質、競品勝出或正式分數結論。

獨立稽核已通過，狀態為 `ROOT_ACCEPTED_SCOPED_INTERRUPTED_CLOSURE_ONLY_NO_MATCHED_SCORE`。稽核核對 8 個所選參照、36 個部分案例紀錄參照及 848 個原始來源的未變更證據；相同 helper／設定、NF4 快取與 Torch 來源前後一致，原 actor 已不存在。這只接受本次中止後的範圍清理，不產生完整比較分數。稽核紀錄為 2,830 bytes，SHA256：`3e8880b0988638176a5c35fb4bde1e000be159638a466ead31a936efbc21dc28`。

聚合欄位見[機器可讀紀錄](decision3-full10-gpu-interrupted-results.json)，方法進度見[研究紀錄](../../research/jev-operation-only-readout.md)。原始請求、逐筆預測、程序身分與私人產物路徑不公開。
