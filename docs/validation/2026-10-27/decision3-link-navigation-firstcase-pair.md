# Decision3：雙頭候選真實瀏覽器單案例配對

日期：2026-10-27

在 GX10 上，將 [465／110 訓練資料的雙頭候選](decision3-link-navigation-cpu-fit.md)與未改動的 NF4 基線，各執行一次原始 V7 瀏覽器開發案例。兩臂的案例、seed `20261014`、16 次決策上限、觀測 profile、頁面判定方式和 NF4 文字服務一致；候選服務先關閉，再執行基線。這個案例已用於選型，不能當作未見資料或官方排行榜成績。

| 指標 | 雙頭候選 | NF4 基線 |
| --- | ---: | ---: |
| 原始獨立頁面 judge 成功 | 0／1 | 0／1 |
| Policy 呼叫／文字呼叫 | 16／0 | 16／1 |
| 實際動作 | 16 | 5 |
| Policy 呼叫 p50／p95 | 6.719／8.549 秒 | 4.057／7.327 秒 |
| 單案例時間 | 125.756 秒 | 87.146 秒 |
| 已記錄 HTTP 嘗試失敗 | 0／16 | 0／17 |
| 成本 | 未知 | 未知 |

p95 是各臂 16 次 policy 呼叫的 nearest-rank 值；只有單一案例，不代表整體延遲分布。候選臂額外記錄 HTTPX 客戶端內容、GPU 見證和固定來源檢查，基線沒有完全相同的記錄開銷，因此這些呼叫時間不能作為純模型速度的公平比較。硬體、程序限制及逐次取得的 GPU forward 租約相同；NF4 為其決策和文字服務使用 178 次受序列化 GPU forward，候選為 16 次。

兩臂均因 16 次決策預算耗盡而結束，live DOM judge 均判定目標未達成，且驗證本身沒有報錯。候選的 16 個動作全部是 CLICK，在兩個記錄目標之間分別出現 8 次，沒有 TYPE_TEXT；每次記錄的頁面指紋都標示為變化。此資料顯示重複操作與操作選擇值得優先調查，但不能單憑目標字串相同或指紋變化，推定每次實際 DOM 狀態都完全相同。前一個 [text-history 候選](decision3-text-history-firstcase-pair.md)在同一開發案例也為 0／1，當時為 3 次 policy 呼叫及 2 個動作；兩次只各測一遍，無法判定一般化改善或退化。

## 驗證與限制

原 runner 在真實頁面執行獨立 judge，並保存其 fresh DOM 觀測；另一個 CPU 程序將兩組保存的觀測重新交給相同 judge，重算結果與原紀錄一致。這是保存觀測的重播，不是第二次開啟真實頁面。候選的 16 筆 HTTPX 客戶端請求與回應逐一對應服務端同次綁定、GPU 見證和已關閉的瀏覽器程序；客戶端內容可能經 HTTPX 解碼，並非原始 HTTP 傳輸框架。獨立 FP64 重算 568 個合法機率位置，原生與同次零增量最大絕對誤差均為 `3.0418974e-7`，候選為 `3.4917349e-7`。這核對了候選的數值與接線，不代表決策正確。

外層監督取樣 133 次，ComfyUI 佇列始終為空，最低可用記憶體約 96.95 GiB；候選與外層工作單元的 cgroup 均已消失，候選 CUDA 配置歸零。租約只約束參與此協定的程序。數值稽核紀錄 SHA256 `59c085c0449376d334d53837ad7ae9df65832cb533b9375245e1bd8e278cc7d8`；DOM 重播 `7f6a23b074ebeb0b5049114c985589e3890471d13829167d15e2d6262c2c497b`；外層關閉紀錄 `8161d22d9bdbb217f5f1edca439bb59a72d612ebb6bd1b7a3b7515b79025ff4f`。原始請求、頁面及瀏覽器軌跡只保留在本機私人 artifacts，未收錄於本報告或訓練集。

此次使用的本機監督器、封存來源和輸出位於 `artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/`。同一台機器上可從儲存庫根目錄，以新的檔名重播兩項稽核；它們不重新呼叫模型或瀏覽器：

```sh
.venv/bin/python -B -s artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/root-link-pair-candidate-audit-v1.py --output "$PWD/artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/root-link-pair-capture-diagnostic-20261027-01/candidate-numeric-audit-replay-02.json"
.venv/bin/python -B -s artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/root-link-pair-dom-replay-v1.py --output "$PWD/artifacts/jev-runtime/decision3-text-history-combined-fit-cpu5-20261027-01/root-link-pair-capture-diagnostic-20261027-01/dom-replay-audit-replay-02.json"
```

這些本機 artifacts 和權重未包含在 Git 提交中；跨機器重現整個瀏覽器試驗還需要相同已釘選模型、原始網頁條件與監督環境。兩套 JevBench 尚未對此候選完成新的有效評測，也尚未在未見任務證明其優於基線。
