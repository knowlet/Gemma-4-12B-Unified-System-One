# Decision3 語意操作頭：單案例瀏覽器配對診斷

日期：2026-10-27（Asia/Taipei）。在 GX10 上，以同一個既有公開 development 案例配對比較語意操作頭候選與 NF4 基準。兩臂均 **0／1**：runner 當次的最終 DOM 判定及事後使用固定原始 judge 重播所留存 DOM 觀察，皆判定未完成。此次重播沒有重新啟動瀏覽器，結果仍待外部接受。

| 指標 | NF4 基準 | 語意候選 |
| --- | ---: | ---: |
| 最終 DOM 成功 | 0／1 | 0／1 |
| Policy 呼叫 | 16 | 3 |
| Text helper 呼叫 | 1 | 1 |
| 已執行動作 | 5 | 2 |
| 案例總耗時 | 82.573 秒 | 29.480 秒 |
| Policy 呼叫延遲 p50／p95 | 4.299／6.819 秒 | 6.734／8.546 秒 |

此處的 policy 延遲分別來自 16 筆與 3 筆，使用 nearest-rank。兩臂的 capture、數值見證與保護程序不同，總耗時也受呼叫次數及提早停止影響，因此這些數字**不能作為公平速度排名**。成本未量測。基準在 decision budget 階段結束；候選紀錄了 premature DONE，且 runner 回報 public navigation scope 錯誤。候選的彙總記錄有 1 筆 request block 及 3 筆 scope violation 記錄；這些記錄不足以重建各事件的完整發生時間與次序，不能據此斷言因果。

兩臂依原有配對 actor 使用同一案例身分、Chromium、V7 observation profile 與 helper、seed、單次重複、16 步上限和 NF4 text helper；最終結果使用 runner 的原始 DOM judge。這是已接觸的公開 development 診斷，並非未見案例、完整 Jev Ultrafast 分數或官方榜單；單筆結果也不足以判斷泛化或勝負。

## 候選傳輸與數值重算

獨立 CPU 稽核在另一程序使用固定的同一 auditor 重新讀取見證，對應 3／3 筆候選 client bytes、HTTP prepared／socket-write 事件、binding 和 witness，並重算 3 份見證的 77 個合法機率。baseline／GPU zero 的最大絕對誤差均為 **1.5582×10⁻⁷**，candidate 為 **1.2755×10⁻⁷**。這確認提供之見證的數值一致性及資料配對，不單獨證明實際 GPU forward 時間或任務成功。

候選 HTTPX capture 共 4 筆，其中 3 筆可依序對應 policy binding，另 1 筆未配對，因 capture 未保存 endpoint，其歸屬為 **unknown**；runner 另回報 1 次 text helper 呼叫，兩者不能僅憑計數相等就認定為同一筆。capture 觀察到的 transport exception 與非成功 HTTP response 均為 0，但無法證明 capture 涵蓋所有傳輸嘗試，也不能由此宣稱整體失敗率為零。capture PID 與受控 client closure 的已追蹤 PID 相符；此檔案對帳本身不獨立證明完整來源鏈。

候選計畫的 55 個 source pin 均於執行後重雜湊一致；publisher manifest 與驗證器選定的模型／程式檔也於磁碟重驗一致。該驗證器略過 LICENSE、NOTICE、requirements、README 與 assets 等項目，且磁碟驗證不證明執行中每個 GPU backbone tensor 的內容。

## 資源與收尾

共用 cooperative GPU lease 留存 **3 對 Decision3** 及 **193 對 NF4** acquired／completed 記錄，逐對身分及時間序相符，測後 lock 可重新取得。外層有 84 次主機取樣，各次 ComfyUI running／pending 均為零、可用記憶體均高於持續門檻；最低約 97.25 GiB，最長相鄰間隔約 2.034 秒。抽樣無法證明間隙內的狀態，協作 lease 也無法排除未使用該 lease 的 GPU 工作。

候選 worker 回報最終 CUDA allocated bytes 為零，候選與外層 unit 的 cgroup 均已消失；兩臂受控瀏覽器 client 的 process-group closure 均為 confirmed。NF4 receipt 在輸出當時仍標示 `process_exit_confirmed: false` 和 `complete_pending_external_closure`；後續外層 unit 已結束且其 cgroup 不存在。這兩份證據的時間和範圍不同，不能把 receipt 改稱為自行確認了 baseline process exit，也不代表整台機器的 GPU 或共享記憶體已歸零。

獨立 DOM 重播聚合檔 SHA256 `58995bfc2cd0abe008a6090bab4dc5cb11292be36b2f8535a82313dfbab3ccbd`；獨立 semantic 傳輸與數值稽核聚合檔 SHA256 `660d10f75a8e8daeba600051723b89a2fe4ec38c3e4486311f3ebce4b68d4fdd`。原始請求、回應、DOM 和逐題內容未納入此公開報告。
