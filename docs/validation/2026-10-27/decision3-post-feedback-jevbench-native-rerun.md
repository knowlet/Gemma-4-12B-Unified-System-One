# Decision3：feedback 實驗後的兩套 JevBench 原生重跑

日期：2026-10-27（Asia/Taipei）。兩套評測分別執行，並由獨立 CPU scorer 重算；結果仍待外部接受。以下只列聚合資料，未納入題目、回答或原始執行紀錄。

兩輪都使用 Decision3 的**原生通用路徑**，沒有啟用瀏覽器 DONE review feedback；瀏覽器 `operation` 和 `click_target` 的語意增量頭亦未參與。這些分數不能證明 feedback 或增量頭的效果，也不能代表瀏覽器任務成功率。

## Benchmark Heaven

獨立重算得到 **159／231 正確（68.8312%）**，**231／231** 筆格式有效；Brier 為 **0.4397679**，ECE 為 **0.1133564**，序數 MAE 為 **0.4533887**。分母是本次這 231 題，不能延伸為所有情境的正確率；這些指標也不是官方 composite 或榜單排名。

獨立稽核彙總包含 924 個事件、231 對租約與 94 次主機取樣。聚合結果 SHA256 為 `3cf7dcdbdd0a58442983a93dcb35904d8bdc197ec6b4b65e46c6a12227ea32b7`。

## JevBench coherence MINI

獨立重算一致性為 **69.6405／100**，95% case bootstrap 區間為 **67.0155–71.9375／100**。五個維度分數如下：

| 維度 | 分數／100 |
| --- | ---: |
| BAT | 100.0000 |
| CHO | 69.4444 |
| LOG | 68.3333 |
| MEA | 34.3137 |
| REP | 76.1111 |

本次包含 240 個案例、1,200 個計畫試驗、1,468 筆 outcome、1,248 次模型 callback；獨立稽核彙總包含 4,992 個事件、1,248 對租約及 760 次主機取樣。這些分母分別描述案例、試驗、結果與呼叫，不能互換。聚合結果 SHA256 為 `45cbbb3b245574fc906275c0f47a73d1bf52abc844da57e1c6e6dc1c3f06d7be`。

## 比較與限制

[先前原生重跑](decision3-post-semantic-jevbench-native-rerun.md)的 Heaven 聚合值也是 159／231、Brier 0.4397679、序數 MAE 0.4533887；coherence 也報告 69.6405／100。數字一致不自行證明兩輪所有模型檔、題組、計時範圍或執行條件完全相同，也不能推論這次 feedback 實驗造成改善或退化。

本輪啟動前以 CPU 核對 **20／20 runtime 檔案**的磁碟雜湊及讀取前後檔案身分與時間資訊，包含兩個 tensor 檔。計畫建立前的 `preflight.json`（SHA256 `d9fcf0c8beb573670b7acbafc0f62ce2dacd33a378ad2c2da6d2ff73cb8ce10c`）只保存這項檢查的彙總，沒有逐檔 receipt；13:16 重新串流 20／20 的結果是一次性記憶體檢查回報，未保存逐檔結果。這些檢查不是 GPU 執行期間各權重內容的獨立證明。完整 manifest 共 30 項，另外 **10 項非 runtime 檔案或資產未驗證**。先前報告記載 19／19 runtime 檔案，但未取得足以解釋兩輪數量差異的明細，不能視為相同核對口徑。

兩個獨立稽核均確認監督 cgroup 已消失，且 worker 回報 CUDA allocated bytes 為零；這不證明整機 GPU 獨占或其他程序的資源已釋放。兩個工作單位均保留為 active/exited、Result 為 success、ExecMainStatus 為 0、ControlGroup 空；coherence 的 MainPID 為 0。這些工作單位不應稱為 inactive。Heaven 啟動前與 94 次主機取樣的 ComfyUI 執行及等待佇列均為 0，最長取樣間隔約 0.5237 秒。coherence 的 760 次主機取樣中可用記憶體最低為 114,450,710,528 bytes，佇列最大值均為 0，最長取樣間隔約 0.5303 秒。coherence 首次取樣晚於 Heaven 最後一次取樣約 80.606 秒，但取樣無法證明間隔內的狀態。所選記憶體門檻未記入 closure，無法僅憑 closure 獨立核對。

執行後的本機記憶體檢查分別回報 Heaven 的計畫來源 **52／52**、coherence 的 **58／58** 項相符，未保存逐檔重驗 receipt。持久的 scorer 稽核彙總中，Heaven 核對 45 個儲存庫來源、coherence 核對 51 個儲存庫來源；各有 7 個外部來源不在相應 scorer 稽核的讀取範圍。兩輪無 GPU 全域獨占證據，背景工作負載和計時條件不足以公平比較速度；成本與耗電未量測。公開資料可能曾被模型或開發過程接觸，這不是 sealed 評測，也不能據此聲稱擊敗其他模型。

另外在 13:30:18 UTC 保存了一份事後重驗彙總 `postrun-source-model-recheck-v1.json`（SHA256 `e23dae89b4ca0223f6208083649cf7029c7dec39597d119da3fc894c9247affc`），回報來源 52／52、58／58 及 runtime 20／20 相符。它是另一次事後檢查，不能作為 13:16 啟動前檢查的持久證據，也不能證明 GPU 執行期間的內容或檔案持續未變。
