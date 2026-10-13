# Mac 本機 jev-ultrafast 整合與效能紀錄

日期：2026-10-13（UTC）。既有訓練版 Gemma 4 12B System One 已透過真實
S1 決策介面、同模型文字 helper 與隔離 Chromium 執行本機 fixture。
保留的 baseline 獨立 DOM 驗證成功率為 **travel 0/3、completion 0/5**；失敗任務的
時間保留為失敗證據，不當作成功速度。

後續 opt-in Noul policy 的單次 development click 已真正 **1/1 完成**：
CLICK → DONE，fresh DOM 通過，2 次決策／9 次 native 推論，104.689 秒。
[獨立報告與原始證據](noul-development/report.md)保留配對分數、source freeze、
資源與限制；這不改寫 baseline，也不宣稱 heldout 泛化或穩定加速。

另有一次[凍結多任務有界窗口](heldout-window/report.md)：預定 24 次，執行 6 次，
2 成功、4 失敗、18 未執行；包含清理的總時間 851.564 秒。
103 次 native server responses（10 暖機、93 測量）全部核對；四次 client 取消
仍有 server 回應，未補算成功。這是未完成的 candidate arm，沒有 paired baseline，
完整群組存活驗證也仍未知。公開 evidence/audit 加遮罩收據共 20 份，
模型輸入及數值未改變；模型 package 關閉後全部 9 檔 SHA 再驗證一致。

固定輸入比較各有十次測量：Q4_K_M 決策 p50／p95 為 **6.294／8.289 秒**，
Q8_0 為 **9.588／10.588 秒**。兩組依序執行，沒有控制熱狀態與 OS cache；
這是模型固定輸入比較，不能推算 agent 速度或通用可靠性。

- [完整報告與量測界線](report.md)
- [機器可讀重算結果](results.json)
- [所有公開 raw attempts 與 resource logs](raw/)
- [隱私遮罩收據](raw/sanitization-manifest.json)：104 份 raw text files
  加一份收據，共 5,748,456 bytes（5.48 MiB）。保留原始／公開 SHA256、
  路徑替換紀錄與未改變的模型 request hashes；機率、計時及 token counts 不變。
- [Jev feature branch 與重現指令](https://github.com/knowlet/jev-ultrafast/blob/db122e8960655404c1062e9108f8e033275bb228/docs/local-benchmark.md)
- [離線分析腳本](https://github.com/knowlet/jev-ultrafast/blob/db122e8960655404c1062e9108f8e033275bb228/scripts/analyze_local_results.py)、
  [模型／瀏覽器 benchmark driver](https://github.com/knowlet/jev-ultrafast/blob/db122e8960655404c1062e9108f8e033275bb228/scripts/local_benchmark.py)、
  [隔離程序 launcher](https://github.com/knowlet/jev-ultrafast/blob/db122e8960655404c1062e9108f8e033275bb228/scripts/run_local_benchmark.py)

實際 S1 執行來源是 `eef847f01b63c86f05df81027a959cf3c658d54b`。
本報告分支基於公開 main `332f80db535983eca22322927f065a62159b7526`，
沒有把新版來源或之後加入的 Jev CI／portable paths 當成原始測量來源。
原模型、校準與 processor 均保留，沒有重新訓練、下載權重或改寫既有 checkpoint。

[所有日期的驗證索引](../validation/README.md)保留此前的公開 JevBench、
coherence 與 CPU replay 證據；本機 fixture 不代表 sealed evaluation 或正式排名。
