# Decision3：語意瀏覽器配對後的兩套 JevBench 重跑

日期：2026-10-27（Asia/Taipei）。在 GX10 的 NVIDIA GB10 上以新的執行目錄、程序、租約與結果，依序完成兩套公開評測，並分別以固定官方 scorer 做 CPU 獨立重算。原始結果保留為待外部接受；下列是重算後的公開研究結果。

兩次都使用 Decision3 Edge **原生通用路徑**。瀏覽器 `operation` 與 `click_target` 兩個增量頭全程 bypass，沒有使用新語意瀏覽器候選的輸出。因此，這兩個分數不能用來判定該候選的能力、是否退化或真實瀏覽器成功率。

## Benchmark Heaven public231

固定[官方來源 `bb05a335`](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc)，完成 **231／231** 題、嚴格有效格式 **231／231**，正確 **159／231（68.8312%）**。Brier 為 **0.4397679**，ECE 為 **0.1133564**，序數 MAE 為 **0.4533887**。原 scorer 的逐題與彙總結果均由獨立稽核重算；這是公開題目上的準確率與校準指標，不是官方 composite 或榜單排名。

本次保存的 scorer、native 和傳輸紀錄綁定 924 個分階段事件與 231 對租約。完整受保護 callback 的 p50／p95 為 **0.112／0.250 秒**，共 231 筆；其範圍包含檢查與留證，排除模型載入與後續 scorer 寫檔。

## JevBench coherence MINI

固定[官方來源 `e187336`](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3)，以新的記憶體快取完成 **1,248／1,248** 次模型 callback；240 個案例、50 項關係、1,200 個計畫試驗產生 1,468 筆 outcome。官方 scorer 整體一致性為 **69.6405／100**，95% case bootstrap 區間為 **67.0155–71.9375／100**；1,000 次 bootstrap 與原結果由獨立重播核對。

| 維度 | 分數／100 |
| --- | ---: |
| BAT | 100.0000 |
| CHO | 69.4444 |
| LOG | 68.3333 |
| MEA | 34.3137 |
| REP | 76.1111 |

獨立稽核核對 4,992 個分階段事件與 1,248 對租約，並以捕捉的回應重跑官方 SDK。完整受保護 callback 的 p50／p95 為 **0.285／0.426 秒**，共 1,248 筆；計時包含 guard 和證據同步，不包含時間日誌自身寫入及 SDK 快取命中。

## 執行與解讀範圍

兩輪分別使用新工作單位、輸出與租約；監督取樣顯示 coherence 在 Heaven 的最後一次取樣後開始。工作單位配置為 5 個 CPU 核心、8 GiB 記憶體上限、零 swap；Heaven 的工作／單次呼叫上限為 600／60 秒，coherence 為 1,800／180 秒。它們是資源上限，不是效能保證。

Heaven 有 94 次、coherence 有 756 次主機取樣，取樣均顯示 ComfyUI 佇列空閒且可用記憶體達基本門檻；最長取樣間隔分別約 0.525／0.527 秒。取樣不能證明間隔內狀態；兩次啟動參數指定的記憶體門檻均為 34,709,107,508 bytes，但關閉紀錄未保存該設定，不能只由關閉紀錄獨立核對。兩個 worker 結束時回報 CUDA allocated bytes 為零，所屬 cgroup 已消失；不代表整台機器 GPU 或共享記憶體歸零。

執行前曾驗證 Heaven 的 52 個及 coherence 的 58 個計畫來源 pin；獨立 scorer 稽核分別重驗 45 個與 51 個儲存庫公開檔案，各有 7 個外部來源不在該稽核的讀取範圍。另以 CPU 串流核對模型 manifest 要求的 **19／19** 個 runtime 檔案，包含兩個 tensor 檔；它證明預檢時磁碟檔案相符，不能單憑此證明 GPU 執行期間每個張量的內容。

兩套延遲計時範圍不同，且稽核未證明所有背景負載均無重疊；這些數字不能用於公平的速度排名或宣稱加速。每次成本與耗電未量測。公開資料可能曾暴露於訓練或過往調整，沒有 sealed 評測、榜單提交或競品勝出結論。

聚合稽核輸出（含結尾換行）的 SHA256 分別為 Heaven `a76bfe4603229d934e0e023c7b08db747204ca3e1375bf5565687b243b2a9844`、coherence `0646eeb261a6ae993eefaad6e233fe73be011cbe3eede498aa17810334289cff`。在有本次完整本機證據及固定官方 checkout 的環境，可從儲存庫根目錄重新計分；這些指令不呼叫模型，也不輸出逐題內容：

```sh
env -u PYTHONPATH .venv/bin/python -B -s scripts/jevbench_heaven_scorer_audit.py \
  --run-root "$HEAVEN_EVIDENCE" --official-root "$HEAVEN_PUBLIC_CHECKOUT"
env -u PYTHONPATH .venv/bin/python -B -s scripts/jevbench_coherence_scorer_audit.py \
  --run-root "$COHERENCE_EVIDENCE" --official-root "$COHERENCE_PUBLIC_CHECKOUT"
```

兩份計畫 SHA256 分別為 `3cf7193f422350468c2910bbaca391cf3ff3b17a895aeabc3f9119dc7a513a41` 與 `899996c917b81f22d48cd30225860e62ecc983b7f8f4538548217cbfbc7dad01`；原始題目、回答、私人 cache 與本機證據未納入本報告或 Git 提交。
