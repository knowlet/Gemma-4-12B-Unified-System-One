# GX10 本機 JevBench 評分

2026-10-22，在實作前已下載並查閱兩套官方來源，另比對本專案既有單模型 campaign。這次固定發布權重的 BF16 causal multislot profile，先完成 legacy public231，再跑官方 coherence mini；不以 browser 5/10 代替此評分。

| 來源 | 已讀實作與授權 | 採用方式 |
| --- | --- | --- |
| [Benchmark Heaven bb05a335](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc) | tasks.py、scoring.py、summarize.py、adapters/base.py，MIT | 直接呼叫本專案既有 SHA 固定 public231 runner；三份資料及六份 source 已逐檔符合 config。 |
| [JevBench e187336](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) | api.evaluate/from_callable、engine.run、backends.CallableBackend、results.Report、suite manifest，Apache-2.0／NOTICE | 原封不動 import 官方 SDK。當日最新 origin/main 就是此 SHA，沒有 scorer 或 suite 變更。 |
| 本專案 apps/modal/jevbench_campaign.py | 單次 UnifiedDecisionModel 載入後執行 public／coherence，專案 MIT | 僅參考呼叫方式；改為本機受外部 owner 管理，沒有 Modal 呼叫或資料上傳。 |

官方 coherence SDK 唯一必要 dependency 是 httpx>=0.27，本機 ComfyUI Python 已有 0.28.1；以 clone 的 src 路徑載入，不新增 dependency。mini 固定 240 cases、50 relations、1,200 tests、1,248 requests、seed 0。官方 from_callable 接收 answers map；Report.save 同時保留設定、分數與完整 result。

新 worker 只負責一次模型載入與明確命名的固定評測。直接重用 scripts/jev_native_v2_browser.py 的 ResourceJournal，保留 73,958,270,912-byte BF16 載入／forward floor、零 resident／swap credit；使用 src/s1/browser_native_resources.py 已 SHA 固定的 H15 allocator_cap 設定自身 40 GiB native allocator。allocator 與自身 memcg 是不同限制，不加總為額外實體記憶體，也不宣稱 CUDA free 為獨立 VRAM。

主程序與實際 backbone forward hook 都套用同一資源檢查。環境拒絕、deadline 與證據寫入失敗須拋出 BaseException 子類，防止官方 public runner 把環境中止當成模型答錯後繼續。外層 owner 檢查 ComfyUI idle，若工作開始只終止此次自有 unit；不卸載或清除 ComfyUI。

固定模型 knowlet/Gemma-4-12B-Unified-System-One@a66f836b56605039fe040f330180e336d19b3362，原始 BF16 head、發布溫度 2.7830344470383452、context 16384，無 LoRA、cache 或 compile。runtime source、官方 source／suite、config 與 checkpoint streaming SHA 在執行前後核對。raw request／answer／latency 僅保留本機。新的實際 GPU 分數尚未執行；純測試不代表模型品質或排名。
