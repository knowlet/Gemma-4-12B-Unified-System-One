# Decision3 Edge：GX10 多核心長輸入診斷

同一個自造長輸入在 GX10 上，10 個高頻核心的本次呼叫為 **23.304 秒**，
16 核為 **24.256 秒**，5 核兩次分別為 **28.225／28.321 秒**。
四次原生呼叫都返回，但均超過原本的 **20 秒合作式 callback 期限**，
因此全部保留失敗，沒有授權瀏覽器動作。

這是未適配的 `vllm-sr/d3-edge` 原模型診斷，revision 為
`cab71e6ad3461b8086d51ddb572c656babd63b0b`；未載入 operation delta。
測量來源提交為 `adf9f83143a852f46a646a11c736ffc7f3810265`，
各次額外診斷來源另以 SHA256 固定。這次沒有執行瀏覽器任務，也沒有重算品質分數。

| 執行順序／CPU 設定 | 長 callback 秒數 | owned unit 記憶體峰值 bytes | memory.max 事件 | 20 秒期限 |
| --- | ---: | ---: | ---: | --- |
| 1：5 核，15–19 | 28.225329 | 4,294,967,296 | 175 | 未通過 |
| 2：16 核，4–19 | 24.256422 | 3,400,183,808 | 0 | 未通過 |
| 3：5 核重跑，15–19 | 28.321324 | 3,368,226,816 | 0 | 未通過 |
| 4：10 核，5–9、15–19 | 23.303843 | 3,366,998,016 | 0 | 未通過 |

每次都重新載入一次模型，先執行一個短輸入 warmup，再執行一個長輸入。
表中的時間包含 callback 前置準備、原生 backend、驗證與 hook 清理；
機器可讀紀錄另列 backend 與 first-layer 到 final-norm 的時間，後者不包含完整 embedding／readout。
記憶體是整個選定 unit 的峰值，包含 wrapper、framework 與 cache，並非原生 forward 專屬峰值。
四次均無 OOM 或 OOM kill。

第一次 5 核碰到 4 GiB 上限，因此在 16 核之後另行重跑相同 5 核設定。
重跑沒有 memory.max 事件，延遲仍約 28.3 秒；記憶體壓力不足以單獨解釋目前的延遲。
各次開始時 ComfyUI 的 running／pending 均為 0，可用記憶體均超過 120 GB。
這組測試沒有隨機化或多次交錯重複，不能建立完整的核心擴展曲線。
「高頻核心」依 GX10 sysfs 的最大頻率設定選擇，沒有量測執行中的實際頻率。

四次使用相同完整輸入：四個 head 的 token 長度為
3117／4196／4579／3126，候選數為 7／38／36／1；有效 token 共 15,018，
padding 後為 18,316。實際首層與 final norm 觀察形狀均為
`4 × 4579 × 1024`。沒有截斷輸入、刪除候選或改變問題順序。
BF16 backbone、FP32 `[255, 1024]` readout、T1、原始 attention／mask
與 Torch reference kernels 均保持不變；只改 CPU affinity、quota 與 intra-op threads。

短、長兩個輸入的原生回應也在私有證據中逐項核對。以第一次 5 核為基準，
另外三次各比較 164 個合法候選機率，最大絕對差均為 0，8/8 個 head 的 argmax 相同。
這只支持這兩個自造輸入的數值一致性，不代表瀏覽器品質或其他輸入已等價。

原始 owner 的四次 scoped closure 均通過，各有兩次 clear process scans，
來源保持不變、cgroup 已消失，沒有本次模型程序留下。
原始 `completed=false`、`continuation_authorized=false`、`model_qualified=false`
與 whole-tree 限制均保留；成功清理不會把期限失敗重新判成通過。
合作式 20 秒檢查在原生呼叫返回後拒絕，並非 20 秒硬取消。
整個診斷仍受 120 秒硬 watchdog、4 GiB／swap0、原 host reserve 與
30,414,140,212 bytes 啟動門檻約束；未聲稱通過原 native.execute 的 hard20 smoke。

兩套 JevBench 的[既有原模型結果](decision3-edge-cpu4-coherence-complete-results.md)
仍分開報分：Benchmark Heaven public231 為 158/231（68.398268%），
JevBench coherence MINI 為 70.58660130718954/100（1,248/1,248 完整）。
這些是先前 CPU4 的品質評測，沒有套用到本次 CPU 設定的延遲比較或新的 operation 候選模型。
[先前瀏覽器比較](decision3-v7-long-page-timeout-results.md)的完整候選分數仍為 null。

下一步量測各 decoder layer 的耗時，再針對主要瓶頸驗證優化。
完整瀏覽器成功率與競品比較仍須另外完成，保留原時限與所有失敗分母。

[機器可讀彙總](decision3-edge-cpu-long-input-diagnostic-results.json)包含四次時間、資源、
數值比較及核對來源 SHA256；公開內容沒有原始請求／回應、DOM、gold、
逐候選機率或模型資料，也不聲稱完整公開重播驗收。
