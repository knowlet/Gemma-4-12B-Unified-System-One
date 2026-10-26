# BF16 mmap/offload：GX10 小型 CPU 實測

2026-10-23（台灣時間）。已使用真實 Torch、Accelerate、safetensors 與 PEFT 完成自行產生的小型 BF16 模型測量。**mmap 與 CPU hook 數值路徑可運作，但 Parameter 身分與例外復原不能直接沿用現有正式模型的檢查。** 本次不是 12B、GPU 或瀏覽器品質評測。

| 項目 | 實測 |
| --- | --- |
| 自產 safetensors | 232 bytes；embedding 8×4、projection 4×4，head 與 embedding 共用 storage |
| Actor 時間 | 13.738 秒，包含套件匯入；外層控制流程 14.928 秒 |
| 程序峰值 RSS | 915,947,520 bytes（873.52 MiB） |
| 自有 cgroup 記憶體峰值 | 733,605,888 bytes（699.62 MiB）；與 RSS 的計量範圍不同 |
| 執行限制 | 2 GiB、swap 0、30 秒、CPU 配額 100%、觀測主執行緒 affinity 為 CPU0 |
| 套件 | Torch 2.13.0+cu130、Accelerate 1.15.0、PEFT 0.21.0、safetensors 0.8.0 |

## 實際成立與不成立的行為

- `safe_open` 關閉後，tensor 數值及指標仍有效；同 CPU／BF16 的 `.to()` 保留原物件。`/proc` mapping 對應自產檔案的正確 inode 與 offsets。這只證明這個小型案例，不能保證完整 checkpoint 的所有載入分支都不複製。
- 實際 Accelerate hooks 保留傳入的外部 state dict。base 與非零 LoRA 各兩次正常 forward 與各自參照完全相同，LoRA 關閉／再開的輸出也符合參照。
- **共用 Parameter 物件身分沒有保留。** 原本 embedding/head 是同一物件；offload 加明確 detach 後成為不同物件，但 CPU storage 仍共用。沒有偷偷 retie。
- **body exception 不會自動完成 post hook。** 刻意觸發錯誤後，projection 留在 CPU，其他參數在 meta；明確移除 hooks 後才全部回到 CPU。原始錯誤與殘留狀態均保留。
- 已保存的 LoRA 參數 `requires_grad=True`；這份快照位於第一次 disable/enable context 之後。程式先前明確設為 false，但缺少該 context 前的配對快照，因此 false→true 屬來源支持的推論。inference-mode meta 參數亦曾無 version counter，不能宣稱完整物件／狀態身分不變。

Torch CUDA context 起訖均未初始化；被記錄的真實 availability 查詢皆為 false／0，自有操作使用 CPU。guard 在 Torch 匯入後才安裝，未監測的匯入區間與 native calls 不在零 CUDA 呼叫保證內，沒有 GPU 傳輸測量。

## 執行與證據範圍

新 unit `jev-d3-mmap-cpu-20261023-03.service` 經過雙 interpreter 身分交接，再進入 Torch。原 owner boundary、Comfy observer 與服務權威來源保持不變；啟動時要求全機 reserve 加 2 GiB，運行時保持原 reserve。192 次運行觀察都執行門檻檢查，只保存前 63 次與最後一次；加啟動／結束共 66 份保存觀察，其可用記憶體最低為 28,244,250,624 bytes。這不是完整時間區間的最低值或全機硬性上限。

所有保存觀察皆為 Comfy HTTP idle。父流程在 5 秒交接期限前 authenticated stop 自有 unit；cgroup 消失、兩次程序掃描皆空，來源前後一致。這證明選定 unit 與觀測到的 cohort 清理，沒有宣稱對同 UID 程序逃逸或所有程序樹的強制隔離。

前兩次執行都在 Torch 匯入前拒絕：第一次錯誤假設使用者 cgroup 有 cpuset 介面；第二次錯誤拒絕 Linux limits 欄位尾端空白。新版使用實際 affinity 雙讀，並在維持精確數字與單位下接受尾端空白。兩次原始失敗、清理與來源紀錄完整保留。Actor 有 9 項 stdlib 回歸，最終 owner 有 16 項，皆通過；真實測量另有獨立保存資料複核。

主要證據：

- [原始 actor receipt](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/bf16-mmap-offload-cpu-owner-20261023-03/actual-01/fixture/receipt.json)
- [owner receipt 與觀測](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/bf16-mmap-offload-cpu-owner-20261023-03/actual-01/receipt.json)
- [actor 來源與選定版本](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/bf16-mmap-offload-cpu-20261023-01/selection.json)
- [先前來源研究](/home/albert_hsieh/Gemma-4-12B-Unified-System-One/docs/research/2026-10-23-bf16-mmap-layer-offload-feasibility.md)

23 個關鍵小型來源使用 SHA，5 個 native binaries 僅核對 stat，並非完整 runtime 資格。自產 tensor 留在本機，Git 只保存產生程式、hash、數值與文字證據。

## 對下一步的影響

值得繼續做真實單層的有界記憶體／時間量測，再決定是否實作完整 offload loader；新 profile 必須處理共用權重與失敗復原，重新驗證狀態。小型 CPU 結果不能外推 GPU／UMA、12B 容量、8192 context、20 秒服務時間或 N3／Q95 資格。原 Q95 初始門檻仍為 73,958,271,732 bytes。

正式結果不變：Benchmark Heaven public231 **195/231（84.42%）**；JevBench coherence MINI **41.26/100**；目前瀏覽器最佳 **4/10**。六款 Decision3 的同條件比較未完成，沒有勝過其他模型的證據。
