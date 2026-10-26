# Decision3 Edge：原生 typed GPU 呼叫實測

日期：2026-10-25（Asia/Taipei）。固定版本的原始 Edge 在 GX10 GPU 載入一次，完成兩次未修改代數的原生 typed 呼叫，涵蓋 Choice／Noul／Score、12 個問題。獨立稽核接受此原生 API 診斷；本次沒有經過瀏覽器 DTO、載入 operation delta，或執行瀏覽器與兩套 JevBench 任務。

| 呼叫 | 完整輸入與實際批次 | 全呼叫時間 |
| --- | --- | --- |
| 混合九問題 | 選項數依原順序為 `[255,2,2,1,2,10,3,2,4]`；實際批次 8＋1，保留尾批；各問題 113–2157 tokens | 2.9819842079887167 秒 |
| 長輸入三問題 | 三種型別各一題；完整長度 5117／5118／5122 tokens；實際批次 3 | 1.245128178037703 秒 |

模型載入為 **3.384253732045181 秒**；原始啟動程序以 exit code 0 結束，整次為 **23.00653588300338 秒**。第二次呼叫發生在第一次之後，屬熱態觀測；兩個輸入不同，不能據此計算對照增益或宣稱穩定延遲、SLO。這也不驗證完整 16k context。

稽核獨立核對 284 個原生回傳的機率純量，每個 Noul 回答只計一個純量。完整 payload、問題及 criteria 順序保留，原始 W0 未改；沒有重新命名問題後套用 delta，也沒有 FP64 數值重播或精度等價聲明。先前 [109 筆候選的 GPU 數值驗證](decision3-affordance-operation-gpu-numerical-results.md)保留自己的證據範圍，本次不繼承其候選資格。

每次呼叫仍受原 **20 秒協作期限**約束，包含來源驗證、雜湊、CUDA fence、留證與 hook 清理；整體 120 秒及 owner command 113 秒上限保持原值。執行中反覆核對來源與原 owner 綁定，實際實體 GPU 符合所選裝置，原 child 有兩次穩定身分樣本。原所選範圍的收尾通過，當前單位、cgroup 與 child 均已不存在；不宣稱禁止程序逸出或完整 DSO 相依閉包。

此程序的 CUDA tensor allocator 峰值為 **3,311,346,688 bytes**，收尾配置量為 **0 bytes**。這只描述本程序的 allocator，不能代表整台機器的 GPU 或共享記憶體使用量歸零。執行前另有 19 項自造控制通過，耗時 0.447 秒，並通過 Ruff 與來源同儕審查；這些控制與實際模型呼叫分開記錄。

| 本次品質或評測結果 | 分數 |
| --- | --- |
| 瀏覽器完整十場景比較 | null |
| Benchmark Heaven 新分數 | null |
| JevBench coherence 新分數 | null |
| 官方排名 | 無 |

本次沒有競品勝出或模型品質結論。[先前中止的完整比較](decision3-full10-gpu-interrupted-results.md)仍是基線 9 筆部分紀錄、其中 4 筆成功，候選未執行；其完整分數仍為 null，不能記為候選 0/10。

獨立稽核狀態為 `ROOT_ACCEPTED_NATIVE_TYPED_GPU_DIAGNOSTIC_ONLY`，紀錄大小 3,119 bytes，SHA256：`bf6f79fd136629d0c0ff09bfe1a0bce79ed4697cb0b0e551a792a3885dd49287`。聚合欄位見[機器可讀紀錄](decision3-native-typed-gpu-results.json)，方法進度見[研究紀錄](../../research/jev-operation-only-readout.md)。公開內容不包含原始請求、預測、標籤、程序／裝置識別資訊或私人產物路徑。
