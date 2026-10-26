# 不讀內容的均勻機率對照：coherence 89.41分

固定輸出均勻機率的對照，在原官方 JevBench coherence MINI 實測得到 **89.4118/100**；graded coherence 為 **98.3626/100**。對照沒有使用模型，回答函式忽略 state 與指令內容，只依問題類型與選項數輸出 Choice1/K、Score1/L、Noul0.5。

| 原官方維度 | 分數（0–100） |
| --- | ---: |
| REP 表示一致性 | 100 |
| BAT 批次隔離 | 100 |
| MEA 機率測度約束 | 47.0588 |
| LOG 邏輯一致性 | 100 |
| CHO 條件選擇一致性 | 100 |

這個實驗具體說明：高 coherence 分數本身不足以證明內容理解或瀏覽器操作能力。它不代表某個模型只輸出均勻分布，也尚未解決全部真實場景失敗原因。這個對照沒有 accuracy／瀏覽器分數，不列模型排名，不宣稱擊敗其他模型。

原始1248次不同請求、240 cases、50 relations、1200 planned tests全部完成；官方輸出1468 outcomes，無 skipped outcomes／incomplete relations／errors。使用全新的記憶體快取，保留原seed0、trials1、repeats1、concurrency1、minimize=False、tolerance0.05、min_n5、bootstrap1000與min_coverage0.95，沒有更換題目、關係或評分方式。完整數值見[JSON](jevbench-uniform-control-results.json)。

所有12072個回答 head 都是相同機率平手。本實驗沒有測量 first-argmax 的操作正確性；不能把平手後選到的第一項當成語義能力。

原模型工作階段結束並確認原始清理後，五項假資料測試、AST、Ruff、格式與預設不計算入口通過。對照在本機單核心、512MiB、CPU30秒／wall45秒限制中以7.555秒、exit0完成；這不是模型推論速度。主程序再核對原官方78個來源與2個資料集metadata、請求範圍及彙總算術。原始結果檔保留原狀，另有[限於指標診斷的封閉審查](../../../artifacts/jev-runtime/decision3-uniform-metric-control-20261024-01/root-closed-control-review-01.json)。未宣稱第二位審查者直接讀取原件；獨立範圍限於來源與提供數字的一致性。

測試固定coherence revision e18733694623aa93058e279c3534f8b8e2edefa3。沒有載入模型、瀏覽器、外部服務或外部網路。公開內容僅含來源、合成測試、typed aggregates與雜湊參照；不公開題目、gold、原始回應、case IDs、範例、權重或特徵。下一步的模型選擇與改善仍以正確決策、完整時限和真正瀏覽器任務驗收。
