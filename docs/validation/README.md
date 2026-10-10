# 驗證與發布證據索引

更新日期：2026-10-10（Asia/Taipei）。各日期目錄保存當時的人口、執行來源與收據；後續結果另立紀錄，保留既有封存檔與報告。

| 日期 | 入口 | 範圍 |
| --- | --- | --- |
| 2026-10-10 | [本輪後續](2026-10-10/README.md)、[公開發布與整理](2026-10-10/publication.md) | CPU03／CPU04 重播、五份公開證據下載核對、乾淨環境測試修正與 111.24 MiB 重複 ZIP 清理；實際狀態以該日收據為準。 |
| 2026-10-08 | [八組優化實驗](2026-10-08/breakthrough/README.md)、[完整結果](2026-10-08/breakthrough/results.md) | Jev-Omni 差距、提示與 Head／LoRA 固定對照、完整人口與校驗限制；此封存保存當時六組完整稽核、兩組機率連結未完成的狀態。 |
| 2026-10-08 | [JevBench campaign](2026-10-08/jevbench/README.md)、[campaign audit](2026-10-08/jevbench/campaign-audit.md) | 市面模型的公開 231 題 screening、獨立 coherence 評測及推論實驗；公開分數不代表正式榜單名次。 |
| 2026-10-07 | [GGUF 記憶體優化](2026-10-07/gguf-memory/README.md) | 原發布模型的執行緩衝區量測與有界原生等價檢查，與模型品質評測分開記錄。 |
| 2026-10-07 | [NVFP4 發布驗證](2026-10-07/nvfp4/README.md)、[Jev-Omni 配對比較](2026-10-07/jev-comparison.md) | NVFP4 轉換、獨立校準、完整保留人口及歷史發布收據；另保存固定 128 題與原生媒體、HTTP 負載的模型比較。 |
| 2026-10-04 | [BF16／MLX 發布](2026-10-04/README.md)、[GGUF 發布](2026-10-04/gguf/README.md) | 公開模型檔案與 immutable revisions 的驗證；GGUF 保存自己的轉換、原生 S1 執行、校準與評測證據。 |
| 2026-10-02 | [MPS／MLX 與訓練驗證](2026-10-02/README.md)、[訓練版完整評測](2026-10-02/release/summary/README.md) | 真實 checkpoint 的本機 profiling，以及 BoolQ 訓練版在獨立校準、256 題 fresh test 與原生媒體上的結果；不同工作負載保留各自人口。 |
| 2026-10-01 | [Live validation](2026-10-01/live-validation.md) | 基線、checkpoint、訓練、完整性與服務測試的實際執行紀錄；執行完成與品質、SLO、正式接受分別判定。 |
| 2026-09-30 | [實作驗證](2026-09-30/README.md)、[PR #2 修正](2026-09-30/pr2-review-fixes.md) | 多模態推論與訓練 smoke、API／評測契約及 review 修正；小型 fixture 不視為代表性品質 benchmark。 |

八組實驗的完整證據重組方式見 [分片與校驗說明](2026-10-08/breakthrough/evidence-download.md)。各日報告保留實際的 `not_run`、失敗與未完成項目；新收據不改寫歷史狀態。
