# 驗證與發布證據索引

更新日期：2026-10-24（Asia/Taipei）。各日期目錄保存當時的人口、執行來源與收據；後續結果另立紀錄，保留既有封存檔與報告。

| 日期 | 入口 | 範圍 |
| --- | --- | --- |
| 2026-10-24 | [Decision3 兩套 JevBench](2026-10-24/decision3-edge-cpu4-coherence-complete-results.md)、[fresh 瀏覽器比較](2026-10-24/decision3-v7-long-page-timeout-results.md)、[多核心長輸入診斷](2026-10-24/decision3-edge-cpu-long-input-diagnostic-results.md) | Edge 原模型 public231 為 158/231、一致性 70.59/100；operation 候選瀏覽器比較因長頁面逾時未完成。相同自造長輸入的 CPU10／CPU16／CPU5 呼叫約 23.30／24.26／28.3 秒，均未通過原 20 秒期限；尚無競品勝出證據。 |
| 2026-10-22 | [兩套 JevBench](2026-10-22/jevbench-both-fresh-gx10-results.md)、[瀏覽器基準](2026-10-22/jev-current-nf4-raw-complete-results.md)、[V2 對照](2026-10-22/jev-current-nf4-v2-complete-results.md)、[原生訓練](2026-10-22/jev-native-v2-complete-training-results.md)、[Decision3 Edge](2026-10-22/decision3-edge-observed-browser-results.md)、[失效操作回饋](2026-10-22/jev-native-v2-feedback-complete-results.md) | GX10 新推論：public231 為 195/231、一致性 41.26/100；相同 NF4 helper 的十場景基準 2/10、V2 4/10，增加兩題且無退步。142 次訓練與 37 題零更新開發集完成。Edge 十題觀察為 0/10，收尾受檢查工具崩潰回報干擾，完整執行驗證仍未通過；回饋實驗完整核對為 3/10，未採用，V2 維持 4/10。輔助模型設定不同，尚未證明跨模型優勢。 |
| 2026-10-13 | [Mac 本機 Jev 整合](../local-jev-benchmark-20261013/README.md)、[完整報告](../local-jev-benchmark-20261013/report.md) | 原生 S1 決策與同模型文字 helper、固定輸入 Q4/Q8 比較、隔離瀏覽器 fixture 的 0/3 travel 與 0/5 completion 結果；原始失敗、量測界線與公開遮罩收據完整保留。 |
| 2026-10-10 | [本輪後續](2026-10-10/README.md)、[公開發布與整理](2026-10-10/publication.md)、[PR 後續修正](2026-10-10/review-followup.md) | CPU03／CPU04 重播、五份公開證據下載核對、批次上限與預設下載修正、乾淨環境測試及 111.24 MiB 重複 ZIP 清理；實際狀態以該日收據為準。 |
| 2026-10-08 | [八組優化實驗](2026-10-08/breakthrough/README.md)、[完整結果](2026-10-08/breakthrough/results.md) | Jev-Omni 差距、提示與 Head／LoRA 固定對照、完整人口與校驗限制；此封存保存當時六組完整稽核、兩組機率連結未完成的狀態。 |
| 2026-10-08 | [JevBench campaign](2026-10-08/jevbench/README.md)、[campaign audit](2026-10-08/jevbench/campaign-audit.md) | 市面模型的公開 231 題 screening、獨立 coherence 評測及推論實驗；公開分數不代表正式榜單名次。 |
| 2026-10-07 | [GGUF 記憶體優化](2026-10-07/gguf-memory/README.md) | 原發布模型的執行緩衝區量測與有界原生等價檢查，與模型品質評測分開記錄。 |
| 2026-10-07 | [NVFP4 發布驗證](2026-10-07/nvfp4/README.md)、[Jev-Omni 配對比較](2026-10-07/jev-comparison.md) | NVFP4 轉換、獨立校準、完整保留人口及歷史發布收據；另保存固定 128 題與原生媒體、HTTP 負載的模型比較。 |
| 2026-10-04 | [BF16／MLX 發布](2026-10-04/README.md)、[GGUF 發布](2026-10-04/gguf/README.md) | 公開模型檔案與 immutable revisions 的驗證；GGUF 保存自己的轉換、原生 S1 執行、校準與評測證據。 |
| 2026-10-02 | [MPS／MLX 與訓練驗證](2026-10-02/README.md)、[訓練版完整評測](2026-10-02/release/summary/README.md) | 真實 checkpoint 的本機 profiling，以及 BoolQ 訓練版在獨立校準、256 題 fresh test 與原生媒體上的結果；不同工作負載保留各自人口。 |
| 2026-10-01 | [Live validation](2026-10-01/live-validation.md) | 基線、checkpoint、訓練、完整性與服務測試的實際執行紀錄；執行完成與品質、SLO、正式接受分別判定。 |
| 2026-09-30 | [實作驗證](2026-09-30/README.md)、[PR #2 修正](2026-09-30/pr2-review-fixes.md) | 多模態推論與訓練 smoke、API／評測契約及 review 修正；小型 fixture 不視為代表性品質 benchmark。 |

八組實驗的完整證據重組方式見 [分片與校驗說明](2026-10-08/breakthrough/evidence-download.md)。各日報告保留實際的 `not_run`、失敗與未完成項目；新收據不改寫歷史狀態。
