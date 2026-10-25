# 驗證與發布證據索引

更新日期：2026-10-25（Asia/Taipei）。各日期目錄保存當時的人口、執行來源與收據；後續結果另立紀錄，保留既有封存檔與報告。

| 日期 | 入口 | 範圍 |
| --- | --- | --- |
| 2026-10-25 | [原生 GPU 完整 coherence MINI](2026-10-25/decision3-native-coherence-gpu-results.md) | 原始 Edge 一致性 69.64／100，完整 240 案例、50 關係、1,200 檢查。已核對的 921 筆 GPU 前段結果接續 327 次新增呼叫，原請求共 1,248 個；完整快取重播與獨立計分均通過。與 Benchmark Heaven 159／231（68.83%）分開報分；兩段計時分列，候選完整瀏覽器分數仍為 null，未建立競品勝出。 |
| 2026-10-25 | [原生 GPU 完整 Benchmark Heaven](2026-10-25/decision3-native-heaven-gpu-results.md) | 原始 Edge 在 GX10 GPU 完整作答 231 題，159／231（68.83%），格式有效 231／231；載入後完整 predict 的 p50／p95 約 0.386／0.523 秒。獨立核對全部原請求、回傳、計分、775 個原生機率純量及範圍收尾。未載入 operation delta；coherence 另見上列接續紀錄，候選完整瀏覽器比較仍為 null，尚未建立競品勝出。 |
| 2026-10-25 | [原生 typed GPU 實測](2026-10-25/decision3-native-typed-gpu-results.md) | 原始 Edge 載入一次，兩次原生 Choice／Noul／Score 呼叫涵蓋 12 個問題；混合九問題實際批次 8＋1，長輸入完整保留 5117／5118／5122 tokens。獨立核對 284 個回傳機率純量及原所選範圍收尾。沒有載入 operation delta 或執行瀏覽器／兩套 JevBench，未新增品質分數、SLO 或競品勝出結論；先前中止的完整比較分數仍為 null。 |
| 2026-10-25 | [可變操作選項收集與長輸入後續](2026-10-25/decision3-affordance-collection-results.md)、[特徵獨立重播](2026-10-25/decision3-affordance-feature-replay-results.md)、[109 筆合併訓練](2026-10-25/decision3-affordance-combined-fit-results.md)、[GX10 原生 GPU 短／長輸入](2026-10-25/decision3-gx10-native-gpu-qualification-results.md)、[新候選 GPU 數值驗證](2026-10-25/decision3-affordance-operation-gpu-numerical-results.md)、[GPU 排程與服務準備](2026-10-25/decision3-gpu-forward-lease-preparation-results.md)、[完整十場景 GPU 中止紀錄](2026-10-25/decision3-full10-gpu-interrupted-results.md) | 新 delta 已載入 GPU；16 次呼叫涵蓋零差值控制、13 筆完整訓練輸入與兩筆結構控制。獨立核對 1,324 個機率、100 次最高機率選項及 35 個完整保留的非動作回答，包含 1–4 個問題及 255 選項邊界。13 筆候選呼叫約 0.66–0.84 秒，發生在首次控制呼叫之後；尚非瀏覽器成功率或穩定延遲。GPU 共用排程的 22 項結構控制及長時間 owner 的短控制已完成。完整比較已啟動，因新的 ComfyUI 工作中止；基線完成 9 筆基礎設施紀錄，其中 4 筆成功，第 10 筆中斷，候選服務尚未啟動，不能記為 0/10。中止後範圍清理通過獨立稽核；完整比較與兩套 JevBench 新候選分數仍為 null，尚無競品勝出結論。先前原始 Edge 4579-token GPU 與 CPU10 長輸入診斷另行保留。 |
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
