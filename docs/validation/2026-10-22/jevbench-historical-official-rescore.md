# JevBench 兩套評測：歷史回答的官方重算

2026-10-22 已實際執行兩套官方評分器，完整重算 2026-10-08 的 A100 回答。這次重算耗時 12.97 秒，沒有載入模型或執行新的 GX10 推論。

模型固定為 `knowlet/Gemma-4-12B-Unified-System-One@a66f836b56605039fe040f330180e336d19b3362`，BF16 與發布校準設定；兩個 execution profile 分開報告。

| 歷史 profile | Benchmark Heaven 公開題準確率 | JevBench coherence mini，0–100 | Coherence 95% CI |
| --- | ---: | ---: | ---: |
| 預設 causal multislot | 197/231＝85.28% | 40.50 | 37.73–43.11 |
| independent development profile | 197/231＝85.28% | 71.96 | 69.27–74.33 |

兩組公開題均為完整 231 題、231 個有效回答、strict schema validity 100%，沒有重新正規化。ECE 均為 0.0484954，Brier 均為 0.2295357。逐題正誤完全一致。Choice 為 120/139，Noul 為 62/74，Score 為 15/18；這些是公開題 screening 的結果。

Coherence 依官方規則先算 relation 分數，再對五個 dimension 等權平均；不是 pooled accuracy。每個 profile 完整回放 1,248 個原始 request，涵蓋 240 cases、50 checks、1,200 planned tests，errors 為 0。另逐項精確比對全部 1,468 個 outcome rows；1,468 不能當 overall 分母。區間採官方 1,000 次 case bootstrap。

| Coherence dimension，0–100 | causal | independent |
| --- | ---: | ---: |
| BAT：批次獨立性 | 38.33 | 96.67 |
| CHO：選項集合一致性 | 20.83 | 66.67 |
| LOG：邏輯一致性 | 69.58 | 85.42 |
| MEA：機率測度一致性 | 15.44 | 37.99 |
| REP：表述一致性 | 58.33 | 73.06 |

這些數字支持優先檢查批次情境下的輸出穩定性：公開單題準確率相同，批次一致性卻明顯不同。Independent 是觀察 causal 問題後的 development profile，不能稱為未接觸評測資料的獨立測試，也不能把 71.96 當作發布預設的分數。兩組原始硬體分別是 A100-SXM4-80GB 和 A100 80GB PCIe，來源紀錄也不同；需要相同 GX10 環境的對照，才能隔離差異來源。

官方來源與重現範圍：

- Benchmark Heaven scorer 固定 [bb05a335](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc)。當日 current revision `609a664e81f46e03731600bd7bd9f1b7a61257c5` 的六份 scorer source 和三份公開 task 檔案，已逐檔核對完全相同；仍是 legacy public231。官方 v1.6.1 的 P300＋S1200 是另一個 population，這次沒有取得或執行完整 release draw，不能換算 [Capability、Composite 或正式排名](https://benchmarkheaven.com/jev-models/methodology)。
- Coherence 直接使用 [JevBench e187336](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) 的官方 SDK 與 mini suite。完整 package、suite 和原始 cache 已核對，未修改 scorer 或容忍值。
- 原始 archive 的 8,219 個 members 已完整驗證 SHA、size、順序與 CRC。重播所有 deterministic outcomes 與原始 probabilities；aggregate float tolerance 為 `1e-12`。原始 causal run01 只記錄 public runner hash，沒有完整 package／campaign execution inventory；本次重播不能補強當年未記錄的執行來源。

原始 requests、answers、cache、完整逐題 audit 和權重留在本機。發布的[彙總 JSON](jevbench-historical-official-rescore.json)只含分數、計數、範圍及 hash references；沒有新的 latency、成本、sealed score 或 browser completion 主張。瀏覽器歷史完成率仍為 5/10，須另以真實場景重新驗證。
