# Decision3 Edge：GX10 CPU4 兩套 JevBench 結果

Benchmark Heaven public231 的完整準確度為 **158/231（68.3983%）**。JevBench coherence MINI 只完成 **1234/1248** 次請求，沒有合格的完整分數。整輪因原定 10000 秒上限結束，1479 次呼叫完成1465次；原命令在10000.291秒以124結束；root exit1、owner.completed=false 和失敗紀錄保留。

| 分開報告的評測 | Decision3 Edge CPU4 本輪 | 先前 Gemma BF16 GPU |
| --- | ---: | ---: |
| public231 準確度 | 68.3983%（158/231） | 84.4156%（195/231） |
| public231 Brier（低較好） | 0.4402665 | 0.2286229 |
| public231 ECE（低較好） | 0.1039798 | 0.0559900 |
| coherence MINI（0–100） | 未完成，無分 | 41.2631 |

這輪 public231 準確度低16.0173個百分點，尚未超過 Gemma。兩個硬體與模型設定不同，不構成公平的速度勝負或綜合排名。先前 Gemma 的來源與精確數值見[既有報告](../2026-10-22/jevbench-both-fresh-gx10-results.md)。

本輪使用固定 vllm-sr/d3-edge revision cab71e6ad3461b8086d51ddb572c656babd63b0b，CPU16–19／四執行緒、2GiB、swap0、BF16 backbone、FP32 原 readout、temperature1、最多batch4、context16384超長即拒絕、不截斷，沒有 permutation averaging 或訓練。兩套官方評分來源保持原固定版本，沒有修改題目、選項、分數或時間限制。

完整 public231 階段在失敗前完成了全部呼叫、模型／來源／執行身分認證與耐久紀錄；同一 invocation 的第一個 coherence callback 提供其最後交接已返回的正向證據。主程序在原始終局之後核對 exact retry04 啟動、59個選定來源、134個模型／runtime／owner／評測紀錄、原public彙總及原本的正向 authenticated_stop 清理。僅保留 public231 的規則已在前置來源中明定；沒有將失敗整輪改成成功，也沒有以之後的空白程序掃描修補原旗標。

public231 完整 callback 中位數0.323260秒、p95 5.028488秒、最大7.281688秒。這個範圍不含模型啟動、HTTP或瀏覽器，不能用來認證瀏覽器的20秒決策限制。觀測到的整輪自身記憶體峰值為2144661504 bytes；整輪未完成，不宣稱完整容量合格。

終局審查程式的 AST、Ruff、格式化與預設不讀實際結果的入口均通過。格式化前後 AST 相同；適用的 public-only 審查在單核心／512MiB／CPU30秒上限中以2.153秒、exit0完成。完整1479審查沒有執行，因本輪不符合其前提。

第二個審查程序的原件唯讀核對被自動審查拒絕兩次，理由是仍套用先前 source-only 限制。它沒有讀到原件；其獨立範圍僅為來源審查與主程序提供數字的一致性確認。本報告不宣稱完成獨立原件驗證。主程序原件審查的[純彙總紀錄](../../../artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-06/root-public-stage-only-actual04-result-review-01.json)保留原始證據的路徑、大小與SHA，實際題目、gold、回應、DOM、權重、特徵與錯誤全文不公開。精確數值另見[JSON](decision3-edge-cpu4-jevbench-results.json)。

接下來依序驗證 CPU16／4GiB 的新設定、量測真實原模型推論，取得其獨立容量與清理紀錄，再執行兩套評測與瀏覽器整合。均勻機率對照僅用於解釋 coherence 指標範圍，不算模型能力、瀏覽器成績或競品勝利。目標仍在進行中。
