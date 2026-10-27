# DONE 覆核路由：單案例配對診斷

日期：2026-10-27。這次在先前已使用的單一 development 案例上，對照啟用 DONE 覆核的 Decision3 semantic 候選與原 NF4 baseline，沿用原 judge。新 run 保存的 runner judge／outcome 彙總顯示兩臂皆為 **0/1**；尚未獨立重跑 DOM judge。這不是未見案例測試，也不是完整評測或排行榜成績。

| 執行臂 | 原 judge | 動作 | Policy 呼叫 | Text 呼叫 | 結束原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| Decision3 semantic + DONE 覆核 | 0/1 | 2 | 5 | 1 | `public_navigation_scope` 阻擋 |
| NF4 baseline | 0/1 | 5 | 16 | 1 | `decision_budget` 耗盡 |

候選端觀察到 1 次覆核與 1 次排除 DONE 後的重選；兩次呼叫已完成事件、runner、client capture 與服務綁定的對帳。覆核問題使用 Decision3 原生 `completion_evidence` head，operation 與 click-target 兩個 semantic 增量頭均未套用在該問題上。模型的覆核結論本身不是任務成功證據。

獨立 CPU FP64 重播核對 5/5 同次 semantic 見證、capture bytes 與 binding，結果與 actor 的數值彙總一致。共觀察 6 筆 capture，其中 1 筆未配對，歸屬及 capture 是否涵蓋所有 transport 嘗試均未知；事件對帳也只涵蓋有記錄的覆核及重選。此紀錄不推論兩臂耗時具有公平可比性。兩臂都未通過此案例，現有證據不足以宣稱候選打贏 baseline 或其他模型。

獨立稽核另核對執行計畫的 22/22 覆核相關來源、候選計畫的 59/59 來源及 actor／supervisor 釋出版本雜湊。兩個受監督單元均正常結束，服務回報 CUDA 配置歸零。執行計畫 SHA-256 為 `410198a185324d357847360ceb114e77de86fed4c1d6209925f080a284bf00e6`。
