# Decision3 scroll-state：首案例配對結果

日期：2026-10-27

新 equal candidate（273 筆 operation、65 筆 click_target 訓練）與 NF4 baseline 在同一輪、相同原始案例中，依 candidate → baseline 順序各執行一次，兩者皆為 **0/1**。

| 分支 | Policy calls | Text calls | Actions | 結果 |
| --- | ---: | ---: | ---: | --- |
| 新 equal candidate | 4 | 4 | 4 | blocked，任務未完成 |
| NF4 baseline | 5 | 1 | 4 | 過早 DONE，目標未滿足 |

Candidate 四次都對同一欄位執行 TYPE_TEXT，且四次輸入內容相同；第一次 `page_changed=True`，後三次皆為 `False`。逐步觀察顯示，第一次輸入後該欄位的可見值已與後續重複輸入內容相同。模型的原生請求也包含非空的欄位值與逐步增加的近期操作歷史，因此這次重複輸入無法歸因於欄位值在傳送途中遺失。Baseline 依序執行 CLICK／SELECT／TYPE_TEXT／CLICK，隨後在目標尚未滿足時回報 DONE。兩個分支均無執行錯誤。

首案例的獨立審核已接受，報告 SHA256 為 `f1477637165af62a656a54ee3a88d582cdb6d45c2a8a074f48bc95244cb4c057`。原生 4 次呼叫涵蓋 16 個 heads、144 個 legal positions，FP64 重算最大誤差為 `2.6725622070955524e-7`；兩次 API 正規化後結果均未改變。兩個原始 scope 的關閉確認均成功，所選 CUDA allocator 的終止值為 0。完整配對的原始工作階段以 exit code 0 終止，經過時間為 252.095 秒。先前的完整 GPU 數值驗證另見[兩批 GPU 測試進度](decision3-scroll-state-gpu-two-shard-progress.md)。

同一案例的舊 225-row candidate 曾連續重複 CLICK 四次，且沒有 text call；新 candidate 則連續重複輸入四次，仍未完成任務。這只是單一、曾用於模型選擇的 DEV 案例，沒有新未見案例的泛化、完整十案例表現或優於其他模型的證據。

新 candidate 尚未執行 JevBench coherence 與 Benchmark Heaven 兩套評測，兩套均無分數。
