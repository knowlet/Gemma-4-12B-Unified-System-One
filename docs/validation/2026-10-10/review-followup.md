# 發布後 PR 修正與驗證

日期：2026-10-10（Asia/Taipei）。[PR #7](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/7)新增三項已重現的修正，來源提交為 `1900d7e024463910531c2bffb2d95c834a630f01` 與 `5f9efb71f6350f76b53ed58748eb7216325e9770`。已公開的研究 tag `research-2026-10-10-jevbench` 仍綁定 `8c94514d7d0c341f7c4a9290c2ceac921c0cee8e`；五份原始 Release assets、ZIP 與歷史報告維持相同內容。

## 批次記憶體與評測身分

在實際 `/decide/batch` 邊界，合法的 16 個請求、每個 64 題，原本會讓 independent 模式形成單次 1,024 rows 的 backbone forward，並同時保留 1,024 份 prepared tensors。修正將全域待處理提示及每次相容 forward 均限制為 **8**。1,024 種不同 media shapes 也受全域上限約束；每個 causal 請求仍保留完整多題提示與順序。輸出的請求、題目順序及原生 candidate／full-vocabulary readout 均有邊界回歸核對。

Gemma backend metadata 現在保存已載入模型實際的 `compile_mode`、`max_forward_batch_size` 與 `max_prepared_prompts`。測試透過真正 benchmark evaluator 和 report writer 核對 JSON 內容，避免以要求的模式代替實際生效的模式。修改前五個邊界用例失敗；修改後針對性測試 **357 passed、2 skipped、1 個既有 warning**，API-only 環境 **252 passed、1 module skipped、1 個既有 warning**。後者實際沒有 Torch 或 Modal。

Compile wrapper 回歸驗證模式分派與原生契約，未執行真實 compiler 數值或硬體效能量測。批次上限是受控資源界線，沒有將 CPU 小模型測試解讀成 12B GPU 記憶體或效能 benchmark。

## 預設證據下載

原預設入口在 GPU stage 結束後呼叫遠端單次 ZIP export；已有兩次 600 秒 timeout 的失敗證據。現在預設入口直接使用既有、有界的本機 Volume downloader，最多 16 份並行讀取，逐檔核對 manifest、SHA256 及來源收據。下載發生錯誤時保留部分輸出與失敗紀錄；GPU stage 已失敗時，完成可取得的證據下載後仍重新拋出原始錯誤。

回歸使用實際預設入口、既有 downloader 和 207／210-member 輸入，核對成功、訓練權重保留、原 stage 錯誤、下載失敗及雙重失敗的例外關係。真正 Modal 1.6.0 SDK 的同步 CLI／entrypoint 與 synchronizer 邊界亦有離線測試，只有 provider context 與 I/O 被替代；確認使用者入口可呼叫本機 async downloader。針對性測試 **48 passed**；API-only 環境 **33 passed、15 skipped**，六個預設入口回歸仍執行。

這次沒有新增 GPU 訓練或重新執行雲端 export；上述離線驗證不代表新的一次雲端 experiment 已完成。

## 證據與發布狀態

逐檔 SHA256、實際命令與後續完整驗證結果見 [修正收據](review-followup.json)。公開下載、tag／asset 身分及可恢復清理見 [發布與整理收據](publication.md)。品質結論維持六組完整稽核、兩組校準 metadata 未完成；模型未升級，sealed evaluation 與正式榜單提交未執行。

## 完整本機與來源 CI

提交 `5f9efb71f6350f76b53ed58748eb7216325e9770` 的單次完整本機驗證為 **1735 passed、3 skipped、1 個既有 warning**。Ruff、格式與 diff check 全部成功；16 份相關來源的執行前後 SHA256 相同。最初沙盒執行有兩項本機 HTTP 測試在 socket bind 時因 PermissionError 被擋住（1,733 passed、2 failed）；完整紀錄保留，未修改來源即重新驗證。原始輸出另存於 [驗證日誌](review-followup-logs/)，受阻的完整 pytest 輸出以 gzip 無損保存；預設 export 的針對性日誌明示由原工具輸出事後保存，未假裝成新的測試執行。

來源 push 與 PR CI 的八個 repository jobs 全部成功，包含各自的 contracts（Python 3.11／3.12／3.13）及 inference。此收據記錄受測來源提交；後續只整理文件的最終 head 檢查由 [PR 檢查頁](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/7/checks)提供。外部 skipped／neutral 不視為完成審查。
