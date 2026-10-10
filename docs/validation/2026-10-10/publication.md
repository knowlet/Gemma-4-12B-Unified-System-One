# 公開發布與檔案整理收據

日期：2026-10-10（Asia/Taipei）。[研究預覽版](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/tag/research-2026-10-10-jevbench)已公開，tag 綁定 `8c94514d7d0c341f7c4a9290c2ceac921c0cee8e`；[PR #7](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/pull/7)保存後續測試相容性修正及本次整理紀錄。研究 snapshot、舊來源、報告與 ZIP 不改寫。

## 公開下載核對

五份 Release assets 已全部以未登入方式下載，完整大小、SHA256 與 GitHub 的 digest 相符。兩個 ZIP 均完成 CRC 核對，原始證據包含 26,135 個檔案，CPU 補充證據包含 36 個檔案。所有驗證下載暫存已移除。

| 紀錄 | 內容 |
| --- | --- |
| [完整發布收據](publication.json) | 五份實際下載、HTTP 狀態、完整 SHA256、ZIP CRC、tag／asset 身分；保留續傳與失敗分段下載的執行邊界。 |
| [獨立公開核對](publication-independent-verification.json) | 13 項核對通過：公開 API、tag、五個 server digests 及小型檔案的匿名實際下載。 |
| [ZIP 發布前獨立核對](cpu-replay-evidence-20261010.verification.json) | 112 項本機核對通過，包含完整逐檔來源、CRC、人口與原始測試紀錄；其「尚未發布」欄位是當時狀態。 |
| [下載與 ZIP 內路徑](evidence-layout.md) | Release 檔案與舊本機 `artifacts/…` 路徑的對照。 |

公開結果維持 **529,136 個回答機率逐值一致、兩項校準 metadata 差異、六組完整稽核／兩組未完成**。本次是研究工具與證據發布；正式模型維持原版，sealed evaluation 未執行、排行榜未提交。

## 乾淨環境的測試修正

發布後 GitHub 檢查重現兩項測試環境問題：24 個 arithmetic 用例需要可選的 Torch，另有兩個 analyzer failure-state 測試讀取 Git 忽略的本機資料。修正只涉及兩份測試：在沒有 Torch 時跳過依賴它的用例，保留 40 個非 Torch 用例；用獨立、綁定 SHA 的暫存輸入驗證八組 `not_run`、失敗與來源遭修改的狀態。

完整環境針對性測試 **144 passed**；真正 API-only、沒有研究 artifacts 的乾淨環境 **67 passed、77 skipped**。Ruff、格式與 diff check 通過，命令、原始摘要及受測檔案身分見 [測試修正收據](ci-portability.json)。先前 **1,712 passed、3 skipped、1 個既有 warning** 仍屬封存之 `0aa2b86` 來源的完整本機驗證，不改寫成新的測試執行。

修正提交 `a6372a91bf91bfc001aa1ceb629baca191f1aeb2` 的 [push CI](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/actions/runs/38015530900) 與 [PR CI](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/actions/runs/38015534168) 各有 contracts（Python 3.11／3.12／3.13）和 inference 四項成功，共八項。具體 job IDs 與完成時間見 [CI 收據](ci-source-fix.json)。最後文件提交的狀態另由 PR #7 的檢查頁記錄；外部 skipped／neutral 不視為完成審查。

## 已完成的整理

[清理收據](cleanup.json)記錄唯一刪除項目：Git 忽略的重複完整 `newcloud-evidence.zip`，釋出 **116,641,459 bytes（111.24 MiB）**。刪除前已核對公開完整下載，並重新串接保留的分片驗證大小與 SHA256 相同；兩份 Git 分片、manifest 及[重組說明](../2026-10-08/breakthrough/evidence-download.md)皆保留，可恢復相同 ZIP。

模型與各格式匯出、原始資料、GPU stage 目錄、CPU03／CPU04 的輸入與失敗／結果、歷史封存、環境與 build 套件均保留。[日期索引](../README.md)提供統一入口；本日新增收據與舊報告分開保存。
