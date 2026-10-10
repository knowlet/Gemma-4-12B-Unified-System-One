# 證據下載與檔案對照

本次研究預覽版的入口為 [GitHub Release](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/tag/research-2026-10-10-jevbench)。下載後先依 manifest 核對大小與 SHA256，再解壓；模型品質、算術校驗與發布狀態分別見 [本日紀錄](README.md)。

| 下載檔 | 內容 |
| --- | --- |
| [newcloud-evidence.zip](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.zip) | 10 月 9 日封存的完整八組實驗、來源、原始回答及報告；26,135 members。 |
| [newcloud-evidence.manifest.json](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/newcloud-evidence.manifest.json) | 上述 ZIP 每檔大小、SHA256、CRC 與來源範圍。 |
| [cpu-replay-evidence-20261010.zip](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/cpu-replay-evidence-20261010.zip) | 本日 CPU 回傳失敗、分段重播、獨立核對、本機測試與完整結果稽核；實際完成狀態以其中收據為準。 |
| [cpu-replay-evidence-20261010.manifest.json](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/releases/download/research-2026-10-10-jevbench/cpu-replay-evidence-20261010.manifest.json) | 本日補充 ZIP 的逐檔大小、SHA256 與 CRC；列出省略的可重組輸入 ZIP。 |

第一個 ZIP 的 SHA256 為 `53cc9767dbffefd29170399e3de9ec538cbfd71d86d3e5ed4251d0c48ae53218`，116,641,459 bytes。Git 內亦保留兩份分片，依 [重組說明](../2026-10-08/breakthrough/evidence-download.md) 可逐位元恢復同一 ZIP。

## 封存報告中的本機路徑

舊報告保持原始位元組，其中 `artifacts/…` 連結表示當時 checkout 的本機路徑。這些檔案已在 ZIP 內，GitHub 網頁無法直接透過那些本機連結讀取。以下對照可找到實際資料：

| 舊本機路徑 | `newcloud-evidence.zip` 內位置 |
| --- | --- |
| `artifacts/jevbench/breakthrough-20261008/<file>` | `research/<file>` |
| `artifacts/jevbench/breakthrough-20261008/runs/20261008-breakthrough-01/ablate/<file>` | `research/runs/20261008-breakthrough-01/ablate/<file>` |
| `artifacts/jevbench/breakthrough-20261008/runs/20261008-breakthrough-01/train-recovered/<file>` | `research/runs/20261008-breakthrough-01/train-recovered/<file>` |
| `configs/experiments/jevbench-breakthrough-20261008.json` | `preflight/configs/experiments/jevbench-breakthrough-20261008.json` |
| 封存時的研究 README／results | `reports/README.md`／`reports/results.md` |

例如嚴格 Mac 稽核在 `research/final-eight-profile-audit-v2.json`，原始執行狀態在 `research/execution-status.json`，當時 CPU03 上傳尚未開始的拒絕紀錄在 `research/cpu-arithmetic-03-approval-rejection-full.json`。本日授權、實際啟動及後續結果另外保存在補充 ZIP，不改寫先前紀錄。

補充 ZIP 使用 `cpu-arithmetic-03/`、`cpu-arithmetic-04/`、`execution/`、`outcome/`、`reports/` 目錄。兩次使用的相同輸入快照仍由 20,142 檔的 input manifest 綁定；ZIP 內包含 verifier、origin、平台設定、完整輸出與核對紀錄。約 24GB 的 HF base 權重未放入這些研究證據包，依原報告的 immutable revision 取得。
