# Decision3：雙頭候選通用路徑的公開 Heaven 評測

日期：2026-10-27。在 GX10 使用全新結果目錄執行固定公開 Heaven `legacy_public231` 的全部 231 題，每題一次、沒有重試或快取重用。這些題目的 head 都不是瀏覽器的 `operation` 或 `click_target`，因此本次明確繞過兩個瀏覽器增量，實際測得的是同一個 Decision3 Edge 原生通用路徑，不是增量頭對瀏覽器或其他 head 的能力證明。

| 指標 | 這次 | [前次原生模型](../2026-10-25/decision3-native-heaven-gpu-results.md) |
| --- | ---: | ---: |
| 原 scorer 正確／完整題數 | 159／231 | 159／231 |
| Accuracy | 68.83% | 68.83% |
| Brier | 0.4397679 | 0.4397679 |
| ECE | 0.1133564 | 0.1133564 |
| 序數 MAE | 0.4533887 | 0.4533887 |
| 嚴格有效格式／實際題數 | 231／231 | 231／231 |
| 完整受保護 callback p50／p95 | 0.112／0.255 秒 | 0.386／0.523 秒 |
| 每千次成本 | 未知 | 未知 |

本次有 231 次模型呼叫，沒有無效回應或失敗題。p50／p95 包含模型請求、同步、來源及狀態檢查與私人證據寫入，排除模型載入及後續官方 scorer 寫檔。前後兩次的驗證程式和記錄開銷不同；本次另有一段 8.68 秒 CPU 擬合的起訖未留存，無法證明是否與評測重疊。因此兩次延遲不能當成乾淨的公平速度比較，也不能由上表宣稱推論速度提升。價格及耗電成本均未量測。

固定官方來源 revision `bb05a335bc809e61b20c0f745d25499a82b326fc`，資料集 SHA256 `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`。獨立 CPU 程序以原官方 scorer 重算全部原題回應及聚合，核對 924 筆分階段事件與 231 對 GPU 租約記錄。它不修改原 scorer，也不輸出個別題目、答案或機率。其來源檢查重驗儲存庫中的 45 個檔案，另外七個外部來源不在該程式的讀取範圍；另以 CPU 獨立重算本次計畫的全部 52 個來源雜湊，並核對固定模型 manifest 及其列入驗證的模型檔案。這不證明執行時每個 GPU backbone 張量的記憶體內容。

外層監督紀錄有 94 次主機觀察，ComfyUI 佇列均為空，最長相鄰取樣間隔約 0.522 秒；各次可用記憶體均達基本門檻，但監督器選用的較高門檻未寫入關閉紀錄，無法由該紀錄獨立核對；工作程序結束時 CUDA 配置量為零，所屬 cgroup 已消失。原 scorer 的完整記錄保留在明確標示尚未接受的本機 staging 中；本報告的數據經上述獨立重算，仍只是公開研究測試，沒有官方 composite、排名或 sealed evaluation。

獨立稽核彙總 SHA256 `7e76cbb23e05ff4550890c4f6007fcba52ff6fe4ed8b1d8f26b3fc03f413838b`，worker 終結紀錄 `fbc6724ac37ed141bba698abcb221350a28c4525cc067aa57372e8801c1abc3b`，外部監督關閉紀錄 `d143ec68493c22b2e5dd437a4d3210c5d56d122440465769a165a5f691bb3118`。在此機器且保留固定公開來源與本機證據時，可從儲存庫根目錄重新驗算：

```sh
env -u PYTHONPATH .venv/bin/python -B -s scripts/jevbench_heaven_scorer_audit.py --run-root artifacts/jev-runtime/decision3-heaven-candidate-check-20261027-01 --official-root /tmp/jev-heaven-public-inventory-20261027
```

此指令不執行模型或輸出原題。固定公開 checkout、本機權重及完整證據不在 Git 提交內，其他環境須先準備與計畫雜湊一致的來源。另一套 JevBench coherence 的[前次原生 MINI 分數](../2026-10-25/decision3-native-coherence-gpu-results.md)為 69.6405／100；它是先前 921＋327 次分段結果，本次沒有對新候選重跑。真實瀏覽器[單案例配對](decision3-link-navigation-firstcase-pair.md)兩臂均為 0／1，仍未證明能通過或超越基線。
