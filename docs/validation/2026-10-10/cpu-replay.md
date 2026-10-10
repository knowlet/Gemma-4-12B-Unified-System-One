# CPU 算術重播與發布前驗證

2026-10-10 收到「上傳，發佈後把檔案整理一下」授權後，已將先前列明的 28,341,101-byte 實驗快照傳回 Modal。輸入 SHA256 為 `fc41051826e8514a0d4b478c19ed471045141b3351e3344d81e8266aa8cdaf18`，20,142 members／84,335,351 uncompressed bytes；兩次皆核對原始來源與完整輸入，不更改資料或訓練配方。

## 已完成的回答重播

CPU04 完整結果已回傳。八組各自的 calibrated 與 unit coherence 人口，共 16 組全部逐值重現：

| 範圍 | 數量 | 與保存回答的差異 |
| --- | ---: | ---: |
| 請求 | 19,968 | 0 個缺失或額外請求 |
| 問題紀錄 | 193,152 | 0 個 native answer field mismatch |
| 機率值 | 529,136 | 0 個 probability mismatch；最大絕對差 0 |
| 每組／每政策 | 1,248 requests、12,072 questions、33,071 probabilities | 全部一致 |

proof.json 為 61,559,340 bytes，SHA256 `81d217c50e626b357f52471edfc3e60e9c77c3f876782b83949fc319060ae8c3`。結果 ZIP 為 12,418,419 bytes，以 48 個不超過 256KiB 的分段回傳，完整 SHA256 `930fc2bca1917fc12d0056d0f02bdb2e01cefe98b3e873cf79ebeb645d8986ac`。來源、順序、分段與整體雜湊、終止收據及 ZIP 完整性都需通過，才接受完整輸出。

這完成了原始 logits、實際保存溫度與回答向量的 Linux CPU 連結核對。公共 231 題的三種政策、保留人口與官方 raw-answer scoring 沿用各自已保存的完整核對，沒有新增模型推論。

## 仍保留的嚴格限制

校準有 24 個型別紀錄，22 個逐值一致，以下兩個紀錄仍不同。因此 proof 的 `exact_arithmetic_match` 保持 **false**，不把回答重播成功等同完整校準或模型接受：

| 候選／型別 | 保存值 | CPU04 重算值 | 差異 |
| --- | --- | --- | --- |
| head／Score temperature，重算 index32 | 0.6309573444801932 | 0.6309573444801934 | 1 binary64 ULP；CE-before／after 相同 |
| LoRA／Noul CE-before 及 CE-after，index40、T=1 | 0.6721651938565587 | 0.6721651938565589 | 各 2 binary64 ULP；溫度相同 |

完整稽核保留原精度門檻與逐欄 metadata 條件，六組完整通過、兩組仍未完成。原先嚴格 Mac 稽核的兩個 FP32 機率差異亦保留；固定 Linux 映像的成功不改寫成 Mac 位元一致。所有機率向量一致與完整校準 metadata 一致是分開判定的結果。

本日完整 analyzer 實際 exit2、190.28 秒，八組 coherence linkage 均為 `verified_same_raw_logits_pinned_cpu`，但 head／LoRA 的 `pinned_cpu_calibration` 仍未通過。JSON 的 aggregate 名稱 `incomplete_vector_verification` 同時要求校準 metadata 通過，並不表示已核對的 19,968 個回答向量有差異；`exact_expected_vectors` 保持 null。完整 JSON／Markdown 與執行收據放在補充 ZIP 的 `outcome/`，不改寫成八組全通過。

## 實際執行與回傳修補

兩次使用原始 image `im-hWNAPF8eRYjs4tA8VG582u`、2 CPU、4,096MiB、600 秒上限、唯讀 `/vol`、網路阻擋、單容器。Torch 2.10.0／NumPy 2.5.3／Linux x86_64；沒有 GPU、模型 forward、訓練、HF download 或 Volume write。平台設定與 terminal call 收據在補充 ZIP 中獨立保存。

CPU03 已上傳並啟動，但大型結果的 SDK blob 回傳嘗試連至 R2 儲存，被網路隔離擋住，terminal failure 為 36.664738 秒；未取得 proof。原失敗、完整平台 log、來源及輸入都保留，不能由這次失敗宣稱算術一致。

CPU04 只改回傳方式，使用 Modal 官方支援的 [generator invocation](https://modal.com/docs/guide/function-invocation-methods) 分段送回同一完整結果。14 個既有算術／輸入函式及 `audit_run` 保持逐字不變；各分段小於 SDK 的大型 blob 門檻，未放寬網路隔離或資源限制。新工具 source SHA256 為 `6bc29167a3f3fcbf891d94a5be5a90735432abafc0ec6e66d56558d7aeaffafb`，本機提交 `0aa2b86`；實際 A100 模型實驗來源仍是原 frozen commit `5a4319783026e32e4542b8d264b8bc9cb25b55ee`。

CPU04 的平台 terminal call `fc-01M4HQGD7MGGHF3FBZGJAW4A7R` 成功，51.399962 秒。獨立核對紀錄 SHA256 為 `8a04e545b5fb1b469fa1ee4e88aa8682b4e7eee7de3d5daf14fcbf98899c38f9`，保存在補充 ZIP 的 `cpu-arithmetic-04/independent-provider-review.json`，同時綁定實際平台設定、輸入、完整向量、校準差異與回傳雜湊。

## 發布前本機驗證

完整單一 process 測試：**1,712 passed、3 skipped、1 個既有 warning**，335.65 秒；Ruff check、format 與 diff check 全部通過。十個受測 source／test 檔案的前後 SHA 相同。分段回傳的針對性測試 173 passed；沒有 Modal 的環境則 108 passed、1 skipped，只有 SDK 序列化檢查跳過，核心大型結果與中斷核對仍執行。

## 模型與榜單結論

公開研究人口仍為原版 197/231、USER 提示對照 200/231、歷史 Jev-Omni 203/231。USER 對照與 Jev 的配對差距區間包含零；已曝光的公開題不構成正式榜單結果。Head／LoRA 有未校準合成題的學習收益，但校準後表現與原生媒體保留退步，本輪不採用新候選。

本次發布研究工具、結果與原始證據；沒有提交 sealed evaluation，formal acceptance 仍未評估、正式 rank／cost／eligibility 未知。既有 [八組完整結果](../2026-10-08/breakthrough/results.md) 與封存 ZIP 保持原始位元組；新結果與失敗另立本日紀錄。下載與封存內路徑對照見 [證據說明](evidence-layout.md)。
