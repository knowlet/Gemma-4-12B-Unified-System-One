# Decision3 可變操作選項訓練資料收集

舊 operation 訓練資料只涵蓋固定 K=8 的選項，瀏覽器實際請求則可能出現 K=4、7、8。這個差異是待驗證的訓練覆蓋缺口，尚未證明它造成模型在真實場景失敗。

[收集器](../scripts/jev_decision3_affordance_collection.py)使用四個新編寫的 train-only 場景：

| 場景 | 內容 | 預期操作選項數 K |
| --- | --- | --- |
| `compact-one` | 單一按鈕 | 4 |
| `compact-two` | 兩個按鈕 | 4 |
| `form-top` | 文字欄位、選單與送出 | 7 |
| `form-middle` | 捲動至段落，再填寫、選取與送出 | 7、8 |

K 由原始 publisher 根據實際觀測建立。表中的數字是收集計畫要求；只有獨立驗證通過，才能確認實際涵蓋。每種 K 都須包含繼續操作與經驗證的完成狀態。

收集時沿用已固定版本的 jev-ultrafast Agent、`choose`、`action_space`、新鮮度檢查與執行器。腳本教師替代 HTTP 回覆與欄位文字產生，不呼叫神經網路。瀏覽器僅能存取本機 loopback fixture；教師的單點分布是執行指示，不能當成模型信心或品質分數。

## 排除清單與輸入

先準備經審查的排除清單，只放目標雜湊及識別碼，不放評測題目、答案或頁面內容。JSON 必須恰有以下五個欄位：

| 欄位 | 值 |
| --- | --- |
| `schema_id` | `decision3-affordance-exclusions-v1` |
| `normalization` | `NFKC-collapse-whitespace-v1` |
| `goal_sha256` | 非空、無重複的 SHA-256 小寫十六進位字串陣列 |
| `scenario_ids` | 無重複的非空識別碼陣列，可為空陣列 |
| `ancestor_ids` | 無重複的非空識別碼陣列，可為空陣列 |

目標雜湊的演算法為 `SHA256(" ".join(unicodedata.normalize("NFKC", goal).split()).encode("utf-8"))`。排除範圍以審查過的清單為準；收集器不會自行擴充成所有可能的評測目標。

另需原始 upstream checkout、明確的 Chromium 可執行檔及所選觀測設定的 helper 雜湊。收集計畫會固定來源、upstream、場景、排除清單與 Chromium 的身分。請保留產生計畫時的程式來源；後續 admission 仍會核對它們。若要同時開發，應從獨立且不再修改的來源樹執行收集。

## 先檢視計畫，再明確收集

下列為介面範例，未執行；請先替換路徑與經審查的 SHA-256。輸出目錄的父目錄須先存在，輸出目錄本身須不存在，且路徑不得經過符號連結。

```bash
python scripts/jev_decision3_affordance_collection.py \
  --output-dir /private/decision3/collection-01 \
  --upstream /path/to/jev-ultrafast \
  --collection-id affordance-train-01 \
  --seed 804 \
  --exclusion-manifest /private/decision3/reviewed-exclusions.json \
  --chromium /path/to/chromium-headless-shell \
  --observation-profile composed-visible-text-settled-v7 \
  --observation-helper-sha256 REVIEWED_V7_HELPER_SHA256 \
  > /private/decision3/prospective.json
```

沒有 `--collect` 時只回傳計畫，不建立輸出目錄，也不啟動瀏覽器。CLI 必須明確指定 `--observation-profile`；settled V7 可用，也可選擇 CLI 列出的原生或其他支援設定。

檢視 `prospective.json` 後，保留完全相同的參數，再加上：

```text
--collect
--collector-sha256 REVIEWED_COLLECTOR_SHA256
--plan-sha256 REVIEWED_PLAN_SHA256
```

`collector` 雜湊取自已審查計畫的 `plan.collector_sha256`；計畫雜湊取自最外層 `plan_sha256`。後者是 `plan` 以 UTF-8、`ensure_ascii=False`、`separators=(",", ":")`、`allow_nan=False` 序列化後的 SHA-256，**不是**整份排版後 `prospective.json` 的檔案雜湊。收集前會重新產生計畫並比對，來源或參數改變即拒絕。

## 獨立 admission 與完整 wire 投影

收集完成只得到 `completed_pending_admission`。DOM、原始請求、CDP journal、場景軌跡與收據都留在私人輸出目錄。不要只憑 `completed=true` 取出資料。

以下 API 範例未執行；請從同一個未修改的來源樹呼叫，並填入獨立審查後的雜湊：

```python
import sys
from pathlib import Path

sys.path[:0] = [
    "/path/to/frozen-source/scripts",
    "/path/to/frozen-source/src",
]
import jev_decision3_affordance_collection as collection

output = Path("/private/decision3/collection-01")
admitted = collection.admit_training_collection(
    output,
    expected_receipt_sha256="REVIEWED_RECEIPT_FILE_SHA256",
    expected_plan_sha256="REVIEWED_PLAN_SHA256",
)
rows = collection.compile_wire_projection(admitted)
```

`expected_receipt_sha256` 是 `receipt.json` 原始檔案位元組的 SHA-256。Admission 會重驗來源與觀測設定、loopback 網路範圍、所有場景及步驟、原始 Agent history、CDP 命令歸屬、新鮮度與動作順序、實際 DOM 轉移、DONE 條件與瀏覽器關閉。遺失、重疊、未分配的命令或未完整涵蓋 K=4、7、8 都會拒絕整批資料。

投影保留完整原始 `payload`、欄位及選項順序、未使用的 head 與只有單一選項的 head，不改寫 `generated-actions-none` 別名。`teacher_choice`、`valid_next_targets`、`loss_mask` 與證據識別碼位於原始請求之外；只有 operation 參與此收集格式的 loss，target 與文字仍保留作為執行證據。無 target 的操作使用 `target=null`；`BLOCKED` 可留在選項中，但不能成為正向訓練標籤。

即使 admission 通過，投影中的 `fit_eligible` 仍為 `false`。這個 API 提供經重播驗證的資料，不會自動訓練、授予模型資格或產生模型分數。

## 失敗與結果界線

瀏覽器、觀測設定、來源或關閉檢查失敗時，收據保留錯誤並標示 `failed_quarantined`；未完成或外來 schema 的收集不能通過 admission。保留失敗目錄，下一次嘗試使用全新的輸出與 collection ID。

原始執行器的 SELECT 使用 JavaScript 派送事件；其證據依原始執行器與 CDP 記錄連結驗證，不能宣稱所有事件都是 `isTrusted=true`。一般單元測試中的假請求與假 CDP 記錄只檢驗拒絕邊界，不代表真正瀏覽器收集成功。

這項收集始終回報 `model_calls=0`、`model_score=null`，不替代真正模型在瀏覽器中的評測，也不改寫兩套 JevBench 的個別分數。
