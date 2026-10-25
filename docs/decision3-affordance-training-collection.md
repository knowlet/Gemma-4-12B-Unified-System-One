# Decision3 可變操作選項訓練資料收集

此收集器最初用來補足固定 K=8 訓練資料與實際可變選項之間的覆蓋缺口。新增的 checked-control 設定另提供勾選控制項與後續欄位操作的通用場景；兩種設定分別驗證，均不提供模型品質分數。

[收集器](../scripts/jev_decision3_affordance_collection.py)透過 `--scenario-profile` 明確選擇 train-only 場景來源：

| 場景設定 | 來源 | 場景族 | 預期操作選項數 K |
| --- | --- | --- | --- |
| `affordance-v1`（預設） | [原始場景模組](../scripts/jev_decision3_affordance_scenarios.py) | 四個可變選項場景 | 4、7、8 |
| `checked-control-progress-v1`（須明確選定） | [checked-control 場景模組](../scripts/jev_decision3_checked_control_scenarios.py) | `radio-two`、`radio-three` | 6 |

預設設定保留原來四個場景與 K4／K7／K8 的完整要求：

| 場景 | 內容 | 預期操作選項數 K |
| --- | --- | --- |
| `compact-one` | 單一按鈕 | 4 |
| `compact-two` | 兩個按鈕 | 4 |
| `form-top` | 文字欄位、選單與送出 | 7 |
| `form-middle` | 捲動至段落，再填寫、選取與送出 | 7、8 |

新設定的兩個場景族各有兩個或三個 radio 控制項，另有文字欄位、選單與送出按鈕，不需要捲動。種子會輪替可見標籤、所需值與 DOM 順序。教師路徑依實際狀態選取尚未勾選的目標 radio，再填寫、選取、送出，最後重新驗證 DONE；已勾選的 radio 仍保持可見且可點擊。研究依據與資料界線見 [checked-control 狀態進度設計](research/jev-checked-control-progress.md)。

K 由原始 publisher 根據實際觀測建立，不注入或刪減選項。表中的數字是各設定的收集計畫要求；只有獨立驗證通過，才能確認實際涵蓋。每種 K 都須包含繼續操作與經驗證的完成狀態，新設定的 K6 不會放寬預設設定的 K4／K7／K8 要求。

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

每份新計畫另固定以下設定欄位：

| 欄位 | 用途 |
| --- | --- |
| `scenario_profile` | 明確選定 `affordance-v1` 或 `checked-control-progress-v1` |
| `scenario_schema_id` | 所選場景模組的 schema；新設定為 `decision3-owned-checked-control-scenarios-v1` |
| `scenario_namespace` | 所選訓練身分範圍；新設定為 `decision3-owned-checked-control-training-v1` |
| `expected_menu_counts` | 預設為 `[4, 7, 8]`；新設定為 `[6]`，仍須由真實觀測證明 |

來源庫存包含收集器內的設定登錄與所選場景模組；新設定也固定原始模組，涵蓋共用輔助程式。重新規劃、收據、軌跡與 admission 都會核對設定身分；profile、schema 或 namespace 不符時，在教師或 DOM 重播前拒絕。缺少設定欄位的舊計畫只能解讀為原始場景，不能讓新 namespace 退回預設設定。新場景的 scenario、ancestor 及正規化目標雜湊仍須通過同一套排除清單，已曝光的評測案例與目標不得轉為訓練標籤。

## 先檢視計畫，再明確收集

下列為介面範例，未執行；請先替換路徑與經審查的 SHA-256。輸出目錄的父目錄須先存在，輸出目錄本身須不存在，且路徑不得經過符號連結。

```bash
python scripts/jev_decision3_affordance_collection.py \
  --output-dir /private/decision3/collection-01 \
  --upstream /path/to/jev-ultrafast \
  --collection-id checked-control-train-01 \
  --seed 804 \
  --scenario-profile checked-control-progress-v1 \
  --exclusion-manifest /private/decision3/reviewed-exclusions.json \
  --chromium /path/to/chromium-headless-shell \
  --observation-profile composed-visible-text-settled-v7 \
  --observation-helper-sha256 REVIEWED_V7_HELPER_SHA256 \
  > /private/decision3/prospective.json
```

沒有 `--collect` 時只回傳計畫，不建立輸出目錄，也不啟動瀏覽器。CLI 必須明確指定 `--observation-profile`；settled V7 可用，也可選擇 CLI 列出的原生或其他支援設定。

上述範例選擇新 checked-control 設定。若要使用原來四個場景，可指定 `--scenario-profile affordance-v1`；省略此參數也會保留原始預設。收集時必須沿用檢視過的場景設定及所有其他參數。

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

`expected_receipt_sha256` 是 `receipt.json` 原始檔案位元組的 SHA-256。Admission 會重驗來源與觀測設定、loopback 網路範圍、所有場景及步驟、原始 Agent history、CDP 命令歸屬、新鮮度與動作順序、實際 DOM 轉移、DONE 條件與瀏覽器關閉。遺失、重疊、未分配的命令或未完整涵蓋所選設定的 K 都會拒絕整批資料：預設要求恰為 K=4、7、8，新設定要求恰為 K=6。

投影保留完整原始 `payload`、欄位及選項順序、未使用的 head 與只有單一選項的 head，不改寫 `generated-actions-none` 別名。`teacher_choice`、`valid_next_targets`、`loss_mask` 與證據識別碼位於原始請求之外；只有 operation 參與此收集格式的 loss，target 與文字仍保留作為執行證據。無 target 的操作使用 `target=null`；`BLOCKED` 可留在選項中，但不能成為正向訓練標籤。

Operation-only 監督可提供勾選完成後轉向 TYPE_TEXT 或 SELECT 的訊號，但不會直接訓練同為 CLICK 的 radio 與送出按鈕之間的 target 選擇。Target 選擇若仍失敗，需要另行驗證的 target 適配；不能只憑這份資料格式宣稱重複點擊問題已解決。原始 API 動作、判定規則與 16 步上限保持不變。

即使 admission 通過，投影中的 `fit_eligible` 仍為 `false`。這個 API 提供經重播驗證的資料，不會自動訓練、授予模型資格或產生模型分數。

## 失敗與結果界線

瀏覽器、觀測設定、來源或關閉檢查失敗時，收據保留錯誤並標示 `failed_quarantined`；未完成或外來 schema 的收集不能通過 admission。保留失敗目錄，下一次嘗試使用全新的輸出與 collection ID。

原始執行器的 SELECT 使用 JavaScript 派送事件；其證據依原始執行器與 CDP 記錄連結驗證，不能宣稱所有事件都是 `isTrusted=true`。一般單元測試中的假請求與假 CDP 記錄只檢驗拒絕邊界，不代表真正瀏覽器收集成功。

這項收集始終回報 `model_calls=0`、`model_score=null`，不替代真正模型在瀏覽器中的評測，也不改寫兩套 JevBench 的個別分數。

2026-10-25 的 [checked-control 收集驗收](validation/2026-10-25/decision3-checked-control-collection-results.md)已接受兩個場景的 10 筆完整原生資料，實際皆為 K6，並通過獨立 admission。這是模型呼叫為 0 的資料收集；後續特徵擷取、訓練與真正模型評測仍需各自的新證據。
